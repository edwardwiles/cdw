# ============================================================================
# Restricted-immutable-workspace production port (2026-07-24), task section 4:
# checkpoint fingerprint hardening.
#
# Root cause of the stale "cold_verified_seed.jls" crash (confirmed live, not
# guessed -- see docs/RESTRICTED_WORKSPACE_STALE_CHECKPOINT_ROOT_CAUSE_2026-07-24.md):
# that file is a bespoke ad-hoc NamedTuple, serialized directly by
# production_runs/cm_campaign_2026-07-22/preflight/seed_chainA_at_delta.jl,
# OUTSIDE the versioned CMCheckpointV6 pipeline in this file's sibling
# cm_checkpoint.jl. It carries only {w, Delta_dual, W, draw_seed, cm_L,
# contrasts, schema, bi, ...} -- no destination_sample/D_dest field at all.
# It was written when :all_legacy (square, D_dest=D=20) was the only
# destination-sample convention, so its `w` has total length 400 (1 + a
# 399-long z_free, matching build_pivot_elimination's D*Ddest-1 = 20*20-1).
# Current production defaults to destination_sample=:exclude_row (D_dest=19),
# whose pe.other_idx has length 379 (D*Ddest-1 = 20*19-1) -- calling
# `x_free_from_w(seedB.w, pe)` under the CURRENT pe therefore indexes
# `pe.other_idx[380]` on a 379-element vector: BoundsError, not a "379 vs 380
# off-by-one in the free vector itself" -- the free vector is 399 long, one
# whole destination-row longer than the current convention expects, and nothing
# in the load path checked destination_sample/D_dest before using it.
#
# This file adds an explicit, versioned fingerprint carried alongside any
# future "benchmark seed" checkpoint (a verified x_free point kept around for
# reuse across sessions, distinct from the full CMCheckpointV6 mid-campaign
# checkpoint schema in cm_checkpoint.jl), and a validating loader that
# refuses a mismatched file with a specific, actionable error instead of
# either (a) crashing on an unrelated BoundsError three call-frames away, or
# (b) silently padding/truncating/reinterpreting the stored vector.
# ============================================================================
const BENCHMARK_SEED_SCHEMA = 1

"""
    RestrictedWorkspaceBenchmarkSeed

A fingerprinted, family-agnostic benchmark seed point: a verified `w`
(pivot-reduced free vector, `x_free_from_w(w, pe)` convention) plus every
piece of provenance needed to detect whether it is still valid for a given
live `ctx`/`pe`/family config before use. Every field listed in the
production-port task's section 4 fingerprint list is present explicitly
(not merely derivable): destination_sample, D/D_dest (active destination
map), raw_active_cells, n_free_coords, pivot_lin (gravity pivot), family
(restriction family), K_mean, K_pair, mean_target_layout, and the
draw/data checksums.
"""
struct RestrictedWorkspaceBenchmarkSeed
    schema::Int
    created_at::String
    family::Symbol                  # :calibration | :origin_zc | :cm_meanzc
    # ---- data/draw fingerprint ----
    W::Int
    draw_design::Symbol
    draw_seed::Int
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    # ---- active-destination-layout fingerprint ----
    destination_sample::Symbol      # :all_legacy | :exclude_row
    D::Int
    D_dest::Int
    raw_active_cells::Int           # D*D_dest
    n_free_coords::Int              # D*D_dest - 1 (pivot-reduced free length)
    # ---- gravity pivot fingerprint ----
    pivot_lin::Int
    # ---- restriction-family fingerprint ----
    K_mean::Int
    K_pair::Int
    mean_target_layout::String       # e.g. "scalar_nu_per_k" (CM+meanZC) / "SharedByPowerLayout" (origin-ZC) / "none" (calibration)
    contrasts::Symbol
    cm_L::Int
    # ---- payload ----
    w::Vector{Float64}
    Delta_dual::Float64
    delta_budget::Float64
    verified_at::String
end

"""
    benchmark_seed_fingerprint_string(s::RestrictedWorkspaceBenchmarkSeed) -> String

One-line human-readable fingerprint summary, for logs and mismatch error messages.
"""
function benchmark_seed_fingerprint_string(s)
    return "family=$(s.family) destination_sample=$(s.destination_sample) D=$(s.D) D_dest=$(s.D_dest) " *
           "raw_active_cells=$(s.raw_active_cells) n_free_coords=$(s.n_free_coords) pivot_lin=$(s.pivot_lin) " *
           "K_mean=$(s.K_mean) K_pair=$(s.K_pair) mean_target_layout=$(s.mean_target_layout) " *
           "W=$(s.W) draw_design=$(s.draw_design) draw_seed=$(s.draw_seed) contrasts=$(s.contrasts) cm_L=$(s.cm_L) " *
           "draw_checksum_uniform=$(s.draw_checksum_uniform[1:12])... draw_checksum_transformed=$(s.draw_checksum_transformed[1:12])..."
end

"""
    save_benchmark_seed(path, ctx, pe, w; family, K_mean=0, K_pair=0, mean_target_layout="none",
                         contrasts=:orthonormal, cm_L=0, Delta_dual, delta_budget) -> path

Builds and atomically writes a fully-fingerprinted `RestrictedWorkspaceBenchmarkSeed`. Atomic
write (serialize-to-.tmp-then-mv) mirrors `save_cm_checkpoint`'s own discipline.
"""
function save_benchmark_seed(path::AbstractString, ctx, pe, w::Vector{Float64};
                              family::Symbol, K_mean::Int = 0, K_pair::Int = 0,
                              mean_target_layout::String = "none",
                              contrasts::Symbol = :orthonormal, cm_L::Int = 0,
                              Delta_dual::Float64, delta_budget::Float64)
    Ddest = _ctx_ddest(ctx)
    s = RestrictedWorkspaceBenchmarkSeed(
        BENCHMARK_SEED_SCHEMA, string(now()), family,
        size(ctx.U, 1), ctx.draw_design, ctx.draw_seed,
        ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
        hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :all_legacy,
        ctx.D, Ddest, ctx.D * Ddest, pe.D * pe.Ddest - 1,
        pe.pivot_lin, K_mean, K_pair, mean_target_layout, contrasts, cm_L,
        copy(w), Delta_dual, delta_budget, string(now()))
    tmp = path * ".tmp"
    serialize(tmp, s)
    mv(tmp, path; force = true)
    return path
end

struct BenchmarkSeedMismatch <: Exception
    msg::String
end
Base.showerror(io::IO, e::BenchmarkSeedMismatch) = print(io, "BenchmarkSeedMismatch: ", e.msg)

"""
    load_and_validate_benchmark_seed(path, ctx, pe; expected_family=nothing) -> Vector{Float64}

Deserializes a `RestrictedWorkspaceBenchmarkSeed` and validates EVERY fingerprint field against
the live `ctx`/`pe` before calling `x_free_from_w`. Throws `BenchmarkSeedMismatch` (listing every
field that disagrees, old vs new) on any mismatch -- never pads, truncates, or reinterprets the
stored vector. This is the single call production/benchmark scripts should use in place of raw
`deserialize(path)` + a partial `@assert`.
"""
function load_and_validate_benchmark_seed(path::AbstractString, ctx, pe; expected_family::Union{Nothing,Symbol} = nothing)
    isfile(path) || throw(BenchmarkSeedMismatch("no file at $path"))
    raw = deserialize(path)
    if !(raw isa RestrictedWorkspaceBenchmarkSeed)
        throw(BenchmarkSeedMismatch(
            "file at $path is not a RestrictedWorkspaceBenchmarkSeed (got $(typeof(raw))). " *
            "If this is a pre-2026-07-24 ad-hoc seed file (e.g. cm_campaign_2026-07-22's " *
            "cold_verified_seed.jls: a bare NamedTuple with no destination_sample/D_dest field), " *
            "it predates the destination_sample=:exclude_row convention and its `w` may be sized " *
            "for the OLD :all_legacy (square, D_dest=D) layout -- regenerate a fresh seed with " *
            "save_benchmark_seed instead of attempting to load it directly."))
    end
    Ddest = _ctx_ddest(ctx)
    live = (
        destination_sample = hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :all_legacy,
        D = ctx.D, D_dest = Ddest, raw_active_cells = ctx.D * Ddest, n_free_coords = pe.D * pe.Ddest - 1,
        pivot_lin = pe.pivot_lin, W = size(ctx.U, 1), draw_design = ctx.draw_design, draw_seed = ctx.draw_seed,
        draw_checksum_uniform = ctx.draw_meta.checksum_uniform, draw_checksum_transformed = ctx.draw_meta.checksum_transformed,
    )
    mismatches = String[]
    live.destination_sample == raw.destination_sample || push!(mismatches, "destination_sample: file=$(raw.destination_sample) live=$(live.destination_sample)")
    live.D == raw.D || push!(mismatches, "D: file=$(raw.D) live=$(live.D)")
    live.D_dest == raw.D_dest || push!(mismatches, "D_dest: file=$(raw.D_dest) live=$(live.D_dest)")
    live.raw_active_cells == raw.raw_active_cells || push!(mismatches, "raw_active_cells: file=$(raw.raw_active_cells) live=$(live.raw_active_cells)")
    live.n_free_coords == raw.n_free_coords || push!(mismatches, "n_free_coords: file=$(raw.n_free_coords) live=$(live.n_free_coords)")
    live.pivot_lin == raw.pivot_lin || push!(mismatches, "pivot_lin: file=$(raw.pivot_lin) live=$(live.pivot_lin)")
    live.W == raw.W || push!(mismatches, "W: file=$(raw.W) live=$(live.W)")
    live.draw_design == raw.draw_design || push!(mismatches, "draw_design: file=$(raw.draw_design) live=$(live.draw_design)")
    live.draw_seed == raw.draw_seed || push!(mismatches, "draw_seed: file=$(raw.draw_seed) live=$(live.draw_seed)")
    live.draw_checksum_uniform == raw.draw_checksum_uniform || push!(mismatches, "draw_checksum_uniform: file=$(raw.draw_checksum_uniform[1:12])... live=$(live.draw_checksum_uniform[1:12])...")
    live.draw_checksum_transformed == raw.draw_checksum_transformed || push!(mismatches, "draw_checksum_transformed: file=$(raw.draw_checksum_transformed[1:12])... live=$(live.draw_checksum_transformed[1:12])...")
    if expected_family !== nothing && raw.family != expected_family
        push!(mismatches, "family: file=$(raw.family) expected=$(expected_family)")
    end
    length(raw.w) == 1 + raw.n_free_coords || push!(mismatches, "w length: file has $(length(raw.w)), expected 1+n_free_coords=$(1+raw.n_free_coords) (the file's OWN fingerprint is internally inconsistent)")
    if !isempty(mismatches)
        throw(BenchmarkSeedMismatch(
            "checkpoint at $path FAILS fingerprint validation against the live context -- refusing " *
            "to load (not padding, truncating, or reinterpreting). Mismatched field(s):\n  " *
            join(mismatches, "\n  ") *
            "\nFile fingerprint: " * benchmark_seed_fingerprint_string(raw)))
    end
    return x_free_from_w(raw.w, pe)
end
