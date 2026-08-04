# Task "fixed-state FULL-vs-REDUCED inner A/B", step 4: versioned sentinel panel.
#
# Assembles points from REAL sources -- production checkpoints already written by the outer
# completion session's own W=20,000/W=100,000 canonical-CLI runs (results/canonical_runner/,
# read-only, never mutated), plus real archived historical forensic points (eval18, and the
# unrestricted W=100,000 replay set) -- NOT arbitrary random vectors, per task section 4's
# explicit instruction. Every point is stored as a REDUCED-native `w_profiled` vector
# (`[gp; r_free]`), decodable to the full FULL A matrix via `decode_outer_profiled` (step 3's own
# verified round trip) -- so ONE stored representation serves both arms; the FULL arm is
# reconstructed from it at solve time, not stored twice.
#
# Read-only with respect to results/canonical_runner/** (never written to) and every full_aod_diag
# file (only `include`d).

const D4X = joinpath(dirname(dirname(dirname(@__DIR__))), "full_aod_diag", "d4_exact")
const REPO_ROOT = dirname(dirname(dirname(@__DIR__)))
const RESULTS_ROOT = joinpath(REPO_ROOT, "results", "canonical_runner")
const HISTORICAL_SCRATCH = "/bbkinghome/edav/repo_scratch/profiled-functional-readiness-closeout-2026-08-03"

for f in ["context_real_d20.jl", "draw_design.jl",
          "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "blas_thread_policy.jl", "knitro_outer_algorithm.jl", "production_backend_manifest.jl",
          "incumbent_logic.jl", "cm_hessian_subblock_profiling.jl", "production_bundle_api.jl", "country_resolve.jl",
          "cm_exact_cache_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl"]
    include(joinpath(D4X, f))
end

include(joinpath(@__DIR__, "frozen_manifest.jl"))
using .FixedStateInnerABFrozenManifest
using Serialization, JLD2, Printf

const CANONICAL_FAMILIES = (:unrestricted, :flexible_cm, :common_frechet, :origin_zc, :cm_meanzc)

"""
    build_ctx_pe(mode::Symbol) -> (ctx, pe, sci)

ONE context/pivot-elimination per mode, shared by every point at that mode -- every point in the
bank at a given mode decodes through the identical `(ctx, pe)`, matching task section 2's "use one
manifest object to construct both arms."
"""
function build_ctx_pe(mode::Symbol)
    sci = mode === :mode_a ? mode_a_scientific_manifest() :
          mode === :mode_b ? mode_b_scientific_manifest() :
          error("build_ctx_pe: mode must be :mode_a or :mode_b, got :$mode")
    ctx = d20_real_setup_design(W = sci.W, δ = 1.0, find_smallest = true,
        draw_design = sci.draw_design, draw_seed = sci.draw_seed,
        destination_sample = sci.destination_sample,
        exclude_diagonal_gravity = sci.exclude_diagonal_gravity,
        gravity_exclude_cells = sci.gravity_exclude_cells,
        σHat = sci.sigma)
    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    # Korea/Brazil anchor-tie override -- the SAME hardcoded (korea_idx=14, brazil_idx=3) every
    # real REDUCED production driver uses (bin/run_profiled_model.jl:236-239's own comment: "the
    # SAME hardcoded ... every existing D20 REDUCED production driver in this directory uses").
    # Omitting this (build_anchor_spec_from_ctx(ctx) with no override, caught live during this
    # session) builds a DIFFERENT, mismatched pivot/anchor spec than the one real checkpoints were
    # written under -- decoding their zfree through the wrong spec would silently produce a
    # self-consistent-looking (still gravity-feasible by construction) but WRONG economic state,
    # not an error. Must match production exactly, not use the function's own empty default.
    korea_idx, brazil_idx = 14, 3
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
    gauge = build_anchor_gauge(z_calib, spec)
    pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
    return (ctx = ctx, pe = pe, sci = sci, z_calib = z_calib, gp0 = θ0[3+D])
end

"""
    parse_flat_vector(path) -> Vector{Float64}

Parses the flat Julia array-literal `.txt` format used by the eval18/historical-replay forensic
points (`[gp, r_free...]`, e.g. `eval18_captured_point_2026-08-02.txt` --
`verify_eval18_current_code_2026-08-03.jl:85-86` is the pre-existing parse this mirrors exactly).
"""
function parse_flat_vector(path::AbstractString)
    s = read(path, String)
    return Vector{Float64}(eval(Meta.parse(s)))
end

"""
    point_from_w(w, ctx, pe; eta_nu=Float64[]) -> NamedTuple

Decodes a REDUCED-native `w_profiled` into the full economic state via step 3's own verified
round trip (`decode_outer_profiled`) -- the ONE place a point's full A/gp gets reconstructed.
"""
function point_from_w(w::Vector{Float64}, ctx, pe; eta_nu::Vector{Float64} = Float64[])
    dec = decode_outer_profiled(w, ctx, pe)
    g = gravity_from_logz(dec.z_full, ctx)
    return (w = w, gp = dec.gp, z_full = dec.z_full, Aod_levels = dec.Aod_levels,
        eta_nu = eta_nu, nu = isempty(eta_nu) ? Float64[] : exp.(eta_nu), gravity_residual = g)
end

"""
    load_reduced_checkpoint_point(family, W) -> NamedTuple or nothing

Reads `results/canonical_runner/reduced_<family>_W<W>_delta1.0/checkpoint.jls` (CMCheckpointV11,
already `include`d via cm_checkpoint.jl) -- read-only, never mutated. Returns `nothing` (not an
error) if the directory doesn't exist for some (family, W) cell, so a caller can report an honest
gap rather than crash.
"""
function load_reduced_checkpoint_point(family::Symbol, W::Int)
    dir = joinpath(RESULTS_ROOT, "reduced_$(family)_W$(W)_delta1.0")
    path = joinpath(dir, "checkpoint.jls")
    isfile(path) || return nothing
    cp = load_cm_checkpoint_v11(path)
    # Real bug, caught live during this task's own step 6/campaign smoke test: cp.g/cp.zfree are
    # the CURRENT outer iterate at whatever wall-clock instant the checkpoint fired
    # (cp.checkpoint_reason=:wall_interval for every one of these real checkpoints) -- NOT
    # necessarily feasible at all, since KNITRO's own search legitimately visits infeasible
    # points between feasible ones. cp.best_feasible (a separate field) is the actual verified
    # feasible incumbent's own w -- confirmed by direct inspection: for the real unrestricted
    # W=20,000 checkpoint, cp.g=0.9590607665008352 vs cp.best_feasible.w[1]=0.9590607598370081 --
    # close but NOT identical, confirming these are genuinely two different points, not the same
    # value read two ways. Using cp.g/cp.zfree here silently built a P1 "ordinary feasible" point
    # bank entry that was NOT actually verified feasible -- it happened to still pass REDUCED's
    # own screen at Mode A scale (screens can be permissive), but failed FULL's stricter
    # inner-feasibility pre-check once bridged, which is what surfaced this. Use the real
    # verified-feasible point when the checkpoint has one; only fall back to the raw iterate (with
    # a loud warning, not silently) if best_feasible is nothing.
    # Free-nu families' best_feasible NamedTuple uses w_econ/eta_nu (separate fields), not a
    # single w -- confirmed live (FieldError surfaced this immediately): fields
    # (gp, w_econ, eta_nu, Delta, n_eval, t_elapsed) for origin_zc/cm_meanzc vs (gp, w, Delta,
    # n_eval, t_elapsed) for the fixed-nu families. Use best_feasible's OWN eta_nu in the free-nu
    # case (the eta at the verified feasible point), not cp.eta_nu (the top-level, current-iterate
    # field -- same current-vs-incumbent distinction as the w/zfree fix above).
    if cp.best_feasible !== nothing
        if hasproperty(cp.best_feasible, :w_econ)
            w = cp.best_feasible.w_econ
            eta_nu = cp.best_feasible.eta_nu
        else
            w = cp.best_feasible.w
            eta_nu = cp.eta_nu
        end
    else
        @warn "load_reduced_checkpoint_point($family, W=$W): checkpoint has no best_feasible incumbent -- falling back to the raw (possibly infeasible) current iterate cp.g/cp.zfree"
        w = vcat(cp.g, cp.zfree)
        eta_nu = cp.eta_nu
    end
    return (w = w, eta_nu = eta_nu, n_eval = cp.n_eval, n_grad = cp.n_grad,
        wall_elapsed = cp.wall_elapsed, checkpoint_reason = cp.checkpoint_reason,
        best_feasible = cp.best_feasible, source_path = path)
end

struct SentinelPoint
    family::Symbol
    point_id::String
    point_type::String   # "P0"|"P1"|"P2"|"P3"
    mode::Symbol
    W::Int
    w::Vector{Float64}
    eta_nu::Vector{Float64}
    nu::Vector{Float64}
    gp::Float64
    z_full::Matrix{Float64}
    Aod_levels::Vector{Float64}
    gravity_residual::Float64
    source::String
    verification_status::String
end

function build_point_bank(; mode::Symbol)
    built = build_ctx_pe(mode)
    ctx, pe, sci = built.ctx, built.pe, built.sci
    points = SentinelPoint[]

    # ---- P0: calibration point, shared across all 5 families (θ0_up itself). ----
    w0 = reduce_to_w_profiled(built.gp0, built.z_calib, pe)
    p0 = point_from_w(w0, ctx, pe)
    for family in CANONICAL_FAMILIES
        push!(points, SentinelPoint(family, "P0", "P0", mode, sci.W, p0.w, p0.eta_nu, p0.nu, p0.gp,
            p0.z_full, p0.Aod_levels, p0.gravity_residual,
            "ctx.θ0_up (gravity calibration), real D20 dataset $(sci.dataset_version)",
            "gravity-feasible by construction of pivot elimination (residual $(p0.gravity_residual)); " *
            "verified via step-3 decoded-state round trip (docs/audits/profiled-fixed-state-inner-ab-2026-08-04/MASTER.md)"))
    end

    # ---- P1: real checkpoint from the canonical CLI runner (results/canonical_runner/, read-only). ----
    for family in CANONICAL_FAMILIES
        cp = load_reduced_checkpoint_point(family, sci.W)
        if cp === nothing
            @warn "no real P1 checkpoint found for ($family, W=$(sci.W)) -- honest gap, not filled with a synthetic point"
            continue
        end
        p1 = point_from_w(cp.w, ctx, pe; eta_nu = cp.eta_nu)
        push!(points, SentinelPoint(family, "P1", "P1", mode, sci.W, p1.w, p1.eta_nu, p1.nu, p1.gp,
            p1.z_full, p1.Aod_levels, p1.gravity_residual, cp.source_path,
            "real REDUCED checkpoint: n_eval=$(cp.n_eval) n_grad=$(cp.n_grad) " *
            "checkpoint_reason=$(cp.checkpoint_reason) best_feasible=$(cp.best_feasible) " *
            "(re-classification deferred to this task's own step 6 cold-solve run, not re-verified here)"))
    end

    # ---- P3: real archived infeasible/stall points. unrestricted + flexible_cm only -- no real
    # persisted P3 file exists in this repo for common_frechet/origin_zc/cm_meanzc (confirmed by
    # repo-wide search; only logged, unsaved shift-probe numbers exist for those three, which do
    # NOT meet "an exact saved point", so they are left as an honest gap here, not fabricated). ----
    hist_path_2 = joinpath(HISTORICAL_SCRATCH, "historical_unrestricted_eval2_2026-08-04.txt")
    if isfile(hist_path_2)
        w_hist2 = parse_flat_vector(hist_path_2)
        p3u = point_from_w(w_hist2, ctx, pe)
        push!(points, SentinelPoint(:unrestricted, "P3a", "P3", mode, sci.W, p3u.w, p3u.eta_nu, p3u.nu,
            p3u.gp, p3u.z_full, p3u.Aod_levels, p3u.gravity_residual, hist_path_2,
            "real archived W=100,000 historical stall point (idx=2 of a JLD2 dict from a genuine " *
            "production campaign, extracted 2026-08-04); documented nStatus=-300 on current HEAD " *
            "(verify_unrestricted_historical_replay_2026-08-04.jl) -- re-classification deferred to step 6"))
    else
        @warn "historical unrestricted eval2 point not found at $hist_path_2 -- outside git repo, machine-local only"
    end
    hist_path_17 = joinpath(HISTORICAL_SCRATCH, "historical_unrestricted_eval17_2026-08-04.txt")
    if isfile(hist_path_17)
        w_hist17 = parse_flat_vector(hist_path_17)
        p3u2 = point_from_w(w_hist17, ctx, pe)
        push!(points, SentinelPoint(:unrestricted, "P3b", "P3", mode, sci.W, p3u2.w, p3u2.eta_nu, p3u2.nu,
            p3u2.gp, p3u2.z_full, p3u2.Aod_levels, p3u2.gravity_residual, hist_path_17,
            "real archived W=100,000 historical stall point (idx=17); documented nStatus=-300 on " *
            "current HEAD (verify_unrestricted_historical_replay_2026-08-04.jl) -- re-classification deferred to step 6"))
    end

    eval18_path = joinpath(D4X, "eval18_captured_point_2026-08-02.txt")
    if isfile(eval18_path)
        w_eval18 = parse_flat_vector(eval18_path)
        p3f = point_from_w(w_eval18, ctx, pe)
        push!(points, SentinelPoint(:flexible_cm, "P3", "P3", mode, sci.W, p3f.w, p3f.eta_nu, p3f.nu,
            p3f.gp, p3f.z_full, p3f.Aod_levels, p3f.gravity_residual, eval18_path,
            "real captured eval18 point (in-repo), documented nStatus=-400 at production maxit=100, " *
            "nStatus=-300 at maxit=1000 real D20/W=100,000, corroborated by an independent HiGHS LP " *
            "infeasibility certificate (EVAL18_FORENSIC_VERDICT_2026-08-02.md) -- gp=$(p3f.gp) -- " *
            "re-classification deferred to step 6"))

        # ---- P2 for flexible_cm ONLY: real, reproducible interpolation between the calibration
        # point and the real eval18 infeasible point, k=0.3/k=0.6 -- exactly the construction
        # already documented and run in CONTINUATION_2026-08-04.md (real KNITRO solves, feasible,
        # near the eval18 infeasibility boundary), reconstructed here from the two real endpoints
        # rather than re-copied from an unsaved log number. ----
        for (k, label) in ((0.3, "P2a"), (0.6, "P2b"))
            w_k = w0 .+ k .* (w_eval18 .- w0)
            p2 = point_from_w(w_k, ctx, pe)
            push!(points, SentinelPoint(:flexible_cm, label, "P2", mode, sci.W, p2.w, p2.eta_nu, p2.nu,
                p2.gp, p2.z_full, p2.Aod_levels, p2.gravity_residual,
                "constructed: w_calib + $(k)*(w_eval18 - w_calib), reproducing " *
                "CONTINUATION_2026-08-04.md's own k=$(k) near-infeasibility-boundary point",
                "documented feasible+near-boundary at k=$(k) (CONTINUATION_2026-08-04.md); " *
                "re-classification deferred to step 6"))
        end
    else
        @warn "eval18_captured_point_2026-08-02.txt not found at $eval18_path"
    end

    return points
end

if abspath(PROGRAM_FILE) == @__FILE__
    all_points = SentinelPoint[]
    for mode in (:mode_a, :mode_b)
        println("="^90); println("Building point bank, mode=$mode"); println("="^90)
        append!(all_points, build_point_bank(mode = mode))
    end

    out_dir = "/bbkinghome/edav/repo_scratch/profiled-fixed-state-inner-ab-2026-08-04"
    mkpath(out_dir)
    jld2_path = joinpath(out_dir, "FIXED_STATE_POINT_BANK.jld2")
    jldsave(jld2_path; points = all_points)

    csv_path = joinpath(out_dir, "FIXED_STATE_POINT_MANIFEST.csv")
    open(csv_path, "w") do io
        println(io, "family,point_id,point_type,mode,W,gp,gravity_residual,eta_nu_norm,source,verification_status")
        for p in all_points
            eta_norm = isempty(p.eta_nu) ? 0.0 : sqrt(sum(abs2, p.eta_nu))
            src_escaped = replace(p.source, "\"" => "'")
            status_escaped = replace(p.verification_status, "\"" => "'")
            @printf(io, "%s,%s,%s,%s,%d,%.16f,%.3e,%.6f,\"%s\",\"%s\"\n",
                p.family, p.point_id, p.point_type, p.mode, p.W, p.gp, p.gravity_residual, eta_norm,
                src_escaped, status_escaped)
        end
    end

    println()
    println("Total points: ", length(all_points))
    for family in CANONICAL_FAMILIES
        fam_points = filter(p -> p.family == family, all_points)
        types = sort(unique(p.point_type for p in fam_points))
        println("  $family: ", length(fam_points), " points, types present: ", types)
    end
    println("Wrote: ", jld2_path)
    println("Wrote: ", csv_path)
end
