# post_verify_fix_seed_registry.jl -- immutable, content-addressed POST_VERIFY_FIX seed registry
# (task §4) for the six non-ZC rerun chains + four fresh K=3 chains.
#
# Seed priorities per target cell (task §4):
#   S1: old published same-delta incumbent (source: the existing K1 frozen registry,
#       IMMUTABLE_SEED_REGISTRY_K1_2026-08-04.csv / frozen_seeds_K1/, built by seed_registry.jl --
#       reused, not re-derived).
#   S2: exact formerly rejected same-cell point, if recoverable and post-fix verified (source:
#       real reverification replays already performed this task/session -- see
#       targeted_k3_extensions/reverify_stuck/ for origin_zc lower delta=1).
#   S3/S4/S5 are populated live as each chain runs (newly refined smaller-delta points, other old
#   starts, cross-family relaxation seeds) -- not knowable ahead of the campaign itself, so this
#   registry seeds the CAMPAIGN START only; the orchestrator's own MonotoneEnvelope/incumbent
#   inheritance (continuation_polish_orchestrator.jl) handles S3 onward during the run itself.
#
# Every S1/S2 entry here is explicitly labeled `pre_fix_verifier=true` or `pre_fix_verifier=false`
# so a downstream reader can immediately see whether a given seed's own GT/Delta_star was computed
# under the OLD buggy gate (S1, from the K1 registry) or the FIXED gate (S2, from a genuine
# post-fix reverification) -- never conflated.
#
# Usage: julia --project=. post_verify_fix_seed_registry.jl [output_csv_path]

using Dates, SHA, Serialization, CSV

const D4E = @__DIR__
isdefined(Main, :d20_real_setup_design) || include(joinpath(D4E, "full_chain_include.jl"))
isdefined(Main, :OrchestratorRunState) || include(joinpath(D4E, "continuation_polish_orchestrator.jl"))

const K1_REGISTRY_CSV = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/IMMUTABLE_SEED_REGISTRY_K1_2026-08-04.csv"
const POST_VERIFY_FIX_ROOT = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/seed_registry"

"""
    load_k1_registry_as_S1() -> Vector{NamedTuple}

Reads the existing frozen K1 registry CSV (built by seed_registry.jl, chmod 444, never mutated)
and returns each row as an S1 candidate: the old published same-delta incumbent, explicitly tagged
`pre_fix_verifier=true` (its GT/Delta_star were accepted/reported under the OLD strict m_min>0
gate) so no downstream reader can mistake it for a fixed-gate result.
"""
function load_k1_registry_as_S1()
    isfile(K1_REGISTRY_CSV) || error("K1 registry not found at $K1_REGISTRY_CSV -- run seed_registry.jl first")
    rows = CSV.File(K1_REGISTRY_CSV)
    out = NamedTuple[]
    for r in rows
        push!(out, (
            seed_priority = "S1", family = r.family, direction = r.direction, delta = r.delta,
            W = r.W, K_mean = r.K_mean, K_pair = r.K_pair,
            gp = r.gp, GT = r.GT, Delta_star = r.Delta_star,
            source_path = r.frozen_path, w_digest = r.point_digest,
            pre_fix_verifier = true, verified_under_fixed_gate = false,
            note = "old K1 published same-delta incumbent; GT/Delta_star accepted under the PRE-FIX strict m_min>0 gate -- valid but potentially under-optimized, see PRE_FIX_RESULTS_STATUS.json",
        ))
    end
    return out
end

"""
    known_s2_candidates() -> Vector{NamedTuple}

Hand-registered S2 candidates: exact formerly-rejected points that were directly re-solved under
the FIXED gate this task/session and independently confirmed VerifiedSolved. Each entry's
checkpoint is read directly (not re-derived from a log line) so `w`/`gp`/`Delta` are exact.
"""
function known_s2_candidates()
    out = NamedTuple[]
    origzc_path = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/targeted_k3_extensions/reverify_stuck/origin_zc_lower_d1_rerun/origin_zc_lower_EXPLORE_DIRECT_SR1_latest.jls"
    if isfile(origzc_path)
        ckpt = load_cm_checkpoint_v10(origzc_path)
        if ckpt.best_feasible !== nothing
            w = ckpt.best_feasible.w
            digest = bytes2hex(sha256(join(string.(w), ",")))
            push!(out, (
                seed_priority = "S2", family = "origin_zc", direction = "lower", delta = 1.0,
                W = 100_000, K_mean = 1, K_pair = 1,
                gp = ckpt.best_feasible.gp, GT = NaN, Delta_star = ckpt.best_feasible.Delta,
                source_path = origzc_path, w_digest = digest,
                pre_fix_verifier = false, verified_under_fixed_gate = true,
                note = "real formerly-rejected origin_zc lower delta=1 point (feasible=true/verified=false under the old gate), directly re-solved and independently VerifiedSolved under the fixed classify_inner_result; GT not recomputed in this frozen snapshot (EXPLORE_DIRECT_SR1 stage only, not yet polished) -- gp/Delta_star are exact and materially tighter than the old published S1 incumbent (gp=0.998132, Delta=0.372).",
            ))
        end
    end
    return out
end

function freeze_registry(out_csv::String)
    rows = vcat(load_k1_registry_as_S1(), known_s2_candidates())
    isdir(POST_VERIFY_FIX_ROOT) || mkpath(POST_VERIFY_FIX_ROOT)
    header = ["seed_priority", "family", "direction", "delta", "W", "K_mean", "K_pair",
              "gp", "GT", "Delta_star", "source_path", "w_digest", "pre_fix_verifier",
              "verified_under_fixed_gate", "note"]
    open(out_csv, "w") do io
        println(io, join(header, ","))
        for r in rows
            vals = [replace(string(getfield(r, Symbol(h))), "," => ";") for h in header]
            println(io, join(vals, ","))
        end
    end
    chmod(out_csv, 0o444)
    println("Wrote ", length(rows), " seed candidates (", count(r -> r.seed_priority == "S1", rows),
            " S1 + ", count(r -> r.seed_priority == "S2", rows), " S2) to ", out_csv, " (chmod 444, immutable)")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    out_csv = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(POST_VERIFY_FIX_ROOT, "POST_VERIFY_FIX_SEED_REGISTRY_2026-08-04.csv")
    freeze_registry(out_csv)
end
