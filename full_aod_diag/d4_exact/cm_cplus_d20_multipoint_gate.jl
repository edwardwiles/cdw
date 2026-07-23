# Part II.2, 2026-07-23: real D=20/W=80000/L=50 multi-point CM-C+ vs CM-Reference gate, using
# ACTUAL cold-verified incumbents from the completed cm_campaign_2026-07-22 (all 3 chains, all 4
# deltas, finished and cold-verified by 2026-07-23 01:22 EDT -- confirmed via supervisor .out logs
# and `ps` before this script was written; campaign directories are READ-ONLY inputs here, nothing
# in production_runs/ is written by this script). Reconstructs each point via the EXACT recipe
# cm_cold_verify.jl itself uses (same worktree, same commit lineage): rebuild ctx from the
# checkpoint's own recorded (W,delta,draw_design,draw_seed), assert draw checksums match, x_free =
# x_free_from_w(best_feasible.w, pe). Runs Reference and C+ gradient callbacks 3x each (post
# warmup) and reports medians, not single observations, per the brief.
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, LinearAlgebra, Statistics, Serialization

lp(xs...) = (println(xs...); flush(stdout))
t0 = time()
elapsed() = round(time() - t0, digits = 1)

const CAMPAIGN = "/bbkinghome/edav/gravity_robustness/production_runs/cm_campaign_2026-07-22"
points = [
    ("chain1_delta0.1", joinpath(CAMPAIGN, "chain1", "delta_0.1", "stage_latest.jls")),
    ("chain1_delta1.0", joinpath(CAMPAIGN, "chain1", "delta_1.0", "stage_latest.jls")),
    ("chain1_delta2.0", joinpath(CAMPAIGN, "chain1", "delta_2.0", "stage_latest.jls")),
    ("chain2_delta1.0", joinpath(CAMPAIGN, "chain2", "delta_1.0", "stage_latest.jls")),
    ("chain3_delta1.0", joinpath(CAMPAIGN, "chain3", "delta_1.0", "stage_latest.jls")),
]

results = NamedTuple[]

for (label, ckpt_path) in points
    lp("="^100)
    lp("[", elapsed(), "s] POINT: ", label, "  (", ckpt_path, ")")
    lp("="^100)
    ckpt = load_cm_checkpoint(ckpt_path)
    ckpt.best_feasible === nothing && (lp("  SKIP: no best_feasible in this checkpoint"); continue)

    ctx = d20_real_setup_design(W = ckpt.W, δ = ckpt.delta, find_smallest = ckpt.find_smallest,
                                 draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed)
    pe = build_pivot_elimination(ctx)
    if ctx.draw_meta.checksum_uniform != ckpt.draw_checksum_uniform ||
       ctx.draw_meta.checksum_transformed != ckpt.draw_checksum_transformed
        lp("  SKIP: draw checksum MISMATCH -- refusing to evaluate against a different problem instance")
        continue
    end
    D = ctx.D; D2 = D^2; W = size(ctx.U, 1)
    pcx = build_cm_production_context(ctx, CS; L = ckpt.cm_L, contrasts = ckpt.cm_contrasts, probs = ckpt.cm_probs)
    ctx_cm = pcx.ctx_cm; cctx = pcx.cctx

    w = ckpt.best_feasible.w
    xf = x_free_from_w(w, pe)
    K, base_check, verify_check = cm_production_value_verified(xf, pcx)
    cold_diff = abs(verify_check.Delta_dual - ckpt.best_feasible.Delta)
    lp("[", elapsed(), "s] re-cold-verify: Delta_dual=", verify_check.Delta_dual, " (checkpoint reported ", ckpt.best_feasible.Delta, ") |diff|=", cold_diff, " inner_status=", verify_check.inner_status)
    winner_here, _, gap_here = compute_winners(base_check.θ_full0, ctx)

    base, verify = archC_verified_state(xf, ctx_cm, cctx)

    pool = build_grad_workspace_pool(W)
    ws = build_lfix_factorized_workspace(D, W)

    # warmup (not timed) -- JIT/compile, first-call allocation noise
    _ = cm_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true, h_mode = :adaptive)
    _ = cm_production_gradient_cplus(xf, pcx, ctx, pe, pool, ws; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())

    nrep = 3
    ref_walls = Float64[]; ref_allocs = Float64[]; ref_gcs = Float64[]
    cp_walls = Float64[]; cp_allocs = Float64[]; cp_gcs = Float64[]
    local g_ref, g_cp
    for r in 1:nrep
        GC.gc()
        s = @timed cm_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true, h_mode = :adaptive)
        g_ref = s.value[1]
        push!(ref_walls, s.time); push!(ref_allocs, s.bytes / 1e6); push!(ref_gcs, s.gctime)
    end
    for r in 1:nrep
        GC.gc()
        s = @timed cm_production_gradient_cplus(xf, pcx, ctx, pe, pool, ws; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
        g_cp = s.value[1]
        push!(cp_walls, s.time); push!(cp_allocs, s.bytes / 1e6); push!(cp_gcs, s.gctime)
    end

    maxerr = maximum(abs.(g_ref .- g_cp))
    relerr = maxerr / max(maximum(abs.(g_ref)), 1e-12)
    cosang = dot(g_ref, g_cp) / (norm(g_ref) * norm(g_cp) + 1e-300)
    nflip = count(sign.(g_ref[2:end]) .!= sign.(g_cp[2:end]))

    m_ref_wall = median(ref_walls); m_cp_wall = median(cp_walls)
    @printf "[%s] label=%s delta=%.2f Delta_dual=%.6f\n" "RESULT" label ckpt.delta verify.Delta_dual
    @printf "  max|Δg|=%.3e relerr=%.3e cos=%.10f sign-mismatches=%d/%d\n" maxerr relerr cosang nflip D2-1
    @printf "  Reference: median wall=%.3fs (reps=%s) median alloc=%.1fMB median gc=%.3fs\n" m_ref_wall string(round.(ref_walls,digits=3)) median(ref_allocs) median(ref_gcs)
    @printf "  C+:        median wall=%.3fs (reps=%s) median alloc=%.1fMB median gc=%.3fs  speedup=%.2fx\n" m_cp_wall string(round.(cp_walls,digits=3)) median(cp_allocs) median(cp_gcs) (m_ref_wall/m_cp_wall)
    passed = maxerr < 1e-6 && cosang > 1 - 1e-8 && nflip == 0
    lp(passed ? "PASS" : "FAIL", ": ", label)
    flush(stdout)

    push!(results, (label = label, delta = ckpt.delta, Delta_dual = verify.Delta_dual, cold_diff = cold_diff,
                     maxerr = maxerr, relerr = relerr, cosine = cosang, sign_mismatches = nflip, D2m1 = D2 - 1,
                     ref_wall_median = m_ref_wall, cp_wall_median = m_cp_wall,
                     ref_alloc_median = median(ref_allocs), cp_alloc_median = median(cp_allocs),
                     speedup = m_ref_wall / m_cp_wall, passed = passed,
                     inner_status = verify.inner_status, n_winners = D))
end

OUTCSV = joinpath(@__DIR__, "..", "..", "results", "cm_cplus_followup", "d20_multipoint_gate.csv")
open(OUTCSV, "w") do io
    println(io, "label,delta,Delta_dual,cold_diff,maxerr,relerr,cosine,sign_mismatches,D2m1,ref_wall_median,cp_wall_median,ref_alloc_median,cp_alloc_median,speedup,passed,inner_status")
    for r in results
        println(io, "$(r.label),$(r.delta),$(r.Delta_dual),$(r.cold_diff),$(r.maxerr),$(r.relerr),$(r.cosine),$(r.sign_mismatches),$(r.D2m1),$(r.ref_wall_median),$(r.cp_wall_median),$(r.ref_alloc_median),$(r.cp_alloc_median),$(r.speedup),$(r.passed),$(r.inner_status)")
    end
end
lp()
lp("Wrote ", OUTCSV, " (", length(results), " points)")
lp("="^100)
n_fail = count(r -> !r.passed, results)
lp("TOTAL: ", length(results) - n_fail, " passed, ", n_fail, " failed (of ", length(results), " points attempted)")
lp("="^100)
lp("[", elapsed(), "s] DONE")
