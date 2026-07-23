# Overnight task 2026-07-22, Section 2.5: real D=20/W=80000/L=50 single-point fixed-point gate
# for the CM-aware C+ gradient backend. NOT a full trajectory (that needs a live outer KNITRO
# loop and materially more wall time / resource contention risk against the live CM campaign) --
# a single base-point comparison: same archC-verified base, both gradient backends evaluated at
# that exact state, timed and diffed. Resource policy: taskset away from cores 0-19 (reserved for
# the live campaign) at the shell level (see launch command in the accompanying log), modest
# thread count.
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
using Printf, LinearAlgebra

# NOTE: cm_config.jl is NOT part of cm_production_stage_runner.jl's actual include chain
# (confirmed by direct read) and has a latent docstring-target parse bug when included
# stand-alone (unrelated to this session's changes -- CMEvalKey's docstring at line 170 fails
# to attach, "cannot document the following expression"), so it is deliberately not included
# here. The only thing this script needed from it is the trivial :equal-grid probs formula,
# inlined below (matches cm_equal_grid_probs's own documented convention exactly,
# `precalc_common_marginals_cdf`'s pre-existing probs===nothing default).
cm_equal_grid_probs(L::Int) = collect(range(1 / L, (L - 1) / L, length = L))

t0 = time()
println("[", round(time()-t0,digits=1), "s] building D=20/W=80000 real-data context...")
flush(stdout)
W = 80000; L = 50
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
probs = cm_equal_grid_probs(L)
println("[", round(time()-t0,digits=1), "s] context ready. D=", D, " W=", size(ctx.U,1))
flush(stdout)

xf_calib = ctx.θ0_up[ctx.free_idx]

for contrasts in (:orthonormal,)   # production's actual contrast-basis decision (docs/CM_PRODUCTION_STATE_2026-07-22.md); :anchored skipped here to bound wall time
    println("[", round(time()-t0,digits=1), "s] building CM production context (L=$L, contrasts=$contrasts)...")
    flush(stdout)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, probs = probs)
    ctx_cm = pcx.ctx_cm; aug = pcx.aug; bins = pcx.bins; cctx = pcx.cctx
    println("[", round(time()-t0,digits=1), "s] CM context ready.")
    flush(stdout)

    println("[", round(time()-t0,digits=1), "s] archC_verified_state (shared base + verify) at calibration...")
    flush(stdout)
    t_solve0 = time()
    base, verify = archC_verified_state(xf_calib, ctx_cm, cctx)
    t_solve = time() - t_solve0
    @printf "[%.1fs] inner solve done in %.2fs. inner_status=%d Delta_dual=%.8f primal_dual_gap=%.3e\n" (time()-t0) t_solve verify.inner_status verify.Delta_dual verify.primal_dual_gap
    flush(stdout)

    pool = build_grad_workspace_pool(size(ctx.U,1))
    ws = build_lfix_factorized_workspace(D, size(ctx.U,1))

    println("[", round(time()-t0,digits=1), "s] Reference gradient callback (threaded=true)...")
    flush(stdout)
    GC.gc()
    t_ref0 = time()
    stats_ref = @timed cm_production_gradient(xf_calib, pcx, ctx, pe; base = base, threaded = true, h_mode = :adaptive)
    t_ref = time() - t_ref0
    g_ref, meta_ref = stats_ref.value
    @printf "[%.1fs] Reference: wall=%.3fs alloc=%.1fMB gc=%.3fs\n" (time()-t0) t_ref (stats_ref.bytes/1e6) stats_ref.gctime
    flush(stdout)

    println("[", round(time()-t0,digits=1), "s] C+ gradient callback (threaded=true)...")
    flush(stdout)
    GC.gc()
    t_cp0 = time()
    stats_cp = @timed cm_production_gradient_cplus(xf_calib, pcx, ctx, pe, pool, ws; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    t_cp = time() - t_cp0
    g_cp, meta_cp = stats_cp.value
    @printf "[%.1fs] C+:        wall=%.3fs alloc=%.1fMB gc=%.3fs\n" (time()-t0) t_cp (stats_cp.bytes/1e6) stats_cp.gctime
    flush(stdout)

    maxerr = maximum(abs.(g_ref .- g_cp))
    relerr = maxerr / max(maximum(abs.(g_ref)), 1e-12)
    cosang = dot(g_ref, g_cp) / (norm(g_ref) * norm(g_cp) + 1e-300)
    nflip = count(sign.(g_ref[2:end]) .!= sign.(g_cp[2:end]))
    @printf "[%s] D=20 fixed-point: max|Δg|=%.3e relerr=%.3e cos=%.10f sign-mismatches=%d/%d speedup(wall)=%.2fx\n" contrasts maxerr relerr cosang nflip D2-1 (t_ref/t_cp)
    println(maxerr < 1e-6 && cosang > 1 - 1e-8 && nflip == 0 ? "PASS: D=20/$contrasts CM-C+ matches CM-Reference at real production point" :
                                                                "FAIL: D=20/$contrasts CM-C+ DISAGREES with CM-Reference")
    flush(stdout)
end
println("[", round(time()-t0,digits=1), "s] DONE")
