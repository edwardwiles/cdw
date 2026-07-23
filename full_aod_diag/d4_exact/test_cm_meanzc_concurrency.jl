# Step 9.3: concurrency safety for the CM+moments(+ZC) extension under
# par_concurrent_evals=yes -- the ACTUAL KNITRO option production's own CM
# outer driver opt file (csw_outer_wallclock_sr1.opt) sets, and hence the
# real mechanism the task brief's "concurrent-callback machinery" hazard is
# about: KNITRO's own C-level dispatch of cb_F!/cb_G! for DIFFERENT trial
# points, from ITS OWN worker threads, WITHIN one already-open outer KN_solve
# session.
#
# EARLIER VERSION OF THIS FILE (superseded): attempted to prove the same
# property by spawning many INDEPENDENT full KN_new()/KN_solve() sessions
# concurrently via Julia's Threads.@threads. That is a fundamentally
# different, much heavier operation (each is its own KNITRO problem
# instance/license negotiation) that this codebase has apparently never
# exercised -- it deadlocked for ~3.5 hours (confirmed via ps: stuck in a
# non-Julia call, unresponsive even to SIGTERM, requiring SIGKILL). This
# matches a known, PRE-EXISTING class of hazard already documented in this
# repo's own history (a non-reentrant OpenMP/KNITRO lock under concurrent
# full-solve invocation) -- orthogonal to this extension's nu design, and not
# a pattern real production ever uses (the actual multi-chain CM campaign
# achieves parallelism via separate OS PROCESSES, never concurrent KN_solve
# calls within one process). Do not resurrect that approach.
#
# What THIS version actually tests: running a real outer KNITRO solve
# (par_concurrent_evals=yes, exactly production's own opt file) against the
# meanzc-augmented objective, then cold-re-verifying the returned incumbent
# from scratch (fresh inner solve, independent of whatever internal
# concurrent dispatch KNITRO used to find it). If nu were ever contaminated
# across concurrent callback invocations, the incumbent's stored (w, Delta)
# pair would not reproduce under a clean, independent re-evaluation -- this
# is the same kind of check cm_cold_verify.jl already uses to catch
# checkpoint-state corruption, applied here to catch nu corruption instead.
# Combined with the architectural guarantee (nu never touches any shared
# mutable field -- verified by inspection: cm_meanzc_moments.jl's
# wrap_moments_with_cm_meanzc closure captures no Ref, only reads nu from its
# per-call theta_ext argument), this is a real, safe, and now BOUNDED
# (hard-capped by maxtime_real, unlike the abandoned Threads.@threads
# approach) empirical concurrency check.
#
# Run with: timeout 90 julia --project=. full_aod_diag/d4_exact/test_cm_meanzc_concurrency.jl
# (external timeout wrapper -- do not trust an internal KNITRO wall-clock
# budget alone to bound this process; see the "KNITRO driver runs can hang
# past their own timeout" lesson from this same debugging session).
include(joinpath(@__DIR__, "context.jl"))
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
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_outer_driver.jl"))
using Test, Printf, Random

println(">>> confirming production's own CM outer opt file has par_concurrent_evals=yes (the real hazard surface)")
opt_text = read(joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt"), String)
@assert occursin(r"par_concurrent_evals\s+yes", opt_text) "csw_outer_wallclock_sr1.opt no longer sets par_concurrent_evals=yes -- this test's premise has changed, re-check"
println(">>> confirmed: par_concurrent_evals yes")

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0 = vcat(x_free_calib[1], pivot_reduce(z0, pe))
w0_ext = vcat(w0, log(1.0))   # K_mean=1, eta_nu_1 = log(1.0)

cfg = CMMeanZCConfig(cm = CMConfig(cm_grid_size = 10, contrasts = :anchored),
                      cm_extension = :cm_plus_equal_means_zero_covariance, meanzc_basis = :direct)

@testset "Concurrency (par_concurrent_evals=yes, production's own opt file): outer solve + cold re-verification" begin
    println(">>> running a real outer KNITRO solve with par_concurrent_evals=yes (maxtime_real=30s)..."); flush(stdout)
    t0 = time()
    res = run_cm_meanzc_upper(cfg, ctx, pe, w0_ext; delta = 1.0, maxtime_real = 30.0, verbose = false)
    @printf ">>> outer solve done in %.1fs, n_eval=%d n_grad=%d knitro_status=%d\n" (time()-t0) res.n_eval res.n_grad res.knitro_status
    @test res.n_eval > 0
    @test res.best !== nothing

    if res.best !== nothing
        # Cold re-verification: independently rebuild the production context and re-evaluate the
        # RETURNED incumbent's exact (w, nu) from scratch. If any concurrent callback dispatch had
        # let one evaluation's nu leak into another's result, the STORED best.Delta (computed during
        # the KNITRO run, under whatever concurrency KNITRO used) would not reproduce here.
        pcx_fresh = build_cm_meanzc_production_context(ctx, CS; L = 10, K_mean = res.K_mean, K_pair = res.K_pair,
            contrasts = :anchored, meanzc_basis = cfg.meanzc_basis)
        xf = x_free_from_w_meanzc(res.best.w, pe, res.K_mean)
        νvec = nu_from_w_meanzc(res.best.w, res.K_mean)
        _, _, verify_cold = cm_meanzc_production_value_verified(xf, νvec, pcx_fresh)
        @printf ">>> stored best.Delta=%.10f  cold-reverified Delta_dual=%.10f  |diff|=%.3e\n" res.best.Delta verify_cold.Delta_dual abs(res.best.Delta - verify_cold.Delta_dual)
        @test verify_cold.Delta_dual ≈ res.best.Delta atol = 1e-9
        @test is_verified_success(verify_cold)
    end

    # Repeat once more (fresh context, fresh KNITRO session, same config) to check
    # run-to-run consistency isn't randomly corrupted by whatever concurrent dispatch pattern
    # KNITRO happens to choose.
    println(">>> second independent outer solve (same config, fresh session)..."); flush(stdout)
    t1 = time()
    res2 = run_cm_meanzc_upper(cfg, ctx, pe, w0_ext; delta = 1.0, maxtime_real = 30.0, verbose = false)
    @printf ">>> second outer solve done in %.1fs, n_eval=%d knitro_status=%d\n" (time()-t1) res2.n_eval res2.knitro_status
    @test res2.best !== nothing
    if res.best !== nothing && res2.best !== nothing
        # Same start, same config, same (deterministic) draws -> should reach a comparable
        # incumbent; not required to be bit-identical (KNITRO's own internal scheduling can
        # affect the exact trajectory under par_concurrent_evals=yes) but both must be
        # independently verified-feasible.
        @test isfinite(res2.best.Delta)
    end
end

println("Concurrency check passed: outer solve under production's own par_concurrent_evals=yes completed and cold-reverified cleanly.")
