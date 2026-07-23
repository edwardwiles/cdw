# Integration smoke test for cm_meanzc_outer_driver.jl / cm_meanzc_config.jl: a SHORT (not
# converged) real KNITRO outer solve at D=4, both cm_extension=:cm_only (byte-identical
# delegation check against run_cm_upper) and cm_extension=:cm_plus_equal_means_zero_covariance
# (the new extended path). Not a scientific trial (no claim about a converged kappa) -- proves
# the assembled config/driver/production/moments stack actually runs end-to-end through a real
# KNITRO outer loop, matching the task brief's "current production call path" requirement for
# the disabled (:cm_only) arm.
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
using Test, Printf

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0 = vcat(x_free_calib[1], pivot_reduce(z0, pe))

const MAXTIME = 15.0   # short -- smoke test only, no convergence claim

println("="^100)
println("Smoke test A: cm_extension=:cm_only delegates byte-identically to run_cm_upper")
println("="^100)
@testset "cm_only delegation" begin
    # NOTE: run_cm_meanzc_upper's :cm_only branch calls run_cm_upper directly (verified by code
    # inspection -- one line, no wrapper logic in between) so "byte-identical delegation" is a
    # static code fact, not something to re-derive empirically. What IS worth checking
    # empirically is bounded by maxtime_real being a WALL-CLOCK budget: two SEPARATE timed runs
    # of the identical deterministic algorithm generally do NOT reach the same n_eval/xsol (the
    # second benefits from JIT warmup from the first, confirmed live: 29 vs 342 evals in the
    # "same" 15s budget) -- that is expected KNITRO/wall-clock behavior, not evidence of
    # divergent code paths. Compare only the FIRST callback evaluation (at w0 itself, before any
    # timing-dependent iteration count enters), which IS deterministic and must match exactly.
    cfg_off = CMMeanZCConfig(cm = CMConfig(cm_grid_size = 10, contrasts = :anchored), cm_extension = :cm_only)
    res_via_meanzc_driver = run_cm_meanzc_upper(cfg_off, ctx, pe, w0; delta = 1.0, maxtime_real = MAXTIME, verbose = false)

    pcx0 = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored)
    res_direct = run_cm_upper(pcx0, ctx, pe, w0; delta = 1.0, maxtime_real = MAXTIME, verbose = false)

    @test res_via_meanzc_driver.n_eval > 0 && res_direct.n_eval > 0
    @test isfinite(res_via_meanzc_driver.trace[1].Delta) && isfinite(res_direct.trace[1].Delta)
    @test res_via_meanzc_driver.trace[1].gp == res_direct.trace[1].gp == w0[1]
    @test res_via_meanzc_driver.trace[1].Delta ≈ res_direct.trace[1].Delta atol = 1e-10
    @test res_via_meanzc_driver.trace[1].feasible == res_direct.trace[1].feasible
    @printf "  meanzc-driver: n_eval=%d kappa=%s knitro_status=%d | direct: n_eval=%d kappa=%s knitro_status=%d (eval COUNTS differ because maxtime_real is wall-clock, not a bug -- see note above)\n" res_via_meanzc_driver.n_eval string(res_via_meanzc_driver.kappa) res_via_meanzc_driver.knitro_status res_direct.n_eval string(res_direct.kappa) res_direct.knitro_status
end
println()

println("="^100)
println("Smoke test B: cm_extension=:cm_plus_equal_means_zero_covariance runs end-to-end")
println("="^100)
@testset "cm_plus_equal_means_zero_covariance short outer run" begin
    cfg_zc = CMMeanZCConfig(cm = CMConfig(cm_grid_size = 10, contrasts = :anchored),
                             cm_extension = :cm_plus_equal_means_zero_covariance, meanzc_basis = :direct)
    K_mean, K_pair = meanzc_resolve_K(cfg_zc)
    @test (K_mean, K_pair) == (1, 1)

    w0_ext = vcat(w0, log(1.0))   # eta_nu_1 = log(1.0), matches Exp(1)'s theoretical mean
    res = run_cm_meanzc_upper(cfg_zc, ctx, pe, w0_ext; delta = 1.0, maxtime_real = MAXTIME, verbose = false)

    @test res.n_eval > 0
    @test res.K_mean == 1 && res.K_pair == 1
    @test length(res.nu_bounds) == 1
    @test res.best !== nothing   # a real KNITRO run at delta=1.0 from a near-calibration start should find SOME feasible point
    if res.best !== nothing
        @test isfinite(res.kappa)
        @printf "  n_eval=%d  n_grad=%d  kappa=%s  best.Delta=%.6f  nu_box_interior=%s\n" res.n_eval res.n_grad string(res.kappa) res.best.Delta string(res.nu_box_interior)
    end
end

println()
println("All driver/config smoke tests passed (short runs, no convergence claim).")
