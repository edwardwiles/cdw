# Continuation session (2026-07-23), Section 11: D=10/D=20 fixed-point inner-solver
# microbenchmarks. Per the governing prompt's explicit instruction, this does NOT launch a
# long outer campaign -- it builds a representative D=10 (K=101 economic moments) and D=20
# (K=401 economic moments) fixture each and benchmarks ONE cold inner CC dual solve at the
# fixture's own true (population-Pareto) point, varying BLAS thread count only (Julia
# coordinate parallelism is inactive here -- there is no outer coordinate loop in a single
# fixed-point inner solve).
#
# Usage: julia --project=. scripts/melitz_d10_d20_inner_microbenchmark.jl

using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function bench_one(D, W; blas_threads=(1, 2, 4, 8, 16, 20))
    println("\n", "="^100)
    @printf("D=%d  W=%d  K=%d economic moments\n", D, W, D^2 + 1)
    println("="^100)
    # Continuation session note: the fixture generator's default `min_participation_prob=0.01`
    # gate is tuned for D=4 and is essentially ALWAYS violated at D=10/D=20 with the default
    # tau/f/A calibration (confirmed systematically, not a seed-luck issue: 14/15 tested seeds
    # at D=10 failed this gate, clustering at 0.004-0.0097, well below 0.01) -- a genuine
    # finding that this generator's default calibration does not scale to larger D without
    # retuning, not a bug in this script. Relaxed to 0.002 HERE ONLY, for this diagnostic
    # inner-solver microbenchmark's own purposes (which needs a valid, gravity-exact fixture
    # of the right SIZE, not necessarily one recalibrated to the same participation-probability
    # standard as the D=4 economic fixture) -- not used for any economic/outer-search result.
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=W,
        min_participation_prob=(D == 4 ? 0.01 : 0.002))
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    # jac_h fix (continuation4, cc_algo/PsiObjectiveBundle.jl): this script is a pure fixed-theta
    # inner-solve microbenchmark (never an outer theta-gradient search on this bundle) -- the
    # unconditional dense jac_h this bundle used to allocate was the actual root cause of this
    # exact script hanging/thrashing at D=20 (docs/melitz_optimization_report_2026-07-23_continuation3.md
    # Section 9). needs_outer_moment_jacobian=false skips that dead allocation entirely.
    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt, needs_outer_moment_jacobian=false)
    ctx = obj.γ

    rows = NamedTuple[]
    for nb in blas_threads
        BLAS.set_num_threads(nb)
        # cold solve: clear warm-start cache, force a fresh KNITRO attempt from the default start.
        obj.use_cached_x = false
        obj.x .= NaN
        CS = CounterfactualSensitivity
        CS.reset_jac_h_counters!()
        t0 = time()
        local nStatus, x
        bytes = @allocated begin
            _, x, nStatus = CS.inner_loop_internal(obj, theta0)
        end
        wall = time() - t0
        accepted = nStatus in (0, -100, -101, -103)
        counters = try
            CS.jac_h_counters_snapshot()
        catch
            nothing
        end
        @printf("  BLAS threads=%3d  wall=%.4fs  nStatus=%d accepted=%s  bytes=%d\n",
            nb, wall, nStatus, accepted, bytes)
        push!(rows, (D=D, W=W, blas_threads=nb, wall=wall, nStatus=nStatus,
            accepted=accepted, bytes=bytes, counters=counters))
    end
    BLAS.set_num_threads(1)   # restore this repo's standing hard-cap default
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    all_rows = NamedTuple[]
    append!(all_rows, bench_one(10, 20_000))
    append!(all_rows, bench_one(20, 20_000))
    append!(all_rows, bench_one(20, 80_000; blas_threads=(1, 2, 4, 8, 16, 20)))
    println("\n", "="^100)
    println("SUMMARY")
    println("="^100)
    @printf("%-6s %-10s %-14s %-10s %-10s %-14s\n", "D", "W", "blas_threads", "wall(s)", "nStatus", "bytes")
    for r in all_rows
        @printf("%-6d %-10d %-14d %-10.4f %-10d %-14d\n", r.D, r.W, r.blas_threads, r.wall, r.nStatus, r.bytes)
    end
end
