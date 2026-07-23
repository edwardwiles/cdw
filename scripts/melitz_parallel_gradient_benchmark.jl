# Phase II.12 (this continuation session): wall-time/allocation/scaling benchmark of
# :method_b_localized_parallel vs the serial :method_b_localized and full :method_b, at
# production scale (D=4/W=20,000), for whatever Threads.nthreads() this Julia process was
# launched with -- run this script once per desired thread count via
# JULIA_NUM_THREADS=<n> julia --project=. scripts/melitz_parallel_gradient_benchmark.jl
# (a single Julia process cannot change its own thread pool size at runtime, so the thread-
# count sweep itself is driven externally, e.g. scripts/melitz_parallel_gradient_sweep.sh).
#
# Requires BIT-IDENTICAL (`==`, not `isapprox`) output vs. the serial localized backend --
# no floating-point-order-dependent reduction across threads, since each coordinate's own
# output column is computed independently from the shared, read-only base state.

using Printf, Random
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ
    n = length(theta0)
    d = ctx.moment_layout.num_moments
    W = size(obj.U, 1)
    nt = Threads.nthreads()

    mj_full = make_melitz_moments_jacobian_b(1e-4)
    mj_loc = make_melitz_moments_jacobian_b_localized(1e-4)
    mj_par = make_melitz_moments_jacobian_b_localized_parallel(1e-4)

    rng = MersenneTwister(11)
    theta_probe = theta0 .+ 0.02 .* randn(rng, n)

    K1, G1 = zeros(W, n), zeros(W, d, n)
    K2, G2 = zeros(W, n), zeros(W, d, n)
    K3, G3 = zeros(W, n), zeros(W, d, n)
    # warm up (JIT) all three paths first
    mj_full(K1, G1, theta_probe, obj.U, obj)
    mj_loc(K2, G2, theta_probe, obj.U, obj)
    mj_par(K3, G3, theta_probe, obj.U, obj)
    exact_vs_loc = K2 == K3 && G2 == G3
    exact_vs_full = K1 == K3 && G1 == G3
    @printf("threads=%d  bit-exact parallel vs serial-localized: %s   vs full-B: %s\n",
        nt, exact_vs_loc, exact_vs_full)
    exact_vs_loc || error("PARALLEL BACKEND NOT BIT-EXACT vs serial localized -- must not proceed")

    n_reps = 7
    t_full = @elapsed for _ in 1:n_reps
        mj_full(K1, G1, theta_probe, obj.U, obj)
    end
    t_loc = @elapsed for _ in 1:n_reps
        mj_loc(K2, G2, theta_probe, obj.U, obj)
    end
    t_par = @elapsed for _ in 1:n_reps
        mj_par(K3, G3, theta_probe, obj.U, obj)
    end
    bytes_par = @allocated mj_par(K3, G3, theta_probe, obj.U, obj)
    gc_before = Base.gc_num()
    mj_par(K3, G3, theta_probe, obj.U, obj)
    gc_after = Base.gc_num()

    @printf("\n  full   :method_b                  : %.4fs/call  (%d reps)\n", t_full / n_reps, n_reps)
    @printf("  serial :method_b_localized         : %.4fs/call  (%d reps)\n", t_loc / n_reps, n_reps)
    @printf("  parallel :method_b_localized_parallel (threads=%d): %.4fs/call  (%d reps)  %d bytes/call\n",
        nt, t_par / n_reps, n_reps, bytes_par)
    @printf("\n  speedup parallel vs serial-localized : %.2fx\n", t_loc / t_par)
    @printf("  speedup parallel vs full-B            : %.2fx\n", t_full / t_par)
    @printf("  efficiency (speedup/threads)           : %.1f%%\n", 100 * (t_loc / t_par) / nt)
    println(@sprintf("THREADCOUNT_RESULT nthreads=%d t_loc=%.6f t_par=%.6f speedup=%.4f efficiency_pct=%.2f bytes=%d",
        nt, t_loc / n_reps, t_par / n_reps, (t_loc / t_par), 100 * (t_loc / t_par) / nt, bytes_par))
end
