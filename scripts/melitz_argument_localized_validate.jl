# Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
# D") Section 3: correctness validation of the new argument-localized gradient backends
# (:B_argument_localized_serial, :B_argument_localized_parallel) against the already-
# validated :B_localized (and, transitively, full :B) at D=4 production scale.
#
# Requires BIT-IDENTICAL (`==`) K_jac/G_jac agreement at the base point, several random
# perturbations, and every free coordinate individually (not just aggregate equality --
# checks column-by-column so a localized bug in one coordinate's dependency handling cannot
# hide behind coincidental agreement elsewhere).
#
# Usage: JULIA_NUM_THREADS=<n> julia --project=. scripts/melitz_argument_localized_validate.jl

using Printf, Random
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function run_validation(; D=4, W=20_000, seed=29, n_random_points=5)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ
    n = length(theta0)
    d = ctx.moment_layout.num_moments
    W_ = size(obj.U, 1)
    println("D=$D  n=$n  K=$d  W=$W_  Threads.nthreads()=", Threads.nthreads(),
        "  Threads.maxthreadid()=", Threads.maxthreadid())

    mj_loc = make_melitz_moments_jacobian_b_localized(1e-4)
    mj_arg_serial = make_melitz_moments_jacobian_b_argument_localized_serial(1e-4)
    mj_arg_parallel = make_melitz_moments_jacobian_b_argument_localized_parallel(1e-4)

    rng = MersenneTwister(7)
    all_ok = true
    for trial in 0:n_random_points
        theta_probe = trial == 0 ? copy(theta0) : theta0 .+ 0.02 .* randn(rng, n)

        K1, G1 = zeros(W_, n), zeros(W_, d, n)
        K2, G2 = zeros(W_, n), zeros(W_, d, n)
        K3, G3 = zeros(W_, n), zeros(W_, d, n)
        mj_loc(K1, G1, theta_probe, obj.U, obj)
        mj_arg_serial(K2, G2, theta_probe, obj.U, obj)
        mj_arg_parallel(K3, G3, theta_probe, obj.U, obj)

        k_ok_serial = K1 == K2
        g_ok_serial = G1 == G2
        k_ok_parallel = K1 == K3
        g_ok_parallel = G1 == G3

        # per-coordinate column check (catches a bug hiding behind aggregate ==)
        per_coord_serial_ok = true
        per_coord_parallel_ok = true
        for k in 1:n
            if !(@views G1[:, :, k] == G2[:, :, k])
                per_coord_serial_ok = false
                @printf("  trial=%d coord=%d SERIAL MISMATCH max|diff|=%.3e\n", trial, k,
                    maximum(abs.(G1[:, :, k] .- G2[:, :, k])))
            end
            if !(@views G1[:, :, k] == G3[:, :, k])
                per_coord_parallel_ok = false
                @printf("  trial=%d coord=%d PARALLEL MISMATCH max|diff|=%.3e\n", trial, k,
                    maximum(abs.(G1[:, :, k] .- G3[:, :, k])))
            end
        end

        ok = k_ok_serial && g_ok_serial && k_ok_parallel && g_ok_parallel &&
             per_coord_serial_ok && per_coord_parallel_ok
        all_ok &= ok
        @printf("trial=%d bit-exact: serial=%s parallel=%s\n", trial,
            k_ok_serial && g_ok_serial && per_coord_serial_ok,
            k_ok_parallel && g_ok_parallel && per_coord_parallel_ok)
    end

    # small-N probe-call skip path (calculate_grad_k!'s own 2-draw call)
    K4, G4 = zeros(2, n), zeros(2, d, n)
    mj_arg_serial(K4, G4, theta0, @view(obj.U[1:2, :]), obj)
    small_n_ok = all(iszero, G4)
    @printf("small-N probe skip path (serial): all-zero G_jac = %s\n", small_n_ok)
    K5, G5 = zeros(2, n), zeros(2, d, n)
    mj_arg_parallel(K5, G5, theta0, @view(obj.U[1:2, :]), obj)
    small_n_ok_p = all(iszero, G5)
    @printf("small-N probe skip path (parallel): all-zero G_jac = %s\n", small_n_ok_p)

    println(all_ok && small_n_ok && small_n_ok_p ? "ALL CHECKS PASSED" : "*** FAILURES FOUND ***")
    return all_ok && small_n_ok && small_n_ok_p
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = run_validation()
    exit(ok ? 0 : 1)
end
