# Phase II.11/12 (screening-continuation session): wall-time/allocation benchmark of
# :method_b_localized vs full :method_b, at production scale (D=4/W=20,000), now that both
# correctness gates (dependency-map superset, bit-exact Jacobian match) have passed.
#
# Usage: julia --project=. scripts/melitz_localized_gradient_benchmark.jl

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

    mj_full = make_melitz_moments_jacobian_b(1e-4)
    mj_loc = make_melitz_moments_jacobian_b_localized(1e-4)

    rng = MersenneTwister(11)
    theta_probe = theta0 .+ 0.02 .* randn(rng, n)

    K1, G1 = zeros(W, n), zeros(W, d, n)
    K2, G2 = zeros(W, n), zeros(W, d, n)
    # warm up (JIT) both paths first
    mj_full(K1, G1, theta_probe, obj.U, obj)
    mj_loc(K2, G2, theta_probe, obj.U, obj)
    @printf("bit-exact at production scale (D=4,W=%d): K match=%s G match=%s\n", W, K1 == K2, G1 == G2)

    n_reps = 5
    t_full = @elapsed for _ in 1:n_reps
        mj_full(K1, G1, theta_probe, obj.U, obj)
    end
    t_loc = @elapsed for _ in 1:n_reps
        mj_loc(K2, G2, theta_probe, obj.U, obj)
    end
    bytes_full = @allocated mj_full(K1, G1, theta_probe, obj.U, obj)
    bytes_loc = @allocated mj_loc(K2, G2, theta_probe, obj.U, obj)

    @printf("\nfull   :method_b          : %.4fs/call  (%d reps, %.4fs total)  %d bytes/call\n",
        t_full / n_reps, n_reps, t_full, bytes_full)
    @printf("localized :method_b_localized: %.4fs/call  (%d reps, %.4fs total)  %d bytes/call\n",
        t_loc / n_reps, n_reps, t_loc, bytes_loc)
    @printf("\nspeedup: %.2fx wall, %.2fx fewer bytes\n", t_full / t_loc, bytes_full / max(1, bytes_loc))

    # dependency-map statistics: average number of cells touched per coordinate vs the full D^2
    depmap = melitz_localized_dependency_map(ctx)
    avg_cells = sum(length(dep.cells) for dep in depmap) / length(depmap)
    n_link = count(dep -> dep.touches_link, depmap)
    @printf("\ndependency map: avg %.2f cells/coordinate (full D^2=%d), %d/%d coordinates touch the link column\n",
        avg_cells, ctx.D^2, n_link, length(depmap))
end
