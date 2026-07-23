# Phase II.10 (screening-session continuation): reproduce the outer-gradient floor under
# the CURRENT production configuration (post Phase I screening fixes), confirming Method
# B's ~30 free coordinates x plus/minus = 60 displaced hard-moment evaluations per gradient,
# and the resulting ~1s/gradient x ~26 gradients per short trajectory floor.
#
# Usage: julia --project=. scripts/melitz_gradient_floor_reproduction.jl

using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    MELITZ_PROFILE[] = true
    melitz_profile_reset!()

    data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ
    n_free = length(theta0)
    @printf("free coordinates: %d  (Method B: %d displaced moment builds/gradient)\n", n_free, 2 * n_free)

    t0 = time()
    res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=1e-2, direction=:upper,
        gradient_backend=:B, theta_box=0.10, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
        cutoff_constraint_backend=:linear)
    wall = time() - t0
    @printf("trajectory wall=%.2fs n_ga_calls=%d n_fc_calls=%d\n", wall, res.n_ga_calls, res.n_fc_calls)

    rows = melitz_profile_report(; trajectory_total_s=wall)
    ga = get(MELITZ_PROF.stats, :ga_divergence_gradient, nothing)
    if ga !== nothing && ga.count > 0
        @printf("\nga_divergence_gradient: count=%d total_s=%.3f mean_ms=%.1f (= %.3fs/gradient)\n",
            ga.count, ga.total_ns / 1e9, (ga.total_ns / ga.count) / 1e6, (ga.total_ns / ga.count) / 1e9)
        @printf("implied gradient-floor share of trajectory wall: %.1f%%\n", 100 * (ga.total_ns / 1e9) / wall)
    end
end
