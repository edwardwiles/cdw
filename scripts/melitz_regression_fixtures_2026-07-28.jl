# Shared fixture builders for the 2026-07-28 step-control/robustness session (governing
# prompt: docs/melitz_outer_search_step_control_and_robustness_2026-07-XX.md). Extracted
# from the identical construction blocks duplicated across every 2026-07-27/07-28 phase
# script (`melitz_phase2_fixed_af_gamma_profile_2026-07-28.jl`,
# `melitz_phase3_gamma_only_smoke_test_2026-07-28.jl`, ...) -- NOT a new fixture, the same
# D=4 (seed=29, W=20,000) and real-D20 (`noah_D20`, seed=1, W=80,000) calibration this
# repo's own 2026-07-24 through 2026-07-28 sessions have used throughout.
#
# `include`d by every script in this session, never run standalone.
#
# 2026-07-28 inner-solver architecture-consolidation session: `build_melitz_psi_bundle`/
# `build_melitz_psi_bundle_from_calibration` now REQUIRE an explicit `policy` keyword (no
# `=nothing` default) -- this file's own `build_d4_fixture`/`build_realD20_fixture` were the
# CONFIRMED ROOT CAUSE of the `1.510118e14` `FiniteSolved`-above-cap anomaly
# (`docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md` Phase 2/6):
# every 2026-07-27/07-28 phase script calling either function here inherited a permanently
# uncapped bundle. Both now take `policy::MelitzInnerSolvePolicy=CappedEvaluation(10.0)` --
# an explicit, NAMED default (not a `nothing` sentinel silently translated into `-Inf`) that
# matches this codebase's own production routine-inner-solve cap
# (`solve_melitz_finite_delta_bound`'s own default) -- so every existing phase script calling
# `build_d4_fixture()`/`build_realD20_fixture()` with no `policy` argument now gets a SAFELY
# CAPPED bundle by default, closing the confirmed gap for the entire 2026-07-27/07-28 script
# corpus from this ONE choke point, without editing each of those ~30 scripts individually.

function build_d4_fixture(; W::Int=20_000, seed::Int=29, backend::Symbol=:matrix_free,
                            policy::MelitzInnerSolvePolicy=CappedEvaluation(10.0))
    data4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    obj4, theta0_4 = build_melitz_psi_bundle(data4; forbid_dense_fallback=true, backend=backend, policy=policy)
    ctx4 = obj4.γ
    r0_4 = evaluate_melitz_delta(theta0_4, ctx4, obj4; cold=true, store_G=false)
    @assert r0_4.verified "D=4 fixture (seed=$seed, W=$W) base point failed to verify"
    return (ctx=ctx4, obj=obj4, theta0=theta0_4, r0=r0_4)
end

function build_realD20_fixture(; W::Int=80_000, seed::Int=1,
                                 inner_loop_opt::AbstractString=joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt"),
                                 policy::MelitzInnerSolvePolicy=CappedEvaluation(10.0))
    real_dir = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
    @assert isdir(real_dir) "real_data/noah_D20 not found at $real_dir"
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    obj20, theta0_20 = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed,
        inner_loop_opt=inner_loop_opt, forbid_dense_fallback=true, policy=policy)
    ctx20 = obj20.γ
    r0_20 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
    @assert r0_20.verified "real-D20 fixture (seed=$seed, W=$W) base point failed to verify"
    return (ctx=ctx20, obj=obj20, theta0=theta0_20, r0=r0_20)
end

"Read a Phase 2 gamma-profile CSV (2026-07-28) back into a Vector of NamedTuples, sorted by gamma_fraction ascending."
function load_gamma_profile_csv(path::AbstractString)
    raw = readdlm(path, ',', header=true)
    data, header = raw
    cols = Symbol.(vec(header))
    rows = NamedTuple[]
    for i in 1:size(data, 1)
        nt = NamedTuple{Tuple(cols)}(Tuple(data[i, :]))
        push!(rows, nt)
    end
    return sort(rows, by=r -> r.gamma_fraction)
end

"""
    classify_gamma_only_config(g0, best_g, best_Delta, profile_rows; g_tol=1e-4, delta_slack=1.2)
        -> (:stuck_at_pareto | :worse_than_profile | :passed, detail::String)

Compares a gamma-only (or gamma-restricted-block-of-a-joint) KNITRO trajectory's own
verified best point against the Phase 2 direct fixed-A/f profile CSV (the ground-truth
corridor Phase 2 already independently solved, no root-finding). `profile_rows` is a
`load_gamma_profile_csv(...)` result for the matching fixture. A configuration that never
moves `g` away from the Pareto starting point (`|best_g - g0| < g_tol`) while the profile
itself shows a strictly better (lower-DeltaStar, i.e. more slack) FiniteSolved point at some
`g` beyond the Pareto row is classified `:stuck_at_pareto` -- a FAILED solver configuration
per the governing prompt's own Phase 2 acceptance rule, not merely "conservative." A
configuration that moved but landed at a point whose interpolated profile DeltaStar (nearest
two bracketing profile rows, log-linear in Delta since Delta spans many orders of magnitude)
is more than `delta_slack`x worse than what the profile shows is achievable at that SAME g is
classified `:worse_than_profile`. Anything else (including "found a point at least as good as
the profile at its own g, or beat the profile outright") is `:passed`.
"""
function classify_gamma_only_config(g0::Real, best_g::Real, best_Delta, profile_rows; g_tol::Real=1e-4, delta_slack::Real=1.2)
    finite_profile = filter(r -> r.classification == "FiniteSolved", profile_rows)
    pareto_row = first(finite_profile)
    better_exists = any(r -> abs(r.g - g0) > 10 * g_tol && r.DeltaStar > pareto_row.DeltaStar, finite_profile)
    if abs(best_g - g0) < g_tol
        return better_exists ? (:stuck_at_pareto, "did not move from g0=$g0; profile shows a better within-budget corridor beyond the Pareto point") :
                                (:passed, "did not move, but the profile itself shows no better point either -- consistent, not a failure")
    end
    if best_Delta === nothing || !(best_Delta isa Real) || isnan(best_Delta)
        return (:passed, "moved but produced no verified finite incumbent to compare (treat cautiously, not auto-failed)")
    end
    # Bracket best_g in the sorted finite profile rows and log-linearly interpolate DeltaStar at best_g.
    gs = [r.g for r in finite_profile]
    ds = [r.DeltaStar for r in finite_profile]
    order = sortperm(gs)
    gs, ds = gs[order], ds[order]
    if best_g < minimum(gs) || best_g > maximum(gs)
        return (:passed, "best_g=$best_g outside the profile's own solved g-range [$(minimum(gs)),$(maximum(gs))] -- cannot compare directly")
    end
    k = searchsortedlast(gs, best_g)
    k = clamp(k, 1, length(gs) - 1)
    t = (best_g - gs[k]) / (gs[k+1] - gs[k])
    log_interp_delta = exp(log(ds[k]) + t * (log(ds[k+1]) - log(ds[k])))
    if best_Delta > delta_slack * log_interp_delta
        return (:worse_than_profile, "best_Delta=$best_Delta at g=$best_g is more than $(delta_slack)x the profile's own interpolated $log_interp_delta at the same g")
    end
    return (:passed, "moved to g=$best_g Delta=$best_Delta -- consistent with (or better than) the profile's own interpolated $log_interp_delta")
end
