# Governing prompt (2026-07-28 outer-search gamma-profile session), Phase 3: gamma-only
# KNITRO smoke test against the Phase 2 direct fixed-A/f profile.
#
# Restricts the outer NLP to gamma-only movement by setting the A/f block of theta_box to
# EXACTLY 0 (KNITRO sees those coordinates as fixed variables, lb==ub==theta0 value) --
# A/f stay on the identical fixed-A/f path Phase 2 used, so the KNITRO trajectory's own
# best feasible point is directly comparable to the Phase 2 profile at the SAME g.
#
# Three configurations, same starting point/options/cap/algorithm each:
#   unscaled          -- var_scale=nothing, objective_scale=nothing (KNITRO's raw defaults)
#   scaled_no_objscale -- var_scale=[s_g,1,1,...] (Phase 7's own s_g=1e-4), objective_scale=nothing
#                         (this session's Phase 1 hypothesis: objective stays raw theta[1],
#                         so KNITRO's own internal scaled-space gradient is only +-s_g)
#   scaled_with_objscale -- SAME var_scale, objective_scale=s_g (this session's Phase 1 fix:
#                         restores an order-1 scaled-space objective gradient)
#
# Algorithm: Active Set (Phase 8's own 2026-07-27 selection -- the one combination that
# reached genuine xtol convergence without runaway at both D=4 and real D=20 last session).
#
# Usage: julia --project=. -t 20 scripts/melitz_phase3_gamma_only_smoke_test_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)
const S_G = 1e-4
const CAP = 10.0

function gamma_only_box(n::Int, g_radius::Real)
    box = zeros(n)
    box[1] = g_radius
    return box
end

function run_config(label, ctx, obj, theta0, delta, direction, g_radius, inner_opt, outer_opt; var_scale, objective_scale)
    n = length(theta0)
    box = gamma_only_box(n, g_radius)
    vc = var_scale === nothing ? nothing : collect(Float64.(theta0))
    local res
    wall = @elapsed begin
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            delta_evaluation_cap=CAP, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
            theta_box=box, cutoff_constraint_backend=:linear,
            inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
            var_scale=var_scale, var_center=vc,
            backend=:matrix_free, forbid_dense_fallback=true,
            objective_scale=objective_scale)
    end
    cv = res.cold_verified_incumbent
    best_g = cv === nothing ? theta0[1] : cv.eval.theta_free[1]
    best_Delta = cv === nothing ? NaN : cv.eval.Delta
    moved = best_g != theta0[1]
    @printf("  [%-22s] nStatus=%5d wall=%6.2fs n_fc=%3d n_ga=%3d n_above_cap=%3d  g0=%.6f -> best_g=%.6f (moved=%s)  Delta=%s\n",
        label, res.nStatus, wall, res.n_fc_calls, res.n_ga_calls, res.n_above_cap_reject,
        theta0[1], best_g, moved, best_Delta === NaN ? "NA" : @sprintf("%.4e", best_Delta))
    return (config=label, nStatus=res.nStatus, wall=wall, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls,
        n_above_cap=res.n_above_cap_reject, n_numerical_failure=res.n_numerical_failure_reject,
        g0=theta0[1], best_g=best_g, moved=moved, best_Delta=best_Delta,
        terminal_g=res.terminal_eval.theta_free[1])
end

function main()
    BLAS.set_num_threads(1)
    rows = NamedTuple[]

    println("="^100); println("D=4, delta=1e-2, upper, gamma-only, Active Set"); println("="^100)
    data4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    obj4, theta0_4 = build_melitz_psi_bundle(data4; forbid_dense_fallback=true, backend=:matrix_free)
    ctx4 = obj4.γ
    n4 = length(theta0_4)
    r0_4 = evaluate_melitz_delta(theta0_4, ctx4, obj4; cold=true, store_G=false)
    @assert r0_4.verified
    vs4 = ones(n4); vs4[1] = S_G
    inner_opt4 = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    outer_active = joinpath(REPO, "melitz_outer_finite_delta_alg_active_2026-07-27.opt")
    @assert isfile(outer_active) "expected Phase 8's own Active-Set outer .opt file at $outer_active"

    push!(rows, run_config("D4_unscaled", ctx4, obj4, theta0_4, 1e-2, :upper, 0.3, inner_opt4, outer_active;
        var_scale=nothing, objective_scale=nothing))
    push!(rows, run_config("D4_scaled_no_objscale", ctx4, obj4, theta0_4, 1e-2, :upper, 0.3, inner_opt4, outer_active;
        var_scale=vs4, objective_scale=nothing))
    push!(rows, run_config("D4_scaled_with_objscale", ctx4, obj4, theta0_4, 1e-2, :upper, 0.3, inner_opt4, outer_active;
        var_scale=vs4, objective_scale=S_G))

    println("\n" * "="^100); println("real D=20, delta=1.0, upper, gamma-only, Active Set"); println("="^100)
    real_dir = joinpath(REPO, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    BLAS.set_num_threads(20)
    inner_opt20 = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    obj20, theta0_20 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=inner_opt20, forbid_dense_fallback=true)
    ctx20 = obj20.γ
    n20 = length(theta0_20)
    r0_20 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
    @assert r0_20.verified
    vs20 = ones(n20); vs20[1] = S_G

    push!(rows, run_config("D20_unscaled", ctx20, obj20, theta0_20, 1.0, :upper, 0.15, inner_opt20, outer_active;
        var_scale=nothing, objective_scale=nothing))
    push!(rows, run_config("D20_scaled_no_objscale", ctx20, obj20, theta0_20, 1.0, :upper, 0.15, inner_opt20, outer_active;
        var_scale=vs20, objective_scale=nothing))
    push!(rows, run_config("D20_scaled_with_objscale", ctx20, obj20, theta0_20, 1.0, :upper, 0.15, inner_opt20, outer_active;
        var_scale=vs20, objective_scale=S_G))
    BLAS.set_num_threads(1)

    outfile = joinpath(OUTDIR, "melitz_phase3_gamma_only_smoke_test_2026-07-28.csv")
    open(outfile, "w") do io
        cols = keys(rows[1])
        println(io, join(cols, ","))
        for r in rows
            println(io, join([r[c] for c in cols], ","))
        end
    end
    println("\nDONE. CSV written to ", outfile)
    return rows
end

main()
