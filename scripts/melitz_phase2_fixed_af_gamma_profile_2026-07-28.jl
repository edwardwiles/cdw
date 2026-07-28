# Governing prompt (2026-07-28 outer-search gamma-profile session), Phase 2: mandatory
# fixed-A/f gamma profile at D=4 and real D=20 -- the central diagnostic this session's
# governing prompt requires before any outer-KNITRO search result may be trusted.
#
# "Fixed A/f, vary gamma" -- same derivation as the 2026-07-24 companion report
# (docs/melitz_real_d20_outer_benchmark_2026-07-24.md Section 6.0, scripts/
# melitz_real_d20_fixed_af_profile_2026-07-24.jl): under the active :logf outer
# parameterization, theta_free = (g=log(gamma_prime_j), A_free[1:D^2-1], f_free[1:D^2-2]).
# A is reconstructed from A_free ALONE; every off-domestic-focal free f cell from f_free
# ALONE; only f[j,j] depends on gamma (derive_fjj_from_autarky_cutoff, the zhat'_jj=1
# normalization this governing prompt explicitly requires NOT be frozen). So holding
# theta_free[2:end] fixed at the calibrated value while moving theta_free[1]=g alone IS the
# fixed-A/f restriction -- no new coordinate system needed.
#
# Analytical endpoint (Delta->infinity upper GT bound), reusing the closed form already
# derived+validated 2026-07-24 (scripts/melitz_real_d20_theoretical_gamma_bound_2026-07-24.jl):
#   kappa_min  = lambda_jj^(1/(sigma-1))      [lambda_jj = domestic trade share at theta0]
#   g_ceiling  = log( (kappa_min / (w_prime/w[target]))^(sigma-1) )
# kappa_of_g(g) = (w_prime/w[target]) * exp(g)^(1/(sigma-1)) satisfies kappa_of_g(g_ceiling)
# == kappa_min exactly. No root-finding is used anywhere in this script -- the grid below
# is a PREDETERMINED, closed-form fraction-of-welfare-distance mapping, evaluated directly.
#
# Profile evaluation cap = 50 (governing prompt's own suggested diagnostic cap, wider than
# the production outer cap of 10, since this is a one-shot 12-point curve, not a repeated
# per-outer-iteration screen). Continuation warm start: grid points are evaluated in
# ambition order (fraction 0 -> 1, i.e. least-extreme g first), reusing obj's own
# obj.x/obj.use_cached_x warm-start state across calls (melitz_classified_inner_solve's
# default warm_start_source=:previous) plus the MelitzDualBank's own continuation/screening
# state -- never re-cold-started mid-grid, matching the governing prompt's explicit "only
# reuse duals from fully solved finite points, no cold retry" instruction.
#
# Usage:
#   julia --project=. -t 20 scripts/melitz_phase2_fixed_af_gamma_profile_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const PROFILE_CAP = 50.0
const FRACTIONS = [0.00, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.995, 1.00]

kappa_of_g(g::Real, wratio::Real, sigma::Real) = wratio * exp(g)^(1 / (sigma - 1))
g_of_kappa(kappa::Real, wratio::Real, sigma::Real) = log((kappa / wratio)^(sigma - 1))

"Domestic trade share lambda_jj and w_prime/w[target] at theta0, computed from the model's own baseline equilibrium (fixture-agnostic: works for the leaked-DGP D=4 path and the calibrated real-D20 path identically, since both build a MelitzOuterCtx with the same fields)."
function wage_ratio_and_lambda_jj(theta0, ctx)
    state = melitz_outer_state(theta0, ctx)
    j = ctx.target_country
    lambda_jj = state.equilibrium.trade_flow[j, j] / state.equilibrium.expenditure[j]
    wratio = 1.0 / ctx.w[j]   # w_prime (autarky counterfactual wage, normalized to 1) / w[target]
    return wratio, lambda_jj
end

struct ProfileRow
    fixture::String
    gamma_fraction::Float64
    g::Float64
    gamma_prime::Float64
    wage_ratio::Float64
    kappa_ratio::Float64
    GT::Float64
    DeltaStar::Float64
    classification::String
    inner_status::Int
    inner_iterations::Int
    obj_callbacks::Int
    grad_callbacks::Int
    hess_callbacks::Int
    wall_s::Float64
    cutoff_min_slack::Float64
    gravity_residual_A::Float64
    gravity_residual_f::Float64
    moment_residual::Float64
    primal_dual_gap::Float64
end

function run_profile(fixture::String, ctx, obj, theta0::Vector{Float64}, sigma::Real)
    j = ctx.target_country
    wratio, lambda_jj = wage_ratio_and_lambda_jj(theta0, ctx)
    kappa_pareto = kappa_of_g(theta0[1], wratio, sigma)
    kappa_min = lambda_jj^(1 / (sigma - 1))
    g_ceiling = g_of_kappa(kappa_min, wratio, sigma)
    @printf("[%s] theta0[1]=g_pareto=%.6f  kappa_pareto=%.6f  GT_pareto=%.6f\n", fixture, theta0[1], kappa_pareto, 1 - kappa_pareto)
    @printf("[%s] analytical ceiling: lambda_jj=%.6f  kappa_min=%.6f  GT_ceiling=%.6f  g_ceiling=%.6f\n",
        fixture, lambda_jj, kappa_min, 1 - kappa_min, g_ceiling)

    bank = MelitzDualBank(16)
    rows = ProfileRow[]
    for frac in FRACTIONS
        kappa_f = kappa_pareto + frac * (kappa_min - kappa_pareto)
        g_f = frac == 0.0 ? theta0[1] : g_of_kappa(kappa_f, wratio, sigma)
        theta = copy(theta0); theta[1] = g_f
        c0 = melitz_backend_counters_snapshot()
        t0 = time()
        result = melitz_classified_inner_solve(obj, theta, ctx; delta_evaluation_cap=PROFILE_CAP, bank=bank)
        wall = time() - t0
        c1 = melitz_backend_counters_snapshot()
        objcb = c1.matrix_free_objective_calls - c0.matrix_free_objective_calls
        gradcb = c1.matrix_free_gradient_calls - c0.matrix_free_gradient_calls
        hesscb = c1.matrix_free_hessian_calls - c0.matrix_free_hessian_calls

        if result isa FiniteSolved
            ev = evaluate_melitz_delta_from_solution(theta, ctx, obj, result.Delta, result.x, result.nStatus)
            kappa_r = kappa_of_g(g_f, wratio, sigma)
            row = ProfileRow(fixture, frac, g_f, exp(g_f), wratio, kappa_r, 1 - kappa_r, result.Delta,
                "FiniteSolved", result.nStatus, objcb, objcb, gradcb, hesscb, wall,
                ev.min_slack, ev.equilibrium_check === nothing ? NaN : ev.equilibrium_check.gravity_residual_A,
                ev.equilibrium_check === nothing ? NaN : ev.equilibrium_check.gravity_residual_f,
                maximum(abs, ev.moment_residuals), ev.primal_dual_gap)
            @printf("  frac=%.3f g=%9.5f kappa=%.6f GT=%.6f  Delta=%.4e  FiniteSolved nStatus=%d  wall=%.2fs\n",
                frac, g_f, kappa_r, 1 - kappa_r, result.Delta, result.nStatus, wall)
        elseif result isa AboveEvaluationCap
            kappa_r = kappa_of_g(g_f, wratio, sigma)
            row = ProfileRow(fixture, frac, g_f, exp(g_f), wratio, kappa_r, 1 - kappa_r, NaN,
                "AboveEvaluationCap50", -1, objcb, objcb, gradcb, hesscb, wall,
                NaN, NaN, NaN, NaN, NaN)
            @printf("  frac=%.3f g=%9.5f kappa=%.6f GT=%.6f  AboveEvaluationCap50 (lower_bound=%.4e, source=%s)  wall=%.2fs\n",
                frac, g_f, kappa_r, 1 - kappa_r, result.certified_lower_bound, result.source, wall)
        elseif result isa InfiniteDeltaCertified
            kappa_r = kappa_of_g(g_f, wratio, sigma)
            row = ProfileRow(fixture, frac, g_f, exp(g_f), wratio, kappa_r, 1 - kappa_r, NaN,
                "InfiniteDeltaCertified", -2, objcb, objcb, gradcb, hesscb, wall,
                NaN, NaN, NaN, NaN, NaN)
            @printf("  frac=%.3f g=%9.5f kappa=%.6f GT=%.6f  InfiniteDeltaCertified (col=%d, [%.3e,%.3e], kind=%s)  wall=%.2fs\n",
                frac, g_f, kappa_r, 1 - kappa_r, result.column, result.lo, result.hi, result.kind, wall)
        else # NumericalFailure
            kappa_r = kappa_of_g(g_f, wratio, sigma)
            row = ProfileRow(fixture, frac, g_f, exp(g_f), wratio, kappa_r, 1 - kappa_r, NaN,
                "NumericalFailure", result.nStatus, objcb, objcb, gradcb, hesscb, wall,
                NaN, NaN, NaN, NaN, NaN)
            @printf("  frac=%.3f g=%9.5f kappa=%.6f GT=%.6f  NumericalFailure nStatus=%d  wall=%.2fs\n",
                frac, g_f, kappa_r, 1 - kappa_r, result.nStatus, wall)
        end
        push!(rows, row)
        flush(stdout)
    end
    return rows
end

function write_csv(path, rows::Vector{ProfileRow})
    open(path, "w") do io
        println(io, "fixture,gamma_fraction,g,gamma_prime,wage_ratio,kappa_ratio,GT,DeltaStar,classification,inner_status,inner_iterations,objective_callbacks,gradient_callbacks,hessian_callbacks,wall_s,cutoff_min_slack,gravity_residual_A,gravity_residual_f,moment_residual,primal_dual_gap")
        for r in rows
            @printf(io, "%s,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8e,%s,%d,%d,%d,%d,%d,%.4f,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                r.fixture, r.gamma_fraction, r.g, r.gamma_prime, r.wage_ratio, r.kappa_ratio, r.GT, r.DeltaStar,
                r.classification, r.inner_status, r.inner_iterations, r.obj_callbacks, r.grad_callbacks, r.hess_callbacks,
                r.wall_s, r.cutoff_min_slack, r.gravity_residual_A, r.gravity_residual_f, r.moment_residual, r.primal_dual_gap)
        end
    end
end

function main()
    BLAS.set_num_threads(1)
    all_rows = ProfileRow[]

    println("="^100); println("D=4 FIXTURE (seed=29, W=20,000)"); println("="^100)
    data4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    cfg4 = MelitzInnerSolveConfig(:diagnostic; delta_evaluation_cap=PROFILE_CAP)
    obj4, theta0_4 = build_melitz_psi_bundle(data4; forbid_dense_fallback=true, inner_solve_config=cfg4)
    ctx4 = obj4.γ
    r0_4 = evaluate_melitz_delta(theta0_4, ctx4, obj4; cold=true, store_G=false)
    @assert r0_4.verified "D=4 base point failed to verify"
    rows4 = run_profile("D4_seed29_W20000", ctx4, obj4, theta0_4, ctx4.sigma)
    append!(all_rows, rows4)
    write_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_d4_2026-07-28.csv"), rows4)

    println("\n" * "="^100); println("Real D=20 (noah_D20, W=80,000, seed=1)"); println("="^100)
    real_dir = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
    @assert isdir(real_dir) "real_data/noah_D20 not found at $real_dir"
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    BLAS.set_num_threads(20)
    cfg20 = MelitzInnerSolveConfig(:diagnostic; delta_evaluation_cap=PROFILE_CAP)
    inner_opt_capped = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
    obj20, theta0_20 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=inner_opt_capped, forbid_dense_fallback=true, inner_solve_config=cfg20)
    ctx20 = obj20.γ
    r0_20 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
    @assert r0_20.verified "real-D20 base point failed to verify"
    rows20 = run_profile("realD20_seed1_W80000", ctx20, obj20, theta0_20, ctx20.sigma)
    append!(all_rows, rows20)
    write_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"), rows20)
    BLAS.set_num_threads(1)

    write_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_combined_2026-07-28.csv"), all_rows)
    println("\nDONE. CSVs written to ", OUTDIR)
    return all_rows
end

main()
