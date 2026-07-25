# Continuation session (2026-07-24), Phase 6: fixed-A/f scalar delta=1 upper-GT-bound
# profile at the real D=20 calibrated reference point.
#
# "Fixed A/f" under the ACTIVE :logf outer parameterization (derivation, addendum Section
# 7): theta_free = (log(gamma_prime_j), A_free[1:D^2-1], f_free_free[1:D^2-2]). A is fully
# reconstructed from A_free ALONE (expand_free_theta: `logA_full = pivot_expand(A_free,
# ctx.A_pivot)`, no gamma dependence) and every OFF-domestic-focal free f cell is
# reconstructed from f_free_free ALONE plus a pivot offset that depends only on f[j,j] (not
# on gamma directly) -- so holding theta_free[2:end] fixed at the CALIBRATED values holds
# EVERY genuine free A/f primitive fixed at its calibrated level, by construction. The one
# object that is NOT frozen is f[j,j] (the focal domestic fixed cost), which is DERIVED from
# gamma via `derive_fjj_from_autarky_cutoff` -- exactly the theorem's own zhat'_jj=1
# normalization requires (main prompt Section 6's explicit instruction NOT to freeze f_jj).
# So "fixed A/f, vary gamma" is EXACTLY: theta_free[1] free, theta_free[2:end] pinned at
# calib's own theta_free -- no new coordinate system needed.
#
# Usage: julia --project=. scripts/melitz_real_d20_fixed_af_profile_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS
using Roots: find_zero, Bisection

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    return calib
end

# kappa_j = (w'_j/w_j) * gamma_prime_j^(1/(sigma-1)); GT_j = 1 - kappa_j.
function kappa_of_g(g::Real, calib)
    gamma_prime = exp(g)
    return (calib.w_prime / calib.w[calib.target_country]) * gamma_prime^(1 / (calib.sigma - 1))
end

function main(; W::Int=80_000, seed::Int=1)
    calib = load_calibration()
    BLAS.set_num_threads(16)
    obj, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed, inner_loop_opt=INNER_OPT)
    ctx = obj.γ
    cache = MelitzDeltaEvalCache(64)

    g_calib = theta_calib[1]
    kappa_calib = kappa_of_g(g_calib, calib)
    @printf("Calibrated reference: g=log(gamma_prime)=%.6f  gamma_prime=%.6f  kappa=%.6f  GT=%.6f\n",
        g_calib, exp(g_calib), kappa_calib, 1 - kappa_calib)

    Delta_fixed(g) = begin
        theta = copy(theta_calib); theta[1] = g
        r = evaluate_melitz_delta(theta, ctx, obj; cold=false, cache=cache)
        r
    end

    println("\n" * "="^100)
    println("PHASE 6.1: coarse grid, Delta_fixed(g) moving toward the upper-bound direction (g falling => kappa falling => GT rising)")
    println("="^100)
    # upper GT bound = minimize kappa = minimize gamma_prime = decrease g below g_calib.
    grid = g_calib .- [0.0, 0.02, 0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = NamedTuple[]
    for g in grid
        r = Delta_fixed(g)
        kappa = kappa_of_g(g, calib)
        @printf("  g=%.4f  gamma_prime=%.6f  kappa=%.6f  GT=%.6f  Delta=%.4e  nStatus=%d  verified=%s  min_slack=%.4f\n",
            g, exp(g), kappa, 1 - kappa, r.Delta, r.nStatus, r.verified, r.min_slack)
        push!(results, (g=g, kappa=kappa, GT=1 - kappa, Delta=r.Delta, nStatus=r.nStatus, verified=r.verified, min_slack=r.min_slack))
        flush(stdout)
        # Early stop: once we have a verified Delta<=1 point immediately followed by a
        # Delta>1 point, the bracket is already in hand -- continuing the coarse grid further
        # into a MORE extreme (and, near this fixture's fragile conditioning, potentially very
        # slow/non-convergent) region wastes wall-clock for no additional bracketing value.
        if length(results) >= 2 && results[end-1].verified && results[end-1].Delta <= 1.0 && results[end].Delta > 1.0
            println("  (bracket found -- stopping the coarse grid early)")
            break
        end
    end

    println("\n" * "="^100)
    println("PHASE 6.2: bracket + solve Delta_fixed(g) == 1")
    println("="^100)
    # find bracket: first grid point with Delta<1 and a neighbor with Delta>1 (or =0/small)
    verified_ok = filter(r -> r.verified, results)
    below1 = filter(r -> r.Delta <= 1.0, verified_ok)
    above1 = filter(r -> r.Delta > 1.0, results)
    if isempty(below1) || isempty(above1)
        println("  Could not bracket Delta_fixed(g)=1 from the coarse grid alone -- refining.")
    end
    g_lo = isempty(below1) ? g_calib : minimum(r.g for r in below1)  # smallest g (most extreme) still <=1
    g_hi = isempty(above1) ? g_calib - 0.6 : maximum(r.g for r in above1 if r.g < g_lo; init=g_lo - 0.05)
    @printf("  bracket: [g_hi=%.4f (Delta>1 side), g_lo=%.4f (Delta<=1 side)]\n", g_hi, g_lo)

    # xatol loosened from a naive 1e-6 to 1e-3: each bisection evaluation here is a REAL
    # cold KNITRO solve near the fragile Delta~1 boundary (not a closed-form function), and
    # this fixture's own conditioning (Section 3's seed-sensitivity finding) makes points in
    # this regime genuinely slow -- 1e-3 in g-space (interval width ~0.03) needs ~5
    # evaluations instead of ~15, a materially different wall-clock cost for a benchmark
    # whose downstream use (Phase 7's starting point, Phase 8's comparison) does not need
    # g resolved to 1e-6.
    root_fn(g) = Delta_fixed(g).Delta - 1.0
    g_star = find_zero(root_fn, (g_hi, g_lo), Bisection(); xatol=1e-3)
    r_star = Delta_fixed(g_star)
    kappa_star = kappa_of_g(g_star, calib)
    @printf("\n  g_star=%.8f  gamma_prime_star=%.8f  kappa_fixed=%.8f  GT_fixed=%.8f\n",
        g_star, exp(g_star), kappa_star, 1 - kappa_star)
    @printf("  Delta(g_star)=%.6e  nStatus=%d  verified=%s  min_slack=%.6f\n",
        r_star.Delta, r_star.nStatus, r_star.verified, r_star.min_slack)
    @printf("  gravity residual A/f at g_star = %.3e / %.3e\n",
        r_star.equilibrium_check.gravity_residual_A, r_star.equilibrium_check.gravity_residual_f)

    println("\n" * "="^100)
    println("PHASE 6.2b: cold/full-value verification of g_star")
    println("="^100)
    theta_star_vec = copy(theta_calib); theta_star_vec[1] = g_star
    r_cold = evaluate_melitz_delta(theta_star_vec, ctx, obj; cold=true)
    @printf("  COLD re-solve: Delta=%.6e  nStatus=%d  lfd_ok=%s  min_slack=%.6f\n",
        r_cold.Delta, r_cold.nStatus, r_cold.lfd_ok, r_cold.min_slack)

    @printf("\nSUMMARY (Phase 6 incumbent):\n")
    @printf("  g_fixed=%.8f  gamma_prime_fixed=%.8f  kappa_fixed=%.8f  GT_fixed=%.8f  Delta=%.6e\n",
        g_star, exp(g_star), kappa_star, 1 - kappa_star, r_cold.Delta)
    @printf("  cache hits=%d misses=%d\n", cache.hits, cache.misses)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return theta_star_vec, r_cold, calib, ctx, obj
end

main()
