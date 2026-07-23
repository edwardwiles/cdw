# Synthetic D=4 (or general D) Melitz economy generator -- POPULATION-Pareto construction
# (addendum "Population-Pareto fake data and iceberg-cost closure", superseding the
# exact-sample-correction closure described in the main body of docs/melitz_delta_star.md
# Section 11 and this file's own prior header comment).
#
# CRITICAL SEQUENCING RULE (addendum Section 2): never construct trade-flow targets first
# and then adjust (A,f) to satisfy gravity while retaining the old targets. This
# construction NEVER computes a trade share/flow until (A,f,w,gamma_prime_target) are ALL
# FINAL:
#   1. Choose tau (iceberg costs, off-diagonal) and L (labor endowments) -- exogenous.
#   2. Choose a raw heterogeneous A, project onto the A-gravity restriction EXACTLY via
#      minimum-L2 `project_to_gravity_manifold` (NOT the single-cell `GravityPivot` used by
#      delta_star.jl's OWN outer coordinate system -- found live, again, that dumping an
#      entire gravity correction onto one cell can force backwards export-selection
#      (q_od < q_oo) or push cutoffs outside a well-conditioned range; L2 projection spreads
#      it evenly and was already the established fix for exactly this failure mode, see
#      `project_to_gravity_manifold`'s docstring in equilibrium.jl).
#   3. Choose the D^2-1 free f-cells' RAW (pre-gravity) log-levels (domestic cells drawn
#      systematically CHEAPER than export cells -- economically sensible "cheaper to sell
#      at home" structure, and avoids the backwards-export-selection failure found when
#      domestic/export levels are drawn symmetrically) -- but do NOT fix f[j,j] yet.
#   4. SOLVE for `gamma_prime_target` (hence `f[j,j]` via `derive_fjj_from_autarky_cutoff`,
#      hence the f-gravity affine offset, hence the whole f matrix, hence the baseline GE)
#      via 1-D bisection on the POPULATION-level focal free-entry LINK residual
#      (`population_focal_link_residual`) -- NOT an arbitrary/ACR-seeded choice. This
#      residual is a genuine equilibrium condition (main prompt Section 4.2) and must hold
#      at numerical precision in POPULATION, not merely shrink with W (addendum Section 8);
#      an arbitrarily chosen gamma_prime_target was found live to leave this residual stuck
#      around -0.3 regardless of W, which is the signature of an uncalibrated parameter,
#      not finite-sample noise.
#   5. At the solved `gamma_prime_target`, the inner `build_at` closure below ALREADY
#      recomputes (f, w, A_final, X, q) self-consistently (it is exactly the function being
#      root-found on) -- reuse that final evaluation directly, never a separately "observed"
#      X that (A,f) are later bent to match.
#   6. Verify feasibility (q>=1, export>=domestic) and the well-conditioned-fixture
#      criterion (min reference participation probability -- main prompt Section 7) on the
#      FINAL cutoff matrix; verify both gravity restrictions, factor-market clearing, and
#      the focal link residual to machine precision.
#
# Reference draws (`z_draws`) are then generated at whatever `W` is requested; the addendum
# EXPECTS (Section 3) that equal finite-sample weights will NOT exactly satisfy the
# population moments at finite W, and that the gap shrinks as W grows -- this is the
# intended finite-sample behavior, not a defect to "correct" via `fstar_solver.jl` (which is
# now archived as an optional debugging utility only, never called from this path).

using Random: MersenneTwister, randn!, rand!
using LinearAlgebra: diag
using Statistics: std
using Roots: find_zero, Bisection

"""
    generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1,
                                 seed=29, W=20_000, draw_mode=:halton, ...)
        -> MelitzSyntheticData

Population-Pareto synthetic Melitz economy (addendum construction). `A`, `f` are
gravity-EXACT by construction (min-L2 projection); baseline wages `w` are a genuine
general-equilibrium OUTPUT (`melitz_solve_wages_ge`), not a normalization -- do NOT expect
`w[target_country]==1`. `gamma_prime_target` is SOLVED (not chosen) to zero the
POPULATION-level focal free-entry link residual exactly. `eq.trade_flow` is the POPULATION
closed-form value implied by the FINAL (A,f,w) (never a separately-chosen target). The
noise scales below were tuned (main prompt Section 7) so that, at D=4/seed=29, every
bilateral cell's reference Pareto participation probability exceeds `min_participation_prob`
(default 0.01) -- comfortably >500 active draws at the W=80,000 benchmark -- while cutoffs
remain feasible (`>=1`, export-selection satisfied).

GATE A5 NOTE (docs/melitz_delta_star.md): the default seed changed from 2 to 29, and the
gravity projection now uses `project_to_gravity_manifold_weighted` (domestic cells
down-weighted), when `gravity_residuals`/`gravity_coefficient_vector` switched from
`doubleDiff` to the canonical `withinTransform` -- the coefficient vector's structure
changed (now diagonal-dominated), so seed=2's fixture stopped being export-selection
feasible under the corrected transform (reproduced live: 100% of the first 60 seeds
failed under `doubleDiff`'s replacement before the weighted projection was added; ~9% of
seeds pass with it, of which 29 is the first). This is a re-tuning of an arbitrary fixture
draw, not an economic finding.
"""
function generate_fake_melitz_data(; D::Int=4, sigma::Float64=2.5, theta_star::Float64=6.8,
                                    target_country::Int=1, seed::Int=29, W::Int=20_000,
                                    draw_mode::Symbol=:halton,
                                    tau_offdiag_logrange::Tuple{Float64,Float64}=(0.05, 0.13),
                                    L_range::Tuple{Float64,Float64}=(0.95, 1.25),
                                    logA_noise_sd::Float64=0.04,
                                    logf_domestic_mean::Float64=-0.55, logf_export_mean::Float64=-0.30,
                                    logf_noise_sd::Float64=0.06,
                                    gamma_prime_bracket::Tuple{Float64,Float64}=(0.2, 1.5),
                                    min_participation_prob::Float64=0.01,
                                    ge_damping::Float64=0.1)
    rng = MersenneTwister(seed)
    j = target_country

    # 1. tau, L -- exogenous, no gravity/GE constraint needed at this stage.
    tau = ones(Float64, D, D)
    for o in 1:D, d in 1:D
        o == d && continue
        tau[o, d] = exp(tau_offdiag_logrange[1] + (tau_offdiag_logrange[2] - tau_offdiag_logrange[1]) * rand(rng))
    end
    L = L_range[1] .+ (L_range[2] - L_range[1]) .* rand(rng, D)

    # 2. raw heterogeneous A, gravity-projected EXACTLY (min-L2, spreads the correction).
    logA_raw = logA_noise_sd .* randn(rng, D, D)
    c_full = gravity_coefficient_vector(D, tau)
    # Gate A5: as with f (below), weight the A-gravity projection away from DOMESTIC (o==d)
    # cells for the same reason -- |c| is diagonal-dominated under withinTransform, so an
    # unweighted min-L2 projection disproportionately perturbs A[o,o], which feeds into the
    # domestic cutoff and can push it (in either direction, depending on sign) far enough
    # to break export-selection even before f is touched.
    a_proj_weights = vec([o == d ? 1000.0 : 1.0 for o in 1:D, d in 1:D])
    A = exp.(reshape(project_to_gravity_manifold_weighted(vec(logA_raw), c_full, 0.0, a_proj_weights), D, D))

    # 3. raw (pre-gravity) log-levels of the D^2-1 free f-cells -- domestic systematically
    # cheaper than export. f[j,j] is NOT fixed yet (depends on the gamma_prime_target being
    # solved for in step 4).
    jj_lin = od2lin(j, j, D)
    f_free_lin = [i for i in 1:D^2 if i != jj_lin]
    logf_raw = [lin2od(i, D)[1] == lin2od(i, D)[2] ? logf_domestic_mean + logf_noise_sd * randn(rng) :
                logf_export_mean + logf_noise_sd * randn(rng) for i in f_free_lin]
    # Gate A5: under withinTransform's coefficient vector, |c| is largest on DOMESTIC
    # cells (see project_to_gravity_manifold_weighted's docstring), so weight the f-gravity
    # projection to route the correction onto EXPORT cells instead (weight 1) and leave
    # domestic cells (weight 1000) nearly untouched -- an unweighted min-L2 projection was
    # verified live to break export-selection on 100% of 60 tried seeds after the
    # doubleDiff -> withinTransform switch.
    f_proj_weights = [lin2od(i, D)[1] == lin2od(i, D)[2] ? 1000.0 : 1.0 for i in f_free_lin]

    """
    Given a trial gamma_prime_target, rebuild (f, w, A_final, X, q, expenditure)
    self-consistently. `derive_fjj_from_autarky_cutoff` needs `A[j,j]` -- but the GE solve's
    per-destination `gamma_d==1` rescaling (`melitz_solve_wages_ge`) ALSO rescales column j
    of A (country j is itself a destination), so the FINAL A[j,j] actually paired with
    f[j,j] everywhere else (moments, ex-post checks) differs from the pre-rescale A[j,j]
    used to derive it -- found live: this mismatch alone left the autarky zero-profit
    condition at z=1 off by ~0.28 (nowhere near zero), even though `derive_fjj_from_autarky_cutoff`
    is algebraically exact given the RIGHT A_jj. Fixed by iterating this inner fixed point
    (A[j,j] guess -> f_jj -> GE solve -> new A_final[j,j] -> re-derive f_jj -> ...) to
    self-consistency (converges in a handful of iterations; the coupling is weak since
    A[j,j]'s GE rescaling factor depends only weakly, through theta_star-power column
    sums, on f_jj).
    """
    function build_at(gamma_prime_target::Real)
        A_jj_guess = A[j, j]
        local f, f_jj, w, A_final, X, q, expenditure
        for _ in 1:50
            f_jj = derive_fjj_from_autarky_cutoff(gamma_prime_target, 1.0, 1.0, A_jj_guess, L[j], sigma)
            g0_f = c_full[jj_lin] * log(f_jj)
            logf_free_final = project_to_gravity_manifold_weighted(logf_raw, c_full[f_free_lin], g0_f, f_proj_weights)
            f = zeros(Float64, D, D)
            f[j, j] = f_jj
            for (k, i) in enumerate(f_free_lin)
                o, d = lin2od(i, D)
                f[o, d] = exp(logf_free_final[k])
            end
            w, A_final, _ = melitz_solve_wages_ge(L, tau, A, f, sigma, theta_star; damping=ge_damping)
            abs(A_final[j, j] - A_jj_guess) < 1e-14 * max(1.0, abs(A_jj_guess)) && break
            A_jj_guess = A_final[j, j]
        end
        X, q, expenditure = population_X(w, L, tau, A_final, f, sigma, theta_star)
        return (f=f, f_jj=f_jj, w=w, A=A_final, X=X, q=q, expenditure=expenditure)
    end

    # 4. solve for gamma_prime_target: the POPULATION-level focal free-entry link residual
    # must vanish exactly (main prompt Section 4.2) -- NOT an arbitrary/ACR-seeded choice
    # (an arbitrary choice was found live to leave this residual stuck around -0.3
    # regardless of W, the signature of an uncalibrated parameter, not sampling noise).
    function link_residual(gamma_prime_target::Real)
        r = build_at(gamma_prime_target)
        primitives_trial = MelitzPrimitives(D, sigma, theta_star, j, tau, r.w, r.A, r.f, gamma_prime_target)
        cf_trial = MelitzCounterfactual(j, 1.0, 1.0 * L[j], 1.0, 1.0 * L[j])
        return population_focal_link_residual(primitives_trial, r.X, r.q, r.f_jj, cf_trial)
    end
    gamma_prime_target = find_zero(link_residual, gamma_prime_bracket, Bisection(); xatol=1e-12)

    # 5. final evaluation at the solved gamma_prime_target.
    r = build_at(gamma_prime_target)
    f, f_jj, w, A_final, X, zhat, expenditure = r.f, r.f_jj, r.w, r.A, r.X, r.q, r.expenditure

    primitives = MelitzPrimitives(D, sigma, theta_star, j, tau, w, A_final, f, gamma_prime_target)
    price_power = vec(sum(X, dims=1)) ./ expenditure # should be ==1 to machine precision (GE-enforced)
    eq = MelitzEquilibrium(expenditure, price_power, zhat, X)

    expenditure_prime = 1.0 * L[j] # w_prime[j] = 1 (numeraire), autarky counterfactual only
    counterfactual = MelitzCounterfactual(j, 1.0, expenditure_prime, 1.0, expenditure_prime)

    # 6. verify support / export-selection / well-conditioned-fixture restrictions on the
    # FINAL cutoff matrix, and gravity + factor-market + focal-link identities to machine
    # precision.
    min_cutoff = minimum(zhat)
    min_cutoff >= 1.0 || error("generate_fake_melitz_data: zhat >= 1 violated (min=$min_cutoff); retune")
    for o in 1:D, d in 1:D
        d == o && continue
        zhat[o, d] >= zhat[o, o] || error(
            "generate_fake_melitz_data: export-selection zhat[$o,$d]>=zhat[$o,$o] violated")
    end
    min_prob = minimum(pareto_tail_prob(zhat[o, d], theta_star) for o in 1:D, d in 1:D)
    min_prob >= min_participation_prob || error(
        "generate_fake_melitz_data: min participation probability $min_prob < $min_participation_prob; retune")

    gravity_residual_A, gravity_residual_f = gravity_residuals(primitives)
    @assert abs(gravity_residual_A) < 1e-8 "A gravity restriction not satisfied: $gravity_residual_A"
    @assert abs(gravity_residual_f) < 1e-8 "f gravity restriction not satisfied: $gravity_residual_f"
    @assert maximum(abs.(w .* L .- vec(sum(X, dims=2)))) < 1e-8 "factor-market clearing (w*L == row-sum X) violated"
    @assert maximum(abs.(price_power .- 1.0)) < 1e-8 "baseline price-index normalization gamma_d==1 violated"
    @assert abs(population_focal_link_residual(primitives, X, zhat, f_jj, counterfactual)) < 1e-8 "population focal link residual not zeroed"

    @assert std(log.(A_final)) > 0.01 "A matrix is too close to trivial"
    @assert std(log.(f)) > 0.01 "f matrix is too close to trivial"
    @assert std(withinTransform(A_final)) > 1e-4 "withinTransform(A) has no genuine bilateral variation"
    @assert std(withinTransform(f)) > 1e-4 "withinTransform(f) has no genuine bilateral variation"

    z_draws = pareto_draws(W, D, theta_star; seed=seed + 1, mode=draw_mode)

    return MelitzSyntheticData(primitives, eq, counterfactual, L, z_draws, seed)
end
