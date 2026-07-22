# Deterministic synthetic D=4 (or general D) Melitz economy generator.
# See docs/melitz_delta_star.md Sections 1.5-1.8, 10 for the derivations used here.
#
# Construction order (mirrors the corrected equilibrium.jl architecture):
#   1. Choose tau, L (labor endowment), raw trade-share heterogeneity.
#   2. Project the raw shares' double-differenced component so the trade-flow DATA
#      itself satisfies the identified-composite restriction <T, DDlogX + theta*T> = 0
#      (docs Sec 1.7) -- this is what makes BOTH gravity restrictions on A and f
#      simultaneously satisfiable via a single cutoff projection in step 4.
#      Note DD(log X) = DD(log lambda) exactly (expenditure_d is a pure destination
#      effect and vanishes under doubleDiff), so this can be done directly on lambda,
#      decoupled from the (nonlinear) wage solve.
#   3. Solve wages (melitz_solve_wages), get expenditure = w.*L, X = lambda.*expenditure'.
#   4. Fix zhat[target,target] at its Sec 1.8 value; project the *other* D^2-1 raw
#      cutoffs so <T, DDlogzhat> hits the value the f-gravity-restriction requires
#      (equivalently, given step 2, the A-restriction too -- verified, not assumed).
#   5. Build A, f, entrant-mass-consistent moments via `build_equilibrium` +
#      `entry_cost_from_free_entry` (all closed form, no iteration beyond step 3's
#      wage solve).

using Random: MersenneTwister, randn!, rand!
using LinearAlgebra: diag
using Statistics: std

"""
    generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1,
                                 seed=1234, W=20_000, ...) -> MelitzSyntheticData

Deterministic synthetic Melitz economy generator for the D=4 Delta-star benchmark.
Nontrivial, heterogeneous `tau`, `A`, `f` (never all-ones); both gravity restrictions
satisfied to numerical (solver) tolerance by construction; target country's autarky
cutoff exactly 1; all other required inequality restrictions (`zhat>=1`,
`zhat_od>=zhat_oo`) verified.
"""
function generate_fake_melitz_data(; D::Int=4, sigma::Float64=2.5, theta_star::Float64=6.8,
                                    target_country::Int=1, seed::Int=1234, W::Int=20_000,
                                    tau_offdiag_range::Tuple{Float64,Float64}=(0.15, 0.55),
                                    L_range::Tuple{Float64,Float64}=(0.8, 2.5),
                                    N_range::Tuple{Float64,Float64}=(0.6, 2.2),
                                    zhat_base::Float64=2.2, zhat_noise_sd::Float64=0.12,
                                    lambda_noise_sd::Float64=0.18, lambda_diag_bonus::Float64=0.6,
                                    lambda_effect_sd::Float64=0.35)
    rng = MersenneTwister(seed)

    # 1. tau: diag=1, heterogeneous off-diagonal iceberg costs
    tau = ones(Float64, D, D)
    for o in 1:D, d in 1:D
        o == d && continue
        tau[o, d] = exp(tau_offdiag_range[1] + (tau_offdiag_range[2] - tau_offdiag_range[1]) * rand(rng))
    end

    # labor endowments (primitive, chosen)
    L = L_range[1] .+ (L_range[2] - L_range[1]) .* rand(rng, D)

    T = doubleDiff(tau) # DD(log tau); row 1 and column 2 are exactly zero by construction
    TT = sum(T .^ 2)
    TT > 0 || error("generate_fake_melitz_data: doubleDiff(tau) is degenerate (D<3?), cannot project")

    # 2. raw trade-share heterogeneity (origin+destination effects vanish under DD, so
    # only the noise matters for the gravity projection -- but the effects still give
    # genuine cross-sectional heterogeneity in levels/shares).
    origin_eff = lambda_effect_sd .* randn(rng, D)
    dest_eff = lambda_effect_sd .* randn(rng, D)
    raw_loglambda = zeros(Float64, D, D)
    for o in 1:D, d in 1:D
        raw_loglambda[o, d] = origin_eff[o] + dest_eff[d] + lambda_noise_sd * randn(rng)
        o == d && (raw_loglambda[o, d] += lambda_diag_bonus)
    end

    # Project so the DATA's own identified composite <T, DDlogX + theta*T> = 0 holds.
    # DD(log X) = DD(log lambda) exactly (expenditure_d is a destination fixed effect).
    required_inner_X = -theta_star * TT
    current_inner = sum(T .* doubleDiff(exp.(raw_loglambda)))
    c1 = (required_inner_X - current_inner) / TT
    loglambda = raw_loglambda .+ c1 .* T
    lambda_unnorm = exp.(loglambda)
    lambda = lambda_unnorm ./ sum(lambda_unnorm, dims=1)

    # 3. solve wages (Ricardian repo's own damped-Jacobi fixed point, reused verbatim)
    w = melitz_solve_wages(lambda, L)
    w ./= w[target_country] # renormalize to the target-country numeraire

    expenditure = w .* L
    X = lambda .* expenditure'

    @assert isapprox(vec(sum(X, dims=1)), expenditure; rtol=1e-8) "column balance failed"
    @assert isapprox(vec(sum(X, dims=2)), w .* L; rtol=1e-8) "row balance (income=sales) failed"

    # verify the identified-composite condition actually holds on the constructed X
    DDlogX = doubleDiff(X)
    gravity_composite_residual = sum(T .* DDlogX) + theta_star * TT
    @assert abs(gravity_composite_residual) < 1e-6 * max(1.0, abs(theta_star * TT)) "composite gravity condition failed to construct"

    # entrant mass: genuinely free (docs Sec 1.4); target_country's value is free too
    # (Sec 1.8 pins the CUTOFF, not N) -- chosen like every other origin.
    N = N_range[1] .+ (N_range[2] - N_range[1]) .* rand(rng, D)

    # 4. cutoffs: target_country's own cell is derived (Sec 1.8); every other cell is a
    # free computational parameterization, projected to satisfy the (now-consistent)
    # gravity restriction on f (equivalently, given step 2, on A too).
    zhat_tt_required = target_baseline_cutoff_for_autarky(expenditure[target_country],
                                                            w[target_country],
                                                            X[target_country, target_country],
                                                            theta_star)
    raw_logzhat = zeros(Float64, D, D)
    for o in 1:D, d in 1:D
        raw_logzhat[o, d] = log(zhat_base) + 0.15 * (d == o ? -1.0 : 1.0) + zhat_noise_sd * randn(rng)
    end
    raw_logzhat[target_country, target_country] = log(zhat_tt_required)

    T_masked = copy(T)
    T_masked[target_country, target_country] = 0.0
    TmaskTmask = sum(T_masked .^ 2)
    TmaskTmask > 0 || error("generate_fake_melitz_data: masked doubleDiff(tau) is degenerate")

    required_inner_zhat = -sum(T .* DDlogX) / theta_star
    current_inner_zhat = sum(T .* doubleDiff(exp.(raw_logzhat)))
    c2 = (required_inner_zhat - current_inner_zhat) / TmaskTmask
    logzhat = raw_logzhat .+ c2 .* T_masked
    zhat = exp.(logzhat)
    @assert isapprox(zhat[target_country, target_country], zhat_tt_required; rtol=1e-10) "target cell was perturbed by projection"

    # 5. assemble A, f, entrant-mass-consistent equilibrium (closed form)
    A, f, C, eq = build_equilibrium(X, N, w, tau, expenditure, zhat, sigma, theta_star)
    f_entry = [entry_cost_from_free_entry(C[o, :], zhat[o, :], w[o], sigma, theta_star) for o in 1:D]

    primitives = MelitzPrimitives(D, sigma, theta_star, target_country, tau, w, A, f, f_entry)
    counterfactual = solve_autarky_counterfactual(primitives, eq)

    # verify support / export-selection restrictions
    min_cutoff, max_cutoff = extrema(zhat)
    min_cutoff >= 1.0 || error("generate_fake_melitz_data: zhat >= 1 violated (min=$min_cutoff); retune zhat_base/noise")
    for o in 1:D, d in 1:D
        d == o && continue
        zhat[o, d] >= zhat[o, o] || error(
            "generate_fake_melitz_data: export-selection zhat[$o,$d]>=zhat[$o,$o] violated")
    end

    # verify both gravity restrictions numerically
    gravity_residual_A = sum(T .* doubleDiff(A))
    gravity_residual_f = sum(T .* doubleDiff(f))
    @assert abs(gravity_residual_A) < 1e-6 "A gravity restriction not satisfied: $gravity_residual_A"
    @assert abs(gravity_residual_f) < 1e-6 "f gravity restriction not satisfied: $gravity_residual_f"

    # verify heterogeneity requirements (never trivially-all-ones)
    @assert std(log.(A)) > 0.01 "A matrix is too close to trivial"
    @assert std(log.(f)) > 0.01 "f matrix is too close to trivial"
    @assert std(doubleDiff(A)) > 1e-4 "doubleDiff(A) has no genuine variation"
    @assert std(doubleDiff(f)) > 1e-4 "doubleDiff(f) has no genuine variation"

    z_draws = pareto_draws(W, D, theta_star; seed=seed + 1)

    return MelitzSyntheticData(primitives, eq, counterfactual, z_draws, seed)
end
