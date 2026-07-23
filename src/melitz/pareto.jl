# Reference (F*) Pareto distribution: draws + analytical tail moments.
# See docs/melitz_delta_star.md Section 1.3 for the derivations below.
#
# Convention: z ~ Pareto(scale=1, shape=theta_star), density theta_star * z^(-theta_star-1)
# on [1, infinity). theta_star > sigma-1 is required for E[z^(sigma-1)] to be finite.

# Bare `using Random` (not a selective import) -- `cc_algo/rhalton.jl`, included below,
# calls unqualified `shuffle` and `Random.seed!` and expects the including scope to have
# the full Random stdlib in scope, exactly as `cc_algo/include_cc_algo.jl` provides it
# elsewhere in the repo.
using Random

const MELITZ_RHALTON_PATH = joinpath(dirname(dirname(@__DIR__)), "cc_algo", "rhalton.jl")
if isfile(MELITZ_RHALTON_PATH) && !isdefined(@__MODULE__, :rhalton)
    include(MELITZ_RHALTON_PATH)
end

"""
    pareto_draws(W, D, theta_star; seed, mode=:halton) -> Matrix{Float64} (W x D)

Deterministic Pareto(1, theta_star) reference draws, one per origin per Monte-Carlo row
(matches the repository's own `UoModel=1` convention: draws are indexed by *origin*, not
by origin-destination pair). Generated once and reused throughout parameter solving,
moment construction, and Delta-star evaluation, never resampled.

`mode=:halton` (default, addendum Section 11): uses the repository's own scrambled-Halton
generator (`cc_algo/rhalton.jl`'s `rhalton(n, d; singleseed)`), inverse-Pareto-CDF
transformed -- deterministic given the seed, lower discrepancy than pseudorandom draws for
the same `W`. `mode=:pseudorandom`: the original `MersenneTwister`-based draw, kept as an
explicit opt-in robustness check (addendum Section 11: "retain a pseudorandom mode only as
an optional robustness test").
"""
function pareto_draws(W::Int, D::Int, theta_star::Real; seed::Int, mode::Symbol=:halton)
    u = if mode == :halton
        rhalton(W, D; singleseed=seed)
    elseif mode == :pseudorandom
        rng = MersenneTwister(seed)
        uu = zeros(Float64, W, D)
        rand!(rng, uu)
        uu
    else
        throw(ArgumentError("pareto_draws: unknown mode $mode (expected :halton or :pseudorandom)"))
    end
    z = similar(Matrix{Float64}(undef, W, D))
    @. z = (1.0 - u)^(-1.0 / theta_star)
    return z
end

"""
    pareto_tail_prob(zhat, theta_star) -> Pr(z > zhat)

`zhat^(-theta_star)` for `zhat >= 1`, `z ~ Pareto(1, theta_star)`.
"""
pareto_tail_prob(zhat::Real, theta_star::Real) = zhat^(-theta_star)

"""
    pareto_tail_power_mean(zhat, sigma, theta_star) -> E[z^(sigma-1) * 1{z > zhat}]

Closed form `theta_star/(theta_star-sigma+1) * zhat^(sigma-1-theta_star)`, requires
`theta_star > sigma - 1`. This is the sufficient statistic behind every aggregate Melitz
moment under the Pareto benchmark (docs Section 1.3, equation (*)).
"""
function pareto_tail_power_mean(zhat::Real, sigma::Real, theta_star::Real)
    theta_star > sigma - 1 || throw(ArgumentError("theta_star must exceed sigma-1"))
    return theta_star / (theta_star - sigma + 1) * zhat^(sigma - 1 - theta_star)
end

"""
    pareto_tail_power_mean_numeric(zhat, sigma, theta_star; kwargs...) -> Float64

High-precision numerical-integration cross-check for `pareto_tail_power_mean`, used only
in tests (docs requires comparing every analytical formula against numerical integration
or a very large Monte Carlo sample).
"""
function pareto_tail_power_mean_numeric(zhat::Real, sigma::Real, theta_star::Real;
                                         upper::Real=1e8, rtol::Real=1e-10)
    # integrand: z^(sigma-1) * theta_star * z^(-theta_star-1) = theta_star * z^(sigma-2-theta_star)
    # substitute z = zhat / (1-t) style adaptive quadrature is overkill here; use a simple
    # high-order composite Simpson on a log grid, which is smooth and rapidly convergent
    # for the decaying power-law integrand.
    n = 200_000
    a, b = log(zhat), log(upper)
    h = (b - a) / n
    s = 0.0
    for i in 0:n
        logz = a + i * h
        z = exp(logz)
        integrand = theta_star * z^(sigma - 1 - theta_star) # includes Jacobian dz = z dlogz folded in via z^(sigma-1)*density*z
        w = (i == 0 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        s += w * integrand
    end
    return s * h / 3
end
