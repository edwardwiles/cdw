# ============================================================================
# Part 7: a GENUINELY sample-exact Frechet negative control for the full
# (D+2)-moment problem.
#
# The earlier "sample-exact" control (run_part4_mc_stability.jl) checked that
# max|E_F[G]| (the POPULATION-calibrated moment, evaluated at a finite W
# sample) and delta*(A*) both shrink as W grows -- true, but only
# APPROXIMATELY zero at any finite W (an artifact of finite-sample MC noise
# around the population target), not EXACTLY zero. That is not actually
# "sample exact": it conflates "the population target is a good
# approximation at this W" with "the uniform likelihood ratio exactly
# rationalizes this sample".
#
# This file instead RECENTERS every moment column by its own frozen sample
# mean at the reference point (A*, gamma'_frechet), computed on the EXACT
# Monte Carlo draw matrix U the inner problem will use. At theta=theta_ref
# EXACTLY, p=uniform (1/W each draw) is then an EXACT feasible point for
# every one of the D+2 moments (not merely approximately so), so
# delta*(theta_ref)=0 up to pure solver tolerance -- establishing the true
# numerical floor for "detecting genuine descent away from A*", per the
# task's explicit purpose (not merely "divergence can't be negative").
# ============================================================================

"""
    make_sample_exact_moments(base_moments_fn!, d, θref, U, γobj)

See module docstring. Returns a moments!(K,G,θ,U,obj) function that calls
`base_moments_fn!` then subtracts the FROZEN column means computed once at
(θref, U). K (the gamma'_focal target itself, not part of G) is untouched.
"""
function make_sample_exact_moments(base_moments_fn!::Function, d::Int, θref::Vector{Float64}, U::Matrix{Float64}, γobj)
    W = size(U, 1)
    Kref = zeros(W); Gref = zeros(W, d)
    base_moments_fn!(Kref, Gref, θref, U, (γ=γobj,))
    means = vec(sum(Gref, dims=1) ./ W)
    return function (K, G, θ, Uarg, obj)
        base_moments_fn!(K, G, θ, Uarg, obj)
        nrow = size(G, 1)
        @inbounds for j in 1:d
            @views G[1:nrow, j] .-= means[j]
        end
        return nothing
    end
end
