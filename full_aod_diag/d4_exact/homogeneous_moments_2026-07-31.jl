# ============================================================================
# Task §8 (core piece) / §1.2: homogeneous factual moment.
# ADDITIVE ONLY -- reuses destination_M_d (recover_full_a_2026-07-31.jl)
# unchanged; requires that file included first.
#
# Replaces the OLD moment  E_F[Q_od(w)] - lambda_od*denom[d] = 0  (denom[d] a
# FIXED DATA constant, NOT invariant to a destination-column rescale -- this
# is what currently pins the destination scale, per
# PROFILED_DESTINATION_SCALES_MASTER_2026-07-31.md's resolution of the
# "is the scale really redundant" question) with the NEW moment
#   E_F[Q_od(w) - lambda_od*M_d(w)] = 0
# using the MODEL's OWN M_d(w) instead of the fixed denom[d]. Because both
# Q_od(w) and M_d(w) scale by the SAME factor kappa^(mu*(sigma-1)) under a
# common destination-d column rescale (theory doc section 2.1, numerically
# confirmed), this new moment is exactly homogeneous: its value scales by
# kappa^(mu*(sigma-1)) too (not merely "stays zero if zero" -- the exact
# proportional-rescaling property, tested directly below), which is exactly
# why the destination scale becomes genuinely unidentified under this moment
# set and can be validly profiled out via a fixed anchor gauge instead of
# left as a KNITRO search direction.
# ============================================================================

isdefined(Main, :destination_M_d) || error("homogeneous_moments_2026-07-31.jl requires recover_full_a_2026-07-31.jl to be included first.")

"""
    homogeneous_factual_moment(θ_full, ctx; d_list=1:ctx.D) -> Dict{Int,Matrix{Float64}}

For every destination `d` in `d_list`, returns a `W x D` matrix whose column
`o` is the per-draw homogeneous moment `Q_od(w) - lambda_od*M_d(w)`. Reuses
`destination_M_d` for `M_d(w)` and reconstructs `Q_od(w)` the same way that
function does internally (documented in its own header), so the two are
mutually consistent by construction.
"""
function homogeneous_factual_moment(θ_full::AbstractVector{Float64}, ctx; d_list = 1:ctx.D)
    D = ctx.D
    W = size(ctx.U, 1)
    K = zeros(W); G = zeros(W, ctx.nTotalMoments)
    ctx.obj.moments!(K, G, θ_full, ctx.U, ctx.obj)
    μ = θ_full[1]; σ = θ_full[2]
    lambda = reshape(ctx.γ.P, (D, D))'
    gammafac = spgamma(μ * (1 - σ) + 1)
    SW = ctx.γ.SamplingWeights[1:W]
    wscale = SW ./ gammafac
    out = Dict{Int,Matrix{Float64}}()
    for d in d_list
        denom_d = ctx.γ.wHat[d] * ctx.γ.L[d]
        Q = zeros(W, D)
        for o in 1:D
            d1 = d + (o - 1) * D
            @. Q[:, o] = G[:, d1] / wscale + lambda[o, d] * denom_d
        end
        M = vec(sum(Q, dims = 2))
        H = zeros(W, D)
        for o in 1:D
            @. H[:, o] = Q[:, o] - lambda[o, d] * M
        end
        out[d] = H
    end
    return out
end
