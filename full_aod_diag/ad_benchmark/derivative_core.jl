# ============================================================================
# §3-4: canonical moment_map and envelope_scalar, matching production EXACTLY.
#
# moment_map(θ; ctx) -> H, an N×(d+2) matrix: H[:,1]=K, H[:,2]=1, H[:,3:end]=G.
# This is precisely what cc_algo/outer_loop_functions.jl::calculate_jac_θ_autodiff!
# differentiates (reshape(obj.jac_h[1:N,:,:], N*(d+2), l) — i.e. m = N*(d+2)).
#
# envelope_scalar_div(θ; ctx) -> scalar, reproducing EXACTLY the contraction
# cc_algo/PsiObjectiveBundle.jl:216-222 computes post-hoc from the dense
# Jacobian for the DIVERGENCE-BUDGET outer constraint (∂c_∂θ[1,:]):
#   s(θ) = (1e10/N) * Σ_draws arg1[draw] * Σ_{j=1}^{outer_constr_index-1} λ[j]*G[draw,j](θ)
# with λ, arg1 held FIXED at their values from the real inner solve at the
# benchmark θ (envelope theorem: λ,arg1 are the current-inner-solution
# multipliers/weights; production does NOT differentiate through them for
# this constraint — that block, uniquely among the outer constraints, has NO
# ift! correction in the source, i.e. it genuinely IS a pure envelope scalar).
#
# NOTE on the gravity constraint (∂c_∂θ[2,:]): production ALSO applies an
# implicit-function-theorem correction (ift!, PsiObjectiveBundle.jl:225-236)
# through how the inner solution x*(θ) itself moves — a genuine total
# derivative, not a pure envelope scalar. That correction is NOT reproduced
# here (out of scope per the call-graph audit's finding that it needs a
# separate IFT re-derivation, not just an AD-backend swap) and is called out
# explicitly in the final report rather than silently approximated.
# ============================================================================

"in-place canonical moment map: H (N×(d+2)) = [K 1 G], matching obj.H layout exactly."
function moment_map!(H::AbstractMatrix, θ::AbstractVector, U::AbstractMatrix, γobj)
    K = @view H[:, 1]
    G = @view H[:, 3:end]
    EK_moments_gammanorm_directgp!(K, G, θ, U, (γ = γobj,))
    H[:, 2] .= 1.0
    return H
end

"out-of-place moment map (ForwardDiff-safe: H's eltype follows θ)."
function moment_map(θ::AbstractVector, U::AbstractMatrix, γobj, d::Int)
    N = size(U, 1)
    H = zeros(eltype(θ), N, d + 2)
    moment_map!(H, θ, U, γobj)
    return H
end

"""
    envelope_scalar_div_ctx(θ, ctx)

Same as above, using a NamedTuple ctx = (U, γobj, λ, arg1, d, outer_constr_index)
so it has the single-argument signature ForwardDiff.gradient/Enzyme need.
"""
function envelope_scalar_div_ctx(θ::AbstractVector, ctx)
    N = size(ctx.U, 1)
    T = eltype(θ)
    H = zeros(T, N, ctx.d + 2)
    moment_map!(H, θ, ctx.U, ctx.γobj)
    Gj = @view H[:, 3:1+ctx.outer_constr_index]   # G[:,1:outer_constr_index-1]... see index note below
    # H columns: 1=K, 2=const, 3:2+d = G[1:d]. The constraint uses G[:,1:outer_constr_index-1]
    # (production's jac_h[:,3:1+outer_constr_index,i] on H's own column indexing,
    # i.e. H columns 3 .. 1+outer_constr_index = G columns 1 .. outer_constr_index-1).
    s = zero(T)
    @inbounds for draw in 1:N
        acc = zero(T)
        for j in 1:ctx.outer_constr_index-1
            acc += ctx.λ[j] * Gj[draw, j]
        end
        s += ctx.arg1[draw] * acc
    end
    return (1e10 / N) * s
end
