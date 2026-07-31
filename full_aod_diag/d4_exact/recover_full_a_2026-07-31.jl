# ============================================================================
# Task §16 / theory doc section 2.2: exact gamma-normalized full-A recovery
# from a working-gauge (not necessarily gamma-normalized) point.
# ADDITIVE ONLY -- reads via ctx.obj.moments!, writes nothing to ctx or any
# trusted file.
#
# c_d = gamma_tilde_d^{-1/(mu*(sigma-1))},  A_full[:,d] = c_d * A_working[:,d]
#
# where gamma_tilde_d := E_F[M_d(working)] / denom[d], mu*(sigma-1) is the
# exact homogeneity exponent numerically confirmed in
# test_profiled_destination_scale_invariance_2026-07-31.jl. See
# PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md section 2.2 for the
# derivation and section 0 for where mu*(sigma-1) comes from in the
# production code (fast_range_screen.jl's own factorization).
# ============================================================================

isdefined(Main, :spgamma) || (using SpecialFunctions: gamma as spgamma)
using Statistics: mean

"""
    destination_M_d(θ_full, ctx; d_list=1:ctx.D) -> Dict{Int,Vector{Float64}}

Evaluates `ctx.obj.moments!` at `θ_full` and reconstructs, for every
destination `d` in `d_list`, the per-draw raw total `M_d(ω)` (length W)
EXACTLY as `hFunction!` computes it internally (`moments/hFunction.jl:83-88`),
by inverting the known `G[ω,d1] = Q_od(ω) - λ_od*denom[d]` post-processing
(sampling-weight and gammafac factors, both ω/(o,d)-independent-within-a-row
scalars, are recovered from `ctx.γ` directly, not assumed to be 1). See
`test_profiled_destination_scale_invariance_2026-07-31.jl`'s
`extract_Q_M` for the same derivation (duplicated here as a small,
self-contained, documented function rather than including that test file).
"""
function destination_M_d(θ_full::AbstractVector{Float64}, ctx; d_list = 1:ctx.D)
    D = ctx.D
    W = size(ctx.U, 1)
    K = zeros(W); G = zeros(W, ctx.nTotalMoments)
    ctx.obj.moments!(K, G, θ_full, ctx.U, ctx.obj)
    μ = θ_full[1]; σ = θ_full[2]
    lambda = reshape(ctx.γ.P, (D, D))'
    gammafac = spgamma(μ * (1 - σ) + 1)
    SW = ctx.γ.SamplingWeights[1:W]
    wscale = SW ./ gammafac
    out = Dict{Int,Vector{Float64}}()
    for d in d_list
        denom_d = ctx.γ.wHat[d] * ctx.γ.L[d]
        M = zeros(W)
        for o in 1:D
            d1 = d + (o - 1) * D
            @. M += G[:, d1] / wscale + lambda[o, d] * denom_d
        end
        out[d] = M
    end
    return out
end

"""
    recover_gamma_normalized_full_A(z_working, ctx; d_list=1:ctx.D) -> (z_full, c, gamma_tilde)

`z_working` (D x Ddest, `log(Aod_theta)` at a working-gauge point) -> `z_full`
(same shape, exactly gamma-normalized for every destination in `d_list`:
`c[d] = gamma_tilde[d]^(-1/(mu*(sigma-1)))`, `z_full[:,d] = z_working[:,d] .+
log(c[d])`). `θ_full_working` (the base θ vector `z_working` was embedded in)
supplies μ, σ, and everything else `destination_M_d` needs -- passed via
`ctx` and the caller's own θ construction (this function only needs `ctx` and
a full θ vector, not a separate `AnchorSpec`; recovery is defined on the
FULL A matrix, independent of which coordinate layer produced it).
"""
function recover_gamma_normalized_full_A(θ_working::AbstractVector{Float64}, ctx; d_list = 1:ctx.D)
    D = ctx.D
    Aod_offset = ctx.Aod_offset
    μ = θ_working[1]; σ = θ_working[2]
    e_exponent = μ * (σ - 1)
    M = destination_M_d(θ_working, ctx; d_list = d_list)
    c = ones(D)
    gamma_tilde = ones(D)
    for d in d_list
        denom_d = ctx.γ.wHat[d] * ctx.γ.L[d]
        gt = mean(M[d]) / denom_d
        gamma_tilde[d] = gt
        c[d] = gt^(-1 / e_exponent)
    end
    Aod_θ_working = reshape(θ_working[Aod_offset+1:Aod_offset+D^2], (D, D))
    z_working = log.(Aod_θ_working)
    z_full = copy(z_working)
    for d in d_list
        z_full[:, d] .+= log(c[d])
    end
    return z_full, c, gamma_tilde
end
