# ============================================================================
# Task §16 / theory doc section 2.2: exact gamma-normalized full-A recovery
# from a working-gauge (not necessarily gamma-normalized) point.
# ADDITIVE ONLY -- writes nothing to ctx or any trusted file.
#
# CORRECTED 2026-07-31 (same day, user stop): an earlier version of this file
# read the LEGACY dense G/K moment matrix via `ctx.obj.moments!`, which only
# exists on the pre-hardening `PsiObjectiveBundleImplicit` bundle that
# `d4_exact_setup()`'s `ctx.obj` happens to be. This repo's production stack
# has moved to genuinely no-H/no-G/no-K `OperatorPsiBundle`s (per this
# session's own audit findings AND explicit user correction: "we use newer
# bundles that do not define G or H or K"). Rebuilt to use
# `build_compressed_factual(θ_full, ctx)` (compressed_moments.jl) instead --
# the actual winner-compressed representation the production operator path
# uses, reading `wval`/`winner`/`Pmat`/`denom`/`cf_raw` DIRECTLY rather than
# reconstructing them by inverting dense-G post-processing. This is not just
# a style fix: it is now genuinely representative of what production
# computes, not a legacy reference path.
#
# c_d = gamma_tilde_d^{-1/(mu*(sigma-1))},  A_full[:,d] = c_d * A_working[:,d]
#
# where gamma_tilde_d := E_F[M_d(working)] / denom[d], mu*(sigma-1) is the
# exact homogeneity exponent (re-confirmed against this corrected path in
# test_profiled_destination_scale_invariance_2026-07-31.jl). See
# PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md section 2.2 for the
# derivation and section 0 for where mu*(sigma-1) comes from in the
# production code (fast_range_screen.jl's own factorization).
# ============================================================================

isdefined(Main, :build_compressed_factual) || error("recover_full_a_2026-07-31.jl requires compressed_moments.jl (and its own defensive includes of winner_certificate.jl/active_layout.jl) to be included first.")
using Statistics: mean

"""
    destination_M_d(θ_full, ctx; d_list=1:ctx.D, check_ties=false) -> Dict{Int,Vector{Float64}}

For every destination `d` in `d_list`, returns the per-draw raw total
`M_d(ω)` (length W) read DIRECTLY off `build_compressed_factual`'s `wval`
field -- since only the winning origin contributes to the destination total,
`M_d(ω) = cf.wval[ω, slot]` exactly, no reconstruction needed (unlike the
legacy dense-G path this replaces, which had to invert a fixed
post-processing scale to recover raw values).
"""
function destination_M_d(θ_full::AbstractVector{Float64}, ctx; d_list = 1:ctx.D, check_ties::Bool = false)
    cf = build_compressed_factual(θ_full, ctx; check_ties = check_ties)
    cf.D_dest == cf.D || error("destination_M_d: expected a square (D_dest==D) context for this session's D=4 gates; got D=$(cf.D), D_dest=$(cf.D_dest)")
    out = Dict{Int,Vector{Float64}}()
    for d in d_list
        s = dest_slot(ctx, d)
        out[d] = cf.wval[:, s]
    end
    return out
end

"""
    destination_Q_od(θ_full, ctx, d) -> Matrix{Float64}  (W x D)

Per-draw `Q_od(ω)` for every origin `o`, destination `d`: `wval[ω,slot]` if
`o` is the draw's winner, else `0.0`. Read directly off `cf.winner`/`cf.wval`.
"""
function destination_Q_od(θ_full::AbstractVector{Float64}, ctx, d::Int; check_ties::Bool = false)
    cf = build_compressed_factual(θ_full, ctx; check_ties = check_ties)
    D = cf.D
    s = dest_slot(ctx, d)
    W = cf.W
    Q = zeros(W, D)
    @inbounds for w in 1:W
        Q[w, cf.winner[w, s]] = cf.wval[w, s]
    end
    return Q
end

"""
    recover_gamma_normalized_full_A(θ_working, ctx; d_list=1:ctx.D) -> (z_full, c, gamma_tilde)

`θ_working` (a full θ vector at a working-gauge, not necessarily
gamma-normalized, point) -> `z_full` (D x Ddest `log(Aod_theta)`, exactly
gamma-normalized for every destination in `d_list`):
`c[d] = gamma_tilde[d]^(-1/(mu*(sigma-1)))`, `z_full[:,d] = z_working[:,d]
.+ log(c[d])`.
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
        s = dest_slot(ctx, d)
        cf = build_compressed_factual(θ_working, ctx; check_ties = false)
        denom_d = cf.denom[s]
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
