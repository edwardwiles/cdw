# ============================================================================
# Compressed winner-form representation of the FACTUAL bilateral moments.
#
# ADDITIVE / DIAGNOSTIC ONLY. Does NOT modify the trusted dense constructor
# (full_aod_diag/moments_gammanorm.jl::EK_moments_gammanorm_directgp!), the
# tie-breaking rule (misc/smoothMinIndNew!.jl::MinInd!), lfix_incremental.jl,
# or composite_gradient_fast.jl. It re-derives -- in a single self-contained
# module -- the same winner-form contraction that lfix_incremental.jl's
# `build_lfix_base_cache`/`contrib0` already exploits (and self-validates), but
# packaged as a standalone compressed representation + dual contraction with
# the full column-normalization bookkeeping (SamplingWeights / NormalizeMoments
# / usePMM / gammafac) made explicit so it is provably correct beyond the one
# specific ctx config, and with exact-tie detection (reusing lfix's
# `detect_price_ties`/`TiedWinnerError`) built in from the start.
#
# EXACT FACTUAL FORMULA (verified line-by-line against hFunction.jl +
# moments_gammanorm.jl, UoModel==1 branch, counterType==1 autarky):
#
#   Raw hFunction! output, bilateral column d1 = d + (o-1)*D:
#       r_{s,(o,d)} = pTσ_{s,o,d} * 1{o = winner_{s,d}}  -  P_{(o,d)} * denom_d
#     winner_{s,d} = argmin_o  price_{s,o,d},   price = constCons_{o,d} / U_{s,o}^{-μ}
#     pTσ_{s,o,d}  = constConsσ_{o,d} / Uσ_{s,o}^{-μ}          (the "winning CES value" v)
#     constCons_{o,d}  = wHat_o * AodPow_{o,d} * τ_{o,d}
#     constConsσ_{o,d} = wHat_o^{1-σ} * (AodPow_{o,d} * τ_{o,d})^{1-σ}
#     denom_d = γ_d^σ * (wHat_d * L_d) = wHat_d * L_d      (γ_d ≡ 1 normalization)
#     P = observed bilateral shares λ̂  (ctx.γ.P, reshaped)
#
#   Post-processing applied by EK_moments_gammanorm_directgp! (in order):
#     (1) cols 1..D^2+1 divided by gammafac = Γ(μ(1-σ)+1)
#     (2) if usePMM:  col j -= PMM_j     (j = 1..numMomentsSimple)
#     (3) if NormalizeMoments: col j *= 1/σ_Moments_j   (j ∉ moments_without_var)
#     (4) col j *= SamplingWeights_s   (all j)
#   =>  G_{s,j} = SW_s * nrm_j * ( r_{s,j} * gdiv_j  -  usePMM*PMM_j )
#       gdiv_j = 1/gammafac if j ≤ D^2+1 else 1 ;  nrm_j = 1/σ_M_j (or 1)
#
#   NOTE (correction to the user's schematic G_{·d,s}=v_{sd}(e_{w}-λ̂_{·d})):
#   the centering term is DRAW-INDEPENDENT (−P_{od}·denom_d), NOT scaled by the
#   per-draw winner value v_{sd}. So the exact fixed-dual contraction over the
#   bilateral block is
#       Σ_{o,d} β_{od} G_{s,(o,d)}
#         = SW_s * [ Σ_d κ_{win_sd,d} v_{s,d}  +  Σ_d C_d  −  usePMM·<β,nrm·PMM> ]
#   with κ_{o,d} = β_{o,d}·nrm_{o,d}·gdiv_{o,d}  and  C_d = −denom_d Σ_o κ_{o,d} P_{o,d}
#   -- a per-draw O(D) sum (winner pick per destination) plus draw-independent
#   constants, i.e. O(W·D) total, vs O(W·D^2) for the dense mat-vec.
#
#   The counterfactual price-index column (d1 = D^2+1) is a single extra column,
#   r_{s} = constConsσ'_{bi,bi} / Uσ_{s,bi}^{-μ} − denom'_{bi}  (primed quantities),
#   handled exactly (O(W)) alongside the bilateral block.
# ============================================================================

using LinearAlgebra: dot
using SpecialFunctions: gamma as spgamma

"""
    CompressedFactual

Compressed winner-form representation of the factual moment matrix over the
inner-dual columns 1..oci-1 (= D^2 bilateral + 1 counterfactual price index).
Stores O(W·D) + O(D^2) data, NOT the dense O(W·D^2) matrix. Everything needed
to reproduce `dot(β, G[s,1:oci-1])` exactly for any dual β.
"""
struct CompressedFactual
    D::Int
    W::Int
    oci::Int
    # --- winner-form bilateral block ---
    winner::Matrix{Int}         # W x D : argmin origin per (draw, destination)
    wval::Matrix{Float64}       # W x D : v_{s,d} = pTσ of the winning origin
    # --- fixed (draw-independent) data used by the contraction ---
    Pmat::Matrix{Float64}       # D x D : observed bilateral shares, Pmat[o,d]=P[d+(o-1)D]
    denom::Vector{Float64}      # D
    gdiv::Vector{Float64}       # length oci-1 : 1/gammafac (cols ≤ D^2+1) else 1
    nrm::Vector{Float64}        # length oci-1 : NormalizeMoments factor (or 1)
    PMM::Vector{Float64}        # length oci-1 : per-moment PMM (used only if usePMM==1)
    usePMM::Int
    SW::Vector{Float64}         # W : sampling weights
    gammafac::Float64
    # --- counterfactual price-index column (col D^2+1) ---
    cf_raw::Vector{Float64}     # W : raw hFunctionCounter! value (before post-proc)
    cf_col::Int                 # = D^2+1 (0 if this column is not an inner-dual column)
    # --- tie bookkeeping ---
    n_tied::Int
    tied_examples::Vector{Tuple{Int,Int}}
end

"""
    build_compressed_factual(θ_full, ctx; check_ties=true) -> CompressedFactual

Build the compressed winner-form representation at `θ_full`. Winner-finding is
O(W·D^2) (irreducible: MinInd! compares D origins per (draw,destination)), but
only the WINNING origin's σ-value is evaluated per (draw,destination) (O(W·D)),
vs the dense path evaluating all D and zeroing losers.

If `check_ties` and any (draw,destination) has 2+ origins bit-exactly tied at
the row-min price, throws `TiedWinnerError` (reusing lfix_incremental.jl's type)
-- the one-winner assumption does not hold there (see MEMORY tie-bug note).
"""
function build_compressed_factual(θ_full::AbstractVector, ctx; check_ties::Bool = true)
    γo = ctx.γ
    D = ctx.D; U = ctx.U; W = size(U, 1)
    μ = θ_full[1]; σ = θ_full[2]
    ind = γo.indicators
    oci = ctx.obj.outer_constr_index

    # ---- draw-independent per-cell constants (D x D) ----
    lambda = reshape(γo.P, (D, D))'                       # lambda[d1']... = observed shares
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
    Aod = Aod_θ .* γo.cHat .* (((γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ γo.cHat) .^ (-μ)
    constCons = [γo.wHat[o] * AodPow[o, d] * γo.τ[o, d] for o in 1:D, d in 1:D]
    wPow = [γo.wHat[o]^(1 - σ) for o in 1:D]
    constConsσ = [wPow[o] * (AodPow[o, d] * γo.τ[o, d])^(1 - σ) for o in 1:D, d in 1:D]
    denom = [γo.wHat[d] * γo.L[d] for d in 1:D]           # γ_d ≡ 1
    Pmat = [γo.P[d + (o - 1) * D] for o in 1:D, d in 1:D]  # Pmat[o,d]=P[d1]

    # BIT-IDENTICAL to hFunction!/MinInd!: price = constCons/UPow, UPow=U^{-μ}
    # (NOT constCons*U^μ -- the division form is what MinInd! compares, so the
    #  winner index and the exact-tie boundary match the dense path bit-for-bit).
    UPow = U .^ (-μ)                                       # W x D  (UoModel==1: o1=o)
    UσPow = γo.Uσ .^ (-μ)                                  # W x D

    winner = Matrix{Int}(undef, W, D)
    wval = Matrix{Float64}(undef, W, D)
    tied = Tuple{Int,Int}[]
    n_tied = 0
    @inbounds for d in 1:D
        for s in 1:W
            # first pass: exact row-min (matches MinInd!'s minimum(x))
            best = constCons[1, d] / UPow[s, 1]; bo = 1
            for o in 2:D
                p = constCons[o, d] / UPow[s, o]
                if p < best
                    best = p; bo = o
                end
            end
            winner[s, d] = bo
            wval[s, d] = constConsσ[bo, d] / UσPow[s, bo]
            if check_ties
                # MinInd! sets xInd[o]=1 for EVERY o with price <= min: count them.
                c = 0
                for o in 1:D
                    (constCons[o, d] / UPow[s, o] <= best) && (c += 1)
                end
                if c > 1
                    n_tied += 1
                    length(tied) < 5 && push!(tied, (s, d))
                end
            end
        end
    end
    if check_ties && n_tied > 0
        throw(TiedWinnerError(n_tied, tied))
    end

    # ---- normalization / post-processing vectors over inner-dual columns ----
    gammafac = spgamma(μ * (1 - σ) + 1)
    ncol = oci - 1
    gdiv = [j <= D^2 + 1 ? 1.0 / gammafac : 1.0 for j in 1:ncol]
    NM = ind.NormalizeMoments
    without = γo.moments_without_var
    nrm = [(NM == 1 && !(j in without)) ? 1.0 / γo.σ_Moments[j] : 1.0 for j in 1:ncol]
    usePMM = ind.usePMM
    PMMv = usePMM == 1 ? Float64[γo.PMM[j] for j in 1:ncol] : zeros(ncol)
    SW = γo.SamplingWeights[1:W]

    # ---- counterfactual price-index column (D^2+1), if it is an inner-dual col ----
    cf_col = D^2 + 1
    cf_raw = zeros(W)
    if cf_col <= ncol
        bi = ctx.bi
        wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
        wPrime_bi = wPrime[bi]                       # ==1
        τPrime_bi = γo.τPrime[bi, bi]
        LPrime_bi = γo.LPrime[bi]
        AodPow_bibi = AodPow[bi, bi]                 # same factual AodPow (hFunctionCounter! is passed AodPow)
        γ_prime_bi = θ_full[3 + D]
        constConsσ_bibi = wPrime_bi^(1 - σ) * (AodPow_bibi * τPrime_bi)^(1 - σ)
        denom_cf = γ_prime_bi^σ * (wPrime_bi * LPrime_bi)
        UσPow_bi = @view(γo.Uσ[:, bi]) .^ (-μ)
        @. cf_raw = constConsσ_bibi / UσPow_bi - denom_cf
    else
        cf_col = 0
    end

    return CompressedFactual(D, W, oci, winner, wval, Pmat, denom, gdiv, nrm,
        PMMv, usePMM, SW, gammafac, cf_raw, cf_col, 0, Tuple{Int,Int}[])
end

"""
    compressed_dual_contraction(β, cf::CompressedFactual) -> Vector{W}

Exact compressed evaluation of `t_s = Σ_{j=1}^{oci-1} β_j · G_{s,j}` for every
draw s, WITHOUT materializing the dense G. β has length oci-1 (= D^2 bilateral
followed by the counterfactual column). O(W·D) work, O(D^2) setup.
"""
function compressed_dual_contraction(β::AbstractVector, cf::CompressedFactual)
    D = cf.D; W = cf.W
    length(β) == cf.oci - 1 || error("β length $(length(β)) != oci-1 = $(cf.oci-1)")

    # κ_{o,d} = β_{(o,d)} · nrm · gdiv   (bilateral cols)
    κ = Matrix{Float64}(undef, D, D)
    C = zeros(D)                                  # C_d = −denom_d Σ_o κ_{o,d} P_{o,d}
    @inbounds for d in 1:D
        acc = 0.0
        for o in 1:D
            j = d + (o - 1) * D
            k = β[j] * cf.nrm[j] * cf.gdiv[j]
            κ[o, d] = k
            acc += k * cf.Pmat[o, d]
        end
        C[d] = -cf.denom[d] * acc
    end
    Csum = sum(C)

    # counterfactual column coefficient
    κ_cf = cf.cf_col > 0 ? β[cf.cf_col] * cf.nrm[cf.cf_col] * cf.gdiv[cf.cf_col] : 0.0

    # PMM constant term  usePMM·Σ_j β_j nrm_j PMM_j
    pmmterm = 0.0
    if cf.usePMM == 1
        @inbounds for j in 1:(cf.oci - 1)
            pmmterm += β[j] * cf.nrm[j] * cf.PMM[j]
        end
    end

    t = Vector{Float64}(undef, W)
    @inbounds for s in 1:W
        acc = Csum
        for d in 1:D
            acc += κ[cf.winner[s, d], d] * cf.wval[s, d]
        end
        acc += κ_cf * cf.cf_raw[s]
        t[s] = cf.SW[s] * (acc - pmmterm)
    end
    return t
end

"""
    materialize_dense_factual(cf::CompressedFactual) -> Matrix{Float64}

DIAGNOSTIC: reconstruct the dense W x (oci-1) factual moment matrix from the
compressed representation (bilateral + counterfactual columns), applying the
exact post-processing. For equivalence checks against obj.moments!'s G[:,1:oci-1].
"""
function materialize_dense_factual(cf::CompressedFactual)
    D = cf.D; W = cf.W; ncol = cf.oci - 1
    G = zeros(W, ncol)
    @inbounds for d in 1:D, s in 1:W
        wo = cf.winner[s, d]
        v = cf.wval[s, d]
        for o in 1:D
            j = d + (o - 1) * D
            r = (o == wo ? v : 0.0) - cf.Pmat[o, d] * cf.denom[d]
            G[s, j] = cf.SW[s] * cf.nrm[j] * (r * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        end
    end
    if cf.cf_col > 0
        j = cf.cf_col
        @inbounds for s in 1:W
            G[s, j] = cf.SW[s] * cf.nrm[j] * (cf.cf_raw[s] * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        end
    end
    return G
end

"""
    materialize_dense_factual!(Gview, cf::CompressedFactual)

ADDITIVE (continuation 8, live-integration): in-place variant of
`materialize_dense_factual`, writing into a caller-supplied `W x (oci-1)`
view/matrix instead of allocating a fresh one. Used by the live compressed
Hessian-callback adapter (`compressed_live.jl`) to fill `obj.H`'s existing
dense G columns from an already-built `CompressedFactual` -- this is cheaper
than a from-scratch dense `moments!` call because `cf.winner`/`cf.wval`
(the expensive part: winner search + per-winner sigma-value) are already
computed; this is pure O(W*D) broadcast-equivalent write, no search, no
sigma-value evaluation for losers. Identical formula to
`materialize_dense_factual`, just avoiding the allocation -- not separately
re-derived.
"""
function materialize_dense_factual!(Gview::AbstractMatrix, cf::CompressedFactual)
    D = cf.D; W = cf.W; ncol = cf.oci - 1
    size(Gview) == (W, ncol) || error("materialize_dense_factual!: size(Gview)=$(size(Gview)) != (W,oci-1)=($W,$ncol)")
    @inbounds for d in 1:D, s in 1:W
        wo = cf.winner[s, d]
        v = cf.wval[s, d]
        for o in 1:D
            j = d + (o - 1) * D
            r = (o == wo ? v : 0.0) - cf.Pmat[o, d] * cf.denom[d]
            Gview[s, j] = cf.SW[s] * cf.nrm[j] * (r * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        end
    end
    if cf.cf_col > 0
        j = cf.cf_col
        @inbounds for s in 1:W
            Gview[s, j] = cf.SW[s] * cf.nrm[j] * (cf.cf_raw[s] * cf.gdiv[j] - cf.usePMM * cf.PMM[j])
        end
    end
    return Gview
end
