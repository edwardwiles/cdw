# ============================================================================
# Structured (Architecture-C-style) Hessian for the COMBINED CDF+truncated-
# power fixed-Fréchet block (eq.37+38), :cumulative basis (port-prep
# 2026-07-24). This is NEW relative to the pre-omit-ROW reconciliation
# archive (experiment/fullA-fixed-frechet-basis-draft-reconciliation-2026-07-24),
# which explicitly left this unbuilt ("no fast structured Hessian exists yet
# for the truncated-power feature block" -- its own Part VIII item 3).
#
# Reuses the existing production `CMBinHessCtx` / `build_bin_tables!` /
# `prefix_sum_tables!` (cm_hessian_architectures.jl) COMPLETELY UNCHANGED for
# the CDF-CDF and core-CDF pieces (identical algebra to
# cm_frechet_moments.jl/cm_frechet_bases.jl's CDF-only kernel -- ported
# separately as `hessian_cm_frechet_structured!` in this branch's
# `cm_frechet_hessian.jl`, NOT duplicated here). Adds THREE new raw weighted
# bin tables for the POWER family and the CDF x POWER cross family, built in
# one extra O(W*(D^2 + D*(NCORE+1))) sweep sharing the SAME bin indices
# `Bidx` (both features use the same threshold grid `z = targets.thresholds`).
#
# Derivation and the full cross-block algebra: see
# docs/FIXED_FRECHET_STRUCTURED_POWER_HESSIAN_2026-07-24.md. Validated
# against dense Architecture A at D=4 in
# test_frechet_power_hessian_d4_gates.jl -- treat this file's formulas as
# UNTRUSTED until that gate passes; do not cite a κ/Δ* number computed
# through this path before that gate has actually been run and shown PASS in
# this branch's own test logs.
#
# Column layout (matches cm_frechet_bases.jl's `feature_set=:cdf_power`
# layout exactly): [core (NCORE)] [CDF-contrast (nO*L)] [CDF-common (L)]
# [POWER-contrast (nO*L)] [POWER-common (L)]. ncm_fam = nO*L + L (one
# family's width); total ncm = 2*ncm_fam.
# ============================================================================

using LinearAlgebra: BLAS

"""
    FrechetPowerBinHessCtx

`cctx`: a genuine `CMBinHessCtx` built via `build_cm_bin_ctx(ctx, aug_cdf)`
against the CDF-ONLY `aug` (width `ncm_fam = D*L`) -- its `Ttab`/`Stab`/`CT`/
`CScum` fields serve the CDF-CDF and core-CDF blocks completely unchanged.
This struct adds the POWER-family and CDF×POWER-cross tables alongside it;
`Hfull` here is sized for the FULL combined `(NCORE + 2*ncm_fam)` problem,
NOT `cctx.Hfull` (which stays `(NCORE+ncm_fam)`-sized and is simply unused
for output -- `build_bin_tables!`/`prefix_sum_tables!` only ever read/write
`cctx.Ttab/Stab/CT/CScum`, never `cctx.Hfull`, so this is a safe reuse, not a
sizing conflict).
"""
struct FrechetPowerBinHessCtx
    cctx::CMBinHessCtx
    Upow::Matrix{Float64}       # W x D, U.^pw precomputed once (pw = (1-σ)/θ*)
    pw::Float64
    p::Vector{Float64}          # CDF targets, t_l* = p_l
    tpow::Vector{Float64}       # POWER targets, t_power_l*
    # raw (un-prefix-summed) new tables
    Ttab_pp::Array{Float64,4}   # D x D x L1 x L1   (POWER-weight both sides)
    Ttab_cp::Array{Float64,4}   # D x D x L1 x L1   (CDF-weight x-side, POWER-weight y-side)
    Stab_p::Array{Float64,3}    # D x (NCORE+1) x L1  (POWER-weight; last core col = constant 1 -> marginal)
    # prefix-summed (restricted to 1:L)
    CT_pp::Array{Float64,4}     # D x D x L x L
    CT_cp::Array{Float64,4}     # D x D x L x L
    CScum_p::Array{Float64,3}   # D x (NCORE+1) x L
    Hfull::Matrix{Float64}      # (NCORE + 2*ncm_fam) x (NCORE + 2*ncm_fam) scratch
end

"""
    build_frechet_power_bin_ctx(ctx, aug_cdf, targets::FrechetReferenceTargets) -> FrechetPowerBinHessCtx

`aug_cdf` must be a `build_cm_frechet_augmented_obj_archB(ctx, CS, targets; ...)`
result (CDF-ONLY, `ncm == D*length(targets.probs)`) -- its `.ncore`/`.L`/
`.origins`/`.refIndex1`/`.z`/`.contrasts` describe the shared layout both
families use. `targets` supplies `theta_star`/`sigma` (-> `pw`) and both
target vectors.
"""
function build_frechet_power_bin_ctx(ctx, aug_cdf, targets::FrechetReferenceTargets)
    cctx = build_cm_bin_ctx(ctx, aug_cdf)   # existing production constructor, UNCHANGED
    D = cctx.D; L = cctx.L; NCORE = cctx.NCORE
    W = size(ctx.U, 1)
    L1 = L + 1
    pw = (1.0 - targets.sigma) / targets.theta_star
    Upow = ctx.U .^ pw
    Hn = NCORE + 2 * cctx.ncm   # cctx.ncm == nO*L + L (one family's width)
    return FrechetPowerBinHessCtx(cctx, Upow, pw, targets.targets, targets.power_targets,
        zeros(D, D, L1, L1), zeros(D, D, L1, L1), zeros(D, NCORE + 1, L1),
        zeros(D, D, L, L), zeros(D, D, L, L), zeros(D, NCORE + 1, L),
        Matrix{Float64}(undef, Hn, Hn))
end

"Raw weighted bin tables for the POWER family and the CDF×POWER cross family. O(W*(D^2 + D*(NCORE+1)))."
function build_frechet_power_bin_tables!(fctx::FrechetPowerBinHessCtx, E::AbstractMatrix{Float64}, w::AbstractVector{Float64})
    cctx = fctx.cctx
    D = cctx.D; NCORE = cctx.NCORE; Bidx = cctx.Bidx; Upow = fctx.Upow
    Tpp = fctx.Ttab_pp; Tcp = fctx.Ttab_cp; Sp = fctx.Stab_p
    fill!(Tpp, 0.0); fill!(Tcp, 0.0); fill!(Sp, 0.0)
    W = size(E, 1)
    @inbounds for s in 1:W
        ws = w[s]
        for x in 1:D
            bx = Bidx[s, x]
            uxs = Upow[s, x]
            wux = ws * uxs
            for j in 1:NCORE
                Sp[x, j, bx] += wux * E[s, j]
            end
            Sp[x, NCORE + 1, bx] += wux   # constant-1 "core" column -> family marginal table
        end
        for x in 1:D
            bx = Bidx[s, x]
            uxs = Upow[s, x]
            for y in 1:D
                by = Bidx[s, y]
                uys = Upow[s, y]
                Tpp[x, y, bx, by] += ws * uxs * uys
                Tcp[x, y, bx, by] += ws * uys      # CDF-weight (=1) on x, POWER-weight on y
            end
        end
    end
    return nothing
end

function _frechet_prefix_sum_2d!(CT::Array{Float64,4}, T::Array{Float64,4}, D::Int, L::Int)
    @inbounds for x in 1:D, y in 1:D
        for l in 1:L
            for lp in 1:L
                v = T[x, y, l, lp]
                v += (l > 1 ? CT[x, y, l-1, lp] : 0.0)
                v += (lp > 1 ? CT[x, y, l, lp-1] : 0.0)
                v -= (l > 1 && lp > 1) ? CT[x, y, l-1, lp-1] : 0.0
                CT[x, y, l, lp] = v
            end
        end
    end
    return nothing
end

function _frechet_prefix_sum_1d!(CS::Array{Float64,3}, S::Array{Float64,3}, D::Int, NC::Int, L::Int)
    @inbounds for x in 1:D, j in 1:NC
        acc = 0.0
        for l in 1:L
            acc += S[x, j, l]
            CS[x, j, l] = acc
        end
    end
    return nothing
end

"2D-prefix-sum Ttab_pp/Ttab_cp into CT_pp/CT_cp; 1D-prefix-sum Stab_p into CScum_p."
function frechet_power_prefix_sum!(fctx::FrechetPowerBinHessCtx)
    D = fctx.cctx.D; L = fctx.cctx.L; NCORE = fctx.cctx.NCORE
    _frechet_prefix_sum_2d!(fctx.CT_pp, fctx.Ttab_pp, D, L)
    _frechet_prefix_sum_2d!(fctx.CT_cp, fctx.Ttab_cp, D, L)
    _frechet_prefix_sum_1d!(fctx.CScum_p, fctx.Stab_p, D, NCORE + 1, L)
    return nothing
end

"""
    hessian_cm_frechet_cdf_power_structured!(h, obj, fctx::FrechetPowerBinHessCtx)

Architecture-C Hessian callback for the combined CDF+POWER fixed-Fréchet
block (`feature_set=:cdf_power`, `:cumulative` basis). Assembles all six
sub-blocks of

    [ H_EE    H_E,CDFc   H_E,CDFcm   H_E,POWc   H_E,POWcm  ]
    [ ...     H_CDFc,CDFc H_CDFc,CDFcm H_CDFc,POWc H_CDFc,POWcm ]
    [ ...     ...         H_CDFcm,CDFcm H_CDFcm,POWc H_CDFcm,POWcm ]
    [ ...     ...         ...          H_POWc,POWc  H_POWc,POWcm  ]
    [ ...     ...         ...          ...          H_POWcm,POWcm ]

(symmetric; only the upper triangle is computed then mirrored). The CDF-CDF
sub-blocks use EXACTLY the formulas in this branch's `cm_frechet_hessian.jl`
(`hessian_cm_frechet_structured!`); the POWER-POWER sub-blocks are the
structural analogue with `CT_pp`/`CScum_p`/`tpow` in place of
`CT`/`CScum`/`p`; the CDF×POWER cross sub-blocks are new (see the doc note
at the top of this file for the derivation).
"""
function hessian_cm_frechet_cdf_power_structured!(h, obj, fctx::FrechetPowerBinHessCtx)
    cctx = fctx.cctx
    @unpack H, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins
    ncm_fam = cctx.ncm
    @assert ncm_fam == nO * L + L
    p = fctx.p; tpow = fctx.tpow
    R = cctx.R

    E = @view H[:, 2:1+NCORE]
    build_bin_tables!(cctx, E, w)
    prefix_sum_tables!(cctx)
    build_frechet_power_bin_tables!(fctx, E, w)
    frechet_power_prefix_sum!(fctx)

    Hfull = fctx.Hfull
    fill!(Hfull, 0.0)
    n = NCORE + 2 * ncm_fam
    @assert size(Hfull, 1) == n

    CS_ = cctx.CScum; CT = cctx.CT
    CS_p = fctx.CScum_p; CT_pp = fctx.CT_pp; CT_cp = fctx.CT_cp

    cdf_off = NCORE
    cdf_common_off = NCORE + nO * L
    pow_off = NCORE + ncm_fam
    pow_common_off = NCORE + ncm_fam + nO * L

    # ---- H_EE ----
    Ews = cctx.Ews
    @views Ews .= E .* sqrt.(w)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    BLAS.gemm!('T', 'N', 1 / M, Ews, Ews, 0.0, HEE)

    # ---- CDF-CDF sub-blocks (unchanged formulas, same as hessian_cm_frechet_structured!) ----
    Hraw_EC = Matrix{Float64}(undef, NCORE, nO)
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            for j in 1:NCORE
                Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
            end
        end
        cols = cdf_off + (l-1)*nO + 1 : cdf_off + l*nO
        block = R === nothing ? Hraw_EC : Hraw_EC * R
        @views Hfull[1:NCORE, cols] .= block
        @views Hfull[cols, 1:NCORE] .= transpose(block)
    end
    Hraw_CC = Matrix{Float64}(undef, nO, nO)
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, pp) in enumerate(origins)
                Hraw_CC[oi, pi] = (CT[o, pp, l, lp] - CT[o, refIndex1, l, lp] - CT[refIndex1, pp, l, lp] + CT[refIndex1, refIndex1, l, lp]) / M
            end
            rows = cdf_off + (l-1)*nO + 1 : cdf_off + l*nO
            cols = cdf_off + (lp-1)*nO + 1 : cdf_off + lp*nO
            block = R === nothing ? Hraw_CC : (R' * Hraw_CC * R)
            @views Hfull[rows, cols] .= block
        end
    end
    Wsum_E = Vector{Float64}(undef, NCORE)
    @inbounds for j in 1:NCORE
        Wsum_E[j] = CS_[refIndex1, j, L] + cctx.Stab[refIndex1, j, L+1]
    end
    Hraw_Ecommon = Matrix{Float64}(undef, NCORE, L)
    @inbounds for l in 1:L
        for j in 1:NCORE
            Hraw_Ecommon[j, l] = (CS_[refIndex1, j, l] - p[l] * Wsum_E[j]) / M
        end
    end
    cdf_common_cols = cdf_common_off+1 : cdf_common_off+L
    @views Hfull[1:NCORE, cdf_common_cols] .= Hraw_Ecommon
    @views Hfull[cdf_common_cols, 1:NCORE] .= transpose(Hraw_Ecommon)
    @inbounds for l in 1:L
        for lp in 1:L
            v = (CT[refIndex1, refIndex1, l, lp] - p[lp]*CT[refIndex1, refIndex1, l, l] - p[l]*CT[refIndex1, refIndex1, lp, lp] + p[l]*p[lp]*M) / M
            Hfull[cdf_common_off + l, cdf_common_off + lp] = v
        end
    end
    Hraw_common_contrast = Matrix{Float64}(undef, L, nO)
    @inbounds for lp in 1:L
        for l in 1:L
            for (oi, o) in enumerate(origins)
                Hraw_common_contrast[l, oi] = (CT[refIndex1, o, l, lp] - CT[refIndex1, refIndex1, l, lp] -
                                                p[l]*CT[o, o, lp, lp] + p[l]*CT[refIndex1, refIndex1, lp, lp]) / M
            end
        end
        block_cc = R === nothing ? Hraw_common_contrast : Hraw_common_contrast * R
        ccols = cdf_off + (lp-1)*nO + 1 : cdf_off + lp*nO
        @views Hfull[cdf_common_cols, ccols] .= block_cc
        @views Hfull[ccols, cdf_common_cols] .= transpose(block_cc)
    end

    # ---- POWER-POWER sub-blocks (structural analogue, CT_pp/CScum_p/tpow) ----
    Hraw_EC_p = Matrix{Float64}(undef, NCORE, nO)
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            for j in 1:NCORE
                Hraw_EC_p[j, oi] = (CS_p[o, j, l] - CS_p[refIndex1, j, l]) / M
            end
        end
        cols = pow_off + (l-1)*nO + 1 : pow_off + l*nO
        block = R === nothing ? Hraw_EC_p : Hraw_EC_p * R
        @views Hfull[1:NCORE, cols] .= block
        @views Hfull[cols, 1:NCORE] .= transpose(block)
    end
    Hraw_CC_p = Matrix{Float64}(undef, nO, nO)
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, pp) in enumerate(origins)
                Hraw_CC_p[oi, pi] = (CT_pp[o, pp, l, lp] - CT_pp[o, refIndex1, l, lp] - CT_pp[refIndex1, pp, l, lp] + CT_pp[refIndex1, refIndex1, l, lp]) / M
            end
            rows = pow_off + (l-1)*nO + 1 : pow_off + l*nO
            cols = pow_off + (lp-1)*nO + 1 : pow_off + lp*nO
            block = R === nothing ? Hraw_CC_p : (R' * Hraw_CC_p * R)
            @views Hfull[rows, cols] .= block
        end
    end
    Hraw_Ecommon_p = Matrix{Float64}(undef, NCORE, L)
    @inbounds for l in 1:L
        for j in 1:NCORE
            Hraw_Ecommon_p[j, l] = (CS_p[refIndex1, j, l] - tpow[l] * Wsum_E[j]) / M
        end
    end
    pow_common_cols = pow_common_off+1 : pow_common_off+L
    @views Hfull[1:NCORE, pow_common_cols] .= Hraw_Ecommon_p
    @views Hfull[pow_common_cols, 1:NCORE] .= transpose(Hraw_Ecommon_p)
    @inbounds for l in 1:L
        for lp in 1:L
            v = (CT_pp[refIndex1, refIndex1, l, lp] - tpow[lp]*CS_p[refIndex1, NCORE+1, l] -
                 tpow[l]*CS_p[refIndex1, NCORE+1, lp] + tpow[l]*tpow[lp]*M) / M
            Hfull[pow_common_off + l, pow_common_off + lp] = v
        end
    end
    Hraw_common_contrast_p = Matrix{Float64}(undef, L, nO)
    @inbounds for lp in 1:L
        for l in 1:L
            for (oi, o) in enumerate(origins)
                Hraw_common_contrast_p[l, oi] = (CT_pp[refIndex1, o, l, lp] - CT_pp[refIndex1, refIndex1, l, lp] -
                                                  tpow[l]*CS_p[o, NCORE+1, lp] + tpow[l]*CS_p[refIndex1, NCORE+1, lp]) / M
            end
        end
        block_cc = R === nothing ? Hraw_common_contrast_p : Hraw_common_contrast_p * R
        ccols = pow_off + (lp-1)*nO + 1 : pow_off + lp*nO
        @views Hfull[pow_common_cols, ccols] .= block_cc
        @views Hfull[ccols, pow_common_cols] .= transpose(block_cc)
    end

    # ---- CDF x POWER cross sub-blocks (NEW) ----
    Hraw_cross_cc = Matrix{Float64}(undef, nO, nO)   # rows: CDF-contrast origin, cols: POWER-contrast origin
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, pp) in enumerate(origins)
                Hraw_cross_cc[oi, pi] = (CT_cp[o, pp, l, lp] - CT_cp[o, refIndex1, l, lp] - CT_cp[refIndex1, pp, l, lp] + CT_cp[refIndex1, refIndex1, l, lp]) / M
            end
            rows = cdf_off + (l-1)*nO + 1 : cdf_off + l*nO
            cols = pow_off + (lp-1)*nO + 1 : pow_off + lp*nO
            block = R === nothing ? Hraw_cross_cc : (R' * Hraw_cross_cc * R)
            @views Hfull[rows, cols] .= block
            @views Hfull[cols, rows] .= transpose(block)
        end
    end
    # CDF-contrast(o,l) x POWER-common(l')
    @inbounds for l in 1:L
        for lp in 1:L
            vraw = Vector{Float64}(undef, nO)
            for (oi, o) in enumerate(origins)
                vraw[oi] = (CT_cp[o, refIndex1, l, lp] - CT_cp[refIndex1, refIndex1, l, lp] -
                            tpow[lp] * (CT[o, o, l, l] - CT[refIndex1, refIndex1, l, l])) / M
            end
            v = R === nothing ? vraw : R' * vraw
            cdf_cols_l = cdf_off + (l-1)*nO + 1 : cdf_off + l*nO
            Hfull[cdf_cols_l, pow_common_off + lp] .= v
            Hfull[pow_common_off + lp, cdf_cols_l] .= v
        end
    end
    # CDF-common(l) x POWER-contrast(p,l')
    @inbounds for lp in 1:L
        for l in 1:L
            vraw = Vector{Float64}(undef, nO)
            for (pi, pp) in enumerate(origins)
                vraw[pi] = (CT_cp[refIndex1, pp, l, lp] - CT_cp[refIndex1, refIndex1, l, lp] -
                            p[l] * CS_p[pp, NCORE+1, lp] + p[l] * CS_p[refIndex1, NCORE+1, lp]) / M
            end
            v = R === nothing ? vraw : R' * vraw
            pow_cols_lp = pow_off + (lp-1)*nO + 1 : pow_off + lp*nO
            Hfull[cdf_common_off + l, pow_cols_lp] .= v
            Hfull[pow_cols_lp, cdf_common_off + l] .= v
        end
    end
    # CDF-common(l) x POWER-common(l')
    @inbounds for l in 1:L
        for lp in 1:L
            v = (CT_cp[refIndex1, refIndex1, l, lp] - tpow[lp]*CT[refIndex1, refIndex1, l, l] -
                 p[l]*CS_p[refIndex1, NCORE+1, lp] + p[l]*tpow[lp]*M) / M
            Hfull[cdf_common_off + l, pow_common_off + lp] = v
            Hfull[pow_common_off + lp, cdf_common_off + l] = v
        end
    end

    # symmetrize defensively (analytically symmetric; absorbs FP-order noise)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            h[k] = 0.5 * (Hfull[i, j] + Hfull[j, i])
            k += 1
        end
    end
    return h
end

"Architecture C's hess_cb_builder for the combined CDF+POWER fixed-Fréchet block."
function archC_frechet_cdf_power_hess_cb_builder(fctx::FrechetPowerBinHessCtx)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_frechet_cdf_power" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_frechet_cdf_power_structured!(evalResult.hess, o, fctx)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
