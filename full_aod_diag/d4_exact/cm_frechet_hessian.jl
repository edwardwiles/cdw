# ============================================================================
# Fixed-Fréchet-marginals Architecture C (structured Hessian) extension
# (2026-07-23). Purely additive: reuses `CMBinHessCtx`, `build_bin_tables!`,
# `prefix_sum_tables!` (cm_hessian_architectures.jl) COMPLETELY UNCHANGED --
# neither reads `cctx.ncm` (verified by direct inspection), so a genuine
# `CMBinHessCtx` sized for `ncm = D*L` (the frechet count) is a fully valid
# drop-in argument to both. Only the final Hessian-block ASSEMBLY step
# (`hessian_cm_frechet_structured!` below) is new -- the expensive O(W*...)
# table-building pass is 100% shared with flexible CM.
#
# Derivation: docs/FIXED_FRECHET_MARGINALS_MATH_NOTE_2026-07-23.md §7.
# Correctness gate: D=4 dense-vs-structured comparison against Architecture A
# (task brief §10.1) -- the algebra below is verified numerically there, not
# trusted from the derivation alone.
# ============================================================================

"""
    build_cm_frechet_bin_ctx(ctx, aug_frechet) -> (cctx=CMBinHessCtx, p=Vector{Float64})

`aug_frechet` is a `build_cm_frechet_augmented_obj(_archB)` result (needs
`L`, `origins`, `refIndex1`, `z`, `ncore`, `ncm` (= D*L), `contrasts`,
`targets`). Builds a genuine `CMBinHessCtx` (same type flexible CM's
Architecture C uses) with `ncm` set to the FRECHET count and `Hfull` sized
accordingly; `cctx.ncm`/`cctx.Hfull` are never read by
`build_bin_tables!`/`prefix_sum_tables!` (verified), so this is a fully
valid, type-correct reuse, not a workaround.
"""
function build_cm_frechet_bin_ctx(ctx, aug_frechet)
    L = aug_frechet.L; D = ctx.D; origins = aug_frechet.origins; nO = length(origins)
    refIndex1 = aug_frechet.refIndex1; z = aug_frechet.z
    NCORE = aug_frechet.ncore; ncm = aug_frechet.ncm   # = D*L
    R = aug_frechet.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Bidx = Int.(compute_bin_indices(ctx.U, z))   # see cm_frechet_moments.jl's note on the ambiguous compute_bin_indices dispatch
    W = size(ctx.U, 1)
    L1 = L + 1
    cctx = CMBinHessCtx(L, D, nO, origins, refIndex1, z, Bidx, NCORE, ncm, aug_frechet.contrasts, R,
        zeros(D, D, L1, L1), zeros(D, NCORE, L1), zeros(D, D, L, L), zeros(D, NCORE, L),
        Matrix{Float64}(undef, W, NCORE), Matrix{Float64}(undef, NCORE + ncm, NCORE + ncm))
    return (cctx = cctx, p = aug_frechet.targets.targets)
end

"""
    hessian_cm_frechet_structured!(h, obj, fctx)

Architecture C Hessian callback for fixed Fréchet marginals. `fctx` is a
`build_cm_frechet_bin_ctx` result. Reuses `build_bin_tables!`/
`prefix_sum_tables!` UNCHANGED, then assembles:
  - H_EE, H_EC[contrast block], H_CC[contrast,contrast] -- IDENTICAL formulas
    to `hessian_cm_structured!` (cm_hessian_architectures.jl), just placed at
    the same leading `nO*L` CM columns (unchanged positions).
  - H_E,common / H_common,common / H_common,contrast -- NEW, math note §7.
"""
function hessian_cm_frechet_structured!(h, obj, fctx)
    cctx = fctx.cctx; p = fctx.p
    @unpack H, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins
    ncm = cctx.ncm   # D*L
    @assert ncm == nO * L + L

    E = @view H[:, 2:1+NCORE]
    build_bin_tables!(cctx, E, w)
    prefix_sum_tables!(cctx)

    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)

    # ---- H_EE (unchanged) ----
    Ews = cctx.Ews
    @views Ews .= E .* sqrt.(w)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    BLAS.gemm!('T', 'N', 1 / M, Ews, Ews, 0.0, HEE)

    # ---- H_EC[contrast], H_CC[contrast,contrast] -- unchanged formulas, unchanged column positions ----
    CS_ = cctx.CScum
    Hraw_EC = Matrix{Float64}(undef, NCORE, nO)
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            for j in 1:NCORE
                Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
            end
        end
        cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
        block_ec = cctx.R === nothing ? Hraw_EC : Hraw_EC * cctx.R
        @views Hfull[1:NCORE, cols] .= block_ec
        @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
    end
    CT = cctx.CT
    Hraw_CC = Matrix{Float64}(undef, nO, nO)
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, pp) in enumerate(origins)
                Hraw_CC[oi, pi] = (CT[o, pp, l, lp] - CT[o, refIndex1, l, lp] - CT[refIndex1, pp, l, lp] + CT[refIndex1, refIndex1, l, lp]) / M
            end
            rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
            block = cctx.R === nothing ? Hraw_CC : (cctx.R' * Hraw_CC * cctx.R)
            @views Hfull[rows, cols] .= block
        end
    end

    # ---- NEW: H_E,common / H_common,common / H_common,contrast ----
    # Wsum_E[j] = sum_s w_s E[s,j] = CS_[x,j,L] + Stab[x,j,L+1], independent of x -- use refIndex1.
    Stab = cctx.Stab
    Wsum_E = Vector{Float64}(undef, NCORE)
    @inbounds for j in 1:NCORE
        Wsum_E[j] = CS_[refIndex1, j, L] + Stab[refIndex1, j, L+1]
    end
    common_offset = NCORE + nO * L   # trailing block starts right after the contrast block

    # H_E,common[j,l]
    Hraw_Ecommon = Matrix{Float64}(undef, NCORE, L)
    @inbounds for l in 1:L
        for j in 1:NCORE
            Hraw_Ecommon[j, l] = (CS_[refIndex1, j, l] - p[l] * Wsum_E[j]) / M
        end
    end
    common_cols = common_offset + 1 : common_offset + L
    @views Hfull[1:NCORE, common_cols] .= Hraw_Ecommon
    @views Hfull[common_cols, 1:NCORE] .= transpose(Hraw_Ecommon)

    # H_common,common[l,l']
    @inbounds for l in 1:L
        for lp in 1:L
            v = (CT[refIndex1, refIndex1, l, lp] - p[lp]*CT[refIndex1, refIndex1, l, l] -
                 p[l]*CT[refIndex1, refIndex1, lp, lp] + p[l]*p[lp]*M) / M
            Hfull[common_offset + l, common_offset + lp] = v
        end
    end

    # H_common,contrast[l,(o,l')] -- raw (pre-R) then congruence on the contrast side only
    Hraw_common_contrast = Matrix{Float64}(undef, L, nO)   # rows: common l, cols: raw contrast origin at fixed l'
    @inbounds for lp in 1:L
        for l in 1:L
            for (oi, o) in enumerate(origins)
                Hraw_common_contrast[l, oi] = (CT[refIndex1, o, l, lp] - CT[refIndex1, refIndex1, l, lp] -
                                                p[l]*CT[o, o, lp, lp] + p[l]*CT[refIndex1, refIndex1, lp, lp]) / M
            end
        end
        block_cc = cctx.R === nothing ? Hraw_common_contrast : Hraw_common_contrast * cctx.R
        contrast_cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
        @views Hfull[common_cols, contrast_cols] .= block_cc
        @views Hfull[contrast_cols, common_cols] .= transpose(block_cc)
    end

    # symmetrize defensively (analytically symmetric; absorbs FP-order noise -- same pattern as hessian_cm_structured!)
    n = NCORE + ncm
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            h[k] = 0.5 * (Hfull[i, j] + Hfull[j, i])
            k += 1
        end
    end
    return h
end

"Architecture C's hess_cb_builder for fixed Fréchet marginals: closes over a `build_cm_frechet_bin_ctx` result."
function archC_frechet_hess_cb_builder(fctx)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_frechet" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_frechet_structured!(evalResult.hess, o, fctx)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
