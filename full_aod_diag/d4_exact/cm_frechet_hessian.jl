# ============================================================================
# Fixed-Fréchet-marginals Architecture C (structured Hessian) extension,
# CDF-ONLY (eq.37 alone, `frechet_feature_set=:cdf_only` legacy/diagnostic
# path). Port-prep 2026-07-24, adapted from the pre-omit-ROW reconciliation
# archive. Reuses `CMBinHessCtx`, `build_bin_tables!`, `prefix_sum_tables!`
# (cm_hessian_architectures.jl) COMPLETELY UNCHANGED -- neither reads
# `cctx.ncm` (verified by direct inspection), so a genuine `CMBinHessCtx`
# sized for `ncm = D*L` (the frechet CDF-only count) is a fully valid
# drop-in argument to both. Only the final Hessian-block ASSEMBLY step
# (`hessian_cm_frechet_structured!` below) is new relative to flexible CM --
# the expensive O(W*...) table-building pass is 100% shared.
#
# For the FULL paper spec (`:cdf_power`, the default), see
# `cm_frechet_power_hessian_structured.jl` instead -- this file's kernel is
# consumed only when `frechet_feature_set=:cdf_only` is explicitly selected.
#
# Correctness gate: D=4 dense-vs-structured comparison against Architecture A
# in test_frechet_bases_d4_gates.jl.
# ============================================================================

"""
    build_cm_frechet_bin_ctx(ctx, aug_frechet) -> (cctx=CMBinHessCtx, p=Vector{Float64})

`aug_frechet` is a `build_cm_frechet_augmented_obj(_archB)` result (CDF-only,
needs `.L`, `.origins`, `.refIndex1`, `.z`, `.ncore`, `.ncm` (= D*L),
`.contrasts`, `.targets`). Delegates to production's own
`build_cm_bin_ctx(ctx, aug)` (cm_hessian_architectures.jl) -- unchanged from
flexible CM's own constructor, since `aug_frechet` already exposes the same
field set `build_cm_bin_ctx` consumes; no hand-rolled positional
`CMBinHessCtx(...)` call needed (production's struct has grown to 17 fields
with pre-sized scratch tensors -- see docs/CM_PRODUCTION_HOOK_INTERFACE_SPEC_2026-07-24.md
§1 -- so a raw positional constructor would be fragile against future field
additions; `build_cm_bin_ctx` is the sanctioned entry point).
"""
function build_cm_frechet_bin_ctx(ctx, aug_frechet)
    cctx = build_cm_bin_ctx(ctx, aug_frechet)
    return (cctx = cctx, p = aug_frechet.targets.targets)
end

"""
    hessian_cm_frechet_structured!(h, obj, fctx)

Architecture C Hessian callback for fixed Fréchet marginals, CDF-only.
`fctx` is a `build_cm_frechet_bin_ctx` result. Reuses `build_bin_tables!`/
`prefix_sum_tables!` UNCHANGED, then assembles:
  - H_EE, H_EC[contrast block], H_CC[contrast,contrast] -- IDENTICAL formulas
    to `hessian_cm_structured!` (cm_hessian_architectures.jl), just placed at
    the same leading `nO*L` CM columns (unchanged positions).
  - H_E,common / H_common,common / H_common,contrast -- pinned to the
    analytic F* target `p[l]` rather than an empirical reference moment.
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

    # ---- H_E,common / H_common,common / H_common,contrast ----
    Wsum_E = Vector{Float64}(undef, NCORE)
    @inbounds for j in 1:NCORE
        Wsum_E[j] = CS_[refIndex1, j, L] + cctx.Stab[refIndex1, j, L+1]
    end
    common_offset = NCORE + nO * L

    Hraw_Ecommon = Matrix{Float64}(undef, NCORE, L)
    @inbounds for l in 1:L
        for j in 1:NCORE
            Hraw_Ecommon[j, l] = (CS_[refIndex1, j, l] - p[l] * Wsum_E[j]) / M
        end
    end
    common_cols = common_offset + 1 : common_offset + L
    @views Hfull[1:NCORE, common_cols] .= Hraw_Ecommon
    @views Hfull[common_cols, 1:NCORE] .= transpose(Hraw_Ecommon)

    @inbounds for l in 1:L
        for lp in 1:L
            v = (CT[refIndex1, refIndex1, l, lp] - p[lp]*CT[refIndex1, refIndex1, l, l] -
                 p[l]*CT[refIndex1, refIndex1, lp, lp] + p[l]*p[lp]*M) / M
            Hfull[common_offset + l, common_offset + lp] = v
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
        block_cc = cctx.R === nothing ? Hraw_common_contrast : Hraw_common_contrast * cctx.R
        contrast_cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
        @views Hfull[common_cols, contrast_cols] .= block_cc
        @views Hfull[contrast_cols, common_cols] .= transpose(block_cc)
    end

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

"Architecture C's hess_cb_builder for fixed Fréchet marginals, CDF-only."
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
