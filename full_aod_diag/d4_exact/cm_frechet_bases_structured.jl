# ============================================================================
# Fast (Architecture C) structured Hessian for the Q1 (interval) basis, CDF
# feature family (task brief Part III.6). Reuses `build_bin_tables!` UNCHANGED
# -- it already accumulates RAW per-bin tables (`Ttab`, `Stab`) BEFORE
# `prefix_sum_tables!` turns them into cumulative (`CT`, `CScum`) tables. The
# interval basis needs exactly the raw tables, so this file skips
# `prefix_sum_tables!` entirely and reads `Ttab`/`Stab` directly -- the O(W*L)
# table-building pass (the expensive part) is 100% shared with the cumulative
# (Q0) structured backend in cm_frechet_hessian.jl; only the O(D^2*L^2)
# assembly formulas differ (cumulative-probability targets p[l] -> bin-mass
# targets dp[l]; prefix-summed cross terms CT[x,y,l,lp] -> raw Ttab[x,y,l,lp],
# which is exactly zero off-diagonal whenever x==y since a single draw lands
# in exactly one bin -- verified by the D=4 gate, not just asserted).
#
# Q2 (whitened) is NOT a fourth independent kernel: it is the congruence
# transform of this Q1 kernel's output, `H_Q2 = T2' H_Q1 T2`, `g_Q2 = T2' g_Q1`
# (task brief Part III.6's literal instruction), cheap because T2 only mixes
# the L-sized quantile axis within each (origin-or-common) column group -- see
# `frechet_whiten_block_transform` in cm_frechet_bases.jl.
# ============================================================================

"""
    build_cm_frechet_interval_bin_ctx(ctx, aug_frechet_interval) -> (cctx=CMBinHessCtx, dp=Vector{Float64})

`aug_frechet_interval` is a `build_cm_frechet_augmented_obj_basis(...; basis=:interval,
feature_set=:cdf_only)` result. Reuses the exact same `CMBinHessCtx` struct/constructor as the
cumulative path (`build_cm_frechet_bin_ctx`) -- `Ttab`/`Stab` are populated identically by
`build_bin_tables!` regardless of which basis will read them.
"""
function build_cm_frechet_interval_bin_ctx(ctx, aug_frechet_interval)
    L = aug_frechet_interval.L; D = ctx.D; origins = aug_frechet_interval.origins; nO = length(origins)
    refIndex1 = aug_frechet_interval.refIndex1; z = aug_frechet_interval.z
    NCORE = aug_frechet_interval.ncore; ncm = aug_frechet_interval.ncm
    R = aug_frechet_interval.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Bidx = Int.(compute_bin_indices(ctx.U, z))
    W = size(ctx.U, 1)
    L1 = L + 1
    cctx = CMBinHessCtx(L, D, nO, origins, refIndex1, z, Bidx, NCORE, ncm, aug_frechet_interval.contrasts, R,
        zeros(D, D, L1, L1), zeros(D, NCORE, L1), zeros(D, D, L, L), zeros(D, NCORE, L),
        Matrix{Float64}(undef, W, NCORE), Matrix{Float64}(undef, NCORE + ncm, NCORE + ncm))
    p = aug_frechet_interval.targets.targets
    dp = Vector{Float64}(undef, L)
    dp[1] = p[1]
    @inbounds for l in 2:L
        dp[l] = p[l] - p[l-1]
    end
    return (cctx = cctx, dp = dp)
end

"""
    hessian_cm_frechet_interval_structured!(h, obj, fctx)

Q1 (interval) analogue of `hessian_cm_frechet_structured!` (cm_frechet_hessian.jl). Formula
substitution only: `CScum`/`CT` (prefix-summed) -> `Stab`/`Ttab` (raw), `p[l]` (cumulative
probability target) -> `dp[l]` (bin-probability-mass target). Calls `build_bin_tables!` but
DELIBERATELY DOES NOT call `prefix_sum_tables!` -- the raw tables ARE the interval-basis
quantities, no summation step needed.
"""
function hessian_cm_frechet_interval_structured!(h, obj, fctx)
    cctx = fctx.cctx; dp = fctx.dp
    @unpack H, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins
    ncm = cctx.ncm
    @assert ncm == nO * L + L

    E = @view H[:, 2:1+NCORE]
    build_bin_tables!(cctx, E, w)
    # NOTE: no prefix_sum_tables! call -- Ttab/Stab (raw) are exactly the interval-basis tables.

    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)

    Ews = cctx.Ews
    @views Ews .= E .* sqrt.(w)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    BLAS.gemm!('T', 'N', 1 / M, Ews, Ews, 0.0, HEE)

    Stab = cctx.Stab; Ttab = cctx.Ttab
    Hraw_EC = Matrix{Float64}(undef, NCORE, nO)
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            for j in 1:NCORE
                Hraw_EC[j, oi] = (Stab[o, j, l] - Stab[refIndex1, j, l]) / M
            end
        end
        cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
        block_ec = cctx.R === nothing ? Hraw_EC : Hraw_EC * cctx.R
        @views Hfull[1:NCORE, cols] .= block_ec
        @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
    end
    Hraw_CC = Matrix{Float64}(undef, nO, nO)
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, pp) in enumerate(origins)
                Hraw_CC[oi, pi] = (Ttab[o, pp, l, lp] - Ttab[o, refIndex1, l, lp] - Ttab[refIndex1, pp, l, lp] + Ttab[refIndex1, refIndex1, l, lp]) / M
            end
            rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
            block = cctx.R === nothing ? Hraw_CC : (cctx.R' * Hraw_CC * cctx.R)
            @views Hfull[rows, cols] .= block
        end
    end

    Wsum_E = Vector{Float64}(undef, NCORE)
    @inbounds for j in 1:NCORE
        s = 0.0
        for l in 1:L+1
            s += Stab[refIndex1, j, l]
        end
        Wsum_E[j] = s
    end
    common_offset = NCORE + nO * L

    Hraw_Ecommon = Matrix{Float64}(undef, NCORE, L)
    @inbounds for l in 1:L
        for j in 1:NCORE
            Hraw_Ecommon[j, l] = (Stab[refIndex1, j, l] - dp[l] * Wsum_E[j]) / M
        end
    end
    common_cols = common_offset + 1 : common_offset + L
    @views Hfull[1:NCORE, common_cols] .= Hraw_Ecommon
    @views Hfull[common_cols, 1:NCORE] .= transpose(Hraw_Ecommon)

    @inbounds for l in 1:L
        for lp in 1:L
            joint = (l == lp) ? Ttab[refIndex1, refIndex1, l, l] : 0.0
            marg_l = Ttab[refIndex1, refIndex1, l, l]
            marg_lp = Ttab[refIndex1, refIndex1, lp, lp]
            v = (joint - dp[lp]*marg_l - dp[l]*marg_lp + dp[l]*dp[lp]*M) / M
            Hfull[common_offset + l, common_offset + lp] = v
        end
    end

    Hraw_common_contrast = Matrix{Float64}(undef, L, nO)
    @inbounds for lp in 1:L
        for l in 1:L
            for (oi, o) in enumerate(origins)
                Hraw_common_contrast[l, oi] = (Ttab[refIndex1, o, l, lp] - Ttab[refIndex1, refIndex1, l, lp] -
                                                dp[l]*Ttab[o, o, lp, lp] + dp[l]*Ttab[refIndex1, refIndex1, lp, lp]) / M
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

"Architecture C's hess_cb_builder for the Q1 (interval) fixed-Fréchet CDF-only basis."
function archC_frechet_interval_hess_cb_builder(fctx)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_frechet_interval" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_frechet_interval_structured!(evalResult.hess, o, fctx)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
