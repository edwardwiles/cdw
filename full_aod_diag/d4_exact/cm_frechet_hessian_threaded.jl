# Threaded Architecture-C structured Hessian for fixed-Fréchet CDF-only
# (frechet_feature_set=:cdf_only). Task: FIXED_FRECHET_INNER_SOLVER production-feasibility
# 2026-07-24 (CDF-only addendum). Mirrors cm_hessian_threaded.jl's
# hessian_cm_structured_v2!/archC_hess_cb_builder_v2 exactly, applied to
# cm_frechet_hessian.jl::hessian_cm_frechet_structured! instead of
# cm_hessian_architectures.jl::hessian_cm_structured!. The only new work relative to the flexible-CM
# v2 kernel is the CDF-only "common" block tail (H_E,common / H_common,common / H_common,contrast),
# copied verbatim from cm_frechet_hessian.jl -- NOT threaded (it's O(NCORE*L) / O(L^2), a few
# hundred thousand flops at D=20/L=50, dwarfed by the O(W*D*NCORE) bin-table pass that IS threaded).
#
# Requires cm_hessian_threaded.jl (ThreadLocalBinScratch, build_thread_local_scratch,
# build_bin_tables_threaded!, prefix_sum_tables_threaded!) to be included first.

"""
    hessian_cm_frechet_structured_v2!(h, obj, fctx; threaded_bins=false, tls=nothing, use_syrk=true)

Threaded/syrk drop-in for `hessian_cm_frechet_structured!` (cm_frechet_hessian.jl), CDF-only fixed
Fréchet. Produces the SAME packed upper-triangular Hessian to within float-accumulation-order
differences (validated at D=4 and bounded D=20 in test_frechet_hessian_threaded_gates.jl).
"""
function hessian_cm_frechet_structured_v2!(h, obj, fctx; threaded_bins::Bool = false,
                                            tls::Union{Nothing,ThreadLocalBinScratch} = nothing,
                                            use_syrk::Bool = true)
    cctx = fctx.cctx; p = fctx.p
    @unpack H, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins
    ncm = cctx.ncm   # D*L
    @assert ncm == nO * L + L

    E = @view H[:, 2:1+NCORE]
    if threaded_bins
        tls === nothing && error("hessian_cm_frechet_structured_v2!(threaded_bins=true) requires tls (build_thread_local_scratch(cctx))")
        build_bin_tables_threaded!(cctx, tls, E, w)
        prefix_sum_tables_threaded!(cctx)
    else
        build_bin_tables!(cctx, E, w)
        prefix_sum_tables!(cctx)
    end

    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)

    # ---- H_EE: syrk or gemm ----
    Ews = cctx.Ews
    @views Ews .= E .* sqrt.(w)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    if use_syrk
        BLAS.syrk!('U', 'T', 1 / M, Ews, 0.0, HEE)
        @inbounds for i in 1:NCORE, j in 1:(i-1)
            HEE[i, j] = HEE[j, i]
        end
    else
        BLAS.gemm!('T', 'N', 1 / M, Ews, Ews, 0.0, HEE)
    end

    # ---- H_EC[contrast], H_CC[contrast,contrast] -- verbatim from cm_frechet_hessian.jl ----
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

    # ---- H_E,common / H_common,common / H_common,contrast -- verbatim, not threaded (small) ----
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

"Threaded/syrk KNITRO Hessian-callback builder, CDF-only fixed Fréchet. Mirrors archC_hess_cb_builder_v2."
function archC_frechet_hess_cb_builder_v2(fctx; threaded_bins::Bool = false,
                                           tls::Union{Nothing,ThreadLocalBinScratch} = nothing,
                                           use_syrk::Bool = true)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_frechet_v2" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_frechet_structured_v2!(evalResult.hess, o, fctx; threaded_bins = threaded_bins, tls = tls, use_syrk = use_syrk)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
