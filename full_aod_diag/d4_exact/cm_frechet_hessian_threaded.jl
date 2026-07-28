# Phase 0 gate closure (2026-07-26, production-audit task): threaded Architecture-C Hessian
# variant for the common-Fréchet level block. Closes the disclosed gap in
# `cm_frechet_hessian.jl::archC_frechet_hess_cb_builder`'s own docstring: "The threaded bin-table
# variant (hessian_cm_structured_v2!) is NOT extended for the level block yet".
#
# Design: per COMMON_FRECHET_HESSIAN_ARCHITECTURE_2026-07-25.md, the level block's three new
# Hessian sub-blocks (H_E,level / H_CM,level / H_level,level) are pure linear combinations of the
# SAME Ttab/CT/Stab/CScum tables the CM block already reads -- no separate O(W) pass. This means
# threading the bin-table construction (`build_bin_tables_threaded!`/`prefix_sum_tables_threaded!`,
# cm_hessian_threaded.jl, UNCHANGED) automatically threads the dominant cost for the level block
# too; only the small O(D*NCORE*L + D^2*L^2) assembly tail stays serial, exactly as it does for
# plain CM's own H_EC/H_CC in `hessian_cm_structured_v2!`.
#
# Both the H_EC/H_CC tail and the three level-block correction terms below are copied VERBATIM from
# `hessian_cm_structured_v2!` (cm_hessian_threaded.jl) and `hessian_cm_frechet_structured!`
# (cm_frechet_hessian.jl) respectively -- not re-derived -- per this project's own stated
# methodology for this feature ("reuse, don't rebuild the bin-contingency tables"; "the H_EC/H_CC
# tail is copied verbatim from the original to minimize the chance of a second divergent bug
# site"). Only the dispatch on `threaded_bins`/`tls` is new.
#
# Depends on (must already be included): cm_hessian_architectures.jl (CMBinHessCtx, build_bin_tables!,
# prefix_sum_tables!, _fill_cm_HEE!), cm_hessian_threaded.jl (ThreadLocalBinScratch,
# build_bin_tables_threaded!, prefix_sum_tables_threaded!), cm_frechet_hessian.jl (this file's
# serial sibling, for the docstring cross-reference and the level_targets semantics).

"""
    hessian_cm_frechet_structured_v2!(h, obj, cctx::CMBinHessCtx, level_targets; threaded_bins=false, tls=nothing)

Threaded-bin-table Architecture-C Hessian callback for `marginal_restriction=:common_frechet`,
producing the byte-identical packed Hessian as `hessian_cm_frechet_structured!` (serial) up to
floating-point summation-order noise -- see `test_cm_frechet_threaded_hessian_gates.jl` for the
correctness gate. `use_syrk` is intentionally omitted (matching `hessian_cm_structured_v2!`'s own
note: `_fill_cm_HEE!`'s shared dense fallback always uses `gemm!`; H_EE dispatch is identical
either way).
"""
function hessian_cm_frechet_structured_v2!(h, obj, cctx::CMBinHessCtx, level_targets::Vector{Float64};
                                            threaded_bins::Bool = false,
                                            tls::Union{Nothing,ThreadLocalBinScratch} = nothing)
    # True no-H operator bundle (2026-07-28 continuation): same fix as the serial
    # hessian_cm_frechet_structured! (cm_frechet_hessian.jl) -- see that function's own comment.
    @unpack M, arg0, arg2, ddPsi! = obj
    H = _dense_H_or_nothing(obj)
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins
    ncm_cm = nO * L
    ncm_level = ncm - ncm_cm
    @assert ncm_level == L "hessian_cm_frechet_structured_v2!: cctx.ncm=$(cctx.ncm) inconsistent with D*L (got ncm_level=$ncm_level, expected L=$L) -- was cctx built from a :common_frechet aug?"
    invsqrtD = 1.0 / sqrt(D)

    # No-moments/no-composite-G task (2026-07-28): `E` no longer constructed eagerly -- see the
    # identical change/rationale in cm_hessian_architectures.jl::hessian_cm_structured!.
    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)
    cf = cctx.core_cf_ref[]

    HEE = @view Hfull[1:NCORE, 1:NCORE]
    _fill_cm_HEE!(HEE, w, obj, cctx, H, M)   # UNCHANGED -- shared winner-pair backend; may rebuild cctx.core_ws/core_ws_for for this cf

    # Winner-aware H_ER phase (2026-07-27), Section 3 Part A: SAME decision function as the serial
    # hessian_cm_frechet_structured! (cm_frechet_hessian.jl) and as flexible-CM's own
    # hessian_cm_structured_v2! (cm_hessian_threaded.jl) -- reused, not re-derived.
    use_winner_bin = _cm_cross_hessian_wants_winner_bin(cctx, cf)
    if threaded_bins
        tls === nothing && error("hessian_cm_frechet_structured_v2!(threaded_bins=true) requires tls (build_thread_local_scratch(cctx))")
        build_bin_tables_threaded!(cctx, tls, H, w; fill_S = !use_winner_bin)
        prefix_sum_tables_threaded!(cctx; fill_S = !use_winner_bin)
    else
        build_bin_tables!(cctx, H, w; fill_S = !use_winner_bin)
        prefix_sum_tables!(cctx; fill_S = !use_winner_bin)
    end

    local wctx, cross_ws
    if use_winner_bin
        record_winner_cross_hessian_call!()
        wctx = serial_ctx(cctx.core_ws)
        cross_ws = _ensure_cm_cross_scratch!(cctx, wctx.ncolI, D, L)
        winner_pair_cross_hessian_fill!(wctx, cross_ws, obj, cctx.Bidx)
    else
        record_dense_cross_hessian_call!()
    end

    CS_ = cctx.CScum
    CT = cctx.CT

    # ---- H_EC, H_CC: verbatim from hessian_cm_structured_v2! ----
    Hraw_EC = cctx.Hraw_EC
    @inbounds for l in 1:L
        if use_winner_bin
            winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, cross_ws, l, origins, refIndex1, M)
        else
            for (oi, o) in enumerate(origins)
                for j in 1:NCORE
                    Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
                end
            end
        end
        cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
        block_ec = if cctx.R === nothing
            Hraw_EC
        else
            mul!(cctx.block_ec, Hraw_EC, cctx.R)
        end
        @views Hfull[1:NCORE, cols] .= block_ec
        @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
    end

    # harmonization task (2026-07-28): extracted to the shared fill_cm_HCC! (cm_hessian_architectures.jl).
    fill_cm_HCC!(Hfull, cctx, M)

    # ---- level-block terms: verbatim from hessian_cm_frechet_structured! ----
    Bidx = cctx.Bidx
    Wraw = size(Bidx, 1)
    Wtab = zeros(D, L + 1)
    @inbounds for s in 1:Wraw
        ws = w[s]
        for x in 1:D
            Wtab[x, Bidx[s, x]] += ws
        end
    end
    T1 = zeros(D, L)
    @inbounds for x in 1:D
        acc = 0.0
        for l in 1:L
            acc += Wtab[x, l]
            T1[x, l] = acc
        end
    end
    Wtot = sum(w)

    # ---- H_E,level: winner-aware (Section 3 Part B) -- see hessian_cm_frechet_structured!'s own
    # comment (cm_frechet_hessian.jl) for the full rationale; verbatim dispatch logic, just threaded
    # bin-table construction above it. ----
    level_off = NCORE + ncm_cm
    if use_winner_bin
        Esum_wb = Vector{Float64}(undef, NCORE)
        winner_pair_cross_hessian_esum!(Esum_wb, wctx, cross_ws, w, Wtot)
        colsum = Vector{Float64}(undef, NCORE)
        @inbounds for l in 1:L
            tl = level_targets[l]
            winner_pair_cross_hessian_colsum!(colsum, wctx, cross_ws, l)
            for j in 1:NCORE
                v = invsqrtD * colsum[j] / M - tl * Esum_wb[j] / M
                Hfull[j, level_off + l] = v
                Hfull[level_off + l, j] = v
            end
        end
    else
        # No-moments/no-composite-G task (2026-07-28): `E` constructed lazily, only here.
        record_dense_frechet_g!()
        E = @view H[:, 2:1+NCORE]
        Esum = Vector{Float64}(undef, NCORE)
        mul!(Esum, E', w)
        @inbounds for l in 1:L
            tl = level_targets[l]
            for j in 1:NCORE
                acc = 0.0
                for o in 1:D
                    acc += CS_[o, j, l]
                end
                v = invsqrtD * acc / M - tl * Esum[j] / M
                Hfull[j, level_off + l] = v
                Hfull[level_off + l, j] = v
            end
        end
    end

    Hraw_cmlevel = Vector{Float64}(undef, nO)
    @inbounds for l in 1:L
        for lp in 1:L
            tlp = level_targets[lp]
            for (oi, o) in enumerate(origins)
                acc_o = 0.0
                acc_ref = 0.0
                for p in 1:D
                    acc_o += CT[o, p, l, lp]
                    acc_ref += CT[refIndex1, p, l, lp]
                end
                Hraw_cmlevel[oi] = invsqrtD * (acc_o - acc_ref) / M - tlp * (T1[o, l] - T1[refIndex1, l]) / M
            end
            cm_rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            col = level_off + lp
            block_cmlevel = cctx.R === nothing ? Hraw_cmlevel : cctx.R' * Hraw_cmlevel
            @views Hfull[cm_rows, col] .= block_cmlevel
            @views Hfull[col, cm_rows] .= block_cmlevel
        end
    end

    invD = 1.0 / D
    @inbounds for l in 1:L
        tl = level_targets[l]
        sum_T1_l = sum(@view T1[:, l])
        for lp in 1:L
            tlp = level_targets[lp]
            acc = 0.0
            for o in 1:D, p in 1:D
                acc += CT[o, p, l, lp]
            end
            sum_T1_lp = sum(@view T1[:, lp])
            Hfull[level_off + l, level_off + lp] =
                invD * acc / M - tlp * invsqrtD * sum_T1_l / M - tl * invsqrtD * sum_T1_lp / M + tl * tlp * Wtot / M
        end
    end

    # harmonization task (2026-07-28): shared pack_upper_cm_hessian! -- see the serial
    # hessian_cm_frechet_structured!'s identical note for the bit-exact-no-op verification.
    n = NCORE + ncm
    pack_upper_cm_hessian!(h, Hfull, NCORE, n)
    return h
end

"""
    archC_frechet_hess_cb_builder_v2(cctx, level_targets; threaded_bins=false, tls=nothing)

KNITRO Hessian-callback builder wrapping `hessian_cm_frechet_structured_v2!`, mirroring
`archC_hess_cb_builder_v2`'s wiring exactly (same `@prof` label suffixed `_v2`,
`_INNER_CALL_COUNTERS[].n_hess_calls` bookkeeping).
"""
function archC_frechet_hess_cb_builder_v2(cctx::CMBinHessCtx, level_targets::Vector{Float64};
                                           threaded_bins::Bool = false,
                                           tls::Union{Nothing,ThreadLocalBinScratch} = nothing)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_frechet_v2" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_frechet_structured_v2!(evalResult.hess, o, cctx, level_targets; threaded_bins = threaded_bins, tls = tls)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
