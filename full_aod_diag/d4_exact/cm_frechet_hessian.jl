# ================================================================================================
# Fixed Fréchet as flexible CM plus a common-level anchor -- Part III (Architecture-C Hessian).
#
# See docs/COMMON_FRECHET_HESSIAN_ARCHITECTURE_2026-07-25.md for the full derivation. Summary: the
# existing CMBinHessCtx (cm_hessian_architectures.jl) already builds its weighted bin-contingency
# tables (`Ttab`/`CT`: D x D x L x L, prefix-summed; `Stab`/`CScum`: D x NCORE x L, prefix-summed)
# over ALL D origins -- needed for CM's own (o,ref)-DIFFERENCE formula, which reads
# `CT[o,p,l,l'] - CT[o,ref,l,l'] - CT[ref,p,l,l'] + CT[ref,ref,l,l']`. The level restriction (a SUM
# over all D origins with weight u=1/sqrt(D), not a difference) needs exactly the same tables, just
# a different linear combination:
#   H_E,level[j,l]        = (1/sqrt(D)) * (1/M) * sum_{o=1}^D CScum[o,j,l]
#   H_CM,level[(o,l),l']  = (1/sqrt(D)) * (1/M) * [sum_p CT[o,p,l,l'] - sum_p CT[ref,p,l,l']]
#   H_level,level[l,l']   = (1/(D*M))   * sum_{o,p=1}^D CT[o,p,l,l']
# (derivation: for any two origin-coefficient vectors alpha,beta in R^D at thresholds l,l', the
# cross Hessian term is (1/M)*sum_{o,p} alpha_o*beta_p*CT[o,p,l,l'] -- CM's alpha/beta are e_o-e_ref
# (or (e_o-e_ref) rows of R for :orthonormal), level's is always u=ones(D)/sqrt(D), giving the sums
# above directly). NO new O(W) pass -- build_bin_tables!/prefix_sum_tables! (cm_hessian_architectures.jl)
# are called UNCHANGED and reused verbatim; only the small O(D*NCORE*L + D^2*L^2)-scale assembly
# step gains the three new blocks above. H_EE (winner-pair backend) is completely unaffected --
# `_fill_cm_HEE!` is reused UNCHANGED, since H_EE depends only on the core/economic columns.
#
# :orthonormal contrasts: the level side of every cross block is UNROTATED (level is always the
# single direction u, never mixed with R); only the CM side rotates, exactly mirroring how
# `hessian_cm_structured!`'s own H_EC/H_CC blocks rotate (right-multiply by R for a "column" CM
# index, left-multiply by R' for a "row" CM index).
# ================================================================================================

"""
    hessian_cm_frechet_structured!(h, obj, cctx::CMBinHessCtx, level_targets::Vector{Float64})

Architecture-C Hessian callback for `marginal_restriction=:common_frechet`. Requires `cctx` to have
been built (via the UNCHANGED `build_cm_bin_ctx`) from an `aug` with `aug.ncm = D*L` (i.e. from
`build_cm_frechet_level_augmented_obj`/`build_cm_frechet_production_context`, NOT plain
`build_cm_augmented_obj`) -- `cctx.ncm` sizes `Hfull` and the packed output, and the column layout
this function writes matches `wrap_moments_with_cm_frechet_archB`'s own
`[core | CM ((D-1)*L) | level (L) | gravity]` layout exactly (CM block first, level block last, both
INSIDE the `NCORE+1 : NCORE+ncm` moment range). `level_targets` (`aug.level_targets`,
`sqrt(D)*p_l`) is required because -- UNLIKE the CM block, whose raw features already have zero
target baked in by construction (`f_o - f_ref`) -- the level feature has a NONZERO target
(`level_l(omega) = u'f_l(omega) - target_l`), and an additive per-draw-CONSTANT shift in a moment
column DOES change the `E'diag(w)E`-type quadratic Hessian form (unlike the FG/gradient side, which
only sees the target through a harmless constant shift of `arg0`). See
`docs/COMMON_FRECHET_HESSIAN_ARCHITECTURE_2026-07-25.md` for the full correction-term derivation.

Same precondition as `hessian_cm_structured!`: `obj.arg0` must already reflect the current
(zeta,lambda) (`_archC_prep_for_hessian!` first).
"""
function hessian_cm_frechet_structured!(h, obj, cctx::CMBinHessCtx, level_targets::Vector{Float64})
    @unpack H, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins
    ncm_cm = nO * L
    ncm_level = ncm - ncm_cm
    @assert ncm_level == L "hessian_cm_frechet_structured!: cctx.ncm=$(cctx.ncm) inconsistent with D*L (got ncm_level=$ncm_level, expected L=$L) -- was cctx built from a :common_frechet aug?"
    invsqrtD = 1.0 / sqrt(D)

    E = @view H[:, 2:1+NCORE]
    build_bin_tables!(cctx, E, w)     # UNCHANGED (cm_hessian_architectures.jl) -- covers all D origins already
    prefix_sum_tables!(cctx)          # UNCHANGED

    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    _fill_cm_HEE!(HEE, w, obj, cctx, E, M)   # UNCHANGED -- winner-pair backend, unaffected by level block

    CS_ = cctx.CScum
    CT = cctx.CT

    # ---- H_EC (core x CM), H_CC (CM x CM): IDENTICAL to hessian_cm_structured! ----
    Hraw_EC = cctx.Hraw_EC
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            for j in 1:NCORE
                Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
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

    Hraw_CC = cctx.Hraw_CC
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, p) in enumerate(origins)
                Hraw_CC[oi, pi] = (CT[o, p, l, lp] - CT[o, refIndex1, l, lp] - CT[refIndex1, p, l, lp] + CT[refIndex1, refIndex1, l, lp]) / M
            end
            rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
            block = if cctx.R === nothing
                Hraw_CC
            else
                mul!(cctx.RtHraw_CC, cctx.R', Hraw_CC)
                mul!(cctx.block_cc, cctx.RtHraw_CC, cctx.R)
            end
            @views Hfull[rows, cols] .= block
        end
    end

    # ---- NEW: marginal weighted-count table T1[x,l] = sum_s w_s*1{bin(s,x)<=l} (D x L), Wtot, Esum.
    # Needed because (unlike CM's own zero-target raw features) the level feature has a NONZERO
    # target subtracted -- see this function's docstring for the correction-term derivation.
    # O(W*D) for Wtab/T1 (cheap vs build_bin_tables!'s own O(W*(D*NCORE+D^2))), O(W*NCORE) for Esum
    # (one BLAS gemv). ----
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
    Esum = Vector{Float64}(undef, NCORE)
    mul!(Esum, E', w)

    # ---- NEW: H_E,level (core x level), O(D*NCORE*L) + O(NCORE*L) correction ----
    level_off = NCORE + ncm_cm   # level columns are level_off+1 : level_off+L
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

    # ---- NEW: H_CM,level (CM x level), O(D^2*L^2) worst case (same order as H_CC's own loop) ----
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

    # ---- NEW: H_level,level (level x level), O(D^2*L^2) + O(D*L^2) correction ----
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

    # symmetrize defensively (analytically symmetric; absorbs FP-order noise, same pattern as
    # hessian_cm_structured!'s own final step)
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

"""
    archC_frechet_hess_cb_builder(cctx::CMBinHessCtx, level_targets::Vector{Float64})

Serial-only KNITRO callback builder for `hessian_cm_frechet_structured!`, mirroring
`archC_hess_cb_builder`'s non-threaded branch exactly (`cm_hessian_architectures.jl`). The threaded
bin-table variant (`hessian_cm_structured_v2!`) is NOT extended for the level block yet -- disclosed
as a follow-up, not silently degraded: `cctx.use_threaded_bins` is simply ignored by this builder
(always serial), which is correct-but-slower rather than wrong.
"""
function archC_frechet_hess_cb_builder(cctx::CMBinHessCtx, level_targets::Vector{Float64})
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_frechet" begin
            _archC_prep_for_hessian!(o, xloc)
            hessian_cm_frechet_structured!(evalResult.hess, o, cctx, level_targets)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
