# D=20 profiling task (flexible_cm/common_frechet, 2026-07-28): self-include the opt-in
# `@cmhess_prof` sub-block timing macro's defining file if not already loaded -- see the identical
# guard/rationale in cm_hessian_architectures.jl.
isdefined(Main, :CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED) || include(joinpath(@__DIR__, "cm_hessian_subblock_profiling.jl"))

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
    CMFrechetExtension(D, L, nO, NCORE, level_targets)

Harmonization task (2026-07-28): the narrow, genuinely Fréchet-only "CM-F" (common-level anchor)
state -- owns `level_targets` (previously threaded as a bare positional argument through every
Fréchet function) plus the level-block Hessian scratch (`Wtab`/`T1`/`Esum_wb`/`colsum`/
`Hraw_cmlevel`), which `hessian_cm_frechet_structured!`/`_v2!` previously reallocated FRESH on
every single KNITRO Hessian callback (`zeros(D, L+1)`, `zeros(D, L)`, two `Vector{Float64}(undef,
NCORE)`, `Vector{Float64}(undef, nO)`) -- now sized once here and reused, the same persistent-
scratch pattern `CMBinHessCtx` already uses for `Hraw_EC`/`Hraw_CC`/etc. Does NOT own or duplicate
any economic state, CM bins/contrasts, or `CMBinHessCtx` fields -- only what §10 of the
harmonization task explicitly scopes to the extension. Defined here (not in cm_frechet_level.jl,
where `level_targets` itself is computed) because every current include-list ordering loads this
file no later than that one, and the two real equivalence-gate scripts load it strictly earlier.
"""
mutable struct CMFrechetExtension
    level_targets::Vector{Float64}
    Wtab::Matrix{Float64}
    T1::Matrix{Float64}
    Esum_wb::Vector{Float64}
    colsum::Vector{Float64}
    Hraw_cmlevel::Vector{Float64}
end

function CMFrechetExtension(D::Int, L::Int, nO::Int, NCORE::Int, level_targets::Vector{Float64})
    return CMFrechetExtension(level_targets, zeros(D, L + 1), zeros(D, L),
        Vector{Float64}(undef, NCORE), Vector{Float64}(undef, NCORE), Vector{Float64}(undef, nO))
end

"""
    _fill_frechet_level_blocks!(Hfull, cctx, w, H, M, use_winner_bin, wctx, cross_ws, ext::CMFrechetExtension)

Harmonization task (2026-07-28): the genuinely Fréchet-only "CM-F" computation -- the three common-
level anchor blocks H_E,level / H_CM,level / H_level,level -- extracted verbatim from the former
`hessian_cm_frechet_structured!` (this was previously the tail of that function's own separate copy
of the ENTIRE H_EE/H_EC/H_CC computation; everything ABOVE this point is now the one shared
`hessian_cm_structured!`, cm_hessian_architectures.jl, that flexible CM already uses). Called from
`hessian_cm_structured!`/`_v2!` only when `extension !== nothing`.
"""
function _fill_frechet_level_blocks!(Hfull, cctx::CMBinHessCtx, w, H, M, use_winner_bin::Bool, wctx, cross_ws, ext::CMFrechetExtension)
    NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins
    ncm_cm = nO * L
    ncm_level = ncm - ncm_cm
    @assert ncm_level == L "_fill_frechet_level_blocks!: cctx.ncm=$(cctx.ncm) inconsistent with D*L (got ncm_level=$ncm_level, expected L=$L) -- was cctx built from a :common_frechet aug?"
    invsqrtD = 1.0 / sqrt(D)
    level_targets = ext.level_targets
    CS_ = cctx.CScum
    CT = cctx.CT

    # ---- marginal weighted-count table T1[x,l] = sum_s w_s*1{bin(s,x)<=l} (D x L), Wtot, Esum.
    # Needed because (unlike CM's own zero-target raw features) the level feature has a NONZERO
    # target subtracted -- see this file's header docstring for the correction-term derivation.
    # O(W*D) for Wtab/T1 (cheap vs build_bin_tables!'s own O(W*(D*NCORE+D^2))), O(W*NCORE) for Esum
    # (one BLAS gemv). ----
    Bidx = cctx.Bidx
    Wraw = size(Bidx, 1)
    # Wtab/T1 are persistent (ext), reused across calls -- Wtab is an accumulator (`+=`) so it must
    # be explicitly zeroed each call; T1/Esum_wb/colsum/Hraw_cmlevel below are all fully overwritten
    # per call (direct assignment or a from-scratch BLAS/loop fill), so no reset is needed for those.
    Wtab = ext.Wtab
    local T1, Wtot
    @cmhess_prof "level_table_prep" begin
        fill!(Wtab, 0.0)
        @inbounds for s in 1:Wraw
            ws = w[s]
            for x in 1:D
                Wtab[x, Bidx[s, x]] += ws
            end
        end
        T1 = ext.T1
        @inbounds for x in 1:D
            acc = 0.0
            for l in 1:L
                acc += Wtab[x, l]
                T1[x, l] = acc
            end
        end
        Wtot = sum(w)
    end

    # ---- H_E,level (core x level), O(D*NCORE*L) + O(NCORE*L) correction ----
    # Winner-aware H_ER phase (2026-07-27), Section 3 Part B: only THIS block ever reads
    # sum_o CS_[o,j,l] / Esum[j] -- H_CM,level and H_level,level below use CT/T1/Wtot alone, never
    # E, and are UNCHANGED. Under :winner_bin, `winner_pair_cross_hessian_colsum!`/`_esum!`
    # (winner_pair_cross_hessian.jl) replace both dense reads with O(D)/O(1)-per-entry lookups from
    # the SAME cumulative tables the H_EC block already built via `winner_pair_cross_hessian_fill!`
    # -- no dense `E`/`obj.H` read at all in the fast path.
    level_off = NCORE + ncm_cm   # level columns are level_off+1 : level_off+L
    @cmhess_prof "H_EF" if use_winner_bin
        Esum_wb = ext.Esum_wb
        winner_pair_cross_hessian_esum!(Esum_wb, wctx, cross_ws, w, Wtot)
        colsum = ext.colsum
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
        Esum = ext.Esum_wb
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

    # ---- H_CM,level (CM x level), O(D^2*L^2) worst case (same order as H_CC's own loop) ----
    Hraw_cmlevel = ext.Hraw_cmlevel
    @cmhess_prof "H_CF" @inbounds for l in 1:L
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

    # ---- H_level,level (level x level), O(D^2*L^2) + O(D*L^2) correction ----
    invD = 1.0 / D
    @cmhess_prof "H_FF" @inbounds for l in 1:L
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
    return Hfull
end

"""
    _resolve_frechet_ext!(cctx::CMBinHessCtx, level_targets::Vector{Float64}) -> CMFrechetExtension

Harmonization task (2026-07-28): lazily builds and caches a `CMFrechetExtension` on
`cctx.frechet_ext_cache` (the same "typed Any cache field, built once, reused every call" pattern
`core_ws`/`cmlookup_st` already use on this struct), keeping every EXISTING external caller of
`archC_frechet_hess_cb_builder` (which historically passed a bare `level_targets::Vector{Float64}`)
unchanged, while still giving `hessian_cm_frechet_structured!`/`_v2!` a persistent, concretely-typed
extension object instead of reallocating scratch on every KNITRO Hessian callback. `level_targets`
identity is not re-checked on the fast path (this cctx is built once per campaign for one fixed
`level_targets` vector, exactly like every other "fixed once `cctx` is built" field on this struct).
"""
function _resolve_frechet_ext!(cctx::CMBinHessCtx, level_targets::Vector{Float64})
    cached = cctx.frechet_ext_cache
    cached isa CMFrechetExtension && return cached
    ext = CMFrechetExtension(cctx.D, cctx.L, cctx.nO, cctx.NCORE, level_targets)
    cctx.frechet_ext_cache = ext
    return ext
end

"""
    archC_frechet_hess_cb_builder(cctx::CMBinHessCtx, level_targets::Vector{Float64})

KNITRO callback builder for the common-Fréchet level-block Hessian, mirroring
`archC_hess_cb_builder`'s dispatch exactly (`cm_hessian_architectures.jl`): dispatches to the
threaded bin-table variant (`hessian_cm_frechet_structured_v2!`, `cm_frechet_hessian_threaded.jl`)
when `cctx.use_threaded_bins` is true (the production default, set by `build_cm_bin_ctx` --
validated to agree with the serial path to ~1e-14 and measured 4.6x faster at a real D=20/
W=80,000/L=50 point, `test_cm_frechet_threaded_hessian_gates.jl`), falling back to the original
serial `hessian_cm_frechet_structured!` unchanged when `cctx.use_threaded_bins` is false. Closes
the disclosed gap in the prior session's own verdict ("the threaded bin-table variant is NOT
extended for the level block yet").

Harmonization task (2026-07-28): resolves (and caches, see `_resolve_frechet_ext!`) a
`CMFrechetExtension` from `level_targets` -- the public signature is unchanged so every existing
caller (production, tests, diagnostics) needs no update.
"""
function archC_frechet_hess_cb_builder(cctx::CMBinHessCtx, level_targets::Vector{Float64})
    frechet_ext = _resolve_frechet_ext!(cctx, level_targets)
    if cctx.use_threaded_bins
        return (kc, cb, evalRequest, evalResult, userParams) -> begin
            o = userParams
            xloc = evalRequest.x
            @prof "inner_dual_hessian_callback_archC_frechet" begin
                _prep_dual_index_for_archC!(cctx, o, xloc)
                hessian_cm_structured_v2!(evalResult.hess, o, cctx, frechet_ext; threaded_bins = true, tls = cctx.tls)
            end
            _INNER_CALL_COUNTERS[].n_hess_calls += 1
            return 0
        end
    end
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC_frechet" begin
            # True no-H operator bundle (2026-07-28 continuation): was hardcoded to the DENSE-only
            # `_archC_prep_for_hessian!` (unconditional `obj.H` read) -- unlike this same function's
            # own threaded branch above (and flexible-CM's `archC_hess_cb_builder`, both branches),
            # which already call the dense-G-free dispatcher `_prep_dual_index_for_archC!`. This was
            # a real, pre-existing asymmetry between the threaded/serial branches, not something
            # introduced by the no-H bundle -- it just happened to be harmless before (redundantly
            # re-reading a real `obj.H` that already existed) and only became a hard failure once a
            # bundle with no `H` field at all was constructed. Fixed to match the threaded branch.
            _prep_dual_index_for_archC!(cctx, o, xloc)
            hessian_cm_structured!(evalResult.hess, o, cctx, frechet_ext)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end

"""
    hessian_cm_frechet_structured!(h, obj, cctx::CMBinHessCtx, level_targets::Vector{Float64})

Harmonization task (2026-07-28): thin backward-compatibility wrapper. The real implementation is
now the shared `hessian_cm_structured!` (cm_hessian_architectures.jl) with a resolved
`CMFrechetExtension` -- kept here (not deleted) because several pre-existing, still-referenced
diagnostic/gate scripts (`test_cm_frechet_threaded_hessian_gates.jl`,
`test_frechet_winner_bin_her_wiring_d4.jl`/`_d20.jl`, `test_frechet_d20_gates.jl`/`_L50.jl`,
`test_frechet_hessian_structured_vs_dense_d4.jl`, `diag_frechet_hardpoint_2026-07-27.jl`) call this
exact name with a bare `level_targets::Vector{Float64}`, not through `archC_frechet_hess_cb_builder`.
"""
function hessian_cm_frechet_structured!(h, obj, cctx::CMBinHessCtx, level_targets::Vector{Float64})
    return hessian_cm_structured!(h, obj, cctx, _resolve_frechet_ext!(cctx, level_targets))
end
