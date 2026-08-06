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
    # production Hessian allocation audit (2026-08-02): persistent output buffer for the
    # `cctx.R' * Hraw_cmlevel` product in `_fill_frechet_level_blocks!`'s H_CM,level loop (below),
    # mirroring `CMBinHessCtx.block_ec`'s identical persistent-buffer-for-an-R-congruence-product
    # pattern (cm_hessian_architectures.jl) -- that block's own header comment documents the exact
    # same fix already applied there for H_EC ("Hraw_EC * cctx.R allocated FRESH on every one of
    # the L=50 threshold-block iterations"); this field/loop had not received the same fix. Sized
    # nO (matches Hraw_cmlevel's own length); `nothing` when R===nothing (that branch returns
    # Hraw_cmlevel directly, no product needed, mirroring block_ec's own Union{Nothing,...} use).
    block_cmlevel::Union{Nothing,Vector{Float64}}
    # 2026-08-05 (paired-basis-preconditioning pilot, common-Fréchet two-family extension):
    # `fam2`-only scratch for the NEW H_E,levelpow / H_CM(cdf),levelpow / H_CM(pow),level /
    # H_level,levelpow / H_levelpow,levelpow blocks -- see `_fill_frechet_level_blocks!`'s own
    # updated docstring for the full derivation. `Wtab_pow`/`T1_pow` are the Pow-weighted,
    # REFLECTED (`1{bin>l}`, not `<=l`) analogs of `Wtab`/`T1` (needed because levelpow, like CM's
    # own eq.36, uses the reflected `z<Z_l ⟺ U>c` indicator -- see `theoretical_u_threshold`'s own
    # docstring for why the plain level block gets away with unreflected `<=` and levelpow does
    # not). `nothing` for every single-family context (`fam2=false`), matching `CMBinHessCtx`'s
    # own `Union{Nothing,...}` convention for POW-only fields -- zero extra allocation otherwise.
    Wtab_pow::Union{Nothing,Matrix{Float64}}
    T1_pow::Union{Nothing,Matrix{Float64}}
    Hraw_cmlevelpow::Union{Nothing,Vector{Float64}}     # H_CM(cdf),levelpow raw column (nO)
    block_cmlevelpow::Union{Nothing,Vector{Float64}}
    Hraw_cmpowlevel::Union{Nothing,Vector{Float64}}     # H_CM(pow),level AND H_CM(pow),levelpow raw column (nO) -- reused sequentially within one (l,lp) iteration, see _fill_frechet_level_blocks!
    block_cmpowlevel::Union{Nothing,Vector{Float64}}
    colsum_pow::Union{Nothing,Vector{Float64}}          # H_E,levelpow winner-bin scratch (NCORE+1), mirrors `colsum` above
end

function CMFrechetExtension(D::Int, L::Int, nO::Int, NCORE::Int, level_targets::Vector{Float64};
        R::Union{Nothing,AbstractMatrix{Float64}} = nothing, fam2::Bool = false)
    return CMFrechetExtension(level_targets, zeros(D, L + 1), zeros(D, L),
        Vector{Float64}(undef, NCORE), Vector{Float64}(undef, NCORE), Vector{Float64}(undef, nO),
        R === nothing ? nothing : Vector{Float64}(undef, nO),
        fam2 ? zeros(D, L + 1) : nothing, fam2 ? zeros(D, L) : nothing,
        fam2 ? Vector{Float64}(undef, nO) : nothing,
        fam2 ? (R === nothing ? nothing : Vector{Float64}(undef, nO)) : nothing,
        fam2 ? Vector{Float64}(undef, nO) : nothing,
        fam2 ? (R === nothing ? nothing : Vector{Float64}(undef, nO)) : nothing,
        fam2 ? Vector{Float64}(undef, NCORE) : nothing)
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
    fam2 = cctx.n_families == 2
    ncm_cdf = nO * L
    ncm_level = fam2 ? 2L : L
    ncm_cm = ncm - ncm_level
    @assert ncm_cm == (fam2 ? 2ncm_cdf : ncm_cdf) "_fill_frechet_level_blocks!: cctx.ncm=$(cctx.ncm) inconsistent with D*L/n_families (got ncm_cm=$ncm_cm, ncm_cdf=$ncm_cdf, fam2=$fam2) -- was cctx built from a :common_frechet aug?"
    invsqrtD = 1.0 / sqrt(D)
    invD = 1.0 / D
    # 2026-08-05 (common-Fréchet two-family extension): `ext.level_targets` is the FULL, CONCATENATED
    # `[level_cdf_targets (L) | level_pow_targets (L)]` vector when fam2 (matches
    # `build_cm_frechet_level_augmented_obj`'s own `LEVEL = hcat(LEVEL_cdf, LEVEL_pow)` column
    # layout) -- sliced here rather than threaded as a separate constructor argument, since
    # `_resolve_frechet_ext!` caches ONE `CMFrechetExtension` per cctx lifetime and this slice is
    # O(L), negligible next to this function's own O(D^2*L^2) blocks.
    level_targets = fam2 ? (@view ext.level_targets[1:L]) : ext.level_targets
    level_targets_pow = fam2 ? (@view ext.level_targets[L+1:2L]) : nothing
    CS_ = cctx.CScum
    CT = cctx.CT
    # 2026-08-05 (common-Fréchet two-family extension): eq.36's own reflected (`1{U>c}`, not `<=c`)
    # bilinear tables -- direct reuse of `fill_cm_HCC!`'s own `CT12_use`/`CT22_use` construction
    # (cm_hessian_architectures.jl), recomputed here rather than shared/cached since this function
    # runs once per Hessian callback exactly like that one does. `CT12_use[x,y,l,lp]` = CDF-side
    # (index x, `<=l`) x POW-side (index y, `>lp` reflected); `CT22_use[x,y,l,lp]` = both sides POW
    # (`>l`,`>lp` reflected).
    CT12_use = fam2 ? _build_reflected_bilinear(cctx.Ttab12, cctx.CT12, D, L; reflect_x = false, reflect_y = true) : nothing
    CT22_use = fam2 ? _build_reflected_bilinear(cctx.Ttab22, cctx.CT22, D, L; reflect_x = true, reflect_y = true) : nothing

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

    # ---- 2026-08-05 (common-Fréchet two-family extension): T1_pow[x,l] = sum_s w_s*Pow[s,x]*
    # 1{bin(s,x)>l} (D x L), the REFLECTED (`>l`, not `<=l`) Pow-weighted twin of T1 above -- needed
    # because levelpow's own raw feature uses the SAME `1{U>c}` reflected convention as CM's own
    # eq.36 sub-block (see fill_cm_HCC!'s own BUG FIX comment, cm_hessian_architectures.jl, for the
    # derivation of why eq.36 needs reflection and eq.35/plain-level does not). Built the same way
    # `Wtab`/`T1` are (one extra O(W*D) pass, Pow-weighted; grand-total-minus-cumsum for the
    # reflection, mirroring `winner_pair_cross_hessian_cm_block!`'s own `*Total_pow .- *CScum_pow`
    # idiom). `nothing` when !fam2 (single-family common-Fréchet is completely unaffected, zero
    # extra allocation or work). ----
    local T1_pow
    if fam2
        @cmhess_prof "levelpow_table_prep" begin
            Pow = cctx.Pow
            Wtab_pow = ext.Wtab_pow
            fill!(Wtab_pow, 0.0)
            @inbounds for s in 1:Wraw
                ws = w[s]
                for x in 1:D
                    Wtab_pow[x, Bidx[s, x]] += ws * Pow[s, x]
                end
            end
            T1_pow = ext.T1_pow
            @inbounds for x in 1:D
                total = 0.0
                for k in 1:(L+1)
                    total += Wtab_pow[x, k]
                end
                acc = 0.0
                for l in 1:L
                    acc += Wtab_pow[x, l]
                    T1_pow[x, l] = total - acc
                end
            end
        end
    else
        T1_pow = nothing
    end

    # ---- H_E,level (core x level), O(D*NCORE*L) + O(NCORE*L) correction ----
    # Winner-aware H_ER phase (2026-07-27), Section 3 Part B: only THIS block ever reads
    # sum_o CS_[o,j,l] / Esum[j] -- H_CM,level and H_level,level below use CT/T1/Wtot alone, never
    # E, and are UNCHANGED. Under :winner_bin, `winner_pair_cross_hessian_colsum!`/`_esum!`
    # (winner_pair_cross_hessian.jl) replace both dense reads with O(D)/O(1)-per-entry lookups from
    # the SAME cumulative tables the H_EC block already built via `winner_pair_cross_hessian_fill!`
    # -- no dense `E`/`obj.H` read at all in the fast path.
    level_off = NCORE + ncm_cm   # level_cdf columns level_off+1:level_off+L; level_pow (fam2) level_off+L+1:level_off+2L
    level_pow_off = level_off + L
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
        # ---- H_E,levelpow (2026-08-05 two-family extension): same shape as H_E,level above, reading
        # the REFLECTED "_pow" winner-bin tables instead -- see winner_pair_cross_hessian_colsum_pow!'s
        # own docstring. ----
        if fam2
            colsum_pow = ext.colsum_pow
            @inbounds for l in 1:L
                tl_pow = level_targets_pow[l]
                winner_pair_cross_hessian_colsum_pow!(colsum_pow, wctx, cross_ws, l)
                for j in 1:NCORE
                    v = invsqrtD * colsum_pow[j] / M - tl_pow * Esum_wb[j] / M
                    Hfull[j, level_pow_off + l] = v
                    Hfull[level_pow_off + l, j] = v
                end
            end
        end
    else
        # No-moments/no-composite-G task (2026-07-28): `E` constructed lazily, only here.
        record_dense_frechet_g!()
        # moment_representation threading task (2026-07-29): matches _fill_cm_HEE!'s identical
        # guards (cm_hessian_architectures.jl) -- this dense H_E,level fallback is provably
        # unreachable in production (use_winner_bin should always hold at defaults), but was
        # previously unguarded, so it would throw a confusing `MethodError: view(::Nothing, ...)`
        # rather than a clear error for an operator-mode bundle (H is nothing).
        H === nothing && error("cm_frechet_hessian.jl H_E,level fill: reached the dense E=@view(H[...]) fallback for an operator-mode bundle with no H field -- this should be provably unreachable in production (use_winner_bin should always hold); indicates a real configuration bug, not expected behavior.")
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
        # ---- H_E,levelpow, dense fallback (2026-08-05 two-family extension) -- mirrors
        # hessian_cm_structured!'s own dense H_EC(pow) computation (`S2total .- CS2`, the REFLECTED
        # correction), reusing the SAME `Esum` (E has no target, independent of family). Requires
        # `cctx.Stab2`/`CScum2` to have been filled (`fill_S=!use_winner_bin`, i.e. exactly this
        # branch's own precondition -- see `hessian_cm_structured!`'s `build_bin_tables!` call). ----
        if fam2
            S2total = dropdims(sum(cctx.Stab2, dims = 3), dims = 3)   # D x NCORE
            CS2 = cctx.CScum2
            @inbounds for l in 1:L
                tl_pow = level_targets_pow[l]
                for j in 1:NCORE
                    acc = 0.0
                    for o in 1:D
                        acc += S2total[o, j] - CS2[o, j, l]
                    end
                    v = invsqrtD * acc / M - tl_pow * Esum[j] / M
                    Hfull[j, level_pow_off + l] = v
                    Hfull[level_pow_off + l, j] = v
                end
            end
        end
    end

    # ---- H_CM,level (CM x level), O(D^2*L^2) worst case (same order as H_CC's own loop). fam2
    # additionally folds in H_CM(pow),level [user's explicitly-requested "item 1"] and
    # H_CM(cdf),levelpow / H_CM(pow),levelpow [item 2] into the SAME (l,lp) double loop, reusing the
    # SAME CT12_use table already built above for all three -- no extra O(D^2*L^2) passes. ----
    Hraw_cmlevel = ext.Hraw_cmlevel
    Hraw_cmpowlevel = fam2 ? ext.Hraw_cmpowlevel : nothing
    Hraw_cmlevelpow = fam2 ? ext.Hraw_cmlevelpow : nothing
    ncm_pow_off = NCORE + ncm_cdf   # CM-pow rows for threshold l: ncm_pow_off+(l-1)*nO+1 : ncm_pow_off+l*nO
    @cmhess_prof "H_CF" @inbounds for l in 1:L
        for lp in 1:L
            tlp = level_targets[lp]
            # 2026-08-06 (D20 real-data gate, BUG FIX): levelpow's OWN target at lp -- was missing,
            # causing H_CM(cdf),levelpow/H_CM(pow),levelpow below to reuse the WRONG target (either
            # `tlp`, the level-CDF family's target, or `tl_pow=level_targets_pow[l]`, the right
            # family but the wrong index) for their own correction term. Caught only at D20/W=80,000
            # real-data scale (max|diff|~5.7e-6) -- the D4 gate's own small/degenerate target values
            # happened to mask this for both blocks (test_frechet_hessian_structured_vs_dense_d4_
            # twofamily_2026-08-05.jl still reports PASS at 1e-15 for both, i.e. does not by itself
            # certify this fix -- the real evidence is the D20 gate). See _fill_frechet_level_blocks!'s
            # own module-level derivation (top of file) for why the correction term is always
            # `target_B * marginal_A`, B being whichever family the OTHER (non-CM) index belongs to.
            tlp_pow = fam2 ? level_targets_pow[lp] : 0.0
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
            # production Hessian allocation audit (2026-08-02): was `cctx.R' * Hraw_cmlevel` --
            # allocated a fresh length-nO vector on EVERY one of this loop's L*L=2500 iterations
            # (L=50), the dominant recurring allocation for common_frechet (500,000 of ~562,000
            # bytes/callback measured via Profile.Allocs at W=20,000, i.e. ~89% of the family's
            # total and ~96% of its excess over flexible_cm at similar dual dimension). Mirrors
            # `CMBinHessCtx.block_ec`'s identical already-fixed pattern for H_EC's own R-congruence
            # product (cm_hessian_architectures.jl) -- that fix's own header comment describes
            # this EXACT same "allocated FRESH on every threshold-block iteration" defect for a
            # sibling block; this level-block loop had not received the analogous fix.
            block_cmlevel = cctx.R === nothing ? Hraw_cmlevel : mul!(ext.block_cmlevel, cctx.R', Hraw_cmlevel)
            @views Hfull[cm_rows, col] .= block_cmlevel
            @views Hfull[col, cm_rows] .= block_cmlevel

            if fam2
                # H_CM(pow),level[(o,l),lp] (item 1): CM-pow's own origin o at its own reflected
                # threshold l, x level-cdf's implicit sum-over-origins at threshold lp -- CT12_use's
                # CDF-side index (`<=a`) is the SUM-over-origins side here (level has no single
                # origin), so `a=lp`; its POW-side index (`>b` reflected) is CM-pow's own origin,
                # `b=l`. NOTE the (lp,l) argument order -- swapped relative to CT12_use's own (l,lp)
                # loop-variable naming, since here CM-pow (not CM-cdf) supplies the POW-side index.
                tlp_level = level_targets[lp]
                for (oi, o) in enumerate(origins)
                    acc_o = 0.0
                    acc_ref = 0.0
                    for x in 1:D
                        acc_o += CT12_use[x, o, lp, l]
                        acc_ref += CT12_use[x, refIndex1, lp, l]
                    end
                    Hraw_cmpowlevel[oi] = invsqrtD * (acc_o - acc_ref) / M - tlp_level * (T1_pow[o, l] - T1_pow[refIndex1, l]) / M
                end
                cm_rows_pow = ncm_pow_off + (l-1)*nO + 1 : ncm_pow_off + l*nO
                block_cmpowlevel = cctx.R === nothing ? Hraw_cmpowlevel : mul!(ext.block_cmpowlevel, cctx.R', Hraw_cmpowlevel)
                @views Hfull[cm_rows_pow, col] .= block_cmpowlevel
                @views Hfull[col, cm_rows_pow] .= block_cmpowlevel

                # H_CM(cdf),levelpow[(o,l),lp] (item 2, CM interaction): CM-cdf's own origin o at
                # its own threshold l (CDF-side, `<=l`), x levelpow's implicit sum-over-origins at
                # threshold lp (POW-side, `>lp` reflected) -- direct (l,lp) index order, matching
                # CT12_use's own naming exactly.
                col_pow = level_pow_off + lp
                for (oi, o) in enumerate(origins)
                    acc_o = 0.0
                    acc_ref = 0.0
                    for y in 1:D
                        acc_o += CT12_use[o, y, l, lp]
                        acc_ref += CT12_use[refIndex1, y, l, lp]
                    end
                    Hraw_cmlevelpow[oi] = invsqrtD * (acc_o - acc_ref) / M - tlp_pow * (T1[o, l] - T1[refIndex1, l]) / M
                end
                block_cmlevelpow = cctx.R === nothing ? Hraw_cmlevelpow : mul!(ext.block_cmlevelpow, cctx.R', Hraw_cmlevelpow)
                @views Hfull[cm_rows, col_pow] .= block_cmlevelpow
                @views Hfull[col_pow, cm_rows] .= block_cmlevelpow

                # H_CM(pow),levelpow[(o,l),lp]: CM-pow's own origin o at threshold l (POW-side,
                # `>l` reflected) x levelpow's implicit sum-over-origins at threshold lp (POW-side,
                # `>lp` reflected) -- both-POW, so reads CT22_use (not CT12_use), (lp,l) order for
                # the same reason as H_CM(pow),level above (level-pow's sum-over-origins is the
                # FIRST/CDF-slot argument of CT22_use's own (x,y,a,b) convention, CM-pow's o is the
                # second/POW-slot argument).
                for (oi, o) in enumerate(origins)
                    acc_o = 0.0
                    acc_ref = 0.0
                    for x in 1:D
                        acc_o += CT22_use[x, o, lp, l]
                        acc_ref += CT22_use[x, refIndex1, lp, l]
                    end
                    Hraw_cmpowlevel[oi] = invsqrtD * (acc_o - acc_ref) / M - tlp_pow * (T1_pow[o, l] - T1_pow[refIndex1, l]) / M
                end
                block_cmpowlevelpow = cctx.R === nothing ? Hraw_cmpowlevel : mul!(ext.block_cmpowlevel, cctx.R', Hraw_cmpowlevel)
                @views Hfull[cm_rows_pow, col_pow] .= block_cmpowlevelpow
                @views Hfull[col_pow, cm_rows_pow] .= block_cmpowlevelpow
            end
        end
    end

    # ---- H_level,level (level x level), O(D^2*L^2) + O(D*L^2) correction. fam2 additionally folds
    # in H_level,levelpow and H_levelpow,levelpow into the SAME (l,lp) loop. ----
    @cmhess_prof "H_FF" @inbounds for l in 1:L
        tl = level_targets[l]
        sum_T1_l = sum(@view T1[:, l])
        sum_T1pow_l = fam2 ? sum(@view T1_pow[:, l]) : 0.0
        for lp in 1:L
            tlp = level_targets[lp]
            acc = 0.0
            for o in 1:D, p in 1:D
                acc += CT[o, p, l, lp]
            end
            sum_T1_lp = sum(@view T1[:, lp])
            Hfull[level_off + l, level_off + lp] =
                invD * acc / M - tlp * invsqrtD * sum_T1_l / M - tl * invsqrtD * sum_T1_lp / M + tl * tlp * Wtot / M

            if fam2
                tlp_pow = level_targets_pow[lp]
                tl_pow = level_targets_pow[l]
                sum_T1pow_lp = sum(@view T1_pow[:, lp])

                # H_level,levelpow[l,lp]: level-cdf (CDF-side, `<=l`) x levelpow (POW-side, `>lp`
                # reflected) -- direct (l,lp), matching CT12_use's own naming.
                acc12 = 0.0
                for x in 1:D, y in 1:D
                    acc12 += CT12_use[x, y, l, lp]
                end
                Hfull[level_off + l, level_pow_off + lp] =
                    invD * acc12 / M - tlp_pow * invsqrtD * sum_T1_l / M - tl * invsqrtD * sum_T1pow_lp / M + tl * tlp_pow * Wtot / M
                Hfull[level_pow_off + lp, level_off + l] = Hfull[level_off + l, level_pow_off + lp]

                # H_levelpow,levelpow[l,lp]: both POW-side (`>l`,`>lp` reflected) -- CT22_use.
                acc22 = 0.0
                for x in 1:D, y in 1:D
                    acc22 += CT22_use[x, y, l, lp]
                end
                Hfull[level_pow_off + l, level_pow_off + lp] =
                    invD * acc22 / M - tlp_pow * invsqrtD * sum_T1pow_l / M - tl_pow * invsqrtD * sum_T1pow_lp / M + tl_pow * tlp_pow * Wtot / M
            end
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
    ext = CMFrechetExtension(cctx.D, cctx.L, cctx.nO, cctx.NCORE, level_targets; R = cctx.R, fam2 = cctx.n_families == 2)
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
