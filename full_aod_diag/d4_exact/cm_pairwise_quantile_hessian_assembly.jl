# ================================================================================================
# Packed exact-Hessian assembly + KNITRO callback for the CM + pairwise-quantile family (family #7,
# 2026-08-12). This is the file that makes `hessopt=exact` (`ek_inner_cmpq.opt`) runnable.
#
# WHY A SEPARATE FILE FROM `cm_pairwise_quantile_hessian.jl`. That file holds this family's own
# algebra (the new H_R,CM block, the replicated-mu reuse, the row map, the packing primitive) and is
# loaded by the STANDALONE dense oracle, which deliberately loads no production stack at all. This
# file is where the production stack enters: `WinnerPairHessCtx`, `CMBinHessCtx`,
# `WinnerBinCrossScratch`, `OperatorPsiBundle`. Keeping the split means the oracle can keep gating
# the algebra without dragging in `compressed_moments.jl`'s whole chain.
#
# THE SIX BLOCKS, AND WHERE EACH COMES FROM. `r` is LINEAR in the inner variables, so
# `H = (1/W) M' diag(h) M` EXACTLY, `M = [1 | E | G_R | G_CM]`, `h_w = Psi''(r_w)` -- no second-order
# term anywhere. FIVE of the six blocks are therefore supplied by already-gated machinery and this
# file only sequences and places them:
#
#   H_EE     `winner_pair_hessian!` + `WinnerPairHessCtx`             (shared winner-pair backend)
#   H_E,R    `pairwise_quantile_cross_hessian_block!` at replicated mu, sub-selected by `sig`
#            (restructured 2026-08-12 for BOTH families at once -- threaded over slot, transposed
#            accumulation, hoisted bins. There is ONE implementation: the pre-restructure serial
#            code was deleted rather than kept as a switchable reference, so it cannot be turned
#            back on by accident. It is in git if it is ever needed. Correctness is gated by the
#            exact Gram reference in test_cm_pairwise_quantile_real_d4_hessian.jl, which proves the
#            block CORRECT rather than merely unchanged.)
#   H_RR     PQ raw fill + centering at replicated mu, sub-selected by `sig`
#   H_R,CM   `build_cmpq_cross_hess_tables!` + `fill_cmpq_cm_cross_block!`   <- the only new algebra
#   H_CM,CM  CM's `build_bin_tables!` -> `prefix_sum_tables!` -> `fill_cm_HCC!`
#   H_E,CM   CM's `winner_pair_cross_hessian_fill!` -> `winner_pair_cross_hessian_cm_block!` per `l`
#
# The CM half is reached through a `CMBinHessCtx` built by `build_cm_bin_ctx(ctx, aug)` from a
# SYNTHESIZED `aug` NamedTuple -- the same adapter pattern the profiled-restricted production bridge
# already uses. Two properties of that route are load-bearing and are asserted, not assumed:
#
#   * `build_bin_tables!(cctx, nothing, h; fill_S=false)` is safe with `H === nothing` ONLY on the
#     `fill_S=false` branch (the one that never touches dense economic columns); the function hard-
#     errors otherwise. This family is operator-native and has no dense `H` at all, so `fill_S=false`
#     is not an optimization here, it is the only admissible call.
#   * `fill_cm_HCC!` writes at CM's OWN offsets (`NCORE + (l-1)*nO + 1 : NCORE + l*nO`, plus
#     `+ncm_cdf` for eq.36) and fills BOTH triangles. Rather than redirect those offsets, the
#     `ncm x ncm` corner is READ back out of `cctx.Hfull` -- which is exactly where CM puts it, since
#     `cctx.NCORE` is set to this family's own economic width.
#
# `winner_pair_cross_hessian_cm_block!` is PER THRESHOLD BLOCK `l` and needs a `WinnerBinCrossScratch`
# populated once per callback by `winner_pair_cross_hessian_fill!`. CM+ZC's `use_direct_hcz`/
# `ncore_core` splitting at that same call site does NOT apply here: `ncore_core == NCORE` for this
# family (there is no ZC-widened economic block), so the plain `wctx`-based call fills the whole
# `NCORE x nO` slab.
#
# Requires (already loaded): core_exact_hessian.jl, winner_pair_cross_hessian.jl,
# cm_hessian_architectures.jl, cm_hessian_threaded.jl, operator_hessian_weights.jl,
# pairwise_quantile_hessian.jl, pairwise_quantile_cross_hessian.jl,
# cm_pairwise_quantile_{config,moments,hessian,lookup_kernels}.jl.
# ================================================================================================

using LinearAlgebra: mul!

isdefined(Main, :WinnerPairHessCtx) || include(joinpath(@__DIR__, "core_exact_hessian.jl"))
isdefined(Main, :WinnerBinCrossScratch) || include(joinpath(@__DIR__, "winner_pair_cross_hessian.jl"))
isdefined(Main, :CMBinHessCtx) || include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
isdefined(Main, :build_bin_tables_threaded!) || include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
isdefined(Main, :PairwiseQuantileCrossHessScratch) ||
    include(joinpath(@__DIR__, "pairwise_quantile_cross_hessian.jl"))
isdefined(Main, :cmpq_to_pq_row) || include(joinpath(@__DIR__, "cm_pairwise_quantile_hessian.jl"))

"""
    CMPQCoreHessCtx

Campaign-lifetime Hessian state: every buffer below is sized ONCE and refreshed in value on each
Hessian callback, never reallocated (this codebase's standing "no per-callback vectors, matrices,
closures" discipline).

The two blocks that come from the STANDALONE pairwise-quantile family are stored at that family's
FULL row count `n_total_rows(D,L)`, not at this family's `n_cmpq_restr_rows(D,L)`, and the packed
write sub-selects through `sig` -- no `n_restr x n_restr` copy is ever made. That matters at scale:
the copy alone would be 74 MB written + 78 MB read per callback at D=20/L=5. `extract_cmpq_HRR!`/
`extract_cmpq_HER!` still exist and are still gated (dense oracle, check 10b); the real-context
Hessian gate additionally checks that this direct `sig` read reproduces them entry for entry, so the
two routes cannot drift.
"""
mutable struct CMPQCoreHessCtx
    # ---- shapes ----
    NCORE::Int                 # economic width INCLUDING the zeta-paired "ones" column = ncore_econ
    D::Int
    L::Int                     # PQ bin count
    n_restr::Int
    ncm::Int
    Lcm::Int                   # CM grid LEVELS (= cm_grid_size - 1)
    nO::Int
    n_families::Int
    origins::Vector{Int}
    refIndex1::Int
    R::Union{Nothing,Matrix{Float64}}
    # ---- this family's restriction block ----
    op::PairwiseQuantileOperator
    state::CMPQMassState                       # the shared mu (L-1 free masses)
    pq_state::PairwiseQuantileMassState        # the SAME mu replicated into every origin row
    tabs_pq::PairwiseQuantileHessianTables
    tls::PairwiseQuantileThreadScratch
    tabs_x::CMPQCrossHessTables
    cross_hess_scratch::PairwiseQuantileCrossHessScratch
    # ---- CM's own machinery ----
    cctx::CMBinHessCtx
    Pow::Union{Nothing,Matrix{Float64}}
    threaded_bins::Bool
    # ---- economic ----
    core_cf_ref::Ref{Any}
    core_ws::Union{Nothing,WinnerPairHessCtx}
    core_ws_for::Any
    zc_cross_ws::Base.RefValue{Union{Nothing,WinnerZCCrossScratch}}    # feeds H_E,R
    cm_cross_ws::Base.RefValue{Union{Nothing,WinnerBinCrossScratch}}   # feeds H_E,CM
    fg_state::Any              # the CMPairwiseQuantileOperatorState -- for operator_prep_for_hessian!
    # ---- output blocks ----
    hee_packed::Vector{Float64}
    HEQ_pq::Matrix{Float64}    # NCORE x n_total_rows(D,L)   (standalone-PQ column space)
    HEC::Matrix{Float64}       # NCORE x ncm
    HRR_pq::Matrix{Float64}    # n_total_rows x n_total_rows, LOWER TRIANGLE ONLY
    HRC::Matrix{Float64}       # n_restr x ncm
    sig::Vector{Int}           # this family's row -> standalone-PQ row
    n_hess_calls::Int
end

"""
    build_cmpq_hess_ctx(ctx, cmpq, fg_state) -> CMPQCoreHessCtx

ONCE per campaign. `ctx` is the ORIGINAL, unaugmented economic context (`d4_exact_setup` /
`d20_real_setup`-shaped) -- the same object `build_cm_pairwise_quantile_context` was given, NOT the
attached `ctx_cm`, because `build_cm_bin_ctx` reads `ctx.U`/`ctx.D`/`ctx.sigma`/`ctx.muHat` off it.

Nothing scientific is defaulted or invented here: `L`, `G`, `n_families`, `contrasts`, the CM
thresholds and the PQ cutoffs all arrive already resolved on `cmpq`, and the `aug` NamedTuple below
is a pure re-labelling of those same values into the field names `build_cm_bin_ctx` reads.
"""
function build_cmpq_hess_ctx(ctx, cmpq, fg_state)
    D = ctx.D
    L = cmpq.L
    W = cmpq.op.W
    NCORE = cmpq.ncore_econ
    npair = cmpq.op.npair
    Lcm = cmpq.Lcm
    ncm = cmpq.ncm
    nO = cmpq.nO
    n_restr = cmpq.n_restr
    npq = n_total_rows(D, L)

    # ---- CM's Architecture-C context, via the documented adapter ---------------------------------
    # `aug` carries exactly the fields `build_cm_bin_ctx` reads, with CM's own names: its `L` is the
    # GRID LEVEL count (this family's `L` is the PQ bin count -- the genuine API collision
    # `resolve_cm_pairwise_quantile_config` exists to disambiguate), and `ncore` is the economic
    # width. `core_cf_ref` is the SAME box `prime_operator!` publishes into, shared, not copied.
    aug = (L = Lcm, origins = cmpq.origins, refIndex1 = cmpq.refIndex1, z = cmpq.z_cm,
           ncore = NCORE, ncm = ncm, contrasts = cmpq.contrasts,
           core_cf_ref = cmpq.core_cf_ref, n_families = cmpq.n_families)
    cctx = build_cm_bin_ctx(ctx, aug)
    cctx.ncore_core == cctx.NCORE ||
        error("build_cmpq_hess_ctx: cctx.ncore_core=$(cctx.ncore_core) != NCORE=$(cctx.NCORE) -- " *
              "this family has no ZC-widened economic block, so the CM+ZC use_direct_hcz split must " *
              "not be engaged")
    size(cctx.Bidx) == size(cmpq.Bidx) &&
        all(Int(cctx.Bidx[i]) == Int(cmpq.Bidx[i]) for i in eachindex(cctx.Bidx)) ||
        error("build_cmpq_hess_ctx: CM's own bin indices disagree with this family's -- the two are " *
              "built from the same U and the same z_cm and MUST be identical")
    # `Ews` (W x NCORE) is scratch for `_fill_cm_HEE!`'s DENSE fallbacks only, and this family never
    # takes them: H_EE comes from `winner_pair_hessian!` directly, and `ncore_core == NCORE` means
    # the widened-block branch does not exist either. Released rather than carried -- 321 MB at
    # D=20/W=100,000/NCORE=401, for a buffer nothing on this path reads. If some future path ever
    # does read it, this errors loudly (BoundsError) instead of returning a silently wrong block --
    # and the real-context Hessian gate calls `_fill_cm_HEE!` against this exact shrunken cctx, so
    # the claim is checked live rather than argued.
    cctx.Ews = Matrix{Float64}(undef, 0, 0)

    tabs_x = CMPQCrossHessTables(D, npair, L, Lcm + 1; n_families = cmpq.n_families, nO = nO)
    return CMPQCoreHessCtx(NCORE, D, L, n_restr, ncm, Lcm, nO, cmpq.n_families,
        cmpq.origins, cmpq.refIndex1, cmpq.R,
        cmpq.op, cmpq.mass_state, PairwiseQuantileMassState(D, L),
        PairwiseQuantileHessianTables(cmpq.op),
        build_pairwise_quantile_thread_scratch(D, npair, L), tabs_x,
        PairwiseQuantileCrossHessScratch(D, npair, W, NCORE - 1, L),
        cctx, cmpq.Pow, cctx.use_threaded_bins,
        cmpq.core_cf_ref, nothing, nothing,
        Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing),
        Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing),
        fg_state,
        Vector{Float64}(undef, div(NCORE * (NCORE + 1), 2)),
        zeros(NCORE, npq), zeros(NCORE, ncm), zeros(npq, npq), zeros(n_restr, ncm),
        cmpq_pq_row_map(D, L, cmpq.refIndex1), 0)
end

"""
    cmpq_fill_hessian_blocks!(octx::CMPQCoreHessCtx, obj) -> octx

Refreshes all six blocks at the CURRENT `obj.arg0` (the per-draw dual index `r`). Split out of the
callback so a test can drive it directly, without KNITRO, and compare against a dense reference --
the callback below is then only `prep -> this -> pack`.

PRECONDITION: `obj.arg0` already holds `r` at the point of interest (`operator_prep_for_hessian!`,
or an FG call at that same point). Every consumer below recomputes `h = Psi''(r)` from it.
"""
function cmpq_fill_hessian_blocks!(octx::CMPQCoreHessCtx, obj)
    D = octx.D; L = octx.L; Lcm = octx.Lcm; nO = octx.nO; NCORE = octx.NCORE
    W = octx.op.W
    obj.M == W ||
        error("cmpq_fill_hessian_blocks!: obj.M=$(obj.M) != op.W=$W -- the 1/M scaling CM's blocks " *
              "use and the 1/W this family's own blocks use must be the same number")

    cf = octx.core_cf_ref[]
    cf isa CompressedFactual ||
        error("cmpq_fill_hessian_blocks!: core_cf_ref[] is not a CompressedFactual -- " *
              "prime_operator! was not called for this outer point. This family has no dense " *
              "economic fallback by design.")
    if octx.core_ws === nothing || octx.core_ws_for !== cf
        octx.core_ws = build_winner_pair_ctx(cf)
        octx.core_ws_for = cf
    end
    wctx = octx.core_ws::WinnerPairHessCtx
    wctx.ncolI + 1 == NCORE ||
        error("cmpq_fill_hessian_blocks!: wctx.ncolI+1=$(wctx.ncolI + 1) != NCORE=$NCORE -- the " *
              "winner-pair context's economic width disagrees with this family's layout")

    # ---- H_EE: shared winner-pair backend, stays in ITS OWN packed form (never unpacked) ----------
    # Every block below is timed under `@cmhess_prof` (cm_hessian_subblock_profiling.jl), the same
    # opt-in, off-by-default instrumentation CM's own callback uses -- a single Ref check when
    # disabled, which is what production runs with. Labels are `cmpq_*` so a `prof_summary()` after a
    # run separates this family's blocks from anything else in the same store.
    @cmhess_prof "cmpq_H_EE" winner_pair_hessian!(octx.hee_packed, obj, wctx)

    # ---- h = Psi''(r), the one weight vector every remaining block contracts against --------------
    obj.ddPsi!(obj.arg2, obj.arg0)
    h = obj.arg2

    # ---- the replicated-mu adapter: an IDENTITY, not a shortcut (see the hessian file's header) ---
    cmpq_replicate_shared_mu!(octx.pq_state, octx.state)

    # ---- H_RR, as a superset, via the standalone family's gated machinery -------------------------
    @cmhess_prof "cmpq_H_RR_tables" build_pairwise_quantile_hessian_tables!(octx.tabs_pq, octx.op, h, octx.tls)
    @cmhess_prof "cmpq_H_RR_fill" fill_pairwise_quantile_hessian_raw!(octx.HRR_pq, octx.op, octx.tabs_pq)
    @cmhess_prof "cmpq_H_RR_center" center_and_scale_pairwise_quantile_hessian!(octx.HRR_pq, octx.op,
                                                                               octx.pq_state, octx.tabs_pq)

    # ---- H_E,R, likewise ---------------------------------------------------------------------
    zc_ws = ensure_winner_zc_cross_scratch!(octx.zc_cross_ws, W, n_total_rows(D, L))
    @cmhess_prof "cmpq_H_ER" begin
        winner_pair_cross_hessian_zc_prep!(zc_ws, wctx, h)
        pairwise_quantile_cross_hessian_block!(octx.HEQ_pq, wctx, zc_ws, octx.op, octx.pq_state,
                                               octx.tls, h, octx.cross_hess_scratch)
    end

    # ---- H_R,CM: the one genuinely new block -----------------------------------------------------
    @cmhess_prof "cmpq_H_RCM_tables" build_cmpq_cross_hess_tables!(octx.tabs_x, octx.op, octx.cctx.Bidx,
                                                                  h, octx.refIndex1; Pow = octx.Pow)
    @cmhess_prof "cmpq_H_RCM_fill" fill_cmpq_cm_cross_block!(octx.HRC, octx.op, octx.state, octx.tabs_x,
                                                            octx.origins, octx.refIndex1, Lcm, octx.R, W)

    # ---- H_CM,CM: CM's own tables. `H=nothing` is admissible ONLY with fill_S=false ---------------
    @cmhess_prof "cmpq_H_CC_tables" if octx.threaded_bins
        build_bin_tables_threaded!(octx.cctx, octx.cctx.tls, nothing, h; fill_S = false)
        prefix_sum_tables_threaded!(octx.cctx; fill_S = false)
    else
        build_bin_tables!(octx.cctx, nothing, h; fill_S = false)
        prefix_sum_tables!(octx.cctx; fill_S = false)
    end
    @cmhess_prof "cmpq_H_CC_fill" fill_cm_HCC!(octx.cctx.Hfull, octx.cctx, W)

    # ---- H_E,CM: CM's winner-bin cross, filled ONCE then sliced per threshold block ---------------
    # `winner_pair_cross_hessian_fill!` recomputes `ddPsi!(obj.arg2, obj.arg0)` internally: same
    # inputs, same buffer, same values -- `h` above is an alias of `obj.arg2` and is unchanged by it.
    cm_ws = ensure_winner_bin_cross_scratch!(octx.cm_cross_ws, wctx.ncolI, D, Lcm)
    @cmhess_prof "cmpq_H_ECM_fill" winner_pair_cross_hessian_fill!(wctx, cm_ws, obj, octx.cctx.Bidx;
                                                                   Pow = octx.Pow)
    fam2 = octx.n_families == 2
    Hraw_EC = octx.cctx.Hraw_EC
    Hraw_EC2 = fam2 ? octx.cctx.Hraw_EC2 : nothing
    ncm_cdf = nO * Lcm
    R = octx.R
    @cmhess_prof "cmpq_H_ECM_blocks" @inbounds for l in 1:Lcm
        winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, cm_ws, l, octx.origins, octx.refIndex1, W;
                                            Hraw_EC_pow = Hraw_EC2)
        # The `:orthonormal` contrast is applied per threshold block on the RIGHT (`block * R`),
        # matching `hessian_cm_structured!`'s own H_EC convention; `R === nothing` (`:anchored`)
        # writes the raw block. Written straight into HEC's own column range -- no `block_ec`
        # round-trip, since `mul!` can target the destination view directly.
        cols = (l-1)*nO+1 : l*nO
        if R === nothing
            @views octx.HEC[:, cols] .= Hraw_EC
        else
            @views mul!(octx.HEC[:, cols], Hraw_EC, R)
        end
        if fam2
            cols2 = ncm_cdf + (l-1)*nO+1 : ncm_cdf + l*nO
            if R === nothing
                @views octx.HEC[:, cols2] .= Hraw_EC2
            else
                @views mul!(octx.HEC[:, cols2], Hraw_EC2, R)
            end
        end
    end
    return octx
end

"""
    cmpq_hess_cb_builder(octx::CMPQCoreHessCtx) -> callback

The KNITRO exact-Hessian callback. `userParams` arrives as `obj` (the `OperatorPsiBundle`) -- see
`inner_loop_KNITRO_cmpairwisequantile_operator`'s own adapter, which passes `userParams.obj`.

`operator_prep_for_hessian!` is called FIRST, on `evalRequest.x`: KNITRO does not guarantee an FG
call at the same point immediately precedes each Hessian call, so the same-point cache is treated as
an optimization, never as a precondition. Its miss path recomputes `r` through the family's own
`dual_index!` -- the identical code path the FG functor runs, with no dense `G` read.
"""
function cmpq_hess_cb_builder(octx::CMPQCoreHessCtx)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        obj = userParams
        @cmhess_prof "cmpq_prep_r" operator_prep_for_hessian!(octx.fg_state, evalRequest.x)
        cmpq_fill_hessian_blocks!(octx, obj)
        # Hoisted ONCE with a type assertion: re-reading `evalResult.hess` per entry through the
        # untyped callback signature was measured at ~660 ns/entry for the standalone family and was
        # the entire cost of its packed write. The assertion is deliberate -- if KNITRO.jl ever
        # changes that field's type this errors loudly instead of silently reverting.
        hess_out = evalResult.hess::Vector{Float64}
        HCC = @view octx.cctx.Hfull[octx.NCORE+1 : octx.NCORE+octx.ncm,
                                    octx.NCORE+1 : octx.NCORE+octx.ncm]
        @cmhess_prof "cmpq_pack" pack_cmpq_hessian!(hess_out, octx.hee_packed, octx.HEQ_pq, octx.HEC,
                                                   octx.HRR_pq, octx.HRC, HCC, octx.sig,
                                                   octx.NCORE, octx.n_restr, octx.ncm)
        octx.n_hess_calls += 1
        return 0
    end
end
