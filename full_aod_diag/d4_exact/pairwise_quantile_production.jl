# ================================================================================================
# Real production wiring for the pairwise-quantile-independence restriction: an OperatorPsiBundle
# (no-H) obj, a callable FG state (PairwiseQuantileOperatorState, mirrors OriginZCOperatorState
# exactly), and a real KNITRO Hessian callback combining H_EE (winner_pair_hessian!, UNCHANGED),
# H_E,R (pairwise_quantile_cross_hessian_block!), and H_MM/H_MP/H_PP (fill_pairwise_quantile_
# hessian_raw!/center_and_scale_pairwise_quantile_hessian!). Mirrors cm_originzc_lookup_kernels.jl
# + cm_originzc_lookup_production.jl's structure, generalized to this restriction's own operator.
# ================================================================================================

isdefined(Main, :economic_forward!) || include(joinpath(@__DIR__, "economic_operator.jl"))
isdefined(Main, :cf_build) || include(joinpath(@__DIR__, "compressed_factual_buffer_reuse.jl"))
isdefined(Main, :OperatorPsiBundle) || include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
isdefined(Main, :CallbackHealthRecord) || include(joinpath(@__DIR__, "cm_callback_health.jl"))
isdefined(Main, :WinnerPairHessCtx) || include(joinpath(@__DIR__, "core_exact_hessian.jl"))
isdefined(Main, :WinnerZCCrossScratch) || include(joinpath(@__DIR__, "winner_pair_cross_hessian.jl"))

"""
    build_pairwise_quantile_augmented_obj(ctx, layout, Z, Q) -> (obj_pq, ncore_econ, core_cf_ref, op, D)

`ctx.obj` (`obj0`) is the economic-only `PsiObjectiveBundleImplicit` (e.g. from `d4_exact_setup`).
Builds the true no-H `OperatorPsiBundle` for `[economic | marginal | pair]`, mirroring
`cm_originzc_moments.jl`'s `:operator` construction verbatim (same field list, same
`outer_constr_index_new` convention), plus this restriction's operator over the FRECHET
productivity draws `Z` and the FIXED cutoffs `Q` (version B -- both are required arguments, built by
`pairwise_quantile_frechet_features` and `pairwise_quantile_fixed_cutoffs` from an explicit
`cutoff_source`; there is no cutoff default and none is invented here).
"""
function build_pairwise_quantile_augmented_obj(ctx, layout::PairwiseQuantileMassLayout,
                                               Z::AbstractMatrix{Float64}, Q::AbstractMatrix{Float64})
    obj0 = ctx.obj
    ncore_econ = obj0.d
    D = ctx.D
    D == layout.D || error("build_pairwise_quantile_augmented_obj: ctx.D=$D != layout.D=$(layout.D)")
    n_rows = n_total_rows(D, layout.L)
    outer_constr_index_new = obj0.outer_constr_index + n_rows
    core_cf_ref = Ref{Any}(nothing)
    obj_pq = OperatorPsiBundle(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, l = obj0.l, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
        inner_loop_opt = obj0.inner_loop_opt)
    size(Z) == size(ctx.U) ||
        error("build_pairwise_quantile_augmented_obj: size(Z)=$(size(Z)) != size(ctx.U)=$(size(ctx.U))")
    op = PairwiseQuantileOperator(Z, layout.L, Q)
    return (obj_pq = obj_pq, ncore_econ = ncore_econ, core_cf_ref = core_cf_ref, op = op, D = D)
end

"""
    PairwiseQuantileOperatorState

Per-inner-solve callable FG state, mirroring `OriginZCOperatorState` exactly: `core_cf_ref` is the
SAME `Ref{Any}` `prime_operator!` publishes a fresh `CompressedFactual` into (called once per outer
point, BEFORE the KNITRO solve starts -- same lifecycle point `reset_for_solve!` below refreshes
`mass_state`).
"""
mutable struct PairwiseQuantileOperatorState
    obj::Any
    ncore1::Int
    op::PairwiseQuantileOperator
    mass_state::PairwiseQuantileMassState
    core_cf_ref::Ref{Any}
    econ_ws::Union{Nothing,Any}
    econ_ws_for::Any
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    econ_buf::Vector{Float64}
    n_fg_calls::Int
end

function PairwiseQuantileOperatorState(obj, ncore1::Int, op::PairwiseQuantileOperator,
        mass_state::PairwiseQuantileMassState, core_cf_ref::Ref{Any})
    W = op.W
    return PairwiseQuantileOperatorState(obj, ncore1, op, mass_state, core_cf_ref,
        nothing, nothing, zeros(W), zeros(W), zeros(W), 0)
end

"""
    reset_for_solve!(st::PairwiseQuantileOperatorState, raw_masses, layout) -> st

Call ONCE per outer point, BEFORE the KNITRO solve starts: decodes the raw outer coordinates into
this restriction's free bin masses (`set_pairwise_quantile_masses!`) -- the version-B form of the
"decode once per outer point, never inside a callback" requirement, satisfied structurally by this
lifecycle placement.

Under version B the BIN ASSIGNMENT is no longer refreshed here at all: the cutoffs are campaign
constants, so `op.bin` was built once in the operator's constructor and never changes. Only the
masses are per-outer-point state.
"""
function reset_for_solve!(st::PairwiseQuantileOperatorState, raw_masses::AbstractVector{Float64},
        layout::PairwiseQuantileMassLayout)
    set_pairwise_quantile_masses!(st.mass_state, raw_masses, layout)
    st.n_fg_calls = 0
    return st
end

function dual_index!(st::PairwiseQuantileOperatorState, x::AbstractVector{Float64})
    obj = st.obj
    ncore1 = st.ncore1
    op = st.op
    D = op.D; npair = op.npair

    L = op.L; nc = L - 1
    ζ = x[1]
    λ_E = @view x[2:1+ncore1]
    # BUG FIX (found live, 2026-08-09, via a real-KNITRO-context finite-difference check that
    # exactly swapped two marginal-cell columns): `marginal_row(o,a,L)=(o-1)*(L-1)+a`
    # (pairwise_quantile_hessian.jl) is an O-MAJOR flat layout (each origin's `nc=L-1` bins
    # contiguous). Julia's own `reshape(v, D, nc)` is COLUMN-MAJOR, i.e. A-MAJOR
    # (`M[o,a] = v[(a-1)*D+o]`) -- the OPPOSITE convention. `reshape(v, nc, D)'` (reshape into
    # (nc,D) then transpose) gives `M[o,a] = v[(o-1)*nc+a]`, matching marginal_row exactly. The
    # pair layout needs NO such fix: `pair_row(D,pidx,a,b,L) = ...+(b-1)*nc+a` already matches
    # `reshape(v,nc,nc,npair)`'s natural column-major layout (`M[a,b,pidx] = v[(pidx-1)*nc^2+(b-1)*nc+a]`)
    # directly.
    λ_M = reshape(@view(x[2+ncore1 : 1+ncore1+n_mean_flat(D, L)]), nc, D)'
    λ_P = reshape(@view(x[2+ncore1+n_mean_flat(D, L) : 1+ncore1+n_mean_flat(D, L)+n_pair_flat(npair, L)]), nc, nc, npair)

    fill!(st.arg0, -ζ)

    cf = st.core_cf_ref[]
    cf isa CompressedFactual || error("PairwiseQuantileOperatorState: core_cf_ref[] is not a CompressedFactual -- prime_operator! was not called for this outer point")
    if st.econ_ws === nothing || st.econ_ws_for !== cf
        st.econ_ws = economic_operator_workspace(cf)
        st.econ_ws_for = cf
    end
    economic_forward!(st.econ_buf, λ_E, cf, st.econ_ws)
    st.arg0 .-= st.econ_buf

    pairwise_quantile_forward!(st.arg0, λ_M, λ_P, op, st.mass_state)
    return st.arg0
end

function (st::PairwiseQuantileOperatorState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = length(st.arg0)
    ncore1 = st.ncore1
    op = st.op
    D = op.D; npair = op.npair

    ζ = x[1]
    dual_index!(st, x)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        obj.dPsi!(st.arg1, st.arg0)
        g[1] = 1.0 - sum(st.arg1) / M
        cf = st.core_cf_ref[]
        g_E = @view g[2:1+ncore1]
        economic_transpose!(g_E, st.arg1, cf, st.econ_ws)
        g_E .*= -(1.0 / M)

        L = op.L; nc = L - 1
        g_M = reshape(@view(g[2+ncore1 : 1+ncore1+n_mean_flat(D, L)]), nc, D)'
        g_P = reshape(@view(g[2+ncore1+n_mean_flat(D, L) : 1+ncore1+n_mean_flat(D, L)+n_pair_flat(npair, L)]), nc, nc, npair)
        tls = build_pairwise_quantile_thread_scratch(D, npair, L)
        scratch = PairwiseQuantileTransposeScratch(D, npair, L)
        pairwise_quantile_transpose!(g_M, g_P, st.arg1, op, st.mass_state, tls, scratch)
    end

    obj.arg0 .= st.arg0
    st.n_fg_calls += 1
    return f
end

"KNITRO FG callback: mirrors _callbackEvalFG_inner_originzc_operator! exactly (generic on any
callable `st`, no ZC-specific assumption)."
function _callbackEvalFG_inner_pairwisequantile!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    x = evalRequest.x
    f = st(x, evalResult.objGrad)
    evalResult.obj[1] = f <= st.obj.lower_limit ? -KNITRO.KN_INFINITY : f
    return 0
end

"""
    PairwiseQuantileCoreHessCtx

Per-context (campaign-lifetime shape) Hessian state: H_EE via the SHARED `WinnerPairHessCtx`/
`winner_pair_hessian!` (UNCHANGED), H_E,R via `pairwise_quantile_cross_hessian_block!`, H_MM/MP/PP
via `fill_pairwise_quantile_hessian_raw!`/`center_and_scale_pairwise_quantile_hessian!`.

`hee_packed`/`HEQ`/`HRR` are persistent (campaign-lifetime shape, never reallocated) -- fixes the
handover doc's "Two smaller, lower-risk fixes" #1 (dense assembly/packing overhead): the packed
KNITRO output is now written DIRECTLY from these three structured blocks (`_pk_upper` below), no
intermediate `(NCORE+n_rows) x (NCORE+n_rows)` `Hfull` build/mirror/repack (previously ~98MB of
avoidable memory traffic per callback at D=20).
"""
mutable struct PairwiseQuantileCoreHessCtx
    NCORE::Int
    D::Int
    op::PairwiseQuantileOperator
    mass_state::PairwiseQuantileMassState
    tabs::PairwiseQuantileHessianTables
    tls::PairwiseQuantileThreadScratch
    core_cf_ref::Ref{Any}
    core_ws::Union{Nothing,WinnerPairHessCtx}
    core_ws_for::Any
    cross_scratch::Union{Nothing,WinnerZCCrossScratch}
    cross_hess_scratch::PairwiseQuantileCrossHessScratch
    hee_packed::Vector{Float64}
    HEQ::Matrix{Float64}
    HRR::Matrix{Float64}
end

function PairwiseQuantileCoreHessCtx(NCORE::Int, op::PairwiseQuantileOperator, mass_state::PairwiseQuantileMassState,
        core_cf_ref::Ref{Any})
    D = op.D; L = op.L
    n_rows = n_total_rows(D, L)
    ncolI = NCORE - 1
    return PairwiseQuantileCoreHessCtx(NCORE, D, op, mass_state, PairwiseQuantileHessianTables(op),
        build_pairwise_quantile_thread_scratch(D, op.npair, L), core_cf_ref, nothing, nothing, nothing,
        PairwiseQuantileCrossHessScratch(D, op.npair, op.W, ncolI, L),
        Vector{Float64}(undef, NCORE * (NCORE + 1) ÷ 2), zeros(NCORE, n_rows), zeros(n_rows, n_rows))
end

"Row-major upper-triangular packed index of (i,j), i<=j, within an n x n matrix -- the SAME
convention `winner_pair_hessian!` (`core_exact_hessian.jl`) already fills `hee_packed` with (a
single running counter across BOTH its zeta-row and lambda-lambda loops), confirmed by construction
rather than assumed: `_pk_upper(1,1,n)=1`, `_pk_upper(1,2,n)=2`, ..., `_pk_upper(2,2,n)=n+1`."
@inline _pk_upper(i::Int, j::Int, n::Int) = (i - 1) * (n + 1) - div((i - 1) * i, 2) + (j - i + 1)

function pairwisequantile_hess_cb_builder(octx::PairwiseQuantileCoreHessCtx)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        obj = userParams
        NCORE = octx.NCORE; D = octx.D; op = octx.op
        n_rows = n_total_rows(D, op.L)
        n = NCORE + n_rows

        cf = octx.core_cf_ref[]
        cf isa CompressedFactual || error("pairwisequantile_hess_cb_builder: core_cf_ref[] is not a CompressedFactual")
        if octx.core_ws === nothing || octx.core_ws_for !== cf
            octx.core_ws = build_winner_pair_ctx(cf)
            octx.core_ws_for = cf
        end
        wctx = octx.core_ws

        # H_EE (unchanged shared winner-pair backend) -- stays in ITS OWN packed form; read directly
        # at pack time below, never unpacked into a dense corner.
        hee_packed = octx.hee_packed
        winner_pair_hessian!(hee_packed, obj, wctx)

        # h_w = Psi''(R_w) for the restriction Hessian + H_E,R cross block
        obj.ddPsi!(obj.arg2, obj.arg0)
        h = obj.arg2

        build_pairwise_quantile_hessian_tables!(octx.tabs, op, h, octx.tls)
        HRR = octx.HRR
        fill_pairwise_quantile_hessian_raw!(HRR, op, octx.tabs)
        center_and_scale_pairwise_quantile_hessian!(HRR, op, octx.mass_state, octx.tabs)
        # HRR is stored LOWER-TRIANGLE ONLY (2026-08-10): KNITRO is handed a packed UPPER triangle,
        # so every unordered pair is read exactly once and mirroring it into both halves was ~242M
        # redundant stores plus ~242M redundant centering flops per callback for a half nothing
        # reads. `_lo_write!` (pairwise_quantile_hessian.jl) stores each pair at `[max(i,j),min(i,j)]`
        # and the centering pass touches only `I >= J`; the packed write below reads exactly that
        # triangle, by column.

        # WinnerPairHessCtx's own convention: H_EE is (1+ncolI) x (1+ncolI) = NCORE x NCORE, so
        # ncolI = NCORE-1; pairwise_quantile_cross_hessian_block! fills a (ncolI+1) x n_rows = NCORE
        # x n_rows block, row 1 = the SAME "ones"/zeta-paired row H_EE's own row/col 1 already is.
        octx.cross_scratch = ensure_winner_zc_cross_scratch!(Ref{Union{Nothing,WinnerZCCrossScratch}}(octx.cross_scratch),
            op.W, n_rows)
        winner_pair_cross_hessian_zc_prep!(octx.cross_scratch, wctx, h)
        HEQ = octx.HEQ
        pairwise_quantile_cross_hessian_block!(HEQ, wctx, octx.cross_scratch, op, octx.mass_state, octx.tls, h,
            octx.cross_hess_scratch)

        # Direct packed write: select the correct source block per (i,j) instead of assembling a
        # dense Hfull first (handover doc fix #1) -- H_EE via hee_packed's own packed index, H_E,R
        # via HEQ, H_MM/MP/PP via HRR, each already in its final (correctly signed/scaled) form.
        #
        # PERFORMANCE (2026-08-10). At D=20/L=10/W=100,000 this block MEASURED 86.16 s of a 96.72 s
        # Hessian callback -- 46% of the entire 1323 s inner solve (logs/pq_L10_blockprofile.log).
        # Two things were wrong with the obvious loop, and both are fixed here:
        #
        #  1. STRIDE. `HRR[i-NCORE, j-NCORE]` with `i` fixed and `j` running walks a ROW of a
        #     COLUMN-MAJOR matrix: stride `n_rows` = 15,570 doubles = 124 KB, i.e. a cache miss and
        #     usually a TLB miss on essentially every one of ~121M reads. `HRR` is symmetric --
        #     `fill_pairwise_quantile_hessian_raw!` mirrors both triangles and the centering
        #     identity `(H - t*r' - r*t' + S*t*t')/W` is symmetric in (I,J) -- so reading
        #     `HRR[j-NCORE, i-NCORE]` instead walks a CONTIGUOUS COLUMN and returns the identical
        #     value. That is now more than an optimization: it is the ONLY populated triangle (see
        #     above). The D=4 dense oracle checks the packed vector against a full symmetric dense
        #     reference every run, so this is a gated property, not an assumption.
        #  2. SERIAL. The running counter `k` looked sequential, but its value at the start of row
        #     `i` is closed-form, `k0(i) = (i-1)*n - (i-1)*(i-2)/2`, so rows are independent and the
        #     loop threads with no reduction and no ordering concern. Writes are disjoint by
        #     construction: row `i` owns exactly `k0(i)+1 : k0(i)+(n-i+1)`, and every packed slot
        #     belongs to exactly one row.
        #
        # SCHEDULE: `:dynamic`, NOT the `:static` used elsewhere in this restriction. The rows of an
        # upper triangle have length `n-i+1`, so contiguous static chunks are badly imbalanced --
        # the thread owning the first 1/nt of the rows does ~31x the work of the thread owning the
        # last 1/nt at nt=16, capping the speedup near 8x. `:static` is used in the T1-T4 scatter
        # because that loop feeds a REDUCTION whose order must be fixed for bit-reproducibility;
        # here there is no reduction at all (each packed slot is written exactly once, by one
        # thread), so the schedule cannot affect the output by even one ulp.
        #
        # Correctness is unaffected: same values, same packed positions. Any numerical movement here
        # is a bug, not a tolerance -- the D=4 oracle's packed round-trip check gates it.
        Threads.@threads :dynamic for i in 1:n
            k = (i - 1) * n - div((i - 1) * (i - 2), 2)
            @inbounds if i <= NCORE
                for j in i:NCORE
                    k += 1
                    evalResult.hess[k] = hee_packed[_pk_upper(i, j, NCORE)]
                end
                for j in NCORE+1:n
                    k += 1
                    evalResult.hess[k] = HEQ[i, j - NCORE]
                end
            else
                ii = i - NCORE
                @simd for j in i:n
                    evalResult.hess[k+j-i+1] = HRR[j - NCORE, ii]   # column walk; HRR symmetric
                end
            end
        end
        return 0
    end
end

"KNITRO registration, mirrors inner_loop_KNITRO_originzc_operator exactly (generalized to accept any callable `st`)."
function inner_loop_KNITRO_pairwisequantile_operator(obj, st::PairwiseQuantileOperatorState; hess_cb_builder)
    CS.guard_enter_inner_solve!()
    health = CallbackHealthRecord()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        x_initial = CS.inner_loop_initial_values(obj)
        KNITRO.KN_set_var_primal_init_values_all(kc, x_initial)

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], callback_health_guard(_callbackEvalFG_inner_pairwisequantile!, health))
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        n_hess = Ref(0)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_cb_raw = hess_cb_builder(obj)
            hess_cb_adapted = callback_health_guard((kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = hess_cb_raw(kc2, cb2, evalRequest, evalResult, userParams.obj)
                n_hess[] += 1
                return r
            end, health)
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb_adapted)
        end

        KNITRO.KN_solve(kc)
        nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        assert_no_fake_success!("inner_loop_KNITRO_pairwisequantile_operator", health, nStatus, st.n_fg_calls, x_initial, x)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        n_fg = st.n_fg_calls
        KNITRO.KN_free(kc)

        return nStatus, objSol, x, lambda_, n_fg, n_hess[]
    finally
        CS.guard_exit_inner_solve!()
    end
end

"""
    archPQ_base_state(x_free0, raw_masses, econ_ctx, ctx_cm, layout) -> (nStatus, x, obj)

Top-level driver: builds `θ_econ`, primes the economic state (`prime_operator!`), refreshes the
restriction's bins (`reset_for_solve!`), and runs the real KNITRO inner dual solve.

`econ_ctx` MUST be the ORIGINAL, unaugmented economic context (e.g. the plain `ctx` returned by
`d4_exact_setup`, whose `.obj` is the economic-only bundle) -- NOT `ctx_cm` (whose own `.obj` has
been overwritten with the restriction-augmented `OperatorPsiBundle`). `cf_build`/`prime_operator!`
reads dimensionality off `ctx.obj` internally; passing the augmented `ctx_cm` there would corrupt
`cf.oci` (confirmed live: passing `ctx_cm` gave `cf.oci-1=129` instead of the correct economic-only
17, a real bug caught by this session's own D4 KNITRO run, not a hypothetical). Mirrors origin-ZC's
own `octx.econ_ctx` field, which exists for exactly this reason.

`ctx_cm` carries `.obj` (the `OperatorPsiBundle`), `.pq_op`/`.pq_mass_state`/`.pq_core_cf_ref`/
`.pq_hess_ctx`/`.m` (the economic `FreeParamMap`, reused unchanged from `econ_ctx`/`ctx`).
"""
function archPQ_base_state(x_free0::AbstractVector, raw_masses::AbstractVector{Float64}, econ_ctx, ctx_cm, layout::PairwiseQuantileMassLayout)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    prime_operator!(obj, θ_econ0, econ_ctx, ctx_cm.pq_core_cf_ref)

    st = PairwiseQuantileOperatorState(obj, obj.outer_constr_index - 1 - n_total_rows(ctx_cm.pq_op.D, ctx_cm.pq_op.L),
        ctx_cm.pq_op, ctx_cm.pq_mass_state, ctx_cm.pq_core_cf_ref)
    reset_for_solve!(st, raw_masses, layout)

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_pairwisequantile_operator(obj, st;
        hess_cb_builder = _ -> pairwisequantile_hess_cb_builder(ctx_cm.pq_hess_ctx))
    return nStatus, x, obj, n_fg, n_hess
end
