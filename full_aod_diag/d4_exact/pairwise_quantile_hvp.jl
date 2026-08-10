# ================================================================================================
# Exact Hessian-vector product (HVP) for the pairwise-quantile-independence restriction's inner
# dual solve. Handover doc (docs/PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_HANDOVER_2026-08-09.md)
# asked whether H_EE·v needs new derivation (its own "harder half"). It does NOT: this restriction's
# own FG functor (`pairwise_quantile_production.jl::dual_index!`/`(::PairwiseQuantileOperatorState)`)
# already composes `arg0 = -zeta - E*lambda_E - R*lambda_R` via `economic_forward!`/
# `pairwise_quantile_forward!`, and `economic_forward!`/`economic_transpose!` ARE
# `compressed_dual_contraction!`/`compressed_transpose_contraction!` (economic_operator.jl's own
# header: "a thin, explicitly-named wrapper" around them) -- the EXACT primitives
# `compressed_cc_hvp` (compressed_cc_inner.jl) already composes for another family. So the FULL
# combined Hv (H_EE*v_E + H_ER*v_R for the E-block, H_ER'*v_E + H_RR*v_R for the R-block) is just
# `(1/M)*FullG'*diag(h)*(FullG*v)` built from the SAME forward!/transpose! calls the FG callback
# already makes -- no dense G, no dense H_ER, no T1-T4 tables, ever.
#
# Sign-convention derivation (all functions below are already-validated but each carries its OWN
# sign/scale baked in for ITS existing caller, not a "pure" bilinear contraction):
#   - `economic_forward!(out, β, cf, ws)` OVERWRITES out = E*β (no hidden sign, matches
#     `compressed_dual_contraction!`).
#   - `pairwise_quantile_forward!(arg0, λ_M, λ_P, op, state)` ACCUMULATES arg0 -= R*λ (own
#     docstring: "forward ACCUMULATES -(Rλ) into arg0"). Calling it into a fresh zero buffer gives
#     `buf = -(R*λ)`.
#   - `economic_transpose!(grad_E, w, cf, ws)` OVERWRITES grad_E = E'*w, RAW (no sign, no 1/M --
#     own docstring: "callers apply the -(1/M) scale themselves").
#   - `pairwise_quantile_transpose!(g_M, g_P, w, op, state, tls, scratch)` OVERWRITES
#     g_M/g_P = -(1/W)*(R_centered'*w) (own docstring: "-(1/W)*(Mraw - S/5)" etc; R_centered is the
#     restriction's own centered feature matrix, target already subtracted, matching
#     `pairwise_quantile_forward!`'s R exactly).
#
# H = (1/M)*FullG'*diag(h)*FullG is invariant to a common sign flip of FullG's columns, so pick
# J = +[1 | E | R] (matching `compressed_cc_hvp`'s own `r = ddPsq*(p_ζ + cpλ)`, POSITIVE):
#   u = p_ζ*1 + E*p_E + R*p_R
#     = p_ζ*1 + economic_forward!(p_E)  -  [pairwise_quantile_forward! into a zero buffer]  (the
#       forward! buffer already equals -(R*p_R), so subtracting it ADDS R*p_R -- see code)
#   r = h .* u                    (h = Psi''(q), q = CURRENT arg0 at the x KNITRO is asking about,
#                                   recomputed via `dual_index!` every call -- KNITRO may call the
#                                   Hessian/HVP callback at an x different from the last FG call,
#                                   e.g. inside a CG sub-iteration; q cannot be assumed cached,
#                                   exactly the same discipline `compressed_q_at` documents)
#   Hp_ζ = sum(r)/M
#   Hp_E = economic_transpose!(r)/M                      (raw contract, just scale by 1/M)
#   Hp_R = -pairwise_quantile_transpose!(r)               (transpose! already returns -(1/W)*R'r;
#                                                            M=W in this FG's own convention, so
#                                                            negating recovers +(1/M)*R_centered'*r)
#
# Never materializes a dense H_ER or the T1-T4 combo tables -- O(W*(D+npair)) per HVP call, same
# complexity class as one FG evaluation, matching the handover's own stated goal exactly.
# ================================================================================================

isdefined(Main, :PairwiseQuantileOperatorState) || include(joinpath(@__DIR__, "pairwise_quantile_production.jl"))

"""
    PairwiseQuantileHVPScratch(st::PairwiseQuantileOperatorState)

Persistent (campaign-lifetime, built once, reused every HVP callback call) scratch -- no
per-call allocation, matching this codebase's zero-hot-path-allocation discipline. Sized off the
SAME `st` the HVP callback will be called with.
"""
mutable struct PairwiseQuantileHVPScratch
    u::Vector{Float64}          # W
    r::Vector{Float64}          # W  (reused: forward!-buffer, then h.*u, then transpose! input)
    h::Vector{Float64}          # W  (Psi''(q))
    econ_fwd_buf::Vector{Float64}   # W        (economic_forward! output -- E*p_E, one entry per draw)
    econ_buf::Vector{Float64}       # ncore1   (economic_transpose! output -- E'*r)
    gM::Matrix{Float64}         # D x (L-1)   (pairwise_quantile_transpose! output, negated after)
    gP::Array{Float64,3}        # (L-1) x (L-1) x npair
    tls::PairwiseQuantileThreadScratch
    tscratch::PairwiseQuantileTransposeScratch
end

function PairwiseQuantileHVPScratch(st::PairwiseQuantileOperatorState)
    W = st.op.W; D = st.op.D; npair = st.op.npair; ncore1 = st.ncore1; L = st.op.L; nc = L - 1
    return PairwiseQuantileHVPScratch(zeros(W), zeros(W), zeros(W), zeros(W), zeros(ncore1),
        zeros(D, nc), zeros(nc, nc, npair),
        build_pairwise_quantile_thread_scratch(D, npair, L), PairwiseQuantileTransposeScratch(D, npair, L))
end

"""
    pairwise_quantile_hvp!(Hp::AbstractVector, st, sc, x, v) -> Hp

Exact H*v for the FULL inner-dual Hessian (H_EE, H_ER, H_RR, and the ζ row/col all at once), no
dense assembly. `x`/`v` are KNITRO's full n-vectors (`n = 1+ncore1+n_mean_flat(D)+n_pair_flat(npair)`
in the SAME `[ζ|λ_E|λ_M|λ_P]` layout `dual_index!` uses). Writes into caller-supplied `Hp` (same
layout) and returns it -- no allocation.
"""
function pairwise_quantile_hvp!(Hp::AbstractVector{Float64}, st::PairwiseQuantileOperatorState,
        sc::PairwiseQuantileHVPScratch, x::AbstractVector{Float64}, v::AbstractVector{Float64})
    obj = st.obj
    ncore1 = st.ncore1
    op = st.op
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    M = length(st.arg0)

    # q = current arg0 at x (recomputed fresh -- KNITRO may call this at an x other than the last
    # FG call's x; dual_index! also refreshes st.econ_ws if needed).
    q = dual_index!(st, x)
    obj.ddPsi!(sc.h, q)

    p_ζ = v[1]
    p_E = @view v[2:1+ncore1]
    p_M = reshape(@view(v[2+ncore1 : 1+ncore1+n_mean_flat(D, L)]), nc, D)'
    p_P = reshape(@view(v[2+ncore1+n_mean_flat(D, L) : 1+ncore1+n_mean_flat(D, L)+n_pair_flat(npair, L)]), nc, nc, npair)

    cf = st.core_cf_ref[]
    if st.econ_ws === nothing || st.econ_ws_for !== cf
        st.econ_ws = economic_operator_workspace(cf)
        st.econ_ws_for = cf
    end

    fill!(sc.u, p_ζ)
    economic_forward!(sc.econ_fwd_buf, p_E, cf, st.econ_ws)   # econ_fwd_buf = E*p_E  (length W)
    sc.u .+= sc.econ_fwd_buf
    fill!(sc.r, 0.0)
    pairwise_quantile_forward!(sc.r, p_M, p_P, op, st.bin_state)   # r = -(R*p_R)
    sc.u .-= sc.r                                          # u = p_ζ + E*p_E + R*p_R

    sc.r .= sc.h .* sc.u

    Hpζ = sum(sc.r) / M
    economic_transpose!(sc.econ_buf, sc.r, cf, st.econ_ws)   # raw E'*r
    sc.econ_buf ./= M

    pairwise_quantile_transpose!(sc.gM, sc.gP, sc.r, op, st.bin_state, sc.tls, sc.tscratch)
    # transpose! returns -(1/W)*(R_centered'*r); M==W here, so negating gives +(1/M)*R_centered'*r.
    sc.gM .*= -1.0
    sc.gP .*= -1.0

    Hp[1] = Hpζ
    @views Hp[2:1+ncore1] .= sc.econ_buf
    HpM_view = reshape(@view(Hp[2+ncore1 : 1+ncore1+n_mean_flat(D, L)]), nc, D)
    HpM_view .= sc.gM'
    HpP_view = reshape(@view(Hp[2+ncore1+n_mean_flat(D, L) : 1+ncore1+n_mean_flat(D, L)+n_pair_flat(npair, L)]), nc, nc, npair)
    HpP_view .= sc.gP
    return Hp
end

"""
    make_hvp_callback(sc::PairwiseQuantileHVPScratch)

KNITRO HVP callback factory: mirrors `_callbackEvalHV_inner_compressed!`
(compressed_inner_alt_solvers.jl) exactly, generalized to this restriction's combined
[ζ|λ_E|λ_M|λ_P] state. `KN_set_cb_hess`'s 2-arg method (see `C_wrapper.jl`) carries NO separate
user-params slot of its own -- KNITRO passes the SAME `userParams` registered via
`KN_set_cb_user_params(kc, cb, st)` for `cb` to both the FG and the Hessian/HV callback tied to
that `cb` (confirmed by how the existing dense callback, `pairwisequantile_hess_cb_builder`, reads
`userParams.obj`, i.e. `userParams === st`). `sc` (the persistent HVP scratch) is therefore
captured by closure here rather than threaded through `userParams`.
"""
function make_hvp_callback(sc::PairwiseQuantileHVPScratch)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        st = userParams
        x = evalRequest.x
        v = evalRequest.vec
        pairwise_quantile_hvp!(evalResult.hessVec, st, sc, x, v)
        return 0
    end
end

"""
    inner_loop_KNITRO_pairwisequantile_operator_hvp(obj, st) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

HVP variant of `inner_loop_KNITRO_pairwisequantile_operator` (pairwise_quantile_production.jl):
registers `_callbackEvalHV_inner_pairwisequantile!` instead of the dense callback. Asserts
`hessopt==5` (product mode) -- `obj.inner_loop_opt` MUST point at an hvp `.opt` file
(algorithm=cg, hessopt=5), never silently proceeds under the wrong mode.
"""
function inner_loop_KNITRO_pairwisequantile_operator_hvp(obj, st::PairwiseQuantileOperatorState)
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

        hessopt = KNITRO.KN_get_int_param(kc, "hessopt")
        hessopt == 5 || error("inner_loop_KNITRO_pairwisequantile_operator_hvp: expected hessopt=product(5), got $hessopt -- check obj.inner_loop_opt points at an hvp .opt file (algorithm=cg, hessopt=5)")

        sc = PairwiseQuantileHVPScratch(st)
        n_hess = Ref(0)
        hvp_raw = make_hvp_callback(sc)
        hvp_cb_adapted = callback_health_guard((kc2, cb2, evalRequest, evalResult, userParams) -> begin
            r = hvp_raw(kc2, cb2, evalRequest, evalResult, userParams)
            n_hess[] += 1
            return r
        end, health)
        KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hvp_cb_adapted)

        KNITRO.KN_solve(kc)
        nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        assert_no_fake_success!("inner_loop_KNITRO_pairwisequantile_operator_hvp", health, nStatus, st.n_fg_calls, x_initial, x)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        n_fg = st.n_fg_calls
        KNITRO.KN_free(kc)

        return nStatus, objSol, x, lambda_, n_fg, n_hess[]
    finally
        CS.guard_exit_inner_solve!()
    end
end

"""
    archPQ_base_state_hvp(x_free0, raw_cutoffs, econ_ctx, ctx_cm, layout) -> (nStatus, x, obj, n_fg, n_hess)

HVP-variant driver, mirrors `archPQ_base_state` (pairwise_quantile_production.jl) exactly except
for which inner-loop wrapper it calls. `ctx_cm.obj.inner_loop_opt` MUST already point at an hvp
.opt file before calling this (caller's responsibility, same discipline as
`inner_loop_internal_compressed_variant`'s callers).
"""
function archPQ_base_state_hvp(x_free0::AbstractVector, raw_cutoffs::AbstractVector{Float64}, econ_ctx, ctx_cm, layout::PairwiseQuantileCutoffLayout)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    prime_operator!(obj, θ_econ0, econ_ctx, ctx_cm.pq_core_cf_ref)

    st = PairwiseQuantileOperatorState(obj, obj.outer_constr_index - 1 - n_total_rows(ctx_cm.pq_op.D, ctx_cm.pq_op.L),
        ctx_cm.pq_op, ctx_cm.U, ctx_cm.pq_bin_state, ctx_cm.pq_core_cf_ref)
    reset_for_solve!(st, raw_cutoffs, layout)

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_pairwisequantile_operator_hvp(obj, st)
    return nStatus, x, obj, n_fg, n_hess
end
