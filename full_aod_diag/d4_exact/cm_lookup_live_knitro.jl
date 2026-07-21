# ================================================================================================
# Part B: wire the lookup-based FG evaluator (cm_lookup_kernels.jl, already validated to machine
# precision against the dense `obj(x,g)` callable in c12i_validate_lookup_fg.jl) into an ACTUAL
# live KNITRO inner-dual solve, so end-to-end timing (task Part B.4) reflects a REAL solve, not
# just an isolated microkernel. Follows the EXISTING established pattern for this
# (`compressed_live.jl`'s custom KNITRO eval-callback wiring, reused as a template, not
# reinvented) -- register a custom FG callback that bypasses `obj`'s own callable for the hot
# per-iteration loop, keep the STANDARD Hessian callback (calls `obj`'s own callable unchanged,
# reading `obj.H`, which we fully materialize densely up front -- same one-time cost the dense
# baseline pays too, so this is a fair comparison of ONLY the FG-loop cost, not a hidden
# reallocation of work).
#
# Dense H is materialized densely (not lazily/partially, unlike compressed_live.jl) because this
# reformulation's whole point is a DIFFERENT representation of the SAME moments, not a compressed
# winner-form -- the Hessian still needs the literal dense G columns, so there is no compression
# opportunity there and none is claimed.
# ================================================================================================
using KNITRO

"FG callback: `evalResult.obj[1]`/`evalResult.objGrad` from the lookup-based `CMLookupState` callable."
function _callbackEvalFG_inner_cmlookup!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    x = evalRequest.x
    f = st(x, evalResult.objGrad)
    evalResult.obj[1] = f <= st.obj.lower_limit ? -KNITRO.KN_INFINITY : f
    return 0
end

"""
Hessian callback: calls the STANDARD dense callable's own Hessian path (`st.obj(x, h=...)`)
unchanged -- correct regardless of whether `obj.arg0` was last set by the lookup FG callback or
not, because `obj.H` is fully materialized (dense) and the callable recomputes `arg0` from `H`
and `x` itself at the top of its body every call (see `cc_algo/PsiObjectiveBundle.jl`'s
`PsiObjectiveBundleImplicit` callable) -- no cross-callback state threading required for
correctness (unlike `compressed_live.jl`, whose `H` is only ever PARTIALLY/lazily dense).
"""
function _callbackEvalH_inner_cmlookup!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    x = evalRequest.x
    st.obj(x, h = evalResult.hess)
    return 0
end

"Faithful mirror of `inner_loop_KNITRO`/`inner_loop_KNITRO_compressed`, registering the lookup FG/H callbacks with `st::CMLookupState` as userParams."
function inner_loop_KNITRO_cmlookup(obj, st::CMLookupState)
    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
    KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
    KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_cmlookup!)
    KNITRO.KN_set_cb_user_params(kc, cb, st)
    KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

    if KNITRO.KN_get_int_param(kc, "hessopt") == 1
        KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalH_inner_cmlookup!)
    end
    if obj.complement_index != [0 0]
        CS.inner_loop_complementarity_constraints(kc, obj)
    end

    KNITRO.KN_solve(kc)
    nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
    CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
    n_fg = st.n_fg_calls
    KNITRO.KN_free(kc)

    return nStatus, objSol, x, lambda_, n_fg
end

"""
    inner_loop_internal_cmlookup(obj, θ_full, ctx, aug; method=:interval, nthreads_use=1)

Full mirror of `inner_loop_internal(obj::PsiObjectiveBundleImplicit, θ)` (cc_algo): materializes
the DENSE `obj.H` once (same `moments!` call the dense path uses -- includes the dense CM block,
needed for the Hessian callback), builds a fresh `CMLookupState`, and runs the lookup-FG KNITRO
solve. Returns `(K_hard, x, nStatus, n_fg, st)`.
"""
function inner_loop_internal_cmlookup(obj, θ_full::AbstractVector, ctx, aug; method::Symbol = :interval, nthreads_use::Int = 1)
    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_full, ctx.U, obj)
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
    st = CMLookupState(obj, aug.ncore, aug.ncm, aug.L, aug.origins, aug.refIndex1, aug.bins, R;
                        method = method, nthreads_use = nthreads_use)

    nStatus, objSol, x, lambda_, n_fg = inner_loop_KNITRO_cmlookup(obj, st)

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, st
    else
        obj.x .= NaN
        return -1e10, x, nStatus, n_fg, st
    end
end

"""
    evaluate_fullA_cmlookup(x_free, ctx, aug; method=:interval, nthreads_use=1, warm=false) -> NamedTuple

Lookup-mode mirror of `oracle.jl::evaluate_fullA`, same return-field shape (a strict subset
actually used by the validation/benchmark scripts -- Delta_dual/Delta_primal/m_weights/nStatus/
max_abs_moment_kkt_resid/inner_iters/elapsed), built directly (not routed through the dense
`evaluate_fullA`) so its inner-solve step genuinely goes through
`inner_loop_internal_cmlookup` above, not the dense KNITRO wiring.
"""
function evaluate_fullA_cmlookup(x_free::AbstractVector{Float64}, ctx, aug; method::Symbol = :interval,
                                  nthreads_use::Int = 1, warm::Bool = false)
    obj = aug.obj_cm
    t_total0 = time()
    if !warm
        obj.x .= NaN
    end
    θ_full = CS.reconstruct_full(x_free, ctx.m)

    t_inner0 = time()
    K_hard, inner_x, nStatus, n_fg, st = inner_loop_internal_cmlookup(obj, θ_full, ctx, aug; method = method, nthreads_use = nthreads_use)
    t_inner = time() - t_inner0

    inner_iters = try
        CS.INNER_ITERS_TOTAL[]
    catch
        missing
    end

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        return (x_free = collect(x_free), θ_full = θ_full, inner_status = nStatus, n_fg_calls = n_fg,
                Delta_dual = NaN, Delta_primal = NaN, m_weights = Float64[], max_abs_moment_kkt_resid = NaN,
                zeta = NaN, lambda = Float64[], inner_iters = inner_iters,
                elapsed = (total = time() - t_total0, inner = t_inner))
    end

    W = size(obj.U, 1); d = obj.d
    K = zeros(W); G = zeros(W, d)
    obj.moments!(K, G, θ_full, obj.U, obj)

    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = obj(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    Delta_primal = primal_divergence(m_weights)

    ζstar = inner_x[1]; λstar = inner_x[2:end]
    moment_kkt = [abs(sum(m_weights .* G[:, j]) / W) for j in 1:min(length(λstar), size(G, 2))]
    max_abs_moment_kkt_resid = isempty(moment_kkt) ? NaN : maximum(moment_kkt)

    t_total = time() - t_total0
    return (x_free = collect(x_free), θ_full = θ_full, inner_status = nStatus, n_fg_calls = n_fg,
            Delta_dual = Delta_dual, Delta_primal = Delta_primal, m_weights = m_weights,
            max_abs_moment_kkt_resid = max_abs_moment_kkt_resid, zeta = ζstar, lambda = collect(λstar),
            inner_iters = inner_iters, elapsed = (total = t_total, inner = t_inner))
end
