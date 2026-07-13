# ============================================================================
# Generic (method-agnostic) cached outer-loop driver.
#
# Ties together FreeParamMap (differentiate/optimize only the free
# coordinates) and OuterEvalCache (solve the inner CC problem at most once per
# unique free-outer point) around the EXISTING, unmodified `inner_loop_internal`
# and `(Q::PsiObjectiveBundleImplicit)` callable's constraint-VALUE path (cheap,
# reused as-is — only the GRADIENT/Jacobian path is replaced).
#
# Gradients are supplied by the CALLER as free-only closures, because the two
# formulations need genuinely different gradient logic:
#   - full-A: a direct scalar ForwardDiff envelope gradient over x_free for the
#     divergence constraint, plus a closed-form analytic gradient for the
#     gravity equality constraint (see full_aod_diag/gravity_tariff.jl).
#   - sequential-profiled: an envelope-theorem gradient holding the sequential
#     loop's frozen (F, umat, p) fixed (per the existing grad_R_theta pattern
#     in sequential_gravity/run_profiled_D10_methodB.jl), NOT a fresh
#     ForwardDiff pass through the stateful sequential moments closure --
#     doing that would differentiate through the sequential iterations
#     themselves, which the spec explicitly forbids.
# This file only owns: KNITRO registration, the cache, and dispatching to
# whichever closures the caller supplied.
# ============================================================================

"""
    outer_loop_cached(obj, m::FreeParamMap, θ_lb_full, θ_ub_full, θ_init_full;
                       obj_grad_fn!, div_grad_fn!, gravity_grad_fn!=nothing,
                       gravity_value_fn=nothing, use_cache=true, outer_loop_opt)

- `obj`            : a PsiObjectiveBundleImplicit-shaped bundle whose
                      `moments!` computes K/G from the FULL θ vector (unchanged
                      from today's `inner_loop_internal` contract).
- `m`               : FreeParamMap for this method's θ layout.
- `obj_grad_fn!(g_free, x_free)`      : fills the free-only OBJECTIVE gradient.
- `div_grad_fn!(g_free, x_free, θ_full, inner_x)` : fills the free-only
                      divergence-BUDGET-constraint gradient (envelope theorem;
                      may use obj's just-solved inner state).
- `gravity_value_fn(θ_full)::Float64` / `gravity_grad_fn!(g_free, x_free)`
                      : OPTIONAL second outer equality constraint (full-A's
                      exact gravity condition). Leave both `nothing` for the
                      sequential method (gravity enforced inside the inner
                      solve; no separate outer constraint — per spec §8).

Returns (κ_min_or_objective, θ_min_full, x_min_free, nStatus, cache, wall,
opt_err, outer_iters, outer_fc).
"""
function outer_loop_cached(obj, m::FreeParamMap, θ_lb_full::AbstractVector, θ_ub_full::AbstractVector,
        θ_init_full::AbstractVector; obj_grad_fn!, div_grad_fn!,
        gravity_value_fn = nothing, gravity_grad_fn! = nothing,
        use_cache::Bool = true, outer_loop_opt::AbstractString)

    has_gravity = gravity_value_fn !== nothing
    (has_gravity == (gravity_grad_fn! !== nothing)) || error("outer_loop_cached: gravity_value_fn and gravity_grad_fn! must both be provided or both be nothing")

    x_lo, x_hi = pack_bounds_free(θ_lb_full, θ_ub_full, m)
    x_init = pack_free(θ_init_full, m)
    nf = n_free(m)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, nf)
    KNITRO.KN_set_var_lobnds_all(kc, x_lo)
    KNITRO.KN_set_var_upbnds_all(kc, x_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, x_init)

    ncon = 1 + (has_gravity ? 1 : 0)
    cIndices = KNITRO.KN_add_cons(kc, ncon)
    KNITRO.KN_set_con_eqbnds(kc, 1, [cIndices[1]], [0.0])
    if has_gravity
        KNITRO.KN_set_con_eqbnds(kc, 1, [cIndices[2]], [0.0])
    end

    cache = OuterEvalCache(nf, ncon; use_cache = use_cache)

    theta_buf = Vector{Float64}(undef, m.l_full)

    div_g = zeros(nf)
    grav_g = has_gravity ? zeros(nf) : Float64[]

    function compute_constr_values(x_free)
        reconstruct_full!(theta_buf, x_free, m)
        _, inner_x, nStatus, solved, hit, warm = ensure_inner!(cache, obj, x_free, theta_buf)
        cbuf = zeros(ncon)
        obj(inner_x, constr = @view(cbuf[1:1]))
        if has_gravity
            cbuf[2] = gravity_value_fn(theta_buf)
        end
        return cache.objSol, cbuf, nStatus, solved, hit, warm, inner_x
    end

    function compute_grad(x_free, theta_full, inner_x)
        div_gr, computed = ensure_grad!(cache, (g, xf) -> div_grad_fn!(g, xf, theta_full, inner_x), x_free)
        og = zeros(nf); obj_grad_fn!(og, x_free)
        if has_gravity
            gravity_grad_fn!(grav_g, x_free)
            return og, div_gr, grav_g, computed
        else
            return og, div_gr, nothing, computed
        end
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        t0 = time()
        x_free = evalRequest.x
        objSol, cbuf, nStatus, solved, hit, warm, _ = compute_constr_values(x_free)
        evalResult.obj[1] = -objSol
        evalResult.c .= cbuf
        if abs(objSol) == 1e10
            evalResult.c .= 1e9
        end
        log_row!(cache, "F", x_free, solved, false, warm, time() - t0, nStatus)
        return 0
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        t0 = time()
        x_free = evalRequest.x
        reconstruct_full!(theta_buf, x_free, m)
        _, inner_x, nStatus, solved, hit, warm = ensure_inner!(cache, obj, x_free, theta_buf)
        og, dgr, ggr, computed = compute_grad(x_free, copy(theta_buf), inner_x)
        evalResult.objGrad .= -1.0 .* og
        evalResult.jac[1:nf] .= dgr
        if has_gravity
            evalResult.jac[nf+1:2nf] .= ggr
        end
        log_row!(cache, "G", x_free, solved, computed, warm, time() - t0, nStatus)
        return 0
    end

    function cb_FG!(kc2, cb, evalRequest, evalResult, userParams)
        t0 = time()
        x_free = evalRequest.x
        objSol, cbuf, nStatus, solved, hit, warm, inner_x = compute_constr_values(x_free)
        evalResult.obj[1] = -objSol
        evalResult.c .= cbuf
        og, dgr, ggr, computed = compute_grad(x_free, copy(theta_buf), inner_x)
        evalResult.objGrad .= -1.0 .* og
        evalResult.jac[1:nf] .= dgr
        if has_gravity
            evalResult.jac[nf+1:2nf] .= ggr
        end
        if abs(objSol) == 1e10
            evalResult.c .= 1e9
        end
        log_row!(cache, "FG", x_free, solved, computed, warm, time() - t0, nStatus)
        return 0
    end

    jacIndexCons = has_gravity ? vcat(fill(cIndices[1], nf), fill(cIndices[2], nf)) : fill(cIndices[1], nf)
    jacIndexVars = has_gravity ? vcat(xIndices, xIndices) : xIndices

    if KNITRO.KN_get_int_param(kc, "eval_fcga") == 1
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_FG!)
    else
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
        KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = jacIndexCons, jacIndexVars = jacIndexVars)
    end
    KNITRO.KN_set_cb_user_params(kc, cb, nothing)

    _t = time()
    KNITRO.KN_solve(kc)
    wall = time() - _t
    nStatus, objv, x_min, lambda_ = KNITRO.KN_get_solution(kc)
    θ_min_full = reconstruct_full(x_min, m)
    opt_err = begin
        v = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_opt_error(kc, v); Float64(v[])
    end
    outer_iters = begin
        n = Ref{Cint}(0); KNITRO.KN_get_number_iters(kc, n); Int(n[])
    end
    outer_fc = begin
        n = Ref{Cint}(0); KNITRO.KN_get_number_FC_evals(kc, n); Int(n[])
    end
    KNITRO.KN_free(kc)

    return (objective = objv, θ_min_full = θ_min_full, x_min_free = x_min, nStatus = nStatus,
            cache = cache, wall = wall, opt_err = opt_err, outer_iters = outer_iters, outer_fc = outer_fc)
end
