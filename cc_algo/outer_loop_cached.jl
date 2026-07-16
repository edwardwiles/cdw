# ============================================================================
# Generic (method-agnostic) cached outer-loop driver.
#
# Ties together FreeParamMap (differentiate/optimize only the free
# coordinates) and OuterEvalCache (solve the inner CC problem at most once per
# unique free-outer point) around the EXISTING, unmodified `inner_loop_internal`
# and `(Q::PsiObjectiveBundleImplicit)` callable's constraint-VALUE path (cheap,
# reused as-is — only the GRADIENT/Jacobian path is replaced). Constraint-value
# semantics are copy-exact from `cc_algo/outer_loop_functions.jl`'s own
# `outer_loop_constraints!` for `PsiObjectiveBundleImplicit`: constraint row 1
# (`-f*1e10 <= 1e10*δ`, i.e. Delta(theta)<=delta) is an INEQUALITY, and every
# subsequent row (gravity, if present) is an EQUALITY at 0 — both rows come
# from ONE call to `obj(inner_x, constr=buf)` (it already fills both; there is
# no separate gravity-value computation needed).
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
                       has_gravity=false, use_cache=true, outer_loop_opt)

- `obj`            : a PsiObjectiveBundleImplicit-shaped bundle whose
                      `moments!` computes K/G from the FULL θ vector (unchanged
                      from today's `inner_loop_internal` contract), with
                      `obj.outer_constr_index == obj.d` when `has_gravity` (one
                      extra outer-loop moment = gravity, matching how
                      `master_prepare_cc.jl` wires `nOuterLoopMoments`).
- `m`               : FreeParamMap for this method's θ layout.
- `obj_grad_fn!(g_free, x_free)`      : fills the free-only OBJECTIVE gradient.
- `div_grad_fn!(g_free, x_free, θ_full, inner_x)` : fills the free-only
                      divergence-BUDGET-constraint gradient (envelope theorem;
                      may use obj's just-solved inner state).
- `gravity_grad_fn!(g_free, x_free)`  : REQUIRED iff `has_gravity`. Analytic
                      free-only gradient of the TARGET gravity function
                      (e.g. section 9's `g_gravity=(1/N_obs)Σq_tilde·logA`).
                      `nothing`/unused for the sequential method (gravity
                      enforced inside the inner solve; no separate outer
                      constraint — per spec §8).
- `gravity_value_scale` : `obj`'s own gravity column (from `obj.moments!`, via
                      `newGravityMoment!`) may be a DIFFERENT (but
                      proportional) quantity than whatever `gravity_grad_fn!`
                      is the gradient of (e.g. production's raw, unnormalized
                      `sumGrav` vs. section 9's `-sumGrav/N_obs`). Set this to
                      the constant that converts one into the other so the
                      constraint VALUE and GRADIENT KNITRO sees are for the
                      exact same function — a mismatch here reads to KNITRO as
                      genuine infeasibility, not merely slow convergence.

Returns a NamedTuple (objective, θ_min_full, x_min_free, nStatus, cache, wall,
opt_err, outer_iters, outer_fc).
"""
function outer_loop_cached(obj, m::FreeParamMap, θ_lb_full::AbstractVector, θ_ub_full::AbstractVector,
        θ_init_full::AbstractVector; obj_grad_fn!, div_grad_fn!,
        gravity_grad_fn! = nothing, has_gravity::Bool = false, gravity_value_scale::Float64 = 1.0,
        use_cache::Bool = true, outer_loop_opt::AbstractString,
        var_scales::Union{Nothing,AbstractVector{Float64}} = nothing)

    !has_gravity || gravity_grad_fn! !== nothing || error("outer_loop_cached: has_gravity=true requires gravity_grad_fn!")

    x_lo, x_hi = pack_bounds_free(θ_lb_full, θ_ub_full, m)
    x_init = pack_free(θ_init_full, m)
    nf = n_free(m)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, nf)
    KNITRO.KN_set_var_lobnds_all(kc, x_lo)
    KNITRO.KN_set_var_upbnds_all(kc, x_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, x_init)
    # Optional per-variable scaling (additive -- default nothing means unchanged behavior).
    # KN_set_var_scalings_all(kc, scaleFactors, scaleCenters) tells KNITRO's INTERNAL linear
    # algebra to work with x'_i = (x_i - center_i)/scaleFactors_i, so a caller-supplied
    # scaleFactors[i] ~ 1/|d(constraint)/dx_i| makes every free coordinate's EFFECTIVE gradient
    # magnitude ~O(1) in KNITRO's own scaled space, regardless of how differently-sized the raw
    # gradient components are. Precedent: full_aod_diag/solve_scaled.jl (a full-A_od outer-loop
    # variant, different method but the same KNITRO scaling API), used there for a different
    # (ill-conditioning) reason. Introduced here to address a severe cross-variable gradient-scale
    # mismatch found at D=20 real data (gamma'_focal's own constraint-gradient component ~1e8,
    # vs the entire Acol block ~10-11500) that appeared to prevent Acol from being explored at all.
    var_scales === nothing || KNITRO.KN_set_var_scalings_all(kc, collect(Float64, var_scales), zeros(nf))

    ncon = 1 + (has_gravity ? 1 : 0)
    cIndices = KNITRO.KN_add_cons(kc, ncon)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1e10 * obj.δ)         # Delta(theta) <= delta (production's exact scaling)
    if has_gravity
        KNITRO.KN_set_con_eqbnds(kc, 1, [cIndices[2]], [0.0])       # gravity == 0
    end

    cache = OuterEvalCache(nf, ncon; use_cache = use_cache)
    theta_buf = Vector{Float64}(undef, m.l_full)
    grav_g = has_gravity ? zeros(nf) : Float64[]

    function compute_constr_values(x_free)
        reconstruct_full!(theta_buf, x_free, m)
        _, inner_x, nStatus, solved, hit, warm = ensure_inner!(cache, obj, x_free, theta_buf)
        cbuf = zeros(ncon)
        obj(inner_x, constr = @view(cbuf[1:ncon]))     # fills BOTH rows in one call; no separate gravity-value fn needed
        if has_gravity
            # `obj`'s own gravity column is whatever raw quantity `obj.moments!` computed (e.g.
            # production's un-normalized, unflipped sumGrav) -- rescale it to match whatever
            # function `gravity_grad_fn!` is the gradient OF, so constraint VALUE and GRADIENT
            # are for the exact same target (both ==0 either way, but KNITRO needs them
            # consistent, not just individually zero-seeking -- a value/gradient mismatch here
            # reads as genuine infeasibility to KNITRO, not just slow convergence).
            cbuf[2] *= gravity_value_scale
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
    feas_err = begin
        v = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_feas_error(kc, v); Float64(v[])
    end
    outer_iters = begin
        n = Ref{Cint}(0); KNITRO.KN_get_number_iters(kc, n); Int(n[])
    end
    outer_fc = begin
        n = Ref{Cint}(0); KNITRO.KN_get_number_FC_evals(kc, n); Int(n[])
    end
    KNITRO.KN_free(kc)

    return (objective = objv, θ_min_full = θ_min_full, x_min_free = x_min, nStatus = nStatus,
            cache = cache, wall = wall, opt_err = opt_err, feas_err = feas_err,
            outer_iters = outer_iters, outer_fc = outer_fc)
end
