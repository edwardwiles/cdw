# KNITRO callback to evaluate objective and constraints
function callbackEval_and_ConsF_outer!(kc, cb, evalRequest, evalResult, userParams)

    obj = userParams
    θ = evalRequest.x
    objSol, x, nStatus = inner_loop_internal(obj, θ)
    evalResult.obj[1] = -objSol

    obj(x, constr = evalResult.c)

    if abs(objSol) == 1e+10
        evalResult.c .= 1e+9
    end

    return 0

end

# KNITRO callback to evaluate gradient of objective and Jacobian of constraints
function callbackEval_and_ConsG_outer!(kc, cb, evalRequest, evalResult, userParams)

    obj = userParams
    θ = evalRequest.x
    objSol, x, nStatus = inner_loop_internal(obj, θ)

    obj(x, evalResult.objGrad, θ, jac = evalResult.jac)
    evalResult.objGrad .*= -1.0

    return 0

end

# KNITRO callback to evaluate objective, gradient of objective, and Jacobian of constraints
function callbackEval_and_ConsFG_outer!(kc, cb, evalRequest, evalResult, userParams)

    obj = userParams
    θ = evalRequest.x
    objSol, x, nStatus = inner_loop_internal(obj, θ)
    evalResult.obj[1] = -objSol

    obj(x, evalResult.objGrad, θ, constr = evalResult.c, jac = evalResult.jac)
    evalResult.objGrad .*= - 1.0

    if abs(objSol) == 1e+10
        evalResult.c .= 1e+9
    end

    return 0

end

# Solve outer program using KNITRO
function outer_loop(obj::ObjectiveBundle, θ_lb, θ_ub, θ_init; output = false)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, length(θ_init))

    KNITRO.KN_set_var_lobnds_all(kc, θ_lb) # changed to _all
    KNITRO.KN_set_var_upbnds_all(kc, θ_ub) #
    KNITRO.KN_set_var_primal_init_values_all(kc, θ_init) # changed to _all
    
    #=
    KNITRO.KN_set_var_lobnds(kc, θ_lb)
    KNITRO.KN_set_var_upbnds(kc, θ_ub)
    KNITRO.KN_set_var_primal_init_values(kc, θ_init)
``=#
    cIndices = outer_loop_constraints!(kc, obj)

    if KNITRO.KN_get_int_param(kc, "eval_fcga") == 1

        # KNITRO is configured to evaluate the objective function and gradient in one function call
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, callbackEval_and_ConsFG_outer!)

    else

        # KNITRO is configured to evaluate the objective function and gradient in seperate calls
        # This is better for finite differences gradient calculations
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, callbackEval_and_ConsF_outer!)
        KNITRO.KN_set_cb_grad(kc, cb, callbackEval_and_ConsG_outer!,
            jacIndexCons = repeat(cIndices, inner=length(xIndices)),
            jacIndexVars = repeat(xIndices, outer=length(cIndices)))

    end

    KNITRO.KN_set_cb_user_params(kc, cb, obj)

    # EXP instrumentation: reset inner-solve counters for this outer solve (Task A4/B2)
    INNER_SOLVE_COUNT[] = 0; INNER_INFEAS_COUNT[] = 0; INNER_ITERS_TOTAL[] = 0
    _t_outer = time()

    KNITRO.KN_solve(kc)

    nStatus, κ_min, θ_min, lambda_ = KNITRO.KN_get_solution(kc)

    # EXP instrumentation: one-line summary of this outer solve
    let oiters = _kn_num_iters(kc),
        ofc    = _kn_num_fc(kc),
        otime  = _kn_solve_time(kc),
        feas   = _kn_feas_err(kc),
        opt    = _kn_opt_err(kc)
        println(">>> OUTER_SOLVE find_smallest=", obj.find_smallest,
                " status=", nStatus,
                " outer_iters=", oiters,
                " outer_FCevals=", ofc,
                " feas_err=", round(feas; sigdigits=3),
                " opt_err=", round(opt; sigdigits=3),
                " knitro_time_s=", round(otime; digits=2),
                " wall_s=", round(time()-_t_outer; digits=2),
                " inner_solves=", INNER_SOLVE_COUNT[],
                " inner_infeas=", INNER_INFEAS_COUNT[],
                " inner_iters_total=", INNER_ITERS_TOTAL[],
                " obj=", round(κ_min; digits=6)); flush(stdout)
    end

    if !obj.find_smallest
        κ_min *= -1.0
    end

    if output

        runtime = KNITRO.KN_get_solve_time_real(kc)
        feas_error = KNITRO.KN_get_abs_feas_error(kc)
        opt_error = KNITRO.KN_get_abs_opt_error(kc)
        KNITRO.KN_free(kc)
        return (κ_min, θ_min, nStatus, runtime, feas_error, opt_error, lambda_)

    else

        KNITRO.KN_free(kc)
        return (κ_min, θ_min, nStatus, lambda_)

    end

end

# Solve outer program using KNITRO with multi start
function outer_loop_multi(obj::ObjectiveBundle, θ_lb, θ_ub, θ_init, maxsolves, startptrange)

    if maxsolves == 1

        (κ_min, θ_min, nStatus, lambda_) = outer_loop(obj, θ_lb, θ_ub, θ_init)
        return (κ_min, θ_min, nStatus, lambda_)

    else

        Θ = zeros(obj.l, maxsolves)
        κ = zeros(maxsolves)
        Λ = zeros(outer_loop_lambda_length(obj), maxsolves)
        flag = zeros(Int, maxsolves)

        Random.seed!(1234)
        for i in 1:maxsolves
            obj.x .= NaN # Reset inner loop starting value so reproducible if using warm start
            θ_init_rand = θ_init .+ (i > 1) .* (rand(obj.l) * 2.0 .- 1.0) .* startptrange
            θ_init_rand = max.(min.(θ_init_rand, θ_ub), θ_lb)
            (κ[i], Θ[:, i], flag[i], runtime, feas_error, opt_error, Λ[:, i]) = outer_loop(obj, θ_lb, θ_ub, θ_init_rand, output = true)
            println("Iter:  ", i, "   Flag:  ", (flag[i] == 0 ? "   0" : flag[i]), "   Val:  ", round(κ[i], digits = 5), "   Time:  ", round(runtime, digits = 2) , "   Opt Err:  ", round(opt_error, sigdigits = 5), "   Feas Err:  ", round(feas_error, sigdigits = 5)); flush(stdout)
        end
        println(); flush(stdout)

        if !obj.find_smallest
            κ .*= -1.0
        end

        # locally optimal solution
        if sum(flag .== 0) > 0
            ix = collect(1:maxsolves)[(κ .== minimum(κ[flag .== 0]))][1]
            (κ_min, θ_min, nStatus, lambda_) = (κ[ix], Θ[:, ix], flag[ix], Λ[:, ix])
        # feasible, near-optimal solution
        elseif sum(in([-100, -101, -103]).(flag)) > 0
            ix = collect(1:maxsolves)[(κ .== minimum(κ[in([-100, -101, -103]).(flag)]))][1]
            (κ_min, θ_min, nStatus, lambda_) = (κ[ix], Θ[:, ix], flag[ix], Λ[:, ix])
        # feasible, but reached iteration limit
        elseif sum(in([-400, -401, -402]).(flag)) > 0
            ix = collect(1:maxsolves)[(κ .== minimum(κ[in([-400, -401, -402]).(flag)]))][1]
            (κ_min, θ_min, nStatus, lambda_) = (κ[ix], Θ[:, ix], flag[ix], Λ[:, ix])
        # fail to find any feasible point
        else
            κ_min = 1e10
            θ_min = θ_init
            nStatus = 999
            lambda_ = NaN * zeros(outer_loop_lambda_length(obj))
        end

        if !obj.find_smallest
            κ_min *= -1.0
        end

        return (κ_min, θ_min, nStatus, lambda_)

    end
end

function outer_loop_lambda_length(obj::Union{PsiObjectiveBundleExplicit, PsiObjectiveBundleDelta, KLObjectiveBundleExplicit, KLObjectiveBundleDelta})
    return obj.l + obj.d - obj.outer_constr_index + 1
end

function outer_loop_lambda_length(obj::Union{PsiObjectiveBundleImplicit, KLObjectiveBundleImplicit})
    return obj.l + obj.d - obj.outer_constr_index + 2
end

# Set up constraints for outer optimization
function outer_loop_constraints!(kc, obj::Union{PsiObjectiveBundleExplicit, PsiObjectiveBundleDelta, KLObjectiveBundleExplicit, KLObjectiveBundleDelta})

    cIndices = KNITRO.KN_add_cons(kc, obj.d - obj.outer_constr_index + 1)
    #changed to fit 
    KNITRO.KN_set_con_eqbnds(kc, obj.d - obj.outer_constr_index +1,cIndices[1:obj.d - obj.outer_constr_index + 1], zeros(obj.d - obj.outer_constr_index + 1))
 
    #KNITRO.KN_set_con_eqbnds(kc, zeros(obj.d - obj.outer_constr_index + 1))
    return cIndices

end

function outer_loop_constraints!(kc, obj::Union{PsiObjectiveBundleImplicit, KLObjectiveBundleImplicit})

    cIndices = KNITRO.KN_add_cons(kc, obj.d - obj.outer_constr_index + 2)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1e10 * obj.δ)
    
    #changed to fit 
    KNITRO.KN_set_con_eqbnds(kc, obj.d - obj.outer_constr_index +1,cIndices[2:obj.d - obj.outer_constr_index + 2], zeros(obj.d - obj.outer_constr_index + 1))
    
    
    #KNITRO.KN_set_con_eqbnds(kc, cIndices[2:obj.d - obj.outer_constr_index + 2], zeros(obj.d - obj.outer_constr_index + 1))
    return cIndices

end

# Gradients and Jacobians for outer optimization
# Calculates the Jacobian of all moments with respect to θ at first N draws
function calculate_jac_θ!(obj::ObjectiveBundle, θ)

    # ADDITIVE (jac_h audit, diag/fullA-d4-exact-jach-audit): if this object was constructed with
    # needs_outer_moment_jacobian=false (PsiObjectiveBundleImplicit only, see
    # cc_algo/PsiObjectiveBundle.jl), jac_h is a 0x0x0 placeholder -- error clearly here rather than
    # let the autodiff/analytic branch below write into (or bounds-error on) an empty array.
    if hasproperty(obj, :needs_outer_moment_jacobian) && !obj.needs_outer_moment_jacobian
        error("calculate_jac_θ!: this ObjectiveBundle was constructed with needs_outer_moment_jacobian=false " *
              "-- the dense outer-moment Jacobian (jac_h) was never allocated, so the legacy " *
              "ForwardDiff-through-moments!/analytic-Jacobian outer-gradient path (calculate_jac_θ!, " *
              "calculate_jac_θ_autodiff!, ift!) cannot be used on this object. Construct a separate " *
              "object with needs_outer_moment_jacobian=true (the default) if this path is needed, or " *
              "use the free-only envelope-gradient / L_fix-incremental gradient path instead (see " *
              "cc_algo/outer_loop_cached.jl, full_aod_diag/d4_exact/lfix_incremental.jl). " *
              "See docs/fullA_jach_audit.md.")
    end

    JAC_H_POPULATE_COUNT[] += 1
    t0 = time()

    # default to autodiff if no analytical jacobian is passed
    if obj.moments_jacobian! == error
        calculate_jac_θ_autodiff!(obj, θ)
    else
        obj.moments_jacobian!(@view(obj.jac_h[1:obj.N, 1, :]), select_jac_g_from_jac_h(obj, obj.jac_h), θ, @view(obj.U[1:obj.N, :]), obj)
    end

    JAC_H_POPULATE_TIME[] += time() - t0

end

function calculate_jac_θ_autodiff!(obj::ObjectiveBundle, θ)

    f = (H, θ) -> begin
        obj.moments!(@view(H[:, 1]), select_G_from_H(obj, H), θ, @view(obj.U[1:obj.N, :]), obj)
    end

    @views ForwardDiff.jacobian!(
        reshape(obj.jac_h[1:obj.N, :, :], length(1:obj.N) * size(obj.H, 2), obj.l),
        f,
        obj.H_copy[1:obj.N, :],
        θ,
        ForwardDiff.JacobianConfig(f, obj.H_copy[1:obj.N, :], θ), Val{true}())

end

# Calculates the gradient of implicit-dependence counterfactual k with respect to θ
function calculate_grad_k!(g, obj::ObjectiveBundle, θ)

    # default to autodiff if no analytical jacobian is passed
    if obj.moments_jacobian! == error
        calculate_grad_k_autodiff!(g, obj, θ)
    else
        jac_kk = zeros(2, obj.l)
        jac_HH = zeros(2, size(obj.H)[2], obj.l)
        obj.moments_jacobian!(jac_kk, select_jac_g_from_jac_h(obj, jac_HH), θ, @view(obj.U[1:2, :]), obj)
        g .= jac_kk[1, :]
    end

end

function calculate_grad_k_autodiff!(g, obj::ObjectiveBundle, θ)

   f = θ -> begin
       kk = zeros(eltype(θ), 2, 1)
       HH = zeros(eltype(θ), 2, size(obj.H)[2])
       obj.moments!(kk, select_G_from_H(obj, HH), θ, @view(obj.U[1:2, :]), obj)
       return kk[1]
   end

   ForwardDiff.gradient!(g, f, θ, ForwardDiff.GradientConfig(f, θ), Val{true}())

end
