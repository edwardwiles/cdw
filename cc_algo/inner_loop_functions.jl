# ---- lightweight instrumentation counters (reset by outer_loop) ----
# EXP: count inner solves and how many return infeasible per outer solve (Task B2).
const INNER_SOLVE_COUNT  = Ref(0)
const INNER_INFEAS_COUNT = Ref(0)
const INNER_ITERS_TOTAL  = Ref(0)

# EXP: KNITRO.jl v1.2.1 exposes only the low-level (kc, out_ptr)->status getters for these;
# wrap them to return the value directly.
function _kn_num_iters(kc)
    n = Ref{Cint}(0); KNITRO.KN_get_number_iters(kc, n); return Int(n[])
end
function _kn_num_fc(kc)
    n = Ref{Cint}(0); KNITRO.KN_get_number_FC_evals(kc, n); return Int(n[])
end
function _kn_solve_time(kc)
    t = Ref{Cdouble}(0.0); KNITRO.KN_get_solve_time_real(kc, t); return Float64(t[])
end
function _kn_feas_err(kc)
    v = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_feas_error(kc, v); return Float64(v[])
end
function _kn_opt_err(kc)
    v = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_opt_error(kc, v); return Float64(v[])
end

# KNITRO callback to evaluate objective and gradient
function callbackEvalFG_inner!(kc, cb, evalRequest, evalResult, userParams)

    obj = userParams
    x = evalRequest.x
    evalResult.obj[1] = obj(x, evalResult.objGrad)

    return 0

end

function callbackEvalH_inner!(kc, cb, evalRequest, evalResult, userParams)

    obj = userParams
    x = evalRequest.x
    obj(x, h = evalResult.hess)

    return 0

end

# Solve inner program using KNITRO
function inner_loop_KNITRO(obj)

    kc = KNITRO.KN_new()


    

    KNITRO.KN_add_vars(kc, inner_loop_number_variables(obj))

            # set lower bounds for inner optimization
    KNITRO.KN_set_var_lobnds_all(kc, inner_loop_lower_bounds(obj)) #I changed this 13072022

            # set initial values
    KNITRO.KN_set_var_primal_init_values_all(kc, inner_loop_initial_values(obj))#I changed this 13072022
    
            
#=
    # set lower bounds for inner optimization
    KNITRO.KN_set_var_lobnds(kc, inner_loop_lower_bounds(obj))

    # set initial values
    KNITRO.KN_set_var_primal_init_values(kc, inner_loop_initial_values(obj))
=#
    # set objective function and gradient
    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], callbackEvalFG_inner!)
    KNITRO.KN_set_cb_user_params(kc, cb, obj)

    # set options
    KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

    # set Hessian, if required
    if KNITRO.KN_get_int_param(kc, "hessopt") == 1
        inner_loop_hessian(kc, cb, obj)
    end

    # set complementarity constraints, if required
    if obj.complement_index != [0 0]
        inner_loop_complementarity_constraints(kc, obj)
    end

    # run
    KNITRO.KN_solve(kc)
    nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
    INNER_ITERS_TOTAL[] += _kn_num_iters(kc)   # EXP instrumentation
    KNITRO.KN_free(kc)

    return nSTatus, objSol, x, lambda_

end

# Refine KNITRO solution
function inner_loop_KNITRO_refine(obj)

    nSTatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    iter = 1
    while iter <= 3 && nSTatus ∈ [-100, -101, -102, -103] && objSol >= obj.lower_limit
        obj.x .= x
        if obj.use_cached_x
            nSTatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)
        else
            obj.use_cached_x = true
            nSTatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)
            obj.use_cached_x = false
        end
        iter += 1
    end

    return nSTatus, objSol, x, lambda_

end

# Number of variables
inner_loop_number_variables(obj::PsiObjectiveBundleExplicit) = 1 + obj.outer_constr_index
inner_loop_number_variables(obj::PsiObjectiveBundleImplicit) = obj.outer_constr_index
inner_loop_number_variables(obj::PsiObjectiveBundleDelta)    = obj.outer_constr_index

inner_loop_number_variables(obj::KLObjectiveBundleExplicit) = obj.outer_constr_index
inner_loop_number_variables(obj::KLObjectiveBundleImplicit) = obj.outer_constr_index - 1
inner_loop_number_variables(obj::KLObjectiveBundleDelta)    = obj.outer_constr_index - 1

# Lower bounds
inner_loop_lower_bounds(obj::PsiObjectiveBundleExplicit) = vcat(obj.η_min, -KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::PsiObjectiveBundleImplicit) = vcat(-KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::PsiObjectiveBundleDelta)    = vcat(-KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])

inner_loop_lower_bounds(obj::KLObjectiveBundleExplicit) = vcat(obj.η_min, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::KLObjectiveBundleImplicit) = [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1]
inner_loop_lower_bounds(obj::KLObjectiveBundleDelta)    = [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1]

# Initial values
inner_loop_initial_values(obj::PsiObjectiveBundleExplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : vcat(obj.η_min, zeros(obj.outer_constr_index))
inner_loop_initial_values(obj::PsiObjectiveBundleImplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index)
inner_loop_initial_values(obj::PsiObjectiveBundleDelta)    = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index)

inner_loop_initial_values(obj::KLObjectiveBundleExplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : vcat(obj.η_min, zeros(obj.outer_constr_index - 1))
inner_loop_initial_values(obj::KLObjectiveBundleImplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index - 1)
inner_loop_initial_values(obj::KLObjectiveBundleDelta)    = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index - 1)

# Hessian
function inner_loop_hessian(kc, cb, obj::ObjectiveBundle)
    KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalH_inner!)
end

# Complementarity constraints
function inner_loop_complementarity_constraints(kc, obj::PsiObjectiveBundleExplicit)
    KNITRO.KN_set_compcons(kc, zeros(Int32, length(obj.complement_index[:, 1])), Int32.(obj.complement_index[:, 1] .+ 1), Int32.(obj.complement_index[:, 2] .+ 1))
end

function inner_loop_complementarity_constraints(kc, obj::Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta, KLObjectiveBundleExplicit})
    KNITRO.KN_set_compcons(kc, zeros(Int32, length(obj.complement_index[:, 1])), Int32.(obj.complement_index[:, 1] .+ 0), Int32.(obj.complement_index[:, 2] .+ 0))
end

function inner_loop_complementarity_constraints(kc, obj::Union{KLObjectiveBundleImplicit, KLObjectiveBundleDelta})
    KNITRO.KN_set_compcons(kc, zeros(Int32, length(obj.complement_index[:, 1])), Int32.(obj.complement_index[:, 1] .- 1), Int32.(obj.complement_index[:, 2] .- 1))
end

# Call KNITRO solver for end use, correcting the sign if smallest
function inner_loop(obj, θ)

    (val, x, nStatus) = inner_loop_internal(obj, θ)

    if obj.find_smallest == true
        val *= -1.0
    end

    return val, x, nStatus

end

# Call KNITRO solver to pass to outer optimization
function inner_loop_internal(obj::PsiObjectiveBundleExplicit, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, 2] .= 1.0

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO_refine(obj)

    if nStatus == 0 || (nStatus ∈ [-100, -101, -103] && objSol >= obj.lower_limit)
        obj.x .= x
        return objSol, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end

function inner_loop_internal(obj::PsiObjectiveBundleImplicit, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1,1] * (-1.0)^obj.find_smallest

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    # EXP instrumentation: count inner solves and infeasible/failed returns (Task B2).
    # (Replaces the per-solve `@show nStatus/objSol` debug prints, which were pure I/O drag.)
    INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        INNER_INFEAS_COUNT[] += 1
    end

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end

function inner_loop_internal(obj::PsiObjectiveBundleDelta, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, 2] .= 1.0

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return objSol, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end

function inner_loop_internal(obj::KLObjectiveBundleExplicit, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1]), select_G_from_H(obj, obj.H), θ, obj.U, obj)

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    if nStatus == 0 || (nStatus ∈ [-100, -101, -103] && objSol >= obj.lower_limit)
        obj.x .= x
        return objSol, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end

function inner_loop_internal(obj::KLObjectiveBundleImplicit, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H_save = obj.H[1,1] * (-1.0)^obj.find_smallest

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end

function inner_loop_internal(obj::KLObjectiveBundleDelta, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1]), select_G_from_H(obj, obj.H), θ, obj.U, obj)

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return objSol, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end
