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


    # set objective function and gradient

   
    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], callbackEvalFG_inner!)

   
    KNITRO.KN_set_cb_user_params(kc, cb, obj)

    # set options
    KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

    # set Hessian, if required
    if KNITRO.KN_get_int_param(kc, "hessopt") == 1
        inner_loop_hessian(kc, cb, obj)
    end

    # run
    KNITRO.KN_solve(kc)
    nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    return nSTatus, objSol, x, lambda_

end

# Number of variables
inner_loop_number_variables(obj::PsiObjectiveBundleExplicit) = 1 + obj.outer_constr_index
inner_loop_number_variables(obj::PsiObjectiveBundleImplicit) = obj.outer_constr_index
inner_loop_number_variables(obj::PsiObjectiveBundleDelta)    = obj.outer_constr_index

inner_loop_number_variables(obj::KLObjectiveBundleExplicit) = obj.outer_constr_index
inner_loop_number_variables(obj::KLObjectiveBundleImplicit) = obj.outer_constr_index - 1
inner_loop_number_variables(obj::KLObjectiveBundleDelta)    = obj.outer_constr_index - 1

inner_loop_number_variables(obj::PsiObjectiveBundleConditionalExplicit) = 2 * obj.ncov + obj.outer_constr_index - 1
inner_loop_number_variables(obj::PsiObjectiveBundleConditionalImplicit) = obj.ncov + obj.outer_constr_index - 1
inner_loop_number_variables(obj::PsiObjectiveBundleConditionalDelta)    = obj.ncov + obj.outer_constr_index - 1

inner_loop_number_variables(obj::KLObjectiveBundleConditionalExplicit) = obj.ncov + obj.outer_constr_index - 1
inner_loop_number_variables(obj::KLObjectiveBundleConditionalImplicit) = obj.outer_constr_index - 1
inner_loop_number_variables(obj::KLObjectiveBundleConditionalDelta)    = obj.outer_constr_index - 1

# Lower bounds
inner_loop_lower_bounds(obj::PsiObjectiveBundleExplicit) = vcat(obj.η_min, -KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::PsiObjectiveBundleImplicit) = vcat(-KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::PsiObjectiveBundleDelta)    = vcat(-KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])

inner_loop_lower_bounds(obj::KLObjectiveBundleExplicit) = vcat(obj.η_min, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::KLObjectiveBundleImplicit) = [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1]
inner_loop_lower_bounds(obj::KLObjectiveBundleDelta)    = [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1]

inner_loop_lower_bounds(obj::PsiObjectiveBundleConditionalExplicit) = vcat(obj.η_min * ones(obj.ncov), -KNITRO.KN_INFINITY * ones(obj.ncov), [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::PsiObjectiveBundleConditionalImplicit) = vcat(-KNITRO.KN_INFINITY * ones(obj.ncov), [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::PsiObjectiveBundleConditionalDelta)    = vcat(-KNITRO.KN_INFINITY * ones(obj.ncov), [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])

inner_loop_lower_bounds(obj::KLObjectiveBundleConditionalExplicit) = vcat(obj.η_min * ones(obj.ncov), [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_lower_bounds(obj::KLObjectiveBundleConditionalImplicit) = [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1]
inner_loop_lower_bounds(obj::KLObjectiveBundleConditionalDelta)    = [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1]

# Initial values
inner_loop_initial_values(obj::PsiObjectiveBundleExplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : vcat(obj.η_min, zeros(obj.outer_constr_index))
inner_loop_initial_values(obj::PsiObjectiveBundleImplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index)
inner_loop_initial_values(obj::PsiObjectiveBundleDelta)    = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index)

inner_loop_initial_values(obj::KLObjectiveBundleExplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : vcat(obj.η_min, zeros(obj.outer_constr_index - 1))
inner_loop_initial_values(obj::KLObjectiveBundleImplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index - 1)
inner_loop_initial_values(obj::KLObjectiveBundleDelta)    = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index - 1)

inner_loop_initial_values(obj::PsiObjectiveBundleConditionalExplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : vcat(obj.η_min * ones(obj.ncov), zeros(obj.ncov + obj.outer_constr_index - 1))
inner_loop_initial_values(obj::PsiObjectiveBundleConditionalImplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.ncov + obj.outer_constr_index - 1)
inner_loop_initial_values(obj::PsiObjectiveBundleConditionalDelta)    = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.ncov + obj.outer_constr_index - 1)

inner_loop_initial_values(obj::KLObjectiveBundleConditionalExplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : vcat(obj.η_min * ones(obj.ncov), zeros(obj.outer_constr_index - 1))
inner_loop_initial_values(obj::KLObjectiveBundleConditionalImplicit) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index - 1)
inner_loop_initial_values(obj::KLObjectiveBundleConditionalDelta)    = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index - 1)

# Hessian
function inner_loop_hessian(kc, cb, obj::ObjectiveBundle)
    KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalH_inner!)
end

function inner_loop_hessian(kc, cb, obj::ObjectiveBundleConditional)
    obj.nnzH == 0 ? hessian_indices!(obj) : nothing
    KNITRO.KN_set_cb_hess(kc, cb, obj.nnzH, callbackEvalH_inner!, hessIndexVars1 = obj.hessIndexVars1, hessIndexVars2 = obj.hessIndexVars2)
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

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

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

function inner_loop_internal(obj::PsiObjectiveBundleConditionalExplicit, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1:obj.ncov]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, obj.ncov + 1:obj.dm + 1:obj.ncov*(obj.dm + 1)] .= 1.0

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    if nStatus == 0 || (nStatus ∈ [-100, -101, -103] && objSol >= obj.lower_limit)
        obj.x .= x
        return objSol, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end

function inner_loop_internal(obj::PsiObjectiveBundleConditionalImplicit, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1:obj.ncov]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, obj.ncov + 1:obj.dm + 1:obj.ncov*(obj.dm + 1)] .= 1.0
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

function inner_loop_internal(obj::PsiObjectiveBundleConditionalDelta, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1:obj.ncov]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, obj.ncov + 1:obj.dm + 1:obj.ncov*(obj.dm + 1)] .= 1.0

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return objSol, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end

function inner_loop_internal(obj::KLObjectiveBundleConditionalExplicit, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1:obj.ncov]), select_G_from_H(obj, obj.H), θ, obj.U, obj)

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    if nStatus == 0 || (nStatus ∈ [-100, -101, -103] && objSol >= obj.lower_limit)
        obj.x .= x
        return objSol, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end

function inner_loop_internal(obj::KLObjectiveBundleConditionalImplicit, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1:obj.ncov]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
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

function inner_loop_internal(obj::KLObjectiveBundleConditionalDelta, θ)

    # update moments
    obj.moments!(@view(obj.H[:, 1:obj.ncov]), select_G_from_H(obj, obj.H), θ, obj.U, obj)

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return objSol, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end

end