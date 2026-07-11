# KNITRO callback to evaluate objective and constraints
function callbackEval_and_ConsF_outer!(kc, cb, evalRequest, evalResult, userParams)
  try
    obj = userParams
    θ = evalRequest.x

    objSol, x, nStatus = inner_loop_internal(obj, θ)


    evalResult.obj[1] = -objSol

    @info "DIAG ConsF" length(evalResult.c) length(evalResult.obj) obj.d obj.outer_constr_index
    obj(x, constr = evalResult.c)

    if abs(objSol) == 1e+10
        evalResult.c .= 1e+9
    end

    return 0
  catch e
    println(stderr, "=== DIAG: exception in callbackEval_and_ConsF_outer! ===")
    showerror(stderr, e, catch_backtrace()); println(stderr)
    rethrow(e)
  end
end

# KNITRO callback to evaluate gradient of objective and Jacobian of constraints
function callbackEval_and_ConsG_outer!(kc, cb, evalRequest, evalResult, userParams)
  try
    obj = userParams
    θ = evalRequest.x
    objSol, x, nStatus = inner_loop_internal(obj, θ)

    @info "DIAG ConsG" length(evalResult.objGrad) length(evalResult.jac) length(θ) length(x) obj.d obj.outer_constr_index obj.l
    obj(x, evalResult.objGrad, θ, jac = evalResult.jac)
    evalResult.objGrad .*= -1.0

    return 0
  catch e
    println(stderr, "=== DIAG: exception in callbackEval_and_ConsG_outer! ===")
    showerror(stderr, e, catch_backtrace()); println(stderr)
    rethrow(e)
  end
end

# KNITRO callback to evaluate objective, gradient of objective, and Jacobian of constraints
function callbackEval_and_ConsFG_outer!(kc, cb, evalRequest, evalResult, userParams)
  try
    obj = userParams
    θ = evalRequest.x
    objSol, x, nStatus = inner_loop_internal(obj, θ)
    evalResult.obj[1] = -objSol

    @info "DIAG ConsFG" length(evalResult.c) length(evalResult.objGrad) length(evalResult.jac) obj.d obj.outer_constr_index
    obj(x, evalResult.objGrad, θ, constr = evalResult.c, jac = evalResult.jac)
    evalResult.objGrad .*= - 1.0

    if abs(objSol) == 1e+10
        evalResult.c .= 1e+9
    end

    return 0
  catch e
    println(stderr, "=== DIAG: exception in callbackEval_and_ConsFG_outer! ===")
    showerror(stderr, e, catch_backtrace()); println(stderr)
    rethrow(e)
  end
end

# Solve outer program using KNITRO
function outer_loop(obj::ObjectiveBundle, θ_lb, θ_ub, θ_init)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, length(θ_init))

    KNITRO.KN_set_var_lobnds_all(kc, θ_lb) # changed to _all
    KNITRO.KN_set_var_upbnds_all(kc, θ_ub) #
    KNITRO.KN_set_var_primal_init_values_all(kc, θ_init) # changed to _all

    cIndices = outer_loop_constraints!(kc, obj)

    if KNITRO.KN_get_int_param(kc, "eval_fcga") == 1

        # KNITRO is configured to evaluate the objective function and gradient in one function call
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, callbackEval_and_ConsFG_outer!)

    else

        # KNITRO is configured to evaluate the objective function and gradient in seperate calls
        # This is better for finite differences gradient calculations
        #original
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, callbackEval_and_ConsF_outer!)
     
        KNITRO.KN_set_cb_grad(kc, cb, callbackEval_and_ConsG_outer!,
            jacIndexCons = repeat(cIndices, inner=length(xIndices)),
            jacIndexVars = repeat(xIndices, outer=length(cIndices)))

    end

    KNITRO.KN_set_cb_user_params(kc, cb, obj)

  
    KNITRO.KN_solve(kc)

  
    nStatus, κ_min, θ_min, lambda_ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    if !obj.find_smallest
        κ_min *= -1.0
    end

    return (κ_min, θ_min, nStatus)

end

function outer_loop(obj::ObjectiveBundleConditional, θ_lb, θ_ub, θ_init)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, length(θ_init))

    KNITRO.KN_set_var_lobnds(kc, θ_lb)
    KNITRO.KN_set_var_upbnds(kc, θ_ub)
    KNITRO.KN_set_var_primal_init_values(kc, θ_init)

    cIndices, jacIndexCons, jacIndexVars = outer_loop_constraints!(kc, obj)

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, callbackEval_and_ConsF_outer!)

    KNITRO.KN_set_cb_grad(kc, cb, callbackEval_and_ConsG_outer!,
        nV = length(xIndices),
        objGradIndexVars = xIndices,
        jacIndexCons = jacIndexCons,
        jacIndexVars = jacIndexVars)

    KNITRO.KN_set_cb_user_params(kc, cb, obj)

    KNITRO.KN_solve(kc)

    nStatus, κ_min, θ_min, lambda_ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    if !obj.find_smallest
        κ_min *= -1.0
    end

    return (κ_min, θ_min, nStatus)

end

# Set up constraints for outer optimization
function outer_loop_constraints!(kc, obj::Union{PsiObjectiveBundleExplicit, PsiObjectiveBundleDelta, KLObjectiveBundleExplicit, KLObjectiveBundleDelta})

    cIndices = KNITRO.KN_add_cons(kc, obj.d - obj.outer_constr_index + 1)
    #KNITRO.KN_set_con_eqbnds(kc, zeros(obj.d - obj.outer_constr_index + 1))
    #changed to fit 
    KNITRO.KN_set_con_eqbnds(kc, obj.d - obj.outer_constr_index +1,cIndices[1:obj.d - obj.outer_constr_index + 1], zeros(obj.d - obj.outer_constr_index + 1))
    return cIndices

end

function outer_loop_constraints!(kc, obj::Union{PsiObjectiveBundleImplicit, KLObjectiveBundleImplicit})

    cIndices = KNITRO.KN_add_cons(kc, obj.d - obj.outer_constr_index + 2)
    
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1e10 * obj.δ)
    
    #original
    #KNITRO.KN_set_con_eqbnds(kc, cIndices[2:obj.d - obj.outer_constr_index + 2], zeros(obj.d - obj.outer_constr_index + 1))
    
    #changed to fit 
    KNITRO.KN_set_con_eqbnds(kc, obj.d - obj.outer_constr_index +1,cIndices[2:obj.d - obj.outer_constr_index + 2], zeros(obj.d - obj.outer_constr_index + 1))
    
    return cIndices

end

function outer_loop_constraints!(kc, obj::Union{PsiObjectiveBundleConditionalExplicit, PsiObjectiveBundleConditionalDelta, KLObjectiveBundleConditionalExplicit, KLObjectiveBundleConditionalDelta})

    cIndices = KNITRO.KN_add_cons(kc, obj.d - obj.outer_constr_index + 1)
    KNITRO.KN_set_con_eqbnds(kc, zeros(obj.d - obj.outer_constr_index + 1))

    jacIndexCons = repeat(cIndices, inner = obj.dϑ)
    jacIndexVars = Int64[]
    for i in 1:obj.ncov
        jacIndexVars = vcat(jacIndexVars, repeat(make_active_indices(obj, i), outer = obj.dc))
    end
    jacIndexVars .-= 1
    jacIndexVars = convert(Array{Int32}, jacIndexVars)

    return cIndices, jacIndexCons, jacIndexVars

end

function outer_loop_constraints!(kc, obj::Union{PsiObjectiveBundleConditionalImplicit, KLObjectiveBundleConditionalImplicit})

    cIndices = KNITRO.KN_add_cons(kc, obj.d - obj.outer_constr_index + 1 + obj.ncov)
    KNITRO.KN_set_con_upbnds(kc, cIndices[1:obj.ncov], 1e10 * obj.δ * ones(obj.ncov))
    KNITRO.KN_set_con_eqbnds(kc, cIndices[obj.ncov+1:obj.d - obj.outer_constr_index + 1 + obj.ncov], zeros(obj.d - obj.outer_constr_index + 1))

    jacIndexCons = repeat(cIndices, inner = obj.dϑ)
    jacIndexVars = Int64[]
    for i in 1:obj.ncov
        jacIndexVars = vcat(jacIndexVars, make_active_indices(obj, i))
    end
    for i in 1:obj.ncov
        jacIndexVars = vcat(jacIndexVars, repeat(make_active_indices(obj, i), outer = obj.dc))
    end
    jacIndexVars .-= 1
    jacIndexVars = convert(Array{Int32}, jacIndexVars)

    return cIndices, jacIndexCons, jacIndexVars

end

# Gradients and Jacobians for outer optimization
# Calculates the Jacobian of all moments with respect to θ at first N draws
function calculate_jac_θ!(obj::ObjectiveBundle, θ)

    # default to autodiff if no analytical jacobian is passed
    if obj.moments_jacobian! == error
        calculate_jac_θ_autodiff!(obj, θ)
    else
        obj.moments_jacobian!(@view(obj.jac_h[1:obj.N, 1, :]), select_jac_g_from_jac_h(obj, obj.jac_h), θ, @view(obj.U[1:obj.N, :]), obj)
    end

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

# Calculates the gradient of implicit-dependence counterfactual k with respect to θ
function calculate_grad_k!(g, obj::ObjectiveBundleConditional, θ)

    # default to autodiff if no analytical jacobian is passed
    if obj.moments_jacobian! == error
        calculate_grad_k_autodiff!(g, obj, θ)
    else
        @unpack ncov, dd, dϑ = obj
        jac_kk = zeros(2, dϑ)
        jac_GG = zeros(2, 1 + dd, dϑ)
        for i in 1:ncov
            active_indices = make_active_indices(obj, i)
            obj.moments_jacobian!(jac_kk, jac_GG, θ, @view(obj.U[1:2, :]), obj, index = i)
            g[active_indices] .= jac_kk[1, :]
        end
    end

end

function calculate_grad_k_autodiff!(g, obj::ObjectiveBundleConditional, θ)

   f = θ -> begin
       kk = zeros(eltype(θ), 2, obj.ncov)
       HH = zeros(eltype(θ), 2, size(obj.H)[2])
       obj.moments!(kk, select_G_from_H(obj, HH), θ, @view(obj.U[1:2, :]), obj)
       return kk[1, 1]
   end

   ForwardDiff.gradient!(g, f, θ, ForwardDiff.GradientConfig(f, θ), Val{true}())

end

# Calculates the Jacobian of moments with respect to a subvector of θ indexed by
# active_indices and a subset of moments indexed by moment_indices at first N draws
function calculate_jac_θ_subvec!(obj::ObjectiveBundleConditional, θ, i, nonzero_moments)

    # default to autodiff if no analytical jacobian is passed
    if obj.moments_jacobian! == error
        calculate_jac_θ_subvec_autodiff!(obj, θ, i, nonzero_moments)
    else
        obj.moments_jacobian!(@view(obj.jac_h[1:obj.N, 1, :]), @view(obj.jac_h[1:obj.N, 2:end, :]), θ, @view(obj.U[1:obj.N, :]), obj, index = i)
        moment_indices, moment_indices_jac_h, slack_indices_jac_h = make_active_moments(obj, i, nonzero_moments)
        @views obj.jac_h[:, slack_indices_jac_h, :] .= 0.0
    end

end

function calculate_jac_θ_subvec_autodiff!(obj::ObjectiveBundleConditional, θ, i, nonzero_moments)

    @unpack ncov, var_index, dm, dc, outer_constr_index = obj

    active_indices = make_active_indices(obj, i)
    moment_indices, moment_indices_jac_h, slack_indices_jac_h = make_active_moments(obj, i, nonzero_moments)

    f = (h, ϑ) -> begin
        θθ = zeros(eltype(ϑ), obj.l)
        @views θθ[active_indices] .= ϑ
        @views θθ[setdiff(1:obj.l, active_indices)] .= θ[setdiff(1:obj.l, active_indices)]
        obj.moments!(@view(obj.HH[:, 1:obj.ncov]), select_G_from_H(obj, obj.HH), θθ, @view(obj.U[1:obj.N, :]), obj, index = i)
        h .= @view(obj.HH[1:obj.N, moment_indices])
    end

    @views ForwardDiff.jacobian!(
        reshape(obj.jac_h[:, moment_indices_jac_h, :], obj.N * length(moment_indices_jac_h), length(active_indices)),
        f,
        obj.H[1:obj.N, moment_indices],
        θ[active_indices],
        ForwardDiff.JacobianConfig(f, obj.H[1:obj.N, moment_indices], θ[active_indices]), Val{true}())

    @views obj.jac_h[:, slack_indices_jac_h, :] .= 0.0

end

# Make active parameter indices for covariate value i
function make_active_indices(obj::ObjectiveBundleConditional, i)

    return vcat(findall(obj.var_index .== 0), findall(obj.var_index .== i))

end

# Make active H and jacobian indices for covariate value i
function make_active_moments(obj::KLObjectiveBundleConditional, i, nonzero_moments)

    @unpack ncov, d, dm, dc, outer_constr_index = obj

    moment_indices = vcat(i, nonzero_moments .+ ((i-1) * dm + ncov))
    if outer_constr_index <= d
        moment_indices = vcat(moment_indices, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc)
    end

    moment_indices_jac_h = vcat(1, nonzero_moments .+ 1, dm+2:dm+1+dc)
    slack_indices_jac_h = setdiff(1:dm, nonzero_moments) .+ 1

    return moment_indices, moment_indices_jac_h, slack_indices_jac_h

end

function make_active_moments(obj::PsiObjectiveBundleConditional, i, nonzero_moments)

    @unpack ncov, d, dm, dc, outer_constr_index = obj

    moment_indices = vcat(i, nonzero_moments .+ ((i-1) * (dm + 1) + ncov + 1))
    if outer_constr_index <= d
        moment_indices = vcat(moment_indices, ncov*(2+dm)+(i-1)*dc+1:ncov*(2+dm)+i*dc)
    end

    moment_indices_jac_h = vcat(1, nonzero_moments .+ 1, dm+2:dm+1+dc)
    slack_indices_jac_h = setdiff(1:dm, nonzero_moments) .+ 1

    return moment_indices, moment_indices_jac_h, slack_indices_jac_h

end