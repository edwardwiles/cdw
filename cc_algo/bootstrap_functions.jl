# KNITRO callback to evaluate objective and constraints for outer bootstrap optimization
function callbackEval_and_ConsF_boot!(kc, cb, evalRequest, evalResult, userParams)

    (obj, dπ) = userParams
    θ = evalRequest.x
    val, x, nStatus = inner_loop(obj, θ)

    if abs(val) < 1e+10
        evalResult.obj[1] = -dot_λ_dπ(obj, x, dπ)
        evalResult.c[1] = val * 1e+10
        if length(evalResult.c) > 1
            obj(x, constr = @view(evalResult.c[2:end]))
        end
    else
        evalResult.obj[1] = 1e+10
        evalResult.c[1] = (-1.0)^obj.find_smallest * -1e+9
        if length(evalResult.c) > 1
            @view(evalResult.c[2:end]) .= 1e+9
        end
    end

    return 0

end

# Inner product of lambda multipliers with perturbation direction dπ
function dot_λ_dπ(obj::PsiObjectiveBundleExplicit, x, dπ)
    return dot(@view(x[3:end]), dπ)
end

function dot_λ_dπ(obj::Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta}, x, dπ)
    return dot(@view(x[2:end]), dπ)
end

# KNITRO callback to evaluate gradient of objective and Jacobian of constraints
function callbackEval_and_ConsG_boot!(kc, cb, evalRequest, evalResult, userParams)

    (obj, dπ) = userParams
    θ = evalRequest.x
    val, x, nStatus = inner_loop(obj, θ)

    jac_dot_λ_dπ!(evalResult.objGrad, obj, x, dπ)
    evalResult.objGrad .*= -1.0

    if length(evalResult.jac) == obj.l
        obj(x, evalResult.jac, θ)
    else
        @views obj(x, evalResult.jac[1:obj.l], θ, jac = evalResult.jac[obj.l+1:end])
    end
    evalResult.jac[1:obj.l] .*= (-1.0)^obj.find_smallest * 1e+10

    return 0

end

# Derivative of inner product of lambda multipliers with perturbation direction dπ
function jac_dot_λ_dπ!(g, obj::PsiObjectiveBundleExplicit, x, dπ)

    η = x[1]
    λ = @view(x[3:end])
    ift!(η, λ, obj)
    BLAS.gemv!('T', 1.0, @view(obj.∂x_∂θ[3:end, :]), dπ, 0.0, g)

end

function jac_dot_λ_dπ!(g, obj::Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta}, x, dπ)

    λ = @view(x[2:end])
    ift!(λ, obj)
    BLAS.gemv!('T', 1.0, @view(obj.∂x_∂θ[2:end, :]), dπ, 0.0, g)

end

# Solve outer program using KNITRO
function outer_loop_bootstrap(obj::Union{PsiObjectiveBundleExplicit, PsiObjectiveBundleDelta}, θ_lb, θ_ub, θ_min, dπ, ε)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, length(θ_min))

    KNITRO.KN_set_var_lobnds(kc, θ_lb)
    KNITRO.KN_set_var_upbnds(kc, θ_ub)
    KNITRO.KN_set_var_primal_init_values(kc, θ_min)

    cIndices = outer_loop_constraints_boot!(kc, obj, θ_min)

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, callbackEval_and_ConsF_boot!)
    KNITRO.KN_set_cb_grad(kc, cb, callbackEval_and_ConsG_boot!,
        jacIndexCons = repeat(cIndices, inner=length(xIndices)),
        jacIndexVars = repeat(xIndices, outer=length(cIndices)))

    KNITRO.KN_set_cb_user_params(kc, cb, (obj, dπ))

    KNITRO.KN_solve(kc)
    if KNITRO.KN_get_number_iters(kc) == 0
        KNITRO.KN_solve(kc)
    end

    nStatus, κ_min_boot, θ_min_boot, lambda_ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    if isa(obj, PsiObjectiveBundleExplicit)
        if !obj.find_smallest
            κ_min_boot *= -1.0
        end
    end

    return (κ_min_boot, θ_min_boot, nStatus)

end

function outer_loop_bootstrap(obj::PsiObjectiveBundleImplicit, θ_min, dπ, Λ)

    inner_loop(obj, θ_min)
    c = dot_λ_dπ(obj, obj.x, dπ)
    c *= Λ[1] * 1e10

    return c

end

function outer_loop_constraints_boot!(kc, obj::Union{PsiObjectiveBundleExplicit, PsiObjectiveBundleDelta}, θ_min)

    κ_min = inner_loop(obj, θ_min)[1]

    cIndices = KNITRO.KN_add_cons(kc, 1 + obj.d - obj.outer_constr_index + 1)
    if obj.find_smallest
        KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1e+10 * (κ_min + ε))
    else
        KNITRO.KN_set_con_lobnd(kc, cIndices[1], 1e+10 * (κ_min - ε))
    end
    if length(cIndices) > 1
        KNITRO.KN_set_con_eqbnds(kc, cIndices[2:end], zeros(obj.d - obj.outer_constr_index + 1))
    end
    return cIndices

end

# Variance based on outer optimizer set Θ being a singleton
function variance_singleton(obj, θ, V)

    val, x, nStatus = inner_loop(obj, θ)

    if abs(val) < 1e+10
        f = quadratic_form_variance(obj, x, V)
    else
        f = 1e+10
    end

end

function quadratic_form_variance(obj::PsiObjectiveBundleExplicit, x, V)
    return dot(x[3:end], V * x[3:end])
end

function quadratic_form_variance(obj::Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta}, x, V)
    return dot(x[2:end], V * x[2:end])
end
