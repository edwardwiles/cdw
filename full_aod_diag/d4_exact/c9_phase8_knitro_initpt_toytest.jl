# Minimal isolated test: does KN_set_var_primal_init_values_all actually take effect for a
# bound-constrained-only (no general constraints) problem with algorithm=3, dense-gradient
# objective callback registered via KN_add_eval_callback(kc, true, Int32[], cb_F!)?
using KNITRO, Printf
include(joinpath(@__DIR__, "..", "..", "cc_algo", "knitro_compat.jl"))

n = 5
x0 = [1.234, -2.345, 3.456, -4.567, 5.0]  # deliberately NOT at any bound (bounds will be +-8)

kc = KNITRO.KN_new()
KNITRO.KN_set_param_by_name(kc, "algorithm", 3)
KNITRO.KN_set_param_by_name(kc, "outlev", 3)
KNITRO.KN_set_param_by_name(kc, "maxit", 5)
xIndices = KNITRO.KN_add_vars(kc, n)
rc1 = KNITRO.KN_set_var_lobnds_all(kc, fill(-8.0, n))
rc2 = KNITRO.KN_set_var_upbnds_all(kc, fill(8.0, n))
rc3 = KNITRO.KN_set_var_primal_init_values_all(kc, x0)
@printf("return codes: lobnds=%d upbnds=%d priminit=%d\n", rc1, rc2, rc3)

seen_x = Ref{Vector{Float64}}(Float64[])
function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    seen_x[] = copy(x)
    println("cb_F! called with x = ", x)
    evalResult.obj[1] = sum(x .^ 2)
    return 0
end
function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    evalResult.objGrad .= 2 .* x
    return 0
end
cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
KNITRO.KN_set_cb_grad(kc, cb, cb_G!)

KNITRO.KN_solve(kc)
println("First x seen by cb_F!: ", seen_x[])
println("norm(first_x - x0) = ", sum(abs2, seen_x[] .- x0)^0.5)
KNITRO.KN_free(kc)
