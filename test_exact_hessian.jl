# A2: exact OUTER Hessian experiment (implicit autarky case).
#
# The outer objective KNITRO minimizes is f(θ) = -(-1)^find_smallest * k(θ), where k is the
# closed-form counterfactual gains-from-trade (does NOT depend on the inner solution). We can
# therefore supply its EXACT Hessian ∇²k via ForwardDiff. The CONSTRAINTS (divergence budget +
# moment conditions) depend on the inner solution x*(θ); their exact curvature needs a 2nd-order
# implicit-function-theorem through the inner KNITRO solve, which is disproportionate to derive.
# So this test supplies the exact objective-Hessian and ZERO constraint curvature (hessopt=exact),
# and compares convergence/bounds against BFGS(auto) and product_findiff(4).
#
# It duplicates the minimal outer_loop so it does not touch the live source.

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra
using Plots, JLD2
import KNITRO

include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl")
include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl")
include("lfd/include_lfd.jl")
include("misc/include_misc.jl")
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity

params = (
    server=1, user=2, fakeData=1, DFake=4, seedFakeData=889,
    counterType=1, counterExplicit=0, θHat=0, σHat=2.5, baseIndex=2, W=8000,
    seedU=888, importanceSampling=0, importanceSamplingFactor=2, stratifiedSampling=0,
    IndMomentOrder=5, θConstant=0, gravMoment=0, localGravityMoment=0, localGravityCrossMoment=0,
    GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0,
    NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05,
    δGridType=0, δ_ref=1, refIndex1=1, OuterLoop=1, UoModel=1, use_Jacobian=1,
    calc_δ_star_initial=1, Jac_W=8000, theta_init=0, runLFD=1, runLFDCounterFactual=1,
)
setup_output = master_setup(params)
useParams = (; params..., D=setup_output.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(useParams)
prestep_output = master_prestep(setup_output.data, setup_output.counters, useParams)
prep_output = master_prepare_cc(setup_output.data, setup_output.counters, prestep_output, useParams)
@unpack θ_initial, θ_initial_low, θ_initial_up, U, γ, δ_grid, outer_constr_index, inequality_index, nTotalMoments, complement_index = prep_output

# outer θ bounds — replicate ccOuter
θ = copy(θ_initial); l = length(θ); D = setup_output.D
θ_lower = (θ_initial.*0.0001)[:]; θ_upper = (θ_initial.*10000)[:]
θ_lower[2] = θ_initial[2]; θ_upper[2] = θ_initial[2]
θ_upper[1] = 1/(θ_initial[2]-1)-0.001; θ_lower[1] = 0.001
θ_upper[1] = min(θ_upper[1], 1/(θ_initial[2]-1))
Aod_offset = 3 + D
for i in 1:D
    θ_upper[Aod_offset+1+D*(i-1)] = θ_initial[Aod_offset+1+D*(i-1)]
    θ_lower[Aod_offset+1+D*(i-1)] = θ_initial[Aod_offset+1+D*(i-1)]
end
for i in 1:l
    if θ_initial[i] < 0; θ_upper[i] = -10000*θ_initial[i]; θ_lower[i] = 10000*θ_initial[i]; end
end

build_obj(find_smallest) = PsiObjectiveBundleImplicit(
    δ = 1.0, find_smallest = find_smallest, γ = γ, (moments!) = EK_moments!,
    moments_jacobian! = EK_moments_Jacobian!, d = nTotalMoments,
    outer_constr_index = outer_constr_index, inequality_index = inequality_index,
    complement_index = complement_index, l = l, U = U, N = params.Jac_W, lower_limit = -50,
    outer_loop_opt = "csw_outer_loop_settings_cluster.opt", inner_loop_opt = "ek_inner_loop_options.opt")

# exact ∇²k(θ) via ForwardDiff (objective Hessian). k on a 2-row subsample (k doesn't depend on U).
function objective_hessian!(Hmat, obj, θ)
    f = t -> begin
        kk = zeros(eltype(t),2,1); HH = zeros(eltype(t),2,size(obj.H,2))
        obj.moments!(kk, CS.select_G_from_H(obj,HH), t, @view(obj.U[1:2,:]), obj)
        return kk[1]
    end
    ForwardDiff.hessian!(Hmat, f, θ)
end

# outer solve with exact objective-Hessian callback (hessopt must be set to exact=1 in the opt file
# passed in). Returns (κ, θ*, status, iters, feas, opt, time).
function outer_solve_exacthess(obj, θ_lb, θ_ub, θ0, optfile)
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, optfile)
    xIndices = KNITRO.KN_add_vars(kc, length(θ0))
    KNITRO.KN_set_var_lobnds_all(kc, θ_lb); KNITRO.KN_set_var_upbnds_all(kc, θ_ub)
    KNITRO.KN_set_var_primal_init_values_all(kc, θ0)
    cIndices = CS.outer_loop_constraints!(kc, obj)
    # objective+grad+jac callback (mirror callbackEval_and_ConsFG_outer!)
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, function (kc,cb,er,eres,up)
            θl = er.x
            objSol, x, ns = CS.inner_loop_internal(obj, θl)
            eres.obj[1] = -objSol
            obj(x, eres.objGrad, θl, constr = eres.c, jac = eres.jac)
            eres.objGrad .*= -1.0
            if abs(objSol) == 1e10; eres.c .= 1e9; end
            return 0
        end)
    KNITRO.KN_set_cb_user_params(kc, cb, obj)
    # exact Hessian callback: sigma * ∇²(-(-1)^find_smallest k), zero constraint curvature
    Hmat = zeros(length(θ0), length(θ0)); sgn = (-1.0)^obj.find_smallest
    KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, function (kc,cb,er,eres,up)
            objective_hessian!(Hmat, obj, er.x)      # ∇²k
            k = 1
            @inbounds for i in 1:length(θ0), j in i:length(θ0)
                eres.hess[k] = er.sigma * (-sgn) * Hmat[i,j]   # f = -sgn*k  => ∇²f = -sgn ∇²k
                k += 1
            end
            return 0
        end)
    t0 = time(); KNITRO.KN_solve(kc); dt = time()-t0
    ns, κ, θstar, lam = KNITRO.KN_get_solution(kc)
    ni=Ref{Cint}(0); KNITRO.KN_get_number_iters(kc,ni)
    fe=Ref{Cdouble}(0.0); KNITRO.KN_get_abs_feas_error(kc,fe)
    oe=Ref{Cdouble}(0.0); KNITRO.KN_get_abs_opt_error(kc,oe)
    if !obj.find_smallest; κ *= -1.0; end
    KNITRO.KN_free(kc)
    return (κ, θstar, ns, Int(ni[]), Float64(fe[]), Float64(oe[]), dt)
end

optfile = get(ENV, "EXP_HESS_OPT", "scratch_exp/opt_exacthess.opt")
println("HX using optfile=", optfile)
for (nm, fs, θ0) in [("upper", false, θ_initial_up), ("lower", true, θ_initial_low)]
    obj = build_obj(fs)
    κ, θstar, ns, ni, fe, oe, dt = outer_solve_exacthess(obj, θ_lower, θ_upper, θ0, optfile)
    println("HX EXACT_HESS $nm  kappa=", round(κ; digits=6), " status=", ns, " iters=", ni,
            " feas_err=", round(fe;sigdigits=3), " opt_err=", round(oe;sigdigits=3), " time_s=", round(dt;digits=1))
end
println("HX DONE.")
