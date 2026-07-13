# ================================================================================
# Sensitivity-scaled FULL-A_od solve (test: does the lower bound stop stalling?)
# Identification-neutral fix: give KNITRO per-variable scale factors ≈ 1/‖∂E[g]/∂θ_i‖
# so the ~59× weaker A_od directions are put on the same footing as the structural
# ones in the outer KKT geometry. Everything else identical to the full variant.
# Additive: outer_loop_scaled() copies cc_algo/outer_loop and inserts one scaling call.
# ================================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2
import KNITRO
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
const OUT = joinpath(@__DIR__,"out"); isdir(OUT) || mkpath(OUT)

# ---- outer_loop with user variable scalings (copy of cc_algo/outer_loop_functions.jl:outer_loop) ----
function outer_loop_scaled(obj, θ_lb, θ_ub, θ_init, scaleFactors; opt=joinpath(@__DIR__,"csw_outer.opt"))
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, opt)
    xIndices = KNITRO.KN_add_vars(kc, length(θ_init))
    KNITRO.KN_set_var_lobnds_all(kc, θ_lb)
    KNITRO.KN_set_var_upbnds_all(kc, θ_ub)
    KNITRO.KN_set_var_primal_init_values_all(kc, θ_init)
    # *** the one addition: per-variable scale factors (centers = 0) ***
    KNITRO.KN_set_var_scalings_all(kc, collect(Float64, scaleFactors), zeros(length(θ_init)))
    cIndices = CS.outer_loop_constraints!(kc, obj)
    if KNITRO.KN_get_int_param(kc, "eval_fcga") == 1
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, CS.callbackEval_and_ConsFG_outer!)
    else
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, CS.callbackEval_and_ConsF_outer!)
        KNITRO.KN_set_cb_grad(kc, cb, CS.callbackEval_and_ConsG_outer!,
            jacIndexCons = repeat(cIndices, inner=length(xIndices)),
            jacIndexVars = repeat(xIndices, outer=length(cIndices)))
    end
    KNITRO.KN_set_cb_user_params(kc, cb, obj)
    CS.INNER_SOLVE_COUNT[]=0; CS.INNER_INFEAS_COUNT[]=0; CS.INNER_ITERS_TOTAL[]=0
    _t=time(); KNITRO.KN_solve(kc)
    nStatus, κ, θ_sol, lambda_ = KNITRO.KN_get_solution(kc)
    println(">>> SCALED_OUTER find_smallest=", obj.find_smallest, " status=", nStatus,
            " iters=", CS._kn_num_iters(kc), " feas=", round(CS._kn_feas_err(kc);sigdigits=3),
            " opt=", round(CS._kn_opt_err(kc);sigdigits=3), " time=", round(CS._kn_solve_time(kc);digits=1),
            " inner=", CS.INNER_SOLVE_COUNT[], " infeas=", CS.INNER_INFEAS_COUNT[],
            " obj=", round(κ;digits=6)); flush(stdout)
    !obj.find_smallest && (κ *= -1.0)
    KNITRO.KN_free(kc)
    return κ, θ_sol, nStatus
end

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,θ_initial_low,θ_initial_up,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
D=so.D; bi=params.baseIndex; σ=params.σHat; l=length(θ_initial)

# bounds: full variant, exactly as ccOuter.jl
θ_lower=(θ_initial.*0.0001)[:]; θ_upper=(θ_initial.*10000)[:]
θ_lower[2]=θ_initial[2]; θ_upper[2]=θ_initial[2]
θ_upper[1]=min(1/(σ-1)-0.001, 1/(σ-1)); θ_lower[1]=0.001
Aod_offset=3+D
for i in 1:D; θ_upper[Aod_offset+1+D*(i-1)]=θ_initial[Aod_offset+1+D*(i-1)]; θ_lower[Aod_offset+1+D*(i-1)]=θ_initial[Aod_offset+1+D*(i-1)]; end
for i in 1:l; if θ_initial[i]<0; θ_upper[i]=-10000*θ_initial[i]; θ_lower[i]=10000*θ_initial[i]; end; end
fixedmask = [θ_lower[i]==θ_upper[i] for i in 1:l]

obj0 = PsiObjectiveBundleImplicit(δ=1.0,find_smallest=true,γ=γ,(moments!)=EK_moments!,moments_jacobian! = error,
    d=nTotalMoments,outer_constr_index=outer_constr_index,inequality_index=inequality_index,
    complement_index=complement_index,l=l,U=U,N=params.Jac_W,lower_limit=-50,use_cached_x=true,
    outer_loop_opt=joinpath(@__DIR__,"csw_outer.opt"),inner_loop_opt=joinpath(@__DIR__,"ek_inner.opt"))

# ---- scale factors = 1/‖moment-Jacobian column‖ at θ_initial (1.0 for fixed vars) ----
CS.calculate_jac_θ_autodiff!(obj0, copy(θ_initial))
J = reshape(mean(@view(obj0.jac_h[1:obj0.N, 3:2+nTotalMoments, :]), dims=1), nTotalMoments, l)
colnorm = [norm(J[:,i]) for i in 1:l]
scaleFactors = [ (fixedmask[i] || colnorm[i] < 1e-10) ? 1.0 : 1.0/colnorm[i] for i in 1:l ]
println("scaleFactors (free): struct≈", round.(extrema([scaleFactors[i] for i in 1:l if !fixedmask[i] && colnorm[i]>0.5]);sigdigits=3),
        "  A≈", round.(extrema([scaleFactors[i] for i in 1:l if !fixedmask[i] && colnorm[i]<0.5]);sigdigits=3)); flush(stdout)

mkobj(fs)=PsiObjectiveBundleImplicit(δ=1.0,find_smallest=fs,γ=γ,(moments!)=EK_moments!,moments_jacobian! = error,
    d=nTotalMoments,outer_constr_index=outer_constr_index,inequality_index=inequality_index,
    complement_index=complement_index,l=l,U=U,N=params.Jac_W,lower_limit=-50,use_cached_x=true,
    outer_loop_opt=joinpath(@__DIR__,"csw_outer.opt"),inner_loop_opt=joinpath(@__DIR__,"ek_inner.opt"))

println(">>> SCALED UPPER"); κ_up,θ_up,st_up = outer_loop_scaled(mkobj(false), θ_lower,θ_upper,θ_initial_up, scaleFactors)
println(">>> SCALED LOWER"); κ_lo,θ_lo,st_lo = outer_loop_scaled(mkobj(true),  θ_lower,θ_upper,θ_initial_low, scaleFactors)
open(joinpath(OUT,"solve_scaled_summary.txt"),"w") do f
    for ln in ("variant=full+sensitivity-scaling  D=$D",
               @sprintf("kappa_lower = %.6f  (status=%d)   [unscaled full stalled at 0.018113]", κ_lo, st_lo),
               @sprintf("kappa_upper = %.6f  (status=%d)   [unscaled full 0.154415]", κ_up, st_up),
               @sprintf("width       = %.6f", κ_up-κ_lo))
        println(f, ln); println(ln)
    end
end
@save joinpath(OUT,"solve_scaled.jld2") κ_up κ_lo θ_up θ_lo st_up st_lo scaleFactors
println("DONE solve_scaled")
