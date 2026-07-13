# ============================================================================
# D=10, μ FIXED, full A_od-in-outer-loop method ("the method where you don't
# do sequential linearization but instead just do A_od all in the outer loop
# directly", per the user), with THIS SESSION's improvements: γ_d≡1
# normalization + direct-γ' objective (compare_directgp.jl, generalized to
# D=10). Uses the STANDARD dense-ForwardDiff outer gradient (Method A) —
# unlike run_profiled_D10_methodB.jl, this method's gravity moment IS a
# second outer constraint (outer_constr_index = d, not d+1), which needs the
# ift! total-derivative correction the derivative audit explicitly did not
# re-derive as a scalar/Method-B object, so Method A stays correct here.
#
#   julia --project=. full_aod_diag/run_fullA_D10.jl   (needs KNITRO env)
# ============================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(@__DIR__, "moments_gammanorm.jl"))

const OUT = joinpath(@__DIR__,"out"); isdir(OUT) || mkpath(OUT)
const MAXIT_OPT = joinpath(@__DIR__, "csw_outer_25.opt")
const INNEROPT = joinpath(@__DIR__, "ek_inner.opt")
const D10 = 10

params = (server=1,user=2,fakeData=1,DFake=D10,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,θ_initial_low,θ_initial_up,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
D=so.D; bi=params.baseIndex; σ=params.σHat; μHat=γ.μHat
@assert D == D10
Aod_offset = 3 + D
println(">>> D=$D  l=$(length(θ_initial))  nTotalMoments=$nTotalMoments  outer_constr_index=$outer_constr_index  (extra outer constraints = $(nTotalMoments-outer_constr_index+1), gravity needs ift!)")

θ_lower_G = (θ_initial.*0.0001)[:]; θ_upper_G = (θ_initial.*10000)[:]
θ_lower_G[2]=θ_initial[2]; θ_upper_G[2]=θ_initial[2]
θ_upper_G[1]=1/(σ-1)-0.001; θ_lower_G[1]=0.001; θ_upper_G[1]=min(θ_upper_G[1],1/(σ-1))
# μ FIXED, per user request
θ_lower_G[1] = θ_initial[1]; θ_upper_G[1] = θ_initial[1]
for d in 1:D
    idx = 2+d
    θ_upper_G[idx]=θ_initial[idx]; θ_lower_G[idx]=θ_initial[idx]
end
bounds = theoretical_gammaprime_bounds(γ, σ)
println(">>> theoretical γ'_focal bounds: [$(bounds.γp_lo), $(bounds.γp_hi)]  (κ_max=$(bounds.κ_max))")
θ_lower_G[3+D] = bounds.γp_lo; θ_upper_G[3+D] = bounds.γp_hi
for i in 1:length(θ_initial)
    if θ_initial[i] < 0
        θ_upper_G[i] = -10000*θ_initial[i]; θ_lower_G[i] = 10000*θ_initial[i]
    end
end

θ_initial_up_G = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
θ_initial_low_G = build_theta_gammanorm(θ_initial_low, D, bi, μHat, σ)
for v in (θ_initial_up_G, θ_initial_low_G)
    v[3+D] = clamp(v[3+D], bounds.γp_lo, bounds.γp_hi)
end

nfree = count(i -> θ_lower_G[i] != θ_upper_G[i], 1:length(θ_lower_G))
println(">>> [RUN 2: full-A, MU_FIXED] free params=$nfree"); flush(stdout)

mkobj(find_smallest, θi) = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=find_smallest, γ=γ,
    (moments!)=EK_moments_gammanorm_directgp!, moments_jacobian! = error, d=nTotalMoments,
    outer_constr_index=outer_constr_index, inequality_index=inequality_index,
    complement_index=complement_index, l=length(θi), U=U, N=params.Jac_W,
    lower_limit=-50, use_cached_x=true,
    outer_loop_opt=MAXIT_OPT, inner_loop_opt=INNEROPT)

t0=time()
println(">>> RUN2: solving for max(gamma'_focal) [-> kappa_lower]"); flush(stdout)
gp_max, θ_lo, st_lo, _ = outer_loop(mkobj(false, θ_initial_low_G), θ_lower_G, θ_upper_G, θ_initial_low_G)
println(">>> RUN2: solving for min(gamma'_focal) [-> kappa_upper]"); flush(stdout)
gp_min, θ_up, st_up, _ = outer_loop(mkobj(true, θ_initial_up_G), θ_lower_G, θ_upper_G, θ_initial_up_G)
wall=time()-t0

κ_lo = 1 - gp_max^(σ/(σ-1))
κ_up = 1 - gp_min^(σ/(σ-1))

open(joinpath(OUT,"run2_fullA_D10_summary.txt"),"w") do f
    for ln in (
        "[RUN 2] full-A D=$D  baseIndex=$bi  free=$nfree  MU_FIXED=true  maxit=25",
        @sprintf("gamma_p_max = %.6f  (status=%d) -> kappa_lower = %.6f", gp_max, st_lo, κ_lo),
        @sprintf("gamma_p_min = %.6f  (status=%d) -> kappa_upper = %.6f", gp_min, st_up, κ_up),
        @sprintf("width       = %.6f", κ_up-κ_lo),
        @sprintf("total wall  = %.1fs", wall))
        println(f, ln); println(ln)
    end
end
@save joinpath(OUT,"run2_fullA_D10.jld2") κ_up κ_lo gp_max gp_min θ_up θ_lo st_up st_lo
println("RUN2 DONE")
