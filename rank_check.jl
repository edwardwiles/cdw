using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2
include("setup/include_setup.jl"); include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl"); include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl"); include("lfd/include_lfd.jl"); include("misc/include_misc.jl")
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=0,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
θ=copy(θ_initial); D=so.D
obj = PsiObjectiveBundleImplicit(δ=1.0,find_smallest=true,γ=γ,(moments!)=EK_moments!,moments_jacobian! = EK_moments_Jacobian!,d=nTotalMoments,outer_constr_index=outer_constr_index,inequality_index=inequality_index,complement_index=complement_index,l=length(θ),U=U,N=params.Jac_W,lower_limit=-50,outer_loop_opt="csw_outer_loop_settings_cluster.opt",inner_loop_opt="ek_inner_loop_options.opt")
obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj,obj.H), θ, obj.U, obj)
G = obj.H[:, 3:end]
println("RK nTotalMoments = ", nTotalMoments, "   size(G) = ", size(G))
r_all = rank(G; rtol=1e-9)
println("RK rank(full 17-moment G) = ", r_all, "  (=17 means all independent, none droppable)")
# trade-share block per d: cols [d, d+D, d+2D, d+3D]
for d in 1:D
    cols = [d + (o-1)*D for o in 1:D]
    println("RK  d=$d trade-share cols $cols : rank = ", rank(G[:, cols]; rtol=1e-9), " / $D")
end
# 16 trade shares together
println("RK 16 trade-share cols rank = ", rank(G[:, 1:D^2]; rtol=1e-9), " / 16")
# singular values of full G (smallest few) to see conditioning
sv = svd(G).S
println("RK smallest 4 singular values of G: ", round.(sv[end-3:end]; sigdigits=3))
println("RK DONE")
