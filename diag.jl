# Diagnostic: (1) exact moment & parameter layout + redundancy checks;
#             (2) λ_dd and analytical max bound vs the 0.303 upper bound;
#             (3) proper flat profile of a representative outer FG evaluation.

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2
import Profile
include("setup/include_setup.jl"); include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl"); include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl"); include("lfd/include_lfd.jl"); include("misc/include_misc.jl")
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=0,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=1,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
θ=copy(θ_initial); D=so.D; σ=params.σHat; bi=params.baseIndex

obj = PsiObjectiveBundleImplicit(δ=1.0,find_smallest=true,γ=γ,(moments!)=EK_moments!,moments_jacobian! = EK_moments_Jacobian!,d=nTotalMoments,outer_constr_index=outer_constr_index,inequality_index=inequality_index,complement_index=complement_index,l=length(θ),U=U,N=params.Jac_W,lower_limit=-50,outer_loop_opt="csw_outer_loop_settings_cluster.opt",inner_loop_opt="ek_inner_loop_options.opt")

println("DG ============ PARAMETERS (θ, length=$(length(θ))) ============")
println("DG idx : value            role")
lbls = String[]
push!(lbls, "μ (Frechet dispersion, free)")
push!(lbls, "σ (elasticity, FIXED)")
for d in 1:D; push!(lbls, "γ_θ[$d] (baseline aux, free)"); end
push!(lbls, "γ'_θ[baseIndex] (counterfactual aux, free)")
for idx in 1:D^2
    o = ((idx-1) % D) + 1; d = ((idx-1) ÷ D) + 1
    push!(lbls, "A[o=$o,d=$d]" * (o==1 ? " (FIXED=1, normalization)" : " (free)"))
end
for i in 1:length(θ); println("DG $(lpad(i,3)) : $(rpad(round(θ[i];digits=5),16)) $(lbls[i])"); end

println("DG ============ MOMENTS (d=$nTotalMoments) ============")
# fill H at θ_initial
obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj,obj.H), θ, obj.U, obj)
G = obj.H[:, 3:end]                      # W x d moment matrix
cInd = D^2; dInd = D^2 + D
colmean = vec(mean(G; dims=1)); colstd = vec(std(G; dims=1))
println("DG col : mean            std             label")
for c in 1:nTotalMoments
    lbl = if c <= D^2
        o = ((c-1) % D)+1; d = ((c-1) ÷ D)+1; "trade-share moment (o=$o -> d=$d)"
    elseif c <= D^2+D
        "BASELINE price-index moment (d=$(c-D^2))"
    else
        d2 = c-(D^2+D); (d2==bi ? "COUNTERFACTUAL price-index (d=baseIndex=$bi)" : "counterfactual placeholder (d=$d2, autarky: unused)")
    end
    println("DG $(lpad(c,3)) : $(rpad(round(colmean[c];sigdigits=4),16)) $(rpad(round(colstd[c];sigdigits=4),16)) $lbl")
end

println("DG ============ REDUNDANCY CHECKS ============")
# (A) data trade shares sum to 1 over origins for each destination?
P = reshape(γ.P, (D,D))            # P[?]: layout used in code is P[d1]=P[d+(o-1)*D]; reshape col-major -> P[?]
lam = reshape(γ.P,(D,D))'          # lambda[o,d] as in code (lambda=reshape(P,(D,D))')
for d in 1:D
    s = sum(lam[:,d])
    println("DG  sum_o data_share[o->d=$d] = ", round(s; digits=6))
end
# (B) baseline price-index moment == sum of the D trade-share moments for that d ?
for d in 1:D
    tscols = [d + (o-1)*D for o in 1:D]
    lhs = G[:, cInd+d]; rhs = vec(sum(G[:, tscols]; dims=2))
    println("DG  price-index col $(cInd+d) vs sum(trade-share cols $tscols):  max|diff| = ", round(maximum(abs.(lhs .- rhs)); sigdigits=3))
end
# (C) which moment columns are identically zero (degenerate)?
zerocols = [c for c in 1:nTotalMoments if maximum(abs.(G[:,c])) < 1e-12]
println("DG  identically-zero moment columns: ", zerocols)
println("DG  (prep flagged moments_without_var; std==0 columns: ", [c for c in 1:nTotalMoments if colstd[c] < 1e-12], ")")

println("DG ============ ANALYTICAL MAX BOUND ============")
println("DG σ = ", σ, "   1/(σ-1) = ", round(1/(σ-1); digits=5))
for d in 1:D
    ldd = lam[d,d]
    println("DG  d=$d: λ_dd = ", round(ldd;digits=6), "   λ_dd^{1/(σ-1)} = ", round(ldd^(1/(σ-1));digits=6), "   1-λ_dd^{1/(σ-1)} = ", round(1-ldd^(1/(σ-1));digits=6))
end
ldd_bi = lam[bi,bi]
println("DG  baseIndex d=$bi: λ_dd=", round(ldd_bi;digits=6),
        "  =>  λ_dd^{1/(σ-1)} = ", round(ldd_bi^(1/(σ-1));digits=6),
        "   (compare to κ_upper=0.303)")

if get(ENV,"DG_PROFILE","0") == "1"
    println("DG ============ FLAT PROFILE (representative outer FG eval) ============")
    function outer_eval_once(obj, θ)
        objSol,x,ns = CS.inner_loop_internal(obj, θ)
        g = zeros(length(θ)); jl = (nTotalMoments-outer_constr_index+2)*length(θ)
        jac = zeros(jl); c = zeros(nTotalMoments-outer_constr_index+2)
        obj(x, g, θ, constr=c, jac=jac); return objSol
    end
    outer_eval_once(obj,θ); outer_eval_once(obj,θ)   # warmup/compile
    Profile.clear(); Profile.init(n=Int(2e7), delay=0.0005)
    Profile.@profile (for _ in 1:40; outer_eval_once(obj,θ); end)
    open("scratch_exp/results/profile_flat.txt","w") do io
        Profile.print(io; format=:flat, sortedby=:count, mincount=8)
    end
    println("DG wrote scratch_exp/results/profile_flat.txt")
end
println("DG DONE")
