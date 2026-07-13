# ================================================================================
# Additive outer-solve driver: solve the κ upper & lower bounds for either the
# FULL A_od variant (RED=0) or the REDUCED variant (RED=1, only the target-
# destination column A[·,baseIndex] free). Replicates ccOuter.jl's bound logic
# exactly, but with a configurable free-A mask, so the core path is never touched.
# The per-iteration OUTER_SOLVE lines (already instrumented in outer_loop_functions.jl)
# give the convergence trace requested in Section 1.
# Env: RED=0/1 ;  optional SCALE=<knitro scale opt> handled via the .opt file.
# ================================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity

const RED = get(ENV,"RED","0") == "1"
const TAG = (RED ? "reduced" : "full") * get(ENV,"SUFFIX","")
const OUT = joinpath(@__DIR__,"out"); isdir(OUT) || mkpath(OUT)

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,θ_initial_low,θ_initial_up,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
D=so.D; bi=params.baseIndex; σ=params.σHat

# ----- bounds, exactly as in ccOuter.jl -----
θ_lower = (θ_initial.*0.0001)[:]; θ_upper = (θ_initial.*10000)[:]
θ_lower[2]=θ_initial[2]; θ_upper[2]=θ_initial[2]
θ_upper[1]=1/(σ-1)-0.001; θ_lower[1]=0.001; θ_upper[1]=min(θ_upper[1],1/(σ-1))
Aod_offset = 3 + D
for i in 1:D  # A[1,d]=1
    θ_upper[Aod_offset+1+D*(i-1)]=θ_initial[Aod_offset+1+D*(i-1)]
    θ_lower[Aod_offset+1+D*(i-1)]=θ_initial[Aod_offset+1+D*(i-1)]
end
Aidx(o,d)=Aod_offset+o+(d-1)*D
if RED   # fix all A columns d != baseIndex
    for d in 1:D, o in 1:D
        d==bi && continue
        θ_lower[Aidx(o,d)]=θ_initial[Aidx(o,d)]; θ_upper[Aidx(o,d)]=θ_initial[Aidx(o,d)]
    end
end
for i in 1:length(θ_initial)   # negative-θ bound swap (ccOuter)
    if θ_initial[i] < 0
        θ_upper[i] = -10000*θ_initial[i]; θ_lower[i] = 10000*θ_initial[i]
    end
end

nfreeA = count(o->θ_lower[Aidx(o[1],o[2])]!=θ_upper[Aidx(o[1],o[2])], [(o,d) for o in 1:D for d in 1:D])
println(">>> VARIANT=$TAG  free params=", count(i->θ_lower[i]!=θ_upper[i], 1:length(θ_initial)),
        "  freeA=", nfreeA); flush(stdout)

# frozen, isolated opt files (default) so runs don't collide with the shared config
const OUTEROPT = get(ENV,"OPT", joinpath(@__DIR__,"csw_outer.opt"))
const INNEROPT = joinpath(@__DIR__,"ek_inner.opt")
println(">>> using outer opt: ", OUTEROPT); flush(stdout)
mkobj(find_smallest) = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=find_smallest, γ=γ,
    (moments!)=EK_moments!, moments_jacobian! = error, d=nTotalMoments,
    outer_constr_index=outer_constr_index, inequality_index=inequality_index,
    complement_index=complement_index, l=length(θ_initial), U=U, N=params.Jac_W,
    lower_limit=-50, use_cached_x=true,
    outer_loop_opt=OUTEROPT, inner_loop_opt=INNEROPT)

t0=time()
println(">>> $TAG UPPER"); flush(stdout)
κ_up, θ_up, st_up, _ = outer_loop(mkobj(false), θ_lower, θ_upper, θ_initial_up)   # output=false (v1.2.1-safe)
println(">>> $TAG LOWER"); flush(stdout)
κ_lo, θ_lo, st_lo, _ = outer_loop(mkobj(true),  θ_lower, θ_upper, θ_initial_low)
wall=time()-t0

open(joinpath(OUT,"solve_$(TAG)_summary.txt"),"w") do f
    for ln in (
        "variant=$TAG  D=$D  baseIndex=$bi   (feas/opt/time in the >>> OUTER_SOLVE lines above)",
        @sprintf("kappa_lower = %.6f   (status=%d)", κ_lo, st_lo),
        @sprintf("kappa_upper = %.6f   (status=%d)", κ_up, st_up),
        @sprintf("width       = %.6f", κ_up-κ_lo),
        @sprintf("total wall  = %.1fs", wall))
        println(f, ln); println(ln)
    end
end
@save joinpath(OUT,"solve_$(TAG).jld2") κ_up κ_lo θ_up θ_lo st_up st_lo
println("DONE solve $TAG")
