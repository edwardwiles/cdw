# Retry of test_methodB_struct.jl using the REAL, already-validated focal moments machinery
# (EK_moments_focal_norm_directgp!, D=4) instead of a synthetic toy moments function — the toy
# version segfaulted Julia 1.12.6's compiler (a compiler-side crash unrelated to Method B's own
# logic; not chased further, see conversation). Real economics, much smaller D for speed.
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
const ROOT = dirname(dirname(@__DIR__))
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(ROOT, "sequential_gravity", "focal_moments.jl"))
include(joinpath(ROOT, "sequential_gravity", "focal_moments_directgp.jl"))
CS.include(joinpath(ROOT, "sequential_gravity", "PsiObjectiveBundleImplicitMethodB.jl"))

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
D = so.D; focal = params.baseIndex; σ = params.σHat
θr0_orig = build_focal_theta(pp.θ_initial, D, focal)
γf0 = θr0_orig[3]; μ0 = θr0_orig[1]
θr0 = vcat(μ0, σ, θr0_orig[4]/γf0, fill(γf0^(-σ/(μ0*(σ-1))), D))

d = D + 1  # focal-only moments, no gravity column for this isolated test (oci=d+1 either way)
oci = d + 1
INNEROPT = joinpath(ROOT, "full_aod_diag", "ek_inner.opt")
γ = pp.γ; U = pp.U; JacW = params.Jac_W

obj_std = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=true, γ=γ, (moments!)=EK_moments_focal_norm_directgp!,
    moments_jacobian! = error, d=d, outer_constr_index=oci, inequality_index=Int64[], complement_index=[0 0],
    l=length(θr0), U=U, N=JacW, lower_limit=-50.0, use_cached_x=false,
    outer_loop_opt=INNEROPT, inner_loop_opt=INNEROPT)

obj_B = CS.PsiObjectiveBundleImplicitMethodB(δ=1.0, find_smallest=true, γ=γ, (moments!)=EK_moments_focal_norm_directgp!,
    moments_jacobian! = error, d=d, outer_constr_index=oci, inequality_index=Int64[], complement_index=[0 0],
    l=length(θr0), U=U, N=JacW, lower_limit=-50.0, use_cached_x=false,
    outer_loop_opt=INNEROPT, inner_loop_opt=INNEROPT)

objSol_std, x_std, st_std = CS.inner_loop_internal(obj_std, θr0)
objSol_B, x_B, st_B = CS.inner_loop_internal(obj_B, θr0)
println("inner solve status: std=$st_std  B=$st_B  objSol match=", isapprox(objSol_std, objSol_B; rtol=1e-10))
println("x match: ", isapprox(x_std, x_B; rtol=1e-8))

l = length(θr0)
g_std = zeros(l); jac_std = zeros(1*l)
obj_std(x_std, g_std, θr0; jac = jac_std)

g_B = zeros(l); jac_B = zeros(1*l)
obj_B(x_B, g_B, θr0; jac = jac_B)

println("objective gradient match: ", isapprox(g_std, g_B; rtol=1e-8))
relerr = maximum(abs.(jac_std .- jac_B)) / maximum(abs.(jac_std))
println("constraint jac relerr = ", relerr)
@assert relerr < 1e-8 "Method B struct does NOT match production!"
println("METHOD B STRUCT VALIDATED OK (real focal moments, D=$D)")
