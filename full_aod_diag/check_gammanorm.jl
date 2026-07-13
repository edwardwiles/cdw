# Standalone sanity check (no KNITRO needed): does EK_moments_gammanorm! run,
# produce finite output, and does the derived θ_initial_G have small moment
# mismatch (comparable to the old θ_initial's near-zero mismatch)?
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(@__DIR__, "moments_gammanorm.jl"))

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,U,γ,nTotalMoments = pp
D=so.D; bi=params.baseIndex; σ=params.σHat; μHat=γ.μHat
println("D=$D  baseIndex=$bi  σ=$σ  μHat=$μHat  nTotalMoments=$nTotalMoments  length(θ_initial)=$(length(θ_initial))")

bounds = theoretical_gammaprime_bounds(γ, σ)
println("theoretical γ'_focal bounds: [$(bounds.γp_lo), $(bounds.γp_hi)]  κ_max=$(bounds.κ_max)")

θ_initial_G = build_theta_gammanorm(θ_initial, D, bi, μHat, σ)
println("γ'_focal (direct, new gauge, init) = ", θ_initial_G[3+D],
        "  -> within bounds? ", bounds.γp_lo <= θ_initial_G[3+D] <= bounds.γp_hi)
Aod_offset = 3+D
println("A_od column initial values (new gauge), column d: ", [θ_initial_G[Aod_offset+1+D*(d-1)] for d in 1:D])
println("old γ_θ (calibrated, for reference): ", θ_initial[3:2+D])

K_old = zeros(params.W); G_old = zeros(params.W, nTotalMoments)
EK_moments!(K_old, G_old, θ_initial, U, (γ=γ,))
mismatch_old = maximum(abs.(vec(sum(G_old, dims=1)) ./ params.W))
println("OLD variant @ θ_initial: max|mean(G)| = ", mismatch_old, "  any NaN/Inf? ", any(!isfinite, G_old) || any(!isfinite, K_old))

K_new = zeros(params.W); G_new = zeros(params.W, nTotalMoments)
EK_moments_gammanorm!(K_new, G_new, θ_initial_G, U, (γ=γ,))
mismatch_new = maximum(abs.(vec(sum(G_new, dims=1)) ./ params.W))
println("NEW variant @ θ_initial_G: max|mean(G)| = ", mismatch_new, "  any NaN/Inf? ", any(!isfinite, G_new) || any(!isfinite, K_new))
println("mean(K_old)=", sum(K_old)/params.W, "  mean(K_new)=", sum(K_new)/params.W)

# ForwardDiff sanity: does the new moments function survive autodiff (needed for use_Jacobian=0)?
try
    f(θ) = begin
        Kd = zeros(eltype(θ), params.W); Gd = zeros(eltype(θ), params.W, nTotalMoments)
        EK_moments_gammanorm!(Kd, Gd, θ, U, (γ=γ,))
        vec(sum(Gd, dims=1))
    end
    J = ForwardDiff.jacobian(f, θ_initial_G)
    println("ForwardDiff.jacobian OK, size=", size(J), "  any NaN/Inf? ", any(!isfinite, J))
catch e
    println("ForwardDiff FAILED: ", e)
end
println("CHECK DONE")
