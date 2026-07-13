# Direct empirical answer to "does Method B's (forward-mode) cost actually scale with l
# (number of outer params) the way theory predicts?" Benchmarks the SAME envelope-scalar
# ForwardDiff.gradient call at D=4 (l=23, full-A) vs D=10 (l=101 free, full-A), same W=8000,
# steady-state (post-warmup) timing only.
using BenchmarkTools, Printf
using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
const ROOT = dirname(dirname(@__DIR__))
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(dirname(@__DIR__), "moments_gammanorm.jl"))

function build(DFake)
    params = (server=1,user=2,fakeData=1,DFake=DFake,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
    so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
    return so, pp, params
end

function envelope_scalar(θ, U, γobj, λ, arg1, d, oci)
    N = size(U,1); T = eltype(θ)
    K = zeros(T,N); G = zeros(T,N,d)
    EK_moments_gammanorm_directgp!(K, G, θ, U, (γ=γobj,))
    s = zero(T)
    @inbounds for draw in 1:N
        acc = zero(T)
        for j in 1:oci-1
            acc += λ[j]*G[draw,j]
        end
        s += arg1[draw]*acc
    end
    return (1e10/N)*s
end

for DFake in (4, 10)
    so, pp, params = build(DFake)
    D = so.D; μHat = pp.γ.μHat; σ = params.σHat; bi = params.baseIndex
    θ = build_theta_gammanorm(pp.θ_initial, D, bi, μHat, σ)
    l = length(θ)
    oci = pp.outer_constr_index; d = pp.nTotalMoments
    λ = ones(oci-1) .* 0.01
    arg1 = ones(params.W)
    U = pp.U; γobj = pp.γ

    f = θθ -> envelope_scalar(θθ, U, γobj, λ, arg1, d, oci)
    f(θ)  # warmup / compile
    b = @benchmark $f($θ) samples=15 seconds=60
    println("D=$D  l=$l (free outer params)  d=$d (moments)  N=$(params.W):")
    println("  Method B (ForwardDiff scalar gradient of envelope): median=$(round(median(b).time/1e9;digits=4))s  min=$(round(minimum(b).time/1e9;digits=4))s  alloc=$(round(median(b).memory/1e6;digits=1))MB")

    # also time the DENSE Jacobian for comparison (Method A), same config
    fj! = (Hvec, θ) -> begin
        H = reshape(Hvec, params.W, d+2)
        K = @view H[:,1]; G = @view H[:,3:end]
        EK_moments_gammanorm_directgp!(K, G, θ, U, (γ=γobj,))
    end
    Hvec0 = zeros(params.W*(d+2))
    ForwardDiff.jacobian(fj!, Hvec0, θ)  # warmup
    bA = @benchmark ForwardDiff.jacobian($fj!, $Hvec0, $θ) samples=8 seconds=60
    println("  Method A (dense Jacobian only, no contract): median=$(round(median(bA).time/1e9;digits=4))s  min=$(round(minimum(bA).time/1e9;digits=4))s  alloc=$(round(median(bA).memory/1e6;digits=1))MB")
    println("  ratio A/B = $(round(median(bA).time/median(b).time; digits=2))x")
    println()
end
println("SCALING BENCHMARK DONE")
