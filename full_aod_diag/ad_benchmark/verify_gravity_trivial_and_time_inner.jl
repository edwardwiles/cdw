# Two things:
# 1. Verify that the gravity constraint's gradient (currently computed via the generic
#    dense-Jacobian + ift! machinery) is IDENTICAL to a trivial DIRECT gradient of the
#    scalar sumGrav(theta) function alone (no draws loop, no lambda, no arg1) -- confirming
#    the code's own comment ("F-independent gravity/orthogonality moment") means the ift!
#    correction is provably zero here, not just small.
# 2. Directly TIME a single inner KNITRO solve (isolated from any gradient computation) at
#    the D=10 full-A config, to compare against Method B's already-measured 0.057s.
const ROOT = dirname(dirname(@__DIR__))
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2, BenchmarkTools
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(dirname(@__DIR__), "moments_gammanorm.jl"))

const D10 = 10
params = (server=1,user=2,fakeData=1,DFake=D10,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,U,γ,outer_constr_index,nTotalMoments = pp
D=so.D; bi=params.baseIndex; σ=params.σHat; μHat=γ.μHat
θ = build_theta_gammanorm(θ_initial, D, bi, μHat, σ)

# ---- Part 1: is the gravity gradient trivial? ----
# extract sumGrav(theta) as an ISOLATED scalar function: only needs mu and A_od (theta[8:end]
# at D=10... generically Aod_offset+1:end), NOT the draws, NOT lambda/arg1.
Aod_offset = 3 + D
τ = γ.τ; cHat = γ.cHat; wHat = γ.wHat; P = γ.P
lambda = reshape(P, (D, D))'
function sumGrav_scalar(θ)
    T = eltype(θ)
    μ = θ[1]
    Aod_θ = reshape(vcat(θ[Aod_offset+1:Aod_offset+D^2]), (D,D))
    Aod = Aod_θ .* cHat .* (((wHat.*τ)./(wHat[1,1].*τ[1,:]')).^(1/μ)) .* (lambda./lambda[1,:]')
    AodPow = (Aod ./ cHat) .^ (-μ)
    Wτ = withinTransform(τ)
    WA = withinTransform(AodPow)
    s = zero(T)
    for o in 1:D, d in 1:D
        s += Wτ[o,d]*WA[o,d]
    end
    return s
end
g_direct = ForwardDiff.gradient(sumGrav_scalar, θ)
t_direct = @belapsed ForwardDiff.gradient($sumGrav_scalar, $θ)
println("Direct gravity gradient (no draws loop): nnz=", count(x->abs(x)>1e-10,g_direct), " time=$(round(t_direct*1000;digits=3))ms")

# now build the FULL production-path gravity column via the generic dense-Jacobian+ift! route
# and confirm it matches g_direct exactly (proving the ift! correction is provably zero here)
mkobj() = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=true, γ=γ,
    (moments!)=EK_moments_gammanorm_directgp!, moments_jacobian! = error, d=nTotalMoments,
    outer_constr_index=outer_constr_index, inequality_index=Int64[], complement_index=[0 0],
    l=length(θ), U=U, N=params.Jac_W, lower_limit=-50, use_cached_x=false,
    outer_loop_opt=joinpath(dirname(@__DIR__),"csw_outer_25.opt"), inner_loop_opt=joinpath(dirname(@__DIR__),"ek_inner.opt"))
obj = mkobj()
objSol, x, nStatus = CS.inner_loop_internal(obj, θ)
println("inner solve status=$nStatus")
g = zeros(obj.l); jac = zeros((obj.d - obj.outer_constr_index + 2) * obj.l)
obj(x, g, θ; jac = jac)
∂c_∂θ = reshape(jac, obj.l, obj.d - obj.outer_constr_index + 2)'
g_gravity_production = ∂c_∂θ[2, :]  # gravity is the 2nd outer constraint
err = maximum(abs.(g_gravity_production .- g_direct))
relerr = err / maximum(abs.(g_direct))
println("production gravity grad (dense-Jac+ift!) vs direct-scalar: max abs err=$err  relerr=$relerr")

# ---- Part 2: time a single ISOLATED inner KNITRO solve (no gradient at all) ----
t_inner = @belapsed CS.inner_loop_internal($(mkobj()), $θ)
println("\nsingle inner KNITRO solve (D=$D, l=$(length(θ))): $(round(t_inner;digits=4))s")
println("Method B gradient (from earlier benchmark, D=10, l=113): 0.057s")
println("ratio inner-solve / Method-B-gradient = ", round(t_inner/0.057; digits=2))
println("DONE")
