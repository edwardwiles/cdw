# ============================================================================
# Shared setup for the derivative-method audit. Builds the SAME economy/data/
# draws (D=4, autarky, gravity ON, W=Jac_W=8000) used by compare_directgp.jl —
# the γ_d≡1-for-all-d + direct-γ' objective variant, now the best-conditioned
# configuration found (see full_aod_diag/out/compare_directgp_summary.txt) and
# per the user's instruction, the one this audit targets. Include this file
# first from any ad_benchmark script.
# ============================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
const AD_ROOT = dirname(dirname(@__DIR__))
include(joinpath(AD_ROOT,"setup/include_setup.jl")); include(joinpath(AD_ROOT,"prestep/include_prestep.jl"))
include(joinpath(AD_ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(AD_ROOT,"moments/include_moments.jl"))
include(joinpath(AD_ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(AD_ROOT,"lfd/include_lfd.jl")); include(joinpath(AD_ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(dirname(@__DIR__), "moments_gammanorm.jl"))

const AD_PARAMS = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)

function build_ad_context()
    so = master_setup(AD_PARAMS)
    up = (; AD_PARAMS..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(up)
    ps = master_prestep(so.data, so.counters, up)
    pp = master_prepare_cc(so.data, so.counters, ps, up)
    return so, pp
end

"""
    make_obj(γobj, U, nTotalMoments, outer_constr_index, inequality_index, complement_index, l; opt=ek_inner-ish)

Build a PsiObjectiveBundleImplicit identical in structure to compare_directgp.jl's,
using EK_moments_gammanorm_directgp! (the best-conditioned variant this audit targets).
"""
function make_ad_obj(pp, so; N=AD_PARAMS.Jac_W)
    @unpack outer_constr_index, inequality_index, nTotalMoments, complement_index, U, γ = pp
    D = so.D
    return PsiObjectiveBundleImplicit(δ=1.0, find_smallest=true, γ=γ,
        (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error,
        d = nTotalMoments, outer_constr_index = outer_constr_index,
        inequality_index = inequality_index, complement_index = complement_index,
        l = length(pp.θ_initial), U = U, N = N, lower_limit = -50, use_cached_x = true,
        outer_loop_opt = joinpath(@__DIR__, "..", "csw_outer_25.opt"),
        inner_loop_opt = joinpath(@__DIR__, "..", "ek_inner.opt"))
end
