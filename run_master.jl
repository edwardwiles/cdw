# Parametrized experiment runner: identical to master.jl but reads a few overrides
# from environment variables so experiments can be driven without editing source.
#   EXP_USE_JAC   : use_Jacobian (default 1)
# .opt algorithm / hessopt / maxit / tol are swept by editing the .opt files in the harness.

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra
using Plots, JLD2

include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl")
include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl")
include("lfd/include_lfd.jl")
include("misc/include_misc.jl")
using .CounterfactualSensitivity

function main(globalParams)
    setup_output = master_setup(globalParams)
    @unpack data, counters = setup_output
    useParams = globalParams
    useParams = (; useParams..., D = setup_output.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(useParams)
    prestep_output = master_prestep(data, counters, useParams)
    prep_output = master_prepare_cc(data, counters, prestep_output, useParams)
    t0 = time()
    println(">>> OUTER_SOLVE_START ", Dates.format(now(), "HH:MM:SS"))
    cc_output = master_cc_algo(prep_output, useParams)
    println(">>> OUTER_SOLVE_END   ", Dates.format(now(), "HH:MM:SS"), "  elapsed_s=", round(time()-t0; digits=2))
    if useParams.OuterLoop == 0
        master_lfd(useParams, setup_output, prestep_output, prep_output, cc_output)
    end
    return cc_output
end

const EXP_USE_JAC = parse(Int, get(ENV, "EXP_USE_JAC", "1"))

params = (
    server=1, user=2, fakeData=1, DFake=4, seedFakeData=889,
    counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=8000,
    seedU=888, importanceSampling=0, importanceSamplingFactor=2, stratifiedSampling=0,
    IndMomentOrder=5, θConstant=0, gravMoment=0, localGravityMoment=0, localGravityCrossMoment=0,
    GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0,
    NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05,
    δGridType=0, δ_ref=1, refIndex1=1, OuterLoop=1, UoModel=1, use_Jacobian=EXP_USE_JAC,
    calc_δ_star_initial=1, Jac_W=8000, theta_init=0, runLFD=1, runLFDCounterFactual=1,
)

println(">>> EXP_USE_JAC=", EXP_USE_JAC)
main(params)
