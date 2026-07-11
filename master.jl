
# ---- dependencies (Main scope) ----
# Only cc_algo/ is wrapped in a module; setup/, prestep/, prepare_cc/, moments/,
# lfd/ and misc/ are included directly into Main. Every package those files use
# must therefore be imported HERE, and before the includes below — macros such as
# @unpack (Parameters) and @threads (Base.Threads) expand at include/parse time.
using Parameters                              # @unpack (used throughout Main-scope code)
using Base.Threads                            # @threads
using Random, Dates, DelimitedFiles           # seed!, timestamps, writedlm/readdlm
using Distributions, Statistics               # distributions, mean/cov/quantile
using SpecialFunctions                        # gamma (price-index / Frechet moments)
using InvertedIndices                         # Not(...) in the analytic moment Jacobian
using NLsolve, ForwardDiff, Calculus, LinearAlgebra
using Plots, JLD2

include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl")
include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl")
include("lfd/include_lfd.jl")
include("misc/include_misc.jl")

using .CounterfactualSensitivity              # module defined by cc_algo/include_cc_algo.jl above

function main(globalParams)

    # run setup 
    setup_output = master_setup(globalParams)
    @unpack data, counters = setup_output
 
    # add D to the parameters list 
    useParams = globalParams
    useParams = (; useParams..., D = setup_output.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!) 
    
    # check parameters are compatible
    checkParams(useParams)

    # run prestep 
    # nb: general distribution prestep not set up, only implements Frechet 
    prestep_output = master_prestep(data, counters, useParams)

    # prepares all of the objects necessary to run the CC algorithm 
    prep_output = master_prepare_cc(data, counters, prestep_output, useParams)

    @show Dates.format(now(), "HH:MM") # print time  
    cc_output = master_cc_algo(prep_output, useParams)
    if useParams.OuterLoop ==0
        master_lfd(useParams, setup_output, prestep_output, prep_output, cc_output)
    end
	@show Dates.format(now(), "HH:MM") # print time 
end 

## Define global parameters
#  Core CC method (CDW / Christensen–Connault): unrestricted distribution-agnostic
#  gains-from-trade bound for a 4-country simulated (Frechet) example, closed-form
#  Frechet prestep, autarky counterfactual, hybrid-divergence (Psi) outer loop.
#  All high-level restriction moments (common marginals, independence, gravity) and
#  all alternative starting points (Frechet copula, RN, general-distribution) are OFF.
params = (

    # setup parameters
    server=1, # 1 if using server, 0 otherwise (uses server file path if 1)
    user = 2, # 1 = Habib, 2 = Ed (this modular working copy; see setup/setwd.jl)
    fakeData=1, # 1 to generate data, 0 to use from files
    DFake=4, # if using fake data, number of countries to gen data
    seedFakeData = 889, # seed for fake data generation
    counterType=1, # 0 = zero gravity, 1 = autarky, 2 = custom
    counterExplicit=0, # 1 = explicit counterfactuals (uses kappaStar), 0 = implicit

    # prestep parameters
    θHat = 0, # 0 = estimate theta via gravity, otherwise fixes theta hat
    σHat = 2.5, # CES elasticity, maintained throughout
    baseIndex = 2, # country to use as wage normalisation and counterfactual
    W = 8000, # number of Monte Carlo draws of U (paper uses ~80000; reduced here for a fast 4-country example)

    # prepare CC parameters
    seedU = 888,
    importanceSampling = 0, # 1: importance sampling; 2: half the U realizations < 0.1
    importanceSamplingFactor = 2,
    stratifiedSampling = 0, # stratified Exp(1) draws
    IndMomentOrder = 5,
    θConstant=0, # 1 if theta and sigma never vary (precalculate U^((1-σ)/θ))
    gravMoment=0, # =1 impose gravity identification for Frechet
    localGravityMoment=0, # =1 impose model-implied trade elasticity matches θHat
    localGravityCrossMoment=0, # =1 impose local cross-elasticity = 0 (ACR R3)
    GravityMomentFirstApproach=0, # =1 impose mean independence between lnU and lnτ
    sameMarginalsMoment=0, # =1 impose all od pairs have the same U distribution (common marginals)
    NoScalingforSameMartingale=1, # =1 moment-based (no CDF scaling) common-marginals variant
    useCDFforMarginalMatching=0, # =1 match marginals via CDF quantiles instead of moments
    independenceMoment=0, # =1 impose cross-country independence / zero correlation
    momentOrder=5, # # moment conditions to approximate the same-marginal condition
    momentOrderForBaseIndex = 50, # same, for U_baseIndex,baseIndex
    ForceFrechetMarginal = 0, # =1 maintain Frechet marginal, vary only dependence
    OuterScaling = 1, # 1 = Aod model, 0 = Aod fixed
    useParallel=0, # 1 = parallelise the deltas in the outer loop
    usePMM=0, # =1 adjust moments so they are exactly zero for F*
    PMMGammaOnly = 0,
    NormalizeMoments = 0, # =1 divide by moment std dev so all moments are on the same scale
    useConfidenceIntervals = 0, # =1 force moments to lie within a confidence interval around zero
    ConfidenceLevel = 0.05, # confidence level for the moment CI
    δGridType = 0, # 0: {δ_ref}, else: δ_ref*{0.01, 0.1, 0.5, 1, 2}
    δ_ref = 1, # divergence-neighborhood radius δ (see CDW §4; δ ≤ 2 is the paper's range)
    refIndex1 = 1, # refIndex used for CDF conditions
    OuterLoop = 1, # =1 CC Outer (compute the min/max κ bounds); =0 CC Inner (single δ* at θ_initial)
    UoModel = 1, # =1 Uo, =0 Uod
    use_Jacobian = 1, # =1 analytic outer Jacobian (required: moments! uses Float64 caches incompatible with ForwardDiff)
    calc_δ_star_initial = 1, # calculate the starting (minimum feasible) delta
    Jac_W = 8000, # number of simulations for the analytic Jacobian (MUST be <= W)
    theta_init = 0, # initialise from a theta different from thetaHat
    # (starting point: always the closed-form Frechet prestep — the CDW method.
    #  The experimental Frechet-copula / RN / general-distribution starts have been removed.)

    # Post CC Optimization tests
    runLFD = 1,
    runLFDCounterFactual = 1
)

main(params)
