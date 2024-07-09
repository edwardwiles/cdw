
include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl")
include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl")
include("lfd/include_lfd.jl")
include("misc/include_misc.jl")

using Distributions, Statistics, Plots, .CounterfactualSensitivity, JLD2

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
params = (

    # setup parameters 
    server=1, # 1 if using server, 0 otherwise (uses server file path if 1)
    user = 1, # 1 = Habib, 2 = Ed 
    fakeData=1, # 1 to generate data, 0 to use from files
    DFake=4, # if using fake data, number of countries to gen data
    seedFakeData = 889, # seed for fake data generation 
    counterType=1, # 0 = zero gravity, 1 = autarky, 2 = custom (adjust above)
    counterExplicit=0, # 1 = explicit counterfactuals (uses kappaStar), 0 = implicit

    # prestep parameters 
    θHat = 0, # 0 = estimate theta via gravity, otherwise fixes theta hat 
    σHat = 2.5, # CES elasticity, maintained throughout 
    baseIndex = 2, # country to use as wage normalisation and counterfactual
    W = 80000,

    # prepare CC parameters 
    seedU = 888,
    importanceSampling = 0, # 1 : Importance sampling with an exponential weight, 2: Half the realizations of U are smaller than 0.1
    importanceSamplingFactor = 2, 
    IndMomentOrder = 5,
    θConstant=0, # put 1 if theta and sigma never vary, will precalculate U^((1-sigma)/theta)
    gravMoment=0, # = 1 impose gravity identification for Frechet, 0 = do not
    localGravityMoment=0, # = 1 impose model implied trade elasticity mtaches θHat, 0 = do not 
    GravityMomentFirstApproach=0, # = 1 imposes mean independence between lnU and ln tau , 0 = do not
    sameMarginalsMoment=0, # =1 imposes all od pairs have the same U distribution
    independenceMoment=0, # =1 imposes correlation[Uod, Uo'd] = 0 
    momentOrder=5, # number of moment conditions to approximate same marginal condition 
    momentOrderForBaseIndex = 50, # number of moment conditions to approximate same marginal condition for U_baseIndex,baseIndex
    ForceFrechetMarginal = 0, # 1= maintains Frechet marginal and leaves dependency to change
    OuterScaling = 1, # 1= Aod model, 0 = Aod fixed
    useParallel=0, # 1 = parallelise the deltas in outer loop; 0 = do not 
    usePMM=0, # =1 adjust moments so they are exactly zero for F*, =0 do not.
    NormalizeMoments = 1, # =1 divide by moment std dev so all moments are on the same scale
    useConfidenceIntervals = 1, # forces moments to be within a confidence interval (around zero)
    ConfidenceLevel = 0.05, # confidence level for the moments being within the CI
    δGridType = 0, # 0: {1}, else: {0.01, 0.1, 0.5, 1, 2}
    δ_ref = 4, # Grid = δ_ref*{1} or  δ_ref*{0.01, 0.1, 0.5, 1, 2}
    refIndex1 = 1, # refIndex used for CDF conditions
    OuterLoop =0, # =0 CC Inner, =1 CC Outer
    UoModel = 1, # =1 Uo, =0 Uod  
    use_Jacobian = 1, # calculate Jacobian analytically for OuterLoop 
    calc_δ_star_initial = 1, # calculate the starting delta
    Jac_W = 25000, # number of simulations for the Jacobian
    theta_init = 0, # initial parameters based on a theta that is different from thetaHat
    
    # Post CC Optimization tests
    runLFD = 1,
    runLFDCounterFactual = 1
)

main(params)
