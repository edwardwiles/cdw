
include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl")
include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl")
include("misc/include_misc.jl")

using Distributions, Statistics, Plots

function main(globalParams)

    # run setup 
    setup_output = master_setup(globalParams)
    @unpack data, counters = setup_output
 
    # add D to the parameters list 
    useParams = globalParams
    useParams = (; useParams..., D = setup_output.D)    

    # run prestep 
    # nb: general distribution prestep not set up, only implements Frechet 
    prestep_output = master_prestep(data, counters, useParams)

    # prepares all of the objects necessary to run the CC algorithm 
    prep_output = master_prepare_cc(data, counters, prestep_output, useParams)

    #=
    ## test the moment function 
    K = zeros(globalParams.W)
    G = Array{Float64,2}(undef,globalParams.W,prep_output.numMoments)

    obj = (γ = prep_output.γ, a = 5, d = prep_output.numMoments)

    moments!(K,G,prep_output.θ_initial,prep_output.γ.Ū[:,:,1],obj)

    @show sum(G,dims=1)./globalParams.W
    =#

    @show Dates.format(now(), "HH:MM") # print time     

    δ_grid, Θ_upper, κ_upper, Θ_lower, κ_lower = ccOuter(θ_initial, θ_initial, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, momentOrderForBaseIndex, useIndependentCFDs,IndMomentOrder, counterType, useParallel, file_name) # run the outer loop 
    writedlm(file_name, [δ_grid κ_lower κ_upper], ',')

    # store the parameters
   writedlm(string("Theta_initial_Frechet", "_", file_name), θ_initial, ',')

    for i = 1:length(δ_grid)
        writedlm(string("Theta_upper_", δ_grid[i], "_", file_name), Θ_upper[:, i], ',')
        writedlm(string("Theta_lower_", δ_grid[i], "_", file_name), Θ_lower[:, i], ',')
    end

    #= To test stuff with specific theta
    δ_grid  = [1] 
    Θ_upper = copy(θ_initial)
    Θ_lower = copy(θ_initial)
    Θ_upper[:,1] = readdlm("Theta_upper_1_Counter_1_countries_4_baseI2_sGrav0_lGrav0_Marg0_NoSc1_ind0_order5_baseOrder50useCDF_1ForceFrechet_0stratify_0IndCDF_0IndMO_5ISampling_0ISF_2_Frechet_4-3-14.csv", ',')
    Θ_lower[:,1] = readdlm("Theta_lower_1_Counter_1_countries_4_baseI2_sGrav0_lGrav0_Marg0_NoSc1_ind0_order5_baseOrder50useCDF_1ForceFrechet_0stratify_0IndCDF_0IndMO_5ISampling_0ISF_2_Frechet_4-3-14.csv", ',')
    =#
    calculateLFD = true
    if calculateLFD
        LFD_upper = zeros(W, length(δ_grid))
        LFD_lower = zeros(W, length(δ_grid))
        for i = 1:length(δ_grid)
            LFD_upper[:, i] = LFD(Θ_upper[:, i], U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, momentOrderForBaseIndex,useIndependentCFDs,IndMomentOrder, counterType)
            LFD_lower[:, i] = LFD(Θ_lower[:, i], U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder,momentOrderForBaseIndex,useIndependentCFDs,IndMomentOrder, counterType)
        end
        writedlm(string("LFD_up_", file_name), LFD_upper, ',')
        writedlm(string("LFD_low_", file_name), LFD_lower, ',')
    end

    @show Θ_upper[:, 1]
    @show Dates.format(now(), "HH:MM") # print time    

end 

## Define global parameters
params = (

    # setup parameters 
    server=0, # 1 if using server, 0 otherwise (uses server file path if 1)
    user = 2, # 1 = Habib, 2 = Ed 
    fakeData=1, # 1 to generate data, 0 to use from files
    DFake=4, # if using fake data, number of countries to gen data
    seedFakeData = 9389, # seed for fake data generation 
    counterType=1, # 0 = zero gravity, 1 = autarky, 2 = custom (adjust above)
    counterExplicit=0, # 1 = explicit counterfactuals (uses kappaStar), 0 = implicit

    # prestep parameters 
    θHat = 0, # 0 = estimate theta via gravity, otherwise fixes theta hat 
    σHat = 2.5, # CES elasticity, maintained throughout 
    baseIndex = 2, # country to use as wage normalisation and counterfactual
    W = 80000,

    # prepare CC parameters 
    seedU = 888,
    stratifiedSampling = 0, # 1 = generates half simulations with low price realizations
    importanceSampling = 0,
    importanceSamplingFactor = 2, 
    useIndependentCFDs = 0,
    IndMomentOrder = 5,
    θConstant=0, # put 1 if theta and sigma never vary, will precalculate U^((1-sigma)/theta)
    gravMoment=0, # = 1 impose gravity identification for Frechet, 0 = do not
    localGravityMoment=0, # = 1 impose model implied trade elasticity mtaches θHat, 0 = do not 
    localGravityCrossMoment=0, # = 1 impose model implied trade cross elasticity is zero, 0 = do not 
    GravityMomentFirstApproach=0, # = 1 imposes mean independence between lnU and ln tau , 0 = do not
    sameMarginalsMoment=1, # =1 imposes all od pairs have the same U distribution
    NoScalingforSameMartingale=1,# =1 imposes a strict same marginal condition, without allowing for a multiplicative dergree of freedom   
    independenceMoment=0, # =1 imposes correlation[Uod, Uo'd] = 0 
    momentOrder=5, # number of moment conditions to approximate same marginal condition 
    momentOrderForBaseIndex = 50, # number of moment conditions to approximate same marginal condition for U_baseIndex,baseIndex
    useCDFforMarginalMatching=0, # 1= impose same marginal condition using CDF, 0= using moments 
    ForceFrechetMarginal = 0, # 1= maintains Frechet marginal and leaves dependency to change
    useParallel=0, # 1 = parallelise the deltas in outer loop; 0 = do not 
    InitDistributionType=1, # 0 = Frechet, 1 = lognormal, 2= t-dist, 3 = flexible (see genRands. uses corr and Param), 4 = LN productivity correlated with trade costs
    InitDistributionCorr=0.0, #correlation between countries
    InitDistributionParam=1, #parameter of the distribution (Not used for Frechet & LN)
    StarDistributionType=0, # 0 = Frechet, 1 = lognormal, 2= t-dist, 3 = flexible (see genRands. uses corr and Param), 4 = LN productivity correlated with trade costs
    StarDistributionCorr=0.0, #correlation between countries
    StarDistributionParam=1, #parameter of the distribution (Not used for Frechet & LN)
    usePMM=0 # not really implemented anymore should be removed maybe)        

)

main(params)
