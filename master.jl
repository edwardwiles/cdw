
include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("moments/include_moments.jl")
include("misc/include_misc.jl")

function main(globalParams)

    # run setup 
    setup_output = master_setup(globalParams)
    @unpack data, counters = setup_output
 
    # add D to the parameters list 
    useParams = globalParams
    useParams = (; useParams..., D = setup_output.D)    

    # run prestep 
    # nb: general distribution prestep not set up on, only implements Frechet 
    prestep_output = master_prestep(data, counters, useParams)

    return prestep_output

end 

## Define global parameters
test = (

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

    # prepare CC parameters 
    seedU = 888,
    stratifiedSampling = 0, # 1 = generates half simulations with low price realizations
    importanceSampling = 0,
    importanceSamplingFactor = 2

)

main(test)
