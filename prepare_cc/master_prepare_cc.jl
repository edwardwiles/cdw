
function master_prepare_cc(data, counters, prestep_output, globalParams)

    @unpack seedU, W, D, counterType, importanceSampling, importanceSamplingFactor, sameMarginalsMoment, independenceMoment, GravityMomentFirstApproach, gravMoment, localGravityMoment, momentOrder, momentOrderForBaseIndex, baseIndex, ForceFrechetMarginal, IndMomentOrder, δGridType = globalParams

    # set seed
    Random.seed!(seedU)

    SamplingWeight = ones(W)
    # draw base U matrix from exp(1), using stratified or importance sampling as specified in parameters 
    U = drawU(SamplingWeight, globalParams)

    Ū, Uσ = createUDerivatives!(U, prestep_output, globalParams)

    useParams = globalParams
    useParams = (; useParams..., SamplingWeight= SamplingWeight) 

    # if using common marginals with moments methodology, calculate the K moments and put them in Ubar 

    CDF_Moments = zeros(1,1)
   
    if sameMarginalsMoment == 1
        CDFs = precalcCDFs(Ū, useParams)
        CDF_Moments = CDFs.CDF_Moments
    end 

    Ind_Moments = zeros(1, 1)
    IndCDF_Cells = Vector{Vector{Int}}(undef,1)

    if independenceMoment == 1
        Ind_Moments, IndCDF_Cells = precalcIndependence(Ū, useParams)      
    end 

    # trade share moments + gamma + gammaPrime 
    numMoments = D^2 + 2 * D 
    
    if counterType != 1
        numMoments += (D - 1)
    end   

    if GravityMomentFirstApproach == 1 
        numMoments += (1 + (1-sameMarginalsMoment)*D^2)
    end 

    if gravMoment == 1 
        numMoments += 1 
    end 

    if localGravityMoment == 1 
        numMoments += (D-1) * D + D * (D - 1) * (D - 2)
    end 

    if sameMarginalsMoment == 1 
        numMoments += 2 * momentOrder * D^2 + 2 * momentOrderForBaseIndex * D + 2 * D^2
    end 

    if independenceMoment == 1 
        numMoments += (1+ D * (D^2 - floor(Int, D * (1 + D) / 2)) + IndMomentOrder + IndMomentOrder^D)
    end 
    
    # index where outer loop moments start
    outer_constr_index = numMoments + 1 - GravityMomentFirstApproach

    file_name = string("Counter_", counterType, "_countries_", D, "_baseI", baseIndex, "_sGrav", GravityMomentFirstApproach, "_lGrav", localGravityMoment, "_Marg", sameMarginalsMoment, "_ind", independenceMoment, "_order", momentOrder, "_baseOrder",momentOrderForBaseIndex, "ForceFrechet_", ForceFrechetMarginal, "IndMO_", IndMomentOrder, "ISampling_", importanceSampling,"ISF_", importanceSamplingFactor,  "_Frechet", "_", Dates.format(now(), "y-m-d"), ".csv")


    PMM = zeros(numMoments)

    γ, θ_initial = buildObjectsForMoments(globalParams, prestep_output, data, counters.LPrime, counters.τPrime, Uσ, Ū, PMM, CDF_Moments, Ind_Moments, IndCDF_Cells, SamplingWeight)

    if δGridType == 0
        δ_grid =   vcat(1)
    else
        δ_grid =   vcat(0.01, 0.1, 0.5, 1, 2)
    end 

    prep_output = (
        U = U,
        γ = γ,
        θ_initial = θ_initial,
        numMoments = numMoments,
        file_name = file_name,
        δ_grid = δ_grid,
        outer_constr_index = outer_constr_index
    )

    return prep_output

end 