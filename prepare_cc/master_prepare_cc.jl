
function master_prepare_cc(data, counters, prestep_output, globalParams)

    @unpack seedU, W, D, counterType, stratifiedSampling, importanceSampling, importanceSamplingFactor, sameMarginalsMoment, useCDFforMarginalMatching, independenceMoment, NoScalingforSameMartingale, GravityMomentFirstApproach, gravMoment, localGravityMoment, localGravityCrossMoment, momentOrder, momentOrderForBaseIndex, baseIndex, ForceFrechetMarginal, useIndependentCFDs, IndMomentOrder = globalParams

    # set seed
    Random.seed!(seedU)

    ImportanceSamplingWeight = ones(W)
    # draw base U matrix from exp(1), using stratified or importance sampling as specified in parameters 
    U = drawU(ImportanceSamplingWeight, globalParams)

    Ū, Uσ, Σ_od = createUDerivatives!(U, prestep_output, globalParams)

    # if using common marginals with moments methodology, calculate the K moments and put them in Ubar 

    K_ = 1

    CDF_X = zeros(1)
    CDF_X_for_base = zeros(1)

    CDF_Moments = zeros(W, (K_ + 1) * D^2 + momentOrderForBaseIndex + D^2)
    CDF_Moments_for_base = zeros(W, momentOrderForBaseIndex)    

    if sameMarginalsMoment == 1
        if useCDFforMarginalMatching == 0
            fillUBarMoments!(Ū, U, globalParams)
        elseif useCDFforMarginalMatching == 1
            CDFs = precalcCDFs(Ū, globalParams)
            CDF_X = CDFs.CDF_X 
            CDF_X_for_base = CDFs.CDF_X_for_base
            K_ = size(CDFs.CDF_Moments, 2)
        end 
    end 

    Ind_Moments = zeros(1, 1)
    IndCDF_Cells = Vector{Vector{Int}}(undef,1)

    if independenceMoment == 1 && NoScalingforSameMartingale == 1
        IndMoments, IndCDF_Cells = precalcIndependence(Ū, globalParams)
        K_ = size(IndMoments, 2)        
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
        numMoments += (D-1) * D 
    end 

    if localGravityCrossMoment == 1 
        numMoments += D * (D - 1) * (D - 2)
    end 

    if sameMarginalsMoment == 1 
        numMoments += ((2 * momentOrder + 1) * D^2 + momentOrderForBaseIndex)
    end 

    if independenceMoment == 1 
        numMoments += (1+ D * (D^2 - floor(Int, D * (1 + D) / 2))+ (1-useIndependentCFDs)*(D-1)* 2 * momentOrder + useIndependentCFDs*((IndMomentOrder-1) + (IndMomentOrder-1)^D))
    end 

    MomentNames = nameMoments(numMoments, K_, globalParams)
    
    file_name = string("Counter_", counterType, "_countries_", D, "_baseI", baseIndex, "_sGrav", GravityMomentFirstApproach, "_lGrav", localGravityMoment, "_Marg", sameMarginalsMoment, "_NoSc", NoScalingforSameMartingale, "_ind", independenceMoment, "_order", momentOrder, "_baseOrder",momentOrderForBaseIndex, "useCDF_", useCDFforMarginalMatching, "ForceFrechet_", ForceFrechetMarginal, "stratify_", stratifiedSampling, "IndCDF_",useIndependentCFDs, "IndMO_", IndMomentOrder, "ISampling_", importanceSampling,"ISF_", importanceSamplingFactor,  "_Frechet", "_", Dates.format(now(), "y-m-d"), ".csv")

    writedlm(string("MomentNames_", file_name), MomentNames, ',')

    Mτ = calcMτ(data.τData, D)
    PMM = zeros(numMoments)

    γ, θ_initial = buildObjectsForMoments(globalParams, prestep_output, data, counters.LPrime, counters.τPrime, Uσ, Ū, Σ_od, Mτ, PMM, CDF_X, CDF_Moments, Ind_Moments, IndCDF_Cells, ImportanceSamplingWeight)

    prep_output = (
        γ = γ,
        θ_initial = θ_initial,
        numMoments = numMoments,
        MomentNames = MomentNames
    )

    return prep_output

end 