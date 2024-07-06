
function master_prepare_cc(data, counters, prestep_output, globalParams)

    @unpack seedU, W, D, counterType, importanceSampling, importanceSamplingFactor, sameMarginalsMoment, independenceMoment, GravityMomentFirstApproach, gravMoment, localGravityMoment, momentOrder, momentOrderForBaseIndex, baseIndex, ForceFrechetMarginal, IndMomentOrder, δGridType, OuterScaling, fakeData, θConstant, usePMM , UoModel, calc_δ_star_initial, Jac_W, δ_ref, NormalizeMoments= globalParams

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
        CDF_Moments = precalcCDFs(Ū, useParams, prestep_output)
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
        if UoModel == 1
            numMoments += 2 * D + 2 * momentOrderForBaseIndex * D
        else
            numMoments += 2 * momentOrder * D^2 + 2 * D^2 + 2 * momentOrderForBaseIndex * D
        end
    end 

    if independenceMoment == 1 
        numMoments += IndMomentOrder + (D^2 - floor(Int, D * (1 + D) / 2)) * (IndMomentOrder^2)
	
        if UoModel == 0
            numMoments += D * (D^2 - floor(Int, D * (1 + D) / 2)) 
        else
            numMoments += (D^2 - floor(Int, D * (1 + D) / 2)) 
        end        
    end 
    
    # index where outer loop moments start
    outer_constr_index = numMoments + 1 - GravityMomentFirstApproach

    file_name = string("FD_",fakeData,"_Count_", counterType, "_NC_", D, "_bI", baseIndex, "_sG", GravityMomentFirstApproach, "_lG", localGravityMoment, "_Marg", sameMarginalsMoment, "_ind", independenceMoment, "_O", momentOrder, "_bO",momentOrderForBaseIndex, "FF_", ForceFrechetMarginal, "IndMO_", IndMomentOrder, "IS_", importanceSampling,"ISF_", importanceSamplingFactor, "Aod_",OuterScaling, "Fmu_", θConstant, "Uo_", UoModel, "_", Dates.format(now(), "y-m-d"), ".csv")


    PMM = zeros(numMoments)
    σ_Moments = ones(numMoments)

    γ, θ_initial = buildObjectsForMoments(globalParams, prestep_output, data, counters.LPrime, counters.τPrime, Uσ, Ū, PMM, σ_Moments, CDF_Moments, Ind_Moments, IndCDF_Cells, SamplingWeight)

    γ_PMM, δ_star_initial = γPMM(θ_initial, γ, U, numMoments, outer_constr_index, calc_δ_star_initial)


    if δGridType == 0
        δ_grid =   vcat(δ_ref)
    else
        δ_grid =   vcat(0.01, 0.1, 0.5, 1, 2) .* δ_ref
    end 

    δ_grid_filtered = filter(x -> x >= δ_star_initial, δ_grid)

    @show δ_star_initial
    @show δ_grid
    @show δ_grid_filtered

    prep_output = (
        U = U,
        γ = (usePMM ==1 || NormalizeMoments == 1) ? γ_PMM : γ, 
        θ_initial = θ_initial,
        numMoments = numMoments,
        file_name = file_name,
        δ_grid = δ_grid_filtered,
        outer_constr_index = outer_constr_index
    )

    return prep_output

end 