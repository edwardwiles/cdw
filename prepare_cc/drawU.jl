
function drawU(ImportanceSamplingWeight, params)
    @unpack W, D, importanceSampling, stratifiedSampling, importanceSamplingFactor = params 

    U = zeros(W, D * D)

    if importanceSampling == 1
        genExpRandsImportanceSampling!(U, importanceSamplingFactor, ImportanceSamplingWeight)
    elseif stratifiedSampling == 1
        genExpRandsStratified!(U)
    else
        genExpRands!(U)
    end    

    return U 
    
end 