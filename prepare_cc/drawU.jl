
function drawU(SamplingWeight, params)
    @unpack W, D, importanceSampling, importanceSamplingFactor, UoModel = params 

    sizeU = UoModel == 1 ? D : D*D 
    U = zeros(W, sizeU)

    if importanceSampling == 1
        genExpRandsImportanceSampling!(U, importanceSamplingFactor, SamplingWeight)
    elseif importanceSampling == 2
        genExpRandsStratified!(U, SamplingWeight)
    else
        genExpRands!(U)
    end    

    return U 
    
end 