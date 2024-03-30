
function master_prepare_cc(globalParams)

    @unpack seedU, W, D, stratifiedSampling, importanceSampling, importanceSamplingFactor = globalParams

    # set seed
    Random.seed!(seedU)

    # draw base U matrix from exp(1)
    U = zeros(W, D * D)
    ImportanceSamplingWeight = ones(W)

    if importanceSampling == 1
        genExpRandsImportanceSampling!(U, importanceSamplingFactor, ImportanceSamplingWeight)
    elseif stratifiedSampling == 1
        genExpRandsStratified!(U)
    else
        genExpRands!(U)
    end    

end 