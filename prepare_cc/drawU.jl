
function drawU(SamplingWeight, params)
	@unpack W, D, importanceSampling, importanceSamplingFactor = params

	U = zeros(W, D)

	if importanceSampling == 1
		genExpRandsImportanceSampling!(U, importanceSamplingFactor, SamplingWeight)
	elseif importanceSampling == 2
		genExpRandsStratified!(U, SamplingWeight)
	else
		genExpRands!(U)
	end

	return U

end
