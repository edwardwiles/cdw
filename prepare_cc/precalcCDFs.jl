
function precalcCDFs(Ū, params, prestep_output)

	@unpack W, D, momentOrder, momentOrderForBaseIndex, SamplingWeight, ForceFrechetMarginal, refIndex1, σHat, baseIndex = params
	@unpack μHat = prestep_output

	# pre-calculate the quantiles for marginal matching with CDF methodology
	CDF_X_for_base = quantile(Ū[:, refIndex1] .* SamplingWeight[:], range(1 / (momentOrderForBaseIndex), (momentOrderForBaseIndex - 1) / (momentOrderForBaseIndex), length = momentOrderForBaseIndex))

	CDF_Moments_Size = 2 * momentOrderForBaseIndex * D + 2 * D

	CDF_Moments = zeros(W, CDF_Moments_Size) # for each od: I. for each CDF point, i) one truncated moment + ii) CDF + II. first moment and (μHat * (1 - σHat)) moment
	CDF_Moments_for_base = zeros(W, 2 * momentOrderForBaseIndex * D)# for  d = baseIndexbaseIndex, for each o for each CDF point, i) one truncated moment + ii) CDF


	Truncated_moment_11_for_base = zeros(W, momentOrderForBaseIndex)
	CDF_11_X_for_base = zeros(W, momentOrderForBaseIndex)


	for ω ∈ 1:W
		smallest_X_for_base = searchsortedfirst(CDF_X_for_base, Ū[ω, refIndex1])

		for i ∈ smallest_X_for_base:momentOrderForBaseIndex
			CDF_11_X_for_base[ω, i] = 1
			Truncated_moment_11_for_base[ω, i] = Ū[ω, refIndex1]^(μHat * (1 - σHat))
		end
	end

	if ForceFrechetMarginal == 1 # Replace the U11 realizations with their expectation, this way the CDF to match does not change with measure change.

		for i ∈ 1:momentOrderForBaseIndex
			CDF_11_X_for_base_expectation = sum(CDF_11_X_for_base[:, i] .* SamplingWeight[:]) / W
			Truncated_moment_11_for_base_expectation = sum(Truncated_moment_11_for_base[:, i] .* SamplingWeight[:]) / W
			@. CDF_11_X_for_base[:, i] = CDF_11_X_for_base_expectation
			@. Truncated_moment_11_for_base[:, i] = Truncated_moment_11_for_base_expectation
		end

	end

	# for d = baseIndex, cach the CDF realizations for momentOrderForBaseIndex points
	for o ∈ 1:D
		o1_base = o
		for i ∈ 1:momentOrderForBaseIndex
			@. CDF_Moments_for_base[:, o+(i-1)*D] += -CDF_11_X_for_base[:, i]
			@. CDF_Moments_for_base[:, momentOrderForBaseIndex*D+o+(i-1)*D] = -Truncated_moment_11_for_base[:, i]
		end
		for ω ∈ 1:W
			smallest_X = searchsortedfirst(CDF_X_for_base, Ū[ω, o1_base])
			for i ∈ smallest_X:momentOrderForBaseIndex
				CDF_Moments_for_base[ω, o+(i-1)*D] += 1
				CDF_Moments_for_base[ω, momentOrderForBaseIndex*D+o+(i-1)*D] += Ū[ω, o1_base]^(μHat * (1 - σHat))
			end
		end
	end

	for o ∈ 1:D
		@. CDF_Moments[:, o] = Ū[:, o] .- Ū[:, refIndex1] # first moment
		@. CDF_Moments[:, D+o] = Ū[:, o]^(μHat * (1 - σHat)) .- Ū[:, refIndex1]^(μHat * (1 - σHat)) # First Moment of inverse
	end

	offset_for_base_index_moments = 2 * D

	@. CDF_Moments[:, offset_for_base_index_moments+1:end] += CDF_Moments_for_base[:, :]

	return CDF_Moments

end
