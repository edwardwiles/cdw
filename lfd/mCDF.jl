function UCDF(x, o, d, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, D, UoModel)
	o1 = o + (d - 1) * D
	if UoModel == 1
		o1 = o
	end
	ucdf = zeros(length(x), δ_grid_size, 3) # x, δ, lower/initial/upper
	RN = ones(δ_grid_size, 3)
	CDF_Size = length(x)
	W = length(SamplingWeights)
	
	for ω ∈ 1:W

		@. RN[:, 1] = LFD_lower[ω, :] .* (SamplingWeights[ω] / W)
		@. RN[:, 2] = SamplingWeights[ω] ./ W
		@. RN[:, 3] = LFD_upper[ω, :] .* (SamplingWeights[ω] / W)


		smallest_X = searchsortedfirst(x, U[ω, o1])
		for j ∈ smallest_X:CDF_Size
			@. ucdf[j, :, :] += RN[:, :]
		end

	end
	return ucdf
end
