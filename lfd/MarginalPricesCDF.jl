function MarginalPricesCDF(x, d, Aod, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, λData)
	#CDF price to the power 1/μ 
	o1 = 1 + (d - 1) * D
	o2 = D + (d - 1) * D
	pcdf = zeros(length(x), δ_grid_size, 3, 3) #x, domestic/rw/ratio, δ, bound=lower/initial/upper 

	RN = ones(δ_grid_size, 3)
    x_ω = ones(D)
    CDF_Size = length(x)
	for ω ∈ 1:W

		RN[:, 1] = LFD_lower[ω, :] .* (SamplingWeights[ω] / W)
		RN[:, 2] = SamplingWeights[ω] ./ W
		RN[:, 3] = LFD_upper[ω, :] .* (SamplingWeights[ω] / W)

		x_ω[:] = U[ω, o1:o2] .* Aod[:, d] ./ λData[:, d]
		price_domestic = x_ω[d] 
		price_rw = minimum(x_ω[Not(d)])
		price_ratio = price_domestic / price_rw

        smallest_X = searchsortedfirst(x, price_domestic)
        for j = smallest_X:CDF_Size
            @. pcdf[j, 1, :, :] += RN[:, :]
        end

        smallest_X = searchsortedfirst(x, price_rw)
        for j = smallest_X:CDF_Size
            @. pcdf[j, 2, :, :] += RN[:, :]
        end

        smallest_X = searchsortedfirst(x, price_ratio)
        for j = smallest_X:CDF_Size
            @. pcdf[j, 3, :, :] += RN[:, :]
        end

	end
	return pcdf
end
