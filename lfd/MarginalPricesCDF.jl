function MarginalPricesCDF(x, d, Aod, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, λData, D)
	#CDF price to the power 1/μ 
	o1 = 1 + (d - 1) * D
	o2 = D + (d - 1) * D
	pcdf = zeros(length(x), 3, δ_grid_size, 3) #x, domestic/rw/ratio, δ, bound=lower/initial/upper 

	RN = ones(δ_grid_size, 3)
	x_ω = ones(D)
	CDF_Size = length(x)
	W = length(SamplingWeights)
	Ud = ones(D)

	for ω ∈ 1:W

		@. RN[:, 1] = LFD_lower[ω, :] .* (SamplingWeights[ω] / W)
		@. RN[:, 2] = SamplingWeights[ω] ./ W
		@. RN[:, 3] = LFD_upper[ω, :] .* (SamplingWeights[ω] / W)
		@. Ud[:] = U[ω, o1:o2]
		for δ ∈ 1:δ_grid_size
			for up_down ∈ 1:3
				@. x_ω[:] = Ud[:] .* Aod[up_down, δ, :, d] ./ λData[:, d]
				price_domestic = x_ω[d]
				price_rw = minimum(x_ω[Not(d)])
				price_ratio = price_domestic / price_rw

				smallest_X = searchsortedfirst(x, price_domestic)
				@. pcdf[smallest_X:CDF_Size, 1, δ, up_down] += RN[δ, up_down]


				smallest_X = searchsortedfirst(x, price_rw)
				@. pcdf[smallest_X:CDF_Size, 2, δ, up_down] += RN[δ, up_down]


				smallest_X = searchsortedfirst(x, price_ratio)
				@. pcdf[smallest_X:CDF_Size, 3, δ, up_down] += RN[δ, up_down]

			end
		end

	end
	return pcdf
end
