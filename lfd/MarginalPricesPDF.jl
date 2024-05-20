function MarginalPricesPDF(x, d, Aod, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, λData)
	#CDF price to the power 1/μ 
	o1 = 1 + (d - 1) * D
	o2 = D + (d - 1) * D
	ppdf = zeros(length(x), δ_grid_size, 3, 3) #x, δ, bound=lower/initial/upper, domestic/rw/ratio 

	IndRN = ones(3, 3)
	Ind = zeros(3)

	for ω ∈ 1:W

		for j ∈ 1:length(x)-1
			x_ω = U[ω, o1:o2] .* Aod[:, d]
			price_domestic = x_ω[d] / λData[d, d]
			price_rw = minimum(x_ω[Not(d)] ./ λData[Not(d), d])
			Ind[1] = price_domestic >= x[j] && price_domestic <= x[j+1] ? 1 : 0
			Ind[2] = price_rw >= x[j] && price_rw <= x[j+1] ? 1 : 0
			Ind[3] = price_domestic / price_rw >= x[j] && price_domestic / price_rw <= x[j+1] ? 1 : 0
			for δ ∈ 1:δ_grid_size
				@. IndRN[1, :] = Ind[:] .* (LFD_lower[ω, δ] * SamplingWeights[ω] / W)
				@. IndRN[2, :] = Ind[:] .* (SamplingWeights[ω] / W)
				@. IndRN[3, :] = Ind[:] .* (LFD_upper[ω, δ] * SamplingWeights[ω] / W)

				@. ppdf[j, δ, :, :] += IndRN_ω[:]
			end
		end
	end
	return ppdf
end
