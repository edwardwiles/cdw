function MarginalPricesCDF(x, d, Aod, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, λData, D, μ_init, μ_up, μ_low, wHat, τ)
	#CDF price to the power 1/μ
	o1 = 1
	o2 = D

	pcdf = zeros(length(x), 3, δ_grid_size, 3) #x, domestic/rw/ratio/, δ, bound=lower/initial/upper
	ppdf = zeros(length(x), 2) #p_domestic/p_rw
	lfd_functional = zeros(length(x), length(x), δ_grid_size, 3) #p_domestic, p_rw, δ, bound=lower/initial/upper
	RN = ones(δ_grid_size, 3)
	x_ω = ones(D)
	CDF_Size = length(x)
	W = length(SamplingWeights)
	Ud = ones(D)
	μ = [μ_low, μ_init, μ_up]

	for ω ∈ 1:W

		@. RN[:, 1] = LFD_lower[ω, :] .* (SamplingWeights[ω] / W)
		@. RN[:, 2] = SamplingWeights[ω] ./ W
		@. RN[:, 3] = LFD_upper[ω, :] .* (SamplingWeights[ω] / W)
		@. Ud[:] = U[ω, o1:o2]
		for δ ∈ 1:δ_grid_size
			for up_down ∈ 1:3
				@. x_ω[:] = wHat[:] .* τ[:, d] .* (Ud[:] ./ Aod[up_down, δ, :, d]) .^ μ[up_down]
				price_domestic = x_ω[d]
				price_rw = minimum(x_ω[Not(d)])
				price_ratio = price_domestic / price_rw

				smallest_X_d = searchsortedfirst(x, price_domestic)
				@. pcdf[smallest_X_d:CDF_Size, 1, δ, up_down] += RN[δ, up_down]


				smallest_X_rw = searchsortedfirst(x, price_rw)
				@. pcdf[smallest_X_rw:CDF_Size, 2, δ, up_down] += RN[δ, up_down]


				smallest_X_ratio = searchsortedfirst(x, price_ratio)
				@. pcdf[smallest_X_ratio:CDF_Size, 3, δ, up_down] += RN[δ, up_down]

				if up_down == 2
					ppdf[min(smallest_X_d, CDF_Size), 1] += RN[δ, up_down]
					ppdf[min(smallest_X_rw, CDF_Size), 2] += RN[δ, up_down]
				end
				lfd_functional[min(smallest_X_d, CDF_Size), min(smallest_X_rw, CDF_Size), δ, up_down] += RN[δ, up_down]

			end
		end

	end

	for δ ∈ 1:δ_grid_size
		for up_down ∈ 1:3
			for i in 1:length(x)
				@. lfd_functional[:, i, δ, up_down] /= ppdf[:, 1] * ppdf[i, 2]
			end
		end
	end

	return pcdf, lfd_functional
end
