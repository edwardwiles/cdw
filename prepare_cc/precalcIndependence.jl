
function precalcIndependence(Ū, params)

	@unpack W, D, baseIndex, SamplingWeight, IndMomentOrder = params

	Ind_Moments = zeros(1, 1)
	IndCDF_Cells = Vector{Vector{Int}}(undef, 1)
	Ind_Moments = zeros(W, D * (D^2 - floor(Int, D * (1 + D) / 2)) + IndMomentOrder  + IndMomentOrder ^D)

	# Cache U_od U_o'd for the E[U_od U_o'd - E[U_od] ^2] = 0 moment 
	idx_corss_moment = 0
	for d ∈ 1:D
		for o ∈ 1:D
			o1 = o + (d - 1) * D
			for c ∈ 1:D
				for f ∈ 1:D
					c1 = c + (f - 1) * D
					if c1 > o1 && d == f # condition on Uod, Uo'd only
						idx_corss_moment += 1
						@. Ind_Moments[:, idx_corss_moment] = Ū[:, o1] .* Ū[:, c1]
					end
				end
			end
		end
	end

	# Joint CDF = Product of Marginal CFDs

	IndCDF_K = range(1, IndMomentOrder , IndMomentOrder)
	IndCDF_X = quantile(Ū[:, 1] .* SamplingWeight[:], range(1 / IndMomentOrder, (IndMomentOrder - 1) / IndMomentOrder, length = IndMomentOrder))

	# CDF U11
	offset = D * (D^2 - floor(Int, D * (1 + D) / 2))
    cell_size = size(IndCDF_X,1)
	for ω ∈ 1:W
		@inbounds for  i ∈ 1:cell_size 
			Ind_Moments[ω, offset+i] = (Ū[ω, 1] < IndCDF_X[i]) ? 1 : 0
		end
	end

	# Joint CDF
	IndCDF_Cells = collect(with_replacement_combinations(IndCDF_K, D))
	offset = offset + IndMomentOrder 
	for ω ∈ 1:W
		idx_corss_moment = 0
		@inbounds for i ∈ 1:cell_size
			this_cdf = 1
			for o ∈ 1:D
				o1 = o + (baseIndex - 1) * D
				this_cdf *= (Ū[ω, o1] < IndCDF_X[floor(Int, IndCDF_Cells[i][o])]) ? 1 : 0
			end
			idx_corss_moment += 1
			Ind_Moments[ω, offset+idx_corss_moment] = this_cdf
		end
	end
	return (Ind_Moments, IndCDF_Cells)
end