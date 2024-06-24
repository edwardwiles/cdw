
function precalcIndependence(Ū, params)

	@unpack W, D, baseIndex, SamplingWeight, IndMomentOrder, UoModel = params

	Ind_Moments = zeros(1, 1)
	IndCDF_Cells = Vector{Vector{Int}}(undef, 1)
	#Ind_Moments = zeros(W, D * (D^2 - floor(Int, D * (1 + D) / 2)) + IndMomentOrder + IndMomentOrder^D)

	Ind_Moments_size = IndMomentOrder + (D^2 - floor(Int, D * (1 + D) / 2)) * (IndMomentOrder^2)

	if UoModel == 0
		Ind_Moments_size += D * (D^2 - floor(Int, D * (1 + D) / 2))
	else
		Ind_Moments_size += (D^2 - floor(Int, D * (1 + D) / 2))
	end



	Ind_Moments = zeros(W, Ind_Moments_size)

	# Cache U_od U_o'd for the E[U_od U_o'd - E[U_od] ^2] = 0 moment 
	if UoModel == 1
		idx_corss_moment = 0
		for o ∈ 1:D
			for c ∈ 1:D
				if c > o
					idx_corss_moment += 1
					@. Ind_Moments[:, idx_corss_moment] = Ū[:, o] .* Ū[:, c]
				end
			end
		end
	else

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
	end

	# Joint CDF = Product of Marginal CFDs

	IndCDF_K = range(1, IndMomentOrder, IndMomentOrder)
	IndCDF_X = quantile(Ū[:, 1] .* SamplingWeight[:], range(1 / IndMomentOrder, (IndMomentOrder - 1) / IndMomentOrder, length = IndMomentOrder))

	# CDF U11
	offset = D * (D^2 - floor(Int, D * (1 + D) / 2))
	cell_size = size(IndCDF_X, 1)
	for ω ∈ 1:W
		@inbounds for i ∈ 1:cell_size
			Ind_Moments[ω, offset+i] = (Ū[ω, 1] < IndCDF_X[i]) ? 1 : 0
		end
	end

	#Pairewise CFDs

	PairWiseIndCDF_Cells = collect(with_replacement_combinations(IndCDF_K, 2))
	cell_size = size(PairWiseIndCDF_Cells, 1)

	# D^2 x IndMomentOrder^2
	#TO DO, make it D^4xIndMomentOrder^2 
	for ω ∈ 1:W
		idx_corss_moment = 0
		@inbounds for i ∈ 1:cell_size
			this_cdf = 1
			for o_1 ∈ 1:D
				if UoModel == 0
					o1 = o_1 + (baseIndex - 1) * D
				else
					o1 = o_1
				end
				this_cdf *= Ū[ω, o1] < IndCDF_X[floor(Int, PairWiseIndCDF_Cells[i][1])] ? 1 : 0
				for o_2 ∈ 1:D
					if o_2 > o_1
						if UoModel == 0
							o2 = o_2 + (baseIndex - 1) * D
						else
							o2 = o_2
						end
						this_cdf *= (Ū[ω, o2] < IndCDF_X[floor(Int, PairWiseIndCDF_Cells[i][2])]) ? 1 : 0
						idx_corss_moment += 1
						Ind_Moments[ω, offset+idx_corss_moment] = this_cdf
					end
				end

			end
		end
	end

	#=
	# Joint CDF
	IndCDF_Cells = collect(with_replacement_combinations(IndCDF_K, D))
	cell_size = size(IndCDF_Cells, 1)
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
	=#
	return (Ind_Moments, PairWiseIndCDF_Cells)
end
