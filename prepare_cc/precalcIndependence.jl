
function precalcIndependence(Ū, params)

	@unpack W, D, baseIndex, SamplingWeight, IndMomentOrder, refIndex1 = params

	Ind_Moments = zeros(1, 1)
	IndCDF_Cells = Vector{Vector{Int}}(undef, 1)
	#Ind_Moments = zeros(W, D * (D^2 - floor(Int, D * (1 + D) / 2)) + IndMomentOrder + IndMomentOrder^D)

	Ind_Moments_size = IndMomentOrder + (D^2 - floor(Int, D * (1 + D) / 2)) * (IndMomentOrder^2)

	Ind_Moments_size += (D^2 - floor(Int, D * (1 + D) / 2))



	Ind_Moments = zeros(W, Ind_Moments_size)

	# Cache U_od U_o'd for the E[U_od U_o'd - E[U_od] ^2] = 0 moment
	idx_corss_moment = 0
	for o ∈ 1:D
		for c ∈ 1:D
			if c > o
				idx_corss_moment += 1
				@. Ind_Moments[:, idx_corss_moment] = Ū[:, o] .* Ū[:, c]
			end
		end
	end

	# Joint CDF = Product of Marginal CFDs

	IndCDF_K = range(1, IndMomentOrder, IndMomentOrder)
	IndCDF_X = quantile(@view(Ū[:, refIndex1]) .* SamplingWeight[:], range(1 / (IndMomentOrder+1), IndMomentOrder / (IndMomentOrder+1), length = IndMomentOrder))

	# CDF U11
	offset = D^2 - floor(Int, D * (1 + D) / 2)
	cell_size = size(IndCDF_X, 1)
	for ω ∈ 1:W
		@inbounds for i ∈ 1:cell_size
			Ind_Moments[ω, offset+i] = (Ū[ω, refIndex1] <= IndCDF_X[i]) ? 1 : 0
		end
	end

	#Pairewise CFDs
	offset += cell_size
	PairWiseIndCDF_Cells =vec(collect(Base.Iterators.product(Base.Iterators.repeated(IndCDF_K, 2)...)))
	@show PairWiseIndCDF_Cells
	cell_size = size(PairWiseIndCDF_Cells, 1)

	# D^2 x IndMomentOrder^2
	#TO DO, make it D^4xIndMomentOrder^2
	for ω ∈ 1:W
		idx_corss_moment = 0
		@inbounds for i ∈ 1:cell_size

			for o_1 ∈ 1:D
				o1 = o_1
				this_cdf_o1 = Ū[ω, o1] < IndCDF_X[floor(Int, PairWiseIndCDF_Cells[i][1])] ? 1 : 0
				for o_2 ∈ 1:D
					if o_2 > o_1
						o2 = o_2
						this_cdf_o2 = (Ū[ω, o2] < IndCDF_X[floor(Int, PairWiseIndCDF_Cells[i][2])]) ? 1 : 0
						this_cdf = this_cdf_o1*this_cdf_o2
						idx_corss_moment += 1
						Ind_Moments[ω, offset+idx_corss_moment] = this_cdf
					end
				end

			end
		end
	end

	@show IndCDF_X
	@show IndCDF_K
	return (Ind_Moments, PairWiseIndCDF_Cells)
end
