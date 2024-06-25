function pairewiseIndependenceMoment!(Ū, G, D, η, Ind_Moments, IndMomentOrder, IndCDF_Cells, ν, offset, refIndex1, UoModel, W)
	# imposes zero coreelation between Uods, implementation used cached realizations
	# E[U(ref,ref)] = η
	@. G[:, end-offset] += Ū[1:W, refIndex1] .- η[1]

	K_ = size(Ind_Moments, 2)
	cell_size = size(IndCDF_Cells, 1)
	number_of_correlation_pairs = (UoModel== 1 ? 1 : D) * floor(Int, D * (D - 1) / 2)
    number_of_correlation_pairs_baseIndex = floor(Int, D * (D - 1) / 2)
	square_mean = η[1]^2
	# copy all moments
	@. G[:, end-offset-1-K_+1:end-offset-1] += Ind_Moments[1:W, :]
	# substract square mean for the correlation moments
	@. G[:, end-offset-1-K_+1:end-offset-1-K_+number_of_correlation_pairs] -= square_mean

	# substract CDF[i] 
	for i ∈ 1:IndMomentOrder
		@. G[:, end-offset-1-K_+number_of_correlation_pairs+i:end-offset-1-K_+number_of_correlation_pairs+i] -= ν[i]
	end
	# calculate and substract the pairwise CDF
	@inbounds for i ∈ 1:cell_size
		this_cdf = ν[floor(Int, IndCDF_Cells[i][1])] * ν[floor(Int, IndCDF_Cells[i][2])]
	    @. G[:, end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+(i-1)*number_of_correlation_pairs_baseIndex+1:end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+i*number_of_correlation_pairs_baseIndex] -= this_cdf
	end
end

function pairewiseIndependenceMoment_jac!(Ū, jac_G, D, η, Ind_Moments, IndMomentOrder, IndCDF_Cells, ν, offset, refIndex1, η_index, UoModel)
	# imposes zero coreelation between Uods, implementation used cached realizations
	# E[U(ref,ref)] = η
	@. jac_G[:, end-offset, η_index] = -1

	K_ = size(Ind_Moments, 2)
	cell_size = size(IndCDF_Cells, 1)
	number_of_correlation_pairs = (UoModel== 1 ? 1 : D)  * floor(Int, D * (D - 1) / 2)
    number_of_correlation_pairs_baseIndex = floor(Int, D * (D - 1) / 2)
	# substract square mean for the correlation moments
	@. jac_G[:, end-offset-1-K_+1:end-offset-1-K_+number_of_correlation_pairs, η_index] = -2 * η[1]

	# substract CDF[i] 
	for i ∈ 1:IndMomentOrder
		@. jac_G[:, end-offset-1-K_+number_of_correlation_pairs+i:end-offset-1-K_+number_of_correlation_pairs+i, end-IndMomentOrder+i] = -1
	end
    # calculate and substract the pairwise CDF
	@inbounds for i ∈ 1:cell_size
        j1 = floor(Int, IndCDF_Cells[i][1])
        j2 = floor(Int, IndCDF_Cells[i][2])

		this_cdf = ν[j1] * ν[j2]
        
        if j1 == j2
            ∂CDF∂CDF_j = 2*ν[j1]
	        @. jac_G[:, end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+(i-1)*number_of_correlation_pairs_baseIndex+1:end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+i*number_of_correlation_pairs_baseIndex, end-IndMomentOrder+j1] = -∂CDF∂CDF_j
        else
            ∂CDF∂CDF_j1 = ν[j2]
            ∂CDF∂CDF_j2 = ν[j1]
            @. jac_G[:, end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+(i-1)*number_of_correlation_pairs_baseIndex+1:end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+i*number_of_correlation_pairs_baseIndex, end-IndMomentOrder+j1] = -∂CDF∂CDF_j1
            @. jac_G[:, end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+(i-1)*number_of_correlation_pairs_baseIndex+1:end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+i*number_of_correlation_pairs_baseIndex, end-IndMomentOrder+j2] = -∂CDF∂CDF_j2
        end
    end
end
