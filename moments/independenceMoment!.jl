function IndependenceMoment!(Ū, G, D, η, Ind_Moments, IndMomentOrder, IndCDF_Cells, ν, offset, refIndex1)
	# imposes zero coreelation between Uods, implementation used cached realizations
	# E[U(ref,ref)] = η
	@. G[:, end-offset] += Ū[:, refIndex1] .- η

	K_ = size(Ind_Moments, 2)
	cell_size = size(IndCDF_Cells, 1)
	number_of_correlation_pairs = D * floor(Int, D * (D - 1) / 2)
	square_mean = η[1]^2
	# copy all moments
	@. G[:, end-offset-1-K_+1:end-offset-1] += Ind_Moments[:, :]
	# substract square mean for the correlation moments
	@. G[:, end-offset-1-K_+1:end-offset-1-K_+number_of_correlation_pairs] -= square_mean

	# substract CDF[i] 
	for i ∈ 1:IndMomentOrder
		@. G[:, end-offset-1-K_+number_of_correlation_pairs+i:end-offset-1-K_+number_of_correlation_pairs+i] -= ν[i]
	end
	# calculate and substract the joint CDF
	@inbounds for i ∈ 1:cell_size
		this_cdf = 1
		for o ∈ 1:D
			this_cdf *= ν[floor(Int, IndCDF_Cells[i][o])]
		end
		@. G[:, end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+i:end-offset-1-K_+number_of_correlation_pairs+IndMomentOrder+i] -= this_cdf
	end
end