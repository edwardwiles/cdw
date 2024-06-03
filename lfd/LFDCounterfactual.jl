function LFDCounterFactual(lfd_output, cc_output, prestep_output, prep_output, params)
	# function that simulates IID Us for all countries from the (same) LFD marginal of refIndex.
	# the function then calculates the moments and the counterfactuals with this new U
	# This is to test:
	# 1- that all moments are matched under the simulated U
	#    For example we were worried that gamma might not be matced because of the small prices issue.
	# 2- In particular, we check also whether the od CFDs match with a Kolmogorov Smirnov test
	# 3- Finally, we compare the counterfactuals 
	@unpack θ_initial, U, γ, numMoments, δ_grid, outer_constr_index, file_name = prep_output
	@unpack μHat, wHat, λPrime, wPrimeHat, γHat, γPrimeHat, cHat = prestep_output
	@unpack W, baseIndex, refIndex1, θConstant, σHat, sameMarginalsMoment, independenceMoment = params
	@unpack Θ_upper, κ_upper, Θ_lower, κ_lower = cc_output
	@unpack LFD_upper, LFD_lower = lfd_output

	D = length(γ.L)
	δ_grid_size = length(δ_grid)
	Ū = γ.Ū

	V = zeros(W, D * D)
	rand!(V)

	# for each delta, we caluculate the CDF of U_refIndex1 for the upper and lower LFD  
	CDF_Size = 1000 # number of points in the empirical marginal CDF 
	max_CDF = 0.99999 # cap on CDF to avoid getting too close to one due to exrapolation
	CDF_X = zeros(CDF_Size) # the y-axis of the empirical CDF : CDF(X)


	U_CDF_control = zeros(CDF_Size, D^2) # for control, to see if the numerical procedure restitutes the U CDF
	U_CDF_exact = zeros(CDF_Size, D^2)

	U_CDF_up = zeros(CDF_Size, D^2, δ_grid_size)
	U_CDF_down = zeros(CDF_Size, D^2, δ_grid_size)

	# Transfrom f(U) to the be uniform [0.1], this will help with the extrapolation
	X_for_CDF = zeros(W, D^2) # 
	for od ∈ 1:D^2
		@. X_for_CDF[:, od] = exp.(-1 .* Ū[:, od])
	end
	# CDF(X)
	CDF_X = quantile(exp.(-1 .* Ū[:, refIndex1]), range(1 / (CDF_Size), (CDF_Size - 1) / (CDF_Size), length = CDF_Size))


	for od ∈ 1:D^2
		@. U_CDF_exact[:, od] = CDF_X[:]
	end


	for ω ∈ 1:W
		for od ∈ 1:D^2
			smallest_X = searchsortedfirst(CDF_X, X_for_CDF[ω, od])

			@. U_CDF_control[smallest_X:CDF_Size, od] += 1 / W
			@. U_CDF_up[smallest_X:CDF_Size, od, :] += LFD_upper[ω, :] ./ W
			@. U_CDF_down[smallest_X:CDF_Size, od, :] += LFD_lower[ω, :] ./ W

		end
	end


	#calculate the K-S test 

	KS_Test = zeros(D^2, δ_grid_size, 2)
	for δ ∈ 1:δ_grid_size
		for od ∈ 1:D^2
			KS_Test[od, δ, 1] = maximum(abs.(U_CDF_up[:, od, δ] .- U_CDF_up[:, refIndex1, δ]))
			KS_Test[od, δ, 2] = maximum(abs.(U_CDF_down[:, od, δ] .- U_CDF_down[:, refIndex1, δ]))
		end

		savefig(heatmap(reshape(KS_Test[:, δ, 1], (D, D)), fc = cgrad([:white, :dodgerblue4])),
			string("upper_KS_delta_", δ_grid[δ], "_", file_name, ".png"))

		savefig(heatmap(reshape(KS_Test[:, δ, 2], (D, D)), fc = cgrad([:white, :dodgerblue4])),
			string("lower_KS_delta_", δ_grid[δ], "_", file_name, ".png"))
	end

	writedlm(string("KSTest_", file_name), [(KS_Test[:, :, 1])' (KS_Test[:, :, 2])'], ',')



	for i ∈ 1:δ_grid_size
		savefig(
			plot(
				xlabel = "exp(- ̄U)",
				ylabel = "CDF",
				CDF_X,
				[U_CDF_up[:, refIndex1, i] U_CDF_down[:, refIndex1, i] U_CDF_up[:, baseIndex+(baseIndex-1)*D, i] U_CDF_down[:, baseIndex+(baseIndex-1)*D, i]],
				label = ["Upper CDFrefIndex" "Lower CDFrefIndex" "Upper CDFbaseIndex,baseIndex" "Lower CDFbaseIndex,baseIndex"],
				title = string("Marginals for δ = ", δ_grid[i]),
			),
			string("marginals_delta_baseIndex,baseIndex_", δ_grid[i], "_", file_name, ".png"),
		)

		savefig(
			plot(
				xlabel = "exp(- ̄U)",
				ylabel = "CDF",
				CDF_X,
				[U_CDF_up[:, refIndex1, i] U_CDF_down[:, refIndex1, i] U_CDF_up[:, 1+(baseIndex-1)*D, i] U_CDF_down[:, 1+(baseIndex-1)*D, i]],
				label = ["Upper CDFrefIndex" "Lower CDFrefIndex" "Upper CDF1,baseIndex,baseIndex" "Lower CDF1,baseIndex,baseIndex"],
				title = string("Marginals for δ = ", δ_grid[i]),
			),
			string("marginals_delta_U1,baseIndex_", δ_grid[i], "_", file_name, ".png"),
		)
	end




	U_exact = zeros(W, D * D)
	Ū_exact = zeros(W, D * D)

	U_control = zeros(W, D * D)
	Ū_control = zeros(W, D * D)

	U_LFD_up = zeros(W, D * D)
	Ū_LFD_up = zeros(W, D * D)

	U_LFD_down = zeros(W, D * D)
	Ū_LFD_down = zeros(W, D * D)

	Inverse_CDF_up = 0
	Inverse_CDF_down = 0
	Inverse_CDF_control = 0

	Ind_Moments = zeros(1, 1)
	IndCDF_Cells = Vector{Vector{Int}}(undef, 1)
	CDF_Moments = zeros(1, 1)

	Ind_Moments_up = zeros(1, 1)
	IndCDF_Cells_up = Vector{Vector{Int}}(undef, 1)
	CDF_Moments_up = zeros(1, 1)

	Ind_Moments_down = zeros(1, 1)
	IndCDF_Cells_down = Vector{Vector{Int}}(undef, 1)
	CDF_Moments_down = zeros(1, 1)


	#simulate Ū_exact & Ū_control

	for ω ∈ 1:W
		for od ∈ 1:D^2
			Inverse_CDF_idx = searchsortedfirst(U_CDF_exact[:, refIndex1], V[ω, od])
			if Inverse_CDF_idx == 1
				Inverse_CDF_exact = V[ω, od] * CDF_X[1] / U_CDF_exact[1, refIndex1]
			elseif Inverse_CDF_idx == CDF_Size + 1
				Inverse_CDF_exact = (V[ω, od] - U_CDF_exact[end, 1]) * (1 - CDF_X[end]) / (1 - U_CDF_exact[end, refIndex1]) + CDF_X[end]
			else
				Slope_exact = (CDF_X[Inverse_CDF_idx] - CDF_X[Inverse_CDF_idx-1]) / (U_CDF_exact[Inverse_CDF_idx, refIndex1] - U_CDF_exact[Inverse_CDF_idx-1, refIndex1])
				Inverse_CDF_exact = (V[ω, od] - U_CDF_exact[Inverse_CDF_idx-1, refIndex1]) * Slope_exact + CDF_X[Inverse_CDF_idx-1]
			end

			# transform back to exponential 
			Ū_exact[ω, od] = -log(Inverse_CDF_exact)

			Inverse_CDF_idx = searchsortedfirst(U_CDF_control[:, refIndex1], V[ω, od])
			if Inverse_CDF_idx == 1
				Inverse_CDF_control = V[ω, od] * CDF_X[1] / U_CDF_control[1, refIndex1]
			elseif Inverse_CDF_idx == CDF_Size + 1
				Inverse_CDF_control = (V[ω, od] - U_CDF_control[end, refIndex1]) * (1 - CDF_X[end]) / (1 - U_CDF_control[end, refIndex1]) + CDF_X[end]
			else
				Slope_control = (CDF_X[Inverse_CDF_idx] - CDF_X[Inverse_CDF_idx-1]) / (U_CDF_control[Inverse_CDF_idx, refIndex1] - U_CDF_control[Inverse_CDF_idx-1, refIndex1])
				Inverse_CDF_control = (V[ω, od] - U_CDF_control[Inverse_CDF_idx-1, refIndex1]) * Slope_control + CDF_X[Inverse_CDF_idx-1]
			end

			# transform back to exponential
			Ū_control[ω, od] = -log(Inverse_CDF_control)
		end
	end

	for d ∈ 1:D
		for o ∈ 1:D
			o1 = o + (d - 1) * D
			@. U_control[:, o1] = Ū_control[:, o1] .* cHat[o, d]
		end
	end

	if θConstant == 1
		@. U_control[:] = U_control[:] .^ (-μHat) # if theta doesn't vary, then much faster to precalculate this matrix 
		# reduction in computation time due to non-integer exponent substantially dominates higher memory usage
	end

	Uσ_control = U_control .^ (1 - σHat) # precalculate 



	for d ∈ 1:D
		for o ∈ 1:D
			o1 = o + (d - 1) * D
			@. U_exact[:, o1] = Ū_exact[:, o1] .* cHat[o, d]
		end
	end

	if θConstant == 1
		@. U_exact[:] = U_exact[:] .^ (-μHat) # if theta doesn't vary, then much faster to precalculate this matrix 
		# reduction in computation time due to non-integer exponent substantially dominates higher memory usage
	end

	Uσ_exact = U_exact .^ (1 - σHat) # precalculate 


	CDF_Moments_control= zeros(1)
	CDF_Moments_exact= zeros(1)
	Ind_Moments_control= zeros(1)
	IndCDF_Cells_control= zeros(1)
	Ind_Moments_exact= zeros(1)
	IndCDF_Cells_exact= zeros(1)


	if sameMarginalsMoment == 1
		CDF_Moments_control = precalcCDFs(Ū_control, params, prestep_output)
		CDF_Moments_exact = precalcCDFs(Ū_exact, params, prestep_output)
	end

	if independenceMoment == 1
		Ind_Moments_control, IndCDF_Cells_control = precalcIndependence(Ū_LFD_control, params)
		Ind_Moments_exact, IndCDF_Cells_exact = precalcIndependence(Ū_LFD_exact, params)
	end

	# update the fields that depend on ̄U

	γ_control= (wHat = γ.wHat,
	L = γ.L,
	LPrime = γ.LPrime,
	τ = γ.τ,
	τPrime = γ.τPrime,
	P = γ.P,
	PMM = γ.PMM,
	baseIndex = γ.baseIndex,
	indicators = γ.indicators,
	wPrimeHat = γ.wPrimeHat,
	Uσ = Uσ_control,
	Ū = Ū_control,
	μHat = γ.μHat,
	D = γ.D,
	CDF_Moments = CDF_Moments_control,
	Ind_Moments = Ind_Moments_control,
	cHat = γ.cHat,
	IndCDF_Cells = γ.IndCDF_Cells,
	SamplingWeights = γ.SamplingWeights,
	refIndex1 = γ.refIndex1)


	γ_exact= (wHat = γ.wHat,
	L = γ.L,
	LPrime = γ.LPrime,
	τ = γ.τ,
	τPrime = γ.τPrime,
	P = γ.P,
	PMM = γ.PMM,
	baseIndex = γ.baseIndex,
	indicators = γ.indicators,
	wPrimeHat = γ.wPrimeHat,
	Uσ = Uσ_exact,
	Ū = Ū_exact,
	μHat = γ.μHat,
	D = γ.D,
	CDF_Moments = CDF_Moments_exact,
	Ind_Moments = Ind_Moments_exact,
	cHat = γ.cHat,
	IndCDF_Cells = γ.IndCDF_Cells,
	SamplingWeights = γ.SamplingWeights,
	refIndex1 = γ.refIndex1)


	# check that indeed the moments are matched at the initial theta 

	obj = PsiObjectiveBundleDelta(
		#δ = 1,
		#find_smallest = true,
		γ = γ,
		(moments!) = EK_moments!,
		#moments_jacobian! = rust_moments_jacobian!,
		d = numMoments,
		outer_constr_index = outer_constr_index,
		inequality_index = Int64[],
		l = size(θ_initial, 1),
		U = U,
		N = 100,
		outer_loop_opt = "ek_outer_loop_options.opt",
		inner_loop_opt = "ek_inner_loop_options.opt",
		lower_limit = -50)


	obj_control = PsiObjectiveBundleDelta(
		#δ = 1,
		#find_smallest = true,
		γ = γ_control,
		(moments!) = EK_moments!,
		#moments_jacobian! = rust_moments_jacobian!,
		d = numMoments,
		outer_constr_index = outer_constr_index,
		inequality_index = Int64[],
		l = size(θ_initial, 1),
		U = U_control,
		N = 100,
		outer_loop_opt = "ek_outer_loop_options.opt",
		inner_loop_opt = "ek_inner_loop_options.opt",
		lower_limit = -50)


	obj_exact = PsiObjectiveBundleDelta(
		#δ = 1,
		#find_smallest = true,
		γ = γ_exact,
		(moments!) = EK_moments!,
		#moments_jacobian! = rust_moments_jacobian!,
		d = numMoments,
		outer_constr_index = outer_constr_index,
		inequality_index = Int64[],
		l = size(θ_initial, 1),
		U = U_exact,
		N = 100,
		outer_loop_opt = "ek_outer_loop_options.opt",
		inner_loop_opt = "ek_inner_loop_options.opt",
		lower_limit = -50)


	G_exact = zeros(W, numMoments)
	K_exact = zeros(W, 1)

	G_control = zeros(W, numMoments)
	K_control = zeros(W, 1)

	G = zeros(W, numMoments)
	K = zeros(W, 1)

	EK_moments!(K, G, θ_initial, U, obj)
	EK_moments!(K_control, G_control, θ_initial, U_control, obj_control)
	EK_moments!(K_exact, G_exact, θ_initial, U_exact, obj_exact)



	MomentsMean = zeros(numMoments)
	MomentsMean_exact = zeros(numMoments)
	MomentsMean_control = zeros(numMoments)

	MomentsVar = zeros(numMoments)
	MomentsVar_exact = zeros(numMoments)
	MomentsVar_control = zeros(numMoments)

	MomentsTest = zeros(numMoments, 9)

	MomentsMean = mean(G, dims = 1)
	MomentsVar = mean(G .* G, dims = 1) .- MomentsMean .* MomentsMean

	MomentsMean_control = mean(G_control, dims = 1)
	MomentsVar_control = mean(G_control .* G_control, dims = 1) .- MomentsMean_control .* MomentsMean_control

	MomentsMean_exact = mean(G_exact, dims = 1)
	MomentsVar_exact = mean(G_exact .* G_exact, dims = 1) .- MomentsMean_exact .* MomentsMean_exact


	@. MomentsTest[:, 1] = MomentsMean[:]
	@. MomentsTest[:, 2] = sqrt.(MomentsVar[:] ./ W)
	@. MomentsTest[:, 3] = indicative.(abs.(MomentsMean[:]) - 2 .* sqrt.(MomentsVar[:] ./ W))

	@. MomentsTest[:, 4] = MomentsMean_exact[:]
	@. MomentsTest[:, 5] = sqrt.(MomentsVar_exact[:] ./ W)
	@. MomentsTest[:, 6] = indicative.(abs.(MomentsMean_exact[:]) - 2 .* sqrt.(MomentsVar_exact[:] ./ W))


	@. MomentsTest[:, 7] = MomentsMean_control[:]
	@. MomentsTest[:, 8] = sqrt.(MomentsVar_control[:] ./ W)
	@. MomentsTest[:, 9] = indicative.(abs.(MomentsMean_control[:]) - 2 .* sqrt.(MomentsVar_control[:] ./ W))

	writedlm(string("MomentsTest_", file_name), MomentsTest, ',')




	G_upper = zeros(W, numMoments)
	G_lower = zeros(W, numMoments)
	K_upper = zeros(W, 1)
	K_lower = zeros(W, 1)

	G_upper_LFD = zeros(W, numMoments)
	G_lower_LFD = zeros(W, numMoments)
	K_upper_LFD = zeros(W, 1)
	K_lower_LFD = zeros(W, 1)

	κ_upper = zeros(δ_grid_size)
	κ_upper_LFD = zeros(δ_grid_size)

	κ_lower = zeros(δ_grid_size)
	κ_lower_LFD = zeros(δ_grid_size)

	for δ ∈ 1:δ_grid_size

		# Simulate the LFD Us for all ods, using the CDF of refIndex1
		for ω ∈ 1:W
			for od ∈ 1:D^2
				Inverse_CDF_idx = searchsortedfirst(U_CDF_up[:, refIndex1, δ], V[ω, od])
				if Inverse_CDF_idx == 1
					Inverse_CDF_up = V[ω, od] * CDF_X[1] / U_CDF_up[1, refIndex1, δ]
				elseif Inverse_CDF_idx == CDF_Size + 1
					Inverse_CDF_up = (V[ω, od] - U_CDF_up[end, refIndex1, δ]) * (1 - CDF_X[end]) / (1 - U_CDF_up[end, refIndex1, δ]) + CDF_X[end]
					Inverse_CDF_up = max(CDF_X[end], min(Inverse_CDF_up, max_CDF))# ensure monotony + capping at max_CDF
				else
					Slope_up = (CDF_X[Inverse_CDF_idx] - CDF_X[Inverse_CDF_idx-1]) / (U_CDF_up[Inverse_CDF_idx, refIndex1, δ] - U_CDF_up[Inverse_CDF_idx-1, refIndex1, δ])
					Inverse_CDF_up = (V[ω, od] - U_CDF_up[Inverse_CDF_idx-1, refIndex1, δ]) * Slope_up + CDF_X[Inverse_CDF_idx-1]
					Inverse_CDF_up = min(max(Inverse_CDF_up, CDF_X[Inverse_CDF_idx-1]), CDF_X[Inverse_CDF_idx]) # ensure monotony
				end
				# transform back to exponential
				Ū_LFD_up[ω, od] = -log(Inverse_CDF_up)


				Inverse_CDF_idx = searchsortedfirst(U_CDF_down[:, 1], V[ω, od])
				if Inverse_CDF_idx == 1
					Inverse_CDF_down = V[ω, od] * CDF_X[1] / U_CDF_down[1, 1]
				elseif Inverse_CDF_idx == CDF_Size + 1
					Inverse_CDF_down = (V[ω, od] - U_CDF_down[end, 1]) * (1 - CDF_X[end]) / (1 - U_CDF_down[end, 1]) + CDF_X[end]
					Inverse_CDF_down = min(max(Inverse_CDF_down, CDF_X[end]), max_CDF) # ensure monotony + capping at max_CDF
				else
					Slope_down = (CDF_X[Inverse_CDF_idx] - CDF_X[Inverse_CDF_idx-1]) / (U_CDF_down[Inverse_CDF_idx, 1] - U_CDF_down[Inverse_CDF_idx-1, 1])
					Inverse_CDF_down = (V[ω, od] - U_CDF_down[Inverse_CDF_idx-1, 1]) * Slope_down + CDF_X[Inverse_CDF_idx-1]
					Inverse_CDF_down = min(max(Inverse_CDF_down, CDF_X[Inverse_CDF_idx-1]), CDF_X[Inverse_CDF_idx]) # ensure monotony
				end
				# transform back to exponential
				Ū_LFD_down[ω, od] = -log(Inverse_CDF_down)
			end

		end

		for d ∈ 1:D
			for o ∈ 1:D
				o1 = o + (d - 1) * D
				@. U_LFD_down[:, o1] = Ū_LFD_down[:, o1] .* cHat[o, d]
			end
		end

		if θConstant == 1
			@. U_LFD_down[:] = U_LFD_down[:] .^ (-μHat) # if theta doesn't vary, then much faster to precalculate this matrix 
			# reduction in computation time due to non-integer exponent substantially dominates higher memory usage
		end

		Uσ_LFD_down = U_LFD_down .^ (1 - σHat) # precalculate 


		for d ∈ 1:D
			for o ∈ 1:D
				o1 = o + (d - 1) * D
				@. U_LFD_up[:, o1] = Ū_LFD_up[:, o1] .* cHat[o, d]
			end
		end

		if θConstant == 1
			@. U_LFD_up[:] = U_LFD_up[:] .^ (-μHat) # if theta doesn't vary, then much faster to precalculate this matrix 
			# reduction in computation time due to non-integer exponent substantially dominates higher memory usage
		end

		Uσ_LFD_up = U_LFD_up .^ (1 - σHat) # precalculate 




		if sameMarginalsMoment == 1
			CDF_Moments_up = precalcCDFs(Ū_LFD_up, params, prestep_output)
			CDF_Moments_down = precalcCDFs(Ū_LFD_down, params, prestep_output)
		end

		if independenceMoment == 1
			Ind_Moments_up, IndCDF_Cells_up = precalcIndependence(Ū_LFD_up, params)
			Ind_Moments_down, IndCDF_Cells_down = precalcIndependence(Ū_LFD_down, params)
		end


		γ_upper= (wHat = γ.wHat,
		L = γ.L,
		LPrime = γ.LPrime,
		τ = γ.τ,
		τPrime = γ.τPrime,
		P = γ.P,
		PMM = γ.PMM,
		baseIndex = γ.baseIndex,
		indicators = γ.indicators,
		wPrimeHat = γ.wPrimeHat,
		Uσ = Uσ_exact,
		Ū = Ū_exact,
		μHat = γ.μHat,
		D = γ.D,
		CDF_Moments = CDF_Moments_up,
		Ind_Moments = Ind_Moments_up,
		cHat = γ.cHat,
		IndCDF_Cells = IndCDF_Cells_up,
		SamplingWeights = γ.SamplingWeights,
		refIndex1 = γ.refIndex1)

		obj_upper = PsiObjectiveBundleDelta(
			#δ = 1,
			#find_smallest = true,
			γ = γ_upper,
			(moments!) = EK_moments!,
			#moments_jacobian! = rust_moments_jacobian!,
			d = numMoments,
			outer_constr_index = outer_constr_index,
			inequality_index = Int64[],
			l = size(θ_initial, 1),
			U = U_LFD_up,
			N = 100,
			outer_loop_opt = "ek_outer_loop_options.opt",
			inner_loop_opt = "ek_inner_loop_options.opt",
			lower_limit = -50)

		γ_lower= (wHat = γ.wHat,
			L = γ.L,
			LPrime = γ.LPrime,
			τ = γ.τ,
			τPrime = γ.τPrime,
			P = γ.P,
			PMM = γ.PMM,
			baseIndex = γ.baseIndex,
			indicators = γ.indicators,
			wPrimeHat = γ.wPrimeHat,
			Uσ = Uσ_exact,
			Ū = Ū_exact,
			μHat = γ.μHat,
			D = γ.D,
			CDF_Moments = CDF_Moments_down,
			Ind_Moments = Ind_Moments_down,
			cHat = γ.cHat,
			IndCDF_Cells = IndCDF_Cells_down,
			SamplingWeights = γ.SamplingWeights,
			refIndex1 = γ.refIndex1)

		obj_lower = PsiObjectiveBundleDelta(
			#δ = 1,
			#find_smallest = true,
			γ = γ_lower,
			(moments!) = EK_moments!,
			#moments_jacobian! = rust_moments_jacobian!,
			d = numMoments,
			outer_constr_index = outer_constr_index,
			inequality_index = Int64[],
			l = size(θ_initial, 1),
			U = U_LFD_down,
			N = 100,
			outer_loop_opt = "ek_outer_loop_options.opt",
			inner_loop_opt = "ek_inner_loop_options.opt",
			lower_limit = -50)


		# calculate moments using the Frechet U and weight by the LFD
		EK_moments!(K_upper, G_upper, Θ_upper[:, δ], U, obj)
		@. G_upper[:, :] = G_upper[:, :] .* LFD_upper[:, δ]
		@. K_upper[:] = K_upper[:] .* LFD_upper[:, δ]

		EK_moments!(K_lower, G_lower, Θ_lower[:, δ], U, obj)
		@. G_lower[:, :] = G_lower[:, :] .* LFD_lower[:, δ]
		@. K_lower[:] = K_lower[:] .* LFD_lower[:, δ]

		#calculate moments using the new U = U_LFD and no weights
		EK_moments!(K_upper_LFD, G_upper_LFD, Θ_upper[:, δ], U_LFD_up, obj_upper)
		EK_moments!(K_lower_LFD, G_lower_LFD, Θ_lower[:, δ], U_LFD_down, obj_lower)

		# check all moments are zero! 

		MomentsMean_upper = zeros(numMoments, 2)
		MomentsMean_lower = zeros(numMoments, 2)

		MomentsVar_upper = zeros(numMoments, 2)
		MomentsVar_lower = zeros(numMoments, 2)

		MomentsMean_upper[:, 1] = mean(G_upper, dims = 1)
		MomentsVar_upper[:, 1] = var(G_upper, dims = 1)
		MomentsMean_lower[:, 1] = mean(G_lower, dims = 1)
		MomentsVar_lower[:, 1] = var(G_lower, dims = 1)

		MomentsMean_upper[:, 2] = mean(G_upper_LFD, dims = 1)
		MomentsVar_upper[:, 2] = var(G_upper_LFD, dims = 1)
		MomentsMean_lower[:, 2] = mean(G_lower_LFD, dims = 1)
		MomentsVar_lower[:, 2] = var(G_lower_LFD, dims = 1)

		MomentsTest_upper = zeros(numMoments, 6)
		MomentsTest_lower = zeros(numMoments, 6)

		for i ∈ 1:numMoments
			for j ∈ 1:2
				MomentsTest_upper[i, 1+(j-1)*3] = MomentsMean_upper[i, j]
				MomentsTest_upper[i, 2+(j-1)*3] = sqrt(MomentsVar_upper[i, j] / W)
				MomentsTest_upper[i, 3+(j-1)*3] = abs(MomentsMean_upper[i, j]) <= 2 * sqrt(MomentsVar_upper[i, j] / W) ? 1 : 0

				MomentsTest_lower[i, 1+(j-1)*3] = MomentsMean_lower[i, j]
				MomentsTest_lower[i, 2+(j-1)*3] = sqrt(MomentsVar_lower[i, j] / W)
				MomentsTest_lower[i, 3+(j-1)*3] = abs(MomentsMean_lower[i, j]) <= 2 * sqrt(MomentsVar_lower[i, j] / W) ? 1 : 0
			end
		end

		writedlm(string("MomentsTest_upper_", δ_grid[δ], "_", file_name), MomentsTest_upper, ',')
		writedlm(string("MomentsTest_lower_", δ_grid[δ], "_", file_name), MomentsTest_lower, ',')

		# check the counterfactual bounds did not change much
		κ_upper[δ] = mean(K_upper[:,1])
		κ_upper_LFD[δ] = mean(K_upper_LFD[:,1])

		κ_lower[δ] = mean(K_lower[:,1])
		κ_lower_LFD[δ] = mean(K_lower_LFD[:,1])

	end
	writedlm(string("LFDCounterFactual_", file_name), [δ_grid κ_lower κ_lower_LFD κ_upper κ_upper_LFD], ',')
end
