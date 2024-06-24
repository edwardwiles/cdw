function γPMM(θ_initial, γ, U, numMoments, outer_constr_index)

	W = size(U, 1)
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
		#N = 100,
		#lower_limit = -50
		outer_loop_opt = "ek_outer_loop_options.opt",
		inner_loop_opt = "ek_inner_loop_options.opt"
		)

	G = zeros(W, numMoments)
	K = zeros(W, 1)

	EK_moments!(K, G, θ_initial, U, obj)

	γ_PMM = (wHat = γ.wHat,
		L = γ.L,
		LPrime = γ.LPrime,
		τ = γ.τ,
		τPrime = γ.τPrime,
		P = γ.P,
		PMM = mean(G, dims = 1),
		σ_Moments = sqrt.(var(G, dims = 1) ./ W),
		baseIndex = γ.baseIndex,
		indicators = γ.indicators,
		wPrimeHat = γ.wPrimeHat,
		Uσ = γ.Uσ,
		Ū = γ.Ū,
		μHat = γ.μHat,
		D = γ.D,
		CDF_Moments = γ.CDF_Moments,
		Ind_Moments = γ.Ind_Moments,
		cHat = γ.cHat,
		IndCDF_Cells = γ.IndCDF_Cells,
		SamplingWeights = γ.SamplingWeights,
		refIndex1 = γ.refIndex1)


	return γ_PMM

end
