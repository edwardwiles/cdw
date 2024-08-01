function γHat(
	θ_initial,
	γ,
	U,
	numMoments,
	numInnerMoments,
	outer_constr_index,
	outer_constr_index_simple,
	nTotalMoments,
	inner_loop_last_moment_index,
	inequality_index,
	calc_δ_star_initial,
	file_name,
	useConfidenceIntervals,
	ConfidenceLevel,
	NormalizeMoments,
	complement_index,
	PMMGammaOnly,
)

	W = size(U, 1)
	# use the simplest object to calc moments at F* and theta initial
	obj = PsiObjectiveBundleDelta(
		γ = γ,
		(moments!) = EK_moments_simple!,
		d = numMoments,
		outer_constr_index = outer_constr_index_simple,
		inequality_index = Int64[],
		l = size(θ_initial, 1),
		U = U,
		N = 100,# not used because we do not calculate jacobians
		lower_limit = -50,
		outer_loop_opt = "ek_outer_loop_options.opt",
		inner_loop_opt = "ek_inner_loop_options.opt",
	)

	G = zeros(W, numMoments)
	K = zeros(W, 1)

	EK_moments_simple!(K, G, θ_initial, U, obj)

	PMM = mean(G, dims = 1)
	σ_Moments = sqrt.(var(G, dims = 1))

	moments_with_var = findall(>(0), σ_Moments[1:numInnerMoments])
	moments_without_var = findall(==(0), σ_Moments[1:numInnerMoments])

	@show PMM
	@show moments_with_var
	@show moments_without_var

	if PMMGammaOnly == 1
		D = size(γ.τ, 1)

		cInd = D^2
		dInd = D^2 + D
		if γ.indicators.counterType != 1
			cInd = D^2 + D - 1
			dInd = D^2 + 2 * D - 1
		end

		for im in 1:numMoments
			is_gamma_moment = (im ∉ cInd+1:cInd+D && im ∉ dInd+1:dInd+D) ? 0 : 1 #gamma is actually a gamma hat.
			PMM[im] *= is_gamma_moment
		end
	end



	# calculate the confidence set assuming normality of moment estimator

	Moments_CS = zeros(numInnerMoments, 2)
	MomentsCovar = zeros(numInnerMoments, numInnerMoments)
	if useConfidenceIntervals == 1

		MomentsCovar = cov(@view(G[:, moments_with_var]), dims = 1)


		moments_with_var_size = size(MomentsCovar, 1)
		if NormalizeMoments == 1
			for im1 in 1:moments_with_var_size
				for im2 in 1:moments_with_var_size
					MomentsCovar[im1, im2] = MomentsCovar[im1, im2] / (σ_Moments[moments_with_var[im1]] * σ_Moments[moments_with_var[im2]])
				end
			end
		end

		c = 2
		s = NormalizeMoments == 1 ? ones(moments_with_var_size) : σ_Moments[moments_with_var]
		if false
			(c, s) = rectangular_confidence_set(MomentsCovar, ConfidenceLevel)
		end
		Moments_CS[moments_with_var, 1] .= -c * s ./ sqrt(W) #lower bound
		Moments_CS[moments_with_var, 2] .= +c * s ./ sqrt(W) #upper bound
		PMMCS = zeros(numMoments)
		@. PMMCS[moments_with_var] = PMM[moments_with_var] .* s[:] ./ σ_Moments[moments_with_var]
		save_object(string("momentsCS_", file_name, ".jld2"), Moments_CS)
		writedlm(string("momentsCS_", file_name), [Moments_CS[:, 1] Moments_CS[:, 2] PMMCS[:]], ',')
	end



	γ_PMM = (wHat = γ.wHat,
		L = γ.L,
		LPrime = γ.LPrime,
		τ = γ.τ,
		τPrime = γ.τPrime,
		P = γ.P,
		numMomentsSimple = γ.numMomentsSimple,
		PMM = PMM,
		σ_Moments = σ_Moments,
		Moments_CS = Moments_CS,
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
		refIndex1 = γ.refIndex1,
		upper_moment_start_index = γ.upper_moment_start_index,
		moments_without_var = moments_without_var)

	δ_star_initial = 0
	if calc_δ_star_initial == 1
		obj2 =
			useConfidenceIntervals == 1 ? obj :
			PsiObjectiveBundleDelta(
				γ = γ_PMM,
				(moments!) = EK_moments!,
				d = nTotalMoments,
				outer_constr_index = outer_constr_index,
				inequality_index = inequality_index,
				complement_index = complement_index,
				#complement_index = filter!(e->e ∉ moments_without_var && e-inner_loop_last_moment_index ∉ moments_without_var, complement_index) ,
				l = size(θ_initial, 1),
				U = U,
				N = 100,# not used because we do not calculate jacobians
				lower_limit = -50,
				outer_loop_opt = "ek_outer_loop_options.opt",
				inner_loop_opt = "ek_inner_loop_options.opt",
			)
		val, x, nStatus = inner_loop(obj, θ_initial)

		if nStatus ∈ [0, -100, -101, -103]
			δ_star_initial = -val
		else
			δ_star_initial = 1e+10
		end
	end

	@show δ_star_initial

	return γ_PMM, δ_star_initial

end
