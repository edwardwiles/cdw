function buildObjectsForMoments(
	globParams,
	prestep_output,
	data,
	LPrime,
	τPrime,
	Uσ,
	Ū,
	PMM = zeros(1),
	σ_Moments = ones(1),
	CDF_Moments = zeros(1),
	Ind_Moments = zeros(1),
	IndCDF_Cells = Vector{Vector{Int}}(undef, 1),
	SamplingWeights = ones(1))
	# constructs object containing fixed parameters (L, tau, data, etc) to feed into moment functions
	@unpack σHat, baseIndex, counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, GravityMomentFirstApproach, sameMarginalsMoment, independenceMoment, momentOrder, momentOrderForBaseIndex, IndMomentOrder, refIndex1, OuterScaling, usePMM, UoModel =
		globParams
	@unpack μHat, wHat, λPrime, wPrimeHat, γHat, γPrimeHat, cHat = prestep_output
	@unpack λData, LData, τData = data

	D = length(LData)

	indicators = (counterExplicit = counterExplicit,
		counterType = counterType,
		θConstant = θConstant,
		gravMoment = gravMoment,
		localGravityMoment = localGravityMoment,
		GravityMomentFirstApproach = GravityMomentFirstApproach,
		sameMarginalsMoment = sameMarginalsMoment,
		independenceMoment = independenceMoment,
		momentOrder = momentOrder,
		momentOrderForBaseIndex = momentOrderForBaseIndex,
		IndMomentOrder = IndMomentOrder,
		OuterScaling = OuterScaling,
		usePMM = usePMM,
		UoModel= UoModel)

	# remove the entry of w' that is the wage we are normalising to 1
	# as no point in optimising over this (will add it back inside the moment function)
	splice!(wPrimeHat, baseIndex)


	if counterType != 1

		θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, wPrimeHat)


		if OuterScaling == 1 # Aod Model
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, ones(D^2))

			if independenceMoment == 1 # this also means sameMarginalsMoment == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, 1, ones(D^2), range(1 / IndMomentOrder, (IndMomentOrder - 1) / IndMomentOrder, length = IndMomentOrder))
			elseif GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0 # We will need to calculate E[ln ̄U]
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, ones(D^2), zeros(D^2))
			end
		elseif sameMarginalsMoment == 1 # Fix Aod and do not allow Ubar to match
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat)
			if independenceMoment == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, 1, range(1 / IndMomentOrder, (IndMomentOrder - 1) / IndMomentOrder, length = IndMomentOrder))
			end
		else # sameMarginalsMoment == 0 && OuterScaling == 0
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat)
			if GravityMomentFirstApproach == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, zeros(D^2))
			end
		end

		γ = (wHat = wHat,
			L = LData,
			LPrime = LPrime,
			τ = τData,
			τPrime = τPrime,
			P = reshape(λData', (1, D^2)),
			PMM = PMM,
			baseIndex = baseIndex,
			indicators = indicators,
			wPrimeHat = wPrimeHat,
			Uσ = Uσ,
			Ū = Ū,
			μHat = μHat,
			D = D,
			CDF_Moments = CDF_Moments,
			Ind_Moments = Ind_Moments,
			cHat = cHat,
			IndCDF_Cells = IndCDF_Cells,
			SamplingWeights = SamplingWeights,
			refIndex1 = refIndex1,
		)
		return γ, θ_initial

	else
		θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])

		if OuterScaling == 1 # Aod Model
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], ones(D^2))

			if independenceMoment == 1 # this also means sameMarginalsMoment == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], 1, ones(D^2), range(1 / IndMomentOrder, (IndMomentOrder - 1) / IndMomentOrder, length = IndMomentOrder))
			elseif GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0 # We will need to calculate E[ln ̄U]
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], ones(D^2), zeros(D^2))
			end
		elseif sameMarginalsMoment == 1 # Fix Aod and do not allow Ubar to match
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])
			if independenceMoment == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], 1, range(1 / IndMomentOrder, (IndMomentOrder - 1) / IndMomentOrder, length = IndMomentOrder))
			end
		else # sameMarginalsMoment == 0 && OuterScaling == 0
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])
			if GravityMomentFirstApproach == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], zeros(D^2))
			end

		end


		γ = (wHat = wHat,
			L = LData,
			LPrime = LPrime,
			τ = τData,
			τPrime = τPrime,
			P = reshape(λData', (1, D^2)),
			PMM = PMM,
			σ_Moments = σ_Moments,
			baseIndex = baseIndex,
			indicators = indicators,
			wPrimeHat = wPrimeHat,
			Uσ = Uσ,
			Ū = Ū,
			μHat = μHat,
			D = D,
			CDF_Moments = CDF_Moments,
			Ind_Moments = Ind_Moments,
			cHat = cHat,
			IndCDF_Cells = IndCDF_Cells,
			SamplingWeights = SamplingWeights,
			refIndex1 = refIndex1,
		)

		return γ, θ_initial
	end

end
