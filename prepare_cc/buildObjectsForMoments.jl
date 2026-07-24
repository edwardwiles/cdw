function buildObjectsForMoments(
	globParams,
	prestep_output,
	data,
	LPrime,
	τPrime,
	Uσ,
	Ū,
	numMomentsSimple,
	upper_moment_start_index = 1,
	PMM = zeros(1),
	σ_Moments = ones(1),
	Moments_CS = zeros(1),
	CDF_Moments = zeros(1),
	Ind_Moments = zeros(1),
	IndCDF_Cells = Vector{Vector{Int}}(undef, 1),
	SamplingWeights = ones(1),
	moments_without_var = int64[])
	# constructs object containing fixed parameters (L, tau, data, etc) to feed into moment functions
	@unpack σHat,
	baseIndex,
	counterType,
	counterExplicit,
	θConstant,
	gravMoment,
	localGravityMoment,
	GravityMomentFirstApproach,
	sameMarginalsMoment,
	independenceMoment,
	momentOrder,
	momentOrderForBaseIndex,
	IndMomentOrder,
	refIndex1,
	OuterScaling,
	usePMM,
	UoModel,
	NormalizeMoments,
	useConfidenceIntervals =
		globParams
	@unpack μHat, wHat, λPrime, wPrimeHat, γHat, γPrimeHat, cHat, Aod_initial, wPrimeHat_upper, γHat_upper, γPrimeHat_upper, Aod_initial_upper, wPrimeHat_lower, γHat_lower, γPrimeHat_lower, Aod_initial_lower = prestep_output
	@unpack λData, LData, τData = data

	D = length(LData)

	# row_idx: destination excluded from the moment sample (Part A, 2026-07-23
	# omit-ROW-destination release). nothing (default) reproduces today's full-D^2 behavior
	# exactly -- named_dest==1:D. τData/λData are sliced to named_dest columns HERE (the single
	# place raw D x D data becomes D x Ddest for everything downstream: obj.γ.τ, the flattened
	# trade-share vector P, and (via prestep_output.cHat, already sliced in master_prestep.jl)
	# the moments callback's Aod construction). Origin axis (rows) is never sliced.
	row_idx = get(globParams, :row_idx, nothing)
	named_dest = row_idx === nothing ? (1:D) : filter(!=(row_idx), 1:D)
	Ddest = length(named_dest)
	τData = τData[:, named_dest]
	λData = λData[:, named_dest]
	τPrime = τPrime[:, named_dest]

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
		UoModel = UoModel,
		NormalizeMoments = NormalizeMoments,
		useConfidenceIntervals = useConfidenceIntervals)


	θ_initial = get_θ_initial(globParams, μHat, wPrimeHat, γHat, γPrimeHat, Aod_initial)
	θ_initial_upper = get_θ_initial(globParams, μHat, wPrimeHat_upper, γHat_upper, γPrimeHat_upper, Aod_initial_upper)
	θ_initial_lower = get_θ_initial(globParams, μHat, wPrimeHat_lower, γHat_lower, γPrimeHat_lower, Aod_initial_lower)

@show θ_initial
@show θ_initial_upper
@show θ_initial_lower

	γ = (wHat = wHat,
		L = LData,
		LPrime = LPrime,
		τ = τData,
		τPrime = τPrime,
		P = reshape(λData', (1, D * Ddest)),
		PMM = PMM,
		σ_Moments = σ_Moments,
		Moments_CS = Moments_CS,
		baseIndex = baseIndex,
		indicators = indicators,
		wPrimeHat = wPrimeHat,
		Uσ = Uσ,
		# preallocated Float64 scratch for U^{-μ} / Uσ^{-μ} (reused on the Float64 moment path;
		# the ForwardDiff/Dual path still allocates per call since a Float64 buffer can't hold Duals)
		UPow_scratch = zeros(size(Uσ)),
		UσPow_scratch = zeros(size(Uσ)),
		Ū = Ū,
		μHat = μHat,
		D = D,
		CDF_Moments = CDF_Moments,
		Ind_Moments = Ind_Moments,
		cHat = cHat,
		IndCDF_Cells = IndCDF_Cells,
		SamplingWeights = SamplingWeights,
		refIndex1 = refIndex1,
		upper_moment_start_index = upper_moment_start_index,
		numMomentsSimple = numMomentsSimple,
		moments_without_var = moments_without_var,
	)

	return γ, θ_initial, θ_initial_lower, θ_initial_upper
end

function get_θ_initial(globParams, μHat, wPrimeHat, γHat, γPrimeHat, Aod_initial)
	@unpack σHat,
	baseIndex,
	counterType,
	counterExplicit,
	θConstant,
	gravMoment,
	localGravityMoment,
	GravityMomentFirstApproach,
	sameMarginalsMoment,
	independenceMoment,
	momentOrder,
	momentOrderForBaseIndex,
	IndMomentOrder,
	refIndex1,
	OuterScaling,
	usePMM,
	UoModel,
	NormalizeMoments,
	useConfidenceIntervals = globParams


	# remove the entry of w' that is the wage we are normalising to 1
	# as no point in optimising over this (will add it back inside the moment function)
	splice!(wPrimeHat, baseIndex)
	euler_gamma = -0.577216
	if counterType != 1

		θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, wPrimeHat)


		if OuterScaling == 1 # Aod Model
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, Aod_initial)

			if independenceMoment == 1 # this also means sameMarginalsMoment == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, 1, Aod_initial, range(1 / (IndMomentOrder + 1), IndMomentOrder / (IndMomentOrder + 1), length = IndMomentOrder))
			elseif GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0 && UoModel == 0 # We will need to calculate E[ln ̄U]
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, Aod_initial, zeros(D^2))
			end
		elseif sameMarginalsMoment == 1 # Fix Aod and do not allow Ubar to match
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat)
			if independenceMoment == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, 1, range(1 / (IndMomentOrder + 1), IndMomentOrder / (IndMomentOrder + 1), length = IndMomentOrder))
			end
		else # sameMarginalsMoment == 0 && OuterScaling == 0
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat)
			if GravityMomentFirstApproach == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, zeros(D^2))
			end
		end

		return θ_initial

	else
		θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])

		if OuterScaling == 1 # Aod Model
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], Aod_initial)

			if independenceMoment == 1 # this also means sameMarginalsMoment == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], 1, Aod_initial, range(1 / (IndMomentOrder + 1), IndMomentOrder / (IndMomentOrder + 1), length = IndMomentOrder))
			elseif GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0 && UoModel == 0# We will need to calculate E[ln ̄U]
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], Aod_initial, ones(D^2) .* euler_gamma)
			end
		elseif sameMarginalsMoment == 1 # Fix Aod and do not allow Ubar to match
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])
			if independenceMoment == 1
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], 1, range(1 / (IndMomentOrder + 1), IndMomentOrder / (IndMomentOrder + 1), length = IndMomentOrder))
			end
		else # sameMarginalsMoment == 0 && OuterScaling == 0
			θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])
			if GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0 && UoModel == 0
				θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], ones(D^2) .* euler_gamma)
			end

		end
		return θ_initial
	end
end
