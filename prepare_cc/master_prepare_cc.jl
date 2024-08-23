
function master_prepare_cc(data, counters, prestep_output, globalParams)

	@unpack seedU,
	W,
	D,
	counterType,
	importanceSampling,
	importanceSamplingFactor,
	sameMarginalsMoment,
	independenceMoment,
	GravityMomentFirstApproach,
	gravMoment,
	localGravityMoment,
	momentOrder,
	momentOrderForBaseIndex,
	baseIndex,
	ForceFrechetMarginal,
	IndMomentOrder,
	δGridType,
	OuterScaling,
	fakeData,
	θConstant,
	usePMM,
	UoModel,
	calc_δ_star_initial,
	Jac_W,
	δ_ref,
	NormalizeMoments,
	useConfidenceIntervals,
	ConfidenceLevel,
	useFiniteSamplePrestep,
	PMMGammaOnly,
	useFrechetCopulaStartingPoint = globalParams

	# set seed
	Random.seed!(seedU)

	SamplingWeight = ones(W)
	# draw base U matrix from exp(1), using stratified or importance sampling as specified in parameters 
	U = drawU(SamplingWeight, globalParams)

	Ū, Uσ = createUDerivatives!(U, prestep_output, globalParams)

	useParams = globalParams
	useParams = (; useParams..., SamplingWeight = SamplingWeight)

	prestep_output_effective =
		useFiniteSamplePrestep == 0 && useFrechetCopulaStartingPoint == 0 ? prestep_output : (useFrechetCopulaStartingPoint != 0 ? preStepGeneralDistribution(data, counters, globalParams) : preStepGeneralDistribution(data, counters, globalParams, Ū))

	# if using common marginals with moments methodology, calculate the K moments and put them in Ubar 

	CDF_Moments = zeros(1, 1)

	if sameMarginalsMoment == 1
		CDF_Moments = precalcCDFs(Ū, useParams, prestep_output_effective)
	end

	Ind_Moments = zeros(1, 1)
	IndCDF_Cells = Vector{Vector{Int}}(undef, 1)

	if independenceMoment == 1
		Ind_Moments, IndCDF_Cells = precalcIndependence(Ū, useParams)
	end

	# trade share moments + gamma + gammaPrime 
	numMoments = D^2 + 2 * D

	if counterType != 1
		numMoments += (D - 1)
	end

	if GravityMomentFirstApproach == 1
		numMoments += 1

		if UoModel == 0 && sameMarginalsMoment == 0
			numMoments += D^2
		end
	end

	if gravMoment == 1
		numMoments += 1
	end

	if localGravityMoment == 1
		numMoments += (D - 1) * D + D * (D - 1) * (D - 2)
	end

	if sameMarginalsMoment == 1
		if UoModel == 1
			numMoments += 2 * D + 2 * momentOrderForBaseIndex * D
		else
			numMoments += 2 * momentOrder * D^2 + 2 * D^2 + 2 * momentOrderForBaseIndex * D
		end
	end

	if independenceMoment == 1
		numMoments += 1 # E[U[refIndex1]]
		numMoments += IndMomentOrder # CDF
		numMoments += (D^2 - floor(Int, D * (1 + D) / 2)) * (IndMomentOrder^2) # CDF[i,j]=CDF[i]*CDF[j]

		# corr[i,j]=0
		if UoModel == 0
			numMoments += D * (D^2 - floor(Int, D * (1 + D) / 2))
		else
			numMoments += (D^2 - floor(Int, D * (1 + D) / 2))
		end
	end


	nOuterLoopMoments = (GravityMomentFirstApproach == 1) ? 1 : 0
	outer_constr_index = numMoments + 1 - nOuterLoopMoments
	numMomentInnerSimple = numMoments - nOuterLoopMoments
	outer_constr_index_simple = outer_constr_index
	inner_loop_last_moment_index = numMoments - nOuterLoopMoments
	nTotalMoments = numMoments
	inequality_index = Int64[]
	lower_inequality_index = Int64[]
	upper_moment_start_index = 1
	complement_index = [0 0]
	moments_without_var = Int64[]


	if useConfidenceIntervals == 1
		inequality_index = collect(1:2*inner_loop_last_moment_index)
		outer_constr_index = 2 * inner_loop_last_moment_index + 1
		upper_moment_start_index = inner_loop_last_moment_index + 1
		inner_loop_last_moment_index = outer_constr_index - 1
		nTotalMoments = 2 * (numMoments - nOuterLoopMoments) + nOuterLoopMoments
		complement_index = hcat(collect(1:inner_loop_last_moment_index) , collect(inner_loop_last_moment_index+1:2*inner_loop_last_moment_index))
	end

	file_name = string(
		"FD_",
		fakeData,
		"_Count_",
		counterType,
		"_NC_",
		D,
		"_bI",
		baseIndex,
		"_sG",
		GravityMomentFirstApproach,
		"_lG",
		localGravityMoment,
		"_Marg",
		sameMarginalsMoment,
		"_ind",
		independenceMoment,
		"_O",
		momentOrder,
		"_bO",
		momentOrderForBaseIndex,
		"FF_",
		ForceFrechetMarginal,
		"IndMO_",
		IndMomentOrder,
		"IS_",
		importanceSampling,
		"ISF_",
		importanceSamplingFactor,
		"Aod_",
		OuterScaling,
		"Fmu_",
		θConstant,
		"Uo_",
		UoModel,
		"MN",
		NormalizeMoments,
		"P",
		usePMM,
		"CS",
		useConfidenceIntervals,
		"Cor",
		useFrechetCopulaStartingPoint,
		"_",
		Dates.format(now(), "y-m-d"),
		".csv",
	)


	PMM = zeros(numMoments)
	σ_Moments = ones(numMoments)
	Moments_CS = zeros(numMoments, 2)

	γ, θ_initial, θ_initial_low, θ_initial_up = buildObjectsForMoments(
		globalParams,
		prestep_output_effective,
		data,
		counters.LPrime,
		counters.τPrime,
		Uσ,
		Ū,
		numMoments,
		upper_moment_start_index,
		PMM,
		σ_Moments,
		Moments_CS,
		CDF_Moments,
		Ind_Moments,
		IndCDF_Cells,
		SamplingWeight,
		moments_without_var,
	)

	@show numMoments
	@show numMomentInnerSimple
	@show inequality_index

	γ_Hat, δ_star_initial = γHat(
		θ_initial,
		θ_initial_low,
		θ_initial_up,
		γ,
		U,
		numMoments,
		numMomentInnerSimple,
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
		useFrechetCopulaStartingPoint,
	)
	moments_without_var = γ_Hat.moments_without_var
	#complement_index =  filter!(e-> ( e ∉ moments_without_var  && (e-inner_loop_last_moment_index) ∉ moments_without_var ), complement_index) 

	if δGridType == 0
		δ_grid = vcat(δ_ref)
	else
		δ_grid = vcat(0.01, 0.1, 0.5, 1, 2) .* δ_ref
	end

	δ_grid_filtered = filter(x -> x >= δ_star_initial, δ_grid)

	@show δ_star_initial
	@show δ_grid
	@show δ_grid_filtered

	prep_output = (
		U = U,
		γ = (PMMGammaOnly == 1 || usePMM == 1 || NormalizeMoments == 1 || useConfidenceIntervals == 1) ? γ_Hat : γ,
		θ_initial = θ_initial,
		θ_initial_low = θ_initial_low,
		θ_initial_up = θ_initial_up,
		numMoments = numMoments,
		file_name = file_name,
		δ_grid = δ_grid_filtered,
		inequality_index = inequality_index,
		outer_constr_index = outer_constr_index,
		nTotalMoments = nTotalMoments,
		complement_index = complement_index,
	)

	return prep_output

end
