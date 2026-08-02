
"""
    master_prepare_cc(data, counters, prestep_output, globalParams; U=nothing)

`U=nothing` (default): draw U internally exactly as before (`Random.seed!(seedU);
U = drawU(SamplingWeight, globalParams)`) -- unchanged, bit-identical pseudorandom path.
`U` given (a `W x D` `AbstractMatrix{Float64}`, already Exp(1)-transformed): use it directly
instead of drawing -- no seeding, no `drawU` call. This is the single unified entry point for
every draw design (task "unify random-draw production pipeline" 2026-07-30 §6); the pseudorandom
and randomized-Sobol/scrambled-Halton/precomputed paths differ ONLY in whether `U` is supplied,
nowhere else in this function.
"""
function master_prepare_cc(data, counters, prestep_output, globalParams; U::Union{Nothing,AbstractMatrix{Float64}} = nothing)

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
	PMMGammaOnly = globalParams

	# row_idx: destination excluded from the moment/estimation sample (Part A, 2026-07-23
	# omit-ROW-destination release). nothing (default) reproduces today's D^2-moment layout
	# exactly. Only the base autarky+gravity path (counterType==1) is supported with row_idx
	# set -- every CM/marginals/independence extension below hard-errors rather than silently
	# building a wrong-sized moment vector (those layers are out of scope for this release, see
	# docs/DROP_ROW_DESTINATION_EXPERIMENT_REPORT_2026-07-23.md and the Part A plan).
	row_idx = get(globalParams, :row_idx, nothing)
	Ddest = row_idx === nothing ? D : D - 1
	if row_idx !== nothing
		(sameMarginalsMoment == 1 || independenceMoment == 1 || GravityMomentFirstApproach == 1 ||
			localGravityMoment == 1 || PMMGammaOnly == 1 || useConfidenceIntervals == 1) &&
			error("row_idx (omit-ROW-destination) is not supported together with sameMarginalsMoment/independenceMoment/GravityMomentFirstApproach/localGravityMoment/PMMGammaOnly/useConfidenceIntervals -- out of scope for this release.")
		counterType == 1 || error("row_idx (omit-ROW-destination) is only implemented for counterType==1 (autarky); counterType=$(counterType) is out of scope for this release.")
	end

	SamplingWeight = ones(W)

	if U === nothing
		# set seed
		Random.seed!(seedU)
		# draw base U matrix from exp(1), using stratified or importance sampling as specified in parameters
		U = drawU(SamplingWeight, globalParams)
	else
		@assert size(U, 1) == W "U has W=$(size(U,1)) rows, expected $(W)"
	end

	Ū, Uσ = createUDerivatives!(U, prestep_output, globalParams)

	useParams = globalParams
	useParams = (; useParams..., SamplingWeight = SamplingWeight)

	# closed-form Frechet prestep (the CDW method); alternative starting points removed
	prestep_output_effective = prestep_output

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
	else
		# autarky: drop the D baseline price-index moments (each is the exact sum of that
		# destination's D trade-share moments, since shares sum to 1) and the D-1 unused
		# counterfactual placeholders (only baseIndex has a counterfactual under autarky).
		# Keep D*Ddest trade shares + 1 counterfactual price-index moment (Ddest==D, i.e. D^2,
		# unless row_idx excludes a destination -- Part A, 2026-07-23).
		numMoments = D * Ddest + 1
	end

	# we add the condition that E[Ubar] =1
	#numMoments += UoModel == 0 ? D : D^2

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
		numMoments += 1 # CDF[i]<CDF[i+1]

		# corr[i,j]=0
		if UoModel == 0
			numMoments += D * (D^2 - floor(Int, D * (1 + D) / 2))
		else
			numMoments += (D^2 - floor(Int, D * (1 + D) / 2))
		end
	end


	# gravMoment (double-diff ln A ⟂ double-diff ln τ) is F-independent for UoModel=1, so it is an
	# OUTER constraint on θ (the A's) rather than an inner-loop moment matched over F.
	nOuterLoopMoments = ((GravityMomentFirstApproach == 1) ? 1 : 0 ) + ( (independenceMoment == 1) ? 1 : 0 ) + ( (gravMoment == 1) ? 1 : 0 )
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
		"NoJacob_FD_",
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
		"DR_",
		δ_ref,
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

	γ_Hat, δ_star_initial, δ_star_initial_low, δ_star_initial_up = γHat(
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
		PMMGammaOnly)
	moments_without_var = γ_Hat.moments_without_var
	#complement_index =  filter!(e-> ( e ∉ moments_without_var  && (e-inner_loop_last_moment_index) ∉ moments_without_var ), complement_index) 

	if δGridType == 0
		δ_grid = vcat(δ_ref)
	else
		δ_grid = vcat(0.01, 0.1, 0.5, 1, 2) .* δ_ref
		#δ_grid = vcat(1, 2) .* δ_ref

	end

	δ_grid_filtered = filter(x -> x >= δ_star_initial  && x >= δ_star_initial_low  && x >= δ_star_initial_up  , δ_grid)

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
