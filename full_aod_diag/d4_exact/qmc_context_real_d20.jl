# ============================================================================
# Continuation 10, Phase 7: isolated QMC-draw context builder for the real
# D=20 economy. ADDITIVE ONLY -- does not modify context_real_d20.jl,
# prepare_cc/master_prepare_cc.jl, or prepare_cc/drawU.jl. Every production
# entry point (d20_real_setup, build_ad_context_real_d20, master_prepare_cc)
# is untouched; this file defines parallel *_qmc functions that are byte-
# for-byte identical to the production chain except at the single point
# where the W x D "F*" draw matrix U is produced.
#
# Background: master_prepare_cc.jl draws U once (`Random.seed!(seedU);
# U = drawU(SamplingWeight, globalParams)`) and everything downstream
# (createUDerivatives!, buildObjectsForMoments, gammaHat, and hence ctx.U /
# ctx.gamma.Uσ) is a deterministic function of that single U -- confirmed by
# reading master_prepare_cc.jl in full: no other call to Random./rand!
# anywhere in that function or its callees. This means (a) common random
# numbers hold automatically within one optimization (U is drawn once at
# setup and never redrawn inside the KNITRO callback loop -- verified by
# grepping prepare_cc/*.jl and the ctx-building chain), and (b) swapping the
# draw source is a pure "replace U before it enters master_prepare_cc"
# operation with no other code path to worry about.
#
# drawU.jl (prepare_cc/drawU.jl) confirms UoModel=1 (this investigation's
# fixed AD_PARAMS default, unchanged by context_real_d20.jl) uses
# `sizeU = D` (NOT D^2) -- i.e. the F* draw matrix is W x D = 80000 x 20 at
# the real D=20 target, origin-indexed only (U[omega,o], shared across
# destinations d) -- confirmed independently by winners.jl's factual_prices
# (`U = ctx.U`, indexed `U[ω, o]` with no d subscript). This matters for the
# QMC comparison: the Halton/Sobol dimensionality needed is only 20, not 400,
# well inside both generators' safe/well-conditioned regime.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> d20_real_setup, build_ad_context_real_d20, D4X_ROOT, AD_PARAMS
include(joinpath(D4X_ROOT, "prepare_cc", "master_prepare_cc.jl"))   # ensure drawU/genExpRands! etc are in scope (already true via context.jl's includes, kept here for explicitness)

"""
    exp_from_uniform01(U01) -> Matrix{Float64}

Literal copy of prepare_cc/genRands.jl::genExpRands!'s elementwise transform
(`U[i] = -log(1 - U[i])`), extracted so it can be applied to an
externally-supplied uniform-[0,1) matrix (Halton/Sobol) instead of only to
`rand!`'s pseudorandom output. This is NOT a new sampling method -- it is the
model's actual Exp(1) inverse-CDF, unchanged, just called on a different
uniform source. Values must be in [0,1); a value of exactly 1.0 would produce
Inf (guard below matches the fact that rand!()/Halton/Sobol as used here
never hit the closed endpoint in practice, but we clamp defensively).
"""
function exp_from_uniform01(U01::AbstractMatrix{Float64})
    U01c = clamp.(U01, 0.0, prevfloat(1.0))
    return -log.(1.0 .- U01c)
end

"""
    master_prepare_cc_qmc(data, counters, prestep_output, globalParams, U_injected)

Byte-for-byte copy of prepare_cc/master_prepare_cc.jl with exactly one change:
U is supplied by the caller (already Exp(1)-transformed, W x D)  instead of
being drawn internally via `Random.seed!(seedU); drawU(...)`. See the diff
kept in this task's commit message / docs for the exact patch. Defined here
(not by editing the production file) per this task's explicit instruction not
to touch production code.
"""

function master_prepare_cc_qmc(data, counters, prestep_output, globalParams, U_injected::AbstractMatrix{Float64})

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
	# building a wrong-sized moment vector (those layers are out of scope for this release).
	# Ported from prepare_cc/master_prepare_cc.jl verbatim -- this function had predated that
	# port and was missing it (found+fixed as part of threading destination_sample through the
	# QMC context path, 2026-07-26).
	row_idx = get(globalParams, :row_idx, nothing)
	Ddest = row_idx === nothing ? D : D - 1
	if row_idx !== nothing
		(sameMarginalsMoment == 1 || independenceMoment == 1 || GravityMomentFirstApproach == 1 ||
			localGravityMoment == 1 || PMMGammaOnly == 1 || useConfidenceIntervals == 1) &&
			error("row_idx (omit-ROW-destination) is not supported together with sameMarginalsMoment/independenceMoment/GravityMomentFirstApproach/localGravityMoment/PMMGammaOnly/useConfidenceIntervals -- out of scope for this release.")
		counterType == 1 || error("row_idx (omit-ROW-destination) is only implemented for counterType==1 (autarky); counterType=$(counterType) is out of scope for this release.")
	end

	# QMC FORK (diagnostic only, full_aod_diag/d4_exact/qmc_context_real_d20.jl): the ONE
	# and only change from prepare_cc/master_prepare_cc.jl -- U is injected by the caller
	# (pseudorandom / scrambled-Halton / scrambled-Sobol, all already Exp(1)-transformed via
	# the SAME -log(1-u) inverse-CDF prepare_cc/genRands.jl::genExpRands! uses) instead of
	# being drawn here via Random.seed!(seedU); drawU(...). Everything below this point is
	# an EXACT, unmodified copy of production master_prepare_cc.jl.
	@assert size(U_injected, 1) == W "U_injected has W=$(size(U_injected,1)) rows, expected $(W)"
	SamplingWeight = ones(W)
	U = U_injected

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

"Mirrors context_real_d20.jl::build_ad_context_real_d20 but injects U_injected (W x D, already Exp(1)) via master_prepare_cc_qmc instead of drawing it internally."
function build_ad_context_real_d20_qmc(; W::Int, U_injected::AbstractMatrix{Float64}, row_idx::Union{Nothing,Int} = nothing, exclude_diagonal_gravity::Bool = false)
    params = merge(AD_PARAMS, (fakeData = 3, DFake = D20_REAL, W = W, Jac_W = W, row_idx = row_idx,
        exclude_diagonal_gravity = exclude_diagonal_gravity))
    so = master_setup(params)
    @assert so.D == D20_REAL "master_setup returned D=$(so.D), expected $(D20_REAL)"
    up = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(up)
    ps = master_prestep(so.data, so.counters, up)
    pp = master_prepare_cc_qmc(so.data, so.counters, ps, up, U_injected)
    return so, pp, params
end

"""
    d20_real_setup_qmc(; W, U_injected, δ=1.0, find_smallest=true, ...) -> NamedTuple

Mirrors context_real_d20.jl::d20_real_setup exactly (same free-parameter
layout, bounds, PsiObjectiveBundleImplicit wiring, needs_outer_moment_jacobian
default, destination_sample/row_idx/D_dest rectangularization) but takes an
externally-supplied W x D Exp(1) draw matrix (`U_injected`) instead of drawing
U via the production seedU/drawU path. Every existing D=4/D=20 diagnostic
function (evaluate_fullA, compute_winners, build_pivot_elimination,
composite_gradient_at_fast, ...) works unchanged on the returned ctx, exactly
as for d20_real_setup.
"""
function d20_real_setup_qmc(; W::Int, U_injected::AbstractMatrix{Float64}, δ::Float64 = 1.0, find_smallest::Bool = true,
        outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
        inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"),
        needs_outer_moment_jacobian::Bool = false,
        # mirrors d20_real_setup's own destination_sample kwarg exactly (context_real_d20.jl) --
        # :exclude_row is the production default (true dimension shrink, ROW dropped as a
        # destination); :all_legacy is the pre-Part-A square D x D opt-out.
        destination_sample::Symbol = :exclude_row,
        # mirrors d20_real_setup's own exclude_diagonal_gravity kwarg exactly (context_real_d20.jl,
        # 2026-07-30 user-directed fix). `false` default reproduces prior behavior bit-exactly.
        exclude_diagonal_gravity::Bool = false)
    destination_sample in (:exclude_row, :all_legacy) ||
        error("d20_real_setup_qmc: destination_sample must be :exclude_row or :all_legacy, got :$destination_sample")
    row_idx = destination_sample == :exclude_row ? D20_REAL : nothing
    so, pp, params_used = build_ad_context_real_d20_qmc(W = W, U_injected = U_injected, row_idx = row_idx, exclude_diagonal_gravity = exclude_diagonal_gravity)
    Dact = so.D; bi = params_used.baseIndex; σ = params_used.σHat; μHat = pp.γ.μHat
    # mirrors d20_real_setup's own bi/row_idx collision guard exactly (context_real_d20.jl) -- a
    # focal country that is not itself a valid destination in the resolved sample makes GT undefined.
    row_idx === nothing || bi != row_idx ||
        error("d20_real_setup_qmc: focal country (baseIndex=$bi) coincides with the omitted ROW " *
              "destination (row_idx=$row_idx) under destination_sample=:exclude_row -- GT is undefined " *
              "for a focal country that is not itself a valid destination in the resolved sample.")
    Ddest = row_idx === nothing ? Dact : Dact - 1
    @unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp
    Aod_offset = 3 + Dact

    θ0_up = build_theta_gammanorm(θ_initial_up, Dact, bi, μHat, σ)
    bounds = theoretical_gammaprime_bounds(γ, σ)
    θ0_up[3+Dact] = clamp(θ0_up[3+Dact], bounds.γp_lo, bounds.γp_hi)

    θ_lo = (θ0_up .* 0.0001)[:]; θ_hi = (θ0_up .* 10000)[:]
    θ_lo[2] = θ0_up[2]; θ_hi[2] = θ0_up[2]
    θ_lo[1] = θ0_up[1]; θ_hi[1] = θ0_up[1]
    for d in 1:Dact
        θ_lo[2+d] = θ0_up[2+d]; θ_hi[2+d] = θ0_up[2+d]
    end
    θ_lo[3+Dact] = bounds.γp_lo; θ_hi[3+Dact] = bounds.γp_hi

    l_full = length(θ0_up)
    free_idx = vcat(3 + Dact, collect(Aod_offset+1:Aod_offset+Dact*Ddest))
    fixed_idx = vcat(1, 2, collect(3:2+Dact))
    fixed_vals = θ0_up[fixed_idx]
    m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
    @assert CS.n_free(m) == 1 + Dact * Ddest

    Aod_free_pos = [1 + (d - 1) * Dact + o for o in 1:Dact, d in 1:Ddest]
    τ = γ.τ
    q_tilde, N_obs = precompute_q_tilde(τ; exclude_diagonal = exclude_diagonal_gravity)

    obj = CS.PsiObjectiveBundleImplicit(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
        outer_constr_index = outer_constr_index, inequality_index = inequality_index,
        complement_index = complement_index, l = l_full, U = U, N = params_used.Jac_W,
        lower_limit = -50, use_cached_x = true,
        outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
        needs_outer_moment_jacobian = needs_outer_moment_jacobian)
    @assert obj.outer_constr_index == obj.d

    return (so = so, pp = pp, D = Dact, D_dest = Ddest, row_idx = row_idx,
            destination_sample = destination_sample,
            # matches d20_real_setup's active_origins/active_destinations convention exactly
            # (context_real_d20.jl) -- origins never restricted, destinations truncated to
            # 1:Ddest under :exclude_row (omitted destination, row_idx, is always Dact, the last
            # index -- see Aod_free_pos's own `d in 1:Ddest` above).
            active_origins = Base.OneTo(Dact), active_destinations = Base.OneTo(Ddest),
            gravity_sample_version = GRAVITY_SAMPLE_VERSION, theta_calibration_version = THETA_CALIBRATION_VERSION,
            W = W, bi = bi, σ = σ, μHat = μHat, γ = γ, U = U,
            θ0_up = θ0_up, θ_lo = θ_lo, θ_hi = θ_hi, l_full = l_full,
            free_idx = free_idx, fixed_idx = fixed_idx, fixed_vals = fixed_vals, m = m,
            Aod_offset = Aod_offset, Aod_free_pos = Aod_free_pos,
            τ = τ, q_tilde = q_tilde, N_obs = N_obs, exclude_diagonal_gravity = exclude_diagonal_gravity, obj = obj,
            nTotalMoments = nTotalMoments, outer_constr_index = outer_constr_index,
            bounds = bounds, δ = δ, find_smallest = find_smallest)
end
