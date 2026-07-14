# ============================================================================
# γ_d ≡ 1 (all d) normalization for the FULL A_od-in-outer-loop model.
#
# Baseline gamma is now normalized to 1 for EVERY destination d everywhere in
# this codebase (moments/hFunction.jl hardcodes it; see that file), so this
# file no longer needs its own gamma array at all — it just leaves the ENTIRE
# A_od matrix free (no A[1,d]=1 pins anywhere), which is what actually made
# this file's normalization distinct from the old A[1,d]=1 gauge.
#
# Why this is a genuine (not cosmetic) change: γ[d] used to enter `denom[d] =
# γ[d]^σ*gdp[d]`, the normalizer for EVERY trade-share moment of destination d,
# for ALL d — not just baseIndex. So this reparameterizes the whole moment
# map, matching the user's intent: "the scale of A_od is no longer arbitrary
# — the units are exactly such that γ_d happens to be 1."
#
# Derivation of the γ'_focal bound (verify-before-trust; matches and
# generalizes sequential_gravity/focal_moments.jl's already-validated
# focal-only derivation, lines ~130-164, which was itself independently
# cross-checked against EXPERIMENTS_FINDINGS.md's earlier all-A audit):
#   γ'_focal/γ_focal = λ_dd^{μ(σ-1)/σ}   (λ_dd = λ[baseIndex,baseIndex], DATA)
#   κ = 1-(γ'_focal/γ_focal)^{σ/(σ-1)} = 1-λ_dd^μ         (σ/(σ-1) exponent
#       here is the one in the DEFINITION of κ itself, from real income
#       W ∝ γ^{σ/(σ-1)} — see moments!.jl:110-112)
# μ ranges over (0,1/(σ-1)], κ=1-λ_dd^μ strictly increasing in μ (0<λ_dd<1), so
#   κ ∈ [0, 1-λ_dd^{1/(σ-1)}]  ⟺  (γ_focal≡1 normalization) γ'_focal ∈ [λ_dd^{1/σ}, 1]
# NOTE the bound exponent is 1/σ, NOT σ/(σ-1) (that exponent belongs to the κ
# formula, not the γ' bound) — this is the point flagged for verification.
#
# θ LAYOUT: kept the SAME LENGTH as the baseline full-A layout (3+D+D^2), to
# reuse all existing offset arithmetic (Aod_offset, outer_constr_index,
# nTotalMoments, gravity-moment wiring) unchanged. Only the SEMANTICS of two
# slot ranges change:
#   θ[3:2+D]   (old γ_θ slots)          -> now INERT/ignored by the moments
#                                           function; caller must PIN them
#                                           (any value; they do nothing).
#   θ[3+D]     (old γ_prime_θ[baseIndex]) -> now γ'_focal DIRECT (the adjusted
#                                           value), bounded to [λ_dd^{1/σ}, 1].
#   θ[Aod_offset+1 : Aod_offset+D^2]     -> ALL free (no A[1,d]=1 pins).
# ============================================================================

"""
    EK_moments_gammanorm!(K, G, θ, U, obj)

Same as `moments/moments!.jl::EK_moments_simple!`, autarky (`counterType==1`)
only, EXCEPT: γ'_focal enters θ DIRECTLY (bypasses the ΔγA_d_prime*Δγμ_d_prime
adjustment the baseline code applies to γ_prime_θ). Everything else
(Aod/AodPow construction, hFunction!/hFunctionCounter! calls, gravity moment,
PMM/NormalizeMoments/SamplingWeights bookkeeping) is byte-identical to
EK_moments_simple! — only the γ'-derivation block differs.
"""
function EK_moments_gammanorm!(K, G, θ, U, obj)
	@unpack wHat, L, LPrime, τ, τPrime, P, σ_Moments, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, Ū, numMomentsSimple, SamplingWeights, PMM, moments_without_var, UPow_scratch, UσPow_scratch, μPow_cache = obj.γ
	@unpack counterExplicit,
	counterType,
	θConstant,
	gravMoment,
	localGravityMoment,
	GravityMomentFirstApproach,
	sameMarginalsMoment,
	independenceMoment,
	momentOrder,
	momentOrderForBaseIndex,
	IndMomentOrder,
	OuterScaling,
	usePMM,
	NormalizeMoments = indicators

	counterType == 1 || error("EK_moments_gammanorm! only implements counterType==1 (autarky)")

	W = size(U, 1)
	D = size(τ, 1)
	T = eltype(θ)

	μ = θ[1]
	σ = θ[2]
	# θ[3:2+D] (old γ_θ slots) intentionally NOT READ: baseline γ is normalized to 1 everywhere.

	wPrime = copy(obj.γ.wPrimeHat)
	insert!(wPrime, baseIndex, 1)

	Aod = ones(T, D, D)
	AodPow = ones(T, D, D)
	Aod_θ = ones(T, D, D)
	Aod_offset = 3 + D   # counterType_θ_offset==0 under counterType==1; unchanged from EK_moments_simple!
	if OuterScaling == 1
		if independenceMoment == 1
			Aod_offset += 1
		end
		Aod_θ = reshape(θ[Aod_offset+1:Aod_offset+D^2], (D, D))
	end

	lambda = reshape(P, (D, D))'

	if θConstant != 1
		Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
	else
		Aod = Aod_θ
	end
	@. AodPow[:, :] = (Aod[:, :] ./ cHat[:, :]) .^ (-μ)

	# ---- γ'_focal DIRECT (γ_d≡1 baseline handled inside hFunction!) ----
	γ_prime = ones(T, D)
	γ_prime[baseIndex] = θ[3+D]   # γ'_focal DIRECT (adjusted value; γ[baseIndex]≡1)
	# --------------------------------------------------------------------

	if counterExplicit == 0
		counterVal = 1 - γ_prime[baseIndex]^(σ / (σ - 1))
		@. K[:] = counterVal
	end

	if θConstant != 1
		# recomputes into the shared Float64 scratch only if μ's value actually changed since the
		# last call (or falls back to a fresh per-call Dual array if μ itself is being
		# differentiated this call) -- see moments/moments!.jl::ensure_UPow!'s docstring.
		UPow, UσPow = ensure_UPow!(UPow_scratch, UσPow_scratch, μPow_cache, U, Uσ, μ)

		Th = Threads.nthreads()
		Threads.@threads for t ∈ 1:Th
			ix0 = round(Int, (t - 1) / Th * W) + 1
			ix1 = round(Int, t / Th * W)
			hFunction!(@view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wHat, τ, σ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat)
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex)
		end
	else
		Th = Threads.nthreads()
		Threads.@threads for t ∈ 1:Th
			ix0 = round(Int, (t - 1) / Th * W) + 1
			ix1 = round(Int, t / Th * W)
			hFunction!(@view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wHat, τ, σ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat)
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex)
		end
	end

	if gravMoment == 1
		newGravityMoment!(G, τ, D, AodPow, GravityMomentFirstApproach)
	end

	GravityMomentFirstApproach == 0 || error("gammanorm variant: GravityMomentFirstApproach not implemented")
	sameMarginalsMoment == 0 || error("gammanorm variant: sameMarginalsMoment not implemented")
	independenceMoment == 0 || error("gammanorm variant: independenceMoment not implemented")

	if θConstant != 1
		simple_end = D^2 + 1   # counterType==1 reduced layout
		@. G[:, 1:simple_end] /= gamma(μ * (1 - σ) + 1)
	end

	if usePMM == 1
		for im ∈ 1:numMomentsSimple
			@. G[:, im] -= PMM[im]
		end
	end

	if NormalizeMoments == 1
		for im ∈ 1:numMomentsSimple-GravityMomentFirstApproach-independenceMoment
			if im ∉ moments_without_var
				@. G[:, im] *= 1 ./ σ_Moments[im]
			end
		end
	end

	for im ∈ 1:numMomentsSimple
		@. G[:, im] *= SamplingWeights[1:W]
	end
	@. K[:] *= SamplingWeights[1:W]

	return nothing
end

"""
    EK_moments_gammanorm_directgp!(K, G, θ, U, obj)

Identical to `EK_moments_gammanorm!` EXCEPT the objective K is γ'_focal DIRECTLY
(`counterVal = γ_prime[baseIndex]`, since γ[baseIndex]≡1 already), not the
power-transformed κ = 1-(γ'/γ)^{σ/(σ-1)}. Since κ is a STRICTLY MONOTONE
(decreasing, σ>1) transform of γ'_focal alone, extremizing γ'_focal directly
and transforming afterward gives the identical bounds — but K is now a purely
LINEAR function of θ (K ≡ θ[3+D], gradient = the unit vector e_{3+D}), instead
of a nonlinear power of a ratio. Tests whether the power-transform itself was
contributing conditioning trouble beyond the A[1,d]=1-vs-γ_d≡1 gauge choice.

SIGN CONVENTION: since κ is DECREASING in γ'_focal, "smallest κ"
(find_smallest=true in the original κ-objective convention) corresponds to
LARGEST γ'_focal, i.e. callers must swap `find_smallest` relative to the
κ-objective runs to get the same κ-direction, then convert back:
`κ = 1 - γp_result^(σ/(σ-1))` where `γp_result` is what `outer_loop` returns
as its (sign-corrected) extremal value.
"""
function EK_moments_gammanorm_directgp!(K, G, θ, U, obj)
	@unpack wHat, L, LPrime, τ, τPrime, P, σ_Moments, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, Ū, numMomentsSimple, SamplingWeights, PMM, moments_without_var, UPow_scratch, UσPow_scratch, μPow_cache = obj.γ
	@unpack counterExplicit,
	counterType,
	θConstant,
	gravMoment,
	localGravityMoment,
	GravityMomentFirstApproach,
	sameMarginalsMoment,
	independenceMoment,
	momentOrder,
	momentOrderForBaseIndex,
	IndMomentOrder,
	OuterScaling,
	usePMM,
	NormalizeMoments = indicators

	counterType == 1 || error("EK_moments_gammanorm_directgp! only implements counterType==1 (autarky)")

	W = size(U, 1)
	D = size(τ, 1)
	T = eltype(θ)

	μ = θ[1]
	σ = θ[2]
	# θ[3:2+D] (old γ_θ slots) intentionally NOT READ: baseline γ is normalized to 1 everywhere.

	wPrime = copy(obj.γ.wPrimeHat)
	insert!(wPrime, baseIndex, 1)

	Aod = ones(T, D, D)
	AodPow = ones(T, D, D)
	Aod_θ = ones(T, D, D)
	Aod_offset = 3 + D
	if OuterScaling == 1
		if independenceMoment == 1
			Aod_offset += 1
		end
		Aod_θ = reshape(θ[Aod_offset+1:Aod_offset+D^2], (D, D))
	end

	lambda = reshape(P, (D, D))'

	if θConstant != 1
		Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
	else
		Aod = Aod_θ
	end
	@. AodPow[:, :] = (Aod[:, :] ./ cHat[:, :]) .^ (-μ)

	γ_prime = ones(T, D)
	γ_prime[baseIndex] = θ[3+D]

	if counterExplicit == 0
		counterVal = γ_prime[baseIndex]   # THE ONLY CHANGE vs EK_moments_gammanorm!: no power transform
		@. K[:] = counterVal
	end

	if θConstant != 1
		# recomputes into the shared Float64 scratch only if μ's value actually changed since the
		# last call (or falls back to a fresh per-call Dual array if μ itself is being
		# differentiated this call) -- see moments/moments!.jl::ensure_UPow!'s docstring.
		UPow, UσPow = ensure_UPow!(UPow_scratch, UσPow_scratch, μPow_cache, U, Uσ, μ)

		Th = Threads.nthreads()
		Threads.@threads for t ∈ 1:Th
			ix0 = round(Int, (t - 1) / Th * W) + 1
			ix1 = round(Int, t / Th * W)
			hFunction!(@view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wHat, τ, σ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat)
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex)
		end
	else
		Th = Threads.nthreads()
		Threads.@threads for t ∈ 1:Th
			ix0 = round(Int, (t - 1) / Th * W) + 1
			ix1 = round(Int, t / Th * W)
			hFunction!(@view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wHat, τ, σ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat)
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex)
		end
	end

	if gravMoment == 1
		newGravityMoment!(G, τ, D, AodPow, GravityMomentFirstApproach)
	end

	GravityMomentFirstApproach == 0 || error("gammanorm variant: GravityMomentFirstApproach not implemented")
	sameMarginalsMoment == 0 || error("gammanorm variant: sameMarginalsMoment not implemented")
	independenceMoment == 0 || error("gammanorm variant: independenceMoment not implemented")

	if θConstant != 1
		simple_end = D^2 + 1
		@. G[:, 1:simple_end] /= gamma(μ * (1 - σ) + 1)
	end

	if usePMM == 1
		for im ∈ 1:numMomentsSimple
			@. G[:, im] -= PMM[im]
		end
	end

	if NormalizeMoments == 1
		for im ∈ 1:numMomentsSimple-GravityMomentFirstApproach-independenceMoment
			if im ∉ moments_without_var
				@. G[:, im] *= 1 ./ σ_Moments[im]
			end
		end
	end

	for im ∈ 1:numMomentsSimple
		@. G[:, im] *= SamplingWeights[1:W]
	end
	@. K[:] *= SamplingWeights[1:W]

	return nothing
end

"γ_dd (domestic/own trade share) for baseIndex — data constant the theoretical bound uses."
lambda_dd_full(γobj) = begin
	D = size(γobj.τ, 1); bi = γobj.baseIndex
	reshape(γobj.P, (D, D))'[bi, bi]
end

"""
    theoretical_gammaprime_bounds(γobj, σ)

(κ_min,κ_max) = (0, 1-λ_dd^{1/(σ-1)}); implied γ'_focal bounds (λ_dd^{1/σ}, 1)
under the γ_d≡1-for-all-d normalization. Same formula as
sequential_gravity/focal_moments.jl::theoretical_kappa_bounds (that function
is for the focal-only reduced model but the closed-form derivation is
identical, since it only ever depended on baseIndex's γ/γ' pair).
"""
function theoretical_gammaprime_bounds(γobj, σ::Real)
	λdd = lambda_dd_full(γobj)
	κ_max = 1 - λdd^(1 / (σ - 1))
	γp_lo = λdd^(1 / σ)
	γp_hi = 1.0
	return (κ_min = 0.0, κ_max = κ_max, γp_lo = γp_lo, γp_hi = γp_hi)
end

"""
    build_theta_gammanorm(θ_initial, D, baseIndex, μHat, σ)

Build a starting point in the new gauge from the OLD (A[1,d]=1-normalized)
θ_initial, so both variants start from (numerically close to) the same
baseline equilibrium. Derivation (generalizes focal_moments.jl's single-column
compensating-scale trick, §6 of SEQUENTIAL_GRAVITY_PROGRESS.md, to every d):
at θ_initial, Aod_θ≡1 and μ=μHat, so γ[d]_old = γ_θ_initial[d] =: γ0[d]
exactly. A uniform rescale of column d's raw A_od by a constant s_d changes
AodPow[·,d] by s_d^{-μ} (since AodPow=(Aod_θ/cHat)^{-μ}), hence the σ-transformed
winning price by s_d^{μ(σ-1)}; matching the OLD target
`P[d1]*γ0[d]^σ*gdp[d]` to the NEW target `P[d1]*gdp[d]` (γ[d]≡1) requires
`s_d^{μ(σ-1)}*γ0[d]^σ = 1`, i.e. `s_d = γ0[d]^{-σ/(μ(σ-1))}`.
γ'_focal(direct) initial value = θ_initial[3+D]/θ_initial[2+baseIndex] (old
γ'_focal_θ / old γ_baseIndex_θ ratio — the model-consistent value under the
new gauge since θ_initial exactly matches F* under the old gauge).
The old γ_θ slots (θ[3:2+D]) are copied over unchanged (they are now inert;
caller must pin their bounds so the search never moves them).
"""
function build_theta_gammanorm(θ_initial::AbstractVector, D::Int, baseIndex::Int, μHat::Real, σ::Real)
	θ = copy(θ_initial)
	Aod_offset = 3 + D
	for d in 1:D
		γ0_d = θ_initial[2+d]
		s_d = γ0_d^(-σ / (μHat * (σ - 1)))
		for o in 1:D
			θ[Aod_offset + o + (d - 1) * D] = s_d
		end
	end
	θ[3+D] = θ_initial[3+D] / θ_initial[2+baseIndex]   # γ'_focal DIRECT initial value
	return θ
end
