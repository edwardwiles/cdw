"""
    ensure_UPow!(UPow_scratch, UσPow_scratch, μPow_cache, U, Uσ, μ) -> (UPow, UσPow)

Returns U^(-μ), Uσ^(-μ), recomputing into the shared Float64 scratch buffers only if μ's VALUE
has changed since the last call (μPow_cache holds the μ value the scratch currently corresponds
to; starts at NaN so the first call always recomputes). Safe even when θ overall is
ForwardDiff-Dual-typed (e.g. Aod_θ is being differentiated) as long as μ itself carries no
nonzero partials for this particular call -- in that case only Aod/AodPow (not U^(-μ)) actually
needs to be Dual, and hFunction!/hFunctionCounter!'s promote_type-based scratch typing handles
the resulting mixed Float64/Dual arithmetic correctly.

If μ DOES carry nonzero partials (μ itself is being differentiated), the cache is bypassed
entirely and a fresh Dual-typed array is computed every call, exactly as before -- reusing the
cache here would silently drop μ's own derivative.
"""
function ensure_UPow!(UPow_scratch, UσPow_scratch, μPow_cache::Ref{Float64}, U, Uσ, μ)
	W = size(U, 1)
	T = Threads.nthreads()

	if μ isa ForwardDiff.Dual && !iszero(ForwardDiff.partials(μ))
		UPow = zeros(eltype(μ), size(U))
		UσPow = zeros(eltype(μ), size(U))
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			@. UPow[ix0:ix1, :] = U[ix0:ix1, :] ^ (-μ)
			@. UσPow[ix0:ix1, :] = Uσ[ix0:ix1, :] ^ (-μ)
		end
		return UPow, UσPow
	end

	μ_val = ForwardDiff.value(μ)
	if μPow_cache[] !== μ_val
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			@. UPow_scratch[ix0:ix1, :] = U[ix0:ix1, :] ^ (-μ_val)
			@. UσPow_scratch[ix0:ix1, :] = Uσ[ix0:ix1, :] ^ (-μ_val)
		end
		μPow_cache[] = μ_val
	end
	return UPow_scratch, UσPow_scratch
end

function EK_moments_simple!(K, G, θ, U, obj)
	# main function that takes empty K and G, and the parameters, and fills in the moment matrices

	# unpack the gamma (auxiliary parameters) vector
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

	W = size(U, 1)
	D = size(τ, 1)

	# unpack the structural parameters vector
	μ = θ[1]
	σ = θ[2]
	# θ[3:2+D] (old gamma_theta slots): baseline gamma is normalized to 1 for every destination
	# (the A_od matrix absorbs all of the scale freedom), so these slots are only read below to
	# size/seed gamma_prime_theta under autarky (their values do not otherwise enter the moments).
	γ_θ = θ[3:3+D-1]

	γ_prime_θ = copy(γ_θ)

	if counterType != 1
		γ_prime_θ = θ[3+D:3+2*D-1]
	else
		γ_prime_θ[baseIndex] = θ[3+D] # we are not interested in the other gamma_primes
	end

	if counterType != 1 # needs to be adjusted
		wPrime = θ[3+2*D:3+2*D+(D-1)-1]
	else
		wPrime = copy(obj.γ.wPrimeHat)
	end

	# add 1 (normalised wage) into the w' vector at appropriate index
	insert!(wPrime, baseIndex, 1)

	counterType_θ_offset = 0
	if counterType != 1
		counterType_θ_offset = 2 * (D - 1) # D-1 wagesPrime and D-1 gamma_primes
	end



	# NB: element type follows θ so the moment map accepts ForwardDiff Duals
	# (use_Jacobian=0 autodiff path). For Float64 θ this is identical to ones(D,D).
	Aod = ones(eltype(θ), D, D)
	AodPow = ones(eltype(θ), D, D)

	Aod_θ = ones(eltype(θ), D, D)
	Aod_offset = counterType_θ_offset + 3 + D
	if OuterScaling == 1 # Aod model
		if independenceMoment == 1
			Aod_offset += 1
		end
		Aod_θ = reshape(θ[Aod_offset+1:Aod_offset+D^2], (D, D))
	end

	lambda = reshape(P, (D, D))'

	if θConstant != 1
		# adjust Aod such that if μ varies and Aod = 1, the model still matches trade shares for F= Frechet

		Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1/μ)) .* (lambda ./ lambda[1, :]')
	else
		Aod = Aod_θ
	end

	#Aod = Delta^A(μ) Aod_θ see the notes


	@. AodPow[:, :] = (Aod[:, :] ./ cHat[:, :]) .^ (-μ)

	# adjust the gamma_prime (the counterfactual gamma; baseline gamma is normalized to 1, see above)

	γ_prime = copy(γ_prime_θ)

	for d=1:D
		ΔγA_d_prime = Aod_θ[d,d]^(μ*(σ-1)/σ)
		Δγμ_d_prime = ((gamma(μ*(1-σ)+1)/gamma(μHat*(1-σ)+1))^(1/σ))*(lambda[1,d]/lambda[d,d])^((1-σ)*(μ-μHat)/σ)

		γ_prime[d] = γ_prime_θ[d]*ΔγA_d_prime*Δγμ_d_prime

	end

	if counterExplicit == 0
		# insert relevant k function if counterfactual does not depend on U
		# GT defined baseline -> autarky: 1 - (γ'/γ)^{σ/(σ-1)}, with γ[baseIndex]≡1.  NB the code's γ
		# satisfies γ^σ·gdp = E[min_o p_od^{1-σ}] = P^{1-σ}, i.e. γ = P^{(1-σ)/σ} (extra σ vs
		# γ=P^{1-σ}), so real income W ∝ 1/P ∝ γ^{σ/(σ-1)} and the exponent here is σ/(σ-1), not
		# 1/(σ-1).
		counterVal = 1 - γ_prime[baseIndex]^(σ / (σ - 1))
		@. K[:] = counterVal
	end

	#@show μ
	#@show γ_prime[baseIndex]
	#@show Aod
	#@show K[1]
	#@show lambda[baseIndex,baseIndex]^(-μ)-1

	if θConstant != 1
		# update c and U matrices to the exponents relevant for calculating price
		# nb: calculate here as don't want to do it in each hFunction call
		# only do this if theta / sigma ever vary, otherwise we precalculate

		# recomputes into the shared Float64 scratch only if μ's value actually changed since the
		# last call (or falls back to a fresh per-call Dual array if μ itself is being
		# differentiated this call) -- see ensure_UPow!'s docstring.
		UPow, UσPow = ensure_UPow!(UPow_scratch, UσPow_scratch, μPow_cache, U, Uσ, μ)

		T = Threads.nthreads()
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			hFunction!(@view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wHat, τ, σ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat) # fill in G with baseline moments
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex) # fill in G with counterfactual moments, fill in K
		end
	else
		T = Threads.nthreads()
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			hFunction!(@view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wHat, τ, σ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat)
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex)
		end
	end

	if gravMoment == 1
		# The price is p_od = w_o·AodPow·τ_od·U^μ with productivity z_o = U^{-μ}, i.e.
		# MC_od = w_o·τ_od/(A_od·z_o) with the raw structural A_od = 1/AodPow. Hence
		# ΔΔ ln(AodPow) = -ΔΔ ln(A_od), and Σ(ΔΔlnτ)·ΔΔ ln(AodPow)=0 is exactly the gravity
		# consistency condition Σ ΔΔlnτ·ΔΔlnA_od = 0. So pass AodPow (= 1/A_od).
		newGravityMoment!(G, τ, D, AodPow, GravityMomentFirstApproach) # add gravity moment if using
	end


	if GravityMomentFirstApproach == 1
		ν = zeros(D, D) # E[ln ̄U]; always zero since ΔΔ E[ln Ū] = 0 under UoModel=1

		offset = D^2 + 2 * D
		if counterType != 1
			offset += (D - 1)
		end

		GravityMomentFirstApproach!(G, τ, ν, Aod, cHat, D, offset)
	end

	if sameMarginalsMoment == 1
		offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach+independenceMoment
		CDF_Moments_Size = size(CDF_Moments, 2)
		@. G[:, end-offset-CDF_Moments_Size+1:end-offset] =CDF_Moments[1:W, :]
	end

	if independenceMoment == 1
		ηk = θ[counterType_θ_offset+3+D+1:counterType_θ_offset+3+D+1]
		ν_probas = θ[end-IndMomentOrder+1:end]

		offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach + independenceMoment
		offset += sameMarginalsMoment * (2 * momentOrderForBaseIndex * D + 2 * D)
		pairewiseIndependenceMoment!(Ū, G, D, ηk, @view(Ind_Moments[1:W,:]), IndMomentOrder, IndCDF_Cells, ν_probas, offset, refIndex1, W, GravityMomentFirstApproach)
	end

	#To change for other counterfactuals
	if θConstant != 1
	# EK simple moments to normalize by the gamma factor: full D^2+2D layout, or the reduced
	# D^2+1 (trade shares + 1 counterfactual) under autarky's trimmed moment set.
	simple_end = counterType == 1 ? D^2 + 1 : D^2 + 2*D
	@. G[:, 1:simple_end] /= gamma(μ*(1-σ)+1)
	end

	# normalize the moments so we do not require useless precision
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

	#Multiply by ISW which are defaulted to 1 if the methodology is not used
	for im ∈ 1:numMomentsSimple
		@. G[:, im] *= SamplingWeights[1:W]
	end
	@. K[:] *= SamplingWeights[1:W]

end

function EK_moments!(K, G, θ, U, obj)
	# main function that takes empty K and G, and the parameters, and fills in the moment matrices
	@unpack upper_moment_start_index, τ, baseIndex, PMM, σ_Moments, Moments_CS, SamplingWeights, indicators, Ind_Moments, moments_without_var = obj.γ
	@unpack gravMoment, localGravityMoment, GravityMomentFirstApproach, useConfidenceIntervals, usePMM, NormalizeMoments, counterType, independenceMoment, IndMomentOrder = indicators

	EK_moments_simple!(K, @view(G[:, upper_moment_start_index:end]), θ, U, obj)

	if useConfidenceIntervals == 1
		D = size(τ, 1)

		cInd = D^2
		dInd = D^2 + D
		if counterType != 1
			cInd = D^2 + D - 1
			dInd = D^2 + 2 * D - 1
		end
		remove_moments = vcat(cInd+1:cInd+D, dInd+1:dInd+D, moments_without_var)

		if independenceMoment == 1
			offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach
			remove_ind_moments = pairewiseIndependenceMoment_indices_to_remove_from_inequality(@view(G[:, upper_moment_start_index:end]), D, Ind_Moments, IndMomentOrder, offset)
			remove_moments = vcat(remove_moments, remove_ind_moments)
		end

		for im in 1:upper_moment_start_index-1
			keep_inequality_moment = im ∉ remove_moments ? 1 : 0
			@. G[:, upper_moment_start_index-1+im] += keep_inequality_moment * (-1) .* Moments_CS[im, 2]
			@. G[:, im] = keep_inequality_moment * (Moments_CS[im, 1] - Moments_CS[im, 2]) .- G[:, upper_moment_start_index-1+im]
		end


	end
end
