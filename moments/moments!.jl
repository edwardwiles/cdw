function EK_moments_simple!(K, G, θ, U, obj)
	# main function that takes empty K and G, and the parameters, and fills in the moment matrices 

	# unpack the gamma (auxiliary parameters) vector
	@unpack wHat, L, LPrime, τ, τPrime, P, σ_Moments, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, Ū, numMomentsSimple, SamplingWeights, PMM = obj.γ
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
	UoModel,
	NormalizeMoments = indicators

	W = size(U, 1)
	D = size(τ, 1)

	# unpack the structural parameters vector
	μ = θ[1]
	σ = θ[2]
	γ = θ[3:3+D-1]

	γ_prime = copy(γ)

	if counterType != 1
		γ_prime = θ[3+D:3+2*D-1]
	else
		γ_prime[baseIndex] = θ[3+D] # we are not interested in the other gamma_primes
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

	if counterExplicit == 0
		# insert relevant k function if counterfactual does not depend on U
		counterVal = (γ[baseIndex] / γ_prime[baseIndex])^(σ / (σ - 1)) - 1
		@inbounds for i ∈ 1:W
			K[i] = counterVal
		end
	end

	Aod = ones(D, D)
	AodPow = ones(D, D)

	if OuterScaling == 1 # Aod model
		Aod_offset = counterType_θ_offset + 3 + D
		if independenceMoment == 1
			Aod_offset += 1
		elseif GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0
			Aod_offset += D^2
		end
		Aod = reshape(vcat(θ[Aod_offset+1:Aod_offset+D^2]), (D, D))

	end

	@. AodPow[:, :] = (Aod[:, :] ./ cHat[:, :]) .^ (-μ)

	if θConstant != 1
		# update c and U matrices to the exponents relevant for calculating price
		# nb: calculate here as don't want to do it in each hFunction call
		# only do this if theta / sigma ever vary, otherwise we precalculate


		#UPow = copy(U)
		UPow = zeros(eltype(γ), size(U))
		UσPow = zeros(eltype(γ), size(U))
		T = Threads.nthreads()
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			@. UPow[ix0:ix1, :] = U[ix0:ix1, :] .^ (-μ)
			@. UσPow[ix0:ix1, :] = Uσ[ix0:ix1, :] .^ (-μ)
			hFunction!(@view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, UoModel) # fill in G with baseline moments 
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex, UoModel) # fill in G with counterfactual moments, fill in K 
		end
	else
		#hFunction!(G, U, Uσ, wHat, τ, σ, γ, AodPow, L, P,  counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat)
		# hFunctionCounter!(K, G, U, Uσ, wPrime, τPrime, σ, γ_prime, AodPow, LPrime,  counterType, baseIndex)

		T = Threads.nthreads()
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			hFunction!(@view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, UoModel)
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex, UoModel)
		end
	end

	if gravMoment == 1
		newGravityMoment!(G, τ, D, W, γ, AodPow, U, GravityMomentFirstApproach, UoModel) # add gravity moment if using 
	end

	if GravityMomentFirstApproach == 1
		ν = zeros(D, D) # E[ln ̄U]
		offset = 0

		if sameMarginalsMoment == 0 && UoModel == 0
			ν_offset = counterType_θ_offset + 3 + D
			if OuterScaling == 1
				ν_offset += D^2
				if independenceMoment == 1 # not necessary because sameMarginalsMoment ==0
					ν_offset += 1
				end
			end
			ν = reshape(vcat(θ[ν_offset+1:ν_offset+D^2]), (D, D))

			offset = D^2 + 2 * D
			if counterType != 1
				offset += (D - 1)
			end
		end
		GravityMomentFirstApproach!(G, τ, ν, Aod, cHat, D, Ū, offset, UoModel)
	end

	if sameMarginalsMoment == 1
		offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach
		CDF_Moments_Size = size(CDF_Moments, 2)
		@. G[:, end-offset-CDF_Moments_Size+1:end-offset] = CDF_Moments[1:W, :]
	end

	if independenceMoment == 1
		ηk = θ[counterType_θ_offset+3+D+1:counterType_θ_offset+3+D+1]
		ν_probas = θ[end-IndMomentOrder+1:end]

		offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach
		if UoModel == 0
			offset += sameMarginalsMoment * (2 * momentOrder * D^2 + 2 * momentOrderForBaseIndex * D + 2 * D^2)
		else
			offset += sameMarginalsMoment * (2 * momentOrderForBaseIndex * D + 2 * D)
		end
		pairewiseIndependenceMoment!(Ū, G, D, ηk, Ind_Moments, IndMomentOrder, IndCDF_Cells, ν_probas, offset, refIndex1, UoModel, W)

	end

	# normalize the moments so we do not require useless precision 
	if usePMM == 1
		for im ∈ 1:numMomentsSimple
			@. G[:, im] -= PMM[im]
		end
	end

	if NormalizeMoments == 1
		KNITRO_tol = 10^(-6)
		for im ∈ 1:numMomentsSimple
			@. G[:, im] *= σ_Moments[im] > KNITRO_tol^2 ? KNITRO_tol ./ σ_Moments[im] : 1
		end
	end
	#Multiply by ISW which are defaulted to 1 if the methodology is not used
	for im ∈ 1:numMomentsSimple
		@. G[:, im] *= SamplingWeights[:]
	end
	@. K[:] *= SamplingWeights[:]

end

function EK_moments!(K, G, θ, U, obj)
	# main function that takes empty K and G, and the parameters, and fills in the moment matrices 
	@unpack upper_moment_start_index, τ,baseIndex, PMM, σ_Moments, Moments_CS, SamplingWeights, indicators = obj.γ
	@unpack useConfidenceIntervals, usePMM, NormalizeMoments, counterType = indicators

	EK_moments_simple!(K, @view(G[:, upper_moment_start_index:end]), θ, U, obj)

	if useConfidenceIntervals == 1
		D = size(τ, 1)

		if counterType != 1
			cInd = D^2 + D - 1
			dInd = D^2 + 2 * D - 1
		else
			cInd = D^2
			dInd = D^2 + D
		end

		for im in 1:upper_moment_start_index-1
		 is_not_gamma_moment = (im ∉ cInd+1:cInd+D && im ∉ dInd+1:dInd+D) ? 1 : 0 #gamma is actually a gamma hat.
		 @. G[:, upper_moment_start_index-1+im] += is_not_gamma_moment*(-1) .* Moments_CS[im, 2]
		 @. G[:, im] = is_not_gamma_moment*( Moments_CS[im, 1] - Moments_CS[im, 2]) .- G[:, upper_moment_start_index-1+im]
		end
	end
end
