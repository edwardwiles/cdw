function EK_moments_simple!(K, G, θ, U, obj)
	# main function that takes empty K and G, and the parameters, and fills in the moment matrices 

	# unpack the gamma (auxiliary parameters) vector
	@unpack wHat, L, LPrime, τ, τPrime, P, σ_Moments, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, Ū, numMomentsSimple, SamplingWeights, PMM, moments_without_var = obj.γ
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



	Aod = ones(D, D)
	AodPow = ones(D, D)

	Aod_θ = ones(D, D)
	Aod_offset = counterType_θ_offset + 3 + D
	if OuterScaling == 1 # Aod model
		if independenceMoment == 1
			Aod_offset += 1
		#elseif GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0 && UoModel == 0
		#	Aod_offset += D^2
		end
		Aod_θ = reshape(vcat(θ[Aod_offset+1:Aod_offset+D^2]), (D, D))
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

	# adjust the gamma_primes

	γ = copy(γ_θ)
	γ_prime = copy(γ_θ)

	for d=1:D
		ΔγA_d = 1
		for o=1:D
			d1 = d + (o - 1) * D
			ΔγA_d *= Aod_θ[o,d]^(P[d1]*μ*(σ-1)/σ)
		end
		Δγμ_d = ((gamma(μ*(1-σ)+1)/gamma(μHat*(1-σ)+1))^(1/σ))*lambda[1,d]^((1-σ)*(μ-μHat)/σ) 

		ΔγA_d_prime = Aod_θ[d,d]^(μ*(σ-1)/σ)
		#Δγμ_d_prime = (gamma(μ*(1-σ)+1)/gamma(μHat*(1-σ)+1))^(1/σ)
		Δγμ_d_prime = ((gamma(μ*(1-σ)+1)/gamma(μHat*(1-σ)+1))^(1/σ))*(lambda[1,d]/lambda[d,d])^((1-σ)*(μ-μHat)/σ) 

		γ[d] = γ_θ[d]*ΔγA_d*Δγμ_d

		γ_prime[d] = γ_prime_θ[d]*ΔγA_d_prime*Δγμ_d_prime

	end

	if counterExplicit == 0
		# insert relevant k function if counterfactual does not depend on U
		counterVal = (γ[baseIndex] / γ_prime[baseIndex])^(σ / (σ - 1)) - 1
		@. K[:] = counterVal
	end

	#@show μ
	#@show γ[baseIndex]
	#@show γ_prime[baseIndex]
	#@show Aod
	#@show K[1]
	#@show lambda[baseIndex,baseIndex]^(-μ)-1

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
			hFunction!(@view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat, UoModel) # fill in G with baseline moments 
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex, UoModel) # fill in G with counterfactual moments, fill in K 
		end
	else
		#hFunction!(G, U, Uσ, wHat, τ, σ, γ, AodPow, L, P,  counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat)
		# hFunctionCounter!(K, G, U, Uσ, wPrime, τPrime, σ, γ_prime, AodPow, LPrime,  counterType, baseIndex)

		T = Threads.nthreads()
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			hFunction!(@view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat, UoModel)
			hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex, UoModel)
		end
	end

	if gravMoment == 1
		newGravityMoment!(G, τ, D, W, γ, AodPow, U, GravityMomentFirstApproach, UoModel) # add gravity moment if using 
	end

	
	if GravityMomentFirstApproach == 1
		ν = zeros(D, D) # E[ln ̄U]
		
		offset = D^2 + 2 * D
		if counterType != 1
			offset += (D - 1)
		end

		if sameMarginalsMoment == 0 && UoModel == 0
			ν_offset = counterType_θ_offset + 3 + D
			if OuterScaling == 1
				ν_offset += D^2
				if independenceMoment == 1 # not necessary because sameMarginalsMoment ==0
					ν_offset += 1
				end
			end
			ν = reshape(vcat(θ[ν_offset+1:ν_offset+D^2]), (D, D))
		end
		GravityMomentFirstApproach!(G, τ, ν, Aod, cHat, D, Ū, offset, UoModel, sameMarginalsMoment)
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
		if UoModel == 0
			offset += sameMarginalsMoment * (2 * momentOrder * D^2 + 2 * momentOrderForBaseIndex * D + 2 * D^2)
		else
			offset += sameMarginalsMoment * (2 * momentOrderForBaseIndex * D + 2 * D)
		end
		pairewiseIndependenceMoment!(Ū, G, D, ηk, @view(Ind_Moments[1:W,:]), IndMomentOrder, IndCDF_Cells, ν_probas, offset, refIndex1, UoModel, W, GravityMomentFirstApproach)
	end

	#To change for other counterfactuals
	if θConstant != 1
	@. G[:, 1:D^2+2*D] /= gamma(μ*(1-σ)+1)
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
	@unpack gravMoment, localGravityMoment, GravityMomentFirstApproach, useConfidenceIntervals, usePMM, NormalizeMoments, counterType, independenceMoment, UoModel, IndMomentOrder = indicators

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
			remove_ind_moments = pairewiseIndependenceMoment_indices_to_remove_from_inequality(@view(G[:, upper_moment_start_index:end]), D, Ind_Moments, IndMomentOrder, offset, UoModel)
			remove_moments = vcat(remove_moments, remove_ind_moments)
		end

		for im in 1:upper_moment_start_index-1
			keep_inequality_moment = im ∉ remove_moments ? 1 : 0
			@. G[:, upper_moment_start_index-1+im] += keep_inequality_moment * (-1) .* Moments_CS[im, 2]
			@. G[:, im] = keep_inequality_moment * (Moments_CS[im, 1] - Moments_CS[im, 2]) .- G[:, upper_moment_start_index-1+im]
		end


	end
end
