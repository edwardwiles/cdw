function moments!(K, G, θ, U, obj)
	# main function that takes empty K and G, and the parameters, and fills in the moment matrices 

	# unpack the gamma (auxiliary parameters) vector
	@unpack wHat, L, LPrime, τ, τPrime, P, PMM, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, SamplingWeights = obj.γ
	@unpack counterExplicit, counterType, θConstant, gravMoment, localGravityMoment, GravityMomentFirstApproach, sameMarginalsMoment, independenceMoment, momentOrder, momentOrderForBaseIndex, IndMomentOrder, OuterScaling = indicators

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
        elseif GravityMomentFirstApproach == 1 && sameMarginalsMoment ==0
            Aod_offset += D^2
		end
		Aod = reshape(vcat(θ[Aod_offset+1:Aod_offset+D^2]), (D, D))

		@. AodPow[:, :] = Aod[:, :] .^ (-μ)
	end


	if θConstant != 1
		# update c and U matrices to the exponents relevant for calculating price
		# nb: calculate here as don't want to do it in each hFunction call
		# only do this if theta / sigma ever vary, otherwise we precalculate

		#UPow = copy(U)
		UPow = zeros(eltype(γ), size(U))
		UσPow = zeros(eltype(γ), size(U))
		for i ∈ 1:length(UPow)
			UPow[i] = U[i]^(-μ)
			UσPow[i] = Uσ[i]^(-μ)
		end

		hFunction!(G, UPow, UσPow, wHat, τ, σ, γ, AodPow, L, P, PMM, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat) # fill in G with baseline moments 
		hFunctionCounter!(K, G, UPow, UσPow, wPrime, τPrime, σ, γ_prime, AodPow, LPrime, PMM, counterType, baseIndex) # fill in G with counterfactual moments, fill in K 
	else
		hFunction!(G, U, Uσ, wHat, τ, σ, γ, AodPow, L, P, PMM, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat)
		hFunctionCounter!(K, G, U, Uσ, wPrime, τPrime, σ, γ_prime, AodPow, LPrime, PMM, counterType, baseIndex)
	end

	if gravMoment == 1
		newGravityMoment!(G, PMM, τ, D, W, γ, AodPow, U, GravityMomentFirstApproach) # add gravity moment if using 
	end

	if GravityMomentFirstApproach == 1
		ν = zeros(D, D) # E[ln ̄U]
        offset = 0

		if sameMarginalsMoment == 0 
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
		GravityMomentFirstApproach!(G, PMM, τ, ν, Aod, cHat, D, Ū, offset)
	end

	if sameMarginalsMoment == 1
		offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach
		CDF_Moments_Size = size(CDF_Moments, 2)
		@. G[:, end-offset-CDF_Moments_Size+1:end-offset] = CDF_Moments[:, :]
	end

	if independenceMoment == 1
		ηk = θ[counterType_θ_offset+3+D+1:counterType_θ_offset+3+D+1]
		ν_probas = θ[end-(IndMomentOrder-1)+1:end]
		offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach + sameMarginalsMoment * (2 * momentOrder * D^2 + 2 * momentOrderForBaseIndex * D + 2 * D^2)
		IndependenceMoment!(Ū, G, D, ηk, Ind_Moments, IndMomentOrder, IndCDF_Cells, ν_probas, offset, refIndex1)
	end

	#Multiply by ISW which are defaulted to 1 if the methodology is not used
	for im ∈ 1:obj.d
		@. G[:, im] *= SamplingWeights[:]
	end
	@. K[:] *= SamplingWeights[:]
end
