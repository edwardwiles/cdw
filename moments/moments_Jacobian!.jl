function EK_moments_Jacobian_Simple!(jac_K, jac_G, θ, U, obj)
	# main function that takes empty K and G, and the parameters, and fills in the moment matrices 
	# unpack the gamma (auxiliary parameters) vector
	@unpack wHat, L, LPrime, τ, τPrime, P, PMM, σ_Moments, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, SamplingWeights, Ū, numMomentsSimple, moments_without_var = obj.γ
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

	β = 0.01

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
		Δγμ_d_prime = (gamma(μ*(1-σ)+1)/gamma(μHat*(1-σ)+1))^(1/σ)

		γ[d] = γ_θ[d]*ΔγA_d*Δγμ_d

		γ_prime[d] = γ_prime_θ[d]*ΔγA_d_prime*Δγμ_d_prime

	end

	# calc internal jacobians

	∂γprime∂γprimeθ = zeros(D)
	∂γ∂γθ = zeros(D)
	∂γprime∂γθ= zeros(D)
	∂γprime∂Aθ= zeros(D)
	∂A∂Aθ = zeros(D,D)
	∂γ∂Aθ = zeros(D,D)
	∂γ∂μ= zeros(D)
	∂γprime∂μ= zeros(D)
	∂A∂μ =  zeros(D,D)

	for d=1:D
		∂γprime∂γprimeθ[d] = γ_prime[d]/γ_prime_θ[d]
		∂γ∂γθ[d] = γ[d]/γ_θ[d]
		#∂γprime∂γθ[d] = γ_prime[d]/γ_θ[d]
		∂γprime∂γθ[d] = 0 # not implemented yet
		∂γprime∂Aθ[d] = (μ*(σ-1)/σ)*γ_prime[d]/Aod_θ[d,d]
		∂γ∂μ[d] = γ[d]*((1-σ)/σ)*(polygamma(0,μ*(1-σ)+1) +log(lambda[1,d]))
		∂γprime∂μ[d] = γ_prime[d]*((1-σ)/σ)*(polygamma(0,μ*(1-σ)+1) +log(lambda[1,d]/lambda[d,d]) - log(Aod_θ[d,d]))
		for o=1:D
			∂A∂Aθ[o,d] = (Aod[o,d]/cHat[o,d])/Aod_θ[o,d]
			∂γ∂Aθ[o,d] = (μ*(σ-1)*lambda[o,d]/σ)*γ[d]/Aod_θ[o,d]
			∂A∂μ[o,d] = -(1/μ^2)*log(wHat[o]*τ[o,d]/(wHat[1]*τ[1,d]))*(Aod[o,d]/cHat[o,d])
			∂γ∂μ[d] -= γ[d]*((1-σ)/σ)*lambda[o,d]*log(Aod_θ[o,d])
		end
	end


	# I work with this assumption for now θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], Aod, CDF)
	# it doesn't work for variable Aod or variable mu
	# TO DO: make it adaptive to theta  
	@. jac_G[:, :, :] = 0 # assume no variation in mu and sigma, for now
	@. jac_K[:, :] = 0

	# we divide G by  Γ(μ*(1-σ)+1)) to keep moments small when mu increases. 
	# this is why we have to adjust the derivatives by -Γ'/Γ^2 = -polygamma/Γ
	if θConstant != 1
	EK_moments_simple!(@view(jac_K[:,1:1]), @view(jac_G[:,:,1:1]), θ, U, obj)
	@. jac_K[:,1] *= 0 # no need to adjust the Kappa
	if size(jac_G, 2) > D^2+2*D
		@. jac_G[:,D^2+2*D+1:end,1] *= 0 # no need to adjust the other moments
	end
	@. jac_G[:,1:D^2+2*D,1] *= -(1-σ)*polygamma(0,μ*(1-σ)+1)*gamma(μ*(1-σ)+1) 
	end
	# insert relevant k function if counterfactual does not depend on U
	#K = (γ[d] / γ_prime[d])^(σ / (σ - 1)) - 1
	
	∂K∂γ = (σ / (σ - 1))*γ[baseIndex]^((σ / (σ - 1)) - 1)*(1 / γ_prime[baseIndex])^(σ / (σ - 1))
	∂K∂γ_prime = (γ[baseIndex])^(σ / (σ - 1)) * (-(σ / (σ - 1))) * γ_prime[baseIndex]^(-σ / (σ - 1) - 1)
	@. jac_K[:, 2+baseIndex] += ∂K∂γ*∂γ∂γθ[baseIndex]
	@. jac_K[:, 2+D+1] += ∂K∂γ_prime*∂γprime∂γprimeθ[baseIndex]
	
	if θConstant != 1
		@. jac_K[:, 1] += ∂K∂γ*∂γ∂μ[baseIndex]
		@. jac_K[:, 1] += ∂K∂γ_prime*∂γprime∂μ[baseIndex]
	end
	
	if OuterScaling == 1
		@. jac_K[:, Aod_offset + baseIndex + (baseIndex - 1) * D] += ∂K∂γ_prime*∂γprime∂Aθ[baseIndex]
	
		for o=1:D
			jac_index_o = Aod_offset + o + (baseIndex - 1) * D
			@. jac_K[:, jac_index_o] += ∂K∂γ*∂γ∂Aθ[o,baseIndex]
		end
	end


	if θConstant != 1
		# update c and U matrices to the exponents relevant for calculating price
		# nb: calculate here as don't want to do it in each hFunction call
		# only do this if theta / sigma ever vary, otherwise we precalculate
		#UPow = copy(U)

		
		UPow = zeros(eltype(γ), size(U))
		UσPow = zeros(eltype(γ), size(U))
		hFunction_jacobian_copy_only!(jac_G, UPow, UσPow, wHat, τ, σ, γ, Aod, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, μ, UoModel, OuterScaling, Aod_offset, θConstant, ∂γ∂γθ, ∂γ∂μ, ∂γ∂Aθ)

		T = Threads.nthreads()
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			@. UPow[ix0:ix1, :] = U[ix0:ix1, :] .^ (-μ)
			@. UσPow[ix0:ix1, :] = Uσ[ix0:ix1, :] .^ (-μ)
		end
		Threads.@threads for t ∈ 1:T
				ix0 = round(Int, (t - 1) / T * W) + 1
				ix1 = round(Int, t / T * W)
			hFunction_jacobian_calculation!(
				@view(jac_G[ix0:ix1, :, :]),
				@view(UPow[ix0:ix1, :]),
				@view(UσPow[ix0:ix1, :]),
				wHat,
				τ,
				σ,
				γ,
				Aod,
				AodPow,
				L,
				P,
				counterType,
				gravMoment,
				localGravityMoment,
				GravityMomentFirstApproach,
				μHat,
				μ,
				UoModel,
				OuterScaling,
				Aod_offset,
				θConstant,
				∂A∂Aθ, ∂A∂μ, cHat
			)
			
			hFunctionCounter_jacobian!(
				@view(jac_K[ix0:ix1, :]),
				@view(jac_G[ix0:ix1, :, :]),
				@view(UPow[ix0:ix1, :]),
				@view(UσPow[ix0:ix1, :]),
				wPrime,
				τPrime,
				σ,
				γ_prime,
				Aod,
				AodPow,
				LPrime,
				counterType,
				baseIndex,
				μ,
				μHat,
				UoModel,
				OuterScaling,
				Aod_offset,
				θConstant,
				∂γprime∂γprimeθ, ∂γprime∂γθ, ∂γprime∂Aθ, ∂γprime∂μ,∂A∂Aθ, ∂A∂μ, cHat

			) # fill in G with counterfactual moments, fill in K 
			
		end
			 
			else


		hFunction_jacobian_copy_only!(jac_G, U,Uσ, wHat, τ, σ, γ, Aod, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, μ, UoModel, OuterScaling, Aod_offset, θConstant, ∂γ∂γθ, ∂γ∂μ, ∂γ∂Aθ)
		hFunctionCounter_jacobian!(jac_K, jac_G, U, Uσ, wPrime, τPrime, σ, γ_prime, Aod, AodPow, LPrime, counterType, baseIndex, μ,μHat, UoModel, OuterScaling, Aod_offset, θConstant,∂γprime∂γprimeθ, ∂γprime∂γθ, ∂γprime∂Aθ, ∂γprime∂μ,∂A∂Aθ, ∂A∂μ, cHat)


		T = Threads.nthreads()
		Threads.@threads for t ∈ 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			hFunction_jacobian_calculation!(
				@view(jac_G[ix0:ix1, :, :]),
				@view(U[ix0:ix1, :]),
				@view(Uσ[ix0:ix1, :]),
				wHat,
				τ,
				σ,
				γ,
				Aod,
				AodPow,
				L,
				P,
				counterType,
				gravMoment,
				localGravityMoment,
				GravityMomentFirstApproach,
				μHat,
				μ,
				UoModel,
				OuterScaling,
				Aod_offset,
				θConstant,
				∂A∂Aθ, ∂A∂μ, cHat
			)
		end

	end

	if independenceMoment == 1
		η_index = counterType_θ_offset + 3 + D + 1
		ηk = θ[counterType_θ_offset+3+D+1:counterType_θ_offset+3+D+1]
		ν_probas = θ[end-IndMomentOrder+1:end]

		offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach
		if UoModel == 0
			offset += sameMarginalsMoment * (2 * momentOrder * D^2 + 2 * momentOrderForBaseIndex * D + 2 * D^2)
		else
			offset += sameMarginalsMoment * (2 * momentOrderForBaseIndex * D + 2 * D)
		end

		pairewiseIndependenceMoment_jac!(Ū, jac_G, D, ηk, @view(Ind_Moments[1:W, :]), IndMomentOrder, IndCDF_Cells, ν_probas, offset, refIndex1, η_index, UoModel, GravityMomentFirstApproach)
	end

	if GravityMomentFirstApproach == 1
		ν = zeros(D, D) # E[ln ̄U]

		offset = D^2 + 2 * D
		if counterType != 1
			offset += (D - 1)
		end
		
		ν_offset = 0
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
		GravityMomentFirstApproach_Jacobian!(jac_G, τ, ν, Aod, cHat, D, Ū, offset, UoModel, sameMarginalsMoment, Aod_offset, ν_offset, θConstant,cHat, ∂A∂Aθ, ∂A∂μ)
	end


	if NormalizeMoments == 1
		for im ∈ 1:numMomentsSimple - GravityMomentFirstApproach - independenceMoment
			if  im ∉ moments_without_var
				@. jac_G[:, im, :] *=  1 ./ σ_Moments[im]
			end
		end
	end

	if θConstant != 1
	@. jac_G[:, 1:D^2+2*D,:] /= gamma(μ*(1-σ)+1)
	end
	#@show mean(jac_G[:, 1:D^2+2*D, 1], dims=1)
	

end

function EK_moments_Jacobian!(jac_K, jac_G, θ, U, obj)
	# main function that takes empty K and G, and the parameters, and fills in the moment matrices 
	# unpack the gamma (auxiliary parameters) vector

	@unpack upper_moment_start_index, PMM, σ_Moments, Moments_CS, indicators = obj.γ
	@unpack useConfidenceIntervals, usePMM, NormalizeMoments = indicators

	EK_moments_Jacobian_Simple!(jac_K, @view(jac_G[:, upper_moment_start_index:end, :]), θ, U, obj)

	if useConfidenceIntervals == 1
		for im in 1:upper_moment_start_index-1
			@. jac_G[:, im, :] = .-jac_G[:, upper_moment_start_index-1+im, :]
		end
	end
end

