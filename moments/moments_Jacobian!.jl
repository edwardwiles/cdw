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

	Aod = ones(D, D)
	AodPow = ones(D, D)
	Aod_offset = 0
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

	# I work with this assumption for now θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], Aod, CDF)
	# it doesn't work for variable Aod or variable mu
	# TO DO: make it adaptive to theta  
	@. jac_G[:, :, :] = 0 # assume no variation in mu and sigma, for now
	@. jac_K[:, :] = 0
	# insert relevant k function if counterfactual does not depend on U
	#K = (γ[d] / γ_prime[d])^(σ / (σ - 1)) - 1
	@. jac_K[:, 2+baseIndex] = (σ / (σ - 1)) * γ[baseIndex]^((σ / (σ - 1)) - 1) * (1 / γ_prime[baseIndex])^(σ / (σ - 1))
	@. jac_K[:, 2+D+1] = (γ[baseIndex])^(σ / (σ - 1)) * (-(σ / (σ - 1))) * γ_prime[baseIndex]^(-σ / (σ - 1) - 1)

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
				UoModel,
				OuterScaling,
				Aod_offset,
				θConstant,
			) # fill in G with counterfactual moments, fill in K 
		end
		hFunction_jacobian_copy_only!(jac_G, UPow, UσPow, wHat, τ, σ, γ, Aod, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, μ, UoModel, OuterScaling, Aod_offset)
	else


		hFunction_jacobian_copy_only!(jac_G, U,Uσ, wHat, τ, σ, γ, Aod, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, μ, UoModel, OuterScaling, Aod_offset)
		hFunctionCounter_jacobian!(jac_K, jac_G, U, Uσ, wPrime, τPrime, σ, γ_prime, Aod, AodPow, LPrime, counterType, baseIndex, μ, UoModel, OuterScaling, Aod_offset, θConstant)


		T = Threads.nthreads()
		@show T
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
			)
		end

	end
	# actually don't need to do anything for the same marginal moments, they have, by design zero jacobian
	#=if sameMarginalsMoment == 1
		offset = gravMoment + localGravityMoment * ((D - 1) * D + D * (D - 1) * (D - 2)) + GravityMomentFirstApproach
		CDF_Moments_Size = size(CDF_Moments, 2)
		@. jac_G[:, end-offset-CDF_Moments_Size+1:end-offset,:] = 0
	end=#


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

		pairewiseIndependenceMoment_jac!(Ū, jac_G, D, ηk, @view(Ind_Moments[1:W, :]), IndMomentOrder, IndCDF_Cells, ν_probas, offset, refIndex1, η_index, UoModel)
		#=
		T = Threads.nthreads()
		Threads.@threads for t = 1:T
			ix0 = round(Int, (t - 1) / T * W) + 1
			ix1 = round(Int, t / T * W)
			IndependenceMoment_jac!(Ū[ix0:ix1, :], @view(jac_G[ix0:ix1, :, :]), D, ηk, IndMomentOrder, IndCDF_Cells, ν_probas, offset, refIndex1, η_index)
		end
		=#
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
		GravityMomentFirstApproach_Jacobian!(jac_G, τ, ν, Aod, cHat, D, Ū, offset, UoModel, sameMarginalsMoment, Aod_offset, ν_offset)
	end

	if NormalizeMoments == 1
		for im ∈ 1:numMomentsSimple
			if  im ∉ moments_without_var
				@. jac_G[:, im, :] *=  1 ./ σ_Moments[im]
			end
		end
	end


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

