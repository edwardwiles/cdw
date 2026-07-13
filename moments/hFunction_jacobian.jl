# EXP: smoothing width of the SmoothDirac term used to differentiate through the
# argmin-selection indicator in the analytic Jacobian. Overridable via ENV for the
# A3 β-sweep; defaults to the committed value 0.01. (β→∞ ⇒ Dirac term vanishes,
# matching the ForwardDiff a.e. derivative that omits the indicator's boundary term.)
const EXP_BETA = parse(Float64, get(ENV, "EXP_BETA", "0.01"))

function hFunction_jacobian_copy_only!(jac_G, UPow, Uσ, w, τ, σ, γ, Aod, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, μ, UoModel, OuterScaling, Aod_offset, θConstant, ∂γ∂γθ, ∂γ∂μ, ∂γ∂Aθ)
	D = size(τ, 1) # num countries 
	W = size(UPow, 1) # num draws (or goods)

	
	gdp = (w .* L)
	lambda = reshape(P, (D, D))'
	for d in 1:D
		#price index moment wrt gamma
		∂m∂γ = -σ*gdp[d]*γ[d]^(σ-1)
		@. jac_G[:, D^2+d, 2+d] += ∂m∂γ*∂γ∂γθ[d]
		@. jac_G[:, D^2+d, 1] += (θConstant != 1) ? ∂m∂γ*∂γ∂μ[d] : 0
		
		for o in 1:D
			d1 = d + (o - 1) * D

			@. jac_G[:, d1, 2+d] += ∂m∂γ*lambda[o,d]*∂γ∂γθ[d]
			@. jac_G[:, d1, 1] += (θConstant != 1) ? ∂m∂γ*lambda[o,d]*∂γ∂μ[d] : 0

			if OuterScaling == 1
				jac_index_o = Aod_offset + o + (d - 1) * D
				@. jac_G[:, D^2+d, jac_index_o] +=∂m∂γ*∂γ∂Aθ[o,d]
				for c in 1:D
					jac_index_c = Aod_offset + c + (d - 1) * D
					@. jac_G[:, d1, jac_index_c] += ∂m∂γ*lambda[o,d]*∂γ∂Aθ[c,d]
				end
			end
		end
	end	
end

function hFunction_jacobian_calculation!(jac_G, UPow, Uσ, w, τ, σ, γ, Aod, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, μ, UoModel, OuterScaling, Aod_offset, θConstant, ∂A∂Aθ, ∂A∂μ, CHat)
	D = size(τ, 1) # num countries
	W = size(UPow, 1) # num draws (or goods)
	β = EXP_BETA   # EXP: was hard-coded 0.01; now ENV-overridable for the A3 sweep

	gdp = (w .* L)


	if OuterScaling == 1 || θConstant != 1
		# initialise important vectors 
		pricesTemp = zeros(eltype(γ), D)
		pricesTempσ = copy(pricesTemp)

		pricesInd = copy(pricesTemp)
		denom = copy(pricesTemp)
		constCons = zeros(eltype(γ), D, D)
		constConsσ = copy(constCons)
		wPow = zeros(eltype(γ), D)

		ξ = zeros(eltype(γ), D, D)


		# identify the part of G matrix where we put the price index moments; we omit wage moments if autarky 
		if counterType != 1
			cInd = D^2 + D - 1
		else
			cInd = D^2
		end

		for d ∈ 1:D
			wPow[d] = w[d]^(1 - σ) # will need transformed wages many times, so do it once here 
		end

		# construct objects that we will need but that never change with omega
		for d ∈ 1:D
			denom[d] = γ[d]^σ * gdp[d]
			for o ∈ 1:D
				constCons[o, d] = w[o] * AodPow[o, d] * τ[o, d]
				constConsσ[o, d] = wPow[o] * (AodPow[o, d] * τ[o, d])^(1 - σ)
			end
		end

		if localGravityMoment == 1
			for d ∈ 1:D
				for o ∈ 1:D
					d1 = d + (o - 1) * D
					ξ[o, d] = P[d1] * denom[d]
				end
			end
		end

		# loop to fill in the G matrix
		@inbounds for ω ∈ 1:W # main loop over all goods 

			for d ∈ 1:D # loop through all destination countries 

				for o ∈ 1:D # for each origin, construct p_{od}
					o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
					if UoModel == 1
						o1 = o
					end

					pricesTemp[o] = constCons[o, d] / UPow[ω, o1]
					pricesTempσ[o] = constConsσ[o, d] / Uσ[ω, o1]
				end

				#smoothMinIndNew!(pricesInd, pricesTemp, D, tuner) # construct 1\{p_{od}=min_o(p_{od})\}
				minPrice = MinInd2!(pricesInd, pricesTemp, D)

				indSum = 0
				# pricesTemp[o] = pricesTempσ[o] * pricesInd[o]

				for o ∈ 1:D
					if pricesInd[o] == 1 
						for c in 1:D
							# if o or c are close to the min, such that variation in Acd could change min 
							#if (abs(pricesTemp[o]-minPrice) < 2*β) || (abs(pricesTemp[c]- minPrice)< 2*β && abs(pricesTemp[c] - pricesTemp[o]) < 2*β)
							jac_index = Aod_offset + c + (d - 1) * D
							d1 = d + (o - 1) * D
							Min_X_oc = minimum(pricesTemp[Not([o, c])])
							#Min_X_o = o == c ? Min_X_oc : minimum(pricesTemp[Not([o])])
							∂ξ∂Acd = 0
							if o == c
								∂ξ∂Acd = (1 - σ) * pricesInd[o] * pricesTempσ[o] / pricesTemp[o] - pricesTempσ[o] * SmoothDirac(β, pricesTemp[o] - Min_X_oc)# use one sided dirac as I am derivating only on the right
							else
								∂ξ∂Acd = pricesTempσ[o] * indicative(Min_X_oc - pricesTemp[o]) * SmoothDirac(β, pricesTemp[o] - pricesTemp[c])
							end
							#∂P∂A
							∂ξ∂Acd *= -μ * pricesTemp[c] / (Aod[c, d]/CHat[c,d])
							
							if OuterScaling == 1
								jac_G[ω, d1, jac_index] += ∂ξ∂Acd*∂A∂Aθ[c,d]
								jac_G[ω, cInd+d, jac_index] += ∂ξ∂Acd*∂A∂Aθ[c,d]
							end

							jac_G[ω, d1, 1] += ∂ξ∂Acd*∂A∂μ[c,d]
							jac_G[ω, cInd+d, 1] += ∂ξ∂Acd*∂A∂μ[c,d]
							#end
						end
					end
					if pricesInd[o] == 1 && θConstant != 1
						o1 = o + (d - 1) * D 
						d1 = d + (o - 1) * D
						if UoModel == 1
							o1 = o
						end
						∂ξ∂μ = (-1 / μ) * (log(Uσ[ω, o1]) - (1 - σ) * log(AodPow[o, d])) * pricesTempσ[o]
						jac_G[ω, d1, 1] += ∂ξ∂μ
						jac_G[ω, cInd+d, 1] += ∂ξ∂μ
					end
				end
			end

		end

	end

end

function hFunctionCounter_jacobian!(jac_K, jac_G, UPow, Uσ, w, τ, σ, γ, Aod, AodPow, L, counterType, baseIndex, μ,μHat, UoModel, OuterScaling, Aod_offset, θConstant,∂γprime∂γprimeθ, ∂γprime∂γθ, ∂γprime∂Aθ, ∂γprime∂μ,∂A∂Aθ, ∂A∂μ, CHat)
	# same as hFunction, except fills in counterfactual parts of G and fills in K 

	D = size(τ, 1)
	W = size(UPow, 1)
	tuner = -100
	l = D^2

	if counterType != 1
		bInd = D^2
		cInd = D^2 + D - 1
		dInd = D^2 + 2 * D - 1
	else
		cInd = D^2
		dInd = D^2 + D
	end

	gdp = (w .* L)

	pricesTemp = zeros(eltype(γ), D)
	pricesTempσ = copy(pricesTemp)
	pricesInd = copy(pricesTemp)
	pricesCounterVec = zeros(eltype(γ), D^2)
	denom = copy(pricesTemp)
	constCons = zeros(eltype(γ), D, D)
	constConsσ = copy(constCons)
	wPow = zeros(eltype(γ), D)


	for d ∈ 1:D
		wPow[d] = w[d]^(1 - σ)
	end

	for d ∈ 1:D
		denom[d] = γ[d]^σ * gdp[d]
		for o ∈ 1:D
			constCons[o, d] = w[o] * AodPow[o, d] * τ[o, d]
			constConsσ[o, d] = wPow[o] * (AodPow[o, d] * τ[o, d])^(1 - σ)
		end
	end


	for d in 1:D
		#counterfactual price index
		if d == baseIndex
			o1 = d + (d - 1) * D # uncomment to to U_{od} rather than U_o 
			if UoModel == 1
				o1 = d
			end

			∂m∂γPrime = -σ * γ[baseIndex]^(σ - 1) * L[d] * w[d] #


			@. jac_G[:, dInd+d, 2+D+1] += ∂m∂γPrime*∂γprime∂γprimeθ[d] #price index moment
			@. jac_G[:, dInd+d, 2+d] += ∂m∂γPrime*∂γprime∂γθ[d]
			if OuterScaling == 1
				jac_index_Add = Aod_offset + d + (d - 1) * D
				@. jac_G[:, dInd+d, jac_index_Add] +=((-μ * (1 - σ) * constConsσ[d, d] / (Aod[d, d]/CHat[d,d])) ./ Uσ[1:W, o1])*∂A∂Aθ[d,d]
				
				@. jac_G[:, dInd+d, jac_index_Add] += ∂m∂γPrime*∂γprime∂Aθ[d]

			end

			if θConstant != 1
				@. jac_G[:, dInd+d, 1] += (-1/ μ) .* (log.(Uσ[1:W, o1]) .- (1-σ)*log(AodPow[d,d])) .* constConsσ[d, d] ./ Uσ[1:W, o1]
				@. jac_G[:, dInd+d, 1] +=((-μ * (1 - σ) * constConsσ[d, d] / (Aod[d, d]/CHat[d,d])) ./ Uσ[1:W, o1])*∂A∂μ[d,d]
				@. jac_G[:, dInd+d, 1] += ∂m∂γPrime*∂γprime∂μ[d]
			end

		end
	end

end

