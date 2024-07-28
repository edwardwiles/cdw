function hFunction_jacobian_copy_only!(jac_G, UPow, Uσ, w, τ, σ, γ, Aod, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, μ, UoModel, OuterScaling, Aod_offset)
	D = size(τ, 1) # num countries 
	W = size(UPow, 1) # num draws (or goods)


	gdp = (w .* L)

	for d in 1:D
		∂PriceIndex∂γ = σ * γ[d]^(σ - 1) * L[d] * w[d]
		for o ∈ 1:D
			d1 = d + (o - 1) * D
			@. jac_G[:, d1, 2+d] = -P[d1] * ∂PriceIndex∂γ #trade share moment
		end
		@. jac_G[:, D^2+d, 2+d] = -∂PriceIndex∂γ #price index moment
	end
end
function hFunction_jacobian_calculation!(jac_G, UPow, Uσ, w, τ, σ, γ, Aod, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, μHat, μ, UoModel, OuterScaling, Aod_offset, θConstant)
	D = size(τ, 1) # num countries 
	W = size(UPow, 1) # num draws (or goods)
	β = 0.01

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
				MinInd!(pricesInd, pricesTemp, D)

				indSum = 0
				# pricesTemp[o] = pricesTempσ[o] * pricesInd[o]

				for o ∈ 1:D
					if pricesInd[o] == 1 && OuterScaling == 1
						for c in 1:D
							jac_index = Aod_offset + c + (d - 1) * D
							d1 = d + (o - 1) * D
							Min_X_oc = minimum(pricesTemp[Not([o, c])])
							Min_X_o = o != c ? Min_X_oc : minimum(pricesTemp[Not([o])])
							∂ξ∂Acd = 0
							if o == c
								∂ξ∂Acd = (1 - σ) * pricesInd[o] * pricesTempσ[o] / pricesTemp[o] - pricesInd[o] * pricesTempσ[o] * SmoothDirac(β, pricesTemp[o] - Min_X_o)
							else
								∂ξ∂Acd = pricesTempσ[o] * indicative(Min_X_oc - pricesTemp[o]) * SmoothDirac(β, pricesTemp[o] - pricesTemp[c])
							end
							#∂P∂A
							∂ξ∂Acd *= -μ * pricesTemp[c] / Aod[c, d]
							jac_G[ω, d1, jac_index] = ∂ξ∂Acd
							jac_G[ω, cInd+d, jac_index] = ∂ξ∂Acd
						end
					end
					if pricesInd[o] == 1 && θConstant != 1
						d1 = d + (o - 1) * D
						∂ξ∂μ = ln(pricesTempσ[o]) * pricesTempσ[o] / μ
						jac_G[ω, d1, 1] = ∂ξ∂μ
						jac_G[ω, cInd+d, 1] = ∂ξ∂μ
					end
				end
			end

		end

	end

end

function hFunctionCounter_jacobian!(jac_K, jac_G, UPow, Uσ, w, τ, σ, γ, Aod, AodPow, L, counterType, baseIndex, μ, UoModel, OuterScaling, Aod_offset, θConstant)
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
			if OuterScaling == 1
				jac_index_Add = Aod_offset + d + (d - 1) * D
				# -(1-sigma)*mu*p_od^(1-sigma)/A
				@. jac_G[:, cInd+d, jac_index_Add] = (-μ * (1 - σ) * constConsσ[d, d] / Aod[d, d]) ./ Uσ[1:W, o1]
			elseif θConstant != 1
				@. jac_G[:, cInd+d, 1] = (1/μ) .* log.((constConsσ[d, d] / Aod[d, d])./ Uσ[1:W, o1]) .* (constConsσ[d, d] / Aod[d, d]) ./ Uσ[1:W, o1]
			end
			∂PriceIndexPrime∂γPrime = σ * γ[baseIndex]^(σ - 1) * L[d] * w[d] #
			@. jac_G[:, cInd+d, 2+D+1] = -∂PriceIndexPrime∂γPrime #price index moment
		end
	end

end
