function hFunction!(G, UPow, Uσ, w, τ, σ, γ, Aod, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach,independenceMoment, μHat, UoModel)
	# main function to fill in the moment matrix G for the baseline moments

	D = size(τ, 1) # num origins (countries) -- always the full set
	Ddest = size(τ, 2) # num destinations -- Ddest==D unless row_idx excludes ROW (Part A, 2026-07-23)
	W = size(UPow, 1) # num draws (or goods)
	tuner = -100.0 # parameter for smooth mins
	l = D * Ddest

	gdp = (w .* L)

	# initialise important vectors
	pricesTemp = zeros(eltype(γ), D)
	pricesTempσ = copy(pricesTemp)

	pricesInd = copy(pricesTemp)
	denom = zeros(eltype(γ), Ddest)
	constCons = zeros(eltype(γ), D, Ddest)
	constConsσ = copy(constCons)
	wPow = zeros(eltype(γ), D)

	ξ = zeros(eltype(γ), D, Ddest)


	# identify the part of G matrix where we put the price index moments; we omit wage moments if autarky
	if counterType != 1
		cInd = D * Ddest + D - 1
	else
		cInd = D * Ddest
	end

	for d ∈ 1:D
		wPow[d] = w[d]^(1 - σ) # will need transformed wages many times, so do it once here (origin-indexed despite the loop variable name)
	end

	# construct objects that we will need but that never change with omega
	for d ∈ 1:Ddest
		denom[d] = γ[d]^σ * gdp[d]
		for o ∈ 1:D
			constCons[o, d] = w[o] * Aod[o, d] * τ[o, d]
			constConsσ[o, d] = wPow[o] * (Aod[o, d] * τ[o, d])^(1 - σ)
		end
	end

	if localGravityMoment == 1
		for d ∈ 1:Ddest
			for o ∈ 1:D
				d1 = d + (o - 1) * Ddest
				ξ[o, d] = P[d1] * denom[d]
			end
		end
	end

	# loop to fill in the G matrix
	@inbounds for ω ∈ 1:W # main loop over all goods

		for d ∈ 1:Ddest # loop through all (named) destination countries

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

			if localGravityMoment == 1
				max_price, max_idx = findmax(pricesTemp[:])
				offset = gravMoment + GravityMomentFirstApproach+independenceMoment
				localGravityMoment!(G, D, ω, pricesTemp, ξ[:, d], σ, μHat, d, max_price, offset)
				offset += D * (D - 1)
				localGravityCrossMoment!(G, D, ω, pricesTemp, ξ[:, d], σ, d, max_price, offset)
			end

			indSum = 0


			for o ∈ 1:D # using min prices, compute implied expenditure share and fill in G with implied minus data
				d1 = d + (o - 1) * Ddest
				pricesTemp[o] = pricesTempσ[o] * pricesInd[o]
				G[ω, d1] = pricesTemp[o] - P[d1]*denom[d]
				indSum += pricesTemp[o]
			end
			# baseline price-index moment (= sum of this d's trade-share moments). Redundant under
			# autarky's reduced moment set, so only written when the price-index columns exist.
			if counterType != 1
				G[ω, cInd+d] = indSum - denom[d]  # fill in part of G for price index moments (identifies MU parameter)
			end
		end

	end


end

function hFunctionCounter!(K, G, UPow, Uσ, w, τ, σ, γ, Aod, L, counterType, baseIndex, UoModel)
	# same as hFunction, except fills in counterfactual parts of G and fills in K

	D = size(τ, 1)      # num origins -- always the full set
	Ddest = size(τ, 2)   # num destinations -- Ddest==D unless row_idx excludes ROW (Part A, 2026-07-23)
	W = size(UPow, 1)
	tuner = -100
	l = D * Ddest

	gdp = (w .* L)

	pricesTemp = zeros(eltype(γ), D)
	pricesTempσ = copy(pricesTemp)
	pricesInd = copy(pricesTemp)
	pricesCounterVec = zeros(eltype(γ), D * Ddest)
	denom = zeros(eltype(γ), Ddest)
	constCons = zeros(eltype(γ), D, Ddest)
	constConsσ = copy(constCons)
	wPow = zeros(eltype(γ), D)

	if counterType != 1
		bInd = D * Ddest
		cInd = D * Ddest + D - 1
		dInd = D * Ddest + 2 * D - 1
	else
		cInd = D * Ddest
		dInd = D * Ddest + D
	end

	for d ∈ 1:D
		wPow[d] = w[d]^(1 - σ)
	end

	for d ∈ 1:Ddest
		denom[d] = γ[d]^σ * gdp[d]
		for o ∈ 1:D
			constCons[o, d] = w[o] * Aod[o, d] * τ[o, d]
			constConsσ[o, d] = wPow[o] * (Aod[o, d] * τ[o, d])^(1 - σ)
		end
	end

	if counterType != 1

		@inbounds for ω ∈ 1:W

			for d ∈ 1:D

				for o ∈ 1:D
					o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
					if UoModel == 1
						o1 = o
					end
					pricesTemp[o] = constCons[o, d] / UPow[ω, o1]
					pricesTempσ[o] = constConsσ[o, d] / Uσ[ω, o1]

				end

				#smoothMinIndNew!(pricesInd, pricesTemp, D, tuner)
				MinInd!(pricesInd, pricesTemp, D)

				indSum = 0

				for o ∈ 1:D
					o1 = o + (d - 1) * D
					pricesTemp[o] = pricesTempσ[o] * pricesInd[o]
					indSum += pricesTemp[o]
					pricesCounterVec[o1] = pricesTemp[o] * gdp[d] / denom[d]

					if o == d && o == baseIndex
						K[ω] = pricesCounterVec[o1] / gdp[d] # fill in K with own trade share 
					end

				end

				G[ω, dInd+d] = indSum - denom[d] # fill in G with counterfactual price index moments 

			end

			# counterfactual wage moments, omitting for first country 
			for o ∈ 2:D

				tempSum = 0

				for d ∈ 1:D
					d1 = o + (d - 1) * D
					tempSum += pricesCounterVec[d1]
				end

				G[ω, bInd+o-1] = tempSum - gdp[o] # fill in G with counterfactual wage moments 

			end

		end
	else
		# we need only baseIndex, so we do not need to identify the price index for other countries
		o1 = baseIndex + (baseIndex - 1) * D # uncomment to to U_{od} rather than U_o 
		if UoModel == 1
			o1 = baseIndex
		end

		# reduced autarky layout: single counterfactual price-index moment sits right after the
		# D*Ddest trade-share moments (baseline price-index moments dropped as redundant).
		# D*Ddest==D^2 unless row_idx excludes ROW as a destination (Part A, 2026-07-23).
		@. G[:, D*Ddest+1] = constConsσ[baseIndex, baseIndex] ./ Uσ[:, o1] .- denom[baseIndex]
		#=
		@inbounds for ω ∈ 1:W
			# we need only baseIndex, so we do not need to identify the price index for other countries
			for d ∈ 1:D
				if d == baseIndex
					o1 = d + (d - 1) * D # uncomment to to U_{od} rather than U_o 
					if UoModel == 1
						o1 = d
					end
					G[ω, dInd+d] = constConsσ[d, d] / Uσ[ω, o1] - denom[d] # all countries go into autarky. Price index is domestic price.
				end
			end
		end
		=#
	end

end
