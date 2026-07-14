function hFunction!(G, UPow, Uσ, w, τ, σ, Aod, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach,independenceMoment, μHat)
	# main function to fill in the moment matrix G for the baseline moments.
	# Baseline gamma is normalized to 1 for every destination (the A_od matrix absorbs all of the
	# scale freedom instead), so every former `gamma[d]^sigma*gdp[d]` term below is just `gdp[d]`.

	D = size(τ, 1) # num countries
	W = size(UPow, 1) # num draws (or goods)
	tuner = -100.0 # parameter for smooth mins
	l = D^2

	gdp = (w .* L)

	# T must be at least as wide as whatever Aod/UPow/Uσ carry (Float64, or Dual under ForwardDiff
	# differentiation of Aod_theta and/or mu) -- these scratch arrays hold results derived from them.
	T = promote_type(eltype(Aod), eltype(UPow), eltype(Uσ))

	# initialise important vectors
	pricesTemp = zeros(T, D)
	pricesTempσ = copy(pricesTemp)

	pricesInd = copy(pricesTemp)
	denom = copy(pricesTemp)
	constCons = zeros(T, D, D)
	constConsσ = copy(constCons)
	wPow = zeros(T, D)

	ξ = zeros(T, D, D)


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
		denom[d] = gdp[d]
		for o ∈ 1:D
			constCons[o, d] = w[o] * Aod[o, d] * τ[o, d]
			constConsσ[o, d] = wPow[o] * (Aod[o, d] * τ[o, d])^(1 - σ)
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
				pricesTemp[o] = constCons[o, d] / UPow[ω, o]
				pricesTempσ[o] = constConsσ[o, d] / Uσ[ω, o]
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
				d1 = d + (o - 1) * D
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

function hFunctionCounter!(K, G, UPow, Uσ, w, τ, σ, γ, Aod, L, counterType, baseIndex)
	# same as hFunction, except fills in counterfactual parts of G and fills in K

	D = size(τ, 1)
	W = size(UPow, 1)
	tuner = -100
	l = D^2

	gdp = (w .* L)

	pricesTemp = zeros(eltype(γ), D)
	pricesTempσ = copy(pricesTemp)
	pricesInd = copy(pricesTemp)
	pricesCounterVec = zeros(eltype(γ), D^2)
	denom = copy(pricesTemp)
	constCons = zeros(eltype(γ), D, D)
	constConsσ = copy(constCons)
	wPow = zeros(eltype(γ), D)

	if counterType != 1
		bInd = D^2
		cInd = D^2 + D - 1
		dInd = D^2 + 2 * D - 1
	else
		cInd = D^2
		dInd = D^2 + D
	end

	for d ∈ 1:D
		wPow[d] = w[d]^(1 - σ)
	end

	for d ∈ 1:D
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
					pricesTemp[o] = constCons[o, d] / UPow[ω, o]
					pricesTempσ[o] = constConsσ[o, d] / Uσ[ω, o]

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
		o1 = baseIndex

		# reduced autarky layout: single counterfactual price-index moment sits right after the
		# D^2 trade-share moments (baseline price-index moments dropped as redundant).
		@. G[:, D^2+1] = constConsσ[baseIndex, baseIndex] ./ Uσ[:, o1] .- denom[baseIndex]
	end

end
