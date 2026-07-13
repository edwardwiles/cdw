function newGravityMoment!(G, τ, D, W, γ, Aod, U, GravityMomentFirstApproach, UoModel)
	# constructs gravity moment, second approach [without additional parameters] (see theory note)

	deltaτ = doubleDiff(τ)

	meanτ = 0
	for o ∈ 2:D

		meanτ += deltaτ[o, 1]

		for d ∈ 3:D
			meanτ += deltaτ[o, d]
		end
	end
	meanτ /= (D - 1)^2

	if UoModel == 1

		# F-independent gravity/orthogonality moment as an origin+destination fixed-effects regression:
		# Σ_od (within lnτ)·(within lnA) = 0, with A the raw structural term (A_od = 1/AodPow, so
		# ln A = -ln AodPow; the sign is irrelevant for =0). The two-way within transform reproduces
		# the OLS-two-way-FE coefficient exactly (FWL); the old cell double-difference did not.
		Wτ = withinTransform(τ)
		WA = withinTransform(Aod)   # `Aod` here is AodPow (passed from moments!); = -within(ln A_od)
		sumGrav = zero(eltype(WA))
		for o ∈ 1:D
			for d ∈ 1:D
				sumGrav += Wτ[o, d] * WA[o, d]
			end
		end
		@. G[:, end-GravityMomentFirstApproach] = sumGrav
	else

		U_ω = zeros(eltype(γ), size(τ))

		for ω ∈ 1:W
			sumGrav = 0
			@. U_ω[:] = U[ω, :]
			deltaU = doubleDiff(reshape(U_ω, (D, D))[:, :] .* Aod[:, :])

			for o ∈ 2:D

				sumGrav += (deltaτ[o, 1] - meanτ) * deltaU[o, 1]

				for d ∈ 3:D
					sumGrav += (deltaτ[o, d] - meanτ) * deltaU[o, d]
				end
			end
			sumGrav /= (D - 1)^2

			G[ω, end-GravityMomentFirstApproach] = sumGrav
		end
	end

end
