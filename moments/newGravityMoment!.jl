function newGravityMoment!(G, τ, D, Ddest, W, γ, Aod, U, GravityMomentFirstApproach, UoModel)
	# constructs gravity moment, second approach [without additional parameters] (see theory note)
	# D = origin count (always full); Ddest = destination count (Ddest==D unless row_idx excludes
	# ROW as a destination, Part A 2026-07-23 -- τ/Aod are already D x Ddest by this point).

	if UoModel == 1

		# F-independent gravity/orthogonality moment as an origin+destination fixed-effects regression:
		# Σ_od (within lnτ)·(within lnA) = 0, with A the raw structural term (A_od = 1/AodPow, so
		# ln A = -ln AodPow; the sign is irrelevant for =0). The two-way within transform reproduces
		# the OLS-two-way-FE coefficient exactly (FWL); the old cell double-difference did not.
		Wτ = within_transform_rect(τ)
		WA = within_transform_rect(Aod)   # `Aod` here is AodPow (passed from moments!); = -within(ln A_od)
		sumGrav = zero(eltype(WA))
		for o ∈ 1:D
			for d ∈ 1:Ddest
				sumGrav += Wτ[o, d] * WA[o, d]
			end
		end
		@. G[:, end-GravityMomentFirstApproach] = sumGrav
	else
		D == Ddest || error("newGravityMoment!'s UoModel==0 branch (cell double-difference, pinned to destination reference column 2) is not rectangular-aware; only UoModel==1 is supported with Ddest != D.")
		deltaτ = doubleDiff(τ)

		meanτ = 0
		for o ∈ 2:D

			meanτ += deltaτ[o, 1]

			for d ∈ 3:D
				meanτ += deltaτ[o, d]
			end
		end
		meanτ /= (D - 1)^2


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
