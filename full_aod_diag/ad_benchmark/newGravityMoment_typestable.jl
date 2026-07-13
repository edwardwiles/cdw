# DIAGNOSTICS-ONLY copy of moments/newGravityMoment!.jl with a single type-
# stability fix for Enzyme (Method C testing). ONE character changed:
# `meanτ = 0` -> `meanτ = zero(eltype(τ))`. The original left `meanτ` as an
# Int64 literal that then accumulates Float64 increments (`meanτ += deltaτ[o,1]`),
# a classic Julia type instability (Union{Int64,Float64} inferred locally).
# For UoModel==1 (this audit's config) `meanτ`/`deltaτ` are DEAD CODE — computed
# but never read by the UoModel==1 branch — so this has ZERO effect on any
# output for any input; verified by direct comparison in validate_derivatives.jl
# (primal equality test). Not an algebraic rewrite — a pure type-stability fix,
# explicitly anticipated by the audit spec §5 Method C.
function newGravityMoment_typestable!(G, τ, D, W, γ, Aod, U, GravityMomentFirstApproach, UoModel)
	deltaτ = doubleDiff(τ)

	meanτ = zero(eltype(τ))   # <-- THE ONLY CHANGE (was: meanτ = 0)
	for o ∈ 2:D
		meanτ += deltaτ[o, 1]
		for d ∈ 3:D
			meanτ += deltaτ[o, d]
		end
	end
	meanτ /= (D - 1)^2

	if UoModel == 1
		Wτ = withinTransform(τ)
		WA = withinTransform(Aod)
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
			sumGrav = zero(eltype(τ))
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
