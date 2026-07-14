function newGravityMoment!(G, τ, D, Aod, GravityMomentFirstApproach)
	# constructs gravity moment (see theory note)

	# F-independent gravity/orthogonality moment as an origin+destination fixed-effects regression:
	# Σ_od (within lnτ)·(within lnA) = 0, with A the raw structural term (A_od = 1/AodPow, so
	# ln A = -ln AodPow; the sign is irrelevant for =0). The two-way within transform reproduces
	# the OLS-two-way-FE coefficient exactly (FWL).
	#
	# FWL also means the within-transform on the A side is REDUNDANT: Wτ is already orthogonal to
	# the row/column means (by construction of withinTransform), so Σ Wτ·within(x) = Σ Wτ·x for
	# ANY x — verified numerically (scratch/check_fwl.jl, agreement to ~1e-15 across D=3..10, and
	# again with x = ln(A^(-μ)) directly). So we dot Wτ against raw ln(Aod) instead of
	# withinTransform(Aod): identical value, one fewer D×D log+demean pass, and a shorter autodiff
	# chain (no need to differentiate the demeaning operation w.r.t. Aod_θ).
	Wτ = withinTransform(τ)
	lnAod = log.(Aod)   # `Aod` here is AodPow (passed from moments!); ln(AodPow) = -ln(A_od)
	sumGrav = zero(eltype(lnAod))
	for o ∈ 1:D
		for d ∈ 1:D
			sumGrav += Wτ[o, d] * lnAod[o, d]
		end
	end
	@. G[:, end-GravityMomentFirstApproach] = sumGrav

end
