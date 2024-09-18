function GravityMomentFirstApproach!(G, τ, ν, Aod, cHat, D, Ū, offset, UoModel, sameMarginalsMoment)

	# constructs the moment that ΔΔ E[ln U] = ΔΔ ln Aod + ΔΔ ln cHat + ΔΔ E[ln Ū] is mean independent of ΔΔ lnτ


	if sameMarginalsMoment == 0 && UoModel == 0# calculate the first moments of U
		#E[ln Ū] = ν[o,d]
		for o ∈ 1:D
			for d ∈ 1:D
				o1 = o + (d - 1) * D
				@. G[:, offset+o1] = log(Ū[:, o1]) .- ν[o, d]
			end
		end
	end
	# if sameMarginalsMoment == 1 or UoModel ==1, we know that ΔΔ E[ln Ū] = 0

	ΔΔν = doubleDiffLinear(ν)
	ΔΔlncHat = doubleDiff(cHat)
	ΔΔlnAod = doubleDiff(Aod)
	deltaτ = doubleDiff(τ)

	meanτ = mean(deltaτ)
	sumGrav = mean((deltaτ .- meanτ) .* (ΔΔν .+ ΔΔlncHat .- ΔΔlnAod))

	@. G[:, end] = sumGrav# this condition is added last because it is a condition on parameters only, so it goes into the outerloop

end

function GravityMomentFirstApproach_Jacobian!(Jac_G, τ, ν, Aod, cHat, D, Ū, offset, UoModel, sameMarginalsMoment, Aod_offset, ν_offset, OuterScaling, θConstant,CHat, ∂A∂Aθ, ∂A∂μ)

	# constructs the moment that ΔΔ E[ln U] = ΔΔ ln Aod + ΔΔ ln cHat + ΔΔ E[ln Ū] is mean independent of ΔΔ lnτ


	if sameMarginalsMoment == 0 && UoModel == 0# calculate the first moments of U
		#E[ln Ū] = ν[o,d]
		for o ∈ 1:D
			for d ∈ 1:D
				o1 = o + (d - 1) * D
				@. Jac_G[:, offset+o1, ν_offset+o1] = -1
			end
		end
	end
	# if sameMarginalsMoment == 1 or UoModel ==1, we know that ΔΔ E[ln Ū] = 0

	∂Acd∂Acdθ = ones(D,D)
	∂γd∂Acd =  ones(D,D)
	∂Acd∂μ = zeros(D,D)
	for d in 1:D
		for o in 1:D
			o1 = d + (o - 1) * D
			∂γd∂Acd[o,d] = (P[o1]*(σ-1)*μ/σ)*γ[d]/Aod[o,d]
			∂Acd∂Acdθ[o,d] = (w[o]*τ[o,d]/(w[1]*τ[1,d]))^(1/μ - 1/μHat)
			∂Acd∂μ[o,d] = (θConstant != 1) ? (-1/μ^2)*log(w[o]*τ[o,d]/(w[1]*τ[1,d]))*Aod[o,d] : 0
		end
	end



	deltaτ = doubleDiff(τ)
	meanτ = mean(deltaτ)

	if OuterScaling == 1 
	ΔΔlnAod_grad = doubleDiff_grad(Aod)
	for o ∈ 1:D
		for d ∈ 1:D
			@. Jac_G[:, end, Aod_offset+o+D*(d-1)] -= (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o, d, o, d]*CHat[o,d]*∂A∂Aθ[o,d]

			@. Jac_G[:, end, Aod_offset+1+D*(2-1)] -= (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o, d, 1, 2]*CHat[1,2]*∂A∂Aθ[1,2]
			@. Jac_G[:, end, Aod_offset+o+D*(2-1)] -= (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o, d, o, 2]*CHat[o,2]*∂A∂Aθ[o,2]
			@. Jac_G[:, end, Aod_offset+1+D*(d-1)] -= (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o, d, 1, d]*CHat[1,d]*∂A∂Aθ[1,d]

			if θConstant!= 1
			@. Jac_G[:, end, 1] -= (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o, d, o, d]*CHat[o,d]*∂A∂μ[o,d]
			@. Jac_G[:, end, 1] -= (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o, d, 1, 2]*CHat[1,2]*∂A∂μ[1,2]
			@. Jac_G[:, end, 1] -= (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o, d, o, 2]*CHat[o,2]*∂A∂μ[o,2]
			@. Jac_G[:, end, 1] -= (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o, d, 1, d]*CHat[1,d]*∂A∂μ[1,d]
			end

		end
	end
	end
	

	if sameMarginalsMoment == 0 && UoModel == 0
		ΔΔν_grad = doubleDiffLinear_grad(ν)

		for o ∈ 1:D
			for d ∈ 1:D
				@. Jac_G[:, end, ν_offset+o+D*(d-1)] += (deltaτ[o, d] - meanτ) * ΔΔν_grad[o, d, o, d]

				@. Jac_G[:, end, ν_offset+1+D*(2-1)] += (deltaτ[o, d] - meanτ) * ΔΔν_grad[o, d, 1, 2]
				@. Jac_G[:, end, ν_offset+o+D*(2-1)] += (deltaτ[o, d] - meanτ) * ΔΔν_grad[o, d, o, 2]
				@. Jac_G[:, end, ν_offset+1+D*(d-1)] += (deltaτ[o, d] - meanτ) * ΔΔν_grad[o, d, 1, d]			
			end
		end
	end

	@. Jac_G[:, end, :] /= D^2

end
