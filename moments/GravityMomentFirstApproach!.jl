function GravityMomentFirstApproach!(G, τ, ν, Aod, cHat, D, Ū, offset, UoModel, sameMarginalsMoment)

    # constructs the moment that ΔΔ E[ln U] = ΔΔ ln Aod + ΔΔ ln cHat + ΔΔ E[ln Ū] is mean independent of ΔΔ lnτ

    
    if sameMarginalsMoment == 0  && UoModel == 0# calculate the first moments of U
        #E[ln Ū] = ν[o,d]
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D
                @. G[:, offset+o1] =  log(Ū[:, o1]) .- ν[o,d] 
            end
        end
    end
    # if sameMarginalsMoment == 1 or UoModel ==1, we know that ΔΔ E[ln Ū] = 0

    ΔΔν = doubleDiffLinear(ν)
    ΔΔlncHat = doubleDiff(cHat)
    ΔΔlnAod = doubleDiff(Aod)
    deltaτ = doubleDiff(τ)


    meanτ = 0
    for o = 2:D
        meanτ += deltaτ[o, 1]
        for d = 3:D
            meanτ += deltaτ[o, d]
        end
    end
    meanτ /= (D - 1)^2

    sumGrav = 0
    for o = 2:D
        sumGrav += (deltaτ[o, 1] - meanτ) * (ΔΔν[o, 1]+ΔΔlncHat[o,1]+ΔΔlnAod[o,1])
        for d = 3:D
            sumGrav += (deltaτ[o, d] - meanτ) * (ΔΔν[o, d]+ΔΔlncHat[o,d]+ΔΔlnAod[o,d])
        end
    end
    sumGrav /= (D - 1)^2

    @. G[:, end] = sumGrav# this condition is added last because it is a condition on parameters only, so it goes into the outerloop

end

function GravityMomentFirstApproach_Jacobian!(Jack_G, τ, ν, Aod, cHat, D, Ū, offset, UoModel, sameMarginalsMoment , Aod_offset, ν_offset)

    # constructs the moment that ΔΔ E[ln U] = ΔΔ ln Aod + ΔΔ ln cHat + ΔΔ E[ln Ū] is mean independent of ΔΔ lnτ

    
    if sameMarginalsMoment == 0  && UoModel == 0# calculate the first moments of U
        #E[ln Ū] = ν[o,d]
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D
                @. jac_G[:, offset+o1,ν_offset+o1] =  -1 
            end
        end
    end
    # if sameMarginalsMoment == 1 or UoModel ==1, we know that ΔΔ E[ln Ū] = 0


    deltaτ = doubleDiff(τ)
    meanτ = 0
    for o = 2:D
        meanτ += deltaτ[o, 1]
        for d = 3:D
            meanτ += deltaτ[o, d]
        end
    end
    meanτ /= (D - 1)^2

    ΔΔlnAod_grad = doubleDiff_grad(Aod)
    for o = 2:D
        @. Jack_G[:, end, Aod_offset +o+D*(1-1)] += (deltaτ[o, 1] - meanτ) * ΔΔlnAod_grad[o,1,o,1]

        @. Jack_G[:, end, Aod_offset +1+D*(2-1)] += (deltaτ[o, 1] - meanτ) * ΔΔlnAod_grad[o,1,1,2]
        @. Jack_G[:, end, Aod_offset +o+D*(2-1)] += (deltaτ[o, 1] - meanτ) * ΔΔlnAod_grad[o,1,o,2]
        @. Jack_G[:, end, Aod_offset +1+D*(d-1)] += (deltaτ[o, 1] - meanτ) * ΔΔlnAod_grad[o,1,1,d]

        for d = 3:D
            @. Jack_G[:, end, Aod_offset+ o+D*(d-1)] += (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o,d, o,d]

            @. Jack_G[:, end, Aod_offset +1+D*(2-1)] += (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o,d,1,2]
            @. Jack_G[:, end, Aod_offset +o+D*(2-1)] += (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o,d,o,2]
            @. Jack_G[:, end, Aod_offset +1+D*(d-1)] += (deltaτ[o, d] - meanτ) * ΔΔlnAod_grad[o,d,1,d]
    
        end
    end

    if sameMarginalsMoment == 0  && UoModel == 0
        ΔΔν_grad = doubleDiffLinear_grad(ν)
        for o = 2:D
            @. Jack_G[:, end, ν_offset +o+D*(1-1)] += (deltaτ[o, 1] - meanτ) * ΔΔν_grad[o,1,o,1]
    
            @. Jack_G[:, end, ν_offset +1+D*(2-1)] += (deltaτ[o, 1] - meanτ) * ΔΔν_grad[o,1,1,2]
            @. Jack_G[:, end, ν_offset +o+D*(2-1)] += (deltaτ[o, 1] - meanτ) * ΔΔν_grad[o,1,o,2]
            @. Jack_G[:, end, ν_offset +1+D*(d-1)] += (deltaτ[o, 1] - meanτ) * ΔΔν_grad[o,1,1,d]
    
            for d = 3:D
                @. Jack_G[:, end, ν_offset+ o+D*(d-1)] += (deltaτ[o, d] - meanτ) * ΔΔν_grad[o,d, o,d]
    
                @. Jack_G[:, end, ν_offset +1+D*(2-1)] += (deltaτ[o, d] - meanτ) * ΔΔν_grad[o,d,1,2]
                @. Jack_G[:, end, ν_offset +o+D*(2-1)] += (deltaτ[o, d] - meanτ) * ΔΔν_grad[o,d,o,2]
                @. Jack_G[:, end, ν_offset +1+D*(d-1)] += (deltaτ[o, d] - meanτ) * ΔΔν_grad[o,d,1,d]
        
            end
        end
    end

    @. Jack_G[:, end,:] /= (D - 1)^2

end