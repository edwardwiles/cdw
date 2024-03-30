function GravityMomentFirstApproach!(G, PMM, τ, ν, cHat, D, Ū, counterType, sameMarginalsMoment)

    # constructs the moment that ΔΔ E[ln U] = ΔΔ ln cHat + ΔΔ E[ln Ū] is mean independent of ΔΔ lnτ
    # this is NOT the strong gravitymoment in the notes 

    
    if sameMarginalsMoment == 0 # calculate if we do not know the value of the first moment
        refIndex = 1
        refIndex1 = refIndex + (refIndex - 1) * D

        dInd = counterType == 1 ? D^2 + 2 * D : D^2 + (D - 1) + 2 * D
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D
                @. G[:, dInd+o1] =  Ū[:, o1, 1] ./ ν[o, d] .- Ū[:, refIndex1, 1] ./ ν[refIndex, refIndex].- PMM[dInd+o1]
            end
        end
    end
    # if sameMarginalsMoment == 1, we know that E[Ū] = constant * ν_od

    ΔΔlnν = doubleDiff(ν)
    ΔΔlncHat = doubleDiff(cHat)
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
        sumGrav += (deltaτ[o, 1] - meanτ) * (ΔΔlnν[o, 1]+ΔΔlncHat[o,1])
        for d = 3:D
            sumGrav += (deltaτ[o, d] - meanτ) * (ΔΔlnν[o, d]+ΔΔlncHat[o,d])
        end
    end
    sumGrav /= (D - 1)^2

    @. G[:, end] = sumGrav - PMM[end] # this condition is added last because it is a condition on parameters only, so it goes into the outerloop

end