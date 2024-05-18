function GravityMomentFirstApproach!(G, PMM, τ, ν, Aod, cHat, D, Ū, offset)

    # constructs the moment that ΔΔ E[ln U] = ΔΔ ln Aod + ΔΔ ln cHat + ΔΔ E[ln Ū] is mean independent of ΔΔ lnτ
    # this is NOT the strong gravitymoment in the notes 

    
    if sameMarginalsMoment == 0 # calculate the first moments of U
        #E[ln Ū] = ν[o,d]
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D
                @. G[:, offset+o1] =  log(Ū[:, o1]) .- ν[o,d] .- PMM[dInd+o1]
            end
        end
    end
    # if sameMarginalsMoment == 1, we know that ΔΔ E[ln Ū] = 0

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

    @. G[:, end] = sumGrav - PMM[end] # this condition is added last because it is a condition on parameters only, so it goes into the outerloop

end