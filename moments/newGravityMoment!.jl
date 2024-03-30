function newGravityMoment!(G, PMM, τ, D, W, γ, U, GravityMomentFirstApproach)
    # constructs gravity moment, second approach [without additional parameters] (see theory note)

    deltaτ = doubleDiff(τ)

    meanτ = 0
    for o = 2:D

        meanτ += deltaτ[o, 1]

        for d = 3:D
            meanτ += deltaτ[o, d]
        end
    end
    meanτ /= (D - 1)^2



    U_ω = zeros(eltype(γ), size(τ))

    for ω = 1:W
        sumGrav = 0
        U_ω = U[ω, :]
        deltaU = doubleDiff(reshape(U_ω, (D, D)))

        for o = 2:D

            sumGrav += (deltaτ[o, 1] - meanτ) * deltaU[o, 1]

            for d = 3:D
                sumGrav += (deltaτ[o, d] - meanτ) * deltaU[o, d]
            end
        end
        sumGrav /= (D - 1)^2
        G[ω, end-GravityMomentFirstApproach] = 1000.0 * sumGrav - PMM[end-GravityMomentFirstApproach]
    end

end