function gravityMoment!(G, τ, c, D, W, γ, GravityMomentFirstApproach)
    # constructs gravity moment, if using (see theory note)

    #deltaτ = zeros(eltype(γ),size(τ))
    #deltaτ[:] .= doubleDiff(τ)
    #deltaC = zeros(eltype(γ),D,D)
    #deltaC[:] .= doubleDiff(reshape(c,(D,D)))

    deltaτ = doubleDiff(τ)
    deltaC = doubleDiff(reshape(c, (D, D)))

    sumGrav = 0

    for o = 2:D

        sumGrav += deltaτ[o, 1] * deltaC[o, 1]

        for d = 3:D
            sumGrav += deltaτ[o, d] * deltaC[o, d]
        end

    end

    sumGrav /= (D - 1)^2

    for i = 1:W
        G[i, end-GravityMomentFirstApproach] = sumGrav
    end
end

