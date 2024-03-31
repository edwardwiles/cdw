
function createUDerivatives!(U, prestep_output, params)

    @unpack W, D, useCDFforMarginalMatching, sameMarginalsMoment, momentOrder, σHat, θConstant = params 
    @unpack μHat, cHat = prestep_output 

    # we keep a copy of U without exponents or scaling by cHat to calculate E[U^α_k] later - only if we do not use the CDF methodology
    Ū = zeros(W, D * D, 1 + (1-useCDFforMarginalMatching)*sameMarginalsMoment * (momentOrder * 2 - 1)) # α_k \in [1,2, -1,....- momentOrder]

    Ū[:, :, 1] = U[:, :]

    for d = 1:D
        for o = 1:D
            o1 = o + (d - 1) * D
            @. U[:, o1] = U[:, o1] .* cHat[o, d]
        end
    end

    if θConstant == 1
        @. U[:] = U[:] .^ (-μHat) # if theta doesn't vary, then much faster to precalculate this matrix 
        # reduction in computation time due to non-integer exponent substantially dominates higher memory usage
    end

    Uσ = U .^ (1 - σHat) # precalculate 

    Σ_od = zeros(D, D)
    for d = 1:D
        for o = 1:D
            o1 = o + (d - 1) * D
            Σ_od[o, d] = sqrt(var(Ū[:, o1, 1]))
        end
    end

    return Ū, Uσ, Σ_od

end 