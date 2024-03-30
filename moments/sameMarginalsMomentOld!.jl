function sameMarginalsMomentold!(Ū, G, PMM, D, W, X, μ_σ, ν, momentOrder, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, baseIndex)
    # old implementation, not used
    # assures the marginals are equal amongst themselves

    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (3 + D^2)

    Γ = gamma(1 + μ_σ)
    U_pow = zeros(momentOrder + 1)
    factorial_k = zeros(momentOrder)
    for k = 2:momentOrder
        factorial_k[k] = factorial(k)
    end
    #ref_idx = baseIndex + (baseIndex - 1) * D
    refIndex = 1
    refIndex1 = refIndex + (refIndex - 1) * D
    for ω = 1:W
        #=
                G[ω, end-offset-1] =  log((Ū[ω, refIndex1]/ν[baseIndex, baseIndex]))- η[1]
                G[ω, end-offset-1-momentOrder] =  ((Ū[ω, o1]/ν[o, d])^μ_σ)/ Γ - η[momentOrder+1]

                for k =2:momentOrder
                    # E[exp] = k!, so we normalize by it. 
                    G[ω, end-offset-k] =  ((Ū[ω, refIndex1]/ν[baseIndex, baseIndex])^k)/factorial(k) - η[k]
                end
        =#

        #F_1_X = Ū[ω, refIndex1]> X[ω] ? 1 : 0
        U_pow[1] = (Ū[ω, refIndex1] / ν[refIndex, refIndex])^μ_σ
        for k = 1:momentOrder
            U_pow[k+1] = (Ū[ω, refIndex1] / ν[refIndex, refIndex])^k
        end
        for o = 1:D
            for d = 1:D
                #if d == baseIndex # clculate only for the baseIndex destination. This is just to get the gains from trade working
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                #F_od_X = Ū[ω, o1]/ν[o, d] > X[ω] ? 1 : 0
                #G[ω, end-offset-o1] =  F_1_X - F_od_X # E[F_1,1(X)] = E[F_o,d(X)] # this is probably not a good moment condition, The X will find its way into F through the RN, so it is not independent of U_od under F anymore.
                #G[ω, end-offset-o1] =  ((Ū[ω, o1]/ν[o, d])^μ_σ - U_pow[1])/ Γ # negative power moment, minimal restriction on F as this power needs to be defined for the counterfactual to be defined.
                G[ω, end-offset-o1] = log((Ū[ω, o1] / ν[o, d]) / (Ū[ω, refIndex1] / ν[refIndex, refIndex])) # negative power moment, minimal restriction on F as this power needs to be defined for the counterfactual to be defined
                G[ω, end-offset-(1+momentOrder)*D*D-o1] = ((Ū[ω, o1] / ν[o, d])^μ_σ - U_pow[1]) / Γ # negative power moment, minimal restriction on F as this power needs to be defined for the counterfactual to be defined.

                G[ω, end-offset-1*D*D-o1] = Ū[ω, o1] - ν[o, d]
                control_variable = (Ū[ω, o1] / ν[o, d]) - U_pow[2]
                for k = 2:momentOrder
                    this_moment_variable = ((Ū[ω, o1] / ν[o, d])^k - U_pow[k+1]) / factorial_k[k]
                    # E[exp] = k!, so we normalize by it. 
                    G[ω, end-offset-k*D*D-o1] = this_moment_variable - control_variable
                    control_variable = this_moment_variable
                end
                #end
            end
        end
    end
end