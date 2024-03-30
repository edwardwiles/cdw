function sameMarginalsMomentCDF!(Ū, G, PMM, D, W, μ_σ, ν, CDF_X, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, baseIndex)
    # assures the marginals are equal amongst themselves
    # CFDs are equalized at specific quantiles
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach

    #CDF_X is a vector of increasing values of X. 
    # the moment condition is CDF_od(X) = CDF_11(X)
    # E[U_od<=X_k] = E[U_11<=X_k]
    K = length(CDF_X) + 2 # the two additional momets are for lower and upper "wings" 

    refIndex = 1 #do not change
    refIndex1 = refIndex + (refIndex - 1) * D
    CDF_11_X = zeros(W, K)
    for ω = 1:W
        smallest_X = searchsortedfirst(CDF_X, Ū[ω, refIndex1, 1] / ν[refIndex, refIndex]) + 1
        for i = smallest_X:K
            CDF_11_X[ω, i] += 1
        end
    end

    for o = 1:D
        for d = 1:D
            o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
            @. G[:, end-offset-o1] = Ū[:, o1, 1] ./ ν[o, d] .- Ū[:, refIndex1, 1] ./ ν[refIndex, refIndex]

            for i = 1:K
                @. G[:, end-offset-i*D^2-o1] += -CDF_11_X[:, i]
            end
            for ω = 1:W
                smallest_X = searchsortedfirst(CDF_X, Ū[ω, o1, 1] / ν[o, d])

                for i = smallest_X:K
                    G[ω, end-offset-i*D^2-o1] += 1
                end
            end
        end
    end
end

function sameMarginalsMomentCDFNoScaling!(G, PMM, D, W, CDF_Moments, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, baseIndex)
    # imposes Uods have the same CDF. It uses cached realizations as the moment condition does not depend on the parameters 
    K_ = size(CDF_Moments, 2)
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach
    @. G[:, end-offset-K_+1:end-offset] = CDF_Moments[:, :]
end