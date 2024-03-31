
function precalcCDFs(Ū, params)

    @unpack W, D, momentOrder, momentOrderForBaseIndex, importanceSampling, ImportanceSamplingWeight, stratifiedSampling, NoScalingforSameMartingale, ForceFrechetMarginal = params 

    CDF_X = zeros(1)
    CDF_X_for_base = zeros(1)
    # pre-calculate the quantiles for marginal matching with CDF methodology
      
    if importanceSampling == 1
        CDF_X = quantile(Ū[:, 1, 1] .* ImportanceSamplingWeight[:], range(1 / (2 * momentOrder - 2), (2 * momentOrder - 3) / (2 * momentOrder - 2), length=2 * momentOrder - 3))
        CDF_X_for_base = quantile(Ū[:, 1, 1] .* ImportanceSamplingWeight[:], range(1 / (momentOrderForBaseIndex - 1), (momentOrderForBaseIndex - 2) / (momentOrderForBaseIndex - 1), length=momentOrderForBaseIndex - 2))

    elseif stratifiedSampling == 1
        Half_W = floor(Int, W / 2)
        strata_size = floor(Int, Half_W / D^2)

        CDF_X = quantile(Ū[union(strata_size+1:Half_W, strata_size+1+Half_W:W), 1, 1], range(1 / (2 * momentOrder - 2), (2 * momentOrder - 3) / (2 * momentOrder - 2), length=2 * momentOrder - 3))
        CDF_X_for_base = quantile(Ū[union(strata_size+1:Half_W, strata_size+1+Half_W:W), 1, 1], range(1 / (momentOrderForBaseIndex - 1), (momentOrderForBaseIndex - 2) / (momentOrderForBaseIndex - 1), length=momentOrderForBaseIndex - 2))
    else

        CDF_X = quantile(Ū[:, 1, 1], range(1 / (2 * momentOrder - 2), (2 * momentOrder - 3) / (2 * momentOrder - 2), length=2 * momentOrder - 3))
        CDF_X_for_base = quantile(Ū[:, 1, 1], range(1 / (momentOrderForBaseIndex - 1), (momentOrderForBaseIndex - 2) / (momentOrderForBaseIndex - 1), length=momentOrderForBaseIndex - 2))
    end
    

    if NoScalingforSameMartingale == 1 
        K = length(CDF_X) + 2
        CDF_Moments = zeros(W, (K + 1) * D^2 + momentOrderForBaseIndex + D^2)
        CDF_Moments_for_base = zeros(W, momentOrderForBaseIndex)

        refIndex = 1 #do not change
        refIndex1 = refIndex + (refIndex - 1) * D
        CDF_11_X = zeros(W, K)
        CDF_11_X_for_base = zeros(W, momentOrderForBaseIndex)


        for ω = 1:W
            smallest_X = searchsortedfirst(CDF_X, Ū[ω, refIndex1, 1]) + 1
            for i = smallest_X:K
                CDF_11_X[ω, i] = 1
            end

            smallest_X_for_base = searchsortedfirst(CDF_X_for_base, Ū[ω, refIndex1, 1]) + 1
            for i = smallest_X_for_base:momentOrderForBaseIndex
                CDF_11_X_for_base[ω, i] = 1
            end
        end

        if ForceFrechetMarginal == 1 # Replace the realizations with their expectation, this way the CDF does not change with measure change.

            CDF_11_X_expectation = zeros(K)
            CDF_11_X_for_base_expectation = zeros(momentOrderForBaseIndex)
            for i = 1:K
                if importanceSampling ==1
                    CDF_11_X_expectation[i] = sum(CDF_11_X[:,i] .* ImportanceSamplingWeight[:])/W
                elseif stratifiedSampling == 1
                    Half_W = floor(Int, W / 2)
                    CDF_11_X_expectation[i] = ((0.1/0.5)*sum(CDF_11_X[1:Half_W,i]) + (0.9/0.5)*sum(CDF_11_X[Half_W+1:W,i])) /W
                else
                    CDF_11_X_expectation[i] = sum(CDF_11_X[:,i])/W
                end

                @. CDF_11_X[:,i] = CDF_11_X_expectation[i]
            end

            for i = 1:momentOrderForBaseIndex
                if importanceSampling == 1
                    CDF_11_X_for_base_expectation[i] = sum(CDF_11_X_for_base[:,i].* ImportanceSamplingWeight[:])/W
                elseif stratifiedSampling == 1
                    Half_W = floor(Int, W / 2)
                    CDF_11_X_for_base_expectation[i] = ((0.1/0.5)*sum(CDF_11_X_for_base[1:Half_W,i]) + (0.9/0.5)*sum(CDF_11_X_for_base[Half_W+1:W,i])) /W
                else
                    CDF_11_X_for_base_expectation[i] = sum(CDF_11_X_for_base[:,i])/W
                end

                @. CDF_11_X_for_base[:,i] = CDF_11_X_for_base_expectation[i]
            end

        end

        # for o= baseIndex d = baseIndex, cach the CDF realizations for momentOrderForBaseIndex points
        o1_base = baseIndex + (baseIndex - 1) * D
        for i = 1:momentOrderForBaseIndex
            @. CDF_Moments_for_base[:, i] += -CDF_11_X_for_base[:, i]
        end
        for ω = 1:W
            smallest_X = searchsortedfirst(CDF_X_for_base, Ū[ω, o1_base, 1])

            for i = (smallest_X+1):momentOrderForBaseIndex
                CDF_Moments_for_base[ω, i] += 1
            end
        end
        
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                @. CDF_Moments[:, o1] = Ū[:, o1, 1] .- Ū[:, refIndex1, 1]
                @. CDF_Moments[:,end+1-o1] = Ū[:, o1, 1]^(-1)  .- Ū[:, refIndex1, 1]^(-1) # First Moment of inverse


                for i = 1:K
                    @. CDF_Moments[:, o1+i*D^2] += -CDF_11_X[:, i]
                end
                for ω = 1:W
                    smallest_X = searchsortedfirst(CDF_X, Ū[ω, o1, 1])

                    for i = (smallest_X+1):K
                        CDF_Moments[ω, o1+i*D^2] += 1
                    end
                end
            end
        end

        @. CDF_Moments[:, (K+1)*D^2+1:end-D^2] += CDF_Moments_for_base[:, :]        

    end 

    return CDFs(CDF_X = CDF_X, CDF_X_for_base = CDF_X_for_base, CDF_Moments = CDF_Moments, CDF_Moments_for_base = CDF_Moments_for_base)

end 