function uncorrelationMoment!(Ū, G, PMM, D, W, ν, ηk, momentOrder, momentOrderForBaseIndex, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, sameMarginalsMoment)
    # assures the marginals are identical and independent
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach  + sameMarginalsMoment * ((2 * momentOrder) * D^2 + momentOrderForBaseIndex)

    refIndex = 1
    refIndex1 = refIndex + (refIndex - 1) * D
    for ω = 1:W
        # E[U(ref,ref)^k/k!] = ηk
        G[ω, end-offset] = Ū[ω, refIndex1, 1] / ν[refIndex, refIndex] - ηk[1]
    end

    idx_corss_moment = 0
    for d = 1:D
        for o = 1:D
            o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
            for c = 1:D
                for f = 1:D
                    if c > o || f > d
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        idx_corss_moment += 1
                        for ω = 1:W
                            G[ω, end-offset-1-idx_corss_moment] = (Ū[ω, o1, 1] / ν[o, d]) * (Ū[ω, c1, 1] / ν[c, f]) - (ηk[1]) * (ηk[1])
                        end
                    end
                end
            end
        end
    end
end

function uncorrelationMomentNoScaling!(Ū, G, PMM, D, W, ηk, Ind_Moments, momentOrder, momentOrderForBaseIndex, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, sameMarginalsMoment,useIndependentCFDs,IndMomentOrder,IndCDF_Cells, ν)
    # imposes zero coreelation between Uods, implementation used cached realizations
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach + sameMarginalsMoment * ((2 * momentOrder + 1) * D^2 +momentOrderForBaseIndex)
    
    refIndex = 1
    refIndex1 = refIndex + (refIndex - 1) * D


    # E[U(ref,ref)^k/k!] = ηk
    @. G[:, end-offset] += Ū[:, refIndex1, 1] .- ηk[1]

    K_ = size(Ind_Moments, 2) 
    square_mean = ηk[1]^2
    @. G[:, end-offset-1-K_+1:end-offset-1] += Ind_Moments[:, :]

    @. G[:, end-offset-1-K_+1:end-offset-1-K_+ D* floor(Int, D * (D-1) / 2)] -= square_mean  # substract square mean for the correlation moments

    if useIndependentCFDs == 1
        # CDF_i = ν_i
        for i =1:IndMomentOrder-1
            @. G[:, end-offset-1-K_+D*floor(Int, D * (D-1) / 2)+i:end-offset-1-K_+ D*floor(Int, D * (D-1) / 2)+i] -= ν[i]
        end
        # joint CDF
        @inbounds for i in 1:length(IndCDF_Cells)
            this_cdf = 1
            for o=1:D
                this_cdf *= ν[floor(Int,IndCDF_Cells[i][o])]
            end
            @. G[:, end-offset-1-K_+D*floor(Int, D * (D-1) / 2)+IndMomentOrder-1+i:end-offset-1-K_+D*floor(Int, D * (D-1) / 2)+IndMomentOrder-1+i] -= this_cdf
        end

    end
end