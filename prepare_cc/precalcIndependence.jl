
function precalcIndependence(Ū, params)

    @unpack W, D, useIndependentCFDs, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, momentOrder, stratifiedSampling, baseIndex, ImportanceSamplingWeight, IndMomentOrder = params 

    Ind_Moments = zeros(1, 1)
    IndCDF_Cells = Vector{Vector{Int}}(undef,1)

    if useIndependentCFDs == 1 

        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (3 + D^2) + sameMarginalsMoment * (2 * momentOrder) * D^2
        Ind_Moments = zeros(W, D * (D^2 - floor(Int, D * (1 + D) / 2)) + (D-1)* 2 * momentOrder)

        idx_corss_moment = 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for c = 1:D
                    for f = 1:D
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        if c1 > o1 && d == f # condition on Uod, Uo'd only
                            idx_corss_moment += 1
                            @. Ind_Moments[:, idx_corss_moment] = Ū[:, o1, 1] .* Ū[:, c1, 1]
                        end
                    end
                end
            end
        end
        # Equalize the CDF of U_{o}/min_{o-}U_{o'}, only for baseIndex
        RatioMinU = zeros(W, D)
        CDF_Moments_for_ratio_base = zeros(W, 2*momentOrder)
        for ω =1:W
            tempU = Ū[ω, (baseIndex-1)*D+1:baseIndex*D, 1]
            for o = 1:D
                o1 = o + (baseIndex - 1) * D # uncomment to to U_{od} rather than U_o 
                #RatioMinU[ω, o] = minimum(tempU[Not(o)])/Ū[ω, o1, 1]
                RatioMinU[ω, o] = minimum(tempU[Not(o)])
            end
        end
        ## dropping this min stuff
        if stratifiedSampling == 1
            Half_W = floor(Int, W / 2)
            strata_size = floor(Int, Half_W / D^2)
            strata_index = baseIndex + (baseIndex - 1) * D - 1
            CDF_Ratio_X = quantile(RatioMinU[Not(union(1+strata_index*strata_size:(strata_index+1)*strata_size, Half_W+1+strata_index*strata_size:Half_W+(strata_index+1)*strata_size)), baseIndex], range(1 / (2 * momentOrder), (2 * momentOrder - 1) / (2 * momentOrder), length=2 * momentOrder - 1))
        else
            CDF_Ratio_X = quantile(RatioMinU[:, baseIndex], range(1 / (2 * momentOrder), (2 * momentOrder - 1) / (2 * momentOrder), length=2 * momentOrder - 1))
        end

        @show CDF_Ratio_X
        for ω = 1:W
            smallest_X = searchsortedfirst(CDF_Ratio_X, RatioMinU[ω, baseIndex])
            smallest_X = max(1, smallest_X)

            for i = smallest_X:2*momentOrder
                CDF_Moments_for_ratio_base[ω, i] += 1
            end
        end
        offset = D * (D^2 - floor(Int, D * (1 + D) / 2))
        o_index = 0
        for o = 1:D
            if o != baseIndex
                for ω = 1:W
                    smallest_X = searchsortedfirst(CDF_Ratio_X, RatioMinU[ω, o])
                    smallest_X = max(1, smallest_X)
                    # I remove the symmetry condition for now
                    #@. Ind_Moments[ω, offset+o_index*2*momentOrder+smallest_X:offset+(o_index+1)*2*momentOrder] = 1 .- CDF_Moments_for_ratio_base[ω, smallest_X:2*momentOrder]
                end 
                o_index += 1
            end
        end

    elseif useIndependentCFDs == 0
        Ind_Moments = zeros(W, D * (D^2 - floor(Int, D * (1 + D) / 2)) + (IndMomentOrder-1) + (IndMomentOrder-1)^D)

        idx_corss_moment = 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for c = 1:D
                    for f = 1:D
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        if c1 > o1 && d == f # condition on Uod, Uo'd only
                            idx_corss_moment += 1
                            @. Ind_Moments[:, idx_corss_moment] = Ū[:, o1, 1] .* Ū[:, c1, 1]
                        end
                    end
                end
            end
        end

        # Joint CDF = Product of Marginal CFDs

        IndCDF_K = range(1,(IndMomentOrder-1), (IndMomentOrder-1))
        IndCDF_X = quantile(Ū[:, 1, 1] .* ImportanceSamplingWeight[:], range(1/IndMomentOrder, (IndMomentOrder-1) / IndMomentOrder, length=IndMomentOrder-1))

        # CDF U11
        offset = D * (D^2 - floor(Int, D * (1 + D) / 2))
        for ω =1:W
            @inbounds for i in 1:length(IndCDF_X)
                Ind_Moments[ω, offset+i] = (Ū[ω, 1, 1] < IndCDF_X[i]) ? 1 : 0
            end
        end

        # Joint CDF
        IndCDF_Cells = collect(with_replacement_combinations(IndCDF_K,D))
        offset = D * (D^2 - floor(Int, D * (1 + D) / 2)) + IndMomentOrder-1
        for ω =1:W
            idx_corss_moment = 0
            @inbounds for i in 1:length(IndCDF_Cells)
                this_cdf = 1
                for o=1:D
                    o1 =  o + (baseIndex - 1) * D 
                    this_cdf *= (Ū[ω, o1, 1] < IndCDF_X[floor(Int,IndCDF_Cells[i][o])]) ? 1 : 0
                end
                idx_corss_moment +=1
                Ind_Moments[ω, offset+idx_corss_moment] = this_cdf
            end
        end

    end 

    return Ind_Moments, IndCDF_Cells

end 