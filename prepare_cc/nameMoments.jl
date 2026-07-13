
function nameMoments(numMoments, K_, params)

    @unpack D, counterType, GravityMomentFirstApproach, localGravityMoment, gravMoment, localGravityCrossMoment, sameMarginalsMoment, useCDFforMarginalMatching, NoScalingforSameMartingale, momentOrder, momentOrderForBaseIndex, baseIndex, independenceMoment  = params 

    if counterType != 1
        numMoments += (D - 1)
    end

    # # this is to verify the moments are written where they are supposed to
    MomentNames = fill("", numMoments)
    if counterType != 1
        cInd = D^2 + D - 1
    else
        cInd = D^2
    end

    if counterType != 1
        bInd = D^2
        cInd = D^2 + D - 1
        dInd = D^2 + 2 * D - 1
    else
        cInd = D^2
        dInd = D^2 + D
    end


    ##### Naming the Moments
    for d = 1:D
        for o = 1:D # using min prices, compute implied expenditure share and fill in G with implied minus data
            d1 = d + (o - 1) * D
            MomentNames[d1] = "lambda [$o ,$d]"
        end
        if counterType != 1
            MomentNames[cInd+d] = "gamma [$d]"
            MomentNames[dInd+d] = "gammaPrime [$d]"
            MomentNames[bInd+d-1] = "WagePrime [$d] "
        end
    end
    if counterType == 1
        # reduced autarky layout: only the single counterfactual moment at col D^2+1
        # (baseline price-index moments dropped as redundant).
        MomentNames[D^2+1] = "gammaPrime [$baseIndex]"
    end

    if GravityMomentFirstApproach == 1
        MomentNames[end] = "Gravity First Approach"
        if sameMarginalsMoment == 0
            for o = 1:D
                for d = 1:D
                    o1 = o + (d - 1) * D
                    MomentNames[dInd+o1] = "First Moment [$o , $d]"
                end
            end
        end
    end

    for d = 1:D
        counter = 0
        if localGravityMoment == 1
            for o = 1:D
                counter += 1
                if o != d
                    moment_idx = (D - 1) * (d - 1) + counter
                    MomentNames[end-GravityMomentFirstApproach-gravMoment-moment_idx] = "Local ACR [$o, $d]"
                end
            end
        end
    end

    for d = 1:D
        counter = 0
        if localGravityCrossMoment == 1
            for o = 1:D
                if o != d
                    for c = 1:D
                        if c != o && c != d
                            counter += 1
                            moment_idx = localGravityMoment * D * (D - 1) + (D - 1) * (D - 2) * (d - 1) + counter
                            MomentNames[end-GravityMomentFirstApproach-gravMoment-moment_idx] = "Local ACR [$o, $d , $c]"
                        end
                    end
                end
            end
        end
    end


    if gravMoment == 1
        MomentNames[end - GravityMomentFirstApproach] = "Gravity Second Approach"
    end


    if sameMarginalsMoment == 1 && useCDFforMarginalMatching == 0
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (3 + D^2)

        refIndex = 1 #do not change
        refIndex1 = refIndex + (refIndex - 1) * D
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 

                #+ve moments

                MomentNames[end-offset-(1-1)*D*D-o1] = "Same Marginals [$o, $d , k+= 1]"
                for k = 2:momentOrder
                    MomentNames[end-offset-(k-1)*D*D-o1] = "Same Marginals [$o, $d , k+= $k]"
                end
                #-ve moments

                MomentNames[end-offset-(momentOrder+1-1)*D*D-o1] = "Same Marginals [$o, $d , k-= 1]"
                for k = 2:momentOrder
                    MomentNames[end-offset-(momentOrder+k-1)*D*D-o1] = "Same Marginals [$o, $d , k+= $k]"
                end
            end
        end
    elseif sameMarginalsMoment == 1 && useCDFforMarginalMatching == 1 && NoScalingforSameMartingale == 0
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (3 + D^2)

        #CDF_X is a vector of increasing values of X. 
        # the moment condition is CDF_od(X) = CDF_11(X)
        # E[U_od<=X_k] = E[U_11<=X_k]

        #K = length(CDF_X) + 2
        refIndex = 1 #do not change
        refIndex1 = refIndex + (refIndex - 1) * D
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                MomentNames[end-offset-o1] = "Same Marginals CDF [$o, $d , First Moment]"

                for i = 1:K
                    MomentNames[end-offset-i*D^2-o1] = "Same Marginals CDF [$o, $d , k= $i]"
                end
            end
        end
    elseif sameMarginalsMoment == 1 && useCDFforMarginalMatching == 1 && NoScalingforSameMartingale == 1
        #K_ = size(CDF_Moments, 2)
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (3 + D^2)
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                MomentNames[end-offset-K_+o1] = "Same Marginals CDF [$o, $d , First Moment]"
                for i = 1:K
                    MomentNames[end-offset-K_+o1+i*D^2] = "Same Marginals CDF [$o, $d , k= $i]"
                end
            end
        end

        for i = 1:momentOrderForBaseIndex
            MomentNames[end-offset-K_+(K+1)*D^2+i] = "Same Marginals CDF for baseIndex [$baseIndex, $baseIndex , k= $i]"
        end
    end

    if independenceMoment == 1 && NoScalingforSameMartingale == 1
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (3 + D^2) + sameMarginalsMoment * ( (2 * momentOrder) * D^2 + momentOrderForBaseIndex)

        MomentNames[end-offset] = "Uncorrelation [1, 1, First Moment Value]"

        #K_ = size(Ind_Moments, 2)

        idx_corss_moment = 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for c = 1:D
                    for f = 1:D
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        #if c1 > o1
                        if c1 > o1 && d == f
                            idx_corss_moment += 1
                            MomentNames[end-offset-1-K_+idx_corss_moment] = "Uncorrelation [ $o, $d, $c , $f]"
                        end
                    end
                end
            end
        end
    end

    return MomentNames 

end 