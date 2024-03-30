function sameMarginalsMoment!(Ū, G, PMM, D, W, μ_σ, ν, momentOrder, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, baseIndex)
    # assures the marginals are equal amongst themselves
    # scaled Uod moments are equalized
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach
    
    refIndex = 1 #do not change
    refIndex1 = refIndex + (refIndex - 1) * D

    νk = zeros(D, D, momentOrder * 2)
    for o = 1:D
        for d = 1:D

            for k = 1:momentOrder
                α_k = k
                νk[o, d, k] = ν[o, d]^α_k
            end

            for k = 1:momentOrder
                α_k = -1 + 1 / (1 + k)
                νk[o, d, k+momentOrder] = ν[o, d]^α_k
            end
        end
    end

    
    for o = 1:D
        for d = 1:D
            o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 

            #+ve moments

            @. G[:, end-offset-(1-1)*D*D-o1] = Ū[:, o1, 1] ./ νk[o, d, 1] .- Ū[:, refIndex1, 1] ./ νk[refIndex, refIndex, 1]
            for k = 2:momentOrder
                @. G[:, end-offset-(k-1)*D*D-o1] = (Ū[:, o1, k] ./ νk[o, d, k] .- Ū[:, o1, k-1] ./ νk[o, d, k-1]) .- (Ū[:, refIndex1, k] ./ νk[refIndex, refIndex, k] .- Ū[:, refIndex1, k-1] ./ νk[refIndex, refIndex, k-1])
            end
            #-ve moments
            @. G[:, end-offset-(momentOrder+1-1)*D*D-o1] = Ū[:, o1, momentOrder+1] ./ νk[o, d, momentOrder+1] .- Ū[:, refIndex1, momentOrder+1] ./ νk[refIndex, refIndex, momentOrder+1]
            for k = 2:momentOrder
                @. G[:, end-offset-(momentOrder+k-1)*D*D-o1] = (Ū[:, o1, momentOrder+k] ./ νk[o, d, k+momentOrder] .- Ū[:, o1, k+momentOrder-1] ./ νk[o, d, k+momentOrder-1]) .- (Ū[:, refIndex1, k+momentOrder] ./ νk[refIndex, refIndex, k+momentOrder] .- Ū[:, refIndex1, k+momentOrder-1] ./ νk[refIndex, refIndex, k+momentOrder-1])
            end
        end
    end

end