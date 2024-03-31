function fillUBarMoments!(Ū, U, params)

    @unpack momentOrder = params 

    for k = 2:momentOrder
        α_k = k
        theoretical_moment = gamma(1 + α_k) # k->inf => theoretical_moment => inf 
        @. Ū[:, :, k] = U[:, :] .^ α_k
        @. Ū[:, :, k] = Ū[:, :, k] ./ theoretical_moment
    end

    for k = 1:momentOrder
        α_k = -1 + 1 / (1 + k) # k->inf => α_k => -1 
        theoretical_moment = gamma(1 + α_k) # k->inf => theoretical_moment => inf 
        @. Ū[:, :, k+momentOrder] = U[:, :] .^ α_k
        @. Ū[:, :, k+momentOrder] = Ū[:, :, k+momentOrder] ./ theoretical_moment
    end
end 