function moments!(K, G, θ, U, obj)
    # main function that takes empty K and G, and the parameters, and fills in the moment matrices 

    # unpack the gamma (auxiliary parameters) vector
    @unpack wHat, L, LPrime, τ, τPrime, P, PMM, baseIndex, indicators, Uσ, Ū, Σ_od, Mτ, μHat, CDF_X, CDF_Moments, Ind_Moments , cHat, IndCDF_Cells, importanceSamplingWeights= obj.γ
    @unpack counterExplicit, counterType, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, momentOrderForBaseIndex, StarDistributionType, useCDFforMarginalMatching, stratifiedSampling, useIndependentCFDs,IndMomentOrder  = indicators

    W = size(U, 1)
    D = size(τ, 1)

    # unpack the structural parameters vector
    μ = θ[1]
    σ = θ[2]
    γ = θ[3:3+D-1]

    γ_prime = copy(γ)



    if counterType != 1
        γ_prime = θ[3+D:3+2*D-1]
    else
        γ_prime[baseIndex] = θ[3+D] # we are not interested in the other gamma_primes
    end

    if counterType != 1 # needs to be adjusted
        wPrime = θ[3+2*D:3+2*D+(D-1)-1]
    else
        wPrime = copy(obj.γ.wPrimeHat)
    end

    # add 1 (normalised wage) into the w' vector at appropriate index
    insert!(wPrime, baseIndex, 1)

    counterType_θ_offset = 0
    if counterType != 1
        counterType_θ_offset = 2 * (D - 1) # D-1 wagesPrime and D-1 gamma_primes 
    end

    if counterExplicit == 0
        # insert relevant k function if counterfactual does not depend on U
        counterVal = (γ[baseIndex] / γ_prime[baseIndex])^(σ / (σ - 1)) - 1
        @inbounds for i = 1:W
            K[i] = counterVal
        end
    end


    if θConstant != 1
        # update c and U matrices to the exponents relevant for calculating price
        # nb: calculate here as don't want to do it in each hFunction call
        # only do this if theta / sigma ever vary, otherwise we precalculate

        #UPow = copy(U)
        UPow = zeros(eltype(γ), size(U))
        UσPow = zeros(eltype(γ), size(U))
        for i = 1:length(UPow)
            UPow[i] = U[i]^(-μ)
            UσPow[i] = Uσ[i]^(-μ)
        end
        hFunction!(K, G, UPow, UσPow, Ū, wHat, τ, σ, μ, γ, L, P, PMM, counterExplicit, counterType, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, μHat, baseIndex) # fill in G with baseline moments 
        hFunctionCounter!(K, G, UPow, UσPow, wPrime, τPrime, σ, γ_prime, LPrime, P, PMM, counterExplicit, counterType, baseIndex) # fill in G with counterfactual moments, fill in K 
    else

        hFunction!(K, G, U, Uσ, Ū, wHat, τ, σ, μ, γ, L, P, PMM, counterExplicit, counterType, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, μHat, baseIndex)
        hFunctionCounter!(K, G, U, Uσ, wPrime, τPrime, σ, γ_prime, LPrime, P, PMM, counterExplicit, counterType, baseIndex)
    end


    if sameMarginalsMoment == 1
        ν = ones(D, D)
        if NoScalingforSameMartingale == 0
            ν = reshape(vcat(1, θ[counterType_θ_offset+3+2*D:counterType_θ_offset+3+2*D+D^2-2]), (D, D))
        end
        if useCDFforMarginalMatching == 0
            sameMarginalsMoment!(Ū, G, PMM, D, W, (1 - σ) * μ, ν, momentOrder, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, baseIndex)
        elseif NoScalingforSameMartingale == 0
            sameMarginalsMomentCDF!(Ū, G, PMM, D, W, (1 - σ) * μ, ν, CDF_X, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, baseIndex)
        else
            sameMarginalsMomentCDFNoScaling!(G, PMM, D, W, CDF_Moments, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, baseIndex)
        end
    end

    if independenceMoment == 1

        ν = ones(D, D)
        ηk = θ[counterType_θ_offset+3+D+1:counterType_θ_offset+3+D+1]

        if NoScalingforSameMartingale == 0
            ν = reshape(vcat(1, θ[counterType_θ_offset+3+2*D:counterType_θ_offset+3+2*D+D^2-2]), (D, D))
            ηk = θ[counterType_θ_offset+3+2*D+D^2-1:counterType_θ_offset+3+2*D+D^2-1]
            uncorrelationMoment!(Ū, G, PMM, D, W, ν, ηk, momentOrder, momentOrderForBaseIndex, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, sameMarginalsMoment)
        else
            ν = θ[counterType_θ_offset+3+D+2:counterType_θ_offset+3+D+1+IndMomentOrder-1]
            uncorrelationMomentNoScaling!(Ū, G, PMM, D, W, ηk, Ind_Moments, momentOrder, momentOrderForBaseIndex, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, sameMarginalsMoment,useIndependentCFDs,IndMomentOrder,IndCDF_Cells, ν )
        end


        #independenceMoment!(Ū, G, PMM, D, W, ν, ηk, momentOrder, gravMoment, localGravityMoment, GravityMomentFirstApproach,localGravityCrossMoment, sameMarginalsMoment)
    end


    if gravMoment == 1
        newGravityMoment!(G, PMM, τ, D, W, γ, U, GravityMomentFirstApproach) # add gravity moment if using 
    end




    if stratifiedSampling ==1
    #stratified sampling reweighting
    # TO DO:  Handle outerloop moments. 
    half_W = floor(Int, W/2)
    @. G[1:half_W,:] = (0.1/0.5) .* G[1:half_W,:]  
    @. G[half_W+1:W,:] = (0.9/0.5) .* G[half_W+1:W,:]
    end

    if GravityMomentFirstApproach == 1
        if sameMarginalsMoment ==1 && NoScalingforSameMartingale == 1
            ν = ones(D,D)
            GravityMomentFirstApproach!(G, PMM, τ, ν, cHat, D, Ū, counterType, sameMarginalsMoment)
        else
            ν = reshape(vcat(1, θ[counterType_θ_offset+3+2*D:counterType_θ_offset+3+2*D+D^2-2]), (D, D))
            GravityMomentFirstApproach!(G, PMM, τ, ν, cHat, D, Ū, counterType, sameMarginalsMoment)
         end
    end


    #Always multiply by ISW which are defaulted to 1 if the methodology is not used
    for im=1:obj.d
        @. G[:,im] *= importanceSamplingWeights[:] 
    end
    @. K[:] *= importanceSamplingWeights[:] 

end