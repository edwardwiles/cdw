function master_prestep(data, counters, globalParams)
    
    # unpack data 
    lambda = data.λData 
    L = data.LData 
    tau = data.τData 
    
    # unpack counterfactual info 
    tauPrime = counters.τPrime 
    LPrime = counters.LPrime 

    # unpack parameter info
    thetaIn = globalParams.θHat 
    sigma = globalParams.σHat 
    baseIndex = globalParams.baseIndex 
    counterType = globalParams.counterType 
    D = globalParams.D

    # Step 1: Estimate thetaHat via gravity or prespecified
   if thetaIn == 0 # use gravity to estimate theta if no theta prespecified
        deltaLambda = doubleDiff(lambda)
        deltaTau = doubleDiff(tau)
        meanTau = mean(deltaTau)
        thetaHat = -sum(deltaLambda .* (deltaTau .-meanTau)) / sum(deltaTau .* deltaTau .- meanTau^2)
    elseif thetaIn > 0
        thetaHat = thetaIn
    end

    # Step 2: Estimate baseline wages
    wHat = ones(D)
    iterWagesPreStep!(wHat, L, lambda)
    wHat = wHat ./ wHat[baseIndex] # normalise so wage is 1 for specified base country 

    # Step 3: Estimate cHat
    AHat = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (thetaHat)) .* (lambda ./ lambda[1, :]')
    cHat = AHat .^ (-1) # we define c as 1/A 

    # Step 4: Estimate wPrimeHat
    lambdaPrime = Array{Float64,2}(undef, D, D)  # initialise

    wPrimeHat = ones(D)

    if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
        iterWagesTheory!(wPrimeHat, LPrime, AHat, tauPrime, thetaHat, lambdaPrime)
        wPrimeHat[:] = wPrimeHat ./ wPrimeHat[baseIndex] # normalise 
    end

    # Step 5: compute the A_od (outerscaling), if we want to start from a decentered theta_init != thetaHat
    additional_theta = globalParams.theta_init == 0 ?  thetaHat :  globalParams.theta_init
    AHat_additional = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (additional_theta)) .* (lambda ./ lambda[1, :]')
    cHat_additional = AHat_additional .^ (-1) # we define c as 1/A 
    
    Aod_initial = cHat ./ ((cHat_additional) .^(thetaHat/additional_theta))



    # Step 5: Compute gammaHat and gammaPrimeHat
    gammaHat = computeGamma(AHat_additional, tau, wHat, additional_theta, sigma, L)
    gammaPrimeHat = computeGamma(AHat_additional, tauPrime, wPrimeHat, additional_theta, sigma, LPrime)



    output = (μHat= 1 ./ thetaHat,
        wHat=wHat[:],
        cHat=cHat,
        λPrime=lambdaPrime,
        wPrimeHat=wPrimeHat[:],
        γHat=gammaHat[:],
        γPrimeHat=gammaPrimeHat[:],
        Aod_initial = reshape(Aod_initial, D^2)[:])

    return output
end