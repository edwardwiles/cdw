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
        thetaHat = -sum(deltaLambda .* deltaTau) ./ sum(deltaTau .* deltaTau)

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

    # Step 5: Compute gammaHat and gammaPrimeHat
    gammaHat = computeGamma(AHat, tau, wHat, thetaHat, sigma, L)
    gammaPrimeHat = computeGamma(AHat, tauPrime, wPrimeHat, thetaHat, sigma, LPrime)

    output = (μHat= 1 ./ thetaHat,
        wHat=wHat[:],
        cHat=cHat,
        λPrime=lambdaPrime,
        wPrimeHat=wPrimeHat[:],
        γHat=gammaHat[:],
        γPrimeHat=gammaPrimeHat[:])

    return output
end