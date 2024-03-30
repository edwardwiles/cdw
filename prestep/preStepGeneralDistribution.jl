function preStepGeneralDistribution(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType, distributionType, corr, vol, autocorr=0)

    D = size(lambda, 1)
    W = 1000000 # number of products for calculating the expectations, we use 1 million products so to emulate the "theoretical" quantities.
    # construct matrix of draws from distributionType
    U_init = zeros(W, D * D)
    Random.seed!(2^12)
    genRands!(U_init, distributionType, corr, vol, autocorr, tau)

    return preStepGeneralDistribution(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType, U_init)

end

function preStepGeneralDistribution(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType, U_init)
    # does the full pre-step to get the initial guess of parameters (psi^*) that corresponds with a specific distribution
    # the same function as above. But without the need to re-simulate the Us for the calibration.  

    D = size(lambda, 1)
    W = size(U_init, 1)

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

    UPow_init = zeros(W, D * D)
    UPow_init[:] = U_init[:] .^ (-1.0 / thetaHat)

    # use Frechet solution as the initial guess
    AHat_Frechet = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (thetaHat)) .* (lambda ./ lambda[1, :]')
    cHat_Frechet = AHat_Frechet .^ (-1) # we define c as 1/A 


    cHat = ones(D, D)
    # force cHat[1,:] =1
    normalizeAs = true
    if normalizeAs
        for d = 1:D
            function cHat_solver_func!(cHat_x)
                return gFunction_d!(sigma, tau, UPow_init, wHat, vcat(ones(1), cHat_x), lambda, d) # we force the cHat[1,:] =1
            end
            cHat_solver_results = nlsolve(cHat_solver_func!, (cHat_Frechet[2:D, d]) .^ (1.0 / thetaHat)) #, autodiff = :forward
            #@show cHat_solver_results.residual_norm ## check it converged
            cHat[2:D, d] .= abs.(cHat_solver_results.zero)[:]
            #@show cHat[:, d]
        end
    else
        for d = 1:D
            function cHat_solver_func2!(cHat_x)
                return gFunction_d2!(sigma, tau, UPow_init, wHat, cHat_x, lambda, d)
            end
            cHat_solver_results = nlsolve(cHat_solver_func2!, (cHat_Frechet[:, d]) .^ (1.0 / thetaHat)) #, autodiff = :forward
            #@show cHat_solver_results.residual_norm ## check it converged
            cHat[:, d] .= abs.(cHat_solver_results.zero)[:]
            #@show cHat[:, d]
        end
    end
    GC.gc()
    # @show cHat
    # @show lambda
    # @show (gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda)).expenditure_ratio_residual
    # @show (gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda)).trade_share_residual
    # @show (gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda)).expenditure

    # Step 4: Estimate wPrimeHat
    #lambdaPrime = Array{Float64,2}(undef, D, D)  # initialise
    lambdaPrime = zeros(D, D)
    wPrimeHat = ones(D)

    if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
        function wPrimeHat_solver_func!(wHatPrime_x)
            func_output = gFunction!(sigma, tauPrime, UPow_init, wHatPrime_x, cHat, lambda)

            wage_moment = zeros(Float64, D)
            for o = 1:D
                wage_moment[o] = sum(func_output.expenditure_ratio[o, :] .* wHatPrime_x .* LPrime) - wHatPrime_x[o] * LPrime[o]
            end

            return wage_moment
        end

        wPrimeHat_solver_results = nlsolve(wPrimeHat_solver_func!, wHat,)

        @show wPrimeHat_solver_results.residual_norm ## check it converged

        wPrimeHat = abs.(wPrimeHat_solver_results.zero)
        wPrimeHat[:] = wPrimeHat ./ wPrimeHat[baseIndex] # normalise 

        lambdaPrime = (gFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambda)).expenditure_ratio
    else
        for o = 1:D
            lambdaPrime[o, o] = 1
        end
    end

    price_index = ones(D)
    price_index_Prime = ones(D)

    price_index[:] .= ((gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda)).price_index)[:]
    price_index_Prime[:] .= ((gFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambdaPrime)).price_index)[:]

    # Step 5: Compute gammaHat and gammaPrimeHat
    gammaHat = ones(D)
    gammaPrimeHat = ones(D)

    gammaHat[:] = (price_index[:] ./ (wHat[:] .* L[:])) .^ (1.0 / sigma)
    gammaPrimeHat[:] = (price_index_Prime[:] ./ (wPrimeHat[:] .* LPrime[:])) .^ (1.0 / sigma)

    # we were using above cHat^(1/theta), for numerical efficiency. We adjust the power before returning the values.
    @. cHat[:] = cHat[:] .^ (thetaHat)

    # we scale so that cHat represents E[U_od]
    meanU = mean(U_init)
    #@. cHat[:] = cHat[:] .* meanU


    output = (μHat=1 ./ thetaHat,
        wHat=wHat[:],
        cHat=cHat,
        λPrime=lambdaPrime,
        wPrimeHat=wPrimeHat[:],
        γHat=gammaHat[:],
        γPrimeHat=gammaPrimeHat[:])

    return output
end