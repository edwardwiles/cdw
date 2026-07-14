function defineCounter(τData, LData, counterType)
    # define counterfactual features of the model
    D = length(LData)
    LPrime = LData  # if using LPrime \neq L, edit here
    # set counterfactual tau
    if counterType == 0 # zero gravity 
        τPrime = ones(D, D)
    elseif counterType == 1 # autarky
        # NOT `τData .^ Inf`: mathematically 1.0^Inf == 1.0, not Inf, so any off-diagonal pair
        # with τData==1 exactly (a real occurrence in real trade-cost data -- e.g. a near-zero
        # measured trade cost between two countries -- never arises in the synthetic fakeData
        # generators, which is why this was never caught before) would silently stay fully
        # tradable under "autarky" instead of becoming infinitely costly. Construct τPrime
        # directly instead: Inf everywhere off-diagonal, own-country cost unchanged, independent
        # of what τData's off-diagonal values happen to be.
        τPrime = fill(Inf, D, D)
        for i in 1:D
            τPrime[i, i] = τData[i, i]
        end
    elseif counterType == 2 # in between custom alternative 
        τPrime = 0.5 .* τData + 0.5 .* ones(D, D) # edit custom alternate matrix here
    end

    return counters = (τPrime = τPrime, LPrime = LPrime)

end