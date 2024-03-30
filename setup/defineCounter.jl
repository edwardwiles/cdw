function defineCounter(τData, LData, counterType)
    # define counterfactual features of the model
    D = length(LData)
    LPrime = LData  # if using LPrime \neq L, edit here
    # set counterfactual tau
    if counterType == 0 # zero gravity 
        τPrime = ones(D, D)
    elseif counterType == 1 # autarky 
        τPrime = τData .^ Inf
    elseif counterType == 2 # in between custom alternative 
        τPrime = 0.5 .* τData + 0.5 .* ones(D, D) # edit custom alternate matrix here
    end

    return counters = (τPrime = τPrime, LPrime = LPrime)

end