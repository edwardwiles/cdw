function smoothMinIndNew!(xInd, x, D, tuner)
    # function takes a GxD matrix and calculates another GxD matrix
    # with a smooth approximation of indicator functions for row-wise min
    # i.e., it computes the min of each row and indicates if x_{ij} is that min
    # tuner controls sharpness of approximation (close to 0 = very sharp)

    # this is necessary for optimisers that use auto diff, because everything must be smooth (no mins, no indicators)
    # if not using auto diff, can just use min and indicator 

    # smoothing approximation is softmax, see https://en.wikipedia.org/wiki/Softmax_function 
    denom = 0

    xMin = minimum(x)

    for i = 1:D
        xInd[i] = exp((x[i] - xMin) * tuner)
        #xInd[i] = exp((x[i]) * tuner)

        denom += xInd[i]
    end

    for i = 1:D
        xInd[i] /= denom
    end

end

function MinInd!(xInd, x, D)
    # function takes a GxD matrix and calculates another GxD matrix
    # with a smooth approximation of indicator functions for row-wise min
    # i.e., it computes the min of each row and indicates if x_{ij} is that min
    xMin = minimum(x)
    for i = 1:D
        xInd[i] = (x[i] > xMin) ? 0 : 1
    end
end