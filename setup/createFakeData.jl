function createFakeData(D, scale)
    # creates a fake dataset where true DGP is Frechet 
    # to use non-Frechet, would need to take draws from alternative DGP and iterate wages without Frechet simplified formulas 

    # create trade cost matrix
    tradeCosts = Array{Float64,2}(undef, D, D) # initialise a matrix
    createTradeCosts!(tradeCosts, scale, D) # creates DxD matrix of taus 

    # define fake "true" parameters
    A = rand(D, D) .+ 1
    A .= A / A[1, :]' # normalise A_{1d} to 1 for all d
    L = rand(D, 1) .+ 1
    theta = 6

    lambda = Array{Float64,2}(undef, D, D)  # initialise

    w = ones(D)
    iterWagesTheory!(w, L, A, tradeCosts, theta, lambda) # solve for wages under Frechet DGP

    # sanity check, but this should return zero as A and tau are independent

    #@show MartingalDifferenceDivergence2(A .^ -1, tradeCosts, D)

    return lambda, L, tradeCosts
end