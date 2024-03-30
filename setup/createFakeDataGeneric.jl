function createFakeDataGeneric(D, scale)
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

    lambda_fake = Array{Float64,2}(undef, D, D)  # initialise
    lambda = Array{Float64,2}(undef, D, D)  # initialise

    w0 = ones(D)

    W = 10000 # number of products for calculating the expectations
    U_fake = zeros(W, D)
    Random.seed!(1234)
    genRands!(U_fake, 1, 0.3, 1, 0)

    # parameters for iteration 
    tol = 1e-12
    maxIter = 40000
    ϵ = 0.2
    diff = tol + 1
    iter = 1

    w1 = copy(w0)

    while diff > tol && iter < maxIter

        # update wage guess
        w0[:] = w0[:] * (1 - ϵ) + w1[:] * ϵ

        lambda = (gFunction!(2.5, tradeCosts, U_fake, w0, A .^ (-1.0 / theta), lambda_fake)).expenditure_ratio
        w1[:] = lambda .* (w0 .* L) ./ L # implied wages 

        diff = maximum(abs.(w1[:] .- w0[:]))

        if mod1(iter, 500) == 1
            @show diffTheory = diff # print progress occasionally
        end

        iter += 1

    end

    if iter == maxIter
        @show -9999999999999999999 # if failed to converge, print error 
    end

    w0[:] ./= w0[1] # normalise country 1's wage to 1 



    @show lambda

    return lambda, L, tradeCosts
end