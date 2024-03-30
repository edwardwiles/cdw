function iterWagesTheory!(w0, L, A, tau, theta, lambda)
    # find market clear wages under Frechet by iteration

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

        # see note for algebra 
        phi = A .* (tau .* w0) .^ (-theta)

        lambda[:, :] .= phi ./ (sum(phi, dims=1))
        w1[:] = lambda * (w0 .* L) ./ L # implied wages 

        diff = maximum(abs.(w1[:] .- w0[:]))

        if mod1(iter, 50) == 1
            @show diffTheory = diff # print progress occasionally
        end

        iter += 1

    end

    if iter == maxIter
        @show -9999999999999999999 # if failed to converge, print error 
    end

    w0[:] ./= w0[1] # normalise country 1's wage to 1 
    lambda[:,:] .*= 1

end