function iterWagesPreStep!(w0, L, lambda)
    # iterate wages again, diff from earlier function is we use lambda (trade shares), not A and phi 
    # see theory note 

    # could we use eigen decomposition of lambda and eigenvector

    tol = 1e-12
    maxIter = 100000
    ϵ = 0.6
    diff = tol + 1
    iter = 1

    w1 = copy(w0)

    while diff > tol && iter < maxIter

        w0[:] = w0[:] * (1 - ϵ) + w1[:] * ϵ
        w1[:] = lambda * (w0 .* L) ./ L

        diff = maximum(abs.(w1[:] .- w0[:]))

        if mod1(iter, 500) == 1
            @show diffPreStep = diff
        end

        iter += 1

    end

    if iter == maxIter
        @show -9999999999999999999
    end

end