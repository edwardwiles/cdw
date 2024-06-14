function localGravityMoment!(G, D, ω, prices, ξ, σ, μ, d, max_price, offset)
    tuner = -100.0
    β = 0.01
    pricesInd_without_o = copy(prices)
    prices_without_o = copy(prices)
    counter = 0

    for o = 1:D
        counter += 1
        if o != d
            prices_without_o = prices[:]
            prices_without_o[o] = max_price + 10000  # such that we are sure the price from o is not the min
            min_price_without_o, exporter_idx_without_o = findmin(prices_without_o[:])
            smoothMinIndNew!(pricesInd_without_o, prices_without_o, D, tuner)
            A = prices[o]^(1 - σ) * SmoothDirac(β, log(prices[o] / min_price_without_o))
            B = prices[d]^(1 - σ) * pricesInd_without_o[d] * SmoothDirac(β, log(prices[o] / prices[d]))

            moment_idx = (D - 1) * (d - 1) + counter

            G[ω, end-offset-moment_idx] = σ - 1 + A / ξ[o] + B / ξ[d] - 1 / μ 
        end
    end

end