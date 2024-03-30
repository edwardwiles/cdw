function localGravityCrossMoment!(G, PMM, D, ω, prices, ξ, σ, d, max_price, gravMoment, localGravityMoment, GravityMomentFirstApproach)
    tuner = -100.0
    β = 0.01
    pricesInd_without_c = copy(prices)
    prices_without_c = copy(prices)
    counter = 0
    for o = 1:D
        if o != d
            for c = 1:D
                if c != o && c != d
                    counter += 1
                    prices_without_c = prices[:]
                    prices_without_c[c] = max_price + 10000 # such that we are sure the price from c is not the min
                    min_price_without_c, exporter_idx_without_c = findmin(prices_without_c[:])
                    smoothMinIndNew!(pricesInd_without_c, prices_without_c, D, tuner)
                    A = prices[o]^(1 - σ) * pricesInd_without_c[o]SmoothDirac(β, log(prices[c] / prices[o]))
                    B = prices[d]^(1 - σ) * pricesInd_without_c[d] * SmoothDirac(β, log(prices[c] / prices[d]))
                    moment_idx = localGravityMoment * D * (D - 1) + (D - 1) * (D - 2) * (d - 1) + counter
                    G[ω, end-GravityMomentFirstApproach-gravMoment-moment_idx] = A / ξ[o] - B / ξ[d] - PMM[end-GravityMomentFirstApproach-gravMoment-moment_idx]
                end
            end
        end
    end
end