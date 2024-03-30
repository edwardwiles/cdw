function strongGravityMomentold2!(G, PMM, τ, ν, D, W, Ū, Σ, Mτ, counterType)

    # constructs the moment that ΔΔ E[ln U] is mean independent of ΔΔ lnτ  

    dInd = counterType == 1 ? D^2 + 2 * D : D^2 + (D - 1) + 2 * D

    MDD = MartingalDifferenceDivergence(ν, Mτ, D)

    for o = 1:D
        for d = 1:D
            o1 = o + (d - 1) * D
            Σ_od = Σ[o, d]
            ν_od = ν[o, d] / Σ_od
            @. G[:, dInd+o1] = Ū[:, o1] ./ Σ_od .- ν_od .- PMM[dInd+o1]
        end
    end


    @. G[:, end] = MDD - PMM[end]
end