function MartingalDifferenceDivergence2(ν, τ, D)

    # function to calculate the martingale difference divergence as per Su and Zheng (2017)
    # https://www.sciencedirect.com/science/article/pii/S0165176517301805?via%3Dihub

    # should make the implementation more efficient

    ΔΔlnν = doubleDiff(ν)
    ΔΔlnτ = doubleDiff(τ)

    meanν = 0
    for o = 2:D
        meanν += ΔΔlnν[o, 1]
        for d = 3:D
            meanν += ΔΔlnν[o, d]
        end
    end

    meanν /= (D - 1)^2

    ΔΔlnν = ΔΔlnν .- meanν

    MDD = 0.0

    for o = 1:D
        for d = 1:D
            for o1 = 1:D
                for d1 = 1:D
                    if o != o1 || d != d1
                        MDD += -abs(ΔΔlnτ[o, d] - ΔΔlnτ[o1, d1]) * ΔΔlnν[o, d] * ΔΔlnν[o1, d1]
                    end
                end
            end
        end
    end

    MDD /= D^4 - D^2

    return MDD^2
end