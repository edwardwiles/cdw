function MartingalDifferenceDivergence(ν, Mτ, D)

    # function to calculate the martingale difference divergence as per Su and Zheng (2017)
    # https://www.sciencedirect.com/science/article/pii/S0165176517301805?via%3Dihub

    # this matrix version sould be faster

    ΔΔlnν = doubleDiff(ν)

    meanν = 0
    for o = 2:D
        meanν += ΔΔlnν[o, 1]
        for d = 3:D
            meanν += ΔΔlnν[o, d]
        end
    end

    meanν /= (D - 1)^2

    ΔΔlnν = ΔΔlnν .- meanν

    longΔΔlnν = reshape(ΔΔlnν, (D^2, 1))

    MDD = dot(longΔΔlnν, Mτ, longΔΔlnν)

    MDD /= D^4 - D^2

    return MDD^2
end