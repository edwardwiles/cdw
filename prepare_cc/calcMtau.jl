function calcMτ(τ, D)
    #calculates the Mτ for MDD usage 
    Mτ = zeros(D^2, D^2)
    ΔΔlnτ = doubleDiff(τ)
    for o1 = 1:D
        for d1 = 1:D
            od1 = o1 + (d1 - 1) * D
            for o2 = 1:D
                for d2 = 1:D
                    od2 = o2 + (d2 - 1) * D
                    if o2 != o1 || d2 != d1
                        Mτ[od1, od2] = -abs(ΔΔlnτ[o1, d1] - ΔΔlnτ[o2, d2])
                    end
                end
            end
        end
    end
    return Mτ
end