function SmoothDirac(β, x)
    return exp(-(x / β)^2) / (β * sqrt(pi))
end

function SmoothOneSidedDirac(β, x)
    if x<0
        return 0
    else
        return 2*exp(-(x / β)^2) / (β * sqrt(pi))
    end
end