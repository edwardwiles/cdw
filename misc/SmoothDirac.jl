function SmoothDirac(β, x)
    return exp(-(x / β)^2) / (β * sqrt(pi))
end