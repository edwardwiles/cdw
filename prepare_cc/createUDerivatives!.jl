
function createUDerivatives!(U, prestep_output, params)

    @unpack W, D, σHat, θConstant = params
    @unpack μHat, cHat = prestep_output

    # we keep a copy of U without exponents or scaling by cHat
    Ū = zeros(W, D)

    Ū[:, :] = U[:, :]

    if θConstant == 1
        @. U[:] = U[:] .^ (-μHat) # if theta doesn't vary, then much faster to precalculate this matrix
        # reduction in computation time due to non-integer exponent substantially dominates higher memory usage
    end

    Uσ = U .^ (1 - σHat) # precalculate

    return Ū, Uσ

end
