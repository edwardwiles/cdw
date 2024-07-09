# Rectangular CS assuming Gaussianity, as implemented by CC in thge CSW example
function rectangular_confidence_set(V, α)

    l = size(V)[1]

    VEig = eigen(V)

    λ = VEig.values
    @show  λ

    for i =1:l
        λ[i] =   λ[i] < 0 ? 10^(-6) : λ[i]
    end
    Λ = VEig.vectors

    V_SPD = Λ*Diagonal(λ)*inv(Λ)


    S = Matrix(cholesky(Hermitian((V_SPD))).L)
    s = sqrt.(diag(Hermitian((V_SPD))))

    B = 100000
    z = zeros(l, B)

    Random.seed!(1234567)

    for b in 1:B

        z[:, b] = S * randn(l)
        z[:, b] ./= s

    end

    zz = [maximum(abs.(z[:, b])) for b in 1:B]

    return (quantile(zz, 1 - α), s)

end