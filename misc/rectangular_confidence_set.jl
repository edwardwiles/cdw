# Rectangular CS assuming Gaussianity, as implemented by CC in thge CSW example
function rectangular_confidence_set(V, α)

    l = size(V)[1]
#=
    VEig = eigen(V)

    λ = VEig.values
    @show  λ

    for i =0:l-1
        λ[l-i] =   λ[l-i] < 0 ? 0.99*λ[l-i+1] : λ[l-i]
    end
    
    @show  λ

    Λ = VEig.vectors

    V_SPD = Λ*Diagonal(λ)*inv(Λ)

    
    #S = Matrix(cholesky(Hermitian((V_SPD))).L)
    #s = sqrt.(diag(Hermitian((V_SPD))))

   =#
    S = Matrix(cholesky(V; check=false).L)
    s = sqrt.(diag(V))


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