#abstract exponentiation function, subtract max for numerical stability
_exp(x::AbstractVecOrMat) = exp.(x .- maximum(x))
#abstract exponentiation function, subtract max for numerical stability and scale by theta
_exp(x::AbstractVecOrMat, θ::AbstractFloat) = exp.((x .- maximum(x)) * θ)
#softmax algorithm expects stablized eponentiated e
_sftmax(e::AbstractVecOrMat, d::Integer) = (e ./ sum(e, dims = d))

# top level softmax function
function softmax(X::AbstractVecOrMat{T}, dim::Integer)::AbstractVecOrMat where T <: AbstractFloat
    _sftmax(_exp(X), dim)
end

function softmax(X::AbstractVecOrMat{T}, dim::Integer, θ::Float64)::AbstractVecOrMat where T <: AbstractFloat
    _sftmax(_exp(X, θ), dim)
end