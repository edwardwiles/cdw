function within_transform_rect(z::AbstractMatrix)
    # Two-way (origin=row, destination=col) fixed-effects "within" residual of log(z), for a
    # Do x Dd RECTANGULAR panel (Do origins, Dd destinations -- need not be equal):
    #   z̃_od = ln z_od - mean_o(ln z) - mean_d(ln z) + grand_mean(ln z)
    # (subtract both margins, ADD BACK the grand mean, each margin divided by ITS OWN dimension).
    # By Frisch–Waugh–Lovell, the moment Σ_od (within lnτ)·(within lnA) = 0 reproduces EXACTLY
    # the coefficient of an OLS gravity regression with origin + destination fixed effects
    # (verified numerically, gravity_check.jl; square case cross-checked bit-exact against the
    # old withinTransform). eltype-generic so it passes ForwardDiff Duals. Reduces to the old
    # square-only formula exactly when Do==Dd -- see withinTransform below.
    lz = log.(z)
    Do, Dd = size(lz)
    return lz .- (sum(lz, dims = 2) ./ Dd) .- (sum(lz, dims = 1) ./ Do) .+ (sum(lz) / (Do * Dd))
end

using LinearAlgebra: qr

"""
    within_transform_masked(z::AbstractMatrix, mask::AbstractMatrix{Bool}) -> Matrix

Exact two-way (origin, destination) fixed-effects OLS residual of log(z), restricted to the cells
where `mask` is `true` (an INCOMPLETE/unbalanced panel -- e.g. excluding the domestic diagonal
`o==d`). `within_transform_rect`'s closed-form row/col/grand-mean formula is only valid for a
COMPLETE panel; it has no notion of dropping individual cells. This computes the exact OLS
estimator via an explicit origin+destination dummy-variable regression (QR-solved once, D=20 is
tiny), which is the same estimator a two-way-FE regression package (e.g. Stata's reghdfe) converges
to on an unbalanced panel, by construction -- NOT an approximation or an iterative-tolerance result.

Masked-out cells get a residual of exactly `0.0` (excluded from the fit AND from any downstream
moment/constraint that sums `mask_z .* other`, since `false` mask entries never entered the
regression). Reduces to `within_transform_rect(z)` on every unmasked cell when `mask` is all-`true`
(same OLS problem, restated via dummies instead of the closed-form projector).

eltype-generic in `z` (works for `Float64` data or `ForwardDiff.Dual` -- the design matrix is a
plain `Float64` QR factorization of the fixed dummy structure; solving `F \\ y` for a `Dual` `y` is
ordinary linear algebra, so this stays differentiable for live (non-data) matrices like `AodPow`).
"""
function within_transform_masked(z::AbstractMatrix, mask::AbstractMatrix{Bool})
    Do, Dd = size(z)
    size(mask) == (Do, Dd) || throw(DimensionMismatch("within_transform_masked: mask size $(size(mask)) != z size $(size(z))"))
    lz = log.(z)
    T = eltype(lz)
    idx = findall(mask)
    N = length(idx)
    N > 0 || throw(ArgumentError("within_transform_masked: mask excludes every cell"))
    # design: intercept + (Do-1) origin dummies + (Dd-1) destination dummies (drop first level of
    # each for identification, standard dummy-variable-regression two-way-FE convention)
    X = zeros(Float64, N, 1 + (Do - 1) + (Dd - 1))
    y = Vector{T}(undef, N)
    @inbounds for (i, ci) in enumerate(idx)
        o, d = ci[1], ci[2]
        y[i] = lz[o, d]
        X[i, 1] = 1.0
        o > 1 && (X[i, 1 + (o - 1)] = 1.0)
        d > 1 && (X[i, 1 + (Do - 1) + (d - 1)] = 1.0)
    end
    F = qr(X)
    beta = F \ y
    fitted = X * beta
    resid = y .- fitted
    out = zeros(T, Do, Dd)
    @inbounds for (i, ci) in enumerate(idx)
        out[ci] = resid[i]
    end
    return out
end

"Square-panel alias, kept for callers that predate the rectangular generalization; bit-identical to within_transform_rect when size(z,1)==size(z,2)."
withinTransform(z) = within_transform_rect(z)

function doubleDiff(z)
    # computes Delta Delta of variable z, see theory note
    #deltaZ = (log.(z[:, :]) .- log.(z[1, :])) .- (log.(z[:, 2]) .- log.(z[1, 2]))
    D = size(z,1)
    deltaZ = zeros(eltype(z), D, D)   # eltype(z) so ForwardDiff Duals pass through (autodiff path)
    for o=1:D
        @.deltaZ[o,:] = (log.(z[o, :]) .- log.(z[1, :])) .- (log.(z[o, 2]) .- log.(z[1, 2]))
    end

    return deltaZ
end

function doubleDiff(z, d1)
    # computes Delta Delta of variable z, see theory note 
    deltaZ = (log.(z[:, :]) .- log.(z[1, :])) .- (log.(z[:, d1]) .- log.(z[1, d1]))
    return deltaZ
end

function doubleDiffLinear(z)
    # computes Delta Delta of variable z, see theory note 
    #deltaZ = (z[:, :] .- z[1, :]) .- (z[:, 2] .- z[1, 2])
    
    D = size(z,1)
    deltaZ = zeros(D,D)
    for o=1:D
        @.deltaZ[o,:] = (z[o, :] .- z[1, :]) .- (z[o, 2] .- z[1, 2])  
    end

    return deltaZ
end

function doubleDiff_grad(z)
    # computes Delta Delta of variable z, see theory note 
    D = size(z,1)
    deltaZ_grad = zeros(D,D,D,D)

    @.deltaZ_grad[:,:,1,2] += 1/z[1,2] 
    for o =1:D
        @. deltaZ_grad[o,:,o,2] += -1/z[o,2] 
        @. deltaZ_grad[:,o,1,o] += -1/z[1,o] #deltaZ_grad[:,d,1,d] += -1/z[1,d]
        for d=1:D
            deltaZ_grad[o,d,o,d] += 1/z[o,d] 
        end
    end
    return deltaZ_grad
end

function doubleDiffLinear_grad(z)
    # computes Delta Delta of variable z, see theory note 
    D = size(z,1)
    deltaZ_grad = zeros(D,D,D,D)
    
    @.deltaZ_grad[:,:,1,2] += 1
    for o =1:D
        @. deltaZ_grad[o,:,o,2] += -1
        @. deltaZ_grad[:,o,1,o] += -1 #deltaZ_grad[:,d,1,d] += -1
        for d=1:D
            #deltaZ_grad[o,d,1,d] += -1
            deltaZ_grad[o,d,o,d] += 1
        end
    end
    return deltaZ_grad
end