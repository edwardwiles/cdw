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