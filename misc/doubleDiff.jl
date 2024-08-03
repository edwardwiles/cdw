function doubleDiff(z)
    # computes Delta Delta of variable z, see theory note 
    #deltaZ = (log.(z[:, :]) .- log.(z[1, :])) .- (log.(z[:, 2]) .- log.(z[1, 2]))
    D = size(z,1)
    deltaZ = zeros(D,D)
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