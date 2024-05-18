function doubleDiff(z)
    # computes Delta Delta of variable z, see theory note 
    deltaZ = (log.(z[:, :]) .- log.(z[1, :])) .- (log.(z[:, 2]) .- log.(z[1, 2]))
    return deltaZ
end

function doubleDiff(z, d1)
    # computes Delta Delta of variable z, see theory note 
    deltaZ = (log.(z[:, :]) .- log.(z[1, :])) .- (log.(z[:, d1]) .- log.(z[1, d1]))
    return deltaZ
end

function doubleDiffLinear(z)
    # computes Delta Delta of variable z, see theory note 
    deltaZ = (z[:, :] .- z[1, :]) .- (z[:, 2] .- z[1, 2])
    return deltaZ
end