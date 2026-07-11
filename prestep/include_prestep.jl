
include("master_prestep.jl") # run prestep assuming F* = Frechet  

include("iterWagesPreStep!.jl") # solve for w, with lambda fixed (instead of A etc)
include("computeGamma.jl") # gamma function for price index