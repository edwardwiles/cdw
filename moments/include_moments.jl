
# gravity moments 
include("GravityMomentFirstApproach!.jl") # constructs the moment that ΔΔ E[ln U] = ΔΔ ln cHat + ΔΔ E[ln Ū] is mean independent of ΔΔ lnτ
include("newGravityMoment!.jl") # zero correlation between tau and E[U]
include("localGravityMoment!.jl") # local derivative moment from the notes 
include("localGravityCrossMoment!.jl") # local cross-derivative = 0 moments to fully satisfy ACR R3 


# independence 
include("independenceMoment!.jl") # independence of CDFs  

# overall 
include("hFunction.jl") # function to compute EK object such that E[h(U,theta)] = lambda 
include("moments!.jl") # master function that calls all of the above to fill in all of the moments 