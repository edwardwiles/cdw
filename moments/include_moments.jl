
# gravity moments 
include("gravityMoment!.jl") # zero correlation between tau and c (DOES NOT DEMEAN TAU--IS THIS JUST WRONG?)
include("GravityMomentFirstApproach!.jl") # constructs the moment that ΔΔ E[ln U] = ΔΔ ln cHat + ΔΔ E[ln Ū] is mean independent of ΔΔ lnτ
include("newGravityMoment!.jl") # zero correlation between tau and E[U]
include("strongGravityMomentold2!.jl") # constructs the moment that ΔΔ E[ln U] is mean independent of ΔΔ lnτ (XXX Habib should this be deleted / archived?)
include("localGravityMoment!.jl") # local derivative moment from the notes 
include("localGravityCrossMoment!.jl") # local cross-derivative = 0 moments to fully satisfy ACR R3 

# common marginals moments 
include("sameMarginalsMoment!.jl") # impose E[U^k] equal for all od up for K 
include("sameMarginalsMomentOld!.jl") # old version of above? (XXX Habib should this be deleted?)
include("sameMarginalsMomentCDF!.jl") # impose E[1{U < k}] equal for all od for some list k_1, k_2,... 

# independence 
include("independenceMoment!.jl") # XXX Habib I can't tell which method this corresponds to in the notes?
include("independenceMomentold!.jl") # old version of above? (XXX Habib should this be deleted?)
include("uncorrelationMoment!.jl") # impose E[U_od U_o'd'] = E[U^2]

# overall 
include("hFunction.jl") # function to compute EK object such that E[h(U,theta)] = lambda 
include("moments!.jl") # master function that calls all of the above to fill in all of the moments 