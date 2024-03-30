
# smoothing functions to make outer loop amenable to auto diff 
include("smoothMinIndNew!.jl") # smooth version of min function 
include("softmax.jl") # smooth version of max function 
include("SmoothDirac.jl") #smooth version of indicator function 

include("MartingalDifferenceDivergence2.jl") # not sure what this is?
include("MartingalDifferenceDivergence.jl") # XXX Habib -- is this just an old version of above? if so, can we delete it?

include("doubleDiff.jl") # double difference function for gravity regressions 

include("checknan.jl") # not sure what this is for?

include("checkParams.jl") # throws an error if trying to impose impossible combination of restrictions 