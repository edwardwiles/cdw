
# smoothing functions to make outer loop amenable to auto diff 
include("smoothMinIndNew!.jl") # smooth version of min function 
include("softmax.jl") # smooth version of max function 
include("SmoothDirac.jl") #smooth version of dirac delta function 

include("indicative.jl") # calculates the indicative function 

include("doubleDiff.jl") # double difference function for gravity regressions 

include("checkParams.jl") # throws an error if trying to impose impossible combination of restrictions 