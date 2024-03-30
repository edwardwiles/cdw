
include("master_setup.jl")

include("setwd.jl") # set working directory 
include("importData.jl") # imports data 
include("createTradeCosts!.jl")
include("iterWagesTheory!.jl") # solves for w vector using Frechet closed form solutions for trade shares 
include("defineCounter.jl") # defines tau prime 
include("createFakeData.jl") # creates fake data from Frechet 
include("createFakeDataGeneric.jl") # creates fake data from other distributions 

