
function master_setup(globalParams)

    @unpack server, user, fakeData, DFake, counterType, seedFakeData = globalParams 

    setwd(server, user) # set working and data directories

    data = importData(fakeData, DFake, seedFakeData)

    counters = defineCounter(data.τData, data.LData, counterType)    
    D = length(data.LData)    

    # set seed
    # Random.seed!(seed)

    return setup_output = (data = data, counters = counters, D = D)

end 

