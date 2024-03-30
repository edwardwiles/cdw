function importData(fakeData, DFake, seed)

    if fakeData == 1
        Random.seed!(seed) # set different seed for true data generation
        (lambdaData, LData, tauData) = createFakeData(DFake, 1) # generate fake data
    elseif fakeData == 2
        Random.seed!(seed) # set different seed for true data generation
        (lambdaData, LData, tauData) = createFakeDataGeneric(DFake, 1) # generate fake data
    elseif fakeData == 0
        # import data
        lambdaData = readdlm(string(folderData, "/wiodPiMatrix.csv"), ',')
        LData = readdlm(string(folderData, "/wdiL.csv"), ',')
        tauData = readdlm(string(folderData, "/ekTau.csv"), ',') # import data from csv
    end

    return data = (λData = lambdaData, LData = LData, τData = tauData)

end