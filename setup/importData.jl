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
    elseif fakeData == 3
        # Noah's real D=20 dataset (prepare_clean_data/objects_for_julia, 2026-03-04 export),
        # pi.csv/tau.csv/L.csv match the wiodPiMatrix/ekTau/wdiL convention above but under new
        # filenames/location -- ported from trade_robustness_modular_perf @ 778c362
        # (feature/sequential-inversion-perf), where this branch was first added.
        realDataDir = get(ENV, "REAL_DATA_DIR", joinpath(@__DIR__, "..", "real_data", "noah_D20"))
        lambdaData = readdlm(joinpath(realDataDir, "pi.csv"), ',')
        LData = readdlm(joinpath(realDataDir, "L.csv"), ',')
        tauData = readdlm(joinpath(realDataDir, "tau.csv"), ',')
        # Rescale labor/size by a constant (millions) purely for numerical conditioning of the wage
        # calibration -- confirmed a no-op on every downstream result: iterWagesPreStep!'s fixed
        # point is exactly invariant to uniform L-rescaling (the constant cancels in
        # `(lambda*(w0.*L))./L`), and every use of computeGamma's L-dependent gammaHat/gammaPrimeHat
        # downstream is as a RATIO (gammaPrime/gamma, or A-column normalizations), so a uniform
        # per-country scale factor in gammaHat cancels there too.
        LData = LData ./ 1e6
    end

    return data = (λData = lambdaData, LData = LData, τData = tauData)

end