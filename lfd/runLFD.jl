function runLFD(globParams)
    # this function takes the LFD RN derivative and creates PDF/ CDF and correlation graphs
    # U realizations are not saved, becuase they are potentially large files, so we re-generate them here assuming we are using the same seed etc.
    @unpack θHat, σHat, W, baseIndex, server, fakeData, DFake, seed,
    counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder,momentOrderForBaseIndex, useParallel, InitDistributionType, InitDistributionCorr, InitDistributionParam, usePMM, useCDFforMarginalMatching, ForceFrechetMarginal, stratifiedSampling, useIndependentCFDs, IndMomentOrder, importanceSampling, importanceSamplingFactor = globParams

    λData, LData, τData = importData(server, user, fakeData, DFake, seed)
    τPrime, LPrime = defineCounter(τData, LData, counterType)
    D = length(LData)

    # get the pre step values for optimiser starting point, using closed form Frechet 
    preStepOutput = preStep(LData, LPrime, τData, τPrime, λData, θHat, σHat, baseIndex, counterType)


    # set seed
    Random.seed!(seed)

    # construct matrix of draws from exp(1)
    U = zeros(W, D * D)
    ImportanceSamplingWeight = ones(W)

    if importanceSampling == 1
        genExpRandsImportanceSampling!(U, importanceSamplingFactor, ImportanceSamplingWeight)
    elseif stratifiedSampling == 1
        genExpRandsStratified!(U)
    else
        genExpRands!(U)
    end
    
    @unpack μHat, wHat, cHat, λPrime, wPrimeHat, γHat, γPrimeHat = preStepOutput

    # Do μHat*(1-σHat) Moment Matching. This is the U power that enters the price expression
    doPriceMomentMatching = true
    Γ_μ_σ = gamma(1 + μHat * (1 - σHat))
    if doPriceMomentMatching && stratifiedSampling == 0 && importanceSampling == 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D
                μ_σ_moment = 0
                for ω = 1:W
                    μ_σ_moment += ((U[ω, o1])^(μHat * (1 - σHat))) / W
                end
                moment_matching_ratio = (Γ_μ_σ / μ_σ_moment)^(1 / (μHat * (1 - σHat)))
                @show moment_matching_ratio
                @. U[:, o1] = U[:, o1] .* (Γ_μ_σ / μ_σ_moment)
            end
        end
    end

    # we keep a copy of U without exponents or scaling by cHat to calculate E[U] later
    Ū = zeros(W, D * D, 1)
    Half_W = floor(Int, W / 2)

    StrataWeight = zeros(W)
    if importanceSampling == 1
        @. StrataWeight[:] = ImportanceSamplingWeight[:]
    elseif stratifiedSampling == 1
        @. StrataWeight[1:Half_W] = 0.1/0.5
        @. StrataWeight[Half_W+1:W] = 0.9/0.5
    else
        @. StrataWeight[:] = 1.0
    end

    Ū[:, :, 1] = U[:, :]

    @show Dates.format(now(), "HH:MM") # print time 

    #### copy here the name of the file containing the counterfactual bounds
    sourcefilename = "Counter_1_countries_4_baseI2_sGrav0_lGrav0_Marg0_NoSc1_ind0_order5_baseOrder50useCDF_1ForceFrechet_0stratify_0IndCDF_0IndMO_5ISampling_1ISF_2_Frechet_4-3-13.csv"
    ### use the naming convention to get the LFD_up and LFD_low file paths
    LFD_upper = readdlm(string(folderData, "/LFD_up_", sourcefilename), ',') # import data from csv
    LFD_lower = readdlm(string(folderData, "/LFD_low_", sourcefilename), ',') # import data from csv







    u = range(0, 2, length=100)
    u_large = range(0, 50, length=100)

    #δ_grid = [0.01, 0.1, 0.5, 1, 2]
    δ_grid = [0.1, 1]

    for i = 1:length(δ_grid)
        marg_cdf = MarginalPricesCDF(u, baseIndex, i, 0)
        marg_cdf_up = MarginalPricesCDF(u, baseIndex, i, 1)
        marg_cdf_low = MarginalPricesCDF(u, baseIndex, i, -1)


        marg_pdf = MarginalPricesPDF(u, baseIndex, i, 0)
        marg_pdf_up = MarginalPricesPDF(u, baseIndex, i, 1)
        marg_pdf_low = MarginalPricesPDF(u, baseIndex, i, -1)

        savefig(plot(xlabel="Price", ylabel="CDF", u, [marg_cdf[:, 1] marg_cdf_up[:, 1] marg_cdf_low[:, 1]], label=["P" "P+" "P-"], title=string("δ=", δ_grid[i])), string("marginalCDF_", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
        savefig(plot(xlabel="Price", ylabel="CDF", u, [marg_cdf[:, 2] marg_cdf_up[:, 2] marg_cdf_low[:, 2]], label=["PRW" "PRW+" "PRW-"], title=string("δ=", δ_grid[i])), string("marginalCDFRW", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
        savefig(plot(xlabel="Price", ylabel="CDF", u, [marg_cdf[:, 3] marg_cdf_up[:, 3] marg_cdf_low[:, 3]], label=["P/PRW" "P+/PRW+" "P-/PRW-"], title=string("δ=", δ_grid[i])), string("marginalCDFPratio", "delta", δ_grid[i], "_", sourcefilename, "_.png"))

        savefig(plot(xlabel="Price", ylabel="PDF", u[1:end-1], [marg_pdf[:, 1] marg_pdf_up[:, 1] marg_pdf_low[:, 1]], label=["P" "P+" "P-"], title=string("δ=", δ_grid[i])), string("marginalPDF", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
        savefig(plot(xlabel="Price", ylabel="PDF", u[1:end-1], [marg_pdf[:, 2] marg_pdf_up[:, 2] marg_pdf_low[:, 2]], label=["PRW" "PRW+" "PRW-"], title=string("δ=", δ_grid[i])), string("marginalPDFRW", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
        savefig(plot(xlabel="Price", ylabel="PDF", u[1:end-1], [marg_pdf[:, 3] marg_pdf_up[:, 3] marg_pdf_low[:, 3]], label=["P/PRW" "P+/PRW+" "P-/PRW-"], title=string("δ=", δ_grid[i])), string("marginalPDFPratio", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
    end


    for i = 1:length(δ_grid)
        marg_cdf = MarginalPricesCDF(u_large, baseIndex, i, 0)
        marg_cdf_up = MarginalPricesCDF(u_large, baseIndex, i, 1)
        marg_cdf_low = MarginalPricesCDF(u_large, baseIndex, i, -1)


        marg_pdf = MarginalPricesPDF(u_large, baseIndex, i, 0)
        marg_pdf_up = MarginalPricesPDF(u_large, baseIndex, i, 1)
        marg_pdf_low = MarginalPricesPDF(u_large, baseIndex, i, -1)

        savefig(plot(xlabel="Price", ylabel="CDF", u_large, [marg_cdf[:, 1] marg_cdf_up[:, 1] marg_cdf_low[:, 1]], label=["P" "P+" "P-"], title=string("δ=", δ_grid[i])), string("large marginalCDF_", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
        savefig(plot(xlabel="Price", ylabel="CDF", u_large, [marg_cdf[:, 2] marg_cdf_up[:, 2] marg_cdf_low[:, 2]], label=["PRW" "PRW+" "PRW-"], title=string("δ=", δ_grid[i])), string("large marginalCDFRW", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
        savefig(plot(xlabel="Price", ylabel="CDF", u_large, [marg_cdf[:, 3] marg_cdf_up[:, 3] marg_cdf_low[:, 3]], label=["P/PRW" "P+/PRW+" "P-/PRW-"], title=string("δ=", δ_grid[i])), string("large marginalCDFPratio", "delta", δ_grid[i], "_", sourcefilename, "_.png"))

        savefig(plot(xlabel="Price", ylabel="PDF", u_large[1:end-1], [marg_pdf[:, 1] marg_pdf_up[:, 1] marg_pdf_low[:, 1]], label=["P" "P+" "P-"], title=string("δ=", δ_grid[i])), string("large marginalPDF", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
        savefig(plot(xlabel="Price", ylabel="PDF", u_large[1:end-1], [marg_pdf[:, 2] marg_pdf_up[:, 2] marg_pdf_low[:, 2]], label=["PRW" "PRW+" "PRW-"], title=string("δ=", δ_grid[i])), string("large marginalPDFRW", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
        savefig(plot(xlabel="Price", ylabel="PDF", u_large[1:end-1], [marg_pdf[:, 3] marg_pdf_up[:, 3] marg_pdf_low[:, 3]], label=["P/PRW" "P+/PRW+" "P-/PRW-"], title=string("δ=", δ_grid[i])), string("large marginalPDFPratio", "delta", δ_grid[i], "_", sourcefilename, "_.png"))
    end


    y0 = mCDF(u_large, 2, 2, 1, 0)

    for i = 1:length(δ_grid)
        y11l = mCDF(u_large, 1, 1, i, -1)
        y11u = mCDF(u_large, 1, 1, i, 1)
        y12u = mCDF(u_large, 1, 2, i, 1)
        y22u = mCDF(u_large, 2, 2, i, 1)
        y32u = mCDF(u_large, 3, 2, i, 1)
        y42u = mCDF(u_large, 4, 2, i, 1)
        y12l = mCDF(u_large, 1, 2, i, -1)
        y22l = mCDF(u_large, 2, 2, i, -1)
        y32l = mCDF(u_large, 3, 2, i, -1)
        y42l = mCDF(u_large, 4, 2, i, -1)

        savefig(
            plot(xlabel="Unscaled U", ylabel="CDF", u_large, [y0 y12l y22l y32l y42l y11l],
                label=["U22" "U12-" "U22-" "U32-" "U42-" "U11-"],
                title=string("Marginals for lower bound δ = ", δ_grid[i])),
            string("large_lower_marginals_delta_", δ_grid[i], "_", sourcefilename, ".png")
        )

        savefig(plot(xlabel="Unscaled U", ylabel="CDF", u_large, [y0 y12u y22u y32u y42u y11u], label=["U22" "U12+" "U22+" "U32+" "U42+" "U11+"], title=string("Marginals for upper bound δ = ", δ_grid[i])), string("large_upper_marginals_delta_", δ_grid[i], "_", sourcefilename, ".png"))

        savefig(plot(xlabel="Unscaled U", ylabel="CDF", u_large, [y0 y22u y22l], label=["U22" "U22+" "U22-"], title=string("Marginals for reference country δ = ", δ_grid[i])), string("large_marginals_delta_", δ_grid[i], "_", sourcefilename, ".png"))
    end

    y0 = mCDF(u, 2, 2, 1, 0)
    for i = 1:length(δ_grid)
        y11l = mCDF(u, 1, 1, i, -1)
        y11u = mCDF(u, 1, 1, i, 1)
        y12u = mCDF(u, 1, 2, i, 1)
        y22u = mCDF(u, 2, 2, i, 1)
        y32u = mCDF(u, 3, 2, i, 1)
        y42u = mCDF(u, 4, 2, i, 1)
        y12l = mCDF(u, 1, 2, i, -1)
        y22l = mCDF(u, 2, 2, i, -1)
        y32l = mCDF(u, 3, 2, i, -1)
        y42l = mCDF(u, 4, 2, i, -1)
        savefig(plot(xlabel="Unscaled U", ylabel="CDF", u, [y0 y12l y22l y32l y42l y11l], label=["U22" "U12-" "U22-" "U32-" "U42-" "U11-"], title=string("Marginals for lower bound δ = ", δ_grid[i])), string("lower_marginals_delta_", δ_grid[i], "_", sourcefilename, "_.png", ".png"))
        savefig(plot(xlabel="Unscaled U", ylabel="CDF", u, [y0 y12u y22u y32u y42u y11u], label=["U22" "U12+" "U22+" "U32+" "U42+" "U11+"], title=string("Marginals for upper bound δ = ", δ_grid[i])), string("upper_marginals_delta_", δ_grid[i], "_", sourcefilename, "_.png", ".png"))
        savefig(plot(xlabel="Unscaled U", ylabel="CDF", u, [y0 y22u y22l], label=["U22" "U22+" "U22-"], title=string("Marginals for reference country δ = ", δ_grid[i])), string("marginals_delta_", δ_grid[i], "_", sourcefilename, "_.png", ".png"))
    end

    for i = 1:length(δ_grid)

        M = zeros(D^2, D^2)

        M = correlationMatrix(i, 0)
        savefig(heatmap(M, fc=cgrad([:white, :dodgerblue4])),
            string("central_correlation_delta_", δ_grid[i], "_", sourcefilename, ".png"))
        writedlm(string("correlation_matrix_central_delta_", δ_grid[i], "_", sourcefilename, ".csv"), M, ',')

        M2 = zeros(D^2, D^2)
        M2 = correlationMatrix(i, 1)
        savefig(heatmap(M2, fc=cgrad([:white, :dodgerblue4])),
            string("upper_bound_correlation_delta_", δ_grid[i], "_", sourcefilename, ".png"))
        writedlm(string("correlation_matrix_upper_bound_delta_", δ_grid[i], "_", sourcefilename, ".csv"), M2, ',')

        M3 = zeros(D^2, D^2)
        M3 = correlationMatrix(i, -1)
        savefig(heatmap(M3, fc=cgrad([:white, :dodgerblue4])),
            string("lower_bound_correlation_delta_", δ_grid[i], "_", sourcefilename, ".png"))
        writedlm(string("correlation_matrix_lower_bound_delta_", δ_grid[i], "_", sourcefilename, ".csv"), M3, ',')
    end


    @show Dates.format(now(), "HH:MM") # print time 

end