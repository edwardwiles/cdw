import Pkg
Pkg.activate("GravityProject")

#Pkg.add("path="https://github.com/timothymchristensen/CounterfactualSensitivity.jl.git")
#Pkg.add("NLsolve")
#Pkg.add("Optim")

#Pkg.add(path="https://github.com/JuliaNLSolvers/NLsolve.jl")
#Pkg.add(path="https://github.com/JuliaNLSolvers/Optim.jl")
#Pkg.add(path="https://github.com/mauro3/Parameters.jl")
#Pkg.add("BlackBoxOptim")
#Pkg.add("SortingAlgorithms")
#Pkg.add("Bigsimr")

#Pkg.add("MemPool")

#Pkg.add("KNITRO")
#Pkg.add("Plots")
#Pkg.add("InvertedIndices")

#Pkg.upgrade_manifest()
#Pkg.resolve()
#Pkg.update()

using KNITRO
using Random, NLsolve, Optim, SpecialFunctions, DelimitedFiles, Dates, LinearAlgebra, Parameters

using Distributions, Statistics, Bigsimr, Plots, InvertedIndices

using MemPoo
#so code doesn't throw error messages as it tries to display figures.
ENV["GKSwstype"] = "nul"

include("softmax.jl")
include("CounterfactualSensitivity_local.jl")
include("Psi.jl")
include("ObjectiveBundle.jl")
include("KLObjectiveBundle.jl")
include("PsiObjectiveBundle.jl")
include("KLObjectiveBundleConditional.jl")
include("PsiObjectiveBundleConditional.jl")
include("inner_loop_functions.jl")
include("outer_loop_functions.jl")
include("local_sensitivity.jl")


# functions
function setwd(server)
    # sets working directories
    if server == 1 # if working on the econ servers
        cd(raw"/bbkinghome/mhansari/Robustness")
        global folderData = "/bbkinghome/mhansari/Robustness"
    elseif server == 0 # if working on laptop 
        cd(raw"C:/2. MIT/Model Robustness")
        global folderData = "C:/2. MIT/Model Robustness/Data"
    end
end

function checknan(x)
    for i in firstindex(x):64:(lastindex(x)-63)
        s = zero(eltype(x))
        for j in 0:63
            s += x[i+j] * 0
        end
        !isfinite(s) && return false
    end
    return all(isfinite, @view x[max(end - 64, begin):end])
end

function iterWagesTheory!(w0, L, A, tau, theta, lambda)
    # find market clear wages under Frechet by iteration

    # parameters for iteration 
    tol = 1e-12
    maxIter = 40000
    ϵ = 0.2
    diff = tol + 1
    iter = 1

    w1 = copy(w0)

    while diff > tol && iter < maxIter

        # update wage guess
        w0[:] = w0[:] * (1 - ϵ) + w1[:] * ϵ

        # see note for algebra 
        phi = A .* (tau .* w0) .^ (-theta)

        lambda[:, :] .= phi ./ (sum(phi, dims=1))
        w1[:] = lambda * (w0 .* L) ./ L # implied wages 

        diff = maximum(abs.(w1[:] .- w0[:]))

        if mod1(iter, 500) == 1
            @show diffTheory = diff # print progress occasionally
        end

        iter += 1

    end

    if iter == maxIter
        @show -9999999999999999999 # if failed to converge, print error 
    end

    w0[:] ./= w0[1] # normalise country 1's wage to 1 

end

function importData(server, fakeData, DFake, seed)

    setwd(server) # set working and data directories

    if fakeData == 1
        Random.seed!(seed + 1) # set different seed for true data generation
        (lambdaData, LData, tauData) = createFakeData(DFake, 1) # generate fake data
    elseif fakeData == 2
        Random.seed!(seed + 1) # set different seed for true data generation
        (lambdaData, LData, tauData) = createFakeDataGeneric(DFake, 1) # generate fake data
    elseif fakeData == 0
        # import data
        lambdaData = readdlm(string(folderData, "/wiodPiMatrix.csv"), ',')
        LData = readdlm(string(folderData, "/wdiL.csv"), ',')
        tauData = readdlm(string(folderData, "/ekTau.csv"), ',') # import data from csv
    end

    # set seed
    Random.seed!(seed)

    return lambdaData, LData, tauData

end

function defineCounter(τData, LData, counterType)
    # define counterfactual features of the model
    D = length(LData)
    LPrime = LData  # if using LPrime \neq L, edit here
    # set counterfactual tau
    if counterType == 0 # zero gravity 
        τPrime = ones(D, D)
    elseif counterType == 1 # autarky 
        τPrime = τData .^ Inf
    elseif counterType == 2 # in between custom alternative 
        τPrime = 0.5 .* τData + 0.5 .* ones(D, D) # edit custom alternate matrix here
    elseif counterType == 3 # in between custom alternative 
        τPrime = τData  # edit custom alternate matrix here
    end

    return τPrime, LPrime

end

function createTradeCosts!(tradeCosts, scale, D)
    # Construct trade costs based on distance if using fake data 
    # generate random x and y coords for each country, then compute distances, then scale to trade costs 
    params = rand(D, 2) .* 5
    coordMat = zeros(D * D, 5)
    coordMat[:, 1] = repeat(params[:, 1], inner=D)
    coordMat[:, 2] = repeat(params[:, 2], inner=D)
    coordMat[:, 3] = repeat(params[:, 1], outer=D)
    coordMat[:, 4] = repeat(params[:, 2], outer=D)
    coordMat[:, 5] = exp.(0.1 * sqrt.(((coordMat[:, 1] - coordMat[:, 3]) .^ 2 + (coordMat[:, 2] - coordMat[:, 4]) .^ 2)))
    tradeCosts[:, :] = reshape(coordMat[:, 5], (D, D)) .^ scale # convert distance to trade costs (scale is a param)
    nothing
end

function createFakeData(D, scale)
    # creates a fake dataset where true DGP is Frechet 
    # to use non-Frechet, would need to take draws from alternative DGP and iterate wages without Frechet simplified formulas 

    # create trade cost matrix
    tradeCosts = Array{Float64,2}(undef, D, D) # initialise a matrix
    createTradeCosts!(tradeCosts, scale, D) # creates DxD matrix of taus 

    # define fake "true" parameters
    A = rand(D, D) .+ 1
    A .= A / A[1, :]' # normalise A_{1d} to 1 for all d
    L = rand(D, 1) .+ 1
    theta = 6

    lambda = Array{Float64,2}(undef, D, D)  # initialise

    w = ones(D)
    iterWagesTheory!(w, L, A, tradeCosts, theta, lambda) # solve for wages under Frechet DGP

    # sanity check, but this should return zero as A and tau are independent

    @show MartingalDifferenceDivergence2(A .^ -1, tradeCosts, D)

    return lambda, L, tradeCosts
end

function createFakeDataGeneric(D, scale)
    # creates a fake dataset where true DGP is Frechet 
    # to use non-Frechet, would need to take draws from alternative DGP and iterate wages without Frechet simplified formulas 

    # create trade cost matrix
    tradeCosts = Array{Float64,2}(undef, D, D) # initialise a matrix
    createTradeCosts!(tradeCosts, scale, D) # creates DxD matrix of taus 

    # define fake "true" parameters
    A = rand(D, D) .+ 1
    A .= A / A[1, :]' # normalise A_{1d} to 1 for all d
    L = rand(D, 1) .+ 1
    theta = 6

    lambda_fake = Array{Float64,2}(undef, D, D)  # initialise
    lambda = Array{Float64,2}(undef, D, D)  # initialise

    w0 = ones(D)

    W = 10000 # number of products for calculating the expectations
    U_fake = zeros(W, D)
    Random.seed!(1234)
    genRands!(U_fake, 1, 0.3, 1, 0)

    # parameters for iteration 
    tol = 1e-12
    maxIter = 40000
    ϵ = 0.2
    diff = tol + 1
    iter = 1

    w1 = copy(w0)

    while diff > tol && iter < maxIter

        # update wage guess
        w0[:] = w0[:] * (1 - ϵ) + w1[:] * ϵ

        lambda = (gFunction!(2.5, tradeCosts, U_fake, w0, A .^ (-1.0 / theta), lambda_fake)).expenditure_ratio
        w1[:] = lambda .* (w0 .* L) ./ L # implied wages 

        diff = maximum(abs.(w1[:] .- w0[:]))

        if mod1(iter, 500) == 1
            @show diffTheory = diff # print progress occasionally
        end

        iter += 1

    end

    if iter == maxIter
        @show -9999999999999999999 # if failed to converge, print error 
    end

    w0[:] ./= w0[1] # normalise country 1's wage to 1 



    @show lambda

    return lambda, L, tradeCosts
end

function doubleDiff(z)
    # computes Delta Delta of variable z, see theory note 
    deltaZ = (log.(z[:, :]) .- log.(z[1, :])) .- (log.(z[:, 2]) .- log.(z[1, 2]))
    return deltaZ
end

function doubleDiff(z, d1)
    # computes Delta Delta of variable z, see theory note 
    deltaZ = (log.(z[:, :]) .- log.(z[1, :])) .- (log.(z[:, d1]) .- log.(z[1, d1]))
    return deltaZ
end

function calcMτ(τ, D)
    #calculates the Mτ for MDD usage 
    Mτ = zeros(D^2, D^2)
    ΔΔlnτ = doubleDiff(τ)
    for o1 = 1:D
        for d1 = 1:D
            od1 = o1 + (d1 - 1) * D
            for o2 = 1:D
                for d2 = 1:D
                    od2 = o2 + (d2 - 1) * D
                    if o2 != o1 || d2 != d1
                        Mτ[od1, od2] = -abs(ΔΔlnτ[o1, d1] - ΔΔlnτ[o2, d2])
                    end
                end
            end
        end
    end
    return Mτ
end

function SmoothDirac(β, x)
    return exp(-(x / β)^2) / (β * sqrt(pi))
end

function iterWagesPreStep!(w0, L, lambda)
    # iterate wages again, diff from earlier function is we use lambda (trade shares), not A and phi 
    # see theory note 

    # could we use eigen decomposition of lambda and eigenvector

    tol = 1e-12
    maxIter = 100000
    ϵ = 0.6
    diff = tol + 1
    iter = 1

    w1 = copy(w0)

    while diff > tol && iter < maxIter

        w0[:] = w0[:] * (1 - ϵ) + w1[:] * ϵ
        w1[:] = lambda * (w0 .* L) ./ L

        diff = maximum(abs.(w1[:] .- w0[:]))

        if mod1(iter, 500) == 1
            @show diffPreStep = diff
        end

        iter += 1

    end

    if iter == maxIter
        @show -9999999999999999999
    end

end

function computeGamma(c, tau, w, theta, sigma, L)
    # computes gamma and gamma prime, see theory note 
    Phi = sum(c .* (tau .* w) .^ (-theta), dims=1) # added parentesis just to be sure
    priceIndex = (Phi .^ (-(1 - sigma) / theta)) .* gamma((1 + theta - sigma) / theta)
    gammaHat = (priceIndex' ./ (w .* L)) .^ (1 / sigma)
    return gammaHat
end

function preStep(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType)
    # does the full pre-step to get the initial guess of parameters (psi^*) that corresponds with Frechet 

    D = size(lambda, 1)

    # Step 1: Estimate thetaHat via gravity or prespecified
    if thetaIn == 0 # use gravity to estimate theta if no theta prespecified
        deltaLambda = doubleDiff(lambda)
        deltaTau = doubleDiff(tau)
        thetaHat = -sum(deltaLambda .* deltaTau) ./ sum(deltaTau .* deltaTau)

    elseif thetaIn > 0
        thetaHat = thetaIn
    end

    # Step 2: Estimate baseline wages
    wHat = ones(D)
    iterWagesPreStep!(wHat, L, lambda)
    wHat = wHat ./ wHat[baseIndex] # normalise so wage is 1 for specified base country 

    # Step 3: Estimate cHat
    AHat = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (thetaHat)) .* (lambda ./ lambda[1, :]')
    cHat = AHat .^ (-1) # we define c as 1/A 

    # Step 4: Estimate wPrimeHat
    lambdaPrime = Array{Float64,2}(undef, D, D)  # initialise

    wPrimeHat = ones(D)

    if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
        iterWagesTheory!(wPrimeHat, LPrime, AHat, tauPrime, thetaHat, lambdaPrime)
        wPrimeHat[:] = wPrimeHat ./ wPrimeHat[baseIndex] # normalise 
    end

    # Step 5: Compute gammaHat and gammaPrimeHat
    gammaHat = computeGamma(AHat, tau, wHat, thetaHat, sigma, L)
    gammaPrimeHat = computeGamma(AHat, tauPrime, wPrimeHat, thetaHat, sigma, LPrime)

    print(thetaHat)

    output = (μHat=1 ./ thetaHat,
        wHat=wHat[:],
        cHat=cHat,
        λPrime=lambdaPrime,
        wPrimeHat=wPrimeHat[:],
        γHat=gammaHat[:],
        γPrimeHat=gammaPrimeHat[:])

    return output
end
function preStepGeneralDistribution(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType, distributionType, corr, vol, autocorr=0)

    D = size(lambda, 1)
    W = 1000000 # number of products for calculating the expectations, we use 1 million products so to emulate the "theoretical" quantities.
    # construct matrix of draws from distributionType
    U_init = zeros(W, D * D)
    Random.seed!(2^12)
    genRands!(U_init, distributionType, corr, vol, autocorr, tau)

    return preStepGeneralDistribution(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType, U_init)

end

function preStepGeneralDistribution(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType, U_init)
    # does the full pre-step to get the initial guess of parameters (psi^*) that corresponds with a specific distribution
    # the same function as above. But without the need to re-simulate the Us for the calibration.  

    D = size(lambda, 1)
    W = size(U_init, 1)

    # Step 1: Estimate thetaHat via gravity or prespecified
    if thetaIn == 0 # use gravity to estimate theta if no theta prespecified
        deltaLambda = doubleDiff(lambda)
        deltaTau = doubleDiff(tau)
        thetaHat = -sum(deltaLambda .* deltaTau) ./ sum(deltaTau .* deltaTau)
    elseif thetaIn > 0
        thetaHat = thetaIn
    end

    # Step 2: Estimate baseline wages
    wHat = ones(D)
    iterWagesPreStep!(wHat, L, lambda)
    wHat = wHat ./ wHat[baseIndex] # normalise so wage is 1 for specified base country 

    # Step 3: Estimate cHat

    UPow_init = zeros(W, D * D)
    UPow_init[:] = U_init[:] .^ (-1.0 / thetaHat)

    # use Frechet solution as the initial guess
    AHat_Frechet = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (thetaHat)) .* (lambda ./ lambda[1, :]')
    cHat_Frechet = AHat_Frechet .^ (-1) # we define c as 1/A 


    cHat = ones(D, D)
    # force cHat[1,:] =1
    normalizeAs = true
    if normalizeAs
        for d = 1:D
            function cHat_solver_func!(cHat_x)
                return gFunction_d!(sigma, tau, UPow_init, wHat, vcat(ones(1), cHat_x), lambda, d) # we force the cHat[1,:] =1
            end
            cHat_solver_results = nlsolve(cHat_solver_func!, (cHat_Frechet[2:D, d]) .^ (1.0 / thetaHat)) #, autodiff = :forward
            #@show cHat_solver_results.residual_norm ## check it converged
            cHat[2:D, d] .= abs.(cHat_solver_results.zero)[:]
            #@show cHat[:, d]
        end
    else
        for d = 1:D
            function cHat_solver_func2!(cHat_x)
                return gFunction_d2!(sigma, tau, UPow_init, wHat, cHat_x, lambda, d)
            end
            cHat_solver_results = nlsolve(cHat_solver_func2!, (cHat_Frechet[:, d]) .^ (1.0 / thetaHat)) #, autodiff = :forward
            #@show cHat_solver_results.residual_norm ## check it converged
            cHat[:, d] .= abs.(cHat_solver_results.zero)[:]
            #@show cHat[:, d]
        end
    end
    GC.gc()
    # @show cHat
    # @show lambda
    # @show (gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda)).expenditure_ratio_residual
    # @show (gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda)).trade_share_residual
    # @show (gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda)).expenditure

    # Step 4: Estimate wPrimeHat
    #lambdaPrime = Array{Float64,2}(undef, D, D)  # initialise
    lambdaPrime = zeros(D, D)
    wPrimeHat = ones(D)

    if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
        function wPrimeHat_solver_func!(wHatPrime_x)
            func_output = gFunction!(sigma, tauPrime, UPow_init, wHatPrime_x, cHat, lambda)

            wage_moment = zeros(Float64, D)
            for o = 1:D
                wage_moment[o] = sum(func_output.expenditure_ratio[o, :] .* wHatPrime_x .* LPrime) - wHatPrime_x[o] * LPrime[o]
            end

            return wage_moment
        end

        wPrimeHat_solver_results = nlsolve(wPrimeHat_solver_func!, wHat,)

        @show wPrimeHat_solver_results.residual_norm ## check it converged

        wPrimeHat = abs.(wPrimeHat_solver_results.zero)
        wPrimeHat[:] = wPrimeHat ./ wPrimeHat[baseIndex] # normalise 

        lambdaPrime = (gFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambda)).expenditure_ratio
    else
        for o = 1:D
            lambdaPrime[o, o] = 1
        end
    end

    price_index = ones(D)
    price_index_Prime = ones(D)

    price_index[:] .= ((gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda)).price_index)[:]
    price_index_Prime[:] .= ((gFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambdaPrime)).price_index)[:]

    # Step 5: Compute gammaHat and gammaPrimeHat
    gammaHat = ones(D)
    gammaPrimeHat = ones(D)

    gammaHat[:] = (price_index[:] ./ (wHat[:] .* L[:])) .^ (1.0 / sigma)
    gammaPrimeHat[:] = (price_index_Prime[:] ./ (wPrimeHat[:] .* LPrime[:])) .^ (1.0 / sigma)

    # we were using above cHat^(1/theta), for numerical efficiency. We adjust the power before returning the values.
    @. cHat[:] = cHat[:] .^ (thetaHat)

    # we scale so that cHat represents E[U_od]
    meanU = mean(U_init)
    #@. cHat[:] = cHat[:] .* meanU


    output = (μHat=1 ./ thetaHat,
        wHat=wHat[:],
        cHat=cHat,
        λPrime=lambdaPrime,
        wPrimeHat=wPrimeHat[:],
        γHat=gammaHat[:],
        γPrimeHat=gammaPrimeHat[:])

    return output
end

function genExpRands!(U)
    # draw from exp(1)
    rand!(U)
    for i in 1:length(U)
        U[i] = -log(1 - U[i])
    end
end

function genRands!(U, DistributionType, corr, vol, autocorr, τ=zeros(1))
    D = size(U, 2)
    W = size(U, 1)

    if DistributionType == 0 # exp(1)
        rand!(U)
        for i in 1:length(U)
            U[i] = -log(1 - U[i])
        end
    elseif DistributionType == 1 # lognormal 

        Lognormal_Dist = MvLogNormal(Matrix(1.0I, D, D))

        U[:] = (rand(Lognormal_Dist, W)')[:]

    elseif DistributionType == 2 # T_Dist 

        T_Dist = MvTDist(5, Matrix(1.0I, D, D))

        U[:] = exp.((rand(T_Dist, W)'))[:]
    elseif DistributionType == 3 # This is a flexible class, any (strictly positive) distribution could work 
        # for the list of distributions: https://juliastats.org/Distributions.jl/stable/univariate/ 

        target_corr = corr .* ones(D, D) .+ (sqrt(1 - corr^2) .* Matrix(1.0I, D, D))

        #@show target_corr

        margins = UnivariateDistribution[]

        for o = 1:D
            if mod(o, 3) == 0
                push!(margins, Normal())
            elseif mod(o, 3) == 1
                push!(margins, Cauchy())
            else
                push!(margins, SymTriangularDist())
            end
        end

        #adjusted_corr = pearson_match(target_corr, margins)
        #@show adjusted_corr

        U[:] = (rvec(W, target_corr, margins))[:]
        U[:] = exp.(U)[:]

        #U_ = zeros(W+1, D)

        #U_[:] = (rvec(W+1, adjusted_corr, margins))[:]

        #   # add autocorrelation between products 
        #  for l =1:W
        #     U[l, :] = sqrt(1-autocorr^2) .* U_[l, :]  .+ autocorr .* U_[W+1, :]
        # end

        #@show  cor(U, Pearson)
    elseif DistributionType == 4 # builds a lognormal distribution that is correlated with trade costs

        Lognormal_Dist = MvLogNormal(Matrix(1.0I, D, D))

        U[:] = (rand(Lognormal_Dist, W)')[:]

        Normal_dist = Normal(0, vol^2)
        G = zeros(W)
        G = rand(Normal_dist, W)
        n_countries = size(τ, 2)

        for o = 1:n_countries
            for d = 1:n_countries
                o1 = o + (d - 1) * n_countries
                for ω = 1:W
                    U[ω, o1] *= τ[o, d]^(G[ω]^2)
                end
            end
        end
    elseif DistributionType == 5 # builds a Frechet distribution that is correlated with trade costs

        rand!(U)
        for i in 1:length(U)
            U[i] = -log(1 - U[i])
        end

        G = zeros(W)
        rand!(G)
        n_countries = size(τ, 2)

        for o = 1:n_countries
            for d = 1:n_countries
                o1 = o + (d - 1) * n_countries
                for ω = 1:W
                    U[ω, o1] *= τ[o, d]^(-G[ω])
                end
            end
        end

    else
        rand!(U)
    end

    # normalize expecation to be exacly 1 for each origin-destination
    # this helps when we are sampling from different distributions
    MomentMatch = false
    if MomentMatch
        for o = 1:D
            U[:, o] /= mean(U[:, o])
        end
    end
end
function gFunction!(σ, τ, UPow_init, w_, cPow_, lambda)
    # encountered issues as solver was trying negative values so used abs. maybe there is a better solution
    w = abs.(w_)
    cPow = abs.(cPow_)

    D = size(τ, 1) # num countries 
    W = size(UPow_init, 1) # num draws (or goods)
    bilateral_prices_pow = zeros(eltype(cPow), D)
    prices_pow = 0.0
    expenditure = zeros(eltype(cPow), D, D)
    expenditure_ratio_residual = zeros(eltype(cPow), D, D)
    expenditure_ratio = zeros(eltype(cPow), D, D)
    trade_share_residual = zeros(eltype(cPow), D, D)
    price_index = zeros(eltype(cPow), D)
    exporter_idx = 0.0

    for d = 1:D
        for ω = 1:W
            for o = 1:D
                o1 = o + (d - 1) * D
                bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o, d]) / (UPow_init[ω, o1]))^(1 - σ)
            end

            prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
            expenditure[exporter_idx, d] += prices_pow
            price_index[d] += prices_pow

        end

        for o = 1:D
            expenditure_ratio_residual[o, d] = lambda[o, d] / lambda[1, d] - expenditure[o, d] / expenditure[1, d] # we could have some cases where expenditure on country 1 goods is zero...
            expenditure_ratio[o, d] = expenditure[o, d] / price_index[d]
            trade_share_residual[o, d] = expenditure_ratio[o, d] - lambda[o, d]
        end
    end

    #@show expenditure

    return (expenditure_ratio_residual=expenditure_ratio_residual, price_index=price_index ./ W, expenditure_ratio=expenditure_ratio, trade_share_residual=trade_share_residual, expenditure=expenditure / W)

end
function gFunction_d!(σ, τ, UPow_init, w_, cPow_, lambda, d)
    # encountered issues as solver was trying negative values so used abs. maybe there is a better solution
    w = abs.(w_)
    cPow = abs.(cPow_)

    D = size(τ, 1) # num countries 
    W = size(UPow_init, 1) # num draws (or goods)
    bilateral_prices_pow = zeros(eltype(cPow), D)
    bilateral_prices = zeros(eltype(cPow), D)
    prices_pow = 0.0
    expenditure = zeros(eltype(cPow), D)
    expenditure_ratio_residual = zeros(eltype(cPow), D - 1)

    price_index = 0.0
    exporter_idx = 0.0
    DiscreteImplementation = false
    pricesIdx = zeros(eltype(cPow), D)

    for ω = 1:W

        for o = 1:D
            o1 = o + (d - 1) * D
            bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))^(1 - σ)
            bilateral_prices[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))
        end

        if DiscreteImplementation
            prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
            expenditure[exporter_idx] += prices_pow
            price_index += prices_pow
        else
            smoothMinIndNew!(pricesIdx, bilateral_prices, D, -100.0)

            for o = 1:D
                prices_pow = pricesIdx[o] * bilateral_prices_pow[o]
                expenditure[o] += prices_pow
                price_index += prices_pow
            end
        end

    end

    for o = 2:D
        relative_error = lambda[o, d] / lambda[1, d] - expenditure[o] / expenditure[1]
        absolute_error = lambda[o, d] - expenditure[o] / price_index
        #expenditure_ratio_residual[o-1] = lambda[o,d] / lambda[1,d] - expenditure[o] / expenditure[1] # we could have some cases where expenditure on country 1 goods is zero...
        expenditure_ratio_residual[o-1] = absolute_error # this version works as well, sum of lambdas = 1, probably better as this is the moment condition in the algo
    end
    return expenditure_ratio_residual

end
function gFunction_d2!(σ, τ, UPow_init, w_, cPow_, lambda, d)
    # encountered issues as solver was trying negative values so used abs. maybe there is a better solution
    w = abs.(w_)
    cPow = abs.(cPow_)

    D = size(τ, 1) # num countries 
    W = size(UPow_init, 1) # num draws (or goods)
    bilateral_prices_pow = zeros(eltype(cPow), D)
    prices_pow = 0.0
    expenditure = zeros(eltype(cPow), D)
    expenditure_ratio_residual = zeros(eltype(cPow), D)
    pricesIdx = zeros(eltype(cPow), D)
    price_index = 0.0
    exporter_idx = 0.0
    DiscreteImplementation = false

    for ω = 1:W

        for o = 1:D
            o1 = o + (d - 1) * D
            bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))^(1 - σ)
        end


        if DiscreteImplementation
            prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
            expenditure[exporter_idx] += prices_pow
            price_index += prices_pow
        else
            smoothMinIndNew!(pricesIdx, bilateral_prices_pow, D, 100.0) # we are looking for the max, so the positive tuner sign is correct.

            for o = 1:D
                prices_pow = pricesIdx[o] * bilateral_prices_pow[o]
                expenditure[o] += prices_pow
                price_index += prices_pow
            end
        end
    end

    for o = 1:D
        #expenditure_ratio_residual[o-1] = lambda[o,d] / lambda[1,d] - expenditure[o] / expenditure[1] # we could have some cases where expenditure on country 1 goods is zero...
        expenditure_ratio_residual[o] = lambda[o, d] - expenditure[o] / price_index # this version works as well
    end
    return expenditure_ratio_residual

end
function smoothMinIndNew!(xInd, x, D, tuner)
    # function takes a GxD matrix and calculates another GxD matrix
    # with a smooth approximation of indicator functions for row-wise min
    # i.e., it computes the min of each row and indicates if x_{ij} is that min
    # tuner controls sharpness of approximation (close to 0 = very sharp)

    # this is necessary for optimisers that use auto diff, because everything must be smooth (no mins, no indicators)
    # if not using auto diff, can just use min and indicator 

    # smoothing approximation is softmax, see https://en.wikipedia.org/wiki/Softmax_function 
    denom = 0

    xMin = minimum(x)

    for i = 1:D
        xInd[i] = exp((x[i] - xMin) * tuner)
        #xInd[i] = exp((x[i]) * tuner)

        denom += xInd[i]
    end

    for i = 1:D
        xInd[i] /= denom
    end


end

function hFunction!(K, G, UPow, Uσ, Ū, w, τ, σ, μ, γ, L, P, PMM, counterExplicit, counterType, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, μHat)
    # main function to fill in the moment matrix G for the baseline moments 

    D = size(τ, 1) # num countries 
    W = size(UPow, 1) # num draws (or goods)
    tuner = -100.0 # parameter for smooth mins 
    l = D^2

    gdp = (w .* L)

    # initialise important vectors 
    pricesTemp = zeros(eltype(γ), D)
    pricesTempσ = copy(pricesTemp)

    pricesInd = copy(pricesTemp)
    pricesCounterVec = zeros(eltype(γ), D^2)
    denom = copy(pricesTemp)
    constCons = zeros(eltype(γ), D, D)
    constConsσ = copy(constCons)
    wPow = zeros(eltype(γ), D)

    # identify the part of G matrix where we put the price index moments; we omit wage moments if autarky 
    if counterType != 1
        cInd = D^2 + D - 1
    else
        cInd = D^2
    end

    for d = 1:D
        wPow[d] = w[d]^(1 - σ) # will need transformed wages many times, so do it once here 
    end

    # construct objects that we will need but that never change with omega
    for d = 1:D
        denom[d] = γ[d]^σ * gdp[d]
        for o = 1:D
            constCons[o, d] = w[o] * τ[o, d]
            constConsσ[o, d] = wPow[o] * (τ[o, d])^(1 - σ)
        end
    end


    # loop to fill in the G matrix
    @inbounds for ω = 1:W # main loop over all goods 

        for d = 1:D # loop through all destination countries 

            for o = 1:D # for each origin, construct p_{od}

                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                #o1 = o

                pricesTemp[o] = constCons[o, d] / UPow[ω, o1]
                pricesTempσ[o] = constConsσ[o, d] / Uσ[ω, o1]
            end

            smoothMinIndNew!(pricesInd, pricesTemp, D, tuner) # construct 1\{p_{od}=min_o(p_{od})\}

            indSum = 0


            for o = 1:D # using min prices, compute implied expenditure share and fill in G with implied minus data 
                d1 = d + (o - 1) * D
                pricesTemp[o] = pricesTempσ[o] * pricesInd[o]
                G[ω, d1] = pricesTemp[o] / denom[d] - P[d1] - PMM[d1]
                indSum += pricesTemp[o]
            end


            G[ω, cInd+d] = indSum - denom[d] - PMM[cInd+d] # fill in part of G for price index moments (identifies MU parameter)


            if localGravityMoment + localGravityCrossMoment > 0
                ξ = zeros(eltype(γ), D)
                for o = 1:D
                    o1 = o + (d - 1) * D
                    d1 = d + (o - 1) * D
                    pricesTemp[o] = constCons[o, d] / UPow[ω, o1]
                    ξ[o] = P[d1] * denom[d]
                end

                max_price, max_idx = findmax(pricesTemp[:])

                if localGravityMoment == 1
                    μTarget = true ? μHat : μ
                    localGravityMoment!(G, PMM, D, ω, pricesTemp, ξ, σ, μTarget, d, max_price, gravMoment, strongGravityMoment)
                end

                if localGravityCrossMoment == 1
                    localGravityCrossMoment!(G, PMM, D, ω, pricesTemp, ξ, σ, d, max_price, gravMoment, localGravityMoment, strongGravityMoment)
                end
            end


        end

    end


end

function hFunctionCounter!(K, G, UPow, Uσ, w, τ, σ, γ, L, P, PMM, counterExplicit, counterType, baseIndex)
    # same as hFunction, except fills in counterfactual parts of G and fills in K 

    D = size(τ, 1)
    W = size(UPow, 1)
    tuner = -100
    l = D^2

    gdp = (w .* L)

    pricesTemp = zeros(eltype(γ), D)
    pricesTempσ = copy(pricesTemp)
    pricesInd = copy(pricesTemp)
    pricesCounterVec = zeros(eltype(γ), D^2)
    denom = copy(pricesTemp)
    constCons = zeros(eltype(γ), D, D)
    constConsσ = copy(constCons)
    wPow = zeros(eltype(γ), D)

    if counterType != 1
        bInd = D^2
        cInd = D^2 + D - 1
        dInd = D^2 + 2 * D - 1
    else
        cInd = D^2
        dInd = D^2 + D
    end

    for d = 1:D
        wPow[d] = w[d]^(1 - σ)
    end

    for d = 1:D
        denom[d] = γ[d]^σ * gdp[d]
        for o = 1:D
            constCons[o, d] = w[o] * τ[o, d]
            constConsσ[o, d] = wPow[o] * (τ[o, d])^(1 - σ)
        end
    end

    if counterType != 1

        @inbounds for ω = 1:W

            for d = 1:D

                for o = 1:D

                    o1 = o + (d - 1) * D # uncomment for A_{od}
                    #o1 = o

                    pricesTemp[o] = constCons[o, d] / UPow[ω, o1]
                    pricesTempσ[o] = constConsσ[o, d] / Uσ[ω, o1]

                end

                smoothMinIndNew!(pricesInd, pricesTemp, D, tuner)

                indSum = 0

                for o = 1:D
                    o1 = o + (d - 1) * D
                    pricesTemp[o] = pricesTempσ[o] * pricesInd[o]
                    indSum += pricesTemp[o]
                    pricesCounterVec[o1] = pricesTemp[o] * gdp[d] / denom[d]

                    if o == d && o == baseIndex
                        K[ω] = pricesCounterVec[o1] / gdp[d] # fill in K with own trade share 
                    end

                end

                G[ω, dInd+d] = indSum - denom[d] - PMM[dInd+d]# fill in G with counterfactual price index moments 

            end

            # counterfactual wage moments, omitting for first country 
            for o = 2:D

                tempSum = 0

                for d = 1:D
                    d1 = o + (d - 1) * D
                    tempSum += pricesCounterVec[d1]
                end

                G[ω, bInd+o-1] = tempSum - gdp[o] - PMM[bInd+o-1] # fill in G with counterfactual wage moments 

            end

        end
    else
        @inbounds for ω = 1:W
            # we need only baseIndex
            for d = 1:D
                if d == baseIndex
                    o1 = d + (d - 1) * D # uncomment for A_{od}
                    #o1 = d
                    G[ω, dInd+d] = constConsσ[d, d] / Uσ[ω, o1] - denom[d] - PMM[dInd+d] # all countries go into autarky. Price index is domestic price.
                end
            end
        end
    end

end

function gravityMoment!(G, τ, c, D, W, γ, strongGravityMoment)
    # constructs gravity moment, if using (see theory note)

    #deltaτ = zeros(eltype(γ),size(τ))
    #deltaτ[:] .= doubleDiff(τ)
    #deltaC = zeros(eltype(γ),D,D)
    #deltaC[:] .= doubleDiff(reshape(c,(D,D)))

    deltaτ = doubleDiff(τ)
    deltaC = doubleDiff(reshape(c, (D, D)))

    sumGrav = 0

    for o = 2:D

        sumGrav += deltaτ[o, 1] * deltaC[o, 1]

        for d = 3:D
            sumGrav += deltaτ[o, d] * deltaC[o, d]
        end

    end

    sumGrav /= (D - 1)^2

    for i = 1:W
        G[i, end-strongGravityMoment] = sumGrav
    end
end

function newGravityMoment!(G, PMM, τ, D, W, γ, U, strongGravityMoment)
    # constructs gravity moment, if using (see theory note)

    deltaτ = doubleDiff(τ)

    meanτ = 0
    for o = 2:D

        meanτ += deltaτ[o, 1]

        for d = 3:D
            meanτ += deltaτ[o, d]
        end
    end
    meanτ /= (D - 1)^2



    U_ω = zeros(eltype(γ), size(τ))

    for ω = 1:W
        sumGrav = 0
        U_ω = U[ω, :]
        deltaU = doubleDiff(reshape(U_ω, (D, D)))

        for o = 2:D

            sumGrav += (deltaτ[o, 1] - meanτ) * deltaU[o, 1]

            for d = 3:D
                sumGrav += (deltaτ[o, d] - meanτ) * deltaU[o, d]
            end
        end
        sumGrav /= (D - 1)^2
        G[ω, end-strongGravityMoment] = 1000.0 * sumGrav - PMM[end-strongGravityMoment]
    end

end

function sameMarginalsMomentold!(Ū, G, PMM, D, W, X, μ_σ, ν, momentOrder, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, baseIndex)
    # assures the marginals are equal amongst themselves

    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2)

    Γ = gamma(1 + μ_σ)
    U_pow = zeros(momentOrder + 1)
    factorial_k = zeros(momentOrder)
    for k = 2:momentOrder
        factorial_k[k] = factorial(k)
    end
    #ref_idx = baseIndex + (baseIndex - 1) * D
    refIndex = 1
    refIndex1 = refIndex + (refIndex - 1) * D
    for ω = 1:W
        #=
                G[ω, end-offset-1] =  log((Ū[ω, refIndex1]/ν[baseIndex, baseIndex]))- η[1]
                G[ω, end-offset-1-momentOrder] =  ((Ū[ω, o1]/ν[o, d])^μ_σ)/ Γ - η[momentOrder+1]

                for k =2:momentOrder
                    # E[exp] = k!, so we normalize by it. 
                    G[ω, end-offset-k] =  ((Ū[ω, refIndex1]/ν[baseIndex, baseIndex])^k)/factorial(k) - η[k]
                end
        =#

        #F_1_X = Ū[ω, refIndex1]> X[ω] ? 1 : 0
        U_pow[1] = (Ū[ω, refIndex1] / ν[refIndex, refIndex])^μ_σ
        for k = 1:momentOrder
            U_pow[k+1] = (Ū[ω, refIndex1] / ν[refIndex, refIndex])^k
        end
        for o = 1:D
            for d = 1:D
                #if d == baseIndex # clculate only for the baseIndex destination. This is just to get the gains from trade working
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                #F_od_X = Ū[ω, o1]/ν[o, d] > X[ω] ? 1 : 0
                #G[ω, end-offset-o1] =  F_1_X - F_od_X # E[F_1,1(X)] = E[F_o,d(X)] # this is probably not a good moment condition, The X will find its way into F through the RN, so it is not independent of U_od under F anymore.
                #G[ω, end-offset-o1] =  ((Ū[ω, o1]/ν[o, d])^μ_σ - U_pow[1])/ Γ # negative power moment, minimal restriction on F as this power needs to be defined for the counterfactual to be defined.
                G[ω, end-offset-o1] = log((Ū[ω, o1] / ν[o, d]) / (Ū[ω, refIndex1] / ν[refIndex, refIndex])) # negative power moment, minimal restriction on F as this power needs to be defined for the counterfactual to be defined
                G[ω, end-offset-(1+momentOrder)*D*D-o1] = ((Ū[ω, o1] / ν[o, d])^μ_σ - U_pow[1]) / Γ # negative power moment, minimal restriction on F as this power needs to be defined for the counterfactual to be defined.

                G[ω, end-offset-1*D*D-o1] = Ū[ω, o1] - ν[o, d]
                control_variable = (Ū[ω, o1] / ν[o, d]) - U_pow[2]
                for k = 2:momentOrder
                    this_moment_variable = ((Ū[ω, o1] / ν[o, d])^k - U_pow[k+1]) / factorial_k[k]
                    # E[exp] = k!, so we normalize by it. 
                    G[ω, end-offset-k*D*D-o1] = this_moment_variable - control_variable
                    control_variable = this_moment_variable
                end
                #end
            end
        end
    end
end
function sameMarginalsMoment!(Ū, G, PMM, D, W, μ_σ, ν, momentOrder, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, baseIndex)
    # assures the marginals are equal amongst themselves
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2)

    νk = zeros(D, D, momentOrder * 2)
    for o = 1:D
        for d = 1:D

            for k = 1:momentOrder
                α_k = k
                νk[o, d, k] = ν[o, d]^α_k
            end

            for k = 1:momentOrder
                α_k = -1 + 1 / (1 + k)
                νk[o, d, k+momentOrder] = ν[o, d]^α_k
            end
        end
    end

    refIndex = 1 #do not change
    refIndex1 = refIndex + (refIndex - 1) * D
    for o = 1:D
        for d = 1:D
            o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 

            #+ve moments

            @. G[:, end-offset-(1-1)*D*D-o1] = Ū[:, o1, 1] ./ νk[o, d, 1] .- Ū[:, refIndex1, 1] ./ νk[refIndex, refIndex, 1]
            for k = 2:momentOrder
                @. G[:, end-offset-(k-1)*D*D-o1] = (Ū[:, o1, k] ./ νk[o, d, k] .- Ū[:, o1, k-1] ./ νk[o, d, k-1]) .- (Ū[:, refIndex1, k] ./ νk[refIndex, refIndex, k] .- Ū[:, refIndex1, k-1] ./ νk[refIndex, refIndex, k-1])
            end
            #-ve moments
            @. G[:, end-offset-(momentOrder+1-1)*D*D-o1] = Ū[:, o1, momentOrder+1] ./ νk[o, d, momentOrder+1] .- Ū[:, refIndex1, momentOrder+1] ./ νk[refIndex, refIndex, momentOrder+1]
            for k = 2:momentOrder
                @. G[:, end-offset-(momentOrder+k-1)*D*D-o1] = (Ū[:, o1, momentOrder+k] ./ νk[o, d, k+momentOrder] .- Ū[:, o1, k+momentOrder-1] ./ νk[o, d, k+momentOrder-1]) .- (Ū[:, refIndex1, k+momentOrder] ./ νk[refIndex, refIndex, k+momentOrder] .- Ū[:, refIndex1, k+momentOrder-1] ./ νk[refIndex, refIndex, k+momentOrder-1])
            end
        end
    end

end

function sameMarginalsMomentCDF!(Ū, G, PMM, D, W, μ_σ, ν, CDF_X, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, baseIndex)
    # assures the marginals are equal amongst themselves
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2)

    #CDF_X is a vector of increasing values of X. 
    # the moment condition is CDF_od(X) = CDF_11(X)
    # E[U_od<=X_k] = E[U_11<=X_k]

    K = length(CDF_X) + 2
    refIndex = 1 #do not change
    refIndex1 = refIndex + (refIndex - 1) * D
    CDF_11_X = zeros(W, K)
    for ω = 1:W
        smallest_X = searchsortedfirst(CDF_X, Ū[ω, refIndex1, 1] / ν[refIndex, refIndex]) + 1
        for i = smallest_X:K
            CDF_11_X[ω, i] += 1
        end
    end

    for o = 1:D
        for d = 1:D
            o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
            @. G[:, end-offset-o1] = Ū[:, o1, 1] ./ ν[o, d] .- Ū[:, refIndex1, 1] ./ ν[refIndex, refIndex]

            for i = 1:K
                @. G[:, end-offset-i*D^2-o1] += -CDF_11_X[:, i]
            end
            for ω = 1:W
                smallest_X = searchsortedfirst(CDF_X, Ū[ω, o1, 1] / ν[o, d])

                for i = smallest_X:K
                    G[ω, end-offset-i*D^2-o1] += 1
                end
            end
        end
    end
end

function sameMarginalsMomentCDFNoScaling!(G, PMM, D, W, CDF_Moments, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, baseIndex)
    K_ = size(CDF_Moments, 2)
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2)
    @. G[:, end-offset-K_+1:end-offset] = CDF_Moments[:, :]
end

function independenceMomentold!(Ū, G, PMM, D, W, μ_σ, ν, ηk, momentOrder, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, sameMarginalsMoment)
    # assures the marginals are identical and independent
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * (2 + momentOrder) * D^2
    for ω = 1:W
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for k = 1:momentOrder
                    idx_moment = (d - 1) * (D * momentOrder + D * D * momentOrder * momentOrder) + (o - 1) * momentOrder + k
                    G[ω, end-offset-idx_moment] = Ū[ω, o1]^k / factorial(k) - ηk[k] # E[Uod^k/k!] = ηodk
                    for c = 1:D
                        if c > o
                            c1 = c + (d - 1) * D # uncomment to to U_{od} rather than U_o
                            for m = 1:momentOrder
                                idx_corss_moment = d * (D * momentOrder) + (d - 1) * (D * D * momentOrder * momentOrder) + (o - 1) * D * momentOrder * momentOrder + (c - 1) * momentOrder * momentOrder + (k - 1) * momentOrder + m
                                # E[Uod^k/k! Ucd^m/m!] = ηodk*ηcdm
                                G[ω, end-offset-idx_corss_moment] = (Ū[ω, o1]^k) * (Ū[ω, c1]^m) / (factorial(k) * factorial(m)) - ηk[k] * ηk[m]
                            end
                        end
                    end
                end
            end
        end
    end
end

function uncorrelationMoment!(Ū, G, PMM, D, W, ν, ηk, momentOrder, momentOrderForBaseIndex, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, sameMarginalsMoment)
    # assures the marginals are identical and independent
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * ((2 * momentOrder) * D^2 + momentOrderForBaseIndex)

    refIndex = 1
    refIndex1 = refIndex + (refIndex - 1) * D
    for ω = 1:W
        # E[U(ref,ref)^k/k!] = ηk
        G[ω, end-offset] = Ū[ω, refIndex1, 1] / ν[refIndex, refIndex] - ηk[1]
    end

    idx_corss_moment = 0
    for d = 1:D
        for o = 1:D
            o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
            for c = 1:D
                for f = 1:D
                    if c > o || f > d
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        idx_corss_moment += 1
                        for ω = 1:W
                            G[ω, end-offset-1-idx_corss_moment] = (Ū[ω, o1, 1] / ν[o, d]) * (Ū[ω, c1, 1] / ν[c, f]) - (ηk[1]) * (ηk[1])
                        end
                    end
                end
            end
        end
    end
end

function uncorrelationMomentNoScaling!(Ū, G, PMM, D, W, ηk, Ind_Moments, momentOrder, momentOrderForBaseIndex, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, sameMarginalsMoment)
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * ((2 * momentOrder) * D^2 +momentOrderForBaseIndex)

    refIndex = 1
    refIndex1 = refIndex + (refIndex - 1) * D


    # E[U(ref,ref)^k/k!] = ηk
    @. G[:, end-offset] += Ū[:, refIndex1, 1] .- ηk[1]

    K_ = size(Ind_Moments, 2)
    square_mean = ηk[1]^2
    @. G[:, end-offset-1-K_+1:end-offset-1] += Ind_Moments[:, :] .- square_mean

end

function independenceMoment!(Ū, G, PMM, D, W, μ_σ, ν, ηk, momentOrder, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, sameMarginalsMoment)
    # assures the marginals are identical and independent
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * (2 + momentOrder) * D^2

    νk = zeros(D, D, momentOrder + 2)
    for o = 1:D
        for d = 1:D
            for k = 1:momentOrder
                νk[o, d, k] = ν[o, d]^k
            end
            νk[o, d, momentOrder+1] = ν[o, d]^μ_σ
            νk[o, d, momentOrder+2] = log(ν[o, d])
        end
    end

    refIndex = 1
    refIndex1 = refIndex + (refIndex - 1) * D
    for ω = 1:W

        # E[U(ref,ref)^k/k!] = ηk
        G[ω, end-offset-k] = Ū[ω, refIndex1, 1] / νk[refIndex, refIndex, 1] - ηk[1]
        for k = 2:momentOrder
            G[ω, end-offset-k] = (Ū[ω, refIndex1, k] / νk[refIndex, refIndex, k] - Ū[ω, refIndex1, k-1] / νk[refIndex, refIndex, k-1]) - (ηk[k] - ηk[k-1])
        end

        idx_corss_moment = 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for c = 1:D
                    if c < o
                        c1 = c + (d - 1) * D # uncomment to to U_{od} rather than U_o
                        idx_corss_moment += 1
                        G[ω, end-offset-momentOrder-idx_corss_moment] = (Ū[ω, o1, 1] / νk[o, d, 1]) * (Ū[ω, c1, 1] / νk[c, d, 1]) - (ηk[1]) * (ηk[1])
                        for m = 2:momentOrder

                            #idx_corss_moment = (d-1)*(D*(D-1)*momentOrder*momentOrder)/2 + (o-1)*(o-1)*momentOrder*momentOrder+ (c-1)*momentOrder*momentOrder + (k-1)*momentOrder + m

                            # E[Uod Ucd^m/m!] = ηodk*ηcdm
                            idx_corss_moment += 1
                            G[ω, end-offset-momentOrder-idx_corss_moment] = (Ū[ω, o1, 1] / νk[o, d, 1]) * (Ū[ω, c1, m] / νk[c, d, m] - Ū[ω, c1, m-1] / νk[c, d, m-1]) - (ηk[1]) * (ηk[m] - ηk[m-1])
                        end
                    end
                end


                for k = 2:momentOrder
                    for c = 1:D
                        if c > o
                            c1 = c + (d - 1) * D # uncomment to to U_{od} rather than U_o
                            idx_corss_moment += 1
                            G[ω, end-offset-momentOrder-idx_corss_moment] = (Ū[ω, o1, k] / νk[o, d, k] - Ū[ω, o1, k-1] / νk[o, d, k-1]) * (Ū[ω, c1, 1] / νk[c, d, 1]) - (ηk[k] - ηk[k-1]) * (ηk[1])
                            for m = 2:momentOrder
                                #idx_corss_moment = d*(D*momentOrder) + (d-1)*(D*D*momentOrder*momentOrder) + (o-1)*D*momentOrder*momentOrder+ (c-1)*momentOrder*momentOrder + (k-1)*momentOrder + m
                                # E[Uod^k/k! Ucd^m/m!] = ηodk*ηcdm
                                idx_corss_moment += 1
                                G[ω, end-offset-momentOrder-idx_corss_moment] = (Ū[ω, o1, k] / νk[o, d, k] - Ū[ω, o1, k-1] / νk[o, d, k-1]) * (Ū[ω, c1, m] / νk[c, d, m] - Ū[ω, c1, m-1] / νk[c, d, m-1]) - (ηk[k] - ηk[k-1]) * (ηk[m] - ηk[m-1])
                            end
                        end
                    end
                end
            end
        end
    end
end
function MartingalDifferenceDivergence2(ν, τ, D)

    # function to calculate the martingale difference divergence as per Su and Zheng (2017)
    # https://www.sciencedirect.com/science/article/pii/S0165176517301805?via%3Dihub

    # should make the implementation more efficient

    ΔΔlnν = doubleDiff(ν)
    ΔΔlnτ = doubleDiff(τ)

    meanν = 0
    for o = 2:D
        meanν += ΔΔlnν[o, 1]
        for d = 3:D
            meanν += ΔΔlnν[o, d]
        end
    end

    meanν /= (D - 1)^2

    ΔΔlnν = ΔΔlnν .- meanν

    MDD = 0.0

    for o = 1:D
        for d = 1:D
            for o1 = 1:D
                for d1 = 1:D
                    if o != o1 || d != d1
                        MDD += -abs(ΔΔlnτ[o, d] - ΔΔlnτ[o1, d1]) * ΔΔlnν[o, d] * ΔΔlnν[o1, d1]
                    end
                end
            end
        end
    end

    MDD /= D^4 - D^2

    return MDD^2
end
function MartingalDifferenceDivergence(ν, Mτ, D)

    # function to calculate the martingale difference divergence as per Su and Zheng (2017)
    # https://www.sciencedirect.com/science/article/pii/S0165176517301805?via%3Dihub

    # this matrix version sould be faster

    ΔΔlnν = doubleDiff(ν)

    meanν = 0
    for o = 2:D
        meanν += ΔΔlnν[o, 1]
        for d = 3:D
            meanν += ΔΔlnν[o, d]
        end
    end

    meanν /= (D - 1)^2

    ΔΔlnν = ΔΔlnν .- meanν

    longΔΔlnν = reshape(ΔΔlnν, (D^2, 1))

    MDD = dot(longΔΔlnν, Mτ, longΔΔlnν)

    MDD /= D^4 - D^2

    return MDD^2
end
function strongGravityMomentold2!(G, PMM, τ, ν, D, W, Ū, Σ, Mτ, counterType)

    # constructs the moment that ΔΔ E[ln U] is mean independent of ΔΔ lnτ  

    dInd = counterType == 1 ? D^2 + 2 * D : D^2 + (D - 1) + 2 * D

    MDD = MartingalDifferenceDivergence(ν, Mτ, D)

    for o = 1:D
        for d = 1:D
            o1 = o + (d - 1) * D
            Σ_od = Σ[o, d]
            ν_od = ν[o, d] / Σ_od
            @. G[:, dInd+o1] = Ū[:, o1] ./ Σ_od .- ν_od .- PMM[dInd+o1]
        end
    end


    @. G[:, end] = MDD - PMM[end]
end

function strongGravityMoment!(G, PMM, τ, ν, D, W, Ū, Σ, Mτ, counterType, baseIndex, sameMarginalsMoment)

    # constructs the moment that ΔΔ E[ln U] is mean independent of ΔΔ lnτ  

    dInd = counterType == 1 ? D^2 + 2 * D : D^2 + (D - 1) + 2 * D

    if sameMarginalsMoment == 0
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D
                ν_od = ν[o, d]
                @. G[:, dInd+o1] = Ū[:, o1, 1] .- ν_od .- PMM[dInd+o1]
            end
        end
    end

    ΔΔlnν = doubleDiff(ν)
    deltaτ = doubleDiff(τ)


    meanτ = 0
    for o = 2:D
        meanτ += deltaτ[o, 1]
        for d = 3:D
            meanτ += deltaτ[o, d]
        end
    end
    meanτ /= (D - 1)^2

    sumGrav = 0
    for o = 2:D
        sumGrav += (deltaτ[o, 1] - meanτ) * ΔΔlnν[o, 1]
        for d = 3:D
            sumGrav += (deltaτ[o, d] - meanτ) * ΔΔlnν[o, d]
        end
    end
    sumGrav /= (D - 1)^2

    @. G[:, end] = sumGrav - PMM[end]

end

function localGravityMoment!(G, PMM, D, ω, prices, ξ, σ, μ, d, max_price, gravMoment, strongGravityMoment)
    tuner = -100.0
    β = 0.01
    pricesInd_without_o = copy(prices)
    prices_without_o = copy(prices)
    counter = 0

    for o = 1:D
        counter += 1
        if o != d
            prices_without_o = prices[:]
            prices_without_o[o] = max_price + 10000  # such that we are sure the price from o is not the min
            min_price_without_o, exporter_idx_without_o = findmin(prices_without_o[:])
            smoothMinIndNew!(pricesInd_without_o, prices_without_o, D, tuner)
            A = prices[o]^(1 - σ) * SmoothDirac(β, log(prices[o] / min_price_without_o))
            B = prices[d]^(1 - σ) * pricesInd_without_o[d] * SmoothDirac(β, log(prices[o] / prices[d]))

            moment_idx = (D - 1) * (d - 1) + counter

            G[ω, end-strongGravityMoment-gravMoment-moment_idx] = σ - 1 + A / ξ[o] + B / ξ[d] - 1 / μ - PMM[end-strongGravityMoment-gravMoment-moment_idx]
        end
    end

end

function localGravityCrossMoment!(G, PMM, D, ω, prices, ξ, σ, d, max_price, gravMoment, localGravityMoment, strongGravityMoment)
    tuner = -100.0
    β = 0.01
    pricesInd_without_c = copy(prices)
    prices_without_c = copy(prices)
    counter = 0
    for o = 1:D
        if o != d
            for c = 1:D
                if c != o && c != d
                    counter += 1
                    prices_without_c = prices[:]
                    prices_without_c[c] = max_price + 10000 # such that we are sure the price from c is not the min
                    min_price_without_c, exporter_idx_without_c = findmin(prices_without_c[:])
                    smoothMinIndNew!(pricesInd_without_c, prices_without_c, D, tuner)
                    A = prices[o]^(1 - σ) * pricesInd_without_c[o]SmoothDirac(β, log(prices[c] / prices[o]))
                    B = prices[d]^(1 - σ) * pricesInd_without_c[d] * SmoothDirac(β, log(prices[c] / prices[d]))
                    moment_idx = localGravityMoment * D * (D - 1) + (D - 1) * (D - 2) * (d - 1) + counter
                    G[ω, end-strongGravityMoment-gravMoment-moment_idx] = A / ξ[o] - B / ξ[d] - PMM[end-strongGravityMoment-gravMoment-moment_idx]
                end
            end
        end
    end
end

function moments!(K, G, θ, U, obj)
    # main function that takes empty K and G, and the parameters, and fills in the moment matrices 

    # unpack the gamma (auxiliary parameters) vector
    @unpack wHat, L, LPrime, τ, τPrime, P, PMM, baseIndex, indicators, Uσ, Ū, Σ_od, Mτ, μHat, CDF_X, CDF_Moments, Ind_Moments = obj.γ
    @unpack counterExplicit, counterType, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, momentOrderForBaseIndex, StarDistributionType, useCDFforMarginalMatching = indicators

    W = size(U, 1)
    D = size(τ, 1)

    # unpack the structural parameters vector
    μ = θ[1]
    σ = θ[2]
    γ = θ[3:3+D-1]

    γ_prime = copy(γ)



    if counterType != 1
        γ_prime = θ[3+D:3+2*D-1]
    else
        γ_prime[baseIndex] = θ[3+D] # we are not interested in the other gamma_primes
    end

    if counterType != 1 # needs to be adjusted
        wPrime = θ[3+2*D:3+2*D+(D-1)-1]
    else
        wPrime = copy(obj.γ.wPrimeHat)
    end

    # add 1 (normalised wage) into the w' vector at appropriate index
    insert!(wPrime, baseIndex, 1)

    counterType_θ_offset = 0
    if counterType != 1
        counterType_θ_offset = 2 * (D - 1) # D-1 wagesPrime and D-1 gamma_primes 
    end

    if counterExplicit == 0
        # insert relevant k function if counterfactual does not depend on U
        counterVal = (γ[baseIndex] / γ_prime[baseIndex])^(σ / (σ - 1)) - 1
        @inbounds for i = 1:W
            K[i] = counterVal
        end
    end


    if θConstant != 1
        # update c and U matrices to the exponents relevant for calculating price
        # nb: calculate here as don't want to do it in each hFunction call
        # only do this if theta / sigma ever vary, otherwise we precalculate

        #UPow = copy(U)
        UPow = zeros(eltype(γ), size(U))
        UσPow = zeros(eltype(γ), size(U))
        for i = 1:length(UPow)
            UPow[i] = U[i]^(-μ)
            UσPow[i] = Uσ[i]^(-μ)
        end
        hFunction!(K, G, UPow, UσPow, Ū, wHat, τ, σ, μ, γ, L, P, PMM, counterExplicit, counterType, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, μHat) # fill in G with baseline moments 
        hFunctionCounter!(K, G, UPow, UσPow, wPrime, τPrime, σ, γ_prime, LPrime, P, PMM, counterExplicit, counterType, baseIndex) # fill in G with counterfactual moments, fill in K 
    else

        hFunction!(K, G, U, Uσ, Ū, wHat, τ, σ, μ, γ, L, P, PMM, counterExplicit, counterType, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, μHat)
        hFunctionCounter!(K, G, U, Uσ, wPrime, τPrime, σ, γ_prime, LPrime, P, PMM, counterExplicit, counterType, baseIndex)
    end


    if sameMarginalsMoment == 1
        ν = ones(D, D)
        if NoScalingforSameMartingale == 0
            ν = reshape(vcat(1, θ[counterType_θ_offset+3+2*D:counterType_θ_offset+3+2*D+D^2-2]), (D, D))
        end
        if useCDFforMarginalMatching == 0
            sameMarginalsMoment!(Ū, G, PMM, D, W, (1 - σ) * μ, ν, momentOrder, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, baseIndex)
        elseif NoScalingforSameMartingale == 0
            sameMarginalsMomentCDF!(Ū, G, PMM, D, W, (1 - σ) * μ, ν, CDF_X, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, baseIndex)
        else
            sameMarginalsMomentCDFNoScaling!(G, PMM, D, W, CDF_Moments, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, baseIndex)
        end
    end

    if independenceMoment == 1

        ν = ones(D, D)
        ηk = θ[counterType_θ_offset+3+D+1:counterType_θ_offset+3+D+1]

        if NoScalingforSameMartingale == 0
            ν = reshape(vcat(1, θ[counterType_θ_offset+3+2*D:counterType_θ_offset+3+2*D+D^2-2]), (D, D))
            ηk = θ[counterType_θ_offset+3+2*D+D^2-1:counterType_θ_offset+3+2*D+D^2-1]
            uncorrelationMoment!(Ū, G, PMM, D, W, ν, ηk, momentOrder, momentOrderForBaseIndex, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, sameMarginalsMoment)
        else
            uncorrelationMomentNoScaling!(Ū, G, PMM, D, W, ηk, Ind_Moments, momentOrder, momentOrderForBaseIndex, gravMoment, localGravityMoment, strongGravityMoment, localGravityCrossMoment, sameMarginalsMoment)
        end


        #independenceMoment!(Ū, G, PMM, D, W, ν, ηk, momentOrder, gravMoment, localGravityMoment, strongGravityMoment,localGravityCrossMoment, sameMarginalsMoment)
    end


    if gravMoment == 1
        newGravityMoment!(G, PMM, τ, D, W, γ, U, strongGravityMoment) # add gravity moment if using 
    end


    if strongGravityMoment == 1
        ν = reshape(θ[counterType_θ_offset+3+2*D:counterType_θ_offset3+2*D+D^2-1], (D, D))
        strongGravityMoment!(G, PMM, τ, ν, D, W, Ū, Σ_od, Mτ, counterType, baseIndex)
    end

end
function ccInner(θ_initial, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, independenceMoment, momentOrder, counterType)
    # function that runs the CC outer loop 
    D = length(γ.L)

    numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * ((2 * momentOrder) * D^2 + momentOrderForBaseIndex) + independenceMoment * (D^2 - floor(Int, D * (1 + D) / 2) + 1)

    if counterType != 1
        numMoments += (D - 1)
    end

    obj = PsiObjectiveBundleDelta(
        #δ = 1,
        #find_smallest = true,
        γ=γ,
        (moments!)=moments!,
        #moments_jacobian! = rust_moments_jacobian!,
        d=numMoments,
        outer_constr_index=numMoments + 1 - strongGravityMoment,
        inequality_index=Int64[],
        l=size(θ_initial, 1),
        U=U,
        N=100,
        outer_loop_opt="ek_outer_loop_options.opt",
        inner_loop_opt="ek_inner_loop_options.opt",
        lower_limit=-50)

    val, x, nStatus = inner_loop(obj, θ_initial)

    return (val, x, nStatus)
end

function ccOuter(θ_initial_all, θ_Star_all, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, counterType, useParallel, file_name)
    # function that runs the CC outer loop 
    D = length(γ.L)

    numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * ((2 * momentOrder) * D^2 + momentOrderForBaseIndex) + independenceMoment * D * (D^2 - floor(Int, D * (1 + D) / 2) + 1)

    if counterType != 1
        numMoments += (D - 1)
    end

    θ_lower_all = (θ_initial_all.*0.5)[:] # lower bound for parameters in outer loop optimisation 
    θ_upper_all = (θ_initial_all.*1.5)[:] # upper bound for parameters in outer loop optimisation 

    # @. θ_lower_all[:] = min(θ_lower_all[:], (θ_Star_all .* 0.8)[:,1])  # lower bound for parameters in outer loop optimisation 
    # @. θ_upper_all[:] = max(θ_upper_all[:], (θ_Star_all .* 1.2)[:,1])  # upper bound for parameters in outer loop optimisation 


    θ_lower_all[2] = θ_initial_all[2] # fix sigma (second param) as not identified anyway
    θ_upper_all[2] = θ_initial_all[2]

    # for Frechet, there is a single mu parameter that matches the moments.
    #θ_lower_all[1] = θ_initial_all[1] # fix Mu
    #θ_upper_all[1] = θ_initial_all[1]

    θ_lower_all[1] = 1 * min(θ_initial_all[1], θ_Star_all[1])
    θ_upper_all[1] = 1 * max(θ_initial_all[1], θ_Star_all[1])


    # if NoScalingforSameMartingale == 1, then we do not need to scale the Us.
    if strongGravityMoment == 1 || (sameMarginalsMoment == 1 && NoScalingforSameMartingale == 0)
        θ_lower_all[3+D+1:3+D+1+D^2-2] = 0.8 * θ_initial_all[3+D+1:3+D+1+D^2-2] # lower and upper bounds for ν_{od}
        θ_upper_all[3+D+1:3+D+1+D^2-2] = 1.2 * θ_initial_all[3+D+1:3+D+1+D^2-2]
    end


    if independenceMoment == 1 # independence with same marginals
        if NoScalingforSameMartingale == 0
            θ_lower_all[3+D+1+D^2-1:3+D+1+D^2-1] = 0.8 * θ_initial_all[3+D+1+D^2-1:3+D+1+D^2-1] # lower and upper bounds for ν_{od}
            θ_upper_all[3+D+1+D^2-1:3+D+1+D^2-1] = 1.2 * θ_initial_all[3+D+1+D^2-1:3+D+1+D^2-1]
        else
            θ_lower_all[3+D+1:3+D+1] = 0.8 * θ_initial_all[3+D+1:3+D+1] # lower and upper bounds for ν_{od}
            θ_upper_all[3+D+1:3+D+1] = 1.2 * θ_initial_all[3+D+1:3+D+1]
        end
    end

    θ_initial = θ_initial_all
    θ_lower = θ_lower_all
    θ_upper = θ_upper_all

    @show θ_initial_all
    @show θ_initial

    #δ_grid = [0.01, 0.1, 0.5, 1, 2] # vector of deltas to run 
    δ_grid = [1] # vector of deltas to run 

    κ_lower = zeros(length(δ_grid))
    κ_upper = zeros(length(δ_grid))
    Θ_lower = zeros(length(θ_initial), length(δ_grid))
    Θ_upper = zeros(length(θ_initial), length(δ_grid))

    if useParallel == 0 # if parallelisation is off, do deltas one at a time 

        # construct object to input into outer loop function 
        if counterType == 1 # if GT counterfactual, k does not depend on U directly (so create Implicit object)
            obj = PsiObjectiveBundleImplicit(
                δ=1,
                find_smallest=true,
                γ=γ,
                (moments!)=moments!,
                #moments_jacobian! = rust_moments_jacobian!,
                d=numMoments,
                outer_constr_index=numMoments + 1 - strongGravityMoment,
                inequality_index=Int64[],
                l=size(θ_initial, 1),
                U=U,
                #N=25000,
                outer_loop_opt="ek_outer_loop_options.opt",
                inner_loop_opt="ek_inner_loop_options.opt",
                lower_limit=-50)
        else
            obj = PsiObjectiveBundleExplicit(
                δ=1,
                find_smallest=true,
                γ=γ,
                (moments!)=moments!,
                #moments_jacobian! = rust_moments_jacobian!,
                d=numMoments,
                outer_constr_index=numMoments + 1 - strongGravityMoment,
                inequality_index=Int64[],
                l=size(θ_initial, 1),
                U=U,
                #N=25000,
                outer_loop_opt="ek_outer_loop_options.opt",
                inner_loop_opt="ek_inner_loop_options.opt",
                lower_limit=-50)

        end

        # construct object to input into outer loop function 

        obj.find_smallest = false
        θ_1 = copy(θ_initial)

        # run outer loop for upper bound 
        for (i, δ) in enumerate(δ_grid)
            obj.δ = δ
            κ_upper[i], θ_1 = outer_loop(obj, θ_lower, θ_upper, θ_initial)
            Θ_upper[:, i] .= θ_1
            print(κ_upper[i])
        end

        obj.find_smallest = true
        θ_1 = copy(θ_initial)

        # run outer loop for lower bound 
        for (i, δ) in enumerate(δ_grid)
            obj.δ = δ
            κ_lower[i], θ_1 = outer_loop(obj, θ_lower, θ_upper, θ_initial)
            Θ_lower[:, i] .= θ_1
            print(κ_lower[i])
        end

    elseif useParallel == 1

        θ_1 = copy(θ_initial)

        Threads.@threads for i = 1:length(δ_grid)

            if counterType == 1

                obj = PsiObjectiveBundleImplicit(
                    δ=δ_grid[i],
                    find_smallest=true,
                    γ=γ,
                    (moments!)=moments!,
                    #moments_jacobian! = rust_moments_jacobian!,
                    d=numMoments,
                    outer_constr_index=numMoments + 1 - strongGravityMoment,
                    inequality_index=Int64[],
                    #l=size(θ_initial, 1),
                    l=length(θ_initial),
                    U=U,
                    N=20000,
                    outer_loop_opt="ek_outer_loop_options.opt",
                    inner_loop_opt="ek_inner_loop_options.opt",
                    lower_limit=-50)

            else

                obj = PsiObjectiveBundleExplicit(
                    δ=δ_grid[i],
                    find_smallest=true,
                    γ=γ,
                    (moments!)=moments!,
                    #moments_jacobian! = rust_moments_jacobian!,
                    d=numMoments,
                    outer_constr_index=numMoments + 1 - strongGravityMoment,
                    inequality_index=Int64[],
                    #l=size(θ_initial, 1),
                    l=length(θ_initial),
                    U=U,
                    N=20000,
                    outer_loop_opt="ek_outer_loop_options.opt",
                    inner_loop_opt="ek_inner_loop_options.opt",
                    lower_limit=-50)

            end

            κ_lower[i], θ_2 = outer_loop(obj, θ_lower, θ_upper, θ_initial)
            Θ_lower[:, i] .= θ_2
            print(κ_lower[i])
        end

        obj.find_smallest = false
        θ_1 = copy(θ_initial)

        Threads.@threads for i = 1:length(δ_grid)

            if counterType == 1

                obj = PsiObjectiveBundleImplicit(
                    δ=δ_grid[i],
                    find_smallest=false,
                    γ=γ,
                    (moments!)=moments!,
                    #moments_jacobian! = rust_moments_jacobian!,
                    d=numMoments,
                    outer_constr_index=numMoments + 1 - strongGravityMoment,
                    inequality_index=Int64[],
                    #l=size(θ_initial, 1),
                    l=length(θ_initial),
                    U=U,
                    #N=20000,
                    outer_loop_opt="ek_outer_loop_options.opt",
                    inner_loop_opt="ek_inner_loop_options.opt",
                    lower_limit=-50)

            else

                obj = PsiObjectiveBundleExplicit(
                    δ=δ_grid[i],
                    find_smallest=false,
                    γ=γ,
                    (moments!)=moments!,
                    #moments_jacobian! = rust_moments_jacobian!,
                    d=numMoments,
                    outer_constr_index=numMoments + 1 - strongGravityMoment,
                    inequality_index=Int64[],
                    #l=size(θ_initial, 1),
                    l=length(θ_initial),
                    U=U,
                    #N=20000,
                    outer_loop_opt="ek_outer_loop_options.opt",
                    inner_loop_opt="ek_inner_loop_options.opt",
                    lower_limit=-50)

            end

            κ_upper[i], θ_1 = outer_loop(obj, θ_lower, θ_upper, θ_initial)
            Θ_upper[:, i] .= θ_1
            print(κ_upper[i])
        end

    end

    print(δ_grid)
    print(κ_lower)
    print(κ_upper)

    return (δ_grid, Θ_upper, κ_upper, Θ_lower, κ_lower)
end

function buildObjectsForMoments(globParams, preStepOutput, data, LPrime, τPrime, Uσ, Ū, Σ_od, Mτ, μHat, PMM=zeros(1), CDF_X=zeros(1), CDF_Moments=zeros(1), Ind_Moments=zeros(1))
    # constructs object containing fixed parameters (L, tau, data, etc) to feed into moment functions

    @unpack σHat, baseIndex, counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder,momentOrderForBaseIndex, StarDistributionType, useCDFforMarginalMatching = globParams
    @unpack μHat, wHat, λPrime, wPrimeHat, γHat, γPrimeHat, cHat = preStepOutput
    @unpack λData, LData, τData = data

    D = length(LData)

    indicators = (counterExplicit=counterExplicit,
        counterType=counterType,
        θConstant=θConstant,
        gravMoment=gravMoment,
        localGravityMoment=localGravityMoment,
        localGravityCrossMoment=localGravityCrossMoment,
        strongGravityMoment=strongGravityMoment,
        sameMarginalsMoment=sameMarginalsMoment,
        NoScalingforSameMartingale=NoScalingforSameMartingale,
        independenceMoment=independenceMoment,
        momentOrder=momentOrder,
        StarDistributionType=StarDistributionType,
        useCDFforMarginalMatching=useCDFforMarginalMatching)

    # remove the entry of w' that is the wage we are normalising to 1
    # as no point in optimising over this (will add it back inside the moment function)
    splice!(wPrimeHat, baseIndex)



    if counterType != 1

        θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat)


        if strongGravityMoment == 1
            θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, reshape(cHat, (D^2, 1))[:])
        end

        if sameMarginalsMoment == 1 && NoScalingforSameMartingale == 0
            θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, ones(D^2 - 1))
        end

        if sameMarginalsMoment == 1 && NoScalingforSameMartingale == 1
            θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat)
        end

        if independenceMoment == 1
            if NoScalingforSameMartingale == 0
                θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, ones(D^2))
            else
                θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, 1)
            end
        end

        γ = (wHat=wHat,
            L=LData,
            LPrime=LPrime,
            τ=τData,
            τPrime=τPrime,
            P=reshape(λData', (1, D^2)),
            PMM=PMM,
            baseIndex=baseIndex,
            indicators=indicators,
            Uσ=Uσ,
            Ū=Ū,
            Σ_od=Σ_od,
            Mτ=Mτ,
            μHat=μHat,
            D=D,
            CDF_X=CDF_X,
            CDF_Moments=CDF_Moments,
            Ind_Moments=Ind_Moments
        )

        return γ, θ_initial

    else

        θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])

        if strongGravityMoment == 1
            θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], reshape(cHat, (D^2, 1))[:])
        end

        if sameMarginalsMoment == 1 && NoScalingforSameMartingale == 0
            θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], ones(D^2 - 1))
        end

        if sameMarginalsMoment == 1 && NoScalingforSameMartingale == 1
            θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])
        end

        if independenceMoment == 1
            if NoScalingforSameMartingale == 0
                θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], ones(D^2))
            else
                θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], 1)
            end
        end

        γ = (wHat=wHat,
            L=LData,
            LPrime=LPrime,
            τ=τData,
            τPrime=τPrime,
            P=reshape(λData', (1, D^2)),
            PMM=PMM,
            baseIndex=baseIndex,
            indicators=indicators,
            wPrimeHat=wPrimeHat,
            Uσ=Uσ,
            Ū=Ū,
            Σ_od=Σ_od,
            Mτ=Mτ,
            μHat=μHat,
            D=D,
            CDF_X=CDF_X,
            CDF_Moments=CDF_Moments,
            Ind_Moments=Ind_Moments
        )

        return γ, θ_initial
    end

end

function LFD(θ_initial, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, counterType)
    # function that runs the CC outer loop 
    D = length(γ.L)
    W = size(U, 1)
    numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * ((2 * momentOrder) * D^2 + momentOrderForBaseIndex) + independenceMoment * D * (D^2 - floor(Int, D * (1 + D) / 2) + 1)

    if counterType != 1
        numMoments += (D - 1)
    end

    G = zeros(W, numMoments)
    K = zeros(W, 1)

    obj = PsiObjectiveBundleDelta(
        #δ = 1,
        #find_smallest = true,
        γ=γ,
        (moments!)=moments!,
        #moments_jacobian! = rust_moments_jacobian!,
        d=numMoments,
        outer_constr_index=numMoments + 1 - strongGravityMoment,
        inequality_index=Int64[],
        l=size(θ_initial, 1),
        U=U,
        N=100,
        outer_loop_opt="ek_outer_loop_options.opt",
        inner_loop_opt="ek_inner_loop_options.opt",
        lower_limit=-50)

    val, x, nStatus = inner_loop(obj, θ_initial)

    moments!(K, G, θ_initial, U, obj)

    arg0 = zeros(W, 1)
    LFD = zeros(W, 1)

    @show G[1, :]
    @show numMoments
    @show x


    meanLDF = 0
    for ω = 1:W
        arg0[ω] = -x[1] - dot(G[ω, 1:numMoments-strongGravityMoment], x[2:length(x)])
    end

    dPsi!(LFD, arg0)

    G = LFD .* G
    MomentsMean = mean(G, dims=1)
    @show MomentsMean
    return LFD
end

function checkParams(globParams)
    @unpack θHat, σHat, W, baseIndex, server, fakeData, DFake, seed,
    counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, independenceMoment, momentOrder, useParallel, InitDistributionType, InitDistributionCorr, InitDistributionParam, usePMM = globParams

    if strongGravityMoment + sameMarginalsMoment + independenceMoment > 1
        error("parameters are not compatible")
    end
end
function runMainClosedFormeFrechet(globParams)
    # this function runs the outer loop, using the same distribution for the F* and the initial point of the optimizer
    @unpack θHat, σHat, W, baseIndex, server, fakeData, DFake, seed,
    counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder,momentOrderForBaseIndex, useParallel, InitDistributionType, InitDistributionCorr, InitDistributionParam, usePMM, useCDFforMarginalMatching = globParams

    λData, LData, τData = importData(server, fakeData, DFake, seed)
    data = (λData=λData, LData=LData, τData=τData)
    τPrime, LPrime = defineCounter(τData, LData, counterType)
    D = length(LData)

    # get the pre step values for optimiser starting point, using closed form Frecht 
    preStepOutput = preStep(LData, LPrime, τData, τPrime, λData, θHat, σHat, baseIndex, counterType)


    # set seed
    Random.seed!(seed)

    # construct matrix of draws from exp(1)
    U = zeros(W, D * D)
    genExpRands!(U)

    @unpack μHat, wHat, cHat, λPrime, wPrimeHat, γHat, γPrimeHat = preStepOutput

    # Do μHat*(1-σHat) Moment Matching. This is the U power that enters the price expression
    doPriceMomentMatching = true
    Γ_μ_σ = gamma(1 + μHat * (1 - σHat))
    if doPriceMomentMatching
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

    # we keep a copy of U without exponents or scaling by cHat to calculate E[U^α_k] later
    Ū = zeros(W, D * D, 1 + sameMarginalsMoment * (momentOrder * 2 - 1)) # α_k \in [1,2, -1,....- momentOrder]

    Ū[:, :, 1] = U[:, :]

    # pre-calculate the powers of Ū for marginal matching with moments methodology
    if sameMarginalsMoment == 1 && useCDFforMarginalMatching == 0

        for k = 2:momentOrder
            α_k = k
            theoretical_moment = gamma(1 + α_k) # k->inf => theoretical_moment => inf 
            @. Ū[:, :, k] = U[:, :] .^ α_k
            @. Ū[:, :, k] = Ū[:, :, k] ./ theoretical_moment
        end

        for k = 1:momentOrder
            α_k = -1 + 1 / (1 + k) # k->inf => α_k => -1 
            theoretical_moment = gamma(1 + α_k) # k->inf => theoretical_moment => inf 
            @. Ū[:, :, k+momentOrder] = U[:, :] .^ α_k
            @. Ū[:, :, k+momentOrder] = Ū[:, :, k+momentOrder] ./ theoretical_moment
        end
    end

    CDF_X = zeros(1)
    CDF_X_for_ref = zeros(1)
    CDF_Moments = zeros(1, 1)
    CDF_Moments_for_ref = zeros(1, 1)
    # pre-calculate the quantiles for marginal matching with CDF methodology
    if sameMarginalsMoment == 1 && useCDFforMarginalMatching == 1
        CDF_X = quantile(Ū[:, 1, 1], range(1 / (2 * momentOrder - 2), (2 * momentOrder - 3) / (2 * momentOrder - 2), length=2 * momentOrder - 3))
        CDF_X_for_ref = quantile(Ū[:, 1, 1], range(1 / (momentOrderForBaseIndex - 1), (momentOrderForBaseIndex - 2) / (momentOrderForBaseIndex - 1), length= momentOrderForBaseIndex - 2))
    end
    # if marginal matching does not allow scaling then we can cach the moment condition and just copy them in the function "moments!"
    if sameMarginalsMoment == 1 && useCDFforMarginalMatching == 1 && NoScalingforSameMartingale == 1
        K = length(CDF_X) + 2
        CDF_Moments = zeros(W, (K + 1) * D^2 + momentOrderForBaseIndex)
        CDF_Moments_for_ref = zeros(W, momentOrderForBaseIndex)

        refIndex = 1 #do not change
        refIndex1 = refIndex + (refIndex - 1) * D
        CDF_11_X = zeros(W, K)
        CDF_11_X_for_ref = zeros(W, momentOrderForBaseIndex)

        for ω = 1:W
            smallest_X = searchsortedfirst(CDF_X, Ū[ω, refIndex1, 1]) + 1
            for i = smallest_X:K
                CDF_11_X[ω, i] = 1
            end

            smallest_X_for_ref = searchsortedfirst(CDF_X_for_ref, Ū[ω, refIndex1, 1]) + 1
            for i = smallest_X_for_ref:momentOrderForBaseIndex
                CDF_11_X_for_ref[ω, i] = 1
            end
        end

        # o= ref d = ref
        o1_ref = baseIndex + (baseIndex - 1) * D
        for i = 1:momentOrderForBaseIndex
            @. CDF_Moments_for_ref[:, i] += -CDF_11_X_for_ref[:, i]
        end
        for ω = 1:W
            smallest_X = searchsortedfirst(CDF_X_for_ref, Ū[ω, o1_ref, 1])

            for i = (smallest_X+1):momentOrderForBaseIndex
                CDF_Moments_for_ref[ω, i] += 1
            end
        end

        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                @. CDF_Moments[:, o1] = Ū[:, o1, 1] .- Ū[:, refIndex1, 1]

                for i = 1:K
                    @. CDF_Moments[:, o1+i*D^2] += -CDF_11_X[:, i]
                end
                for ω = 1:W
                    smallest_X = searchsortedfirst(CDF_X, Ū[ω, o1, 1])

                    for i = (smallest_X+1):K
                        CDF_Moments[ω, o1+i*D^2] += 1
                    end
                end
            end
        end

        @. CDF_Moments[:, (K+1)*D^2+1:end] += CDF_Moments_for_ref[:, :]
    end

    Ind_Moments = zeros(1, 1)
    #=
    if independenceMoment == 1 && NoScalingforSameMartingale == 1
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * (2 * momentOrder) * D^2
        Ind_Moments = zeros(W, D^4 - floor(Int, D^2 * (1 + D^2) / 2))

        idx_corss_moment = 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for c = 1:D
                    for f = 1:D
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        if c1 > o1
                            idx_corss_moment += 1
                            @. Ind_Moments[:, idx_corss_moment] = Ū[:, o1, 1] .* Ū[:, c1, 1]
                        end
                    end
                end
            end
        end
    end
    =#


    if independenceMoment == 1 && NoScalingforSameMartingale == 1
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * (2 * momentOrder) * D^2
        Ind_Moments = zeros(W, D * (D^2 - floor(Int, D * (1 + D) / 2)))

        idx_corss_moment = 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for c = 1:D
                    for f = 1:D
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        if c1 > o1 && d == f # condition on baseIndex only
                            idx_corss_moment += 1
                            @. Ind_Moments[:, idx_corss_moment] = Ū[:, o1, 1] .* Ū[:, c1, 1]
                        end
                    end
                end
            end
        end
    end

    #numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * (2 * momentOrder) * D^2 + independenceMoment * (D^4 - floor(Int, D^2 * (1 + D^2) / 2) + 1)

    numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * ((2 * momentOrder) * D^2 + momentOrderForBaseIndex) + independenceMoment * D * (D^2 - floor(Int, D * (1 + D) / 2) + 1)


    if counterType != 1
        numMoments += (D - 1)
    end

    # # this is to verify the moments are written where they are supposed to
    MomentNames = fill("", numMoments)
    if counterType != 1
        cInd = D^2 + D - 1
    else
        cInd = D^2
    end

    if counterType != 1
        bInd = D^2
        cInd = D^2 + D - 1
        dInd = D^2 + 2 * D - 1
    else
        cInd = D^2
        dInd = D^2 + D
    end


    for d = 1:D
        for o = 1:D # using min prices, compute implied expenditure share and fill in G with implied minus data 
            d1 = d + (o - 1) * D
            MomentNames[d1] = "lambda [$o ,$d]"
        end
        MomentNames[cInd+d] = "gamma [$d]"
        if counterType != 1
            MomentNames[dInd+d] = "gammaPrime [$d]"
        else
            if d == baseIndex
                MomentNames[dInd+d] = "gammaPrime [$d]"
            end
        end

        if counterType != 1
            MomentNames[bInd+d-1] = "WagePrime [$d] "
        end
    end

    if strongGravityMoment == 1
        MomentNames[end] = "Strong Gravity"
    end

    for d = 1:D
        counter = 0
        if localGravityMoment == 1
            for o = 1:D
                counter += 1
                if o != d
                    moment_idx = (D - 1) * (d - 1) + counter
                    MomentNames[end-strongGravityMoment-gravMoment-moment_idx] = "Local ACR [$o, $d]"
                end
            end
        end
    end

    for d = 1:D
        counter = 0
        if localGravityCrossMoment == 1
            for o = 1:D
                if o != d
                    for c = 1:D
                        if c != o && c != d
                            counter += 1
                            moment_idx = localGravityMoment * D * (D - 1) + (D - 1) * (D - 2) * (d - 1) + counter
                            MomentNames[end-strongGravityMoment-gravMoment-moment_idx] = "Local ACR [$o, $d , $c]"
                        end
                    end
                end
            end
        end
    end


    if gravMoment == 1
        # not implemented
    end


    if sameMarginalsMoment == 1 && useCDFforMarginalMatching == 0
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2)

        refIndex = 1 #do not change
        refIndex1 = refIndex + (refIndex - 1) * D
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 

                #+ve moments

                MomentNames[end-offset-(1-1)*D*D-o1] = "Same Marginals [$o, $d , k+= 1]"
                for k = 2:momentOrder
                    MomentNames[end-offset-(k-1)*D*D-o1] = "Same Marginals [$o, $d , k+= $k]"
                end
                #-ve moments

                MomentNames[end-offset-(momentOrder+1-1)*D*D-o1] = "Same Marginals [$o, $d , k-= 1]"
                for k = 2:momentOrder
                    MomentNames[end-offset-(momentOrder+k-1)*D*D-o1] = "Same Marginals [$o, $d , k+= $k]"
                end
            end
        end
    elseif sameMarginalsMoment == 1 && useCDFforMarginalMatching == 1 && NoScalingforSameMartingale == 0
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2)

        #CDF_X is a vector of increasing values of X. 
        # the moment condition is CDF_od(X) = CDF_11(X)
        # E[U_od<=X_k] = E[U_11<=X_k]

        K = length(CDF_X) + 2
        refIndex = 1 #do not change
        refIndex1 = refIndex + (refIndex - 1) * D
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                MomentNames[end-offset-o1] = "Same Marginals CDF [$o, $d , First Moment]"

                for i = 1:K
                    MomentNames[end-offset-i*D^2-o1] = "Same Marginals CDF [$o, $d , k= $i]"
                end
            end
        end
    elseif sameMarginalsMoment == 1 && useCDFforMarginalMatching == 1 && NoScalingforSameMartingale == 1
        K_ = size(CDF_Moments, 2)
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2)
        for o = 1:D
            for d = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                MomentNames[end-offset-K_+o1] = "Same Marginals CDF [$o, $d , First Moment]"
                for i = 1:K
                    MomentNames[end-offset-K_+o1+i*D^2] = "Same Marginals CDF [$o, $d , k= $i]"
                end
            end
        end
    end

    if independenceMoment == 1 && NoScalingforSameMartingale == 1
        offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * ( (2 * momentOrder) * D^2 + momentOrderForBaseIndex)

        MomentNames[end-offset] = "Uncorrelation [1, 1, First Moment Value]"

        K_ = size(Ind_Moments, 2)

        idx_corss_moment = 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for c = 1:D
                    for f = 1:D
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        #if c1 > o1
                        if c1 > o1 && d == f
                            idx_corss_moment += 1
                            MomentNames[end-offset-1-K_+idx_corss_moment] = "Uncorrelation [ $o, $d, $c , $f]"
                        end
                    end
                end
            end
        end
    end



    file_name = string("Counter_", counterType, "_countries_", D, "_baseI", baseIndex, "_sGrav", strongGravityMoment, "_lGrav", localGravityMoment, "_Marg", sameMarginalsMoment, "_NoSc", NoScalingforSameMartingale, "_ind", independenceMoment, "_order", momentOrder, "useCDF_", useCDFforMarginalMatching, "_Frechet", "_", Dates.format(now(), "y-m-d"), ".csv")

    writedlm(string("MomentNames_", file_name, ".csv"), MomentNames, ',')


    #  @show Ū[1,:,:]

    for d = 1:D
        for o = 1:D
            o1 = o + (d - 1) * D
            @. U[:, o1] = U[:, o1] .* cHat[o, d]
        end
    end


    Σ_od = zeros(D, D)
    for d = 1:D
        for o = 1:D
            o1 = o + (d - 1) * D
            Σ_od[o, d] = sqrt(var(Ū[:, o1, 1]))
        end
    end


    if θConstant == 1
        @. U[:] = U[:] .^ (-μHat) # if theta doesn't vary, then much faster to precalculate this matrix 
        # reduction in computation time due to non-integer exponent substantially dominates higher memory usage
    end

    Uσ = U .^ (1 - σHat) # precalculate 

    #pre calculate the Mτ

    Mτ = calcMτ(τData, D)
    PMM = zeros(numMoments)

    γ, θ_initial = buildObjectsForMoments(globParams, preStepOutput, data, LPrime, τPrime, Uσ, Ū, Σ_od, Mτ, μHat, PMM, CDF_X, CDF_Moments, Ind_Moments)

    @show θ_initial

    @show Dates.format(now(), "HH:MM") # print time 


    δ_grid, Θ_upper, κ_upper, Θ_lower, κ_lower = ccOuter(θ_initial, θ_initial, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, counterType, useParallel, file_name) # run the outer loop 
    writedlm(file_name, [δ_grid κ_lower κ_upper], ',')

    # store the parameters
    writedlm(string("Theta_initial_Frechet", "_", file_name, ".csv"), θ_initial, ',')

    for i = 1:length(δ_grid)
        writedlm(string("Theta_upper_", δ_grid[i], "_", file_name, ".csv"), Θ_upper[:, i], ',')
        writedlm(string("Theta_lower_", δ_grid[i], "_", file_name, ".csv"), Θ_lower[:, i], ',')
    end

    calculateLFD = true
    if calculateLFD
        LFD_upper = zeros(W, length(δ_grid))
        LFD_lower = zeros(W, length(δ_grid))
        for i = 1:length(δ_grid)
            LFD_upper[:, i] = LFD(Θ_upper[:, i], U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, counterType)
            LFD_lower[:, i] = LFD(Θ_lower[:, i], U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, counterType)
        end
        writedlm(string("LFD_up_", file_name, ".csv"), LFD_upper, ',')
        writedlm(string("LFD_low_", file_name, ".csv"), LFD_lower, ',')
    end

    @show Θ_upper[:, 1]
    @show Dates.format(now(), "HH:MM") # print time 

end

function runLFD(globParams)
    # this function runs the outer loop, using the same distribution for the F* and the initial point of the optimizer
    @unpack θHat, σHat, W, baseIndex, server, fakeData, DFake, seed,
    counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, independenceMoment, momentOrder, useParallel, InitDistributionType, InitDistributionCorr, InitDistributionParam, usePMM = globParams

    λData, LData, τData = importData(server, fakeData, DFake, seed)
    data = (λData=λData, LData=LData, τData=τData)
    τPrime, LPrime = defineCounter(τData, LData, counterType)
    D = length(LData)

    # get the pre step values for optimiser starting point, using closed form Frecht 
    preStepOutput = preStep(LData, LPrime, τData, τPrime, λData, θHat, σHat, baseIndex, counterType)


    # set seed
    Random.seed!(seed)

    # construct matrix of draws from exp(1)
    U = zeros(W, D * D)
    genExpRands!(U)

    @unpack μHat, wHat, cHat, λPrime, wPrimeHat, γHat, γPrimeHat = preStepOutput

    # Do μHat*(1-σHat) Moment Matching. This is the U power that enters the price expression
    doPriceMomentMatching = true
    Γ_μ_σ = gamma(1 + μHat * (1 - σHat))
    if doPriceMomentMatching
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

    Ū[:, :, 1] = U[:, :]


    @show Dates.format(now(), "HH:MM") # print time 


    sourcefilename = "GT_countries_4_baseI2_sGrav0_lGrav0_Marg1_NoSc1_ind1_order5useCDF_1_Frechet_4-1-14.csv"
    LFD_upper = readdlm(string(folderData, "/LFD_up_", sourcefilename), ',') # import data from csv
    LFD_lower = readdlm(string(folderData, "/LFD_low_", sourcefilename), ',') # import data from csv

    sourcefilename = "GT_countries_4_baseI2_sGrav0_lGrav0_Marg1_NoSc1_ind1_order5useCDF_1_Frechet_4-1-14.csv"

    function mCDF(x, o, d, i, up_down)
        o1 = o + (d - 1) * D
        mcdf = zeros(length(x))
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W
            for j = 1:length(x)
                mcdf[j] += Ū[ω, o1, 1] < x[j] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W : 0
            end
        end
        return mcdf / nornamization_factor
    end

    function JointPricesCDF(x, d, i, up_down)
        o1 = 1 + (d - 1) * D
        o2 = D + (d - 1) * D
        jcdf = zeros(length(x), length(x))
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W
            x_ω = Ū[ω, o1:o2, 1]
            price_baseIndex = x_ω[d] / λData[d, d]
            price_rw = minimum(x_ω[Not(d)] ./ λData[Not(d), d])
            for i1 = 1:length(x)
                for i2 = 1:length(x)
                    jcdf[i1, i2] += (price_baseIndex < x[i1] && price_rw < x[i2]) ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W : 0
                end
            end
        end
        return jcdf / nornamization_factor
    end

    function MarginalPricesCDF(x, d, i, up_down)
        o1 = 1 + (d - 1) * D
        o2 = D + (d - 1) * D
        pcdf = zeros(length(x), 3)
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W
            for j = 1:length(x)
                x_ω = Ū[ω, o1:o2, 1]
                price_baseIndex = x_ω[d] / λData[d, d]
                price_rw = minimum(x_ω[Not(d)] ./ λData[Not(d), d])
                pcdf[j, 1] += price_baseIndex < x[j] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W : 0
                pcdf[j, 2] += price_rw < x[j] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W : 0
                pcdf[j, 3] += price_baseIndex / price_rw < x[j] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W : 0
            end
        end
        return pcdf / nornamization_factor
    end


    function MarginalPricesPDF(x, d, i, up_down)
        o1 = 1 + (d - 1) * D
        o2 = D + (d - 1) * D
        ppdf = zeros(length(x) - 1, 3)
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W
            for j = 1:length(x)-1
                x_ω = Ū[ω, o1:o2, 1]
                price_baseIndex = x_ω[d] / λData[d, d]
                price_rw = minimum(x_ω[Not(d)] ./ λData[Not(d), d])
                ppdf[j, 1] += price_baseIndex >= x[j] && price_baseIndex <= x[j+1] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / (W * (x[j+1] - x[j])) : 0
                ppdf[j, 2] += price_rw >= x[j] && price_rw <= x[j+1] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / (W * (x[j+1] - x[j])) : 0
                ppdf[j, 3] += price_baseIndex / price_rw >= x[j] && price_baseIndex / price_rw <= x[j+1] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / (W * (x[j+1] - x[j])) : 0
            end
        end
        return ppdf / nornamization_factor
    end

    function PricesCorrelation(d, i)
        o1 = 1 + (d - 1) * D
        o2 = D + (d - 1) * D
        pmean = zeros(3, 2)
        pcorr = zeros(3)
        for ω = 1:W
            x_ω = Ū[ω, o1:o2, 1]
            price_baseIndex = x_ω[d] / λData[d, d]
            price_rw = minimum(x_ω[Not(d)] ./ λData[Not(d), d])

            pmean[1, 1] += price_baseIndex / W
            pmean[2, 1] += LFD_upper[ω, i] * price_baseIndex / W
            pmean[3, 1] += LFD_lower[ω, i] * price_baseIndex / W

            pmean[1, 2] += price_rw / W
            pmean[2, 2] += LFD_upper[ω, i] * price_rw / W
            pmean[3, 2] += LFD_lower[ω, i] * price_rw / W

            pcorr[1] += price_rw * price_baseIndex / W
            pcorr[2] += LFD_upper[ω, i] * price_rw * price_baseIndex / W
            pcorr[3] += LFD_lower[ω, i] * price_rw * price_baseIndex / W
        end

        pcorr[1] -= pmean[1, 1] * pmean[1, 2]
        pcorr[2] -= pmean[2, 1] * pmean[2, 2]
        pcorr[3] -= pmean[3, 1] * pmean[3, 2]

        return (pcorr, pmean)
    end

    function correlationMatrix(i, up_down)

        U_Means = zeros(D^2)
        U_Vars = zeros(D^2)
        corrMatrix = zeros(D^2, D^2)
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / W
        end


        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for ω = 1:W

                    U_Means[o1] += Ū[ω, o1, 1] * (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / (W * nornamization_factor)
                    U_Vars[o1] += Ū[ω, o1, 1] * Ū[ω, o1, 1] * (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / (W * nornamization_factor)
                end
            end
        end

        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                U_Vars[o1] = U_Vars[o1] - U_Means[o1] * U_Means[o1]

            end
        end

        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o               
                for c = 1:D
                    for f = 1:D
                        if c > o || f > d
                            c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                            for ω = 1:W
                                corrMatrix[o1, c1] += (Ū[ω, o1, 1]) .* (Ū[ω, c1, 1]) * (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) / (W * nornamization_factor) - U_Means[o1] * U_Means[c1] / W
                            end
                        end
                    end
                end
            end
        end

        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o               
                for c = 1:D
                    for f = 1:D
                        if c > o || f > d
                            c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                            corrMatrix[o1, c1] = corrMatrix[o1, c1] / sqrt(U_Vars[o1] * U_Vars[c1])
                        end
                    end
                end
            end
        end
        return corrMatrix
    end

    u = range(0, 2, length=100)
    u_large = range(0, 50, length=100)

    #δ_grid = [0.01, 0.1, 0.5, 1, 2]
    δ_grid = [1]

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
        y11 = mCDF(u_large, 1, 1, i, 0)
        y11l = mCDF(u_large, 1, 1, i, -1)
        y11u = mCDF(u_large, 1, 1, i, 1)
        y22 = mCDF(u_large, 1, 1, i, 0)
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
        y11 = mCDF(u, 1, 1, i, 0)
        y11l = mCDF(u, 1, 1, i, -1)
        y11u = mCDF(u, 1, 1, i, 1)
        y22 = mCDF(u, 1, 1, i, 0)
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

function runMainGeneric(globParams)
    # this function runs the outer loop, using the same distribution for the F* and the initial point of the optimizer
    @unpack θHat, σHat, W, baseIndex, server, fakeData, DFake, seed,
    counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, independenceMoment, momentOrder, useParallel, InitDistributionType, InitDistributionCorr, InitDistributionParam, StarDistributionType, StarDistributionCorr, StarDistributionParam, usePMM = globParams

    checkParams(globParams)

    λData, LData, τData = importData(server, fakeData, DFake, seed)
    data = (λData=λData, LData=LData, τData=τData)
    τPrime, LPrime = defineCounter(τData, LData, counterType)
    D = length(LData)

    # generate the realizations of the StarDistribution (F* in CC)
    Random.seed!(seed)
    U = zeros(W, D * D)
    genRands!(U, StarDistributionType, StarDistributionCorr, StarDistributionParam, 0)
    #to get the cHats that matche trade shares 
    preStepOutput_FStar = preStepGeneralDistribution(LData, LPrime, τData, τPrime, λData, θHat, σHat, baseIndex, counterType, U)

    cHat_Star = preStepOutput_FStar.cHat
    μHat_Star = preStepOutput_FStar.μHat

    for d = 1:D
        for o = 1:D
            o1 = o + (d - 1) * D
            @. U[:, o1] = U[:, o1] .* cHat_Star[o, d]
        end
    end

    # we keep a copy of U without exponents to calculate E[U] later
    Ū = zeros(W, D * D)
    Ū[:] = U[:]

    Σ_od = zeros(D, D)
    for d = 1:D
        for o = 1:D
            o1 = o + (d - 1) * D
            Σ_od[o, d] = sqrt(var(Ū[:, o1]))
        end
    end


    # Get the pre step values for optimiser starting point (θ_initial)
    preStepOutput_FInit = preStepGeneralDistribution(LData, LPrime, τData, τPrime, λData, θHat, σHat, baseIndex, counterType, InitDistributionType, InitDistributionCorr, InitDistributionParam)

    @unpack μHat, wHat, cHat, λPrime, wPrimeHat, γHat, γPrimeHat = preStepOutput_FInit


    if θConstant == 1

        @. U[:] = U[:] .^ (-μHat) # if theta doesn't vary, then much faster to precalculate this matrix 
    end

    Uσ = U .^ (1 - σHat) # precalculate 


    Mτ = calcMτ(τData, D)
    print(MartingalDifferenceDivergence(cHat, Mτ, D))

    numMoments = counterType == 1 ? D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (3 + D^2) + sameMarginalsMoment * (2 + momentOrder) * D^2 + independenceMoment * D * D * momentOrder * (1 + D * momentOrder) : D^2 + (D - 1) + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + strongGravityMoment * (1 + D^2) + sameMarginalsMoment * (2 + momentOrder) * D^2 + independenceMoment * D * D * momentOrder * (1 + D * momentOrder)

    PMM = zeros(numMoments)
    ΣMM = zeros(numMoments, numMoments)


    #auxiliary variable
    X = zeros(W, D)
    genExpRands!(X)


    γ, θ_initial = buildObjectsForMoments(globParams, preStepOutput_FInit, data, LPrime, τPrime, Uσ, Ū, X, Σ_od, Mτ, μHat_Star, PMM)
    γStar, θ_Star = buildObjectsForMoments(globParams, preStepOutput_FStar, data, LPrime, τPrime, Uσ, Ū, X, Σ_od, Mτ, μHat_Star, PMM)

    if counterType == 1

        vcHat = reshape(cHat, (D^2, 1))[:]
        vcHat_Star = reshape(cHat_Star, (D^2, 1))[:]
        @show vcHat
        @show vcHat_Star
        if sameMarginalsMoment == 1
            @. θ_initial[3+2*D:3+2*D+D^2-1] = θ_initial[3+2*D:3+2*D+D^2-1] .* vcHat[:] ./ vcHat_Star[:]
        end
        @show θ_initial
    end

    @show Dates.format(now(), "HH:MM") # print time 

    file_name = string("InitDistributionType", InitDistributionType, "_narrow_countries_", D, "_baseIndex", baseIndex, "_fixMu_gravMoment", gravMoment, "_strongGravityMoment", strongGravityMoment, "_localGravityMoment", localGravityMoment, "_PMM", usePMM, "_new2samemarginals", sameMarginalsMoment, "_independenceMoment", independenceMoment, "_order", momentOrder, "_", Dates.format(now(), "y-m-d"), ".csv")
    δ_grid, Θ_upper, κ_upper, Θ_lower, κ_lower = ccOuter(θ_initial, θ_Star, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, independenceMoment, momentOrder, counterType, useParallel, file_name) # run the outer loop 
    writedlm(file_name, [δ_grid κ_lower κ_upper], ',')



    calculateLFD = true

    if calculateLFD
        LFD_upper = zeros(W, length(δ_grid))
        LFD_lower = zeros(W, length(δ_grid))
        for i = 1:length(δ_grid)
            LFD_upper[:, i] = LFD(Θ_upper[:, i], U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, independenceMoment, momentOrder, counterType)
            LFD_lower[:, i] = LFD(Θ_lower[:, i], U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, strongGravityMoment, sameMarginalsMoment, independenceMoment, momentOrder, counterType)
        end
        file_name = string("LFD_InitDistributionType", InitDistributionType, "_narrow_countries_", D, "_baseIndex", baseIndex, "_fixMu_gravMoment", gravMoment, "_strongGravityMoment", strongGravityMoment, "_localGravityMoment", localGravityMoment, "_PMM", usePMM, "_new2samemarginals", sameMarginalsMoment, "_independenceMoment", independenceMoment, "_order", momentOrder, "_", Dates.format(now(), "y-m-d"), ".csv")
        writedlm(file_name, LFD_upper, ',')

        file_name = string("InitDistributionType", InitDistributionType, "_narrow_countries_", D, "_baseIndex", baseIndex, "_fixMu_gravMoment", gravMoment, "_strongGravityMoment", strongGravityMoment, "_localGravityMoment", localGravityMoment, "_PMM", usePMM, "_new2samemarginals", sameMarginalsMoment, "_independenceMoment", independenceMoment, "_order", momentOrder, "_", Dates.format(now(), "y-m-d"), ".csv")
        writedlm(file_name, LFD_lower, ',')
    end

    @show Θ_upper[:, 1]
    @show Dates.format(now(), "HH:MM") # print time 

end


## Define global parameters
globParams = (θHat=0,
    σHat=2.5, # CES elasticity, maintained throughout 
    W=80000, # num draws 
    baseIndex=2, # country to use as wage normalisation and counterfactual 
    server=1, # 1 if using server, 0 otherwise (uses server file path if 1)
    fakeData=1, # 1 to generate data, 0 to use from files
    DFake=16, # if using fake data, number of countries to gen data
    seed=888, # seed for simulations (different from fake data seed)
    counterType=1, # 0 = zero gravity, 1 = autarky, 2 = custom (adjust above)
    counterExplicit=0, # 1 = explicit counterfactuals (uses kappaStar), 0 = implicit
    θConstant=0, # put 1 if theta and sigma never vary, will precalculate U^((1-sigma)/theta)
    gravMoment=0, # = 1 impose gravity identification for Frechet, 0 = do not
    localGravityMoment=0, # = 1 impose model implied trade elasticity mtaches θHat, 0 = do not 
    localGravityCrossMoment=0, # = 1 impose model implied trade cross elasticity is zero, 0 = do not 
    strongGravityMoment=0, # = 1 imposes mean independence between lnU and ln tau , 0 = do not
    sameMarginalsMoment=1, # =1 imposes all od pairs have the same U distribution
    NoScalingforSameMartingale=1,# =1 imposes a strict same marginal condition, without allowing for a multiplicative dergree of freedom   
    independenceMoment=0, # =1 imposes correlation[Uod, Uo'd] = 0 
    momentOrder=5, # number of moment conditions to approximate same marginal condition 
    momentOrderForBaseIndex = 50, # number of moment conditions to approximate same marginal condition for U_baseIndex,baseIndex
    useCDFforMarginalMatching=1, # 1= impose same marginal condition using CDF, 0= using moments 
    useParallel=1, # 1 = parallelise the deltas in outer loop; 0 = do not 
    InitDistributionType=1, # 0 = Frechet, 1 = lognormal, 2= t-dist, 3 = flexible (see genRands. uses corr and Param), 4 = LN productivity correlated with trade costs
    InitDistributionCorr=0.0, #correlation between countries
    InitDistributionParam=1, #parameter of the distribution (Not used for Frechet & LN)
    StarDistributionType=0, # 0 = Frechet, 1 = lognormal, 2= t-dist, 3 = flexible (see genRands. uses corr and Param), 4 = LN productivity correlated with trade costs
    StarDistributionCorr=0.0, #correlation between countries
    StarDistributionParam=1, #parameter of the distribution (Not used for Frechet & LN)
    usePMM=0, # not really implemented anymore should be removed maybe)
)
@show Dates.format(now(), "HH:MM")

runMainClosedFormeFrechet(globParams)
#runLFD(globParams)

@show Dates.format(now(), "HH:MM")
