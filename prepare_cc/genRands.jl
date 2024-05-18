
function genExpRands!(U)
    # draw from exp(1)
    rand!(U)
    for i in 1:length(U)
        U[i] = -log(1 - U[i])
    end
end

function genExpRandsStratified!(U, StratifiedSamplingWeight)
    # draw from exp(1)
    D2 = size(U, 2)
    W = size(U, 1)
    Half_W = floor(Int, W/ 2)
    strata_size = floor(Int, Half_W/D2)
    # !!!! this assumes that W/(2*D^2) is an integer... 

    rand!(U)

    # We stratify U on two intervals [0, 0.1] & [0.1, 1]. We therefore have to weight the realizations by w_1 = 0.1/0.5 and w_2 = 0.9/0.5 when calculating expectations

    @. StratifiedSamplingWeight[1:strata_size] = 0.1./0.5
    @. StratifiedSamplingWeight[strata_size+1:end] = 0.9./0.5

    strata_index = 0
    for od = 1:D2
        @. U[1+strata_index*strata_size:(strata_index+1)*strata_size,od] =  0.1 .* U[1+strata_index*strata_size:(strata_index+1)*strata_size,od]
        @. U[Half_W+1+strata_index*strata_size:(strata_index+1)*strata_size,od] = 0.1 .+ 0.9 .* U[Half_W+1+strata_index*strata_size:(strata_index+1)*strata_size,od]
        strata_index += 1
    end

    for i in 1:length(U)
        U[i] = -log(1 - U[i])
    end


end

function genExpRandsImportanceSampling!(U, λ, ImportanceSampleingWeight)
    # draw from exp(1)
    rand!(U)
    #λ >>1 -> Increase the number of realizations where U is small
    for i in 1:length(U)
        U[i] = -log(1 - U[i])/λ
    end
    
    W = size(U, 1)
    D2 = size(U, 2)
    ImportanceSampleingWeight = ones(W)
    for od = 1:D2
       @. ImportanceSampleingWeight[:] = ImportanceSampleingWeight[:] .* exp.((λ-1) .* U[:,od]) ./ λ
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