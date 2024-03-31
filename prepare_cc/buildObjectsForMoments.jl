function buildObjectsForMoments(globParams, prestep_output, data, LPrime, τPrime, Uσ, Ū, Σ_od, Mτ, PMM=zeros(1), CDF_X=zeros(1), CDF_Moments=zeros(1), Ind_Moments=zeros(1), IndCDF_Cells =Vector{Vector{Int}}(undef,1), importanceSamplingWeights = ones(1) )
    # constructs object containing fixed parameters (L, tau, data, etc) to feed into moment functions

    @unpack σHat, baseIndex, counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder,momentOrderForBaseIndex, StarDistributionType, useCDFforMarginalMatching,stratifiedSampling, useIndependentCFDs,IndMomentOrder = globParams
    @unpack μHat, wHat, λPrime, wPrimeHat, γHat, γPrimeHat, cHat = prestep_output
    @unpack λData, LData, τData = data

    D = length(LData)

    indicators = (counterExplicit=counterExplicit,
        counterType=counterType,
        θConstant=θConstant,
        gravMoment=gravMoment,
        localGravityMoment=localGravityMoment,
        localGravityCrossMoment=localGravityCrossMoment,
        GravityMomentFirstApproach=GravityMomentFirstApproach,
        sameMarginalsMoment=sameMarginalsMoment,
        NoScalingforSameMartingale=NoScalingforSameMartingale,
        independenceMoment=independenceMoment,
        momentOrder=momentOrder,
        StarDistributionType=StarDistributionType,
        useCDFforMarginalMatching=useCDFforMarginalMatching,
        momentOrderForBaseIndex = momentOrderForBaseIndex,
        stratifiedSampling = stratifiedSampling,
        IndMomentOrder = IndMomentOrder,
        useIndependentCFDs = useIndependentCFDs)

    # remove the entry of w' that is the wage we are normalising to 1
    # as no point in optimising over this (will add it back inside the moment function)
    splice!(wPrimeHat, baseIndex)



    if counterType != 1

        θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat)


        if GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0
            θ_initial = vcat(μHat, σHat, γHat, γPrimeHat, wPrimeHat, ones(D^2 - 1))
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
            elseif useIndependentCFDs == 1
                θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], 1, range(1/IndMomentOrder, (IndMomentOrder-1) / IndMomentOrder, length=IndMomentOrder-1))
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
            Ind_Moments=Ind_Moments,
            cHat = cHat,
            IndCDF_Cells = IndCDF_Cells,
            importanceSamplingWeights = importanceSamplingWeights
        )

        return γ, θ_initial

    else

        θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex])

        if GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0
            #θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], reshape(cHat, (D^2, 1))[:])
            θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], ones(D^2 - 1))
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
            elseif useIndependentCFDs == 1
                θ_initial = vcat(μHat, σHat, γHat, γPrimeHat[baseIndex], 1, range(1/IndMomentOrder, (IndMomentOrder-1) / IndMomentOrder, length=IndMomentOrder-1))
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
            Ind_Moments=Ind_Moments,
            cHat = cHat,
            IndCDF_Cells = IndCDF_Cells,
            importanceSamplingWeights = importanceSamplingWeights
        )

        return γ, θ_initial
    end

end