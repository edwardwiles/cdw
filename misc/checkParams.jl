function checkParams(globParams)
    @unpack counterType, sameMarginalsMoment, independenceMoment, usePMM, importanceSampling,use_Jacobian  = globParams

    if sameMarginalsMoment == 0  && independenceMoment == 1
        error("sameMarginalsMoment must be =1 with independenceMoment = 1")
    elseif  importanceSampling != 0
        error("Option not yet handled correctly")
    elseif counterType != 1 && calculateLFD == 1
        error("Option not yet handled correctly")
    elseif counterType != 1 && use_Jacobian == 1
        error("Option not yet handled correctly")
    elseif counterType == 1 && counterExplicit != 0
        error("counterType = 1 means counterExplicit = 0")
    elseif useFrechetCopulaStartingPoint ==1 && useFiniteSamplePrestep == 1
        error("cant Frechet Copula and FiniteSample initial parameters")
    end
end