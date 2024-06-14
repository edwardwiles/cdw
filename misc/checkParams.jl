function checkParams(globParams)
    @unpack counterType, sameMarginalsMoment, independenceMoment, usePMM, importanceSampling,  = globParams

    if sameMarginalsMoment == 0  && independenceMoment == 1
        error("sameMarginalsMoment must be =1 with independenceMoment = 1")
    elseif  importanceSampling != 0
        error("Option not yet handled correctly")
    elseif counterType != 1 && calculateLFD == 1
        error("Option not yet handled correctly")
    end
end