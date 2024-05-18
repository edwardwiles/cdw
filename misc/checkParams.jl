function checkParams(globParams)
    @unpack counterType, sameMarginalsMoment, independenceMoment, usePMM, importanceSampling = globParams

    if sameMarginalsMoment == 0  && independenceMoment == 1
        error("sameMarginalsMoment must be =1 with independenceMoment = 1")
    elseif usePMM != 0 || counterType != 1 || importanceSampling != 0
        error("Option not yet handled correctly")
    end
end