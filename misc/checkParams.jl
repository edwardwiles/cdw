function checkParams(globParams)
    @unpack θHat, σHat, W, baseIndex, server, fakeData, DFake, seed,
    counterType, counterExplicit, θConstant, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, independenceMoment, momentOrder, useParallel, InitDistributionType, InitDistributionCorr, InitDistributionParam, usePMM = globParams

    if GravityMomentFirstApproach + sameMarginalsMoment + independenceMoment > 1
        error("parameters are not compatible")
    end
end