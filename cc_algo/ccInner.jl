function ccInner(θ_initial, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, independenceMoment, momentOrder, momentOrderForBaseIndex,useIndependentCFDs, IndMomentOrder, counterType)
    # function that runs the CC outer loop 
    D = length(γ.L)

    numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (1 + (1-sameMarginalsMoment)*D^2) + sameMarginalsMoment * ((2 * momentOrder + 1) * D^2 + momentOrderForBaseIndex) + independenceMoment * (1+ D * (D^2 - floor(Int, D * (1 + D) / 2))+ (1-useIndependentCFDs)*(D-1)* 2 * momentOrder + useIndependentCFDs*((IndMomentOrder-1) + (IndMomentOrder-1)^D))

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
        outer_constr_index=numMoments + 1 - GravityMomentFirstApproach,
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