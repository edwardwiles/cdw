function ccInner(prep_output, params)
    #runs the CC Inner loop 
    @unpack GravityMomentFirstApproach, counterType, useParallel, EK_moments!, θConstant = params
	@unpack θ_initial, U, γ, numMoments, δ_grid, outer_constr_index = prep_output

    obj = PsiObjectiveBundleDelta(
        #δ = 1,
        #find_smallest = true,
        γ=γ,
        (moments!)=EK_moments!,
        #moments_jacobian! = rust_moments_jacobian!,
        d=numMoments,
        outer_constr_index=outer_constr_index,
        inequality_index=Int64[],
        l=size(θ_initial, 1),
        U=U,
        #N=1000,
        outer_loop_opt="ek_outer_loop_options.opt",
        inner_loop_opt="ek_inner_loop_options.opt",
        lower_limit=-5000000)

    val, x, nStatus = inner_loop(obj, θ_initial)

    return (val, x, nStatus)
end