
function ccOuter(θ_initial_all, θ_Star_all, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, momentOrderForBaseIndex,useIndependentCFDs,IndMomentOrder, counterType, useParallel, file_name)
    # function that runs the CC outer loop 
    D = length(γ.L)

    numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (1 + (1-sameMarginalsMoment)*D^2) + sameMarginalsMoment * ((2 * momentOrder + 1) * D^2 + momentOrderForBaseIndex) + independenceMoment * (1+ D * (D^2 - floor(Int, D * (1 + D) / 2))+ (1-useIndependentCFDs)*(D-1)* 2 * momentOrder + useIndependentCFDs*((IndMomentOrder-1) + (IndMomentOrder-1)^D))

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
    if GravityMomentFirstApproach == 1 || (sameMarginalsMoment == 1 && NoScalingforSameMartingale == 0)
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
                outer_constr_index=numMoments + 1 - GravityMomentFirstApproach,
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
                outer_constr_index=numMoments + 1 - GravityMomentFirstApproach,
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
                    outer_constr_index=numMoments + 1 - GravityMomentFirstApproach,
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
                    outer_constr_index=numMoments + 1 - GravityMomentFirstApproach,
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
                    outer_constr_index=numMoments + 1 - GravityMomentFirstApproach,
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
                    outer_constr_index=numMoments + 1 - GravityMomentFirstApproach,
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