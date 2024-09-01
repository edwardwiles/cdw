
function ccOuter(prep_output, params)
	# function that runs the CC outer loop
	@unpack GravityMomentFirstApproach, counterType, useParallel, EK_moments!, EK_moments_Jacobian!, θConstant, use_Jacobian, Jac_W, independenceMoment, IndMomentOrder, OuterScaling, UoModel, sameMarginalsMoment, ForceFrechetMarginal, baseIndex = params
	@unpack θ_initial, θ_initial_low, θ_initial_up, U, γ, numMoments, δ_grid, outer_constr_index, inequality_index, nTotalMoments, file_name, complement_index = prep_output
	D = length(γ.L)
	δ_grid_size = length(δ_grid)

	#θ_lower = (θ_initial.*0.5)[:] # lower bound for parameters in outer loop optimisation 
	#θ_upper = (θ_initial.*1.5)[:] # upper bound for parameters in outer loop optimisation 

	θ_lower = (θ_initial.*0.0001)[:] # lower bound for parameters in outer loop optimisation 
	θ_upper = (θ_initial.*10000)[:] # upper bound for parameters in outer loop optimisation 


	θ_lower[2] = θ_initial[2] # fix sigma (second param) as not identified anyway
	θ_upper[2] = θ_initial[2]

	# for μ have the largest bounds such that prices are defined under Frechet
	θ_upper[1] = 1 / (θ_initial[2] - 1)
	θ_lower[1] = 0

	# if fixed μ
	if θConstant == 1
		θ_lower[1] = θ_initial[1]
		θ_upper[1] = θ_initial[1]
	else
		θ_upper[1] = min(θ_upper[1], 1 / (θ_initial[2] - 1))
	end


	# Cap CDF probas to one
	if independenceMoment == 1
		@. θ_upper[end-IndMomentOrder+1:end] = 1
		@. θ_lower[end-IndMomentOrder+1:end] = 0
	end


	Aod_offset = 0
	if counterType != 1
		Aod_offset = 2 * (D - 1) # D-1 wagesPrime and D-1 gamma_primes 
	end
	if OuterScaling == 1 # Aod model
		Aod_offset += 3 + D
		if independenceMoment == 1
			Aod_offset += 1
			#elseif GravityMomentFirstApproach == 1 && sameMarginalsMoment == 0 && UoModel == 0
			#	Aod_offset += D^2
		end
		for i in 1:D #A[1,d] = 1
			θ_upper[Aod_offset+1+D*(i-1):Aod_offset+1+D*(i-1)] = θ_initial[Aod_offset+1+D*(i-1):Aod_offset+1+D*(i-1)]
			θ_lower[Aod_offset+1+D*(i-1):Aod_offset+1+D*(i-1)] = θ_initial[Aod_offset+1+D*(i-1):Aod_offset+1+D*(i-1)]
		end

		if ForceFrechetMarginal == 1 # make only A[baesIndex,d] variable. This is an extra restriction that I want to test to see if it gives better bounds

			for d in 1:D
				for i in 1:D #A[1,d] = 1
					if i != baseIndex
						θ_upper[Aod_offset+d+D*(i-1):Aod_offset+d+D*(i-1)] = θ_initial[Aod_offset+d+D*(i-1):Aod_offset+d+D*(i-1)]
						θ_lower[Aod_offset+d+D*(i-1):Aod_offset+d+D*(i-1)] = θ_initial[Aod_offset+d+D*(i-1):Aod_offset+d+D*(i-1)]
					end
				end
			end
		end
	end

	Θ_lenght = size(θ_initial, 1)

	for i in 1:Θ_lenght
		if θ_initial[i] < 0
			θ_upper[i] = -10000 * θ_initial[i]
			θ_lower[i] = 10000 * θ_initial[i]
		end
	end

	κ_lower = zeros(length(δ_grid))
	κ_upper = zeros(length(δ_grid))
	Θ_lower = zeros(length(θ_initial), length(δ_grid))
	Θ_upper = zeros(length(θ_initial), length(δ_grid))

	if useParallel == 0 # if parallelisation is off, do deltas one at a time 
		θ_1 = copy(θ_initial)

		# run outer loop for upper bound 
		for (i, δ) in enumerate(δ_grid)
			if counterType == 1 # if GT counterfactual, k does not depend on U directly (so create Implicit object)
				obj = PsiObjectiveBundleImplicit(
					δ = δ,
					find_smallest = false,
					γ = γ,
					(moments!) = EK_moments!,
					moments_jacobian! = use_Jacobian == 1 ? params.EK_moments_Jacobian! : error,
					d = nTotalMoments,
					outer_constr_index = outer_constr_index,
					inequality_index = inequality_index,
					complement_index = complement_index,
					l = size(θ_initial, 1),
					U = U,
					N = Jac_W,
					lower_limit = -5000,
					outer_loop_opt = "csw_outer_loop_settings_cluster.opt",
					inner_loop_opt = "ek_inner_loop_options.opt",
				)
			else
				obj = PsiObjectiveBundleExplicit(
					δ = δ,
					find_smallest = false,
					γ = γ,
					(moments!) = EK_moments!,
					moments_jacobian! = use_Jacobian == 1 ? params.EK_moments_Jacobian! : error,
					d = nTotalMoments,
					outer_constr_index = outer_constr_index,
					inequality_index = inequality_index,
					complement_index = complement_index,
					l = size(θ_initial, 1),
					U = U,
					N = Jac_W,
					lower_limit = -50,
					outer_loop_opt = "ek_outer_loop_options.opt",
					inner_loop_opt = "ek_inner_loop_options.opt",
				)
			end
			κ_upper[i], θ_1 = outer_loop(obj, θ_lower, θ_upper, θ_initial_up)
			Θ_upper[:, i] .= θ_1
			print(κ_upper[i])

			save_object(string("cc_output_upper_", i, "_", file_name, ".jld2"),
				(κ_upper[i], θ_1))

		end

		θ_1 = copy(θ_initial)

		# run outer loop for lower bound 
		for (i, δ) in enumerate(δ_grid)
			if counterType == 1 # if GT counterfactual, k does not depend on U directly (so create Implicit object)
				obj = PsiObjectiveBundleImplicit(
					δ = δ,
					find_smallest = true,
					γ = γ,
					(moments!) = EK_moments!,
					moments_jacobian! = use_Jacobian == 1 ? params.EK_moments_Jacobian! : error,
					d = nTotalMoments,
					outer_constr_index = outer_constr_index,
					inequality_index = inequality_index,
					complement_index = complement_index,
					l = size(θ_initial, 1),
					U = U,
					N = Jac_W,
					lower_limit = -5000,
					outer_loop_opt = "csw_outer_loop_settings_cluster.opt",
					inner_loop_opt = "ek_inner_loop_options.opt",
				)
			else
				obj = PsiObjectiveBundleExplicit(
					δ = δ,
					find_smallest = true,
					γ = γ,
					(moments!) = EK_moments!,
					moments_jacobian! = use_Jacobian == 1 ? params.EK_moments_Jacobian! : error,
					d = nTotalMoments,
					outer_constr_index = outer_constr_index,
					inequality_index = inequality_index,
					complement_index = complement_index,
					l = size(θ_initial, 1),
					U = U,
					N = Jac_W,
					lower_limit = -50,
					outer_loop_opt = "ek_outer_loop_options.opt",
					inner_loop_opt = "ek_inner_loop_options.opt",
				)
			end
			κ_lower[i], θ_1 = outer_loop(obj, θ_lower, θ_upper, θ_initial_low)
			Θ_lower[:, i] .= θ_1
			print(κ_lower[i])
			save_object(string("cc_output_lower_", i, "_", file_name, ".jld2"),
				(κ_lower[i], θ_1))
		end

	elseif useParallel == 1

		θ_1 = copy(θ_initial)

		Threads.@threads for i in 1:δ_grid_size

			if counterType == 1

				obj = PsiObjectiveBundleImplicit(
					δ = δ_grid[i],
					find_smallest = true,
					γ = γ,
					(moments!) = EK_moments!,
					moments_jacobian! = use_Jacobian == 1 ? params.EK_moments_Jacobian! : error,
					d = nTotalMoments,
					outer_constr_index = outer_constr_index,
					inequality_index = inequality_index,
					complement_index = complement_index,
					#l=size(θ_initial, 1),
					l = length(θ_initial),
					U = U,
					N = Jac_W,
					lower_limit = -50,
					outer_loop_opt = "ek_outer_loop_options.opt",
					inner_loop_opt = "ek_inner_loop_options.opt",
				)

			else

				obj = PsiObjectiveBundleExplicit(
					δ = δ_grid[i],
					find_smallest = true,
					γ = γ,
					(moments!) = EK_moments!,
					moments_jacobian! = use_Jacobian == 1 ? params.EK_moments_Jacobian! : error,
					d = nTotalMoments,
					outer_constr_index = outer_constr_index,
					inequality_index = inequality_index,
					complement_index = complement_index,
					#l=size(θ_initial, 1),
					l = length(θ_initial),
					U = U,
					N = Jac_W,
					lower_limit = -50,
					outer_loop_opt = "ek_outer_loop_options.opt",
					inner_loop_opt = "ek_inner_loop_options.opt")

			end

			κ_lower[i], θ_2 = outer_loop(obj, θ_lower, θ_upper, θ_initial)
			Θ_lower[:, i] .= θ_2
			print(κ_lower[i])
			save_object(string("cc_output_lower_", i, "_", file_name, ".jld2"),
				(κ_lower[i], θ_2))
		end

		obj.find_smallest = false
		θ_1 = copy(θ_initial)

		Threads.@threads for i in 1:δ_grid_size

			if counterType == 1

				obj = PsiObjectiveBundleImplicit(
					δ = δ_grid[i],
					find_smallest = false,
					γ = γ,
					(moments!) = EK_moments!,
					moments_jacobian! = use_Jacobian == 1 ? params.EK_moments_Jacobian! : error,
					d = nTotalMoments,
					outer_constr_index = outer_constr_index,
					inequality_index = inequality_index,
					complement_index = complement_index,
					#l=size(θ_initial, 1),
					l = length(θ_initial),
					U = U,
					N = Jac_W,
					lower_limit = -50,
					outer_loop_opt = "ek_outer_loop_options.opt",
					inner_loop_opt = "ek_inner_loop_options.opt")

			else

				obj = PsiObjectiveBundleExplicit(
					δ = δ_grid[i],
					find_smallest = false,
					γ = γ,
					(moments!) = EK_moments!,
					moments_jacobian! = use_Jacobian == 1 ? params.EK_moments_Jacobian! : error,
					d = numMomennTotalMomentsts,
					outer_constr_index = outer_constr_index,
					inequality_index = inequality_index,
					complement_index = complement_index,
					#l=size(θ_initial, 1),
					l = length(θ_initial),
					U = U,
					N = Jac_W,
					lower_limit = -50,
					outer_loop_opt = "ek_outer_loop_options.opt",
					inner_loop_opt = "ek_inner_loop_options.opt",
				)

			end

			κ_upper[i], θ_1 = outer_loop(obj, θ_lower, θ_upper, θ_initial)
			Θ_upper[:, i] .= θ_1
			print(κ_upper[i])
			save_object(string("cc_output_upper_", i, "_", file_name, ".jld2"),
				(κ_upper[i], θ_1))
		end

	end

	print(δ_grid)
	print(κ_lower)
	print(κ_upper)

	return (Θ_upper, κ_upper, Θ_lower, κ_lower)
end
