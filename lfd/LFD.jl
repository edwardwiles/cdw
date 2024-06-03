function LFD(cc_output, prep_output, params)
	# function that generate the LFD for a particular θ, by running the innerloop

	@unpack δ_grid, file_name, θ_initial, numMoments, U, γ, outer_constr_index = prep_output
	@unpack Θ_upper, Θ_lower = cc_output

	D = length(γ.L)
	W = size(U, 1)

	LFD_upper = zeros(W, length(δ_grid))
	LFD_lower = zeros(W, length(δ_grid))

	obj = PsiObjectiveBundleDelta(
		#δ = 1,
		#find_smallest = true,
		γ = γ,
		(moments!) = EK_moments!,
		#moments_jacobian! = rust_moments_jacobian!,
		d = numMoments,
		outer_constr_index = outer_constr_index,
		inequality_index = Int64[],
		l = size(θ_initial, 1),
		U = U,
		N = 100,
		outer_loop_opt = "ek_outer_loop_options.opt",
		inner_loop_opt = "ek_inner_loop_options.opt",
		lower_limit = -50)

	for i ∈ 1:length(δ_grid)
		# upper LFD
		G = zeros(W, numMoments)
		K = zeros(W, 1)
		arg0 = zeros(W, 1)
		LFD = zeros(W, 1)

		val, x, nStatus = inner_loop(obj, Θ_upper[:, i]) # Represents the lagrangian of the inner optimizer.
		EK_moments!(K, G, Θ_upper[:, i], U, obj)
		## Works only when κ does not depend on U [e.g. Grains from Trade]
		# Equation (25) from CC, page 279 
		# TO DO: make it work for all counterfactuals
		for ω ∈ 1:W
			arg0[ω] = -x[1] - dot(G[ω, 1:outer_constr_index-1], x[2:length(x)])
		end
		dPsi!(LFD, arg0)
		@. LFD_upper[:, i] = LFD[:]

		# Lower LFD
		G = zeros(W, numMoments)
		K = zeros(W, 1)
		arg0 = zeros(W, 1)
		LFD = zeros(W, 1)

		val, x, nStatus = inner_loop(obj, Θ_lower[:, i]) # Represents the lagrangian of the inner optimizer.
		EK_moments!(K, G, Θ_lower[:, i], U, obj)
		## Works only when κ does not depend on U [e.g. Grains from Trade]
		# Equation (25) from CC, page 279 
		# TO DO: make it work for all counterfactuals
		for ω ∈ 1:W
			arg0[ω] = -x[1] - dot(G[ω, 1:outer_constr_index-1], x[2:length(x)])
		end
		dPsi!(LFD, arg0)
		@. LFD_lower[:, i] = LFD[:]
	end

	lfd_output = (LFD_upper = LFD_upper,
		LFD_lower = LFD_lower)

	return lfd_output
end
