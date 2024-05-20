function LFD(θ, prep_output, params)
	# function that generate the LFD for a particular θ, by running the innerloop
	@unpack numMoments, U, γ, outer_constr_index = prep_output
	D = length(γ.L)
	W = size(U, 1)


	G = zeros(W, numMoments)
	K = zeros(W, 1)

	obj = PsiObjectiveBundleDelta(
		#δ = 1,
		#find_smallest = true,
		γ = γ,
		(moments!) = EK_moments!,
		#moments_jacobian! = rust_moments_jacobian!,
		d = numMoments,
		outer_constr_index = outer_constr_index,
		inequality_index = Int64[],
		l = size(θ, 1),
		U = U,
		N = 100,
		outer_loop_opt = "ek_outer_loop_options.opt",
		inner_loop_opt = "ek_inner_loop_options.opt",
		lower_limit = -50)

	val, x, nStatus = inner_loop(obj, θ) # Represents the lagrangian of the inner optimizer.

	moments!(K, G, θ, U, obj)

	arg0 = zeros(W, 1)
	LFD = zeros(W, 1)

	## Works only when κ does not depend on U [e.g. Grains from Trade]
    # Equation (25) from CC, page 279 
    # TO DO: make it work for all counterfactuals
	for ω ∈ 1:W
		arg0[ω] = -x[1] - dot(G[ω, 1:outer_constr_index-1], x[2:length(x)])
	end

	dPsi!(LFD, arg0)

	return LFD
end
