abstract type KLObjectiveBundle <: ObjectiveBundle end

# Objective bundle for the explicit-dependence case
@with_kw mutable struct KLObjectiveBundleExplicit{T} <: KLObjectiveBundle
    δ                   ::Float64
    find_smallest       ::Bool                                                      # find the smallest counterfactual?

    # model-specific options
    γ                   ::T                                                         # model-specific variables, e.g. conditional choice probabilities, or cache variables that save memory allocations
    moments!            ::Function                                                  # modifies vector K and matrix G in-place by evaluating k and g at (θ, U)
    moments_jacobian!   ::Function         = error                                  # calculates Jacobian of K and G in-place at (θ, U)
    d                   ::Int64                                                     # number of moments in g
    l                   ::Int64                                                     # number of elements in θ
    inequality_index    ::Array{Int64,1}                                            # indices of moments in g that are inequalities, not equalities. May be empty: `Int64[]`
    U                   ::Array{Float64,2}                                          # random draws of U, each draw in one row
    M                   ::Int64            = size(U)[1]                             # number of random draws of U

    # solution-specific options
    outer_constr_index  ::Int64            = d + 1                                  # starting at which index of g should the constraints be solved in the outer loop
    inner_loop_opt      ::String                                                    # path to the KNITRO options file
    outer_loop_opt      ::String                                                    # path to the KNITRO options file
    lower_limit         ::Float64          = -KNITRO.KN_INFINITY                    # lower limit for objective function, where applicable
    use_cached_x        ::Bool             = true                                   # use cached (η, λ) values as starting value for the inner loop
    η_min               ::Float64          = 1e-6                                   # truncate η away from zero to avoid numerical instabilities

    # gradient subsampling options
    N                   ::Int64            = M

    # cache variables used to reduce memory allocations
    H                   ::Array{Float64,2} = zeros(M, d + 1)                        # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = zeros(M, d + 1)
    H_subsam            ::Array{Float64,2} = zeros(N, d + 1)
    H_mean              ::Array{Float64,1} = zeros(d + 1)
    H_temp              ::Array{Float64,1} = zeros(d + 1)
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    jac_h               ::Array{Float64,3} = zeros(N, d + 1, l)
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index)        # cache variable that stores the last successful (η, λ)
    ∂x_∂θ               ::Array{Float64,2} = zeros(length(x), l)
    ∂c_∂θ               ::Array{Float64,2} = zeros(d - outer_constr_index + 1, l)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(length(x), length(x))
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(length(x), l)
end

# Inner-loop objective function, gradient, and Jacobian of constraints for K program, explicit-dependence case
function (Q::KLObjectiveBundleExplicit)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))

	@unpack δ, H, arg0, arg1, M, d, find_smallest, outer_constr_index, lower_limit = Q
	η = x[1]
	λ = @view x[2:end]

	# assemble the expression in the expecation operator
	BLAS.gemv!('N', 1.0, @view(H[:, 1:outer_constr_index]), vcat((-1.0)^find_smallest, -λ)./η, 0.0, arg0)

	# objective function value using numerically sound log-sum-exp
	offset = maximum(arg0)
	arg0 .-= offset     # arg0 = a - offset
	arg1 .= exp.(arg0)  # arg1 = exp(a) * exp(-offset)
	lse = log(sum(arg1))
	f = η * (lse + offset - log(M) + δ)

	# outer loop constraint values, if there are any
	if length(constr) > 0
		@views BLAS.gemv!('T', exp(-lse), H[:, 1+outer_constr_index:1+d], arg1, 0.0, constr)
	end

	# gradient w.r.t. (η, λ)
	if length(g) > 0 && length(θ) == 0

		g[1] = -dot(arg0, arg1) * exp(-lse) - offset + f / η # partial w.r.t. η
		@views BLAS.gemv!('T', -1.0 * exp(-lse), H[:, 2:outer_constr_index], arg1, 0.0, g[2:end]) # partials w.r.t. λ

	# gradient w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		@unpack jac_h, H_copy, H_mean, H_temp, N, l, ∂x_∂θ, ∂c_∂θ = Q

		# calculate gradient of objective via envelope theorem
		# update the first N rows of jac_h
		calculate_jac_θ!(Q, θ)
		# store (1/η) * ((-1)^find_smallest * (∂k_∂θ') - λ'(∂g_∂θ')) in jac_h[:, 1, :]
		@views jac_h[:, 1, :] .*= (-1.0)^find_smallest / η
		for i in 1:l
			@views BLAS.gemv!('N', 1.0, jac_h[:, 2:outer_constr_index, i], -λ / η, 1.0, jac_h[:, 1, i])
		end
		@views BLAS.gemv!('T', η / sum(arg1[1:N]), jac_h[:, 1, :], arg1[1:N], 0.0, g)

		# provide Jacobian for outer-loop constraints, if required
		if outer_constr_index <= d

			# implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean for the calculations below
			ift!(η, λ, offset, Q)

			for i in 1:l
				@views BLAS.gemv!('T', 1.0/sum(arg1[1:N]), jac_h[:, 1+outer_constr_index:1+d, i], arg1[1:N], 0.0, ∂c_∂θ[:, i])
				@views BLAS.gemv!('T', 1.0, H[1:N, 1+outer_constr_index:1+d], jac_h[:, 1, i], 1.0, ∂c_∂θ[:, i])
				@views H_temp[1+outer_constr_index:1+d] .= H_mean[1+outer_constr_index:1+d]
				@views H_temp[1+outer_constr_index:1+d] .*= sum(jac_h[:, 1, i])
				@views ∂c_∂θ[:, i] .-= H_temp[1+outer_constr_index:1+d]
			end
			@views ∂c_∂θ .-= 1/η * (BLAS.gemm('T', 'N', 1.0/sum(arg1), H_copy[:, 1+outer_constr_index:1+d], H_copy[:, 1:outer_constr_index]) .- H_mean[1+outer_constr_index:1+d] .* H_mean[1:outer_constr_index]') * ∂x_∂θ

			jac .= (∂c_∂θ')[:]

		end
	end

	# Hessian w.r.t. (η, λ)
	length(h) > 0 ? hessian!(h, η, offset, Q) : nothing

	if f <= lower_limit
		return -KNITRO.KN_INFINITY
	else
		return f
	end

end

# Objective bundle for the implicit-dependence case
@with_kw mutable struct KLObjectiveBundleImplicit{T} <: KLObjectiveBundle
    δ                   ::Float64
    find_smallest       ::Bool                                                      # find the smallest counterfactual?

    # model-specific options
    γ                   ::T                                                         # model-specific variables, e.g. conditional choice probabilities, or cache variables that save memory allocations
    moments!            ::Function                                                  # modifies vector K and matrix G in-place by evaluating k and g at (θ, U)
    moments_jacobian!   ::Function         = error                                  # calculates Jacobian of K and G in-place at (θ, U)
    d                  	::Int64                                                     # number of moments in g
    l                   ::Int64                                                     # number of elements in θ
    inequality_index    ::Array{Int64,1}                                            # indices of moments in g that are inequalities, not equalities. May be empty: `Int64[]`
    U                   ::Array{Float64,2}                                          # random draws of U, each draw in one row
    M                   ::Int64            = size(U)[1]                             # number of random draws of U

    # solution-specific options
    outer_constr_index  ::Int64            = d + 1                                  # starting at which index of g should the constraints be solved in the outer loop
    inner_loop_opt      ::String                                                    # path to the KNITRO options file
    outer_loop_opt      ::String                                                    # path to the KNITRO options file
    lower_limit	        ::Float64          = -KNITRO.KN_INFINITY                    # lower limit for objective function, where applicable
    use_cached_x        ::Bool             = true                                   # use cached λ values as starting value for the inner loop

    # gradient subsampling options
    N                   ::Int64            = M

    # cache variables used to reduce memory allocations
    H                   ::Array{Float64,2} = zeros(M, d + 1)                        # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = zeros(M, d + 1)
    H_save              ::Float64          = 0.0
    H_mean              ::Array{Float64,1} = zeros(d + 1)
    H_temp              ::Array{Float64,1} = zeros(d + 1)
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    jac_h               ::Array{Float64,3} = zeros(N, d + 1, l)
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index - 1)    # cache variable that stores the last successful λ
    ∂x_∂θ               ::Array{Float64,2} = zeros(length(x), l)
    ∂c_∂θ               ::Array{Float64,2} = zeros(d - outer_constr_index + 2, l)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(length(x), length(x))
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(length(x), l)
end

# Inner-loop objective function, gradient, and Jacobian of constraints for K program, implicit-dependence case
function (Q::KLObjectiveBundleImplicit)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))

	@unpack H, arg0, arg1, M, d, outer_constr_index, lower_limit = Q

	# assemble the expression in the expecation operator
	BLAS.gemv!('N', 1.0, @view(H[:, 2:outer_constr_index]), -x, 0.0, arg0)

	# objective function value using numerically sound log-sum-exp
	offset = maximum(arg0)
	arg0 .-= offset     # arg0 = a - offset
	arg1 .= exp.(arg0)  # arg1 = exp(a) * exp(-offset)
	lse = log(sum(arg1))
	f = lse + offset - log(M)

	if length(constr) > 0
		constr[1] = -f * 1e10
		if outer_constr_index <= d
			@views BLAS.gemv!('T', exp(-lse), H[:, 1+outer_constr_index:1+d], arg1, 0.0, constr[2:d - outer_constr_index + 2])
		end
	end

	# gradient w.r.t. λ
	if length(g) > 0 && length(θ) == 0

		@views BLAS.gemv!('T', -1.0 * exp(-lse), H[:, 2:outer_constr_index], arg1, 0.0, g)

	# gradient (and, if necessary, Jacobian of constraints) w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		@unpack find_smallest, jac_h, H_copy, H_mean, H_temp, N, l, ∂x_∂θ, ∂c_∂θ = Q

		# gradient of objective is simply derivative of K wrt θ
		calculate_grad_k!(g, Q, θ)
		g .*= (-1.0)^find_smallest

		# Jacobian for distance constraint
		# update the first N rows of jac_h
		calculate_jac_θ!(Q, θ)
		# store (∂g_∂θ)'λ in jac_h[:, 1, :]
		@views jac_h[:, 1, :] .= 0.0
		for i in 1:l
			@views BLAS.gemv!('N', 1.0, jac_h[:, 2:outer_constr_index, i], x, 1.0, jac_h[:, 1, i])
		end
		# update Jacobian for distance constraint
		@views BLAS.gemv!('T', 1e10/sum(arg1[1:N]), jac_h[:, 1, :], arg1[1:N], 0.0, ∂c_∂θ[1, :])

		# provide Jacobian for outer-loop constraints, if required
		if outer_constr_index <= d

			# implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean for the calculations below
			ift!(x, Q)

			for i in 1:l
				@views BLAS.gemv!('T', 1.0/sum(arg1[1:N]), jac_h[:, 1+outer_constr_index:1+d, i], arg1[1:N], 0.0, ∂c_∂θ[2:end, i])
				@views BLAS.gemv!('T', -1.0, H[1:N, 1+outer_constr_index:1+d], jac_h[:, 1, i], 1.0, ∂c_∂θ[2:end, i])
				@views H_temp[1+outer_constr_index:1+d] .= H_mean[1+outer_constr_index:1+d]
				@views H_temp[1+outer_constr_index:1+d] .*= sum(jac_h[:, 1, i])
				@views ∂c_∂θ[2:end, i] .+= H_temp[1+outer_constr_index:1+d]
			end
			@views ∂c_∂θ[2:end, :] .-= (BLAS.gemm('T', 'N', 1.0/sum(arg1), H_copy[:, 1+outer_constr_index:1+d], H_copy[:, 2:outer_constr_index]) .- H_mean[1+outer_constr_index:1+d] .* H_mean[2:outer_constr_index]') * ∂x_∂θ

		end

		jac .= (∂c_∂θ')[:]

	end

	# Hessian w.r.t. λ
	length(h) > 0 ? hessian!(h, Q) : nothing

	if f <= lower_limit
		return -KNITRO.KN_INFINITY
	else
		return f
	end

end

# Objective bundle for the minimum-divergence problem
@with_kw mutable struct KLObjectiveBundleDelta{T} <: KLObjectiveBundle
    find_smallest       ::Bool             = true

    # model-specific options
    γ                   ::T                                                         # model-specific variables, e.g. conditional choice probabilities, or cache variables that save memory allocations
    moments!            ::Function                                                  # modifies vector K and matrix G in-place by evaluating k and g at (θ, U)
    moments_jacobian!   ::Function         = error                                  # calculates Jacobian of K and G in-place at (θ, U)
    d                  	::Int64                                                     # number of moments in g
    l                   ::Int64                                                     # number of elements in θ
    inequality_index   	::Array{Int64,1}                                            # indices of moments in g that are inequalities, not equalities. May be empty: `Int64[]`
    U                   ::Array{Float64,2}                                          # random draws of U, each draw in one row
    M                   ::Int64            = size(U)[1]                             # number of random draws of U

    # solution-specific options
    outer_constr_index  ::Int64            = d + 1                                  # starting at which index of g should the constraints be solved in the outer loop
    inner_loop_opt      ::String                                                    # path to the KNITRO options file
    outer_loop_opt      ::String                                                    # path to the KNITRO options file
    lower_limit	        ::Float64          = -KNITRO.KN_INFINITY                    # lower limit for objective function, where applicable
    use_cached_x        ::Bool             = true                                   # use cached λ values as starting value for the inner loop

    # gradient subsampling options
    N                   ::Int64            = M

    # cache variables used to reduce memory allocations
    H                   ::Array{Float64,2} = zeros(M, d + 1)                        # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = zeros(M, d + 1)
    H_mean              ::Array{Float64,1} = zeros(d + 1)
    H_temp              ::Array{Float64,1} = zeros(d + 1)
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    jac_h               ::Array{Float64,3} = zeros(N, d + 1, l)
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index - 1)    # cache variable that stores the last successful λ
    ∂x_∂θ               ::Array{Float64,2} = zeros(length(x), l)
    ∂c_∂θ               ::Array{Float64,2} = zeros(d - outer_constr_index + 1, l)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(length(x), length(x))
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(length(x), l)
end

# Inner-loop objective function, gradient, and Jacobian of constraints for Δ^* program
function (Q::KLObjectiveBundleDelta)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))

	@unpack H, arg0, arg1, M, d, outer_constr_index, lower_limit = Q

	# assemble the expression in the expecation operator
	BLAS.gemv!('N', 1.0, @view(H[:, 2:outer_constr_index]), -x, 0.0, arg0)

	# objective function value using numerically sound log-sum-exp
	offset = maximum(arg0)
	arg0 .-= offset     # arg0 = a - offset
	arg1 .= exp.(arg0)  # arg1 = exp(a) * exp(-offset)
	lse = log(sum(arg1))
	f = lse + offset - log(M)

	if length(constr) > 0
		@views BLAS.gemv!('T', exp(-lse), H[:, 1+outer_constr_index:1+d], arg1, 0.0, constr)
	end

	# gradient w.r.t. λ
	if length(g) > 0 && length(θ) == 0

		@views BLAS.gemv!('T', -1.0 * exp(-lse), H[:, 2:outer_constr_index], arg1, 0.0, g)

	# gradient (and, if necessary, Jacobian of constraints) w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		@unpack jac_h, H_copy, H_mean, H_temp, N, l, ∂x_∂θ, ∂c_∂θ = Q

		# update the first N rows of jac_h
		calculate_jac_θ!(Q, θ)
		# store (∂g_∂θ)'λ in jac_h[:, 1, :]
		@views jac_h[:, 1, :] .= 0.0
		for i in 1:l
			@views BLAS.gemv!('N', 1.0, jac_h[:, 2:outer_constr_index, i], x, 1.0, jac_h[:, 1, i])
		end
		# calculate gradient of objective via envelope theorem
		@views BLAS.gemv!('T', -1.0/sum(arg1[1:N]), jac_h[:, 1, :], arg1[1:N], 0.0, g)

		# provide Jacobian for outer-loop constraints, if required
		if outer_constr_index <= d

			# implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean for the calculations below
			ift!(x, Q)

			for i in 1:l
				@views BLAS.gemv!('T', 1.0/sum(arg1[1:N]), jac_h[:, 1+outer_constr_index:1+d, i], arg1[1:N], 0.0, ∂c_∂θ[:, i])
				@views BLAS.gemv!('T', -1.0, H[1:N, 1+outer_constr_index:1+d], jac_h[:, 1, i], 1.0, ∂c_∂θ[:, i])
				@views H_temp[1+outer_constr_index:1+d] .= H_mean[1+outer_constr_index:1+d]
				@views H_temp[1+outer_constr_index:1+d] .*= sum(jac_h[:, 1, i])
				@views ∂c_∂θ[:, i] .+= H_temp[1+outer_constr_index:1+d]
			end
			@views ∂c_∂θ .-= (BLAS.gemm('T', 'N', 1.0/sum(arg1), H_copy[:, 1+outer_constr_index:1+d], H_copy[:, 2:outer_constr_index]) .- H_mean[1+outer_constr_index:1+d] .* H_mean[2:outer_constr_index]') * ∂x_∂θ

			jac .= (∂c_∂θ')[:]

		end
	end

	# Hessian w.r.t. λ
	length(h) > 0 ? hessian!(h, Q) : nothing

	if f <= lower_limit
		return -KNITRO.KN_INFINITY
	else
		return f
	end

end

# Implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean
function ift!(η, λ, offset, obj::KLObjectiveBundleExplicit)

    @unpack jac_h, arg0, arg1, H, H_copy, H_mean, H_subsam, H_temp, l, M, N, outer_constr_index, ∂x_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, inequality_index, η_min = obj

    H_copy .= H
    # store ((-1)^find_smallest * k - λ'g) / η in first column of H
    @views H_copy[:, 1] .= arg0 .+ offset
    H_subsam .= @view(H_copy[1:N, :])
    @views BLAS.gemv!('T', 1.0/sum(arg1), H_copy, arg1, 0.0, H_mean)
    H_copy .*= .√arg1

    # only the normalized product is used below
    @views jac_h[:, 1, :] .*= arg1[1:N]
    @views jac_h[:, 1, :] ./= sum(arg1[1:N])

    # implicit function theorem
    @views BLAS.gemm!('T', 'N', 1.0/sum(arg1), H_copy[:, 1:outer_constr_index], H_copy[:, 1:outer_constr_index], 0.0, ∂∂f_∂∂x)
    @views BLAS.ger!(-1.0, H_mean[1:outer_constr_index], H_mean[1:outer_constr_index], ∂∂f_∂∂x)
    ∂∂f_∂∂x ./= η
    for i in 1:l
        @views BLAS.gemv!('T', -1.0, H_subsam[:, 1:outer_constr_index], jac_h[:, 1, i], 0.0, ∂∂f_∂x∂θ[:, i])
        @views BLAS.gemv!('T', -1.0/sum(arg1[1:N]), jac_h[:, 2:outer_constr_index, i], arg1[1:N], 1.0, ∂∂f_∂x∂θ[2:outer_constr_index, i])
        @views H_temp[1:outer_constr_index] .= H_mean[1:outer_constr_index]
        @views H_temp[1:outer_constr_index] .*= sum(jac_h[:, 1, i])
        @views ∂∂f_∂x∂θ[:, i] .+= H_temp[1:outer_constr_index]
    end
    ∂x_∂θ .= 0.0
    active_set = findall(vcat((η > η_min), [i ∉ inequality_index || (i ∈ inequality_index && λ[i] !== 0.0) for i in 1:outer_constr_index - 1]))
    try
        @views ∂x_∂θ[active_set, :] .= -∂∂f_∂∂x[active_set, active_set] \ ∂∂f_∂x∂θ[active_set, :]
    catch
        @view(∂x_∂θ[active_set, :]) .= -pinv(∂∂f_∂∂x[active_set, active_set]) * @view(∂∂f_∂x∂θ[active_set, :])
    end

end

# Implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean
function ift!(x, obj::Union{KLObjectiveBundleImplicit, KLObjectiveBundleDelta})

    @unpack jac_h, arg1, H, H_copy, H_mean, H_temp, l, M, N, outer_constr_index, ∂x_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, inequality_index = obj

    H_copy .= H
    @views BLAS.gemv!('T', 1.0/sum(arg1), H, arg1, 0.0, H_mean)
    H_copy .*= .√arg1

    # only the normalized product is used below
    @views jac_h[:, 1, :] .*= arg1[1:N]
    @views jac_h[:, 1, :] ./= sum(arg1[1:N])

    # implicit function theorem
    @views BLAS.gemm!('T', 'N', 1.0/sum(arg1), H_copy[:, 2:outer_constr_index], H_copy[:, 2:outer_constr_index], 0.0, ∂∂f_∂∂x)
    @views BLAS.ger!(-1.0, H_mean[2:outer_constr_index], H_mean[2:outer_constr_index], ∂∂f_∂∂x)
    for i in 1:l
        @views BLAS.gemv!('T', 1.0, H[1:N, 2:outer_constr_index], jac_h[:, 1, i], 0.0, ∂∂f_∂x∂θ[:, i])
        @views BLAS.gemv!('T', -1.0/sum(arg1[1:N]), jac_h[:, 2:outer_constr_index, i], arg1[1:N], 1.0, ∂∂f_∂x∂θ[:, i])
        @views H_temp[2:outer_constr_index] .= H_mean[2:outer_constr_index]
        @views H_temp[2:outer_constr_index] .*= sum(jac_h[:, 1, i])
        @views ∂∂f_∂x∂θ[:, i] .-= H_temp[2:outer_constr_index]
    end
    ∂x_∂θ .= 0.0
    active_set = findall([i ∉ inequality_index || (i ∈ inequality_index && x[i] !== 0.0) for i in 1:outer_constr_index - 1])
    try
        @views ∂x_∂θ[active_set, :] .= -∂∂f_∂∂x[active_set, active_set] \ ∂∂f_∂x∂θ[active_set, :]
    catch
        @view(∂x_∂θ[active_set, :]) .= -pinv(∂∂f_∂∂x[active_set, active_set]) * @view(∂∂f_∂x∂θ[active_set, :])
    end

end

# Hessian w.r.t. (η, λ)
function hessian!(h, η, offset, obj::KLObjectiveBundleExplicit)

    @unpack H, H_copy, H_mean, arg0, arg1, outer_constr_index, ∂∂f_∂∂x = obj

    @views H_copy[:, 2:outer_constr_index] .= H[:, 2:outer_constr_index]
    @views H_copy[:, 1] .= arg0 .+ offset
    @views BLAS.gemv!('T', 1.0 / sum(arg1), H_copy[:, 1:outer_constr_index], arg1, 0.0, H_mean[1:outer_constr_index])
    @views H_copy[:, 1:outer_constr_index] .*= .√arg1
    @views BLAS.gemm!('T', 'N', 1.0/sum(arg1), H_copy[:, 1:outer_constr_index], H_copy[:, 1:outer_constr_index], 0.0, ∂∂f_∂∂x)
    @views BLAS.ger!(-1.0, H_mean[1:outer_constr_index], H_mean[1:outer_constr_index], ∂∂f_∂∂x)
    ∂∂f_∂∂x ./= η

    k = 1
    for i in 1:size(∂∂f_∂∂x)[2]
        for j in i:size(∂∂f_∂∂x)[2]
            h[k] = ∂∂f_∂∂x[i, j]
            k += 1
        end
    end

end

# Hessian w.r.t. λ
function hessian!(h, obj::Union{KLObjectiveBundleImplicit, KLObjectiveBundleDelta})

    @unpack H, H_copy, H_mean, arg1, outer_constr_index, ∂∂f_∂∂x = obj

    @views H_copy[:, 2:outer_constr_index] .= H[:, 2:outer_constr_index]
    @views H_copy[:, 2:outer_constr_index] .*= .√arg1
    @views BLAS.gemv!('T', 1.0/sum(arg1), H[:, 2:outer_constr_index], arg1, 0.0, H_mean[2:outer_constr_index])

    @views BLAS.gemm!('T', 'N', 1.0/sum(arg1), H_copy[:, 2:outer_constr_index], H_copy[:, 2:outer_constr_index], 0.0, ∂∂f_∂∂x)
    @views BLAS.ger!(-1.0, H_mean[2:outer_constr_index], H_mean[2:outer_constr_index], ∂∂f_∂∂x)

    k = 1
    for i in 1:size(∂∂f_∂∂x)[2]
        for j in i:size(∂∂f_∂∂x)[2]
            h[k] = ∂∂f_∂∂x[i, j]
            k += 1
        end
    end

end

# Wrapper to evaluate moments! to H
select_G_from_H(obj::KLObjectiveBundle, H) = @view(H[:, 2:end])

# Wrapper to evaluate moments_jacobian! to jac_h
select_jac_g_from_jac_h(obj::KLObjectiveBundle, jac_h) = @view(jac_h[:, 2:end, :])