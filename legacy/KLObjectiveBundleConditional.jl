abstract type KLObjectiveBundleConditional <: ObjectiveBundleConditional end

# Objective bundle for the explicit-dependence case
@with_kw mutable struct KLObjectiveBundleConditionalExplicit{T} <: KLObjectiveBundleConditional
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
    ncov                ::Int64                                                     # number of conditioning values
    var_index           ::Array{Int64,1}                                            # indexes which covariate is associated with which element in θ (0 for all moments)

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
    dd                  ::Int64            = d ÷ ncov
    dm                  ::Int64            = (outer_constr_index - 1) ÷ ncov
    dc                  ::Int64            = (d - outer_constr_index + 1) ÷ ncov
    dϑ                  ::Int64            = sum(var_index .== 0) + sum(var_index .== 1)
    H                   ::Array{Float64,2} = zeros(M, ncov + d)                     # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = zeros(M, 1 + dd)
    H_subsam            ::Array{Float64,2} = zeros(N, 1 + dd)
    H_mean              ::Array{Float64,1} = zeros(1 + dd)
    H_temp              ::Array{Float64,1} = zeros(1 + dd)
    arg0                ::Array{Float64,2} = zeros(M, ncov)
    arg1                ::Array{Float64,2} = zeros(M, ncov)
    lse                 ::Array{Float64,1} = zeros(ncov)
    offset              ::Array{Float64,1} = zeros(ncov)
    jac_h               ::Array{Float64,3} = zeros(N, 1 + dd, dϑ)
    HH                  ::Array{Real,2}    = zeros(Real, N, size(H)[2])
    gg                  ::Array{Float64,1} = zeros(dϑ)
    x                   ::Array{Float64,1} = NaN .* ones(ncov + outer_constr_index - 1)  # cache variable that stores the last successful (η, λ)
    ∂x_∂θ               ::Array{Float64,2} = zeros(1 + dm, dϑ)
    ∂c_∂θ               ::Array{Float64,2} = zeros(dc, dϑ)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(1 + dm, 1 + dm)
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(1 + dm, dϑ)
    nnzH                ::Int32            = zero(Int32)
    hessIndexVars1      ::Array{Int32,1}   = zeros(Int32, 0)
    hessIndexVars2      ::Array{Int32,1}   = zeros(Int32, 0)
end

# Inner-loop objective function, gradient, and Jacobian of constraints for K program, explicit-dependence case
function (Q::KLObjectiveBundleConditionalExplicit)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))

	@unpack δ, ncov, H, arg0, arg1, lse, offset, M, d, dc, dd, dm, find_smallest, outer_constr_index, lower_limit = Q

	ff = zeros(eltype(θ), ncov)
	for i in 1:ncov
		# assemble the expression in the expectation operator
		η = x[i]
		λ = @view x[ncov+(i-1)*dm+1:ncov+i*dm]
		@views arg0[:, i] .= H[:, i] .* (-1.0)^find_smallest / η
		@views BLAS.gemv!('N', 1.0, H[:, ncov+(i-1)*dm+1:ncov+i*dm], -λ / η, 1.0, arg0[:, i])
		# objective function value using numerically sound log-sum-exp
		@views offset[i] = maximum(arg0[:, i])
		@views arg0[:, i] .-= offset[i]        # arg0 = a - offset
		@views arg1[:, i] .= exp.(arg0[:, i])  # arg1 = exp(a) * exp(-offset)
		@views lse[i] = log(sum(arg1[:, i]))
		ff[i] = η * (lse[i] + offset[i] - log(M) + δ)
	end
	f = sum(ff)

	# outer loop constraint values, if there are any
	if length(constr) > 0
		for i in 1:ncov
			@views BLAS.gemv!('T', exp(-lse[i]), H[:, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc], arg1[:, i], 0.0, constr[(i-1)*dc+1:i*dc])
		end
	end

	# gradient w.r.t. (η, λ)
	if length(g) > 0 && length(θ) == 0

		for i in 1:ncov
			# partial w.r.t η
			η = x[i]
			@views g[i] = -dot(arg0[:, i], arg1[:, i]) * exp(-lse[i]) - offset[i] + ff[i] / η
			# partials w.r.t. λ
			@views BLAS.gemv!('T', -1.0 * exp(-lse[i]), H[:, ncov+(i-1)*dm+1:ncov+i*dm], arg1[:, i], 0.0, g[ncov+(i-1)*dm+1:ncov+i*dm])
		end

	# gradient (and, if necessary, Jacobian of constraints) w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		@unpack gg, jac_h, N, l, H_copy, H_mean, H_temp, ∂x_∂θ, ∂c_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, dϑ, inequality_index, η_min = Q

		g .= 0.0

		for i in 1:ncov

			# update the first N rows of jac_h
			nonzero_moments = findall([j ∉ inequality_index || (j ∈ inequality_index && x[ncov+j] !== 0.0) for j in (i-1)*dm+1:i*dm])
			calculate_jac_θ_subvec!(Q, θ, i, nonzero_moments)

			# store ((-1)^find_smallest * (∂k_∂θ) - (∂g_∂θ)'λ) / η in jac_h[:, 1, :]
			η = x[i]
			λ = @view x[ncov+(i-1)*dm+1:ncov+i*dm]
			@views jac_h[:, 1, :] .*= (-1.0)^find_smallest / η
			for j in 1:dϑ
				@views BLAS.gemv!('N', 1.0, jac_h[:, 2:dm+1, j], -λ / η, 1.0, jac_h[:, 1, j])
			end

			# calculate gradient of objective via envelope theorem
			@views BLAS.gemv!('T', η / sum(arg1[1:N, i]), jac_h[:, 1, :], arg1[1:N, i], 0.0, gg)
			# find indices for active subvector ϑ of θ
			active_indices = make_active_indices(Q, i)
			@views g[active_indices] .+= gg

			# provide Jacobian for outer-loop constraints, if required
			if outer_constr_index <= d

				# Implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean
				ift!(x, Q, i)

				for j in 1:dϑ
					@views BLAS.gemv!('T', 1.0/sum(arg1[1:N, i]), jac_h[:, dm+2:dd+1, j], arg1[1:N, i], 0.0, ∂c_∂θ[:, j])
					@views BLAS.gemv!('T', 1.0, H[1:N, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc], jac_h[:, 1, j], 1.0, ∂c_∂θ[:, j])
					@views H_temp[dm+2:dd+1] .= H_mean[dm+2:dd+1]
					@views H_temp[dm+2:dd+1] .*= sum(jac_h[:, 1, j])
					@views ∂c_∂θ[:, j] .-= H_temp[dm+2:dd+1]
				end
				@views ∂c_∂θ .-= 1/η * (BLAS.gemm('T', 'N', exp(-lse[i]), H_copy[:, dm+2:dd+1], H_copy[:, 1:dm+1]) .- H_mean[dm+2:dd+1] .* H_mean[1:dm+1]') * ∂x_∂θ

				jac[(i-1)*dc*dϑ+1:i*dc*dϑ] .= (∂c_∂θ')[:]

			end
		end
	end

	# Hessian w.r.t. λ
	length(h) > 0 ? hessian!(h, x, Q) : nothing

	if f <= lower_limit
		return -KNITRO.KN_INFINITY
	else
		return f
	end

end

# Objective bundle for the implicit-dependence case
@with_kw mutable struct KLObjectiveBundleConditionalImplicit{T} <: KLObjectiveBundleConditional
    δ                   ::Float64
    find_smallest       ::Bool                                                      # find the smallest counterfactual?

    # model-specific options
    γ                   ::T                                                         # model-specific variables, e.g. conditional choice probabilities, or cache variables that save memory allocations
    moments!            ::Function                                                  # modifies vector K and matrix G in-place by evaluating k and g at (θ, U)
    moments_jacobian!   ::Function         = error                                  # calculates Jacobian of K and G in-place at (θ, U)
    d                  	::Int64                                                     # number of moments in g
    l                   ::Int64                                                     # number of elements in θ
    inequality_index   	::Array{Int64,1}                                            # indices of moments in g that are inequalities, not equalities. May be empty: `Int64[]`
    U                   ::Array{Float64,2}                                          # random draws of U, each draw in one row
    M                   ::Int64            = size(U)[1]                             # number of random draws of U
    ncov                ::Int64                                                     # number of conditioning values
    var_index           ::Array{Int64,1}                                            # indexes which covariate is associated with which element in θ (0 for all moments)

    # solution-specific options
    outer_constr_index  ::Int64            = d + 1                                  # starting at which index of g should the constraints be solved in the outer loop
    inner_loop_opt      ::String                                                    # path to the KNITRO options file
    outer_loop_opt      ::String                                                    # path to the KNITRO options file
    lower_limit         ::Float64          = -KNITRO.KN_INFINITY                    # lower limit for objective function, where applicable
    use_cached_x        ::Bool             = true                                   # use cached λ values as starting value for the inner loop

    # gradient subsampling options
    N                   ::Int64            = M

    # cache variables used to reduce memory allocations
    dd                  ::Int64            = d ÷ ncov
    dm                  ::Int64            = (outer_constr_index - 1) ÷ ncov
    dc                  ::Int64            = (d - outer_constr_index + 1) ÷ ncov
    dϑ                  ::Int64            = sum(var_index .== 0) + sum(var_index .== 1)
    H                   ::Array{Float64,2} = zeros(M, ncov + d)                     # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = zeros(M, 1 + dd)
    H_mean              ::Array{Float64,1} = zeros(1 + dd)
    H_temp              ::Array{Float64,1} = zeros(1 + dd)
    H_save              ::Float64          = 0.0
    arg0                ::Array{Float64,2} = zeros(M, ncov)
    arg1                ::Array{Float64,2} = zeros(M, ncov)
    lse                 ::Array{Float64,1} = zeros(ncov)
    offset              ::Array{Float64,1} = zeros(ncov)
    jac_h               ::Array{Float64,3} = zeros(N, 1 + dd, dϑ)
    HH                  ::Array{Real,2}    = zeros(Real, N, size(H)[2])
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index - 1)    # cache variable that stores the last successful λ
    ∂x_∂θ               ::Array{Float64,2} = zeros(dm, dϑ)
    ∂c_∂θ               ::Array{Float64,2} = zeros(dc, dϑ)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(dm, dm)
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(dm, dϑ)
    nnzH                ::Int32            = zero(Int32)
    hessIndexVars1      ::Array{Int32,1}   = zeros(Int32, 0)
    hessIndexVars2      ::Array{Int32,1}   = zeros(Int32, 0)
end

# Inner-loop objective function, gradient, and Jacobian of constraints for K program, implicit-dependence case
function (Q::KLObjectiveBundleConditionalImplicit)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))

	@unpack ncov, H, arg0, arg1, lse, offset, M, d, dc, dd, dm, outer_constr_index, lower_limit = Q

	ff = zeros(eltype(θ), ncov)
	for i in 1:ncov
		# assemble the expression in the expecation operator
		@views BLAS.gemv!('N', 1.0, H[:, ncov+(i-1)*dm+1:ncov+i*dm], -x[(i-1)*dm+1:i*dm], 0.0, arg0[:, i])
		# objective function value using numerically sound log-sum-exp
		@views offset[i] = maximum(arg0[:, i])
		@views arg0[:, i] .-= offset[i]        # arg0 = a - offset
		@views arg1[:, i] .= exp.(arg0[:, i])  # arg1 = exp(a) * exp(-offset)
		@views lse[i] = log(sum(arg1[:, i]))
		ff[i] = lse[i] + offset[i] - log(M)
	end
	f = sum(ff)

	# outer loop constraint values, if there are any
	if length(constr) > 0
		constr[1:ncov] .= -ff .* 1e10
		if outer_constr_index <= d
			for i in 1:ncov
				@views BLAS.gemv!('T', exp(-lse[i]), H[:, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc], arg1[:, i], 0.0, constr[ncov+(i-1)*dc+1:ncov+i*dc])
			end
		end
	end

	# gradient w.r.t. λ
	if length(g) > 0 && length(θ) == 0

		for i in 1:ncov
			@views BLAS.gemv!('T', -exp(-lse[i]), H[:, ncov+(i-1)*dm+1:ncov+i*dm], arg1[:, i], 0.0, g[(i-1)*dm+1:i*dm])
		end

	# gradient (and, if necessary, Jacobian of constraints) w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		@unpack find_smallest, jac_h, N, l, H_copy, H_mean, H_temp, ∂x_∂θ, ∂c_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, dϑ, inequality_index = Q

		# derivative of objective is simply derivative of K wrt θ
		calculate_grad_k!(g, Q, θ)
		g .*= (-1.0)^find_smallest

		for i in 1:ncov

			# update the first N rows of jac_h
			nonzero_moments = findall([j ∉ inequality_index || (j ∈ inequality_index && x[j] !== 0.0) for j in (i-1)*dm+1:i*dm])
			calculate_jac_θ_subvec!(Q, θ, i, nonzero_moments)

			# store (∂g_∂ϑ)'λ in jac_h[:, 1, :]
			λ = @view x[(i-1)*dm+1:i*dm]
			@views jac_h[:, 1, :] .= 0.0
			for j in 1:dϑ
				@views BLAS.gemv!('N', 1.0, jac_h[:, 2:dm+1, j], λ, 1.0, jac_h[:, 1, j])
			end

			# update Jacobian for distance constraint
			@views BLAS.gemv!('T', 1e10/sum(arg1[1:N, i]), jac_h[:, 1, :], arg1[1:N, i], 0.0, jac[(i-1)*dϑ+1:i*dϑ])

			# provide Jacobian for outer-loop constraints, if required
			if outer_constr_index <= d

				# Implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean
				ift!(x, Q, i)

				for j in 1:dϑ
					@views BLAS.gemv!('T', 1.0/sum(arg1[1:N, i]), jac_h[:, dm+2:dd+1, j], arg1[1:N, i], 0.0, ∂c_∂θ[:, j])
					@views BLAS.gemv!('T', -1.0, H[1:N, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc], jac_h[:, 1, j], 1.0, ∂c_∂θ[:, j])
					@views H_temp[dm+2:dd+1] .= H_mean[dm+2:dd+1]
					@views H_temp[dm+2:dd+1] .*= sum(jac_h[:, 1, j])
					@views ∂c_∂θ[:, j] .+= H_temp[dm+2:dd+1]
				end
				@views ∂c_∂θ .-= (BLAS.gemm('T', 'N', exp(-lse[i]), H_copy[:, dm+2:dd+1], H_copy[:, 2:dm+1]) .- H_mean[dm+2:dd+1] .* H_mean[2:dm+1]') * ∂x_∂θ

				jac[ncov*dϑ+(i-1)*dc*dϑ+1:ncov*dϑ+i*dc*dϑ] .= (∂c_∂θ')[:]

			end
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

# Objective bundle for the minimum-divergence problem
@with_kw mutable struct KLObjectiveBundleConditionalDelta{T} <: KLObjectiveBundleConditional
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
    ncov                ::Int64                                                     # number of conditioning values
    var_index           ::Array{Int64,1}                                            # indexes which covariate is associated with which element in θ (0 for all moments)

    # solution-specific options
    outer_constr_index  ::Int64            = d + 1                                  # starting at which index of g should the constraints be solved in the outer loop
    inner_loop_opt      ::String                                                    # path to the KNITRO options file
    outer_loop_opt      ::String                                                    # path to the KNITRO options file
    lower_limit         ::Float64          = -KNITRO.KN_INFINITY                    # lower limit for objective function, where applicable
    use_cached_x        ::Bool             = true                                   # use cached λ values as starting value for the inner loop

    # gradient subsampling options
    N                   ::Int64            = M

    # cache variables used to reduce memory allocations
    dd                  ::Int64            = d ÷ ncov
    dm                  ::Int64            = (outer_constr_index - 1) ÷ ncov
    dc                  ::Int64            = (d - outer_constr_index + 1) ÷ ncov
    dϑ                  ::Int64            = sum(var_index .== 0) + sum(var_index .== 1)
    H                   ::Array{Float64,2} = zeros(M, ncov + d)                     # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = zeros(M, 1 + dd)
    H_mean              ::Array{Float64,1} = zeros(1 + dd)
    H_temp              ::Array{Float64,1} = zeros(1 + dd)
    arg0                ::Array{Float64,2} = zeros(M, ncov)
    arg1                ::Array{Float64,2} = zeros(M, ncov)
    lse                 ::Array{Float64,1} = zeros(ncov)
    offset              ::Array{Float64,1} = zeros(ncov)
    jac_h               ::Array{Float64,3} = zeros(N, 1 + dd, dϑ)
    HH                  ::Array{Real,2}    = zeros(Real, N, size(H)[2])
    gg                  ::Array{Float64,1} = zeros(dϑ)
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index - 1)    # cache variable that stores the last successful λ
    ∂x_∂θ               ::Array{Float64,2} = zeros(dm, dϑ)
    ∂c_∂θ               ::Array{Float64,2} = zeros(dc, dϑ)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(dm, dm)
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(dm, dϑ)
    nnzH                ::Int32            = zero(Int32)
    hessIndexVars1      ::Array{Int32,1}   = zeros(Int32, 0)
    hessIndexVars2      ::Array{Int32,1}   = zeros(Int32, 0)
end

# Inner-loop objective function, gradient, and Jacobian of constraints for Δ^* program
function (Q::KLObjectiveBundleConditionalDelta)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))

	@unpack ncov, H, arg0, arg1, lse, offset, M, d, dc, dd, dm, outer_constr_index, lower_limit = Q

	f = zero(eltype(θ))
	for i in 1:ncov
		# assemble the expression in the expecation operator
		@views BLAS.gemv!('N', 1.0, H[:, ncov+(i-1)*dm+1:ncov+i*dm], -x[(i-1)*dm+1:i*dm], 0.0, arg0[:, i])
		# objective function value using numerically sound log-sum-exp
		@views offset[i] = maximum(arg0[:, i])
		@views arg0[:, i] .-= offset[i]        # arg0 = a - offset
		@views arg1[:, i] .= exp.(arg0[:, i])  # arg1 = exp(a) * exp(-offset)
		@views lse[i] = log(sum(arg1[:, i]))
		f += lse[i] + offset[i] - log(M)
	end

	# outer loop constraint values, if there are any
	if length(constr) > 0
		for i in 1:ncov
			@views BLAS.gemv!('T', exp(-lse[i]), H[:, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc], arg1[:, i], 0.0, constr[(i-1)*dc+1:i*dc])
		end
	end

	# gradient w.r.t. λ
	if length(g) > 0 && length(θ) == 0

		for i in 1:ncov
			@views BLAS.gemv!('T', -exp(-lse[i]), H[:, ncov+(i-1)*dm+1:ncov+i*dm], arg1[:, i], 0.0, g[(i-1)*dm+1:i*dm])
		end

	# gradient (and, if necessary, Jacobian of constraints) w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		@unpack gg, jac_h, N, l, H_copy, H_mean, H_temp, ∂x_∂θ, ∂c_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, dϑ, inequality_index = Q

		g .= 0.0

		for i in 1:ncov

			# update the first N rows of jac_h
			nonzero_moments = findall([j ∉ inequality_index || (j ∈ inequality_index && x[j] !== 0.0) for j in (i-1)*dm+1:i*dm])
			calculate_jac_θ_subvec!(Q, θ, i, nonzero_moments)

			# store (∂g_∂ϑ)'λ in jac_h[:, 1, :]
			λ = @view x[(i-1)*dm+1:i*dm]
			@views jac_h[:, 1, :] .= 0.0
			for j in 1:dϑ
				@views BLAS.gemv!('N', 1.0, jac_h[:, 2:dm+1, j], λ, 1.0, jac_h[:, 1, j])
			end

			# calculate gradient of objective via envelope theorem
			@views BLAS.gemv!('T', -1.0/sum(arg1[1:N, i]), jac_h[:, 1, :], arg1[1:N, i], 0.0, gg)
			# find indices for active subvector ϑ of θ
			active_indices = make_active_indices(Q, i)
			@views g[active_indices] .+= gg

			# provide Jacobian for outer-loop constraints, if required
			if outer_constr_index <= d

				# Implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean
				ift!(x, Q, i)

				for j in 1:dϑ
					@views BLAS.gemv!('T', 1.0/sum(arg1[1:N, i]), jac_h[:, dm+2:dd+1, j], arg1[1:N, i], 0.0, ∂c_∂θ[:, j])
					@views BLAS.gemv!('T', -1.0, H[1:N, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc], jac_h[:, 1, j], 1.0, ∂c_∂θ[:, j])
					@views H_temp[dm+2:dd+1] .= H_mean[dm+2:dd+1]
					@views H_temp[dm+2:dd+1] .*= sum(jac_h[:, 1, j])
					@views ∂c_∂θ[:, j] .+= H_temp[dm+2:dd+1]
				end
				@views ∂c_∂θ .-= (BLAS.gemm('T', 'N', exp(-lse[i]), H_copy[:, dm+2:dd+1], H_copy[:, 2:dm+1]) .- H_mean[dm+2:dd+1] .* H_mean[2:dm+1]') * ∂x_∂θ

				jac[(i-1)*dc*dϑ+1:i*dc*dϑ] .= (∂c_∂θ')[:]

			end
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
function ift!(x, obj::KLObjectiveBundleConditionalExplicit, i)

    @unpack jac_h, arg0, arg1, d, dc, dd, dm, dϑ, H, H_copy, H_mean, H_subsam, H_temp, lse, M, N, ncov, offset, ∂x_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, inequality_index, η_min  = obj

    # store ((-1)^find_smallest * k - λ'g) / η in ith column of H
    @views H_copy[:, 1] .= arg0[:, i] .+ offset[i]
    @views H_copy[:, 2:dm+1] .= H[:, ncov+(i-1)*dm+1:ncov+i*dm]
    @views H_copy[:, dm+2:dd+1] .= H[:, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc]
    H_subsam .= @view(H_copy[1:N, :])
    @views BLAS.gemv!('T', exp(-lse[i]), H_copy, arg1[:, i], 0.0, H_mean)
    @views H_copy .*= .√arg1[:, i]

    # only the normalized product is used below
    @views jac_h[:, 1, :] .*= arg1[1:N, i]
    @views jac_h[:, 1, :] ./= sum(arg1[1:N, i])

    # implicit function theorem
    η = x[i]
    @views BLAS.gemm!('T', 'N', exp(-lse[i]), H_copy[:, 1:dm+1], H_copy[:, 1:dm+1], 0.0, ∂∂f_∂∂x)
    @views BLAS.ger!(-1.0, H_mean[1:dm+1], H_mean[1:dm+1], ∂∂f_∂∂x)
    ∂∂f_∂∂x ./= η

    for j in 1:dϑ
        @views BLAS.gemv!('T', -1.0, H_subsam[:, 1:dm+1], jac_h[:, 1, j], 0.0, ∂∂f_∂x∂θ[:, j])
        @views BLAS.gemv!('T', -1.0/sum(arg1[1:N, i]), jac_h[:, 2:1+dm, j], arg1[1:N, i], 1.0, ∂∂f_∂x∂θ[2:end, j])
        @views H_temp[1:dm+1] .= H_mean[1:dm+1]
        @views H_temp[1:dm+1] .*= sum(jac_h[:, 1, j])
        @views ∂∂f_∂x∂θ[:, j] .+= H_temp[1:dm+1]
    end
    ∂x_∂θ .= 0.0
    active_set = findall(vcat(η > η_min, [j ∉ inequality_index || (j ∈ inequality_index && x[ncov+j] !== 0.0) for j in (i-1)*dm+1:i*dm]))
    try
        @views ∂x_∂θ[active_set, :] .= -∂∂f_∂∂x[active_set, active_set] \ ∂∂f_∂x∂θ[active_set, :]
    catch
        @view(∂x_∂θ[active_set, :]) .= -pinv(∂∂f_∂∂x[active_set, active_set]) * @view(∂∂f_∂x∂θ[active_set, :])
    end

end

# Implicit function theorem to calculate ∂x_∂θ and update jac_h, H_copy, and H_mean
function ift!(x, obj::Union{KLObjectiveBundleConditionalImplicit, KLObjectiveBundleConditionalDelta}, i)

    @unpack jac_h, arg1, d, dc, dd, dm, dϑ, H, H_copy, H_mean, H_temp, lse, M, N, ncov, ∂x_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, inequality_index  = obj

    @views H_copy[:, 2:dm+1] .= H[:, ncov+(i-1)*dm+1:ncov+i*dm]
    @views H_copy[:, dm+2:dd+1] .= H[:, ncov*(1+dm)+(i-1)*dc+1:ncov*(1+dm)+i*dc]
    @views BLAS.gemv!('T', exp(-lse[i]), H_copy[:, 2:dd+1], arg1[:, i], 0.0, H_mean[2:dd+1])
    @views H_copy .*= .√arg1[:, i]

    # only the normalized product is used below
    @views jac_h[:, 1, :] .*= arg1[1:N, i]
    @views jac_h[:, 1, :] ./= sum(arg1[1:N, i])

    # implicit function theorem
    @views BLAS.gemm!('T', 'N', exp(-lse[i]), H_copy[:, 2:dm+1], H_copy[:, 2:dm+1], 0.0, ∂∂f_∂∂x)
    @views BLAS.ger!(-1.0, H_mean[2:dm+1], H_mean[2:dm+1], ∂∂f_∂∂x)

    for j in 1:dϑ
        @views BLAS.gemv!('T', 1.0, H[1:N, ncov+(i-1)*dm+1:ncov+i*dm], jac_h[:, 1, j], 0.0, ∂∂f_∂x∂θ[:, j])
        @views BLAS.gemv!('T', -1.0/sum(arg1[1:N, i]), jac_h[:, 2:dm+1, j], arg1[1:N, i], 1.0, ∂∂f_∂x∂θ[:, j])
        @views H_temp[2:dm+1] .= H_mean[2:dm+1]
        @views H_temp[2:dm+1] .*= sum(jac_h[:, 1, j])
        @views ∂∂f_∂x∂θ[:, j] .-= H_temp[2:dm+1]
    end
    ∂x_∂θ .= 0.0
    active_set = findall([j ∉ inequality_index || (j ∈ inequality_index && x[j] !== 0.0) for j in (i-1)*dm+1:i*dm])
    try
        @views ∂x_∂θ[active_set, :] .= -∂∂f_∂∂x[active_set, active_set] \ ∂∂f_∂x∂θ[active_set, :]
    catch
        @view(∂x_∂θ[active_set, :]) .= -pinv(∂∂f_∂∂x[active_set, active_set]) * @view(∂∂f_∂x∂θ[active_set, :])
    end

end

# Hessian w.r.t. (η, λ)
function hessian!(h, x, obj::KLObjectiveBundleConditionalExplicit)

    @unpack H, H_copy, H_mean, arg0, arg1, dm, ncov, lse, offset, ∂∂f_∂∂x = obj

    k = 1

    for i in 1:ncov

        η = x[i]

        @views H_copy[:, 2:dm+1] .= H[:, ncov+(i-1)*dm+1:ncov+i*dm]
        @views H_copy[:, 1] .= arg0[:, i] .+ offset[i]
        @views BLAS.gemv!('T', exp(-lse[i]), H_copy[:, 1:dm+1], arg1[:, i], 0.0, H_mean[1:dm+1])
        @views H_copy[:, 1:dm+1] .*= .√arg1[:, i]

        # implicit function theorem
        @views BLAS.gemm!('T', 'N', exp(-lse[i]), H_copy[:, 1:dm+1], H_copy[:, 1:dm+1], 0.0, ∂∂f_∂∂x)
        @views BLAS.ger!(-1.0, H_mean[1:dm+1], H_mean[1:dm+1], ∂∂f_∂∂x)
        ∂∂f_∂∂x ./= η

        for i in 1:size(∂∂f_∂∂x)[2]
            for j in i:size(∂∂f_∂∂x)[2]
                h[k] = ∂∂f_∂∂x[i, j]
                k += 1
            end
        end

    end

end


# Hessian w.r.t. λ
function hessian!(h, obj::Union{KLObjectiveBundleConditionalImplicit, KLObjectiveBundleConditionalDelta})

    @unpack H, H_copy, H_mean, arg1, dm, ncov, lse, ∂∂f_∂∂x = obj

    k = 1

    for i in 1:ncov

        @views H_copy[:, 2:dm+1] .= H[:, ncov+(i-1)*dm+1:ncov+i*dm]
        @views BLAS.gemv!('T', exp(-lse[i]), H_copy[:, 2:dm+1], arg1[:, i], 0.0, H_mean[2:dm+1])
        @views H_copy[:, 2:dm+1] .*= .√arg1[:, i]

        @views BLAS.gemm!('T', 'N', exp(-lse[i]), H_copy[:, 2:dm+1], H_copy[:, 2:dm+1], 0.0, ∂∂f_∂∂x)
        @views BLAS.ger!(-1.0, H_mean[2:dm+1], H_mean[2:dm+1], ∂∂f_∂∂x)

        for i in 1:size(∂∂f_∂∂x)[2]
            for j in i:size(∂∂f_∂∂x)[2]
                h[k] = ∂∂f_∂∂x[i, j]
                k += 1
            end
        end

    end

end

# Indices (row-column) for Hessian w.r.t. (η, λ)
function hessian_indices!(obj::KLObjectiveBundleConditionalExplicit)

    @unpack ncov, dm = obj

    obj.nnzH = ncov * (dm + 1) * (dm + 2) ÷ 2
    obj.hessIndexVars1 = zeros(Int32, obj.nnzH)
    obj.hessIndexVars2 = zeros(Int32, obj.nnzH)
    hessIndexVars1 = zeros(Int32, obj.nnzH)
    hessIndexVars2 = zeros(Int32, obj.nnzH)

    k = 1

    for i in 1:ncov

        # (η, η)
        hessIndexVars1[k] = convert(Int32, i)
        hessIndexVars2[k] = convert(Int32, i)
        k += 1

        # (η, λ)
        for j in 1:dm
            hessIndexVars1[k] = convert(Int32, i)
            hessIndexVars2[k] = convert(Int32, ncov + (i-1) * dm + j)
            k += 1
        end

        # (λ, λ)
        for j1 in 1:dm
            for j2 in j1:dm
                hessIndexVars1[k] = convert(Int32, ncov + (i-1) * dm + j1)
                hessIndexVars2[k] = convert(Int32, ncov + (i-1) * dm + j2)
                k += 1
            end
        end

    end

    hessIndexVars1 .-= 1
    hessIndexVars2 .-= 1

    obj.hessIndexVars1 .= hessIndexVars1
    obj.hessIndexVars2 .= hessIndexVars2

end

# Indices (row-column) for Hessian w.r.t. λ
function hessian_indices!(obj::Union{KLObjectiveBundleConditionalImplicit, KLObjectiveBundleConditionalDelta})

    @unpack ncov, dm = obj

    obj.nnzH = ncov * dm * (dm + 1) ÷ 2
    obj.hessIndexVars1 = zeros(Int32, obj.nnzH)
    obj.hessIndexVars2 = zeros(Int32, obj.nnzH)
    hessIndexVars1 = zeros(Int32, obj.nnzH)
    hessIndexVars2 = zeros(Int32, obj.nnzH)

    k = 1

    for i in 1:ncov

        # (λ, λ)
        for j1 in 1:dm
            for j2 in j1:dm
                hessIndexVars1[k] = convert(Int32, (i-1) * dm + j1)
                hessIndexVars2[k] = convert(Int32, (i-1) * dm + j2)
                k += 1
            end
        end

    end

    hessIndexVars1 .-= 1
    hessIndexVars2 .-= 1

    obj.hessIndexVars1 .= hessIndexVars1
    obj.hessIndexVars2 .= hessIndexVars2

end

# Wrapper to evaluate moments! to H
select_G_from_H(obj::KLObjectiveBundleConditional, H) = @view(H[:, obj.ncov+1:end])