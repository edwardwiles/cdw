abstract type PsiObjectiveBundle <: ObjectiveBundle end

# AUD-02 runtime guard (independent-audit remediation): each Psi*Bundle instance owns mutable
# shared scratch (H, H_copy, arg0/1/2, ...) that every objective/gradient/Hessian callback
# mutates in place. KNITRO's own par_concurrent_evals is now set to "no" in every active
# production option file (the primary fix), but this guard is defense-in-depth: it detects --
# rather than silently corrupts state under -- any genuine CROSS-THREAD overlapping entry into
# the SAME bundle instance's callbacks, whether from a misconfigured option file, two solves
# sharing one context, or a future caller that parallelizes across outer points.
#
# Thread-AWARE, not a plain non-reentrant flag: a first cut at this guard (plain
# compare-and-swap Bool) false-positived on a real, legitimate call pattern discovered live
# during validation -- prepare_cc/PMM.jl's delta-star-initial computation calls a bundle's own
# functor from WITHIN a sequential outer solve that already holds it (same thread, nested, no
# race). That pattern is safe: nothing runs concurrently, the inner call completes and its
# buffer writes are fully consumed before the outer call resumes. Tracking the OWNING THREAD ID
# (not just "is someone in here") lets the guard tell the two apart: same-thread nesting
# increments a depth counter and is allowed; a DIFFERENT thread trying to enter while the first
# thread's chain is still active is the actual AUD-02 violation (genuine concurrent mutation of
# shared H/H_copy/arg buffers) and errors immediately.
#
# Not a lock: it does not serialize legitimate concurrent use of DIFFERENT bundle instances
# (those have independent guard-state entries), only flags cross-thread overlapping use of ONE
# instance. Keyed by objectid() via a global IdDict rather than a struct field so it applies
# uniformly to all three bundle types without changing their (widely, positionally-free but
# kwarg-constructed) definitions.
mutable struct _PsiCallbackGuardState
    owner::Int   # Threads.threadid() of the thread currently holding this instance; 0 = free
    depth::Int   # reentrancy depth for `owner` (same-thread nested calls)
end
_PsiCallbackGuardState() = _PsiCallbackGuardState(0, 0)

const _PSI_CALLBACK_ACTIVE = IdDict{Any,_PsiCallbackGuardState}()
const _PSI_CALLBACK_ACTIVE_LOCK = ReentrantLock()
const PSI_CALLBACK_GUARD_ENABLED = Ref(true)
const PSI_CALLBACK_GUARD_VIOLATIONS = Ref(0)

"Call as the first statement of every PsiObjectiveBundle{Explicit,Implicit,Delta} functor and
every hessian! method, before any shared buffer (H, H_copy, arg0/1/2, jac_h, ...) is touched.
Errors immediately on detecting a DIFFERENT thread already active on this SAME bundle instance
(AUD-02: 'concurrent KNITRO evaluations... mutate shared arrays... callback time... races').
Same-thread nested entry (sequential, no race) is allowed and just bumps a depth counter."
function _enter_callback!(obj)
    PSI_CALLBACK_GUARD_ENABLED[] || return nothing
    tid = Threads.threadid()
    lock(_PSI_CALLBACK_ACTIVE_LOCK) do
        st = get!(() -> _PsiCallbackGuardState(), _PSI_CALLBACK_ACTIVE, obj)
        if st.owner == 0
            st.owner = tid
            st.depth = 1
        elseif st.owner == tid
            st.depth += 1
        else
            PSI_CALLBACK_GUARD_VIOLATIONS[] += 1
            error("PsiObjectiveBundle AUD-02 guard: thread $(tid) entered a callback " *
                  "(objective/gradient/Hessian) on a bundle instance while thread $(st.owner) " *
                  "was still active on the SAME instance. This means par_concurrent_evals was " *
                  "effectively enabled for this solve, or two solves/threads share one inner " *
                  "context -- both are unsafe (shared mutable H/H_copy/arg buffers, cache/bank " *
                  "slots). See docs/fullA_independent_audit_remediation.md AUD-02.")
        end
    end
    return nothing
end

"Call in a `finally` clause paired with every `_enter_callback!` call, so the depth counter
clears even if the callback body throws."
function _exit_callback!(obj)
    PSI_CALLBACK_GUARD_ENABLED[] || return nothing
    lock(_PSI_CALLBACK_ACTIVE_LOCK) do
        st = get(_PSI_CALLBACK_ACTIVE, obj, nothing)
        if st !== nothing && st.owner == Threads.threadid()
            st.depth -= 1
            if st.depth <= 0
                st.owner = 0
                st.depth = 0
            end
        end
    end
    return nothing
end

"Reset guard state between independent test runs (mirrors parallelism_guards.jl's guard_reset!)."
function psi_callback_guard_reset!()
    lock(_PSI_CALLBACK_ACTIVE_LOCK) do
        empty!(_PSI_CALLBACK_ACTIVE)
    end
    PSI_CALLBACK_GUARD_VIOLATIONS[] = 0
    return nothing
end

# Objective bundle for the explicit-dependence case
@with_kw mutable struct PsiObjectiveBundleExplicit{T} <: PsiObjectiveBundle
    δ                   ::Float64
    find_smallest       ::Bool                                                      # find the smallest counterfactual?

    # model-specific options
    γ                   ::T                                                         # model-specific variables, e.g. conditional choice probabilities, or cache variables that save memory allocations
    moments!            ::Function                                                  # modifies vector K and matrix G in-place by evaluating k and g at (θ, U)
    moments_jacobian!   ::Function         = error                                  # calculates Jacobian of K and G in-place at (θ, U)
    d                   ::Int64                                                     # number of moments in g
    l                   ::Int64                                                     # number of elements in θ
    inequality_index    ::Array{Int64,1}                                            # indices of moments in g that are inequalities, not equalities. May be empty: `Int64[]`
    complement_index    ::Array{Int64,2}   = [0 0]                                  # for projection inference, inidces of moments in g formed using [lower bounds, upper bounds] for CS for P_2
    U                   ::Array{Float64,2}                                          # random draws of U, each draw in one row
    M                   ::Int64            = size(U)[1]                             # number of random draws of U

    # solution-specific options
    outer_constr_index  ::Int64            = d + 1                                  # starting at which index of g should the constraints be solved in the outer loop
    inner_loop_opt      ::String                                                    # path to the KNITRO options file
    outer_loop_opt      ::String                                                    # path to the KNITRO options file
    lower_limit         ::Float64          = -KNITRO.KN_INFINITY                    # lower limit for objective function, where applicable
    use_cached_x        ::Bool             = false                                  # use cached (η, ζ, λ) values as starting value for the inner loop
    η_min               ::Float64          = 1e-120                                 # truncate η away from zero to avoid numerical instabilities

    # gradient subsampling options
    N                   ::Int64            = M

    # divergence
    Psi!                ::Function         = Psi!
    dPsi!               ::Function         = dPsi!
    ddPsi!              ::Function         = ddPsi!

    # cache variables used to reduce memory allocations
    H                   ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))   # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))   # K and G evaluated at θ, U, γ
    H_subsam            ::Array{Float64,2} = hcat(zeros(N), ones(N), zeros(N, d))   # K and G evaluated at θ, U, γ
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    arg2                ::Array{Float64,1} = zeros(M)
    jac_h               ::Array{Float64,3} = _instrumented_jac_h_default(N, d, l)
    x                   ::Array{Float64,1} = NaN .* ones(1 + outer_constr_index)    # cache variable that stores the last successful (η, ζ, λ)
    ∂x_∂θ               ::Array{Float64,2} = zeros(length(x), l)
    ∂c_∂θ               ::Array{Float64,2} = zeros(d - outer_constr_index + 1, l)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(length(x), length(x))
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(length(x), l)
end

# Inner-loop objective function, gradient, and Jacobian of constraints for K program, explicit-dependence case
function (Q::PsiObjectiveBundleExplicit)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))
	_enter_callback!(Q)
	try

	@unpack δ, H, arg0, arg1, M, d, find_smallest, outer_constr_index, lower_limit, Psi!, dPsi!, ddPsi! = Q
	η = x[1]
	ζ = x[2]
	λ = @view x[3:end]

	# assemble the expression in the expecation operator
	BLAS.gemv!('N', 1.0, @view(H[:, 1:1+outer_constr_index]), vcat((-1.0)^find_smallest, -ζ, -λ)./η, 0.0, arg0)
	Psi!(arg1, arg0)

	# objective function value
	f = η * (sum(arg1) / M + δ) + ζ

	# update the RN derivative if necessary
	if length(g) > 0 || length(constr) > 0
		dPsi!(arg1, arg0)
	end

	# outer loop constraint values, if there are any
	if length(constr) > 0
		@views BLAS.gemv!('T', 1/M, H[:, 2+outer_constr_index:2+d], arg1, 0.0, constr)
	end

	# gradient w.r.t. (η, ζ, λ)
	if length(g) > 0 && length(θ) == 0

		g[1] = -dot(arg0, arg1) / M + (f - ζ) / η # partial w.r.t. η
		g[2] = 1.0 - sum(arg1) / M # partial w.r.t. ζ
		@views BLAS.gemv!('T', -1/M, H[:, 3:1+outer_constr_index], arg1, 0.0, g[3:end]) # partials w.r.t. λ

	# gradient (and, if necessary, Jacobian of constraints) w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		JAC_H_THETA_BRANCH_COUNT[] += 1   # ADDITIVE (jac_h audit): counts entry into this legacy outer-gradient branch

		@unpack jac_h, H_copy, N, l, ∂x_∂θ, ∂c_∂θ = Q

		# update the first N rows of jac_h
		calculate_jac_θ!(Q, θ)
		# store (1/η) * ((-1)^find_smallest * (∂k_∂θ) - (∂g_∂θ)'λ) in jac_h[:, 1, :]
		@views jac_h[:, 1, :] .*= (-1.0)^find_smallest / η
		for i in 1:l
			@views BLAS.gemv!('N', 1.0, jac_h[:, 3:1+outer_constr_index, i], -λ / η, 1.0, jac_h[:, 1, i])
		end
		# calculate gradient of objective via envelope theorem
		@views BLAS.gemv!('T', η/N, jac_h[:, 1, :], arg1[1:N], 0.0, g)

		# provide Jacobian for outer-loop constraints, if required
		if outer_constr_index <= d

			# implicit function theorem to calculate ∂x_∂θ and update jac_h and H_copy for the calculations below
			ift!(η, λ, Q)

			for i in 1:l
				@views BLAS.gemv!('T', 1/N, jac_h[:, 2+outer_constr_index:2+d, i], arg1[1:N], 0.0, ∂c_∂θ[:, i])
				@views BLAS.gemv!('T', 1/N, H[1:N, 2+outer_constr_index:2+d], jac_h[:, 1, i], 1.0, ∂c_∂θ[:, i])
			end
			∂c_∂θ .+= -1/(M * η) * BLAS.gemm('T', 'N', @view(H_copy[:, 2+outer_constr_index:2+d]), @view(H_copy[:, 1:1+outer_constr_index])) * ∂x_∂θ

			jac .= (∂c_∂θ')[:]

		end
	end

	# Hessian w.r.t. (η, ζ, λ)
	length(h) > 0 ? hessian!(h, η, Q) : nothing

	if f <= lower_limit
		return -KNITRO.KN_INFINITY
	else
		return f
	end

	finally
		_exit_callback!(Q)
	end
end

# Objective bundle for the implicit-dependence case
@with_kw mutable struct PsiObjectiveBundleImplicit{T} <: PsiObjectiveBundle
    δ                   ::Float64
    find_smallest       ::Bool                                                      # find the smallest counterfactual?

    # model-specific options
    γ                   ::T                                                         # model-specific variables, e.g. conditional choice probabilities, or cache variables that save memory allocations
    moments!            ::Function                                                  # modifies vector K and matrix G in-place by evaluating k and g at (θ, U)
    moments_jacobian!   ::Function         = error                                  # calculates Jacobian of K and G in-place at (θ, U)
    d                  	::Int64                                                     # number of moments in g
    l                   ::Int64                                                     # number of elements in θ
    inequality_index   	::Array{Int64,1}                                            # indices of moments in g that are inequalities, not equalities. May be empty: `Int64[]`
    complement_index    ::Array{Int64,2}   = [0 0]                                  # for projection inference, inidces of moments in g formed using [lower bounds, upper bounds] for CS for P_2
    U                   ::Array{Float64,2}                                          # random draws of U, each draw in one row
    M                   ::Int64            = size(U)[1]                             # number of random draws of U

    # solution-specific options
    outer_constr_index  ::Int64            = d + 1                                  # starting at which index of g should the constraints be solved in the outer loop
    inner_loop_opt      ::String                                                    # path to the KNITRO options file
    outer_loop_opt      ::String                                                    # path to the KNITRO options file
    lower_limit         ::Float64          = -KNITRO.KN_INFINITY                    # lower limit for objective function, where applicable
    use_cached_x        ::Bool             = false                                  # use cached (ζ, λ) values as starting value for the inner loop

    # divergence
    Psi!                ::Function         = Psi!
    dPsi!               ::Function         = dPsi!
    ddPsi!              ::Function         = ddPsi!

    # gradient subsampling options
    N                   ::Int64            = M

    # cache variables used to reduce memory allocations
    H                   ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))   # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))   # K and G evaluated at θ, U, γ
    H_save              ::Float64          = 0.0
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    arg2                ::Array{Float64,1} = zeros(M)
    # ADDITIVE (jac_h audit, diag/fullA-d4-exact-jach-audit): when false, the dense N x (d+2) x l
    # jac_h tensor below is NOT allocated (a 0x0x0 array is stored instead) and any legacy code path
    # that would populate/read/contract it (calculate_jac_θ!, ift!) errors with a clear diagnostic
    # instead of silently operating on an empty array or an out-of-bounds index. Default TRUE
    # preserves exactly the prior unconditional-allocation behavior for every existing caller that
    # does not pass this kwarg -- see docs/fullA_jach_audit.md for the runtime audit motivating this.
    needs_outer_moment_jacobian ::Bool      = true
    jac_h               ::Array{Float64,3} = needs_outer_moment_jacobian ? _instrumented_jac_h_default(N, d, l) : _skipped_jac_h_default()
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index)        # cache variable that stores the last successful (ζ, λ)
    ∂x_∂θ               ::Array{Float64,2} = zeros(length(x), l)
    ∂c_∂θ               ::Array{Float64,2} = zeros(d - outer_constr_index + 2, l)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(length(x), length(x))
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(length(x), l)
    # ADDITIVE (Melitz screening session, docs/melitz_optimization_report_2026-07-23_screening_session.md
    # Section 4 / this session's Phase I.1): the `lower_limit` early-stop branch immediately below
    # (`if f <= lower_limit; return -KNITRO.KN_INFINITY`) already gives KNITRO a mathematically valid
    # early "unbounded" signal, but previously discarded the exact information that PRODUCED that
    # signal -- the crossing dual point `x`, its own valid Delta lower bound `-f`, and when it
    # happened -- leaving a caller with only KNITRO's post-hoc `nStatus` to go on (which cannot be
    # told apart from a genuine unresolved numerical failure). These four Refs record that
    # information UNCONDITIONALLY whenever the branch fires (a few scalar/vector writes, negligible
    # cost), for ANY caller of this shared bundle type, not just Melitz -- purely additive
    # instrumentation, default state (`false`/`NaN`/empty/`0`) reproduces the exact pre-change
    # observable behavior for every existing caller that does not read these fields (Ricardian's
    # `ccOuter.jl`/`ccInner.jl` `lower_limit=-50` usage included). A caller wanting today's threshold
    # crossing must reset `threshold_crossed[]=false` itself before each inner solve it wants to
    # classify (this struct persists across an entire outer trajectory, so a stale `true` from an
    # earlier solve would otherwise leak into a later, unrelated one).
    threshold_crossed          ::Base.RefValue{Bool}         = Ref(false)           # did `f <= lower_limit` fire on the just-completed inner solve?
    threshold_crossing_bound   ::Base.RefValue{Float64}      = Ref(NaN)             # `-f` at the crossing iterate -- a valid Delta lower bound (weak duality, unconditional)
    threshold_crossing_x       ::Base.RefValue{Vector{Float64}} = Ref(Float64[])    # the (ζ,λ) dual iterate at the crossing
    threshold_crossing_time_ns ::Base.RefValue{UInt64}       = Ref(UInt64(0))       # time_ns() at the crossing, for a caller to compute elapsed-to-crossing against its own solve-start timestamp
end

# Inner-loop objective function, gradient, and Jacobian of constraints for K program, implicit-dependence case
function (Q::PsiObjectiveBundleImplicit)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))
	_enter_callback!(Q)
	try

	@unpack H, arg0, arg1, M, d, outer_constr_index, lower_limit, Psi!, dPsi!, ddPsi! = Q
	ζ = x[1]

	# assemble the expression in the expecation operator
	BLAS.gemv!('N', 1.0, @view(H[:, 2:1+outer_constr_index]), -x, 0.0, arg0)
	Psi!(arg1, arg0)

	# objective function value
	f = sum(arg1) / M + ζ

	# update the RN derivative if necessary
	if length(g) > 0 || length(constr) > 0
		dPsi!(arg1, arg0)
	end

	# outer loop constraint values, if there are any
	if length(constr) > 0
		constr[1] = -f * 1e10
		if outer_constr_index <= d
			@views BLAS.gemv!('T', 1/M, H[:, 2+outer_constr_index:2+d], arg1, 0.0, constr[2:d - outer_constr_index + 2])
		end
	end

	# gradient w.r.t. (ζ, λ)
	if length(g) > 0 && length(θ) == 0

		g[1] = 1.0 - sum(arg1) / M # partial w.r.t. ζ
		@views BLAS.gemv!('T', -1/M, H[:, 3:1+outer_constr_index], arg1, 0.0, g[2:end]) # partials w.r.t. λ

	# gradient (and, if necessary, Jacobian of constraints) w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		JAC_H_THETA_BRANCH_COUNT[] += 1   # ADDITIVE (jac_h audit): counts entry into this legacy outer-gradient branch

		@unpack find_smallest, jac_h, H_copy, N, l, ∂x_∂θ, ∂c_∂θ = Q

		# gradient of objective is simply derivative of K wrt θ
		calculate_grad_k!(g, Q, θ)
		g .*= (-1.0)^find_smallest

		# Jacobian for distance constraint
		# update the first N rows of jac_h
		calculate_jac_θ!(Q, θ)
		# store (∂g_∂θ)'λ in jac_h[:, 1, :]
		λ = @view x[2:end]
		@views jac_h[:, 1, :] .= 0.0
		for i in 1:l
			@views BLAS.gemv!('N', 1.0, jac_h[:, 3:1+outer_constr_index, i], λ, 1.0, jac_h[:, 1, i])
		end
		# update Jacobian for distance constraint
		@views BLAS.gemv!('T', 1e10/N, jac_h[:, 1, :], arg1[1:N], 0.0, ∂c_∂θ[1, :])

		# provide Jacobian for outer-loop constraints, if required
		if outer_constr_index <= d

			# implicit function theorem to calculate ∂x_∂θ and update jac_h and H_copy for the calculations below
			ift!(λ, Q)

			for i in 1:l
				@views BLAS.gemv!('T', 1/N, jac_h[:, 2+outer_constr_index:2+d, i], arg1[1:N], 0.0, ∂c_∂θ[2:d - outer_constr_index + 2, i])
				@views BLAS.gemv!('T',-1/N, H[1:N, 2+outer_constr_index:2+d], jac_h[:, 1, i], 1.0, ∂c_∂θ[2:d - outer_constr_index + 2, i])
			end
			∂c_∂θ[2:d - outer_constr_index + 2, :] .+= -1/M * BLAS.gemm('T', 'N', @view(H_copy[:, 2+outer_constr_index:2+d]), @view(H_copy[:, 2:1+outer_constr_index])) * ∂x_∂θ

		end

		jac .= (∂c_∂θ')[:]

	end

	# Hessian w.r.t. (ζ, λ)
	length(h) > 0 ? hessian!(h, Q) : nothing

	if f <= lower_limit
		# ADDITIVE (Melitz screening session Phase I.1): record the crossing before returning
		# -- see the struct field comments above for why this is unconditional and safe for
		# every existing caller.
		Q.threshold_crossed[] = true
		Q.threshold_crossing_bound[] = -f
		Q.threshold_crossing_x[] = copy(x)
		Q.threshold_crossing_time_ns[] = time_ns()
		return -KNITRO.KN_INFINITY
	else
		return f
	end

	finally
		_exit_callback!(Q)
	end
end

# Objective bundle for the minimum-divergence problem
@with_kw mutable struct PsiObjectiveBundleDelta{T} <: PsiObjectiveBundle
    find_smallest       ::Bool             = true

    # model-specific options
    γ                   ::T                                                         # model-specific variables, e.g. conditional choice probabilities, or cache variables that save memory allocations
    moments!            ::Function                                                  # modifies vector K and matrix G in-place by evaluating k and g at (θ, U)
    moments_jacobian!   ::Function         = error                                  # calculates Jacobian of K and G in-place at (θ, U)
    d                  	::Int64                                                     # number of moments in g
    l                   ::Int64                                                     # number of elements in θ
    inequality_index   	::Array{Int64,1}                                            # indices of moments in g that are inequalities, not equalities. May be empty: `Int64[]`
    complement_index    ::Array{Int64,2}   = [0 0]                                  # for projection inference, inidces of moments in g formed using [lower bounds, upper bounds] for CS for P_2
    U                   ::Array{Float64,2}                                          # random draws of U, each draw in one row
    M                   ::Int64            = size(U)[1]                             # number of random draws of U

    # solution-specific options
    outer_constr_index  ::Int64            = d + 1                                  # starting at which index of g should the constraints be solved in the outer loop
    inner_loop_opt      ::String                                                    # path to the KNITRO options file
    outer_loop_opt      ::String                                                    # path to the KNITRO options file
    lower_limit         ::Float64          = -KNITRO.KN_INFINITY                    # lower limit for objective function, where applicable
    use_cached_x        ::Bool             = false                                  # use cached (ζ, λ) values as starting value for the inner loop

    # divergence
    Psi!                ::Function         = Psi!
    dPsi!               ::Function         = dPsi!
    ddPsi!              ::Function         = ddPsi!

    # gradient subsampling options
    N                   ::Int64            = M

    # cache variables used to reduce memory allocations
    H                   ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))   # K and G evaluated at θ, U, γ
    H_copy              ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))   # K and G evaluated at θ, U, γ
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    arg2                ::Array{Float64,1} = zeros(M)
    # ADDITIVE (memory-scalability continuation session, docs/melitz_optimization_report_2026-07-23_continuation3.md
    # Section 9): mirrors PsiObjectiveBundleImplicit's own needs_outer_moment_jacobian escape hatch
    # (introduced earlier for exactly this reason, see docs/fullA_jach_audit.md) -- an apparent
    # oversight had left PsiObjectiveBundleDelta with an UNCONDITIONAL dense jac_h allocation
    # (~206GB at D=20/W=80,000, quartic in D: N*(d+2)*l with d~D^2, l~2D^2). Call-graph audit
    # (this session) confirms jac_h is NEVER read for this bundle type by any current caller in this
    # repo: `build_melitz_psi_bundle` (the only construction site, src/melitz/delta_star.jl) is used
    # exclusively for FIXED-theta inner CC dual solves (`inner_loop`/`melitz_recover_lfd`/
    # `run_melitz_inner_delta`), and `inner_loop_KNITRO`'s callback (`callbackEvalFG_inner!`) always
    # calls `obj(x, g)` with an EMPTY θ -- the theta-gradient branch of this functor (below, the only
    # branch that touches jac_h) is dead code for every existing caller. The real Melitz outer
    # theta-gradient search uses a SEPARATE `PsiObjectiveBundleImplicit` (`build_melitz_implicit_bundle`,
    # `finite_delta_outer.jl`), which already has its own independent needs_outer_moment_jacobian flag.
    # Default TRUE preserves the exact prior unconditional-allocation behavior for any caller that does
    # not pass this kwarg.
    needs_outer_moment_jacobian ::Bool      = true
    jac_h               ::Array{Float64,3} = needs_outer_moment_jacobian ? _instrumented_jac_h_default(N, d, l) : _skipped_jac_h_default()
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index)        # cache variable that stores the last successful (ζ, λ)
    ∂x_∂θ               ::Array{Float64,2} = zeros(length(x), l)
    ∂c_∂θ               ::Array{Float64,2} = zeros(d - outer_constr_index + 1, l)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(length(x), length(x))
    ∂∂f_∂x∂θ            ::Array{Float64,2} = zeros(length(x), l)
end

# Inner-loop objective function, gradient, and Jacobian of constraints for Δ^* program
function (Q::PsiObjectiveBundleDelta)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))
	_enter_callback!(Q)
	try

	@unpack H, arg0, arg1, M, d, outer_constr_index, lower_limit, Psi!, dPsi!, ddPsi! = Q
	ζ = x[1]

	# assemble the expression in the expecation operator
	BLAS.gemv!('N', 1.0, @view(H[:, 2:1+outer_constr_index]), -x, 0.0, arg0)
	Psi!(arg1, arg0)

	# objective function value
	f = sum(arg1) / M + ζ

	# update the RN derivative, if necessary
	if length(g) > 0 || length(constr) > 0
		dPsi!(arg1, arg0)
	end

	# outer loop constraint values, if there are any
	if length(constr) > 0
		@views BLAS.gemv!('T', 1/M, H[:, 2+outer_constr_index:2+d], arg1, 0.0, constr)
	end

	# gradient w.r.t. (ζ, λ)
	if length(g) > 0 && length(θ) == 0

		g[1] = 1.0 - sum(arg1) / M # partial w.r.t. ζ
		@views BLAS.gemv!('T', -1/M, H[:, 3:1+outer_constr_index], arg1, 0.0, g[2:end]) # partials w.r.t. λ

	# gradient (and, if necessary, Jacobian of constraints) w.r.t. θ
	elseif length(g) > 0 && length(θ) > 0

		JAC_H_THETA_BRANCH_COUNT[] += 1   # ADDITIVE (jac_h audit): counts entry into this legacy outer-gradient branch

		@unpack jac_h, H_copy, N, l, ∂x_∂θ, ∂c_∂θ = Q

		# update the first N rows of jac_h
		calculate_jac_θ!(Q, θ)
		# store (∂g_∂θ)'λ in jac_h[:, 1, :]
		λ = @view x[2:end]
		@views jac_h[:, 1, :] .= 0.0
		for i in 1:l
			@views BLAS.gemv!('N', 1.0, jac_h[:, 3:1+outer_constr_index, i], λ, 1.0, jac_h[:, 1, i])
		end
		# calculate gradient of objective via envelope theorem
		@views BLAS.gemv!('T', -1/N, jac_h[:, 1, :], arg1[1:N], 0.0, g)

		# provide Jacobian for outer-loop constraints, if required
		if outer_constr_index <= d

			# implicit function theorem to calculate ∂x_∂θ and update jac_h and H_copy for the calculations below
			ift!(λ, Q)

			for i in 1:l
				@views BLAS.gemv!('T', 1/N, jac_h[:, 2+outer_constr_index:2+d, i], arg1[1:N], 0.0, ∂c_∂θ[:, i])
				@views BLAS.gemv!('T',-1/N, H[1:N, 2+outer_constr_index:2+d], jac_h[:, 1, i], 1.0, ∂c_∂θ[:, i])
			end
			∂c_∂θ .+= -1/M * BLAS.gemm('T', 'N', @view(H_copy[:, 2+outer_constr_index:2+d]), @view(H_copy[:, 2:1+outer_constr_index])) * ∂x_∂θ

			jac .= (∂c_∂θ')[:]

		end
	end

	# Hessian w.r.t. (ζ, λ)
	length(h) > 0 ? hessian!(h, Q) : nothing

	if f <= lower_limit
		return -KNITRO.KN_INFINITY
	else
		return f
	end

	finally
		_exit_callback!(Q)
	end
end

# Implicit function theorem to calculate ∂x_∂θ and update jac_h and H_copy
function ift!(η, λ, obj::PsiObjectiveBundleExplicit)

    JAC_H_IFT_COUNT[] += 1   # ADDITIVE (jac_h audit): counts calls to the jac_h-reading/contracting ift! path

    @unpack jac_h, arg0, arg1, arg2, H, H_copy, H_subsam, l, M, N, outer_constr_index, ∂x_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, inequality_index, η_min = obj

    ddPsi!(arg2, arg0)

    H_copy .= H
    # store ((-1)^find_smallest * k - ζ - λ'g) / η in first column of H_copy
    @views H_copy[:, 1] .= arg0
    H_subsam .= @view(H_copy[1:N, :])
    H_copy .*= .√arg2

    # store ddΨ() * ((-1)^find_smallest * (∂k_∂θ) - (∂g_∂θ)'λ) / η in jac_h[:, 1, :] as only the product is used below
    @views jac_h[:, 1, :] .*= arg2[1:N]

    # implicit function theorem
    @views BLAS.gemm!('T', 'N', 1/(η * M), H_copy[:, 1:1+outer_constr_index], H_copy[:, 1:1+outer_constr_index], 0.0, ∂∂f_∂∂x)
    @views BLAS.gemm!('T', 'N', -1/N, H_subsam[:, 1:1+outer_constr_index], jac_h[:, 1, :], 0.0, ∂∂f_∂x∂θ)
    for i in 1:l
        @views BLAS.gemv!('T', -1/N, jac_h[:, 3:1+outer_constr_index, i], arg1[1:N], 1.0, ∂∂f_∂x∂θ[3:end, i])
    end
    ∂x_∂θ .= 0.0
    active_set = findall(vcat((η > η_min), true, [i ∉ inequality_index || (i ∈ inequality_index && λ[i] !== 0.0) for i in 1:outer_constr_index - 1]))
    try
        @views ∂x_∂θ[active_set, :] .= -∂∂f_∂∂x[active_set, active_set] \ ∂∂f_∂x∂θ[active_set, :]
    catch
        @view(∂x_∂θ[active_set, :]) .= -pinv(∂∂f_∂∂x[active_set, active_set]) * @view(∂∂f_∂x∂θ[active_set, :])
    end

end

# Implicit function theorem to calculate ∂x_∂θ and update jac_h and H_copy
function ift!(λ, obj::Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta})

    # ADDITIVE (jac_h audit): defensive guard -- if this object was constructed with
    # needs_outer_moment_jacobian=false, jac_h is a 0x0x0 placeholder and MUST NOT be indexed into
    # below (that would either error opaquely on a bounds check or, worse, silently no-op on an
    # empty array). Throw a clear diagnostic instead. In the normal callable flow this is
    # unreachable (calculate_jac_θ! already errors first, see cc_algo/outer_loop_functions.jl), but
    # this guard covers any direct/future call to ift! that bypasses that ordering.
    if hasproperty(obj, :needs_outer_moment_jacobian) && !obj.needs_outer_moment_jacobian
        error("ift!: this PsiObjectiveBundleImplicit was constructed with needs_outer_moment_jacobian=false " *
              "-- jac_h was never allocated, so the legacy implicit-function-theorem outer-Jacobian " *
              "contraction cannot run. See docs/fullA_jach_audit.md.")
    end
    JAC_H_IFT_COUNT[] += 1   # ADDITIVE (jac_h audit): counts calls to the jac_h-reading/contracting ift! path

    @unpack jac_h, arg0, arg1, arg2, H, H_copy, l, M, N, outer_constr_index, ∂x_∂θ, ∂∂f_∂∂x, ∂∂f_∂x∂θ, inequality_index = obj

    ddPsi!(arg2, arg0)

    H_copy .= H
    H_copy .*= .√arg2

    # store ddΨ() * (∂g_∂θ)'λ in jac_h[:, 1, :] as only the product is used below
    @views jac_h[:, 1, :] .*= arg2[1:N]

    # implicit function theorem
    @views BLAS.gemm!('T', 'N', 1/M, H_copy[:, 2:1+outer_constr_index], H_copy[:, 2:1+outer_constr_index], 0.0, ∂∂f_∂∂x)
    @views BLAS.gemm!('T', 'N', 1/N, H[1:N, 2:1+outer_constr_index], jac_h[:, 1, :], 0.0, ∂∂f_∂x∂θ)
    for i in 1:l
        @views BLAS.gemv!('T', -1/N, jac_h[:, 3:1+outer_constr_index, i], arg1[1:N], 1.0, ∂∂f_∂x∂θ[2:end, i])
    end
    ∂x_∂θ .= 0.0
    active_set = findall(vcat(true, [i ∉ inequality_index || (i ∈ inequality_index && λ[i] !== 0.0) for i in 1:outer_constr_index - 1]))
    try
        @views ∂x_∂θ[active_set, :] .= -∂∂f_∂∂x[active_set, active_set] \ ∂∂f_∂x∂θ[active_set, :]
    catch
        @view(∂x_∂θ[active_set, :]) .= -pinv(∂∂f_∂∂x[active_set, active_set]) * @view(∂∂f_∂x∂θ[active_set, :])
    end

end

# Hessian w.r.t. (η, ζ, λ)
function hessian!(h, η, obj::PsiObjectiveBundleExplicit)
    _enter_callback!(obj)
    try

    @unpack H, H_copy, M, arg0, arg2, ddPsi!, outer_constr_index, ∂∂f_∂∂x = obj

    ddPsi!(arg2, arg0)

    @views H_copy[:, 2:1+outer_constr_index] .= H[:, 2:1+outer_constr_index]
    @views H_copy[:, 1] .= arg0
    @views H_copy[:, 1:1+outer_constr_index] .*= .√arg2
    @views BLAS.gemm!('T', 'N', 1/(η * M), H_copy[:, 1:1+outer_constr_index], H_copy[:, 1:1+outer_constr_index], 0.0, ∂∂f_∂∂x)

    k = 1
    for i in 1:size(∂∂f_∂∂x)[2]
        for j in i:size(∂∂f_∂∂x)[2]
            h[k] = ∂∂f_∂∂x[i, j]
            k += 1
        end
    end

    finally
    	_exit_callback!(obj)
    end
end

# Hessian w.r.t. (ζ, λ)
function hessian!(h, obj::Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta})
    _enter_callback!(obj)
    try

    @unpack H, H_copy, M, arg0, arg2, ddPsi!, outer_constr_index, ∂∂f_∂∂x = obj

    ddPsi!(arg2, arg0)

    @views H_copy[:, 2:1+outer_constr_index] .= H[:, 2:1+outer_constr_index]
    @views H_copy[:, 2:1+outer_constr_index] .*= .√arg2
    @views BLAS.gemm!('T', 'N', 1/M, H_copy[:, 2:1+outer_constr_index], H_copy[:, 2:1+outer_constr_index], 0.0, ∂∂f_∂∂x)

    k = 1
    for i in 1:size(∂∂f_∂∂x)[2]
        for j in i:size(∂∂f_∂∂x)[2]
            h[k] = ∂∂f_∂∂x[i, j]
            k += 1
        end
    end

    finally
    	_exit_callback!(obj)
    end
end

# Wrapper to evaluate moments! to H
select_G_from_H(obj::PsiObjectiveBundle, H) = @view(H[:, 3:end])

# Wrapper to evaluate moments_jacobian! to jac_h
select_jac_g_from_jac_h(obj::PsiObjectiveBundle, jac_h) = @view(jac_h[:, 3:end, :])
