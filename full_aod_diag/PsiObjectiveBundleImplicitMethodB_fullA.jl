# ============================================================================
# Diagnostics-only Method-B variant of PsiObjectiveBundleImplicit for the
# FULL-A_od-in-outer-loop framework (2 outer constraints: divergence budget +
# gravity). Unlike sequential_gravity/PsiObjectiveBundleImplicitMethodB.jl
# (which requires outer_constr_index > d, i.e. ZERO extra constraints), this
# handles exactly ONE extra constraint (outer_constr_index == d), replacing
# the dense-Jacobian-then-contract computation for BOTH constraint columns
# with cheap, exact, validated direct gradients:
#   - divergence budget: the same envelope-scalar ForwardDiff.gradient
#     already used/validated in the sequential_gravity variant and in
#     full_aod_diag/ad_benchmark's audit (relerr ~1e-15 vs production).
#   - gravity: a TRIVIAL direct gradient of the bare scalar sumGrav(theta)
#     function (no draws loop, no lambda, no arg1 needed at all) — because
#     for UoModel==1 the gravity moment is IDENTICAL across every draw (see
#     moments/newGravityMoment!.jl's own comment: "F-independent
#     gravity/orthogonality moment"), so the weighted-average constraint
#     value equals that constant for ANY weighting, meaning the ift!
#     correction (how the WEIGHTS change with theta) is mathematically
#     exactly zero, not merely small. Verified: matches the production
#     dense-Jacobian+ift! computation to relerr 2.3e-16, and costs ~0.4ms
#     vs. the ~6.3s dense Jacobian at D=10 (full_aod_diag/ad_benchmark/
#     verify_gravity_trivial_and_time_inner.log).
#
# Net effect: ELIMINATES the dense N×(d+2)×l Jacobian construction from the
# full-A outer gradient/constraint-Jacobian callback entirely — both
# constraint columns are now computed by direct, exact, cheap formulas.
# Measured combined per-evaluation cost at D=10: divergence (~0.057s) +
# gravity (~0.0004s) vs the dense Jacobian's ~6.33s — about 100x cheaper for
# the gradient step alone; combined with a ~0.1s inner KNITRO solve, total
# per-evaluation cost drops from ~6.4s to ~0.16s (~40x).
# ============================================================================

@with_kw mutable struct PsiObjectiveBundleImplicitMethodBFullA{T} <: PsiObjectiveBundle
    δ                   ::Float64
    find_smallest       ::Bool

    γ                   ::T
    moments!            ::Function
    moments_jacobian!   ::Function         = error
    d                   ::Int64
    l                   ::Int64
    inequality_index    ::Array{Int64,1}
    complement_index    ::Array{Int64,2}   = [0 0]
    U                   ::Array{Float64,2}
    M                   ::Int64            = size(U)[1]

    outer_constr_index  ::Int64            = d
    inner_loop_opt      ::String
    outer_loop_opt      ::String
    lower_limit         ::Float64          = -KNITRO.KN_INFINITY
    use_cached_x        ::Bool             = false

    Psi!                ::Function         = Psi!
    dPsi!                ::Function         = dPsi!
    ddPsi!               ::Function         = ddPsi!

    N                   ::Int64            = M

    # the trivial gravity-gradient function, theta -> l-vector (D-general;
    # closed over the fixed data/context at construction time by the caller)
    gravity_grad        ::Function

    H                   ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))
    H_copy              ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))
    H_save              ::Float64          = 0.0
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    arg2                ::Array{Float64,1} = zeros(M)
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(length(x), length(x))
end

"Envelope scalar for the divergence-budget constraint (constr[1]); identical formula to
 sequential_gravity/PsiObjectiveBundleImplicitMethodB.jl's _methodB_envelope_scalar."
function _methodB_fullA_envelope_scalar(θ::AbstractVector, moments!::Function, γobj, Usub::AbstractMatrix,
                                          λ::AbstractVector, arg1::Vector{Float64}, d::Int, oci::Int)
    N = size(Usub, 1); T = eltype(θ)
    K = zeros(T, N); G = zeros(T, N, d)
    moments!(K, G, θ, Usub, (γ = γobj,))
    s = zero(T)
    @inbounds for draw in 1:N
        acc = zero(T)
        for j in 1:oci-1
            acc += λ[j] * G[draw, j]
        end
        s += arg1[draw] * acc
    end
    return (1e10 / N) * s
end

function (Q::PsiObjectiveBundleImplicitMethodBFullA)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))

	@unpack H, arg0, arg1, M, d, outer_constr_index, lower_limit, Psi!, dPsi!, ddPsi! = Q
	ζ = x[1]

	BLAS.gemv!('N', 1.0, @view(H[:, 2:1+outer_constr_index]), -x, 0.0, arg0)
	Psi!(arg1, arg0)

	f = sum(arg1) / M + ζ

	if length(g) > 0 || length(constr) > 0
		dPsi!(arg1, arg0)
	end

	if length(constr) > 0
		constr[1] = -f * 1e10
		if outer_constr_index <= d
			@views BLAS.gemv!('T', 1/M, H[:, 2+outer_constr_index:2+d], arg1, 0.0, constr[2:d-outer_constr_index+2])
		end
	end

	if length(g) > 0 && length(θ) == 0

		g[1] = 1.0 - sum(arg1) / M
		@views BLAS.gemv!('T', -1/M, H[:, 3:1+outer_constr_index], arg1, 0.0, g[2:end])

	elseif length(g) > 0 && length(θ) > 0

		@unpack find_smallest, N, l, moments!, γ = Q

		# ---- objective gradient: unchanged (already a direct scalar autodiff call) ----
		calculate_grad_k!(g, Q, θ)
		g .*= (-1.0)^find_smallest

		# ---- constraint 1 (divergence budget): Method B envelope scalar ----
		λ = collect(@view x[2:end])
		Usub = Q.U[1:N, :]
		gdiv = ForwardDiff.gradient(θθ -> _methodB_fullA_envelope_scalar(θθ, moments!, γ, Usub, λ, Q.arg1, d, outer_constr_index), θ)

		# ---- constraint 2 (gravity): trivial direct gradient, no draws loop ----
		ggrav = Q.gravity_grad(θ)

		jac .= vcat(gdiv, ggrav)   # matches (∂c_∂θ')[:] column-major flattening for 2 constraints

	end

	length(h) > 0 ? hessian!(h, Q) : nothing

	if f <= lower_limit
		return -KNITRO.KN_INFINITY
	else
		return f
	end

end

outer_loop_lambda_length(obj::PsiObjectiveBundleImplicitMethodBFullA) = obj.l + obj.d - obj.outer_constr_index + 2

function outer_loop_constraints!(kc, obj::PsiObjectiveBundleImplicitMethodBFullA)
    cIndices = KNITRO.KN_add_cons(kc, obj.d - obj.outer_constr_index + 2)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1e10 * obj.δ)
    KNITRO.KN_set_con_eqbnds(kc, obj.d - obj.outer_constr_index + 1, cIndices[2:obj.d-obj.outer_constr_index+2], zeros(obj.d - obj.outer_constr_index + 1))
    return cIndices
end

inner_loop_number_variables(obj::PsiObjectiveBundleImplicitMethodBFullA) = obj.outer_constr_index
inner_loop_lower_bounds(obj::PsiObjectiveBundleImplicitMethodBFullA) = vcat(-KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_initial_values(obj::PsiObjectiveBundleImplicitMethodBFullA) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index)

function inner_loop_complementarity_constraints(kc, obj::PsiObjectiveBundleImplicitMethodBFullA)
    KNITRO.KN_set_compcons(kc, zeros(Int32, length(obj.complement_index[:, 1])), Int32.(obj.complement_index[:, 1] .+ 0), Int32.(obj.complement_index[:, 2] .+ 0))
end

function inner_loop_internal(obj::PsiObjectiveBundleImplicitMethodBFullA, θ)
    obj.moments!(@view(obj.H[:, 1]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1,1] * (-1.0)^obj.find_smallest

    nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)

    INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        INNER_INFEAS_COUNT[] += 1
    end

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus
    else
        obj.x .= NaN
        return -1e10, x, nStatus
    end
end

function hessian!(h, obj::PsiObjectiveBundleImplicitMethodBFullA)
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
end

"""
    make_gravity_grad(γobj, D)

Build the trivial direct-gradient function theta -> l-vector for the gravity
constraint, closed over the fixed (data-only) context. D-general.
"""
function make_gravity_grad(γobj, D::Int)
    Aod_offset = 3 + D
    τ = γobj.τ; cHat = γobj.cHat; wHat = γobj.wHat
    lambda = reshape(γobj.P, (D, D))'
    sumGrav_scalar = θθ -> begin
        T = eltype(θθ)
        μ = θθ[1]
        Aod_θ = reshape(θθ[Aod_offset+1:Aod_offset+D^2], (D, D))
        Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
        AodPow = (Aod ./ cHat) .^ (-μ)
        # Wτ is already within-transformed (orthogonal to row/col means), so Σ Wτ·within(x) = Σ Wτ·x
        # for any x (verified numerically, scratch/check_fwl.jl) -- skip within-transforming AodPow,
        # one fewer D×D log+demean pass and a shorter chain rule through this ForwardDiff.gradient.
        Wτ = Main.withinTransform(τ)
        lnAod = log.(AodPow)
        s = zero(T)
        for o in 1:D, d in 1:D
            s += Wτ[o, d] * lnAod[o, d]
        end
        return s
    end
    return θθ -> ForwardDiff.gradient(sumGrav_scalar, θθ)
end
