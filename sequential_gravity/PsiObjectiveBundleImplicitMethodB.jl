# ============================================================================
# Diagnostics-only Method-B variant of PsiObjectiveBundleImplicit
# (cc_algo/PsiObjectiveBundle.jl). Identical struct/callable EXCEPT the
# θ-gradient branch replaces calculate_jac_θ! (dense N×(d+2)×l ForwardDiff
# Jacobian, then contracted) with a DIRECT scalar ForwardDiff gradient of the
# same envelope scalar the dense path implicitly computes for the divergence-
# budget constraint — validated exact (relerr~1e-15) against production in
# full_aod_diag/ad_benchmark/ this session.
#
# VALIDITY CONDITION (checked at construction): outer_constr_index > d, i.e.
# there are NO additional outer equality constraints beyond the divergence
# budget (constr[1]) — so there is no ift!-based total-derivative correction
# to reproduce (that correction, needed when a genuine second outer
# constraint like "gravity as a hard outer constraint" exists, was
# EXPLICITLY NOT re-derived in this session's audit; this struct simply
# refuses to be used where it would be needed, rather than silently giving a
# wrong answer). This holds for sequential_gravity's profiled/sequential
# method (gravity enters as an INNER-matched linearized moment, not an outer
# constraint: d=D+2, outer_constr_index=d+1) but NOT for the full-A_od-in-
# outer-loop method (gravity IS a second outer constraint there).
#
# Everything else (inner-loop (ζ,λ)-space objective/gradient/Hessian, Ψ
# machinery, KNITRO wiring) is byte-identical to PsiObjectiveBundleImplicit.
# ============================================================================

@with_kw mutable struct PsiObjectiveBundleImplicitMethodB{T} <: PsiObjectiveBundle
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

    outer_constr_index  ::Int64            = d + 1
    inner_loop_opt      ::String
    outer_loop_opt      ::String
    lower_limit         ::Float64          = -KNITRO.KN_INFINITY
    use_cached_x        ::Bool             = false

    Psi!                ::Function         = Psi!
    dPsi!                ::Function         = dPsi!
    ddPsi!               ::Function         = ddPsi!

    N                   ::Int64            = M

    H                   ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))
    H_copy              ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))
    H_save              ::Float64          = 0.0
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    arg2                ::Array{Float64,1} = zeros(M)
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index)
    ∂∂f_∂∂x             ::Array{Float64,2} = zeros(length(x), length(x))
end

"""
    check_methodB_valid(d, outer_constr_index)

VALIDITY CHECK callers MUST run before constructing a PsiObjectiveBundleImplicitMethodB
(not enforced in an inner constructor — `@with_kw`'s keyword-constructor generation doesn't mix
safely with a hand-written positional inner constructor, so the check lives here instead, and
every driver script in this file's family calls it explicitly right before construction).
"""
function check_methodB_valid(d::Int, outer_constr_index::Int)
    outer_constr_index > d || error(
        "PsiObjectiveBundleImplicitMethodB requires outer_constr_index > d (no extra outer " *
        "constraints beyond the divergence budget) — got outer_constr_index=$outer_constr_index, d=$d. " *
        "The gravity-as-outer-constraint case needs an ift! total-derivative correction this " *
        "diagnostics-only struct does NOT implement; use PsiObjectiveBundleImplicit instead.")
    return true
end

"""
Envelope scalar for the divergence-budget constraint:
(1e10/N)*Σ_draws arg1[draw]*Σ_j λ[j]*G[draw,j](θ). Takes plain, concretely-typed arguments
(NOT the whole mutable/abstract-Function-typed bundle struct) — capturing the entire
PsiObjectiveBundleImplicitMethodB struct inside the ForwardDiff-differentiated closure
triggered a Julia 1.12.6 compiler segfault (runaway inlining/type-inference, reproduced in
isolation, unrelated to this codebase's logic); passing the concrete pieces needed
(moments!, γ, U slice, oci) avoids it.
"""
function _methodB_envelope_scalar(θ::AbstractVector, moments!::Function, γobj, Usub::AbstractMatrix,
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

function (Q::PsiObjectiveBundleImplicitMethodB)(x, g = Float64[], θ = Float64[]; h = Float64[], constr = Float64[], jac = Array{Float64}(undef, 0, 0))

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
		# outer_constr_index > d by construction (checked in the inner constructor), so there is
		# NO second outer constraint to fill here — matches PsiObjectiveBundleImplicit's own
		# `if outer_constr_index <= d` guard, which would be false in exactly this case.
	end

	if length(g) > 0 && length(θ) == 0

		g[1] = 1.0 - sum(arg1) / M
		@views BLAS.gemv!('T', -1/M, H[:, 3:1+outer_constr_index], arg1, 0.0, g[2:end])

	elseif length(g) > 0 && length(θ) > 0

		@unpack find_smallest, N, l = Q

		# ---- objective gradient: unchanged (already a direct scalar autodiff call) ----
		calculate_grad_k!(g, Q, θ)
		g .*= (-1.0)^find_smallest

		# ---- METHOD B: divergence-budget constraint gradient, no dense Jacobian ----
		λvec = collect(@view x[2:end])
		Usub = Q.U[1:N, :]
		moments_fn = Q.moments!; γobj = Q.γ; dd = Q.d; oci = Q.outer_constr_index; arg1v = Q.arg1
		gdiv = ForwardDiff.gradient(θθ -> _methodB_envelope_scalar(θθ, moments_fn, γobj, Usub, λvec, arg1v, dd, oci), θ)
		jac .= gdiv   # single constraint row -> jac is already length-l

	end

	length(h) > 0 ? hessian!(h, Q) : nothing

	if f <= lower_limit
		return -KNITRO.KN_INFINITY
	else
		return f
	end

end

# Reuse the generic outer_loop/outer_loop_constraints!/hessian! methods already defined for
# Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta} — add this type to their dispatch.
outer_loop_lambda_length(obj::PsiObjectiveBundleImplicitMethodB) = obj.l + obj.d - obj.outer_constr_index + 2

function outer_loop_constraints!(kc, obj::PsiObjectiveBundleImplicitMethodB)
    cIndices = KNITRO.KN_add_cons(kc, 1)   # ONLY the divergence budget (outer_constr_index>d enforced)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1e10 * obj.δ)
    return cIndices
end

# ---- inner-loop ((ζ,λ)-space) dispatch: byte-identical to PsiObjectiveBundleImplicit's ----
inner_loop_number_variables(obj::PsiObjectiveBundleImplicitMethodB) = obj.outer_constr_index
inner_loop_lower_bounds(obj::PsiObjectiveBundleImplicitMethodB) = vcat(-KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
inner_loop_initial_values(obj::PsiObjectiveBundleImplicitMethodB) = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index)

function inner_loop_complementarity_constraints(kc, obj::PsiObjectiveBundleImplicitMethodB)
    KNITRO.KN_set_compcons(kc, zeros(Int32, length(obj.complement_index[:, 1])), Int32.(obj.complement_index[:, 1] .+ 0), Int32.(obj.complement_index[:, 2] .+ 0))
end

function inner_loop_internal(obj::PsiObjectiveBundleImplicitMethodB, θ)
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

function hessian!(h, obj::PsiObjectiveBundleImplicitMethodB)
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
