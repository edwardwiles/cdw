# ============================================================================
# Part 9: the fixed-A* problem as a feasible incumbent for the full-A search.
#
# `outer_solve_fixedA` is `outer_solve_nested_cached`'s twin with ONLY
# gamma'_focal free (A pinned at whatever Acol values θinit carries -- meant
# to be called with θinit=θr0, i.e. A=A*). Since gamma'_focal's AD gradient
# is exact (no winner/argmax dependence on theta[3] -- verified in
# derivative_methods_report.md and directly in focal_moments_directgp.jl),
# this is a plain 1-D AD outer search, not a finite-difference one -- no
# gradient_method choice needed here.
#
# The full-A feasible set contains the fixed-A* feasible set (A* is always a
# feasible choice of A[.,focal]), so:
#   kappa_lower_full <= kappa_lower_fixed   (lower bound: full-A weakly BEATS fixed-A)
#   kappa_upper_full >= kappa_upper_fixed   (upper bound: full-A weakly BEATS fixed-A)
# A violation (beyond numerical tolerance) means the full-A search did NOT
# reach a valid optimum and must be flagged, not reported as an economic
# result -- this is the Part 9 sanity check the task requires.
# ============================================================================

"""
    outer_solve_fixedA(find_smallest, θinit; δ=δ)

θinit's Acol block (θinit[4:3+D]) is held FIXED at its given value throughout
(only gamma'_focal, θ[3], is a free outer KNITRO variable). Returns the same
tuple shape as `outer_solve_nested_cached` (gp, θ_min_full, nStatus, best_θ,
best_κ, best_warm, cache) plus the raw `obj`/`fpmap`/`m!`-state Refs needed to
warm-start a subsequent full-A solve at this endpoint.
"""
function outer_solve_fixedA(find_smallest, θinit; δ::Real=δ)
    d = D + 2; oci = d + 1
    CS.check_methodB_valid(d, oci)
    m!, gcol, lastRmean, best_θ, best_κ, best_warm, lastθ_st, lastRcol_st, dRdθ_st, lastok_st =
        make_stateful_moments(; use_exact_grad = true, find_smallest = find_smallest, δ = δ)
    obj = CS.PsiObjectiveBundleImplicitMethodB(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = m!, moments_jacobian! = error, d = d, outer_constr_index = oci,
        inequality_index = Int64[], complement_index = [0 0], l = length(θinit), U = U, N = JacW,
        lower_limit = -50, use_cached_x = false,
        outer_loop_opt = OUTER_OPT_FILE, inner_loop_opt = INNER_OPT_FILE)

    l_full = length(θinit)
    free_idx = [3]                                    # ONLY gamma'_focal free
    fixed_idx = vcat([1, 2], collect(4:3+D))           # mu, sigma, and ALL of Acol pinned
    fixed_vals = θinit[fixed_idx]
    fpmap = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
    @assert CS.n_free(fpmap) == 1

    θ_lo_fA = copy(θ_lo); θ_hi_fA = copy(θ_hi)
    @views θ_lo_fA[4:3+D] .= θinit[4:3+D]; @views θ_hi_fA[4:3+D] .= θinit[4:3+D]   # pin bounds to match fixed_vals exactly (pack_bounds_free requires this)

    div_grad_fn! = make_seq_div_grad_fn!(obj, fpmap)   # plain AD, exact for the sole free coordinate
    function obj_grad_fn!(g_free, x_free)
        fill!(g_free, 0.0)
        g_free[1] = (-1.0)^find_smallest
    end

    r = CS.outer_loop_cached(obj, fpmap, θ_lo_fA, θ_hi_fA, θinit;
        obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
        has_gravity = false, use_cache = true, outer_loop_opt = OUTER_OPT_FILE)

    gp = r.θ_min_full[3]
    return (gp = gp, θ_min_full = r.θ_min_full, nStatus = r.nStatus, best_θ = best_θ[], best_κ = best_κ[],
            best_warm = best_warm[], cache = r.cache, obj = obj)
end

"""
    exact_inner_divergence_at(θ)

Part 10's post-solve audit primitive: freezes the gravity linearization at
`θ` (a fresh `seq_gravcol`+`grad_R_theta` call -- NOT reusing any stale
cached state), then solves the EXACT full-(D+2) inner CC problem at `θ`
(cold-started KNITRO `inner_loop`, no warm start, no fixed-x approximation).
Returns (δ_star, R_mean, R_lin_residual, nStatus, umat, p).

ALWAYS uses find_smallest=true for the underlying `PsiObjectiveBundleDelta`
(the only valid value for computing genuine delta* -- see
full_gradient_method_wiring.jl's module docstring for the earlier bug this
corrects: delta* has no dependence on which OUTER bound direction produced
`θ`, and threading that bound's find_smallest through here silently negated
delta* for the lower-bound case).
"""
function exact_inner_divergence_at(θ::Vector{Float64})
    frozen = freeze_gravity_linearization(θ, seq_gravcol, grad_R_theta)
    frozen.ok || return (δ_star=Inf, R_mean=Inf, nStatus=-999, gravity_ok=false)
    moments_fn = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, frozen.θ, frozen.Rcol, frozen.gcol, frozen.dRdθ)
    obj = build_fixed_dual_bundle(γ, U, length(θ), D + 2, moments_fn; find_smallest=true)
    δ_star, x_star, nStatus = inner_loop(obj, θ)
    return (δ_star=δ_star, R_mean=frozen.R, nStatus=nStatus, gravity_ok=(abs(frozen.R) <= 5e-4), x_star=x_star, frozen=frozen)
end

"""
    exact_fixedA_divergence_at(γp_target, θref)

Part 10's "same gamma', A held fixed at A*" comparator: solves the exact
full-(D+2) inner problem at theta=(mu,sigma,gamma'_focal=γp_target,
Acol=θref's Acol) -- i.e. the SAME γ' target the moved-A search reached, but
with A pinned at θref's value (normally A*).
"""
function exact_fixedA_divergence_at(γp_target::Real, θref::Vector{Float64})
    θ = copy(θref); θ[3] = γp_target
    return exact_inner_divergence_at(θ)
end
