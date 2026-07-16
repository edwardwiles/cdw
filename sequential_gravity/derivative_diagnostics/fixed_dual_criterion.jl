# ============================================================================
# Part 1.1: the exact, unoptimized fixed-dual criterion Q(A,x) for the
# sequential/profiled method's inner problem, and the identity test
# Q(A_k,x_k*) = delta*(A_k,gamma').
#
# Architecture (see derivative_methods_report.md for the full note):
#   - Q(A,x) is LITERALLY `PsiObjectiveBundleDelta`'s own callable
#     (cc_algo/PsiObjectiveBundle.jl:298-...): f = sum(Psi(arg0))/M + zeta,
#     arg0 = H[:,2:1+oci]*(-x). No new formula is introduced here -- this file
#     only supplies a thin wrapper that (a) recomputes H = [K, 1, G] at an
#     arbitrary theta via the EXACT SAME `moments!` function production uses
#     (hard argmin winner, full hybrid-divergence conjugate via Psi!), and
#     (b) evaluates the SAME struct's callable with x supplied externally
#     (never re-optimized -- x is just data here, not a KNITRO decision
#     variable).
#   - Free outer parameter for this method: theta[4:3+D] = Acol = A_od_theta
#     for the focal destination (Case A per Part 6 -- confirmed by
#     `focal_moments_directgp.jl:19`, `run_profiled_production.jl`'s
#     FreeParamMap free_idx = vcat(3, 4:3+D)). No trade-share-inversion
#     derivative is needed for what this file computes.
# ============================================================================

"""
    build_fixed_dual_bundle(γobj, U, l, d, moments_fn; N=size(U,1), find_smallest=true)

Builds a `PsiObjectiveBundleDelta` identical in structure to `recover_lfd`'s
own construction in `run_profiled_production.jl` (same d, oci=d+1, same
`moments!`), for use as a REUSABLE evaluation object (its `.H` gets
overwritten on every call, never its `.x`).

`find_smallest` MUST match the real outer problem's own `find_smallest`
(lower-bound vs upper-bound search direction) whenever this bundle's result
will be compared against, or substituted into, a real production gradient --
`dual_criterion_fixed_x`'s sign convention reads `obj.find_smallest` from
THIS bundle, not from any other object. The struct's own keyword default
(`true`) silently produced a WRONG sign for the lower-bound case
(find_smallest=false in `run_profiled_production.jl::run_one_bound`) in an
earlier version of `gradient_method_wiring.jl`, which never passed this
argument and therefore always got `find_smallest=true` regardless of which
bound was actually being solved -- caught while building the full-(D+2)
wiring (`full_gradient_method_wiring.jl`); fixed here so every caller must
pass it explicitly at the two production call sites. Standalone diagnostic
drivers that only ever exercise `find_smallest=true` (Parts 1-4's own
identity/FD tests, which construct AND solve with the SAME bundle so the
convention is self-consistent regardless of the flag's true economic
meaning there) are unaffected by the default and need no changes.
"""
function build_fixed_dual_bundle(γobj, U::Matrix{Float64}, l::Int, d::Int, moments_fn::Function;
        N::Int=size(U, 1), find_smallest::Bool=true)
    return PsiObjectiveBundleDelta(γ=γobj, find_smallest=find_smallest, (moments!) = moments_fn, moments_jacobian! = error,
        d=d, outer_constr_index=d + 1, inequality_index=Int64[], complement_index=[0 0],
        l=l, U=U, N=N, lower_limit=-5000.0,
        outer_loop_opt="ek_outer_loop_options.opt", inner_loop_opt="ek_inner_loop_options.opt")
end

"""
    dual_criterion_fixed_x(θ, obj, x_fixed)

Q(A,x_fixed;γ') = the exact fixed-dual scalar criterion at θ (A embedded in
θ[4:3+D]), with the inner dual variables x=(ζ,λ) held FIXED at `x_fixed`
(never re-optimized, no KNITRO call). Recomputes θ's moments (hence winners)
fresh via `obj.moments!` -- the exact hard argmin, no smoothing -- then calls
the bundle's own callable with x supplied and g/θ both empty, which by
construction (see `(Q::PsiObjectiveBundleDelta)(x,g,θ;...)`,
cc_algo/PsiObjectiveBundle.jl:298-300,326-345) skips every optimization/
Jacobian branch and returns exactly `f = sum(Psi(arg0))/M + ζ`.

`obj.U` MUST be the same draws matrix on every call for a valid finite
difference (common random numbers) -- this function never touches obj.U.

Applies `obj.find_smallest`'s sign convention (matching `inner_loop`'s own
`val *= -1.0 if find_smallest` -- cc_algo/inner_loop_functions.jl:165-173) so
this ALREADY returns the same signed quantity as δ*(θ) at the optimum, not
the bundle callable's raw (possibly negated) `f`. This matters for every
downstream consumer (FD gradients, directional derivatives): get it right
once, here, rather than requiring every caller to remember the flip.
"""
function dual_criterion_fixed_x(θ::AbstractVector, obj::PsiObjectiveBundleDelta, x_fixed::Vector{Float64})
    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, 2] .= 1.0
    f = obj(x_fixed, Float64[], Float64[])
    return obj.find_smallest ? -f : f
end

"""
    test_fixed_dual_identity(θ, moments_fn, d, γobj, U; l=length(θ))

Part 1.1's required test: solve the inner problem once (via `recover_lfd`-
equivalent machinery, reusing `inner_loop` directly here so we get `x*`
itself, not just the LFD `p`), then check
`dual_criterion_fixed_x(θ, obj, x*) == δ*(θ)` up to solver tolerance.
Returns (δ_star, Q_fixed, abs_diff, rel_diff, x_star, nStatus).
"""
function test_fixed_dual_identity(θ::Vector{Float64}, moments_fn::Function, d::Int, γobj, U::Matrix{Float64}; l::Int=length(θ), find_smallest::Bool=true)
    obj = build_fixed_dual_bundle(γobj, U, l, d, moments_fn; find_smallest=find_smallest)
    δ_star, x_star, nStatus = inner_loop(obj, θ)
    Q_fixed = dual_criterion_fixed_x(θ, obj, x_star)
    abs_diff = abs(Q_fixed - δ_star)
    rel_diff = abs_diff / max(abs(δ_star), 1e-12)
    return (δ_star=δ_star, Q_fixed=Q_fixed, abs_diff=abs_diff, rel_diff=rel_diff, x_star=x_star, nStatus=nStatus, obj=obj)
end
