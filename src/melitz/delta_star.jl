# Wiring into the existing Christensen-Connault inner minimum-divergence loop.
# See docs/melitz_delta_star.md Section "Reused Ricardian infrastructure".
#
# Reuses cc_algo/PsiObjectiveBundle.jl's PsiObjectiveBundleDelta and
# cc_algo/inner_loop_functions.jl's inner_loop UNMODIFIED -- the Melitz module supplies
# only a moments!(K, G, theta, U, obj) function matching the required contract.
#
# TWO outer-vector representations (main prompt Section 2, addendum Section 6):
#   - the ECONOMIC vector theta_econ (length 2D^2): (log_gamma_prime_j, vec(log A) [D^2],
#     log_f_free [D^2-1, every cell except (target,target)]) -- NOT assumed
#     gravity-feasible;
#   - the FREE vector theta_free (length 2D^2-2): (log_gamma_prime_j, A-pivoted [D^2-1],
#     f-pivoted [D^2-2]) -- gravity-feasible BY CONSTRUCTION via the two GravityPivot
#     eliminations in equilibrium.jl. This is what is actually packed/unpacked as `theta`
#     for `PsiObjectiveBundleDelta` (`l = 2D^2-2`) -- the coordinate system a future
#     finite-delta outer search would optimize over.
#
# Unlike the superseded closure's adapter (valid only at a fixed theta*, since the old
# counterfactual was reused unchanged from construction time), `melitz_moments_adapter!`
# here fully RE-EQUILIBRATES (A, f[target,target], the autarky counterfactual) from
# `theta` on every call -- required now that `gamma_prime_target` is itself part of
# `theta`, and needed for the nearby-perturbation tests (main prompt Section 12).

using LinearAlgebra: dot

"lin2od(i, D) -> (o, d): invert Julia's column-major vec()/reshape() linear index."
lin2od(i::Int, D::Int) = (mod1(i, D), div(i - 1, D) + 1)
"od2lin(o, d, D) -> i: column-major linear index, inverse of lin2od."
od2lin(o::Int, d::Int, D::Int) = o + (d - 1) * D

"""
    melitz_outer_layout(D, target_country) -> NamedTuple

Named index ranges for the ECONOMIC outer vector `theta_econ = vcat(log(gamma_prime_j),
vec(log.(A)), log_f_free)`, length `2*D^2` (main prompt Section 2). `f_free_lin` lists the
`D^2-1` column-major linear indices of the free f-cells (every cell except `(j,j)`, in
increasing order) -- `f[j,j]` is never part of this vector.
"""
function melitz_outer_layout(D::Int, target_country::Int)
    jj_lin = od2lin(target_country, target_country, D)
    f_free_lin = [i for i in 1:D^2 if i != jj_lin]
    return (D=D, target_country=target_country, jj_lin=jj_lin,
            gamma=1:1, A=2:(D^2 + 1), f_free=(D^2 + 2):(2D^2),
            f_free_lin=f_free_lin, length=2D^2)
end

"""
    melitz_pack_outer(gamma_prime_j, A, f, layout) -> theta_econ

Packs the ECONOMIC outer vector. `f[j,j]` is read from `f` for convenience of callers
that already have a full D x D `f` matrix, but is NOT included in the packed vector.
"""
function melitz_pack_outer(gamma_prime_j::Real, A::AbstractMatrix, f::AbstractMatrix, layout)
    f_free_vals = [log(f[lin2od(i, layout.D)...]) for i in layout.f_free_lin]
    return vcat(log(gamma_prime_j), vec(log.(A)), f_free_vals)
end

"""
    melitz_unpack_outer(theta_econ, layout; w_prime_j, expenditure_prime_j, sigma)
        -> (A, f, gamma_prime_j, f_jj)

Unpacks the ECONOMIC outer vector. Does NOT assume gravity feasibility (that is the
FREE-vector representation's job) -- `A` is read off directly, and `f[j,j]` is DERIVED
from `gamma_prime_j`/`A[j,j]` via `derive_fjj_from_autarky_cutoff` (main prompt Section 3;
`f[j,j]` is never itself packed).
"""
function melitz_unpack_outer(theta_econ::AbstractVector, layout;
                              w_prime_j::Real, expenditure_prime_j::Real, sigma::Real)
    D = layout.D
    gamma_prime_j = exp(theta_econ[1])
    A = reshape(exp.(theta_econ[layout.A]), D, D)
    A_jj = A[layout.target_country, layout.target_country]
    f_jj = derive_fjj_from_autarky_cutoff(gamma_prime_j, w_prime_j, 1.0, A_jj, expenditure_prime_j, sigma)
    f = zeros(eltype(A), D, D)
    f[layout.target_country, layout.target_country] = f_jj
    for (k, i) in enumerate(layout.f_free_lin)
        o, d = lin2od(i, D)
        f[o, d] = exp(theta_econ[layout.f_free[k]])
    end
    return A, f, gamma_prime_j, f_jj
end

"""
    build_gravity_pivots(tau, target_country) -> (c_full, A_pivot)

`c_full = gravity_coefficient_vector(D, tau)` (length D^2, equilibrium.jl -- built from
`withinTransform`, Gate A5; equals `vec(withinTransform(tau))` exactly by that transform's
self-adjointness, see that function's docstring) is the SHARED linear coefficient
vector for both gravity restrictions. `A_pivot` is fixed once `tau` is fixed (offset
`g0=0` exactly, since every A cell is free); the f-pivot must be REBUILT on every call
(its offset depends on the current `f[j,j]`, itself a function of `gamma_prime_j`) --
see `expand_free_theta` below.
"""
function build_gravity_pivots(tau::Matrix{Float64}, target_country::Int)
    D = size(tau, 1)
    c_full = gravity_coefficient_vector(D, tau)
    A_pivot = build_gravity_pivot(c_full, 0.0)
    return c_full, A_pivot
end

"""
    reduce_to_free_theta(p::MelitzPrimitives, ctx) -> theta_free

Full `(A, f, gamma_prime_target)` -> the `2D^2-2` FREE gravity-pivoted vector (inverse of
`expand_free_theta`). Requires `p` to already be gravity-feasible (both
`gravity_residuals(p)` components ~0) -- used to seed KNITRO/tests from an already-valid
outer point, not to project an arbitrary point onto the gravity manifold.
"""
function reduce_to_free_theta(p::MelitzPrimitives, ctx)
    D, j = ctx.D, ctx.target_country
    logA_full = vec(log.(p.A))
    A_free = pivot_reduce(logA_full, ctx.A_pivot)

    f_jj = p.f[j, j]
    g0_f = ctx.c_full[ctx.jj_lin] * log(f_jj)
    avoid_f = f_pivot_avoid_index(ctx.A_pivot.pivot, ctx.f_free_lin)
    f_pivot = build_gravity_pivot(ctx.c_full[ctx.f_free_lin], g0_f; avoid=avoid_f)
    logf_free_full = [log(p.f[lin2od(i, D)...]) for i in ctx.f_free_lin]
    f_free = pivot_reduce(logf_free_full, f_pivot)

    return vcat(log(p.gamma_prime_target), A_free, f_free)
end

"""
    expand_free_theta(theta_free, ctx) -> (A, f, gamma_prime_j, f_jj)

The `2D^2-2` FREE gravity-pivoted vector -> full `(A, f)` (both gravity restrictions
EXACTLY satisfied by construction, main prompt Section 5). Order matters: `A` must be
expanded before `f[j,j]` can be derived (needs `A[j,j]`), and `f[j,j]` must be known before
the f-pivot's offset `g0` can be built.
"""
function expand_free_theta(theta_free::AbstractVector{T}, ctx) where {T}
    D, j = ctx.D, ctx.target_country
    nA = D^2 - 1
    log_gamma_prime_j = theta_free[1]
    gamma_prime_j = exp(log_gamma_prime_j)
    A_free = @view theta_free[2:1+nA]
    f_free_free = @view theta_free[2+nA:end]

    logA_full = pivot_expand(A_free, ctx.A_pivot)
    A = exp.(reshape(logA_full, D, D))
    A_jj = A[j, j]

    expenditure_prime_j = ctx.w_prime * ctx.L[j]
    f_jj = derive_fjj_from_autarky_cutoff(gamma_prime_j, ctx.w_prime, 1.0, A_jj, expenditure_prime_j, ctx.sigma)

    g0_f = ctx.c_full[ctx.jj_lin] * log(f_jj)
    avoid_f = f_pivot_avoid_index(ctx.A_pivot.pivot, ctx.f_free_lin)
    f_pivot = build_gravity_pivot(ctx.c_full[ctx.f_free_lin], g0_f; avoid=avoid_f)
    logf_free_full = pivot_expand(f_free_free, f_pivot)

    f = zeros(T, D, D)
    f[j, j] = f_jj
    @inbounds for (k, i) in enumerate(ctx.f_free_lin)
        o, d = lin2od(i, D)
        f[o, d] = exp(logf_free_full[k])
    end

    return A, f, gamma_prime_j, f_jj
end

"""
    melitz_moments_adapter!(K, G, theta, U, obj)

Adapter matching the `(K, G, theta, U, obj) -> nothing` contract required by
`PsiObjectiveBundleDelta`. `theta` is the FREE `2D^2-2` vector. `obj.gamma` (`ctx`) holds
everything besides `theta` needed to reconstruct the model: `D, sigma, theta_star,
target_country, tau, w, w_prime, L, expenditure, cutoff, moment_layout, X_data, c_full,
A_pivot, jj_lin, f_free_lin`. FULLY re-equilibrates `(A, f, gamma_prime_target)` and the
autarky counterfactual from `theta` on every call (general, not fixed-theta*-only --
contrast the superseded closure's adapter).
"""
function melitz_moments_adapter!(K, G, theta, U, obj)
    ctx = obj.γ
    A, f, gamma_prime_j, f_jj = expand_free_theta(theta, ctx)
    D = ctx.D

    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country,
                                   ctx.tau, ctx.w, A, f, gamma_prime_j)
    eq = MelitzEquilibrium(ctx.expenditure, ones(Float64, D), ctx.cutoff, ctx.X_data)
    expenditure_prime = ctx.w_prime * ctx.L[ctx.target_country]
    cf = MelitzCounterfactual(ctx.target_country, ctx.w_prime, expenditure_prime,
                               1.0, expenditure_prime)

    melitz_moments!(K, G, primitives, eq, cf, U, ctx.moment_layout; X_data=ctx.X_data)
    return nothing
end

"""
    build_melitz_psi_bundle(data::MelitzSyntheticData; X_data=data.equilibrium.trade_flow,
                             inner_loop_opt=..., outer_loop_opt=...)
        -> (obj::PsiObjectiveBundleDelta, theta_free::Vector{Float64})

Builds a `PsiObjectiveBundleDelta` wired to the Melitz moments (`d = D^2+1`) and the
FREE gravity-pivoted `theta_free` (`l = 2D^2-2`, main prompt Section 2's preferred
"reuse the gravity pivots" option) reduced from `data.primitives`'s own
`(A, f, gamma_prime_target)`.
"""
function build_melitz_psi_bundle(data::MelitzSyntheticData;
                                  X_data::Matrix{Float64}=data.equilibrium.trade_flow,
                                  inner_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_inner_loop_options.opt"),
                                  outer_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_outer_loop_options.opt"))
    p, eq, cf = data.primitives, data.equilibrium, data.counterfactual
    D = p.D
    j = p.target_country
    z_draws = data.z_draws

    moment_layout = MelitzMomentLayout(D)
    c_full, A_pivot = build_gravity_pivots(p.tau, j)
    outer_layout = melitz_outer_layout(D, j)

    ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau, w=p.w,
           w_prime=cf.w_prime, L=data.L, expenditure=eq.expenditure, cutoff=eq.cutoff,
           moment_layout=moment_layout, X_data=X_data, c_full=c_full, A_pivot=A_pivot,
           jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin)

    theta_free = reduce_to_free_theta(p, ctx)

    obj = PsiObjectiveBundleDelta(
        γ=ctx,
        (moments!)=melitz_moments_adapter!,
        d=moment_layout.num_moments,
        l=length(theta_free),
        inequality_index=Int64[],
        U=z_draws,
        inner_loop_opt=inner_loop_opt,
        outer_loop_opt=outer_loop_opt,
    )

    return obj, theta_free
end

"""
    melitz_primal_divergence(weights, W) -> Float64

Primal phi-divergence `(1/W)*sum(phi(W*weight))` for the SAME hybrid KL/quadratic family
whose convex conjugate is `Psi!`/`dPsi!` (`cc_algo/Psi.jl`): `phi(m) = m*log(m)-m+1` for
`m<=e`, `phi(m) = m^2/(2e)-e/2+1` for `m>e` (the crossover `m=e` matches `Psi`'s own
`arg0<=1`/`arg0>1` branch point, since `m=exp(arg0)` in the KL branch). Ported directly from
`sequential_gravity/run_profiled_production.jl`'s `divergence_of` (the reference
implementation this module's `melitz_recover_lfd` already reuses the dual-recovery recipe
from) -- used to verify primal/dual agreement (main prompt Section 4), not re-derived
differently.
"""
function melitz_primal_divergence(weights::AbstractVector, W::Int)
    e = exp(1)
    acc = 0.0
    @inbounds for p in weights
        m = p * W
        if !(m > 0) || !isfinite(m)
            return Inf
        elseif m <= e
            acc += m * log(m) - m + 1
        else
            acc += m^2 / (2e) - e / 2 + 1
        end
    end
    return acc / W
end

"""
    melitz_recover_lfd(obj::PsiObjectiveBundleDelta, theta) -> MelitzLFDResult

Runs `inner_loop(obj, theta)` (real KNITRO, `cc_algo/inner_loop_functions.jl`,
unmodified) and RECOVERS THE PRIMAL LFD -- not just the dual objective/status (the
superseded closure's `run_melitz_inner_delta` returned only `(val, x, nStatus)`).

Reuses the exact conjugate-derivative recipe already validated (and copy-pasted 11x) in
`sequential_gravity/run_profiled_production.jl`'s `recover_lfd`: after `inner_loop`
returns the KNITRO-optimal dual vector `x = (zeta, lambda...)`, per-draw
`arg0[w] = -x[1] - dot(G[w,1:d], x[2:end])`, then `dPsi!` (the divergence's conjugate
derivative, `cc_algo/Psi.jl`) maps `arg0` to unnormalized LFD weights, normalized by their
sum. This is a SEPARATE, standalone reconstruction from `G`/`x` alone -- it does not read
`K`/`H[:,1]` at all, consistent with `K` being unused by a fixed-theta `inner_loop` call
(docs Section 10).

`lfd_ok` (main prompt Section 4) requires ALL of: `nStatus==0` (checked before ANY
normalization is attempted -- an unbounded/failed dual is never normalized and presented as
an LFD); all dual variables finite; the raw (pre-normalization) density-ratio sum close to
`W` (`normalization_tol`); every imposed moment holding under the recovered weights
(`moment_tol`); and the PRIMAL divergence at the recovered weights
(`melitz_primal_divergence`) agreeing with the dual objective `val`, via a SCALE-AWARE gap
test (Gate A4): `gap <= max(gap_atol, gap_rtol*max(1, |primal|, |dual|))`, default
`gap_atol=1e-10, gap_rtol=1e-6` -- replacing the previous flat `divergence_tol=1e-4`, which
was documented as much larger than `Delta(theta_Fstar)` itself (~1e-6 to 1e-7) and so could
not actually discriminate a bad LFD from a good one at this benchmark's scale.
Strictly stronger than the SUPERSEDED check (finite/nonnegative weights only), which let a
numerically degenerate, `nStatus!=0` reconstruction through as "ok" (docs Section 12).

KNITRO's own KKT diagnostics (`KN_get_abs_opt_error`/`KN_get_abs_feas_error`) are captured
non-invasively via `cc_algo/inner_loop_functions.jl`'s `INNER_LAST_OPT_ERR`/
`INNER_LAST_FEAS_ERR` globals (set on every inner KNITRO solve, safe under the existing
single-flight `guard_enter_inner_solve!` discipline) -- no change to `inner_loop`'s shared
return signature was needed.
"""
function melitz_recover_lfd(obj, theta::AbstractVector; moment_tol::Real=1e-6,
                             normalization_tol::Real=1e-6,
                             gap_atol::Real=1e-10, gap_rtol::Real=1e-6)
    d = obj.d
    W = size(obj.U, 1)
    val, x, nStatus = inner_loop(obj, theta)
    kkt_opt_error = INNER_LAST_OPT_ERR[]
    kkt_feas_error = INNER_LAST_FEAS_ERR[]

    if nStatus != 0 || !all(isfinite, x)
        return MelitzLFDResult(val, x, nStatus, fill(1.0 / W, W), false,
                                fill(NaN, d), NaN, NaN, NaN, NaN,
                                NaN, val, NaN, NaN, NaN, kkt_opt_error, kkt_feas_error)
    end

    K = zeros(W)
    G = zeros(W, d)
    obj.moments!(K, G, theta, obj.U, obj)

    arg0 = zeros(W)
    @inbounds for w in 1:W
        arg0[w] = -x[1] - dot(view(G, w, 1:d), view(x, 2:length(x)))
    end
    LFD = zeros(W)
    dPsi!(LFD, arg0)
    s = sum(LFD)

    if !(isfinite(s) && s > 0) || !all(isfinite, LFD) || !all(>=(0), LFD)
        return MelitzLFDResult(val, x, nStatus, fill(1.0 / W, W), false,
                                fill(NaN, d), NaN, NaN, NaN, NaN,
                                NaN, val, NaN, NaN, NaN, kkt_opt_error, kkt_feas_error)
    end

    normalization_residual = s / W - 1
    weights = LFD ./ s
    moment_residuals = vec(sum(G .* weights, dims=1))
    max_moment_residual = maximum(abs.(moment_residuals))

    primal_divergence = melitz_primal_divergence(weights, W)
    dual_divergence = val
    primal_dual_gap = abs(primal_divergence - dual_divergence)
    gap_tol = max(gap_atol, gap_rtol * max(1.0, abs(primal_divergence), abs(dual_divergence)))

    # Section 4: an unbounded/failed dual (already excluded above via nStatus!=0) must
    # never be normalized and presented as a valid LFD; a BOUNDED dual can still fail to
    # be trustworthy if normalization was already far off pre-renormalization, an imposed
    # moment doesn't actually hold under the recovered weights, or the primal divergence at
    # these weights disagrees with the dual objective `val` -- all checked explicitly here
    # rather than inferred from finiteness alone (the previous, looser definition let
    # exactly this class of untrustworthy LFD through, docs Section 12).
    lfd_ok = abs(normalization_residual) < normalization_tol &&
             max_moment_residual < moment_tol &&
             primal_dual_gap <= gap_tol

    return MelitzLFDResult(val, x, nStatus, weights, lfd_ok, moment_residuals, normalization_residual,
                            minimum(weights), maximum(weights), maximum(abs.(W .* weights .- 1)),
                            primal_divergence, dual_divergence, primal_dual_gap,
                            max_moment_residual, normalization_residual,
                            kkt_opt_error, kkt_feas_error)
end

"""
    run_melitz_inner_delta(data::MelitzSyntheticData; kwargs...) -> (lfd::MelitzLFDResult, obj)

Runs the REAL CC inner minimum-divergence loop at `data.primitives`'s own outer point and
recovers the LFD. `lfd.Delta` is `Delta(theta)` -- should be ~0 for the exact-sample smoke
test (main prompt Section 11), small but nonzero against the closed-form population
target (Section 10's `Delta(theta)` terminology -- this is a fixed-outer-parameter inner
solve, NOT a completed outer `Delta*` search).
"""
function run_melitz_inner_delta(data::MelitzSyntheticData; kwargs...)
    obj, theta = build_melitz_psi_bundle(data; kwargs...)
    lfd = melitz_recover_lfd(obj, theta)
    return lfd, obj
end
