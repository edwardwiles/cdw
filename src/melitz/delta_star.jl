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
using ForwardDiff

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

Session prompt Section 1.1: the A-pivot is restricted to the largest-`|c|` ELIGIBLE
OFF-DIAGONAL cell (`avoid` excludes every `(o,o)` cell). Before this fix, `A_pivot` was
chosen unrestricted -- and since Gate A5's `withinTransform` switch made `|c|`
systematically diagonal-dominated (`f_pivot_domestic_avoid_indices`'s own finding), the
unrestricted choice landed on a DOMESTIC cell in the active production configuration,
never exercised against an outer solve. Off-diagonal pivots preserve coordinate
locality (session prompt Section 1.1): a domestic A-pivot cell feeds directly into that
country's own baseline AND autarky cutoffs (`derive_fjj_from_autarky_cutoff` reads
`A[j,j]` when `j` is the pivoted country), entangling every outer coordinate's effect on
gravity-feasibility with the focal autarky construction; an export-cell pivot does not.
The f-pivot's own off-diagonal + distinct-from-A-pivot restriction is applied where the
f-pivot is actually built (`reduce_to_free_theta`/`expand_free_theta`, via
`f_gravity_pivot_avoid_indices`), since its domain (`f_free_lin`, already excluding
`(j,j)`) is only available there.
"""
function build_gravity_pivots(tau::Matrix{Float64}, target_country::Int)
    D = size(tau, 1)
    c_full = gravity_coefficient_vector(D, tau)
    A_diag_avoid = [od2lin(o, o, D) for o in 1:D]
    A_pivot = build_gravity_pivot(c_full, 0.0; avoid=A_diag_avoid)
    return c_full, A_pivot
end

"""
    gravity_pivot_cells(tau, target_country) -> (A_pivot_od, f_pivot_od)

Session prompt Section 1.1 diagnostic: the physical `(o,d)` cells chosen as the A- and
f-gravity pivots for a given `tau`/`target_country`, using the SAME off-diagonal,
mutually-distinct restriction the active outer coordinate system enforces
(`build_gravity_pivots`/`f_gravity_pivot_avoid_indices`). `g0=0.0` is used for the
f-pivot's offset here (any value would do -- WHICH cell is chosen depends only on `c`
and the avoid-set, never on `g0`; see `pivot_expand`).
"""
function gravity_pivot_cells(tau::Matrix{Float64}, target_country::Int)
    D = size(tau, 1)
    c_full, A_pivot = build_gravity_pivots(tau, target_country)
    jj_lin = od2lin(target_country, target_country, D)
    f_free_lin = [i for i in 1:D^2 if i != jj_lin]
    avoid_f = f_gravity_pivot_avoid_indices(D, f_free_lin, A_pivot.pivot)
    f_pivot = build_gravity_pivot(c_full[f_free_lin], 0.0; avoid=avoid_f)
    A_pivot_od = lin2od(A_pivot.pivot, D)
    f_pivot_od = lin2od(f_free_lin[f_pivot.pivot], D)
    return A_pivot_od, f_pivot_od
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
    avoid_f = f_gravity_pivot_avoid_indices(D, ctx.f_free_lin, ctx.A_pivot.pivot)
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
    avoid_f = f_gravity_pivot_avoid_indices(D, ctx.f_free_lin, ctx.A_pivot.pivot)
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
    melitz_outer_state(theta_free, ctx; obj=nothing, evaluate_inner=false,
                        warm_start=nothing, cold=false) -> NamedTuple

Session prompt Section 1.2: the single authoritative displaced-outer-point state builder.
NEVER reads `ctx.benchmark_cutoff` -- every field that could be stale at a displaced point
is recomputed fresh from `theta_free`, every call:

 1. expands `theta_free` into `(A, f, gamma_prime_j)` (`expand_free_theta`, which itself
    derives `f[j,j]` via the autarky-cutoff condition and rebuilds the f-gravity pivot's
    offset every time -- see that function's docstring);
 2. `f[j,j]` (already returned by 1., surfaced here as its own field for convenience);
 3. recomputes the FULL baseline cutoff matrix from the CURRENT `(A, f)`
    (`melitz_baseline_cutoff`), never `ctx.benchmark_cutoff`;
 4. computes the deterministic cutoff feasibility constraints at that fresh cutoff
    (`melitz_deterministic_cutoff_constraints`, Section 1.3);
 5. constructs the current `MelitzPrimitives`/`MelitzEquilibrium`/`MelitzCounterfactual`
    (the `MelitzEquilibrium` here also carries the FRESH cutoff, not a benchmark one);
 6. OPTIONALLY evaluates the inner CC problem (`evaluate_inner=true`) via
    `melitz_recover_lfd` on the CALLER-SUPPLIED `obj::PsiObjectiveBundleDelta` (built once
    by `build_melitz_psi_bundle` and reused across calls for the SAME `ctx`/`z_draws` --
    this function does not construct a throwaway `obj`, so the caller controls warm-start
    continuity across a sequence of outer points explicitly, via `obj.use_cached_x`/
    `obj.x`, rather than this function silently deciding). `warm_start`, if given, is
    written into `obj.x` before solving (and `obj.use_cached_x` is forced `true`);
    `cold=true` clears `obj.use_cached_x` to `false` (and `obj.x` to `NaN`) first, forcing
    KNITRO's zero-vector default start regardless of whatever the bundle's own cache
    holds -- the caller must set at least one of `cold`/`warm_start` deliberately for every
    call whose starting point matters (both default to "whatever the bundle's `x` cache
    currently holds", i.e. an ordinary warm continuation).

`feasible = (min_slack >= 0)` where `min_slack = min(minimum(g_domestic),
minimum(g_export))` -- the single worst deterministic-constraint margin, negative iff
infeasible. This is a CHEAP (no Monte Carlo, no KNITRO) check available even when
`evaluate_inner=false`.
"""
function melitz_outer_state(theta_free::AbstractVector, ctx; obj=nothing,
                             evaluate_inner::Bool=false, warm_start=nothing, cold::Bool=false)
    D, j = ctx.D, ctx.target_country
    A, f, gamma_prime_j, f_jj = @melitz_profile :outer_state_expand melitz_expand_theta(theta_free, ctx)

    cutoff = @melitz_profile :outer_state_cutoff melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    g_domestic, g_export = @melitz_profile :outer_state_constraints melitz_deterministic_cutoff_constraints(cutoff)
    min_slack = min(minimum(g_domestic), minimum(g_export))

    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, j, ctx.tau, ctx.w, A, f, gamma_prime_j)
    equilibrium = MelitzEquilibrium(ctx.expenditure, ones(Float64, D), cutoff, ctx.X_data)
    expenditure_prime = ctx.w_prime * ctx.L[j]
    counterfactual = MelitzCounterfactual(j, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)

    lfd = nothing
    if evaluate_inner
        obj === nothing && throw(ArgumentError(
            "melitz_outer_state: evaluate_inner=true requires a caller-supplied `obj` " *
            "(build_melitz_psi_bundle's PsiObjectiveBundleDelta, sharing this `ctx`)"))
        if cold
            obj.use_cached_x = false
            obj.x .= NaN
        end
        if warm_start !== nothing
            obj.x .= warm_start
            obj.use_cached_x = true
        end
        lfd = melitz_recover_lfd(obj, theta_free)
    end

    return (theta_free=theta_free, A=A, f=f, gamma_prime_j=gamma_prime_j, f_jj=f_jj,
            cutoff=cutoff, g_domestic=g_domestic, g_export=g_export, min_slack=min_slack,
            feasible=(min_slack >= 0), primitives=primitives, equilibrium=equilibrium,
            counterfactual=counterfactual, lfd=lfd)
end

"""
    melitz_cutoff_constraint_jacobian(theta_free, ctx) -> (J_domestic, J_export)

Session prompt Section 1.3: the EXACT Jacobian of `melitz_deterministic_cutoff_constraints`
with respect to `theta_free`, including the A-gravity-pivot chain rule, the f-gravity-pivot
chain rule, the derived `f[j,j]`, and `gamma_prime` -- ALL captured automatically by
differentiating straight through `expand_free_theta` -> `melitz_baseline_cutoff` ->
`melitz_deterministic_cutoff_constraints` with ForwardDiff, rather than hand-deriving each
piece separately. This is legitimate here (unlike `Delta(theta)`'s hard participation
gate, docs' own documented ForwardDiff pitfall) because the cutoff formula is a SMOOTH
closed-form algebraic expression in `(A, f, w, tau, expenditure)` -- no Monte Carlo, no
`profit>0` Boolean gate anywhere in this call chain. `expand_free_theta`/
`melitz_baseline_cutoff`/`melitz_deterministic_cutoff_constraints` are all written
type-generically (`T = eltype(...)`), so Dual-number propagation is exact, not an
approximation -- validated against central finite differences in the test suite
(`test/melitz/runtests.jl`, "Section 1.3").
"""
function melitz_cutoff_constraints_at(theta_free::AbstractVector, ctx)
    A, f, _, _ = melitz_expand_theta(theta_free, ctx)
    zhat = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    return melitz_deterministic_cutoff_constraints(zhat)
end

function melitz_cutoff_constraint_jacobian(theta_free::AbstractVector, ctx)
    g_domestic_fn(tf) = melitz_cutoff_constraints_at(tf, ctx)[1]
    g_export_fn(tf) = melitz_cutoff_constraints_at(tf, ctx)[2]
    J_domestic = ForwardDiff.jacobian(g_domestic_fn, theta_free)
    J_export = ForwardDiff.jacobian(g_export_fn, theta_free)
    return J_domestic, J_export
end

"""
    melitz_moments_adapter!(K, G, theta, U, obj)

Adapter matching the `(K, G, theta, U, obj) -> nothing` contract required by
`PsiObjectiveBundleDelta`. `theta` is the FREE `2D^2-2` vector. `obj.gamma` (`ctx`) holds
everything besides `theta` needed to reconstruct the model: `D, sigma, theta_star,
target_country, tau, w, w_prime, L, expenditure, benchmark_cutoff, moment_layout, X_data,
c_full, A_pivot, jj_lin, f_free_lin`. FULLY re-equilibrates `(A, f, gamma_prime_target)` and
the autarky counterfactual from `theta` on every call (general, not fixed-theta*-only --
contrast the superseded closure's adapter).

Session prompt Section 1.2: the `MelitzEquilibrium` built here uses a FRESH cutoff
(`melitz_baseline_cutoff(A, f, ...)`, recomputed from THIS call's own `(A,f)`), never
`ctx.benchmark_cutoff` -- `melitz_moments!` itself does not currently read `eq.cutoff`
(only `eq.expenditure`), but constructing `eq` with a stale benchmark-only cutoff at a
displaced `theta` would be a landmine for any future/downstream code that reads `eq.cutoff`
expecting it to be real (e.g. a feasibility screen or diagnostic). `ctx.benchmark_cutoff`
itself (the field name change from the former `ctx.cutoff`) is retained ONLY for reporting
comparisons against the fixture's own benchmark point -- never read here.
"""
function melitz_moments_adapter!(K, G, theta, U, obj)
    ctx = obj.γ
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    D = ctx.D

    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country,
                                   ctx.tau, ctx.w, A, f, gamma_prime_j)
    cutoff = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    eq = MelitzEquilibrium(ctx.expenditure, ones(Float64, D), cutoff, ctx.X_data)
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

`needs_outer_moment_jacobian` (default `true`, memory-scalability continuation session):
threads `PsiObjectiveBundleDelta`'s own escape hatch (`cc_algo/PsiObjectiveBundle.jl`,
mirroring `PsiObjectiveBundleImplicit`'s pre-existing one) through to this constructor.
The default preserves the exact prior unconditional dense-`jac_h` allocation
(`N*(d+2)*l` -- ~206GB at D=20/W=80,000, quartic in `D`). Pass `false` for any caller that
only performs FIXED-theta inner CC dual solves (`inner_loop`/`melitz_recover_lfd`/
`run_melitz_inner_delta`) -- the call-graph audit backing this default confirms `jac_h` is
never read on this path (the theta-gradient branch of the functor, the only branch that
touches it, is never exercised by `inner_loop_KNITRO`'s callback, which always calls
`obj(x, g)` with an empty `θ`). The real Melitz outer theta-gradient search uses a
SEPARATE `PsiObjectiveBundleImplicit` (`build_melitz_implicit_bundle`), unaffected by this
kwarg.
"""
function build_melitz_psi_bundle(data::MelitzSyntheticData;
                                  X_data::Matrix{Float64}=data.equilibrium.trade_flow,
                                  outer_parameterization::Symbol=:logf,
                                  inner_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_inner_loop_options.opt"),
                                  outer_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_outer_loop_options.opt"),
                                  needs_outer_moment_jacobian::Bool=true)
    outer_parameterization in (:logf, :logcutoff) || throw(ArgumentError(
        "outer_parameterization must be :logf or :logcutoff, got $outer_parameterization"))
    p, eq, cf = data.primitives, data.equilibrium, data.counterfactual
    D = p.D
    j = p.target_country
    z_draws = data.z_draws

    moment_layout = MelitzMomentLayout(D)
    c_full, A_pivot = build_gravity_pivots(p.tau, j)
    outer_layout = melitz_outer_layout(D, j)

    ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau, w=p.w,
           w_prime=cf.w_prime, L=data.L, expenditure=eq.expenditure,
           benchmark_cutoff=eq.cutoff,  # reporting ONLY (Section 1.2) -- never read for
                                        # candidate feasibility; melitz_outer_state/
                                        # melitz_moments_adapter! always recompute fresh
           moment_layout=moment_layout, X_data=X_data, c_full=c_full, A_pivot=A_pivot,
           jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin,
           outer_parameterization=outer_parameterization,
           inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt)

    theta_free = melitz_reduce_theta(p, ctx)

    obj = PsiObjectiveBundleDelta(
        γ=ctx,
        (moments!)=melitz_moments_adapter!,
        d=moment_layout.num_moments,
        l=length(theta_free),
        inequality_index=Int64[],
        U=z_draws,
        inner_loop_opt=inner_loop_opt,
        outer_loop_opt=outer_loop_opt,
        needs_outer_moment_jacobian=needs_outer_moment_jacobian,
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
    melitz_recover_lfd_from_solution(val, x, nStatus, theta, obj; kkt_opt_error=NaN,
        kkt_feas_error=NaN, moment_tol=1e-6, normalization_tol=1e-6, gap_atol=1e-10,
        gap_rtol=1e-6) -> MelitzLFDResult

The post-processing half of `melitz_recover_lfd` (which is now a thin wrapper: run
`inner_loop`, then call this) -- factored out (2026-07-23 correctness-repair session, main
prompt Section 2.2) so a caller that has ALREADY obtained a verified inner dual solution
`(val, x, nStatus)` at `theta` -- e.g. the finite-delta outer NLP's own combined callback,
which calls `inner_loop_internal` itself every evaluation -- can recover the LFD/moment-
residual/primal-dual diagnostics WITHOUT launching a SECOND, redundant real KNITRO solve
at the same point.

`val` must already be in the SAME sign/scale convention `inner_loop` returns
(`Delta(theta)`, positive) -- a caller holding a `PsiObjectiveBundleImplicit`'s own
`objSol` (which is `K`-based -- `theta[1]`, NOT `Delta` -- see
`docs/melitz_delta_star.md` Section 3.2) must NOT pass that directly; it must compute
`Delta(theta)` separately (e.g. via the raw functor's `constr[1]/1e10`, Section 18's sign
convention) and pass THAT as `val`.

RECOVERS THE PRIMAL LFD -- not just the dual objective/status (the superseded closure's
`run_melitz_inner_delta` returned only `(val, x, nStatus)`).

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
function melitz_recover_lfd_from_solution(val::Real, x::AbstractVector, nStatus::Integer,
                                           theta::AbstractVector, obj;
                                           kkt_opt_error::Real=NaN, kkt_feas_error::Real=NaN,
                                           moment_tol::Real=1e-6, normalization_tol::Real=1e-6,
                                           gap_atol::Real=1e-10, gap_rtol::Real=1e-6,
                                           G_precomputed::Union{Nothing,AbstractMatrix}=nothing)
    d = obj.d
    W = size(obj.U, 1)

    if nStatus != 0 || !all(isfinite, x)
        return MelitzLFDResult(val, x, nStatus, fill(1.0 / W, W), false,
                                fill(NaN, d), NaN, NaN, NaN, NaN,
                                NaN, val, NaN, NaN, NaN, kkt_opt_error, kkt_feas_error)
    end

    # 2026-07-23 continuation session, Section 4: `G_precomputed`, when supplied, MUST be
    # the moment matrix at this EXACT `theta` -- the caller's responsibility to guarantee
    # (e.g. `finite_delta_outer.jl`'s `register_live_candidate!` passes a view onto
    # `obj.H`'s own G columns, populated by the inner solve that JUST ran at this same
    # `theta`, with nothing mutating it in between). This skips a second, otherwise
    # identical O(W*(D^2+1)) `obj.moments!` build -- the single largest component of the
    # "second authoritative evaluation" cost `docs/melitz_optimization_report_2026-07-23.md`
    # Section A.3/G identified in `fc_candidate_registration` (~65-75ms/FC call). Default
    # (`nothing`) preserves the exact prior behavior (always rebuild G fresh) for every
    # OTHER caller (tests, gradient-lab diagnostics, `evaluate_melitz_delta`'s own ordinary
    # fresh-solve path) that cannot make this same freshness guarantee.
    local G
    if G_precomputed === nothing
        K = zeros(W)
        G = zeros(W, d)
        obj.moments!(K, G, theta, obj.U, obj)
    else
        size(G_precomputed) == (W, d) || throw(DimensionMismatch(
            "melitz_recover_lfd_from_solution: G_precomputed has size $(size(G_precomputed)), " *
            "expected ($W, $d)"))
        G = G_precomputed
    end

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
    melitz_recover_lfd(obj::PsiObjectiveBundleDelta, theta) -> MelitzLFDResult

Runs `inner_loop(obj, theta)` (real KNITRO, `cc_algo/inner_loop_functions.jl`, unmodified)
then delegates to `melitz_recover_lfd_from_solution` (see its docstring for the full
recovery recipe and the `lfd_ok` acceptance rule) -- the ordinary "solve fresh, then
recover" entry point. Use `melitz_recover_lfd_from_solution` directly instead when a
verified `(val, x, nStatus)` at `theta` is already in hand (no redundant KNITRO solve).
"""
function melitz_recover_lfd(obj, theta::AbstractVector; moment_tol::Real=1e-6,
                             normalization_tol::Real=1e-6,
                             gap_atol::Real=1e-10, gap_rtol::Real=1e-6)
    val, x, nStatus = inner_loop(obj, theta)
    kkt_opt_error = INNER_LAST_OPT_ERR[]
    kkt_feas_error = INNER_LAST_FEAS_ERR[]
    return melitz_recover_lfd_from_solution(val, x, nStatus, theta, obj;
        kkt_opt_error=kkt_opt_error, kkt_feas_error=kkt_feas_error,
        moment_tol=moment_tol, normalization_tol=normalization_tol,
        gap_atol=gap_atol, gap_rtol=gap_rtol)
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

# ============================================================================
# Session prompt Section 2: the authoritative fixed-outer-point evaluator + cache.
# ============================================================================

"""
    MelitzDeltaEvalCache(max_size=4)

Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
D") Section 4: THE "heavy-state cache" tier -- each entry is a full `MelitzDeltaEvalResult`,
which carries the `W x num_moments` moment matrix `G` (`nothing` only if a caller passed
`store_G=false`) plus the full `MelitzEquilibriumCheck` diagnostics. At D=20/W=80,000 a
SINGLE entry's `G` alone is `80_000 * 401 * 8` bytes ~ 257MB -- retaining this UNBOUNDED
(this cache's behavior before this session) risks unbounded growth across a long outer
trajectory that keeps calling `evaluate_melitz_delta(...; cache=...)` at new points. Bounded
here to a SMALL default capacity (`4`, matching the main prompt's own "very small... such as
1-4 entries" spec for this tier) via the shared `MelitzLRUOrder` machinery
(`bounded_cache.jl`) -- a cache HIT still requires the SAME `theta_free` key (exact value
equality, unchanged); once evicted, a repeat query is an ordinary cache MISS (a real
`evaluate_melitz_delta` recomputation), not an error. `max_size<=0` disables bounding
entirely (`melitz_lru_evict_until!`'s own escape hatch) -- every production default remains
a small positive integer.

Keyed by `theta_free -> MelitzDeltaEvalResult` with EXACT value equality on the free-
coordinate vector (no rounding/binning -- the gradient laboratory and outer solve both
re-query EXACT points, e.g. `theta`/`theta+h*v`/`theta-h*v` triples, so exact-key lookups
are the useful case; a fuzzy/nearest-point cache is a different, unimplemented data
structure). `evaluate_melitz_delta`'s own gate (`result.verified`) decides what may be
STORED here -- the cache itself never re-checks or overrides that gate.
"""
mutable struct MelitzDeltaEvalCache
    store::Dict{Vector{Float64},MelitzDeltaEvalResult}
    order::MelitzLRUOrder
    max_size::Int
    hits::Int
    misses::Int
    evictions::Int
end
MelitzDeltaEvalCache(max_size::Int=4) = MelitzDeltaEvalCache(
    Dict{Vector{Float64},MelitzDeltaEvalResult}(), MelitzLRUOrder(), max_size, 0, 0, 0)

"""
    evaluate_melitz_delta(theta_free, ctx, obj; warm_start=nothing, cold=false,
                           cache=nothing, store_G=true) -> MelitzDeltaEvalResult

Session prompt Section 2: THE authoritative fixed-outer-point evaluator. Wraps
`melitz_outer_state` (cutoff-safe state, Section 1.2, timed separately as `state_time`)
and `melitz_recover_lfd` (the real inner CC KNITRO solve, timed as `inner_time`) into one
IMMUTABLE `MelitzDeltaEvalResult`, and, when `lfd.nStatus==0`, the full ex-post
equilibrium check (`check_profiled_melitz_equilibrium`) under the RECOVERED LFD weights
(never reference/equal weights).

`verified = state.feasible && lfd.lfd_ok && lfd.nStatus==0` -- the three-way gate. Main
prompt Section 2's caching rule is enforced HERE, not left to the caller: a result is
written into `cache.store` if and only if `result.verified` -- a failed numerical solve
(`nStatus != 0`), an approximate/unverified LFD (`lfd_ok=false`), or a cutoff-infeasible
point (`feasible=false`) is returned to the CALLER (so the caller sees exactly what
happened) but never cached, and never overwrites an existing cached (necessarily
verified) entry for the same `theta_free` with something worse.

`warm_start`/`cold` are forwarded to `melitz_outer_state` (`obj.x`/`obj.use_cached_x`
semantics -- see that function's docstring); the default (both absent) is an ordinary warm
continuation from whatever `obj`'s cache currently holds. This D=4 development pass is
SERIAL by design (main prompt Section 2's own instruction) -- no parallel callback
evaluation is added here; `obj` (and hence its mutable KNITRO-facing cache/scratch) must
not be shared across concurrent callers.
"""
function evaluate_melitz_delta(theta_free::AbstractVector, ctx, obj;
                                warm_start=nothing, cold::Bool=false,
                                cache::Union{Nothing,MelitzDeltaEvalCache}=nothing,
                                store_G::Bool=true)
    if cache !== nothing
        key = Vector{Float64}(theta_free)
        hit = get(cache.store, key, nothing)
        if hit !== nothing
            cache.hits += 1
            melitz_lru_touch!(cache.order, key)  # move-to-MRU on a hit, not just on insert
            MELITZ_PROFILE[] && melitz_record!(:eval_cache_hit, Int64(0))
            return hit
        end
        cache.misses += 1
        MELITZ_PROFILE[] && melitz_record!(:eval_cache_miss, Int64(0))
    end

    t0 = time()
    state = melitz_outer_state(theta_free, ctx)  # cheap: no inner solve (Section 1.2)
    state_time = time() - t0
    melitz_record_seconds!(:outer_state_total, state_time)

    t1 = time()
    if cold
        obj.use_cached_x = false
        obj.x .= NaN
    end
    if warm_start !== nothing
        obj.x .= warm_start
        obj.use_cached_x = true
    end
    lfd = melitz_recover_lfd(obj, theta_free)
    inner_time = time() - t1
    melitz_record_seconds!(cold ? :inner_solve_cold : :inner_solve_warm, inner_time)
    melitz_record_seconds!(:inner_solve_total, inner_time)

    G_out = nothing
    if store_G
        W = size(obj.U, 1)
        K = zeros(W)
        G = zeros(W, obj.d)
        obj.moments!(K, G, theta_free, obj.U, obj)
        G_out = G
    end

    verified = state.feasible && lfd.lfd_ok && lfd.nStatus == 0
    MELITZ_PROFILE[] && melitz_record!(verified ? :eval_verified : :eval_unverified, Int64(0))
    check = verified ? check_profiled_melitz_equilibrium(
        state.primitives, state.equilibrium, state.counterfactual, obj.U, lfd.weights) : nothing

    result = MelitzDeltaEvalResult(
        Vector{Float64}(theta_free), state.A, state.f, state.gamma_prime_j, state.f_jj,
        state.cutoff, state.g_domestic, state.g_export, state.min_slack, state.feasible,
        G_out, lfd.dual_x, lfd.weights, lfd.Delta, lfd.primal_divergence, lfd.dual_divergence,
        lfd.primal_dual_gap, lfd.moment_residuals, lfd.kkt_opt_error, lfd.kkt_feas_error,
        lfd.nStatus, lfd.lfd_ok, verified, check, state_time, inner_time, state_time + inner_time)

    if cache !== nothing && result.verified
        key = Vector{Float64}(theta_free)
        cache.store[key] = result
        melitz_lru_touch!(cache.order, key)
        cache.evictions += melitz_lru_evict_until!(cache.order, cache.store, cache.max_size)
    end

    return result
end

"""
    evaluate_melitz_delta_from_solution(theta_free, ctx, obj_like, Delta_val, x, nStatus;
        kkt_opt_error=NaN, kkt_feas_error=NaN, store_G=false) -> MelitzDeltaEvalResult

2026-07-23 correctness-repair session (main prompt Section 2.2): builds the SAME
authoritative `MelitzDeltaEvalResult` `evaluate_melitz_delta` does, but from a dual
solution `(Delta_val, x, nStatus)` the CALLER has already obtained at this exact
`theta_free` -- e.g. the finite-delta outer NLP's combined callback, which calls
`inner_loop_internal` on its own `PsiObjectiveBundleImplicit` every evaluation anyway.
Launches NO additional real KNITRO solve (`melitz_outer_state`'s cutoff/gravity state and
`melitz_recover_lfd_from_solution`'s LFD reconstruction are both cheap closed-form/O(W)
computations) -- this is what lets the outer callback classify every live evaluation as a
candidate incumbent (Section 2.2 step 1-3) without doubling the trajectory's own KNITRO
solve count.

`obj_like` need only expose `.moments!`/`.U`/`.d` (both `PsiObjectiveBundleDelta` and
`PsiObjectiveBundleImplicit` qualify -- the divergence-relevant moment matrix `G` is
identical between them at a shared `ctx`, see `docs/melitz_delta_star.md` Section 3's file
header). `Delta_val` must be `Delta(theta)` itself (positive), NOT a `PsiObjectiveBundleImplicit`'s
own `K`-based `objSol` -- see `melitz_recover_lfd_from_solution`'s own docstring.

`state_time`/`inner_time` are set to `0.0` (no separate real-time measurement is meaningful
here, since neither computation involves a fresh KNITRO solve) rather than a misleading
wall-clock split; `total_time` likewise `0.0`.

`G_precomputed` (2026-07-23 continuation session, Section 4): forwarded to
`melitz_recover_lfd_from_solution` -- see that function's docstring. Passing the caller's
own already-computed G (guaranteed fresh at this exact `theta_free`) avoids a second
`obj_like.moments!` build; `nothing` (default) preserves the original always-rebuild
behavior.
"""
function evaluate_melitz_delta_from_solution(theta_free::AbstractVector, ctx, obj_like,
                                              Delta_val::Real, x::AbstractVector, nStatus::Integer;
                                              kkt_opt_error::Real=NaN, kkt_feas_error::Real=NaN,
                                              store_G::Bool=false,
                                              G_precomputed::Union{Nothing,AbstractMatrix}=nothing)
    state = melitz_outer_state(theta_free, ctx)
    lfd = melitz_recover_lfd_from_solution(Float64(Delta_val), x, nStatus, theta_free, obj_like;
        kkt_opt_error=kkt_opt_error, kkt_feas_error=kkt_feas_error, G_precomputed=G_precomputed)

    G_out = nothing
    if store_G
        if G_precomputed !== nothing
            G_out = Matrix{Float64}(G_precomputed)
        else
            W = size(obj_like.U, 1)
            Ktmp = zeros(W)
            Gtmp = zeros(W, obj_like.d)
            obj_like.moments!(Ktmp, Gtmp, theta_free, obj_like.U, obj_like)
            G_out = Gtmp
        end
    end

    verified = state.feasible && lfd.lfd_ok && lfd.nStatus == 0
    check = verified ? check_profiled_melitz_equilibrium(
        state.primitives, state.equilibrium, state.counterfactual, obj_like.U, lfd.weights) : nothing

    return MelitzDeltaEvalResult(
        Vector{Float64}(theta_free), state.A, state.f, state.gamma_prime_j, state.f_jj,
        state.cutoff, state.g_domestic, state.g_export, state.min_slack, state.feasible,
        G_out, lfd.dual_x, lfd.weights, lfd.Delta, lfd.primal_divergence, lfd.dual_divergence,
        lfd.primal_dual_gap, lfd.moment_residuals, lfd.kkt_opt_error, lfd.kkt_feas_error,
        lfd.nStatus, lfd.lfd_ok, verified, check, 0.0, 0.0, 0.0)
end
