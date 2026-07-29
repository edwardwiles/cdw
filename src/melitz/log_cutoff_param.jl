# Session prompt Section 5: an experimental parallel outer-coordinate backend,
# `outer_parameterization = :logcutoff`, searching directly over baseline log-cutoffs
# `q_od = log(zhat_od)` instead of `log f_od`. Does NOT replace `:logf`
# (`delta_star.jl`'s `expand_free_theta`/`reduce_to_free_theta`) -- both represent the
# SAME economic feasible set and have the SAME free dimension `2D^2-2`, cross-validated by
# exact fixed-point round-trip tests (Section 5.5).
#
# Economic vector (length 2D^2, Section 5.2): (g, vec(log A) [D^2], q_free [D^2-1, excludes
# (j,j)]) -- structurally identical in SHAPE to the `:logf` economic vector
# (`melitz_outer_layout`), only the second D^2-1-length block's ECONOMIC MEANING differs
# (q instead of log f). Two eliminations give the free `2D^2-2` vector:
#
#   1. A-gravity (Section 5.3): UNCHANGED from `:logf` -- `log A` is linear in `A_free`
#      via the SAME `ctx.A_pivot` (reused, not rebuilt, so both parameterizations agree on
#      which physical A-cell is eliminated -- a deliberate comparability choice, not
#      required by the math).
#   2. q-gravity (Section 5.3, NEW): after substituting `log f_od(q_od, a_od)` (Section
#      5.1's affine formula) into the f-gravity restriction `dot(c_full, vec(log f)) = 0`,
#      the restriction becomes affine in `(a, q)` jointly -- derived analytically below
#      (`build_q_gravity_pivot`/`build_q_gravity_offset`). NOTE: Section 3.1's own
#      basis-probe-vs-analytical two-independent-methods pattern was NOT separately
#      replicated here for this offset formula alone -- instead it is validated a
#      DIFFERENT way (arguably stronger): the full `(a,q) -> (A,f,gamma)` round trip is
#      checked end to end against the ALREADY-validated `:logf` path AND against the
#      TRUE ground-truth `gravity_residuals` (not a second derivation of the same
#      formula) at both random points and the exact fixture (Section 5.5's own tests).
#      A dedicated basis-probe cross-check of `build_q_gravity_offset` in isolation,
#      mirroring `affine_cutoff.jl` exactly, was not built -- flagged as a cheap
#      follow-up, not a gap in what was actually verified.
#
# The focal domestic cell `(j,j)` is NEVER a free q coordinate (Section 5.1): `q[j,j]` is
# DERIVED from `(g, fixed primitives)` alone -- algebraically independent of `A[j,j]`
# (verified below, `derive_qjj_from_autarky_cutoff`'s own docstring) -- then `f[j,j]` is
# recovered from the SAME general log-f formula, required to reproduce
# `derive_fjj_from_autarky_cutoff` exactly (Section 5.1's own instruction).

using LinearAlgebra: dot

"""
    melitz_log_f_from_q(q_od, a_od, w_o, tau_od, expenditure_d, sigma) -> log_f_od

Section 5.1: the exact inverse of the baseline log-cutoff formula
(`affine_cutoff.jl`'s own `q_od = log(markup) + log(w_o) + log(tau_od) - a_od +
(log(sigma)+log(w_o)+log(f_od)-log(expenditure_d))/(sigma-1)`, itself derived directly from
`melitz_C`/`melitz_cutoff`), solved for `log(f_od)`:

    log(f_od) = (sigma-1) * (q_od + a_od - log(markup) - log(w_o) - log(tau_od))
                + log(expenditure_d) - log(sigma) - log(w_o)

Algebraically identical to the session prompt's own Section 5.1 formula with
`markup = sigma/(sigma-1)`.
"""
function melitz_log_f_from_q(q_od::Real, a_od::Real, w_o::Real, tau_od::Real,
                              expenditure_d::Real, sigma::Real)
    markup = melitz_markup(sigma)
    return (sigma - 1) * (q_od + a_od - log(markup) - log(w_o) - log(tau_od)) +
           log(expenditure_d) - log(sigma) - log(w_o)
end

"""
    derive_qjj_from_autarky_cutoff(gamma_prime_j, ctx) -> q_jj

Session prompt Section 5.1: the focal domestic cell's BASELINE log-cutoff `q[j,j] =
log(zhat[j,j])`, derived from `(g=log(gamma_prime_j), fixed primitives)` alone -- NEVER
searched independently, and algebraically INDEPENDENT of `A[j,j]` (the `-a_jj`/`+a_jj`
terms cancel exactly once `log(f_jj)`'s own affine dependence on `a_jj`, via
`derive_fjj_from_autarky_cutoff`, is substituted into the general `q_od` formula --
verified symbolically in this docstring's derivation and numerically in the test suite,
"Section 5: q[j,j] independent of A[j,j]").

Derivation: substitute `log(f_jj) = const_fjj + (sigma-1)*log(A_jj) - g` (the SAME affine
identity `affine_cutoff_map_analytical` uses, Section 3.1's `const_fjj`/`coef_fjj`) into
the general `q_od` formula at `(o,d)=(j,j)`, `tau_jj=1`:

    q_jj = log(markup) + log(w_j) - log(A_jj)
           + (log(sigma) + log(w_j) + log(f_jj) - log(expenditure_j)) / (sigma-1)
         = log(markup) + log(w_j) - log(A_jj)
           + (log(sigma) + log(w_j) + const_fjj - log(expenditure_j)) / (sigma-1)
           + log(A_jj) - g/(sigma-1)
         = [log(markup) + log(w_j) + (log(sigma)+log(w_j)+const_fjj-log(expenditure_j))/(sigma-1)]
           - g/(sigma-1)

the `-log(A_jj)`/`+log(A_jj)` terms cancel EXACTLY -- `q_jj` depends only on `g` and fixed
primitives, matching the session prompt's own claimed form `q[j,j] = qjj_constant -
g/(sigma-1)` (here derived from the FULL fixed-primitive formula, not assumed).
"""
function derive_qjj_from_autarky_cutoff(gamma_prime_j::Real, ctx)
    j = ctx.target_country
    sigma = ctx.sigma
    markup = melitz_markup(sigma)
    w_prime_j = ctx.w_prime
    expenditure_prime_j = ctx.w_prime * ctx.L[j]
    const_fjj = log(expenditure_prime_j) + (1 - sigma) * (log(markup) + log(w_prime_j)) -
                log(sigma) - log(w_prime_j)
    g = log(gamma_prime_j)
    qjj_constant = log(markup) + log(ctx.w[j]) +
                   (log(sigma) + log(ctx.w[j]) + const_fjj - log(ctx.expenditure[j])) / (sigma - 1)
    return qjj_constant - g / (sigma - 1)
end

"""
    build_q_gravity_pivot(ctx) -> GravityPivot

Session prompt Section 5.3: the q-gravity pivot, analogous to `equilibrium.jl`'s f-pivot
but for the SUBSTITUTED (via `melitz_log_f_from_q`) f-gravity restriction expressed in
`(a,q)`-space. Uses the SAME domestic-avoid + A-pivot-avoid restriction as the `:logf`
f-pivot (`f_gravity_pivot_avoid_indices`) -- q lives on the identical `f_free_lin` domain
(every cell except `(j,j)`).

The pivot's own `c` vector is `ctx.c_full[ctx.f_free_lin]` (UNCHANGED: substituting
`log_f(q,a)` scales the restriction's overall magnitude by `(sigma-1)` -- verified in
`build_q_gravity_offset`'s own derivation -- but does not change which linear combination
of `q_free` it constrains, so the SAME coefficient vector/pivot-selection logic applies).
The `g0` offset is set to `0.0` here (placeholder) -- it MUST be rebuilt fresh at every
evaluation via `build_q_gravity_offset` (below). 2026-07-29: the offset depends on `q_jj`
(hence `g`) and fixed data ONLY, never on `A` (see `build_q_gravity_offset`'s own docstring)
-- it is still rebuilt per call rather than cached, since it is cheap (`O(D^2)`, dominated by
`melitz_expand_theta`'s other work) and depends on `g`, which does vary across coordinate
probes.
"""
function build_q_gravity_pivot(ctx)
    D = ctx.D
    avoid_q = f_gravity_pivot_avoid_indices(D, ctx.f_free_lin, ctx.A_pivot.pivot)
    return build_gravity_pivot(ctx.c_full[ctx.f_free_lin], 0.0; avoid=avoid_q)
end

"""
    build_q_gravity_offset(q_jj::Real, ctx) -> g0_q

Session prompt Section 5.3: the q-gravity pivot's affine offset at the focal-cell `q[j,j]`
(2026-07-29: no longer takes `A` -- see this function's own docstring continuation below for
why the "current A" dependence the original derivation anticipated is provably always zero).
Derivation: the f-gravity restriction `dot(c_full, vec(log f)) = 0`
(summed over ALL `D^2` cells, INCLUDING `(j,j)`), substituting `log(f_od) = (sigma-1)*(q_od
+ a_od - log(markup) - log(w_o) - log(tau_od)) + log(expenditure_d) - log(sigma) -
log(w_o)` (`melitz_log_f_from_q`) at every cell, is

    dot(c_full, vec(log f)) = (sigma-1)*dot(c_full, vec(q)) + (sigma-1)*dot(c_full, vec(a))
                               + dot(c_full, const_vec) = 0

where `const_vec[od] = (sigma-1)*(-log(markup)-log(w_o)-log(tau_od)) + log(expenditure_d) -
log(sigma) - log(w_o)` is a FIXED (data-only) vector, and `vec(q)` spans ALL `D^2` cells
(the `D^2-1` free ones AND `q[j,j]`). Dividing by `(sigma-1)` (nonzero, `sigma>1`) and
splitting off the `(j,j)` term (`q[j,j]` is NOT a free coordinate of the q-pivot, whose
domain is only `f_free_lin`):

    dot(c_full[f_free_lin], q_free_full) + c_full[jj_lin]*q_jj + dot(c_full, vec(a))
        + dot(c_full, const_vec)/(sigma-1) = 0

i.e. in `GravityPivot`'s own `dot(c,z)+g0=0` convention (`z=q_free_full`, restricted to
`f_free_lin`), `g0 = c_full[jj_lin]*q_jj + dot(c_full, vec(a)) + dot(c_full,
const_vec)/(sigma-1)` -- an AFFINE function of the CURRENT full `log A` AND `q[j,j]`
(hence, through `q[j,j] = derive_qjj_from_autarky_cutoff(gamma_prime_j,ctx)`, of `g` too).

BUG FIXED (found live via the Section 5.5 cross-parameterization round-trip test, which
failed with `gravity_residual_f ~ 0.0065`, not machine precision): an earlier version of
this function omitted the `c_full[jj_lin]*q_jj` term entirely, silently assuming the
`(j,j)` cell's contribution to the FULL-vector restriction was zero -- it is not, since
`c_full[jj_lin]` is generically nonzero (Gate A5: `|c|` is systematically LARGEST on
diagonal cells under `withinTransform`) and `q[j,j]` is a genuine nonzero baseline
log-cutoff.

2026-07-29 continuation (A_q separation and gradient diagnostics session, Phase 2):
**the `s_a = dot(c_full, vec(log A))` term is dropped entirely -- it is not merely small,
it is EXACTLY ZERO for every admissible `A`, by construction of the A-gravity pivot, not
merely at a calibrated/gravity-satisfying point.** `A` reaching this function always comes
from `pivot_expand(A_free, ctx.A_pivot)` (`expand_free_theta_logcutoff`/`!`), and
`pivot_expand`'s own pivot-cell formula (`logA_full[pivot] = -sum_k
c[other[k]]*A_free[k]/c[pivot]`) makes `dot(c_full, vec(logA_full)) ==
c_full[pivot]*logA_full[pivot] + sum_k c_full[other[k]]*A_free[k] == 0` an ALGEBRAIC
IDENTITY in `A_free`, not a restriction that merely happens to hold at the model's
calibrated point -- true for ANY `A_free`, including deliberately gravity-violating test
inputs. Live-verified this session (`scripts/melitz_aq_phase1_separation_audit_2026-07-29.jl`):
`dot(c_full, vec(logA))` measures `~1e-18` to `~1e-16` (pure floating-point roundoff, not a
"small but real" economic term) under every tested A-perturbation. The PRIOR code computed
this term explicitly every call and added it to `g0_q` -- since it is provably zero, this
was a genuine (if tiny, `~5e-16`) coupling channel through which perturbing `A_free` moved
the q-pivot cell's reconstructed value: NOT strict block separation as literally written,
even though the coupling was numerically negligible for every practical purpose. Dropping
the term makes the q-pivot's offset an EXACT function of `q_jj` (hence of `g` alone) and
fixed data -- q is now algebraically, not merely numerically, independent of every free A
coordinate (verified: perturbing any A_free now leaves every q value BIT-IDENTICAL, not
merely `~1e-16`-close). This also removes the O(D^2) loop over `A`/`log.(A[lin])` this
function used to run every call -- a genuine (small) allocation-free performance win on top
of the correctness improvement, since `A` is no longer read by this function at all.
"""
function build_q_gravity_offset(q_jj::Real, ctx)
    D = ctx.D
    sigma = ctx.sigma
    markup = melitz_markup(sigma)
    c_full = ctx.c_full
    s_const = 0.0
    @inbounds for lin in 1:D^2
        o, d = lin2od(lin, D)
        const_lin = (sigma - 1) * (-log(markup) - log(ctx.w[o]) - log(ctx.tau[o, d])) +
                    log(ctx.expenditure[d]) - log(sigma) - log(ctx.w[o])
        s_const += c_full[lin] * const_lin
    end
    return c_full[ctx.jj_lin] * q_jj + s_const / (sigma - 1)
end

"""
    melitz_free_dim_logcutoff(ctx) -> n

Same as `melitz_free_dim(ctx)` (`affine_cutoff.jl`) -- `2D^2-2` -- the `:logcutoff`
parameterization has the IDENTICAL free dimension to `:logf` (Section 5.2). Kept as a
separate name for documentation clarity at call sites.
"""
melitz_free_dim_logcutoff(ctx) = melitz_free_dim(ctx)

"""
    expand_free_theta_logcutoff(theta_free_q, ctx) -> (A, f, gamma_prime_j, f_jj, q)

The `:logcutoff` analogue of `delta_star.jl`'s `expand_free_theta`: `theta_free_q =
vcat(log(gamma_prime_j), A_free [D^2-1, SAME A-pivot], q_free_free [D^2-2, q-pivoted])`.

Order matters, exactly as in `expand_free_theta`: `A` first (needed for the q-pivot's own
`g0`), then `q[j,j]` (`derive_qjj_from_autarky_cutoff`, independent of `A`, Section 5.1),
then the q-pivot expansion (needs both `A` for its offset AND `q[j,j]` is NOT part of this
pivot's domain -- domain is `f_free_lin`, excluding `(j,j)`), then `f` (every cell,
`melitz_log_f_from_q`, including `f[j,j]` recovered from `q[j,j]`), matching the session
prompt's own instruction to verify `f[j,j]` this way agrees with
`derive_fjj_from_autarky_cutoff`.

Returns the full `(A, f, gamma_prime_j, f_jj)` (SAME shape as `expand_free_theta`, so any
caller needing only the economic primitives can use either backend interchangeably) plus
`q` (the full `D x D` baseline log-cutoff matrix, this parameterization's own "native"
object) for callers that want it directly without recomputing `melitz_baseline_cutoff`.
"""
function expand_free_theta_logcutoff(theta_free_q::AbstractVector{T}, ctx) where {T}
    D, j = ctx.D, ctx.target_country
    nA = D^2 - 1
    log_gamma_prime_j = theta_free_q[1]
    gamma_prime_j = exp(log_gamma_prime_j)
    A_free = @view theta_free_q[2:1+nA]
    q_free_free = @view theta_free_q[2+nA:end]

    logA_full = pivot_expand(A_free, ctx.A_pivot)
    A = exp.(reshape(logA_full, D, D))

    q_jj = derive_qjj_from_autarky_cutoff(gamma_prime_j, ctx)

    q_pivot = build_q_gravity_pivot(ctx)
    g0_q = build_q_gravity_offset(q_jj, ctx)
    q_pivot_g0 = GravityPivot(q_pivot.n, q_pivot.pivot, q_pivot.other, q_pivot.c, g0_q)
    q_free_full = pivot_expand(q_free_free, q_pivot_g0)   # length D^2-1, over f_free_lin domain

    q = zeros(T, D, D)
    q[j, j] = q_jj
    f = zeros(T, D, D)
    sigma = ctx.sigma
    @inbounds for (k, i) in enumerate(ctx.f_free_lin)
        o, d = lin2od(i, D)
        q[o, d] = q_free_full[k]
        f[o, d] = exp(melitz_log_f_from_q(q_free_full[k], log(A[o, d]), ctx.w[o], ctx.tau[o, d],
                                           ctx.expenditure[d], sigma))
    end
    f_jj = derive_fjj_from_autarky_cutoff(gamma_prime_j, ctx.w_prime, 1.0, A[j, j],
                                           ctx.w_prime * ctx.L[j], sigma)
    f[j, j] = f_jj

    return A, f, gamma_prime_j, f_jj, q
end

"""
    reduce_to_free_theta_logcutoff(A, f, gamma_prime_j, ctx) -> theta_free_q

Inverse of `expand_free_theta_logcutoff` (Section 5.5's round-trip equivalence
requirement): given a gravity-feasible `(A, f, gamma_prime_j)` (e.g. from the EXISTING
`:logf` parameterization's own `expand_free_theta`), computes the full baseline log-cutoff
`q = log.(melitz_baseline_cutoff(A,f,...))`, then reduces `(A_free, q_free)` through the
SAME two pivots `expand_free_theta_logcutoff` uses.
"""
function reduce_to_free_theta_logcutoff(A::AbstractMatrix, f::AbstractMatrix,
                                         gamma_prime_j::Real, ctx)
    D = ctx.D
    logA_full = vec(log.(A))
    A_free = pivot_reduce(logA_full, ctx.A_pivot)

    zhat = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    q_full = log.(zhat)
    q_pivot = build_q_gravity_pivot(ctx)
    q_domain_full = [q_full[lin2od(i, D)...] for i in ctx.f_free_lin]
    q_free = pivot_reduce(q_domain_full, q_pivot)

    return vcat(log(gamma_prime_j), A_free, q_free)
end

# ============================================================================
# Section 5 (live wiring): single dispatch point letting every downstream consumer of
# `expand_free_theta`/`reduce_to_free_theta` (moments!, cutoff constraints and their exact
# Jacobian, gradient laboratory Methods A-D, the affine cutoff system's basis-probe
# construction) work UNCHANGED under EITHER parameterization, keyed off
# `ctx.outer_parameterization` (`:logf`, the default, or `:logcutoff`). Both
# parameterizations expand to the SAME PHYSICAL `(A, f, gamma_prime_j, f_jj)` -- only
# theta_free's own coordinate meaning differs -- so no downstream function needs a
# per-parameterization twin; only the expand/reduce step itself is parameterization-
# specific, and every call site that used to call `expand_free_theta`/`reduce_to_free_theta`
# directly now calls this dispatcher instead (delta_star.jl's `melitz_outer_state`,
# `melitz_cutoff_constraints_at`, `melitz_moments_adapter!`; affine_cutoff.jl's
# `melitz_log_cutoff_vec`; gradient_lab.jl's `fixed_active_set_moments`,
# `base_active_mask`, `count_switches`, `method_d_hand_derived`,
# `expand_theta_econ_vector`; finite_delta_outer.jl's
# `melitz_moment_directional_derivative`).
# ============================================================================

"""
    melitz_expand_theta(theta_free, ctx) -> (A, f, gamma_prime_j, f_jj)

Dispatches to `expand_free_theta` (:logf) or `expand_free_theta_logcutoff` (:logcutoff,
discarding its extra native `q` output -- callers that want `q` directly should call
`expand_free_theta_logcutoff` themselves) based on `get(ctx, :outer_parameterization,
:logf)` -- absent defaults to `:logf` so any `ctx` built before this dispatcher existed
(e.g. the Section 5 round-trip tests' own hand-built `ctx` NamedTuples) still works.

2026-07-27 addendum (governing prompt Phase 6): ALSO dispatches on the orthogonal
TECHNOLOGY-coordinate axis, `get(ctx, :technology_coordinate, :logA)` (`technology_coordinate.jl`)
-- `theta_free`'s A-block entries (`theta_free[2:1+nA]`, `nA=D^2-1`) are un-scaled from
`p_A*log(A_od)` back to plain `log(A_od)` BEFORE calling `expand_free_theta`/
`expand_free_theta_logcutoff` (both of which only ever know plain `log(A_od)` units).
`:logA` (`p_A=1`) is an exact no-op, so every existing caller/test that never sets
`ctx.technology_coordinate` is byte-for-byte unaffected. Doing this INSIDE the single
dispatcher (rather than as a separate post-hoc wrapper layer, `technology_coordinate.jl`'s
OWN now-superseded `melitz_reduce_theta_powered`/`melitz_expand_theta_powered`) means every
existing consumer of `melitz_expand_theta`/`melitz_reduce_theta` -- the outer KNITRO
objective/constraint callbacks, EVERY finite-difference gradient backend (`direct_gradient.jl`,
`sorted_crossing_gradient.jl`, `argument_localized_gradient.jl`, all of which perturb
`theta_free` directly and re-expand at each perturbed point), the moment operator, the
screens -- becomes technology-coordinate-correct automatically, with NO separate chain-rule
rescale needed anywhere else: a central finite difference computed by perturbing the POWERED
coordinate and re-expanding through THIS dispatcher at each perturbed point already IS the
correct derivative w.r.t. the powered coordinate, in the FD limit, by construction. (Verified
directly, not merely argued: `docs/melitz_outer_parameterization_comparison_2026-07-26.md`
Section F reports FD-vs-independent-FD chain-rule validation results for all three technology
candidates.)
"""
function melitz_expand_theta(theta_free::AbstractVector, ctx)
    theta_free_plain = melitz_unpower_theta_free(theta_free, ctx)
    if get(ctx, :outer_parameterization, :logf) == :logcutoff
        A, f, gamma_prime_j, f_jj, _q = expand_free_theta_logcutoff(theta_free_plain, ctx)
        return A, f, gamma_prime_j, f_jj
    else
        return expand_free_theta(theta_free_plain, ctx)
    end
end

"""
    expand_free_theta_logcutoff!(state::MelitzExpandedState, theta_free_q_plain::AbstractVector{Float64},
                                  ctx, ws::MelitzThetaExpansionWorkspace) -> state

2026-07-27 continuation (governing prompt Phase 1.2): the mutating, workspace-based
`:logcutoff` counterpart of `delta_star.jl`'s `expand_free_theta!`, completing this session's
own governing prompt's requirement that NO production-supported parameterization silently
fall back to the allocating wrapper. Mirrors `expand_free_theta_logcutoff` above exactly
(same order: `A` first, then `q[j,j]`, then the q-pivot expansion, then `f`), with two
allocation removals beyond simply making the array writes in-place:

1. **q-pivot selection reuses the ALREADY-cached f-pivot parts** (`melitz_cached_f_pivot_parts`,
   Phase 3) instead of calling `build_q_gravity_pivot(ctx)` (which would otherwise rebuild
   `f_gravity_pivot_avoid_indices`/`build_gravity_pivot` from scratch every call). This is
   exact, not an approximation: `build_q_gravity_pivot`'s own `(c, avoid)` inputs
   (`ctx.c_full[ctx.f_free_lin]`, `f_gravity_pivot_avoid_indices(ctx.D, ctx.f_free_lin,
   ctx.A_pivot.pivot)`) are IDENTICAL, term for term, to the f-pivot's own inputs
   (`melitz_build_f_pivot_parts`) -- `build_gravity_pivot`'s pivot choice is a deterministic
   function of `(c, avoid)` alone, never `g0` (`equilibrium.jl`'s own docstring), so the two
   pivots are PROVABLY the same physical cell, only their per-call `g0` offset differs (and
   that offset is rebuilt fresh here regardless, exactly as the allocating path already does).
   Verified directly (test suite, "q-pivot and f-pivot coincide"), not merely argued.
2. **`ws.logf_free_full`** (length `D^2-1`) is reused as `q_free_full` scratch -- safe because
   this field is otherwise UNUSED for the lifetime of a `:logcutoff` ctx (only
   `expand_free_theta!`'s own `:logf`-only body ever writes it, and that function is never
   called for a `:logcutoff` ctx via the single dispatcher below).

`theta_free_q_plain` must already be in plain (un-powered) `log(A_od)` units, exactly like
`expand_free_theta!`'s own contract.
"""
function expand_free_theta_logcutoff!(state::MelitzExpandedState, theta_free_q_plain::AbstractVector{Float64},
                                       ctx, ws::MelitzThetaExpansionWorkspace)
    D, j = ctx.D, ctx.target_country
    nA = D^2 - 1
    log_gamma_prime_j = theta_free_q_plain[1]
    gamma_prime_j = exp(log_gamma_prime_j)
    A_free = @view theta_free_q_plain[2:1+nA]
    q_free_free = @view theta_free_q_plain[2+nA:end]

    pivot_expand!(ws.logA_full, A_free, ctx.A_pivot)
    A = state.A
    @inbounds for i in eachindex(A)
        A[i] = exp(ws.logA_full[i])
    end

    q_jj = derive_qjj_from_autarky_cutoff(gamma_prime_j, ctx)

    c_free, q_pivot_idx, q_other = melitz_cached_f_pivot_parts(ctx)
    g0_q = build_q_gravity_offset(q_jj, ctx)
    q_pivot = GravityPivot(length(c_free), q_pivot_idx, q_other, c_free, g0_q)
    q_free_full = ws.logf_free_full
    pivot_expand!(q_free_full, q_free_free, q_pivot)

    f = state.f
    sigma = ctx.sigma
    @inbounds for (k, i) in enumerate(ctx.f_free_lin)
        o, d = lin2od(i, D)
        f[o, d] = exp(melitz_log_f_from_q(q_free_full[k], log(A[o, d]), ctx.w[o], ctx.tau[o, d],
                                           ctx.expenditure[d], sigma))
    end
    f_jj = derive_fjj_from_autarky_cutoff(gamma_prime_j, ctx.w_prime, 1.0, A[j, j],
                                           ctx.w_prime * ctx.L[j], sigma)
    f[j, j] = f_jj

    state.gamma_prime_j = gamma_prime_j
    state.f_jj = f_jj
    return state
end

"""
    melitz_expand_theta!(state::MelitzExpandedState, theta_free::AbstractVector{Float64},
                          ctx, ws::MelitzThetaExpansionWorkspace) -> state

2026-07-27 continuation (governing prompt Phase 2): the mutating, production-hot-path
counterpart of `melitz_expand_theta` above -- un-scales `theta_free`'s A-block into plain
`log(A_od)` units DIRECTLY INTO `ws.theta_plain` (no new `Vector`, mirroring
`melitz_unpower_theta_free`'s own formula exactly), then delegates to `expand_free_theta!`
(`:logf`) or `expand_free_theta_logcutoff!` (`:logcutoff`) based on
`ctx.outer_parameterization`, exactly mirroring the allocating `melitz_expand_theta`
dispatcher's own logic.

2026-07-27 continuation (governing prompt Phase 1.2): previously threw `ArgumentError` for
`:logcutoff` (disclosed gap from the prior session). Both production-supported
parameterizations now have a genuine in-place expansion path -- every existing consumer of
`melitz_expand_theta` (screens, diagnostics, `gradient_lab.jl`) is UNCHANGED and continues to
call the allocating dispatcher above; only the identified O(n_theta)-per-gradient-call hot
sites (`sorted_crossing_gradient.jl`'s `_fill_compact_direct_columns_crossing_sorted!`,
`direct_gradient.jl`'s `_direct_coordinate_grad`, and their focal-link-fill siblings) call
this mutating entry point instead, and now work correctly regardless of which
parameterization the bundle was built with.
"""
function melitz_expand_theta!(state::MelitzExpandedState, theta_free::AbstractVector{Float64},
                               ctx, ws::MelitzThetaExpansionWorkspace)
    technology_coordinate = get(ctx, :technology_coordinate, :logA)
    p_A = melitz_technology_coordinate_scale(technology_coordinate, ctx)
    logcutoff = get(ctx, :outer_parameterization, :logf) == :logcutoff
    if p_A == 1.0
        # 2026-07-27 continuation (governing prompt Phase 4, memory-traffic audit): `:logA`
        # (p_A=1, the production default technology coordinate -- melitz_technology_coordinate_scale's
        # own no-op case) means the un-scale below is the IDENTITY -- `copyto!(ws.theta_plain,
        # theta_free)` was therefore copying all `2D^2-2` entries of `theta_free` UNCHANGED,
        # every coordinate probe (2x/coordinate), purely so `expand_free_theta!`/
        # `expand_free_theta_logcutoff!` had a `Float64`-typed argument to read -- `theta_free`
        # itself already IS exactly that. Skips the copy entirely in this (default) case;
        # the `p_A != 1.0` branch below is unchanged (a genuine rescale, still needs its own
        # scratch destination).
        if logcutoff
            expand_free_theta_logcutoff!(state, theta_free, ctx, ws)
        else
            expand_free_theta!(state, theta_free, ctx, ws)
        end
        return state
    end
    theta_plain = ws.theta_plain
    copyto!(theta_plain, theta_free)
    nA = ctx.D^2 - 1
    @inbounds for i in 2:1+nA
        theta_plain[i] /= p_A
    end
    if logcutoff
        expand_free_theta_logcutoff!(state, theta_plain, ctx, ws)
    else
        expand_free_theta!(state, theta_plain, ctx, ws)
    end
    return state
end

"""
    melitz_reduce_theta(p::MelitzPrimitives, ctx) -> theta_free

Inverse dispatcher: routes to `reduce_to_free_theta` (:logf) or
`reduce_to_free_theta_logcutoff` (:logcutoff) based on `ctx.outer_parameterization`, THEN
re-scales the A-block into `ctx.technology_coordinate`'s own units (`melitz_power_theta_free`,
exact inverse of `melitz_unpower_theta_free` above) -- see `melitz_expand_theta`'s own
docstring for the full addendum rationale.
"""
function melitz_reduce_theta(p::MelitzPrimitives, ctx)
    theta_free_plain = if get(ctx, :outer_parameterization, :logf) == :logcutoff
        reduce_to_free_theta_logcutoff(p.A, p.f, p.gamma_prime_target, ctx)
    else
        reduce_to_free_theta(p, ctx)
    end
    return melitz_power_theta_free(theta_free_plain, ctx)
end
