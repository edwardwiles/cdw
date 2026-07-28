using LinearAlgebra: I, norm

# Typed inner-solve classification + cheap pre-KNITRO screens.
#
# Session: reduce the cost of problematic inner points in the finite-delta outer search
# (docs/melitz_optimization_report_2026-07-23_continuation.md's #1 finding: failed
# inner-solve attempts are 56-93% of every trajectory's wall time, individual failures up
# to ~55s vs. ~30-70ms for a successful solve). Implements the governing prompt's typed
# classification (Section 2) plus the ADDENDUM's front-loaded, no-routine-cold-retry
# screening order (addendum Sections 1-6), which supersedes the main prompt's own
# warm/cold retry and screen-ordering instructions.
#
# -- Mathematical basis for the two screens implemented this session --
#
# Finite-support feasibility (main prompt Section 3): at a fixed outer point theta, the
# moment system E_p[G[:,k]] = 0, k=1..K is satisfiable by some probability vector p (p>=0,
# sum(p)=1 over the W draws) iff 0 lies in the convex hull of the W rows of G -- iff, for
# EVERY column k considered alone, min(G[:,k]) <= 0 <= max(G[:,k]) (a two-point
# distribution putting weight on that column's argmin/argmax draw attains any value in
# between, so the single-column range test is exactly necessary and sufficient for that
# ONE column's own marginal). This is a NECESSARY condition for the full K-column joint
# problem (a violation on any single column certifies infeasibility of the whole system);
# it is not sufficient (all columns can individually span zero while the joint hull still
# excludes it) -- the K-dimensional convex-hull LP the main prompt's Sections 5-6 describe
# would close that gap, but is NOT implemented this session (see the session report's
# Section C "screen effectiveness" for why the range screen alone was judged sufficient
# for this session's rejection needs).
#
# Stored-dual lower bound (addendum Section 4.3/8.1): the inner CC dual problem minimizes
# a raw functor value `f(zeta,lambda; G) = zeta + (1/M) sum_w Psi*(-zeta-lambda'G[w,:])`
# over UNCONSTRAINED (zeta,lambda) -- every point in R^{1+d} is dual-feasible, so for any
# (zeta,lambda), weak duality gives `f(zeta,lambda; G) >= f* = min f = -Delta(G)`, i.e.
# `-f(zeta,lambda; G) <= Delta(G)` for literally EVERY (zeta,lambda), not just a verified
# optimum. A dual vector `x=(zeta,lambda)` verified optimal at some OTHER moment matrix
# G_old therefore still gives a valid (generally loose) lower bound on Delta at a NEW G,
# at the cost of one functor evaluation (`obj(x)`, no gradient/constraint request -- a
# single BLAS gemv! plus an elementwise Psi! map, no KNITRO call).

# ============================================================================
# Section 2: typed inner-solve result.
#
# 2026-07-24 evaluation-cap-correction session (governing prompt:
# docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md): RENAMED from the prior
# session's `InnerSolved`/`BudgetInfeasible`/`MomentInfeasible` vocabulary to the corrected
# four-way taxonomy the governing prompt specifies. This is not a cosmetic rename -- the old
# name `BudgetInfeasible` was itself part of the conceptual error being corrected: it
# conflated "we aborted before certifying DeltaStar(theta) against the CURRENT outer budget
# delta" with "the point is economically infeasible," which is false whenever the true
# DeltaStar is merely finite-and-over-budget (an ordinary, fully solvable case under the
# corrected semantics, Case A below). The new names describe exactly what was PROVEN, no
# more:
#
#   FiniteSolved            -- DeltaStar(theta) IS finite and this IS its optimized value
#                              (whether <= or > the outer budget delta -- budget-relative
#                              status is a property the CALLER derives from `.Delta`, never
#                              a reason to withhold or substitute this result).
#   AboveEvaluationCap      -- only `DeltaStar(theta) > delta_evaluation_cap` is certified;
#                              the true value is UNKNOWN (finite-above-cap, or infinite).
#   InfiniteDeltaCertified  -- DeltaStar(theta) = +infinity, PROVEN by an exact certificate
#                              (no evaluation cap involved at all -- the feasible set is
#                              empty regardless of any cap).
#   NumericalFailure        -- no certificate of any kind was obtained.
# ============================================================================

"Abstract supertype for a classified inner-solve outcome."
abstract type MelitzInnerResult end

"""
Case A (governing prompt Section 2): the inner CC dual problem was genuinely solved to
KNITRO's accepted optimal status. `Delta` is the ACTUAL, fully-optimized `DeltaStar(theta)`
-- finite by construction of reaching this branch -- `x` is the optimal dual at that
optimum, `nStatus` is KNITRO's own accepted status code. Returned IDENTICALLY whether
`Delta <= delta_evaluation_cap`'s outer budget or not: a point with true `DeltaStar` of
1.5, 2, or 5 must be reported here with its genuine solved value, dual, and (via the
existing envelope-theorem-exact gradient machinery) gradient -- never intercepted early
merely because it exceeds some OUTER budget `delta` (that budget plays no role in whether
this branch is reached; only `delta_evaluation_cap`, Case B below, can abort a solve before
this point).
"""
struct FiniteSolved <: MelitzInnerResult
    Delta::Float64
    x::Vector{Float64}
    nStatus::Int
end

"""
Case B (governing prompt Section 2): a valid dual lower bound certifies
`DeltaStar(theta) > delta_evaluation_cap`, obtained WITHOUT ever converging the inner dual
problem to a genuine optimum -- either the KNITRO-native mid-solve bailout
(`source=:live_dual_threshold`) or a cheap pre-solve screen against an already-known dual
(`source in (:stored_dual, :dual_polish)`).

`certified_lower_bound` is a VALID, FINITE lower bound (unconditional weak duality, file
header) on the TRUE `DeltaStar(theta)` -- it is NOT `DeltaStar(theta)` itself, and must
NEVER be reported/logged as `delta_star`/`Delta=<value>` (governing prompt Section 11: write
`certified_lower_bound=...; delta_star_solved=false`, never `Delta=<value>` for this case).
The true `DeltaStar(theta)` may be ANY finite value `>= certified_lower_bound` (including
values far above the cap) or `+infinity` -- this result proves only the ONE inequality
`DeltaStar(theta) > delta_evaluation_cap`, nothing more precise.

`x` is the finite dual vector that CERTIFIES `certified_lower_bound` (weak duality holds at
ANY finite dual point, not only an optimum) -- kept so a caller that deliberately wants a
fixed-dual surrogate gradient AT THIS CERTIFICATE (Policy B in `finite_delta_outer.jl`, NOT
the default) has it available, though the default outer-callback policy (Policy A) does not
use it at all.

`crossing_time_s`/`crossing_iteration` (governing prompt Section 2's "threshold-crossing
iteration, threshold-crossing time"): populated only for `source==:live_dual_threshold`
(the one genuinely ITERATIVE producer -- `crossing_time_s` is wall-clock elapsed from this
inner-solve attempt's own start to the KNITRO-native bailout) and `:dual_polish` (the damped-
Newton polish loop's own step index at certification, and its own elapsed wall time).
`:stored_dual` is a single, static bank lookup with no iteration to time a "crossing"
against -- reported as `crossing_time_s=0.0, crossing_iteration=0` by convention (documented
here, not a real "first-iteration" claim). `crossing_iteration=-1` for `:live_dual_threshold`
specifically is a DISCLOSED LIMITATION, not a silent omission: `cc_algo/PsiObjectiveBundle.jl`
(deliberately unmodified, shared with the Ricardian model) does not expose a barrier-
iteration counter to its own early-bailout branch, only a timestamp
(`threshold_crossing_time_ns`) -- threading a genuine iteration count through would require
touching that shared file, out of scope for this session.
"""
struct AboveEvaluationCap <: MelitzInnerResult
    certified_lower_bound::Float64
    source::Symbol
    x::Vector{Float64}
    crossing_time_s::Float64
    crossing_iteration::Int
end

"""
    MELITZ_ABOVE_EVALUATION_CAP_SOURCES

Every `source` symbol an `AboveEvaluationCap` result may carry, for validation/testing.
`:stored_dual`/`:dual_polish` (single-shot screens against an already-known dual) and
`:live_dual_threshold` (the KNITRO-native `lower_limit` mid-solve early stop -- see
`melitz_classified_inner_solve`'s docstring below and `cc_algo/PsiObjectiveBundle.jl`'s
`threshold_crossed` fields).
"""
const MELITZ_ABOVE_EVALUATION_CAP_SOURCES = (:stored_dual, :dual_polish, :live_dual_threshold)

"""
Case C (governing prompt Section 2): `DeltaStar(theta) = +infinity`, PROVEN (not merely
suspected, and independent of any `delta_evaluation_cap`) by an exact finite-support
separation certificate -- moment column `column`'s achievable range `[lo, hi]` across all
`W` draws does not contain 0, so no probability vector over the draws can satisfy that
moment: the inner CC feasible set is EMPTY, no finite divergence value exists at all. `kind`
is currently `:range` (the always-on O(W*K) screen) or `:origin_block` (the compressed
origin-block LP screen, `origin_block_screen.jl`) -- both exact certificates; the general
K-dimensional convex-hull LP (main prompt Sections 5-6 of the PRIOR session) is documented
but not implemented, see that session's report Section C.
"""
struct InfiniteDeltaCertified <: MelitzInnerResult
    column::Int
    lo::Float64
    hi::Float64
    kind::Symbol
end

"""
Moments pass every implemented screen but the smooth dual optimum could not be certified
attained. NOT produced by this session's production classifier (`melitz_classified_inner_solve`
below never returns this -- distinguishing a genuine boundary/non-attained-dual case from
an ordinary `NumericalFailure` needs the primal-LP/relative-interior diagnostics main
prompt Section 7 describes, which this session did not implement, see report Section C).
Kept as a real case in the type so a future diagnostic pass can populate it without
another type-hierarchy change.
"""
struct BoundaryFeasible <: MelitzInnerResult
    note::String
end

"""
Case D (governing prompt Section 2/7): no mathematical certificate of any kind was
obtained -- the (single, no-retry) KNITRO attempt returned a status outside the accepted
set, no screen rejected the point first, and no cap-crossing certificate fired either. This
INCLUDES a routine inner time/iteration cap being reached (governing prompt Section 7:
"timeout does not imply infinite DeltaStar... timeout does not imply AboveEvaluationCap...
timeout is NumericalFailure unless a valid lower-bound or infeasibility certificate already
exists") -- `nStatus` in that case is whatever code KNITRO's own `maxtime_real`/`maxit`
inner-solve option reports; if the KNITRO-native threshold ALSO happened to fire before the
time limit was reached, `melitz_classified_inner_solve` returns `AboveEvaluationCap`
instead (checked first, see that function's body) -- a genuine certificate already in hand
is never downgraded to `NumericalFailure` merely because the SAME attempt also ran out of
time. No `DeltaStar` value is ever invented for this case.
"""
struct NumericalFailure <: MelitzInnerResult
    nStatus::Int
end

# ============================================================================
# Section 4.1 (addendum): always-on O(W*K) range screen.
# ============================================================================

"""
    melitz_range_screen(G::AbstractMatrix; guard=0.0) -> Union{Nothing,InfiniteDeltaCertified}

For every column `k` of `G` (`W x num_moments`), tests `min(G[:,k]) <= guard <= max(G[:,k])`.
Returns the FIRST violated column's certificate, or `nothing` if every column's range
spans `guard` (normally `0`). O(W*K), a single pass over `G` (already resident in
`obj.H` after a `moments!` call) -- no extra allocation, no KNITRO call. Applies uniformly
to trade-share columns AND the focal-link column (main prompt Section 4's "for the focal
link, require the draw-level link contribution to span zero" is the SAME test, not a
separate rule -- the focal link is just another column of `G`).
"""
function melitz_range_screen(G::AbstractMatrix{<:Real}; guard::Real=0.0)
    W, K = size(G)
    @inbounds for k in 1:K
        lo, hi = extrema(@view G[:, k])
        if !(lo <= guard <= hi)
            return InfiniteDeltaCertified(k, lo, hi, :range)
        end
    end
    return nothing
end

"""
    melitz_range_screen(op::MelitzMomentOperator; guard=0.0) -> Union{Nothing,InfiniteDeltaCertified}

2026-07-26 closure session (governing prompt Phase 7): a matrix-free equivalent of the dense
range screen above, computed WITHOUT ever materializing `G` -- `O(D^2)` given `op`'s own
already-updated state (`melitz_update_moment_operator!`'s `O(W*D)` cost, amortized across
every screen/objective/gradient/Hessian call at this outer point, NOT repeated here).

Algebra: for trade cell `(o,d)`, `G[w,idx] = coef[o,d]*z_power[w,o] - lambda[o,d]` when draw
`w` is ACTIVE for `d` (`bin[w,o] >= rank[d,o]`), else `G[w,idx] = -lambda[o,d]` (a CONSTANT
across every inactive draw). So the column's value set is
`{-lambda[o,d]} UNION {coef[o,d]*z_power[w,o] - lambda[o,d] : w active}` (the first set is
only present if at least one draw is inactive). Because
`sorted_ctx.sorted_z_power[:,o]` is sorted ASCENDING and `bin` is a non-decreasing function
of `z`, "active for threshold `t`" is exactly the SORTED SUFFIX from `op.first_active_pos[t+1,o]`
to `W` -- so the active set's `z_power` extrema are just its two endpoints:
`sorted_z_power[first_active_pos[t+1,o], o]` (min) and `sorted_z_power[W,o]` (max, the SAME
for every destination at this origin, since the top of the suffix never changes). No sign
assumption on `coef` is made (both `coef*zmin` and `coef*zmax` are computed and min/max'd
directly, robust either way). "At least one inactive draw exists" iff
`rank[d,o] > bin[perm_o[1],o]` (the smallest-z draw's own bin value) -- the complementary
condition to `first_active_pos[1,o]==1` always holding for every destination trivially at
`t=0` only.

The focal link column (`ell`) is already fully dense (no active/inactive gate) -- its range
is a direct `extrema(op.ell)`, exactly as the dense reference computes its own dense columns.

Returns the FIRST violated column's certificate (SAME `InfiniteDeltaCertified` type, same
`(column_index, lo, hi, :range)` fields as the dense method), scanning trade cells in the SAME
`(o,d)` order the dense method's own column index enumeration uses (`trade_index[o,d]`) so a
caller comparing "which column failed first" against the dense reference sees a directly
comparable index.
"""
function melitz_range_screen(op::MelitzMomentOperator; guard::Real=0.0)
    D = op.D
    W = op.W
    sorted_ctx = op.sorted_ctx
    trade_index = op.layout.trade_index
    coef = op.coef
    lambda = op.lambda
    rank = op.rank
    first_active_pos = op.first_active_pos
    sorted_zpow = sorted_ctx.sorted_z_power
    perm = sorted_ctx.permutation

    @inbounds for o in 1:D
        zpow_max_o = sorted_zpow[W, o]
        min_bin_o = Int(op.bin[perm[1, o], o])
        for d in 1:D
            t = rank[d, o]
            lo = Inf
            hi = -Inf
            if t > min_bin_o   # at least one inactive draw exists at this threshold
                v = -lambda[o, d]
                lo = min(lo, v); hi = max(hi, v)
            end
            fpos = first_active_pos[t+1, o]
            if fpos <= W   # at least one active draw exists at this threshold
                zpow_min_active = sorted_zpow[fpos, o]
                v1 = coef[o, d] * zpow_min_active - lambda[o, d]
                v2 = coef[o, d] * zpow_max_o - lambda[o, d]
                lo = min(lo, v1, v2); hi = max(hi, v1, v2)
            end
            if !(lo <= guard <= hi)
                return InfiniteDeltaCertified(trade_index[o, d], lo, hi, :range)
            end
        end
    end

    lo_ell, hi_ell = extrema(op.ell)
    if !(lo_ell <= guard <= hi_ell)
        return InfiniteDeltaCertified(op.layout.focal_link_index, lo_ell, hi_ell, :range)
    end
    return nothing
end

# ============================================================================
# Section 4.3/8.1 (addendum): stored-dual lower-bound screen.
# ============================================================================

"""
    MelitzDualBank(max_size=8; policy=:fifo)

Phase I.6 (this session): `policy` controls which entry is EVICTED once the bank is full
and a new (finite) dual vector arrives -- three options, compared live in the session
report:

  - `:fifo` (default, matches the pre-Phase-I.6 behavior exactly): evict the oldest entry.
  - `:nearest`: evict whichever EXISTING entry is closest (Euclidean) to the incoming
    point -- the incoming point is the most redundant replacement for its own nearest
    neighbor, so this keeps the bank from accumulating near-duplicates in one region.
  - `:diversity`: evict whichever EXISTING entry is closest to ITS OWN nearest neighbor
    within the bank (the single most redundant entry overall, a simple greedy diversity
    heuristic) -- independent of where the incoming point lands.

Every insertion point in this file funnels through `melitz_dual_bank_insert!`, which
unconditionally refuses a non-finite vector (main prompt Section 6: "do not insert NaNs
or nonfinite vectors") -- centralized here rather than trusted at every call site.

Continuation session (2026-07-23), Section 5: `thetas` is a PARALLEL array (same length,
same indexing as `entries`) recording the outer coordinate `theta` each dual vector was
obtained at, when the caller supplies one -- `nothing` at an index means "no theta known
for this entry" (every EXISTING call site/test that calls `melitz_dual_bank_insert!` without
the new optional `theta` kwarg keeps working exactly as before, just with an untagged
entry). This enables `melitz_bank_nearest_theta` (below) to answer "which banked dual is
closest to the point I am ABOUT TO evaluate," the missing piece needed to use the bank as an
actual inner-solve WARM START (not merely a pre-solve screening lower bound, its only use
before this session) -- see `melitz_classified_inner_solve`'s new `warm_start_source` kwarg.
"""
mutable struct MelitzDualBank
    entries::Vector{Vector{Float64}}
    thetas::Vector{Union{Nothing,Vector{Float64}}}
    max_size::Int
    policy::Symbol
end
MelitzDualBank(max_size::Int=8; policy::Symbol=:fifo) =
    MelitzDualBank(Vector{Float64}[], Union{Nothing,Vector{Float64}}[], max_size, policy)

function melitz_dual_bank_insert!(bank::MelitzDualBank, x::AbstractVector;
                                   theta::Union{Nothing,AbstractVector}=nothing)
    xc = collect(Float64.(x))
    all(isfinite, xc) || return nothing
    thetac = theta === nothing ? nothing : collect(Float64.(theta))
    if length(bank.entries) < bank.max_size
        push!(bank.entries, xc)
        push!(bank.thetas, thetac)
        return nothing
    end
    if bank.policy == :fifo
        popfirst!(bank.entries)
        popfirst!(bank.thetas)
        push!(bank.entries, xc)
        push!(bank.thetas, thetac)
    elseif bank.policy == :nearest
        i_near = argmin([norm(e .- xc) for e in bank.entries])
        bank.entries[i_near] = xc
        bank.thetas[i_near] = thetac
    elseif bank.policy == :diversity
        n = length(bank.entries)
        nn_dist = [minimum(norm(bank.entries[i] .- bank.entries[j]) for j in 1:n if j != i) for i in 1:n]
        i_redundant = argmin(nn_dist)
        bank.entries[i_redundant] = xc
        bank.thetas[i_redundant] = thetac
    else
        error("unknown MelitzDualBank policy: $(bank.policy) (expected :fifo, :nearest, or :diversity)")
    end
    return nothing
end

"""
    melitz_bank_nearest_theta(bank, theta) -> (dist, x)

Continuation session (2026-07-23), Section 5: scans `bank` for the THETA-TAGGED entry
(`bank.thetas[i] !== nothing`) closest (Euclidean, in outer-coordinate space) to the query
`theta`, returning its Euclidean distance and its own dual vector `x` -- the natural
"nearest verified bank dual" warm-start candidate. `(Inf, nothing)` if the bank is empty or
has no theta-tagged entries at all (e.g. every entry came from a call site that did not pass
`theta`, or the bank is fresh).
"""
function melitz_bank_nearest_theta(bank::MelitzDualBank, theta::AbstractVector)
    best_dist = Inf
    best_x = nothing
    @inbounds for i in eachindex(bank.entries)
        th = bank.thetas[i]
        th === nothing && continue
        d = norm(th .- theta)
        if d < best_dist
            best_dist = d
            best_x = bank.entries[i]
        end
    end
    return (best_dist, best_x)
end

"""
    melitz_without_lower_limit_bailout(f, obj)

2026-07-24 outer-benchmark-correction session, main prompt Section 3 (bug found LIVE while
validating the corrected `lower_limit_guard`): `cc_algo/PsiObjectiveBundle.jl`'s shared
functor (`(Q::PsiObjectiveBundleImplicit)(...)`) applies its `if f <= lower_limit; return
-KNITRO.KN_INFINITY` early-bailout UNCONDITIONALLY on every call to `obj(...)` -- not only
during a genuine NESTED KN_solve barrier iteration (its intended use, `:live_dual_threshold`
above) but also during a bare DIAGNOSTIC/screening call
(`melitz_stored_dual_lower_bound`/`melitz_bank_best`/`melitz_dual_polish_screen`, all called
OUTSIDE of any KN_solve). Once `lower_limit_guard` is set to a small NUMERICAL value (main
prompt Section 2's own correction, vs. the old `49.0` margin), an ordinary, only-modestly-
infeasible point's TRUE `-f` routinely exceeds `delta+guard` too -- so these screens silently
received `-KNITRO.KN_INFINITY` (`-floatmax(Float64)`) as `f`, i.e. `lb=-f=floatmax(Float64)`,
instead of the TRUE, modest, informative lower bound -- corrupting exactly the constraint
value the main prompt's Section 3 says must never be `floatmax`/a sentinel. Confirmed live:
guard in `{1e-8,1e-6,1e-4}` all reproduced `certified_bound == floatmax(Float64)` exactly for
the stored-dual screen at a genuinely `Delta≈1.06` point.

Root cause is the shared functor's own unconditional check, which this repo's convention
(documented throughout this file and `finite_delta_outer.jl`) deliberately never modifies
(`cc_algo/PsiObjectiveBundle.jl` stays byte-for-byte reusable by the Ricardian model). Fix,
scoped entirely to Melitz's own screening code: temporarily set `obj.lower_limit = -Inf`
(the bailout condition `f <= -Inf` is then unreachable for any finite `f`) around exactly the
bare diagnostic call(s) `fn` makes, then restore the REAL guard-based limit afterward in a
`finally` block -- so a genuinely nested nStatus check inside `CS.inner_loop_internal`
(called separately, never inside this wrapper) still sees and uses the real
`lower_limit`/`threshold_crossed` mechanism exactly as before, unaffected.
"""
function melitz_without_lower_limit_bailout(fn, obj)
    saved_limit = obj.lower_limit
    obj.lower_limit = -Inf
    try
        return fn()
    finally
        obj.lower_limit = saved_limit
    end
end

"""
    melitz_stored_dual_lower_bound(obj, bank) -> Float64

Evaluates every dual vector in `bank` against the CURRENTLY-loaded `obj.H` (caller must
have already run `obj.moments!` for the query `theta`) via the bare functor call
`obj(x)` (no gradient/constraint request -- the cheapest possible call). Returns the
TIGHTEST (maximum) `-f(x)` over the bank, a valid lower bound on `Delta` at this `theta`
(see file header); `-Inf` if the bank is empty. Wrapped in
`melitz_without_lower_limit_bailout` (see that function's docstring) so a tight
evaluation cap never corrupts this TRUE lower bound with the live-solve
`-KNITRO.KN_INFINITY` sentinel.
"""
function melitz_stored_dual_lower_bound(obj, bank::MelitzDualBank)
    isempty(bank.entries) && return -Inf
    melitz_without_lower_limit_bailout(obj) do
        best = -Inf
        for x in bank.entries
            f = obj(x)
            lb = -f
            lb > best && (best = lb)
        end
        best
    end
end

# ============================================================================
# Phase I.5: cheap dual-polishing budget screen.
# ============================================================================

"""
    melitz_dual_polish_screen(obj, x0; delta_evaluation_cap, max_steps=3,
        damping=1e-6, max_backtrack=4) -> Union{Nothing,AboveEvaluationCap}

A small, fixed number of damped Newton steps on the CANONICAL exact dual functor in
`(zeta,lambda)`-space -- `obj(x, g; h=H)` returns the raw objective `f` and fills the exact
gradient `g` and exact Hessian `H` (the SAME functor the inner KNITRO solve itself uses,
`cc_algo/PsiObjectiveBundle.jl`; no separate/approximate objective is introduced), starting
from `x0` (typically the stored-dual bank's own best-lower-bound entry).

No convergence claim is made or needed: by the SAME unconditional weak-duality argument as
`melitz_stored_dual_lower_bound` (file header above), EVERY finite dual iterate visited --
including `x0` itself, before any step is taken -- gives a valid `Delta(theta)` lower bound
`-f(x)`. The moment ANY visited iterate's bound exceeds `delta_evaluation_cap`, this
function returns immediately with a certified `AboveEvaluationCap(lb, :dual_polish, ...)`; if `max_steps` damped
Newton steps complete without a certified rejection, returns `nothing` (not a claim of
feasibility -- merely "this screen did not reject").

The line search is safeguarded per main prompt Section 12: a candidate step is accepted
only if it is FINITE and does not increase the raw objective (`f_try <= f`, i.e. does not
WORSEN the lower bound) -- halved up to `max_backtrack` times, else the polish stops (not
an error; the caller proceeds to a real KNITRO attempt).

2026-07-26 (user-directed simplification): the threshold used to be `delta_evaluation_cap +
guard` (a small additive margin) -- removed. The cap value itself is already the correct
exact threshold; no separate margin is needed (same rationale as
`inner_solve_config.jl`'s own guard removal).
"""
function melitz_dual_polish_screen(obj, x0::AbstractVector; delta_evaluation_cap::Real,
                                    max_steps::Int=3, damping::Real=1e-6, max_backtrack::Int=4)
    melitz_without_lower_limit_bailout(obj) do   # see melitz_without_lower_limit_bailout's docstring
        t0 = time_ns()
        n = length(x0)
        x = collect(Float64.(x0))
        all(isfinite, x) || return nothing
        g = zeros(n)
        H = zeros(n, n)
        f = obj(x, g; h=H)
        isfinite(f) || return nothing
        lb = -f
        lb > delta_evaluation_cap &&
            return AboveEvaluationCap(lb, :dual_polish, copy(x), (time_ns() - t0) / 1e9, 0)

        for step in 1:max_steps
            local dir
            try
                dir = -((H + damping * I) \ g)
            catch
                break   # singular/ill-conditioned Hessian at this iterate -- stop polishing, not an error
            end
            (isempty(dir) || any(!isfinite, dir)) && break

            accepted = false
            step_scale = 1.0
            for _ in 1:max_backtrack
                x_try = x .+ step_scale .* dir
                g_try = zeros(n)
                H_try = zeros(n, n)
                f_try = obj(x_try, g_try; h=H_try)
                if isfinite(f_try) && f_try <= f
                    x, g, H, f = x_try, g_try, H_try, f_try
                    accepted = true
                    break
                end
                step_scale /= 2
            end
            accepted || break

            lb = -f
            lb > delta_evaluation_cap &&
                return AboveEvaluationCap(lb, :dual_polish, copy(x), (time_ns() - t0) / 1e9, step)
        end
        return nothing
    end
end

# ============================================================================
# Orchestration: front-loaded screens, single (no-retry) KNITRO attempt, typed result.
# ============================================================================

"""
    melitz_bank_best(obj, bank) -> (best_lb, best_x)

Scans `bank` once, returning both the tightest stored-dual lower bound (main prompt
Section 8.1, `melitz_stored_dual_lower_bound`'s own computation) AND the entry that
attains it -- the natural starting point for the dual-polish screen (Phase I.5: "starting
from the stored dual with the best current lower bound"). `(-Inf, nothing)` if the bank is
empty. Wrapped in `melitz_without_lower_limit_bailout` (see that function's docstring) so a
tight evaluation cap never corrupts this TRUE lower bound with the live-solve
`-KNITRO.KN_INFINITY` sentinel.
"""
function melitz_bank_best(obj, bank::MelitzDualBank)
    isempty(bank.entries) && return (-Inf, nothing)
    melitz_without_lower_limit_bailout(obj) do
        best_lb = -Inf
        best_x = bank.entries[1]
        for xb in bank.entries
            lb = -obj(xb)
            if lb > best_lb
                best_lb = lb
                best_x = xb
            end
        end
        (best_lb, best_x)
    end
end

"""
    melitz_resolve_warm_start!(obj, bank, theta, warm_start_source) -> Symbol

Continuation session (2026-07-23), Section 5: sets `obj.x`/`obj.use_cached_x` per
`warm_start_source` BEFORE the real KNITRO attempt, and returns the source ACTUALLY applied
(useful for logging -- may differ from the requested source when a bank lookup finds
nothing, e.g. `:bank_nearest` on a bank with no theta-tagged entries falls back to
`:previous`, never erroring or silently doing nothing unexpected):

  - `:previous` (default, matches ALL prior behavior exactly): no-op -- `obj.x`/
    `obj.use_cached_x` are left exactly as the caller/previous solve set them.
  - `:bank_nearest`: `melitz_bank_nearest_theta(bank, theta)`'s own dual vector, the
    theta-tagged bank entry closest to the point about to be evaluated. Falls back to
    `:previous` (a no-op) if no theta-tagged entry exists.
  - `:bank_best_lb`: `melitz_bank_best(obj, bank)`'s own dual vector, the entry giving the
    TIGHTEST stored-dual lower bound against `obj.H`'s CURRENTLY loaded moment matrix
    (already populated at `theta` by the caller before this is invoked). Falls back to
    `:previous` if the bank is empty.
  - `:neutral`: clears the cache (`obj.use_cached_x = false; obj.x .= NaN`), forcing
    `inner_loop_initial_values`'s own `zeros(...)` default -- a diagnostic-only baseline for
    comparing warm-start policies against "no warm start at all," never itself a routine
    cold-RETRY (this remains a single, no-retry attempt either way; addendum Section 1's own
    no-routine-cold-retry policy is unaffected).

`obj.x`/`obj.use_cached_x` must already be correctly sized/initialized fields of `obj`
(true for any live `PsiObjectiveBundleImplicit`, the only type this is ever called on).
"""
function melitz_resolve_warm_start!(obj, bank::MelitzDualBank, theta::AbstractVector,
                                     warm_start_source::Symbol)
    if warm_start_source == :previous
        return :previous
    elseif warm_start_source == :bank_nearest
        _, x_near = melitz_bank_nearest_theta(bank, theta)
        if x_near === nothing
            return :previous
        end
        obj.x .= x_near
        obj.use_cached_x = true
        return :bank_nearest
    elseif warm_start_source == :bank_best_lb
        _, x_best = melitz_bank_best(obj, bank)
        if x_best === nothing
            return :previous
        end
        obj.x .= x_best
        obj.use_cached_x = true
        return :bank_best_lb
    elseif warm_start_source == :neutral
        obj.use_cached_x = false
        obj.x .= NaN
        return :neutral
    else
        throw(ArgumentError("warm_start_source must be :previous, :bank_nearest, " *
            ":bank_best_lb, or :neutral, got $warm_start_source"))
    end
end

"""
    _melitz_classified_inner_solve!(session::MelitzInnerSession, theta;
        range_screen=true, stored_dual_screen=true, dual_polish_screen=false,
        dual_polish_steps=3, origin_block_screen=false, screen_order=:A,
        warm_start_source=:previous, on_result=nothing) -> MelitzInnerResult

**INTERNAL as of the 2026-07-28 inner-solver architecture-consolidation session -- call
`solve_melitz_delta!` (`inner_session.jl`) instead of this function directly.** This is the
one node every sanctioned entry path (outer FC/GA, fixed-point probe, nuisance profile,
final verification, calibration fixture, diagnostic script, test fixture) funnels through
before a `FiniteSolved`/`AboveEvaluationCap`/`InfiniteDeltaCertified`/`NumericalFailure`
verdict can ever be produced -- `solve_melitz_delta!` is the only caller this session adds,
and is itself the ONLY function outside this one that production code/tests should call.

Governing prompt Section 3 (this session): the evaluation cap is no longer a separate
`delta_evaluation_cap::Real` argument passed independently of the bundle's own `lower_limit`
-- that independence was the root cause of the `1.510118e14` `FiniteSolved`-above-cap anomaly
(`docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md`): a caller could
pass `delta_evaluation_cap=10.0` here while `obj.lower_limit` (set once, at bundle
CONSTRUCTION time, by an entirely different code path) remained `-KNITRO.KN_INFINITY`, so the
two numbers silently disagreed. `session::MelitzInnerSession` now carries BOTH `obj` (whose
`lower_limit` was set FROM `session.policy` at construction, asserted consistent by
`MelitzInnerSession`'s own constructor and re-asserted by `solve_melitz_delta!`) and the
`policy` itself (`melitz_policy_cap(session.policy)` is the ONE place this function reads the
cap from) -- there is no longer a second number that could drift out of sync with the first;
`obj`/`ctx`/`bank` are likewise read off `session`, not passed independently.

Addendum Section 3/6's production order, extended by this session's Phase I.3/I.5/I.8: (1)
fill `obj.H`'s moment matrix at `theta` (the same `obj.moments!` call
`CounterfactualSensitivity.inner_loop_internal` would make -- this duplicates that one call
ONLY when every screen passes and a real solve proceeds); (2) the O(W*K) range screen,
ALWAYS first (main prompt Section 8's own screen orders all agree on this); (3) the
remaining enabled screens, in the order named by `screen_order`:

  - `:A` (default): stored-dual, then compressed origin-block, then dual-polish.
  - `:B`: compressed origin-block, then stored-dual, then dual-polish.
  - `:C`: stored-dual, then dual-polish, then compressed origin-block.

(4) if nothing rejects, ONE KNITRO attempt (`CounterfactualSensitivity.inner_loop_internal`),
warm-started per `warm_start_source` (Section 5, continuation session; default `:previous`,
reusing the caller's existing `obj.use_cached_x`/`obj.x` exactly as before this kwarg
existed) -- NO routine cold retry on a numerical failure (addendum Section 1), regardless of
`warm_start_source`.

2026-07-24 evaluation-cap-correction session (governing prompt Section 3): every early-abort
threshold this function relies on is conceptually gated on the evaluation cap, never the
OUTER budget `delta`. The prior (pre-2026-07-24) version took a `delta::Real` kwarg here and
rejected as soon as a certified lower bound exceeded `delta+guard` -- this was EXACTLY the
conceptual error that session's governing prompt diagnosed: a certificate that
`DeltaStar(theta) > delta` (the CURRENT outer budget) is not evidence the point is
unsolvable or that `DeltaStar` is large/infinite, only that it exceeds THIS budget -- a
value of 1.5 at `delta=1` is an ordinary, fully solvable finite point (Case A) that the old
code intercepted and mislabeled before ever finding out. This function has never accepted a
`delta` argument since (governing prompt Section 3: "the budget delta is used only by the
outer nonlinear constraint... it is not the routine inner stopping threshold" -- the caller,
`finite_delta_outer.jl`'s `cb_F!`/`cb_G!`, is the ONLY place `delta` is still used, to form
`c(theta)=DeltaStar(theta)/delta` from a GENUINELY-solved `FiniteSolved.Delta`).

**CORRECTED 2026-07-28 (inner-solver architecture-consolidation session) -- this docstring
previously overclaimed here that "the KNITRO-native mid-solve `lower_limit` bailout... is now
gated on `delta_evaluation_cap` ONLY."** That sentence was not accurate for how `obj.lower_limit`
was actually set on four of this function's five call paths, and is the documented, concrete
mechanism the 2026-07-28 anomaly session identified as a contributing cause of the
`1.510118e14` `FiniteSolved`-above-cap incident (a session reading it at face value had a
textual reason to believe the argument alone controlled the live KNITRO threshold; see
`docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md` Phase 6b and
`docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md`). As of this session there
is no `delta_evaluation_cap` ARGUMENT to this function at all: `session::MelitzInnerSession`
(`inner_session.jl`) carries `obj` (whose `lower_limit` was set from `session.policy` at
CONSTRUCTION time, by whichever sanctioned constructor built it -- `build_melitz_cc_bundle`/
`build_melitz_implicit_bundle`/`build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`,
all of which now take `policy::MelitzInnerSolvePolicy` as a mandatory keyword) and `policy`
itself; this function reads `delta_evaluation_cap = melitz_policy_cap(session.policy)` as a
LOCAL variable derived from the exact same object, never a second independently-suppliable
number. This is not merely a documentation fix -- it is a structural guarantee: there is no
remaining code path in this codebase where the live KNITRO threshold and this function's own
cap-based screening/assert could be built from two different sources. Call `solve_melitz_delta!`
(`inner_session.jl`), never this function directly, from any production or diagnostic caller.

`dual_polish_screen`/`origin_block_screen` (Phase I.5/I.3, default `false` each -- opt-in
until the Phase I.8 screen-order benchmark decides a production default): the former runs
`melitz_dual_polish_screen` from `melitz_bank_best`'s own best-lower-bound entry for
`dual_polish_steps` damped Newton steps; the latter runs `melitz_origin_block_screen`.

`on_result`, if given, is called as `on_result(theta, result)` right before returning --
a zero-risk (default `nothing`, skipped entirely) hook for a diagnostic caller to archive
`theta` alongside its classification (e.g. the session report's warm/cold no-rescue
benchmark, which needs a representative sample of thetas that produced `NumericalFailure`
during a real trajectory) without adding any collection state to this function itself.
"""
function _melitz_classified_inner_solve!(session, theta::AbstractVector;
                                        range_screen::Bool=true,
                                        matrix_free_range_screen::Bool=true,
                                        stored_dual_screen::Bool=true,
                                        dual_polish_screen::Bool=false,
                                        dual_polish_steps::Int=3,
                                        origin_block_screen::Bool=false,
                                        screen_order::Symbol=:A,
                                        warm_start_source::Symbol=:previous,
                                        on_result=nothing)::MelitzInnerResult
    screen_order in (:A, :B, :C) || throw(ArgumentError("screen_order must be :A, :B, or :C, got $screen_order"))
    # Section 3 (this session): obj/ctx/bank/cap all read off the ONE session object --
    # see this function's own docstring for why that closes the anomaly's root cause.
    obj = session.obj
    ctx = session.ctx
    bank = session.bank
    delta_evaluation_cap = melitz_policy_cap(session.policy)
    # 2026-07-26 production-port session: routed through the Melitz-owned
    # melitz_bundle_prepare_at_theta! dispatcher (cc_bundle.jl) instead of hardcoding
    # `CS.select_G_from_H`/`obj.moments!` -- the legacy bundles get the EXACT same two lines
    # via that dispatcher's generic (untyped) method; `obj::MelitzCCBundle` updates the
    # matrix-free operator in place and returns `nothing` (no dense G exists), which the
    # range screen below is guarded against.
    G_now = melitz_bundle_prepare_at_theta!(obj, theta)

    # Continuation session (2026-07-23), Section 12: per-screen call-count/timer
    # instrumentation, using the SAME exception-safe `melitz_record_seconds_outcome!`
    # convention as the inner-solve outcomes above (`profiling.jl`) -- a zero-cost no-op
    # when `MELITZ_PROFILE[]` is off. Each screen's own real work (not the trivial
    # disabled/empty-bank short-circuit) is timed, and its outcome (`:rejected`/`:passed`)
    # recorded under `:screen_range`/`:screen_stored_dual`/`:screen_origin_block`/
    # `:screen_dual_polish` -- lets a live campaign report each screen's ACTUAL cost and
    # incremental rejection rate directly (`melitz_profile_summary()`'s own `count`/
    # `total_s` columns), rather than only inferring it indirectly from wall-time deltas
    # across separate before/after campaign runs (the prior session's own Section 8/12
    # methodology).
    # 2026-07-26 production-port session (Phase 8): `melitz_range_screen` needs a genuine
    # dense `G` (an O(W*K) column-range scan), which does not exist for `MelitzCCBundle`
    # (`G_now === nothing`, from `melitz_bundle_prepare_at_theta!` above).
    #
    # 2026-07-26 closure session (Phase 7): a genuine matrix-free equivalent now exists
    # (`melitz_range_screen(op::MelitzMomentOperator)`, inner_screening.jl above) -- `O(D^2)`
    # given `op`'s already-updated state, no dense `G` materialized, fused into the SAME
    # merge sweep `melitz_update_moment_operator!` already runs (no extra O(W*D) pass).
    # Validated correct against the dense reference (D=4/D=10 synthetic, real D=20 --
    # docs/melitz_production_fast_backend_closure_2026-07-26.md Phase 7: zero mismatches
    # across 37 checked perturbed/feasible points spanning all three) and measured
    # DECISIVELY faster in isolation (real D=20/W=20,000: ~110us matrix-free vs ~33ms dense,
    # ~300x) -- the governing prompt's own bar ("enable it by default only if the measured
    # net return is positive") is clearly met, so `matrix_free_range_screen::Bool=true` by
    # default for `MelitzCCBundle`. `stored_dual_screen`/`dual_polish_screen` below remain
    # fully active for BOTH bundle types regardless (they operate purely through the generic
    # functor `obj(x,g;h=H)`, already bundle-agnostic via duck typing, no dense G needed).
    if range_screen && G_now !== nothing
        t0_range = time_ns()
        cert = melitz_range_screen(G_now)
        MELITZ_PRODUCTION_DENSE_SCREEN_CALLS[] += 1
        melitz_record_seconds_outcome!(:screen_range, cert === nothing ? :passed : :rejected,
            (time_ns() - t0_range) / 1e9)
        if cert !== nothing
            on_result !== nothing && on_result(theta, cert)
            return cert
        end
    elseif matrix_free_range_screen && obj isa MelitzCCBundle
        t0_range = time_ns()
        cert = melitz_range_screen(obj.op)
        MELITZ_MATRIX_FREE_RANGE_SCREEN_CALLS[] += 1
        melitz_record_seconds_outcome!(:screen_range, cert === nothing ? :passed : :rejected,
            (time_ns() - t0_range) / 1e9)
        if cert !== nothing
            on_result !== nothing && on_result(theta, cert)
            return cert
        end
    end

    function try_stored_dual()
        (!stored_dual_screen || isempty(bank.entries)) && return nothing
        t0_sd = time_ns()
        lb, x_best = melitz_bank_best(obj, bank)
        # Governing prompt Section 3: threshold is `delta_evaluation_cap`, never the outer
        # budget `delta` -- "do not reject merely because a stored dual proves
        # DeltaStar>delta" (that is now an ordinary Case A point, solved fully below).
        result = lb > delta_evaluation_cap ?
            AboveEvaluationCap(lb, :stored_dual, copy(x_best), 0.0, 0) : nothing
        melitz_record_seconds_outcome!(:screen_stored_dual, result === nothing ? :passed : :rejected,
            (time_ns() - t0_sd) / 1e9)
        result
    end
    function try_dual_polish()
        (!dual_polish_screen || isempty(bank.entries)) && return nothing
        t0_dp = time_ns()
        _, best_x = melitz_bank_best(obj, bank)
        result = best_x === nothing ? nothing : melitz_dual_polish_screen(obj, best_x;
            delta_evaluation_cap=delta_evaluation_cap, max_steps=dual_polish_steps)
        melitz_record_seconds_outcome!(:screen_dual_polish, result === nothing ? :passed : :rejected,
            (time_ns() - t0_dp) / 1e9)
        result
    end
    function try_origin_block()
        origin_block_screen || return nothing
        t0_ob = time_ns()
        result = melitz_origin_block_screen(theta, ctx, obj)
        melitz_record_seconds_outcome!(:screen_origin_block, result === nothing ? :passed : :rejected,
            (time_ns() - t0_ob) / 1e9)
        result
    end

    steps = screen_order == :A ? (try_stored_dual, try_origin_block, try_dual_polish) :
            screen_order == :B ? (try_origin_block, try_stored_dual, try_dual_polish) :
            (try_stored_dual, try_dual_polish, try_origin_block)   # :C
    for step in steps
        result = step()
        if result !== nothing
            on_result !== nothing && on_result(theta, result)
            return result
        end
    end

    # Main prompt Section 11/this session's Phase I.1: reset the crossing flag
    # UNCONDITIONALLY before every attempt -- `obj` (a `PsiObjectiveBundleImplicit`)
    # persists across the whole outer trajectory, so a stale `true` left over from an
    # earlier, unrelated inner solve must never leak into this one's classification.
    obj.threshold_crossed[] = false
    # Continuation session (2026-07-23), Section 5: apply the requested warm-start policy
    # (default `:previous`, a no-op reproducing all pre-existing behavior) immediately before
    # the one real KNITRO attempt -- never a routine cold retry, addendum Section 1's policy
    # is otherwise unchanged.
    @melitz_profile :fc_warm_start_resolve melitz_resolve_warm_start!(obj, bank, theta, warm_start_source)
    t_solve_start_ns = time_ns()   # evaluation-cap-correction session: base for crossing_time_s below
    # 2026-07-26 production-port session: melitz_bundle_inner_solve! (cc_bundle.jl) dispatches
    # to CS.inner_loop_internal for the legacy bundles (generic method) or the Melitz-owned
    # melitz_cc_inner_loop_internal! for MelitzCCBundle -- theta was already applied to the
    # bundle/operator by melitz_bundle_prepare_at_theta! above, so this call does not repeat
    # that work for the matrix-free path (see that dispatcher's own docstring).
    objSol, x, nStatus = melitz_bundle_inner_solve!(obj, theta)
    # 2026-07-28 anomaly-hardening (docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md):
    # mirror the ALREADY-established `cc_algo/inner_loop_functions.jl` acceptance rule
    # (`nStatus == 0 || (nStatus in [-100,-101,-103] && objSol >= obj.lower_limit)`), which
    # Melitz's own classifier had silently dropped the `objSol >= obj.lower_limit` half of --
    # an approximate/stalled status code (-100/-101/-103, NOT a genuine KKT-optimal 0) whose
    # raw objective already violates the configured cap must not be treated as a clean
    # accepted solve. Confirmed live: a genuinely InfiniteDeltaCertified real-D20 point
    # produced nStatus=-103 with a runaway dual (||x||~2e17) that the OLD unconditional
    # `nStatus in (0,-100,-101,-103)` rule accepted outright, reporting a bogus
    # `FiniteSolved(Delta=1.51e14)`. This gate alone does not fully close that anomaly (it
    # only fires when `obj.lower_limit` is itself finite/active -- see the output-side
    # invariant a few lines below for the case where the cap never reached `obj.lower_limit`
    # at all), but it is real, cheap, precedented hardening and must not regress.
    accepted = nStatus == 0 || (nStatus in (-100, -101, -103) && objSol >= obj.lower_limit)
    if !accepted
        if obj.threshold_crossed[]
            # The KNITRO-native `lower_limit` early-stop branch (`cc_algo/PsiObjectiveBundle.jl`)
            # fired during this attempt: `threshold_crossing_bound[]` is `-f` at the crossing
            # dual iterate, a valid `Delta(theta)` lower bound by the SAME unconditional weak-
            # duality argument as the stored-dual screen (file header above) -- this is a
            # CERTIFICATE, not an unresolved numerical failure, so it must not enter the
            # numerical-failure counter (main prompt Section 11: "must not trigger another
            # start; must not enter a failed-solve counter"). The crossing dual is also a valid
            # (if not necessarily optimal) finite dual point, so it is eligible for the
            # screening bank on the same weak-duality basis as any verified solve.
            # `obj.lower_limit` (set by the CALLER, `build_melitz_implicit_bundle`, from
            # `delta_evaluation_cap`, NOT `delta` -- see that function's docstring) is what
            # made this branch a genuine `delta_evaluation_cap` certificate, not a budget one.
            x_crossing = obj.threshold_crossing_x[]
            melitz_dual_bank_insert!(bank, x_crossing; theta=theta)
            crossing_time_s = (obj.threshold_crossing_time_ns[] - t_solve_start_ns) / 1e9
            result = AboveEvaluationCap(obj.threshold_crossing_bound[], :live_dual_threshold,
                                         copy(x_crossing), crossing_time_s, -1)
            on_result !== nothing && on_result(theta, result)
            return result
        end
        # Phase I.6 (main prompt Section 6): the current code sets `obj.x .= NaN` on a
        # failure, but the RAW value `x` returned here (KNITRO's own `KN_get_solution` at
        # whatever iterate it stopped on) is NOT that poisoned cache field -- it is a real
        # (if unconverged) dual point. Any FINITE such point remains valid for a future
        # lower-bound evaluation by the same unconditional weak-duality argument (file
        # header) regardless of why the solve failed; `melitz_dual_bank_insert!` itself
        # refuses a non-finite vector, so this is always safe to attempt.
        #
        # Governing prompt Section 7: this branch is ALSO reached when a genuine routine
        # inner time/iteration cap (KNITRO's own `maxtime_real`/`maxit` inner-solve options,
        # `nStatus` outside the accepted set) is hit WITHOUT the threshold above having fired
        # first -- correctly `NumericalFailure`, never an invented `AboveEvaluationCap`/
        # `InfiniteDeltaCertified` value: "timeout is NumericalFailure unless a valid
        # lower-bound or infeasibility certificate already exists" (checked above, in order).
        melitz_dual_bank_insert!(bank, x; theta=theta)
        result = NumericalFailure(Int(nStatus))
        on_result !== nothing && on_result(theta, result)
        return result
    end
    localc = zeros(1)
    obj(x, constr=localc)
    Delta_theta = localc[1] / 1e10
    # 2026-07-28 anomaly-hardening (docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md):
    # `FiniteSolved`'s OWN docstring already documents the intended invariant -- "only
    # delta_evaluation_cap ... CAN abort a solve before this point" -- i.e. reaching this
    # branch at all is supposed to mean `Delta_theta` is within the cap; only the OUTER
    # budget `delta` (a different, separate quantity) is allowed to be exceeded here. Live
    # confirmed: a genuinely InfiniteDeltaCertified real-D20 point (independently proven by
    # `melitz_origin_block_screen`) produced `Delta_theta=1.51e14` under `delta_evaluation_cap
    # =10.0` because the requested cap never propagated into `obj.lower_limit` at bundle-
    # construction time (`build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`'s
    # own `inner_solve_config=nothing` default) -- a silent configuration-propagation bug, not
    # a screen or classification-logic defect. This assert converts that failure mode from a
    # silently-mislabeled `FiniteSolved` into a loud, unambiguous, always-on error, exactly
    # mirroring `build_melitz_cc_bundle`'s own no-default `lower_limit` precedent (cc_bundle.jl)
    # and `solve_melitz_finite_delta_bound`'s own `@assert isfinite(obj.lower_limit)`.
    @assert Delta_theta <= delta_evaluation_cap + max(1e-6, 1e-6 * abs(delta_evaluation_cap)) (
        "melitz_classified_inner_solve: INVARIANT VIOLATION -- about to return FiniteSolved " *
        "with Delta=$(Delta_theta) > delta_evaluation_cap=$(delta_evaluation_cap) " *
        "(obj.lower_limit=$(obj.lower_limit), nStatus=$(nStatus)). A capped inner solve must " *
        "never report a genuine FiniteSolved value above its own cap -- either the cap never " *
        "propagated into obj.lower_limit at bundle construction (check the bundle builder's " *
        "inner_solve_config), or this point is not actually finite (check " *
        "melitz_origin_block_screen). See " *
        "docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md.")
    x_copy = collect(Float64.(x))
    melitz_dual_bank_insert!(bank, x_copy; theta=theta)
    result = FiniteSolved(Delta_theta, x_copy, Int(nStatus))
    on_result !== nothing && on_result(theta, result)
    return result
end
