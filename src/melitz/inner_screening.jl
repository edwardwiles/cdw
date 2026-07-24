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
# ============================================================================

"Abstract supertype for a classified inner-solve outcome."
abstract type MelitzInnerResult end

"Verified inner solve: divergence `Delta`, dual vector `x`, KNITRO `nStatus` (in the accepted set {0,-100,-101,-103})."
struct InnerSolved <: MelitzInnerResult
    Delta::Float64
    x::Vector{Float64}
    nStatus::Int
end

"""
Certified `Delta(theta) > delta` obtained WITHOUT a completed KNITRO inner solve (or, in
principle, from a KNITRO solve terminated early on a certified threshold -- not
implemented this session, see report Section 11/D). `lower_bound` is a valid lower bound
on `Delta(theta)`; `source` records which screen produced it (this session implements
only `:stored_dual`; `:scalar` and `:origin_block`, main prompt Sections 8.2/8.3, are
documented as future work, report Section C).
"""
struct BudgetInfeasible <: MelitzInnerResult
    lower_bound::Float64
    source::Symbol
end

"""
    MELITZ_BUDGET_INFEASIBLE_SOURCES

Every `source` symbol a `BudgetInfeasible` result may carry, for validation/testing.
`:stored_dual` (Section 3.3 above) and `:live_dual_threshold` (main prompt Section 11/this
session's Phase I.1: the KNITRO-native `lower_limit` mid-solve early stop, now correctly
classified instead of folding into `NumericalFailure` -- see
`melitz_classified_inner_solve`'s docstring below and
`cc_algo/PsiObjectiveBundle.jl`'s `threshold_crossed` fields).
"""
const MELITZ_BUDGET_INFEASIBLE_SOURCES = (:stored_dual, :live_dual_threshold)

"""
Exact finite-support separation certificate (main prompt Section 3/6): moment column
`column`'s achievable range `[lo, hi]` across all `W` draws does not contain 0, so no
probability vector over the draws can satisfy that moment -- the point is infeasible
independent of any KNITRO attempt. `kind` is currently always `:range` (this session's
only implemented certificate; the K-dimensional convex-hull LP, main prompt Sections 5-6,
is documented but not implemented, see report Section C).
"""
struct MomentInfeasible <: MelitzInnerResult
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

"No mathematical certificate was obtained: the (single, no-retry) KNITRO attempt returned a status outside the accepted set, and no screen rejected the point first."
struct NumericalFailure <: MelitzInnerResult
    nStatus::Int
end

# ============================================================================
# Section 4.1 (addendum): always-on O(W*K) range screen.
# ============================================================================

"""
    melitz_range_screen(G::AbstractMatrix; guard=0.0) -> Union{Nothing,MomentInfeasible}

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
            return MomentInfeasible(k, lo, hi, :range)
        end
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
    melitz_stored_dual_lower_bound(obj, bank) -> Float64

Evaluates every dual vector in `bank` against the CURRENTLY-loaded `obj.H` (caller must
have already run `obj.moments!` for the query `theta`) via the bare functor call
`obj(x)` (no gradient/constraint request -- the cheapest possible call). Returns the
TIGHTEST (maximum) `-f(x)` over the bank, a valid lower bound on `Delta` at this `theta`
(see file header); `-Inf` if the bank is empty.
"""
function melitz_stored_dual_lower_bound(obj, bank::MelitzDualBank)
    isempty(bank.entries) && return -Inf
    best = -Inf
    for x in bank.entries
        f = obj(x)
        lb = -f
        lb > best && (best = lb)
    end
    return best
end

# ============================================================================
# Phase I.5: cheap dual-polishing budget screen.
# ============================================================================

"""
    melitz_dual_polish_screen(obj, x0; delta, guard=1e-6, max_steps=3, damping=1e-6,
        max_backtrack=4) -> Union{Nothing,BudgetInfeasible}

A small, fixed number of damped Newton steps on the CANONICAL exact dual functor in
`(zeta,lambda)`-space -- `obj(x, g; h=H)` returns the raw objective `f` and fills the exact
gradient `g` and exact Hessian `H` (the SAME functor the inner KNITRO solve itself uses,
`cc_algo/PsiObjectiveBundle.jl`; no separate/approximate objective is introduced), starting
from `x0` (typically the stored-dual bank's own best-lower-bound entry).

No convergence claim is made or needed: by the SAME unconditional weak-duality argument as
`melitz_stored_dual_lower_bound` (file header above), EVERY finite dual iterate visited --
including `x0` itself, before any step is taken -- gives a valid `Delta(theta)` lower bound
`-f(x)`. The moment ANY visited iterate's bound exceeds `delta+guard`, this function returns
immediately with a certified `BudgetInfeasible(lb, :dual_polish)`; if `max_steps` damped
Newton steps complete without a certified rejection, returns `nothing` (not a claim of
feasibility -- merely "this screen did not reject").

The line search is safeguarded per main prompt Section 12: a candidate step is accepted
only if it is FINITE and does not increase the raw objective (`f_try <= f`, i.e. does not
WORSEN the lower bound) -- halved up to `max_backtrack` times, else the polish stops (not
an error; the caller proceeds to a real KNITRO attempt).
"""
function melitz_dual_polish_screen(obj, x0::AbstractVector; delta::Real, guard::Real=1e-6,
                                    max_steps::Int=3, damping::Real=1e-6, max_backtrack::Int=4)
    n = length(x0)
    x = collect(Float64.(x0))
    all(isfinite, x) || return nothing
    g = zeros(n)
    H = zeros(n, n)
    f = obj(x, g; h=H)
    isfinite(f) || return nothing
    lb = -f
    lb > delta + guard && return BudgetInfeasible(lb, :dual_polish)

    for _ in 1:max_steps
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
        lb > delta + guard && return BudgetInfeasible(lb, :dual_polish)
    end
    return nothing
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
empty.
"""
function melitz_bank_best(obj, bank::MelitzDualBank)
    isempty(bank.entries) && return (-Inf, nothing)
    best_lb = -Inf
    best_x = bank.entries[1]
    for xb in bank.entries
        lb = -obj(xb)
        if lb > best_lb
            best_lb = lb
            best_x = xb
        end
    end
    return (best_lb, best_x)
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
    melitz_classified_inner_solve(obj, theta, ctx; delta, bank, guard=1e-6,
        range_screen=true, stored_dual_screen=true, dual_polish_screen=false,
        dual_polish_steps=3, origin_block_screen=false, screen_order=:A,
        warm_start_source=:previous, on_result=nothing) -> MelitzInnerResult

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
function melitz_classified_inner_solve(obj, theta::AbstractVector, ctx;
                                        delta::Real, bank::MelitzDualBank,
                                        guard::Real=1e-6,
                                        range_screen::Bool=true,
                                        stored_dual_screen::Bool=true,
                                        dual_polish_screen::Bool=false,
                                        dual_polish_steps::Int=3,
                                        origin_block_screen::Bool=false,
                                        screen_order::Symbol=:A,
                                        warm_start_source::Symbol=:previous,
                                        on_result=nothing)::MelitzInnerResult
    screen_order in (:A, :B, :C) || throw(ArgumentError("screen_order must be :A, :B, or :C, got $screen_order"))
    CS = CounterfactualSensitivity
    G_now = CS.select_G_from_H(obj, obj.H)
    obj.moments!(@view(obj.H[:, 1]), G_now, theta, obj.U, obj)
    obj.H[:, 2] .= 1.0

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
    if range_screen
        t0_range = time_ns()
        cert = melitz_range_screen(G_now)
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
        lb, _ = melitz_bank_best(obj, bank)
        result = lb > delta + guard ? BudgetInfeasible(lb, :stored_dual) : nothing
        melitz_record_seconds_outcome!(:screen_stored_dual, result === nothing ? :passed : :rejected,
            (time_ns() - t0_sd) / 1e9)
        result
    end
    function try_dual_polish()
        (!dual_polish_screen || isempty(bank.entries)) && return nothing
        t0_dp = time_ns()
        _, best_x = melitz_bank_best(obj, bank)
        result = best_x === nothing ? nothing : melitz_dual_polish_screen(obj, best_x; delta=delta,
            guard=guard, max_steps=dual_polish_steps)
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
    melitz_resolve_warm_start!(obj, bank, theta, warm_start_source)
    objSol, x, nStatus = CS.inner_loop_internal(obj, theta)
    accepted = nStatus in (0, -100, -101, -103)
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
            x_crossing = obj.threshold_crossing_x[]
            melitz_dual_bank_insert!(bank, x_crossing; theta=theta)
            result = BudgetInfeasible(obj.threshold_crossing_bound[], :live_dual_threshold)
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
        melitz_dual_bank_insert!(bank, x; theta=theta)
        result = NumericalFailure(Int(nStatus))
        on_result !== nothing && on_result(theta, result)
        return result
    end
    localc = zeros(1)
    obj(x, constr=localc)
    Delta_theta = localc[1] / 1e10
    x_copy = collect(Float64.(x))
    melitz_dual_bank_insert!(bank, x_copy; theta=theta)
    result = InnerSolved(Delta_theta, x_copy, Int(nStatus))
    on_result !== nothing && on_result(theta, result)
    return result
end
