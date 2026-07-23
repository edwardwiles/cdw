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
Bounded FIFO bank of verified inner-solve dual vectors, used ONLY as stored-dual lower
bounds (addendum Section 4.3/8.1) -- this is NOT a warm-start bank (that role is
unchanged this session: `obj.use_cached_x`/`obj.x`'s single most-recent slot, plus the
existing Section 5.1 exact-point cache in `finite_delta_outer.jl`). A verified `x` is
inserted after every accepted inner solve, oldest evicted first once `max_size` is
exceeded.
"""
mutable struct MelitzDualBank
    entries::Vector{Vector{Float64}}
    max_size::Int
end
MelitzDualBank(max_size::Int=8) = MelitzDualBank(Vector{Float64}[], max_size)

function melitz_dual_bank_insert!(bank::MelitzDualBank, x::AbstractVector)
    push!(bank.entries, collect(Float64.(x)))
    while length(bank.entries) > bank.max_size
        popfirst!(bank.entries)
    end
    return nothing
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
# Orchestration: front-loaded screens, single (no-retry) KNITRO attempt, typed result.
# ============================================================================

"""
    melitz_classified_inner_solve(obj, theta, ctx; delta, bank, guard=1e-6,
        range_screen=true, stored_dual_screen=true, on_result=nothing) -> MelitzInnerResult

Addendum Section 3/6's production order: (1) fill `obj.H`'s moment matrix at `theta`
(the same `obj.moments!` call `CounterfactualSensitivity.inner_loop_internal` would make
-- this duplicates that one call ONLY when screens do not reject and a real solve
proceeds; see the session report Section 16 for the measured cost of the duplication);
(2) the O(W*K) range screen; (3) the near-free stored-dual screen; (4) if neither rejects,
ONE KNITRO attempt (`CounterfactualSensitivity.inner_loop_internal`, which reuses the
caller's warm-start state via `obj.use_cached_x`/`obj.x`, unchanged) -- NO routine cold
retry on a numerical failure (addendum Section 1: a numerical failure with no certificate
returns `NumericalFailure` immediately; cold retry is reserved for explicit `:full_value`/
diagnostic call sites, e.g. `evaluate_melitz_delta(...; cold=true)`, which this function
does not touch).

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
                                        on_result=nothing)::MelitzInnerResult
    CS = CounterfactualSensitivity
    G_now = CS.select_G_from_H(obj, obj.H)
    obj.moments!(@view(obj.H[:, 1]), G_now, theta, obj.U, obj)
    obj.H[:, 2] .= 1.0

    if range_screen
        cert = melitz_range_screen(G_now)
        if cert !== nothing
            on_result !== nothing && on_result(theta, cert)
            return cert
        end
    end

    if stored_dual_screen && !isempty(bank.entries)
        lb = melitz_stored_dual_lower_bound(obj, bank)
        if lb > delta + guard
            result = BudgetInfeasible(lb, :stored_dual)
            on_result !== nothing && on_result(theta, result)
            return result
        end
    end

    objSol, x, nStatus = CS.inner_loop_internal(obj, theta)
    accepted = nStatus in (0, -100, -101, -103)
    if !accepted
        result = NumericalFailure(Int(nStatus))
        on_result !== nothing && on_result(theta, result)
        return result
    end
    localc = zeros(1)
    obj(x, constr=localc)
    Delta_theta = localc[1] / 1e10
    x_copy = collect(Float64.(x))
    melitz_dual_bank_insert!(bank, x_copy)
    result = InnerSolved(Delta_theta, x_copy, Int(nStatus))
    on_result !== nothing && on_result(theta, result)
    return result
end
