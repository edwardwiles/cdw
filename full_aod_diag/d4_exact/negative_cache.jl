# ============================================================================
# Typed negative-cache entries + confirm-then-cache (Policy B) for the exact-
# point cache. Negative-cache audit deliverable (docs/fullA_negative_cache_audit.md,
# integration/fullA-negative-cache-audit).
#
# BACKGROUND: oracle.jl's `is_cacheable_result` (unchanged by this file) only
# ever caches a genuine feasible solve (inner_status in FEASIBLE_CODES) or an
# exact screen certificate (inner_status <= -9000, from infeasibility_screen.jl
# / fast_range_screen.jl). A raw, uncertified inner-solver failure (KNITRO
# -300 or otherwise) is NEVER written to the positive `SafeExactCache` -- this
# file does not change that.
#
# WHAT THIS FILE ADDS: a SEPARATE, opt-in negative cache
# (`SafeNegativeCache{K}`, same lock-guarded design as `SafeExactCache{K}`)
# that stores a raw solver failure ONLY after it has been CONFIRMED by a
# second, materially different start returning a compatible failure code.
# This is "Policy B" from the audit brief -- see docs/fullA_negative_cache_audit.md
# Section 5 for why Policy A (cache on first failure) was rejected and Policy C
# (status quo, never cache) was judged unnecessarily conservative once
# confirmation is required.
#
# WHY confirmation, not first-failure: cc_algo/PsiObjectiveBundle.jl's three
# ObjectiveBundle variants (Explicit/Implicit/Delta) all return
# `-KNITRO.KN_INFINITY` (forcing KNITRO's native KN_RC_UNBOUNDED=-300) the
# instant ANY evaluated (zeta,lambda) point -- accepted or a rejected line-
# search trial -- has objective f <= lower_limit (=-50 at every real call
# site: context.jl, context_real_d20.jl, ad_benchmark/setup_context.jl,
# ccOuter.jl, ccInner.jl; KNITRO's own `objrange` is left at its inert 1e+20
# default everywhere). Because f(zeta,lambda) = mean(Psi(arg0)) + zeta is
# globally convex in (zeta,lambda) over its box-unconstrained domain (Psi! is
# convex by construction, arg0 is affine in x, zeta/lambda have no upper or
# generally-applicable lower bound), inf_x f(x) <= f(x) trivially for EVERY
# evaluated x -- so a CONFIRMED f<=-50 crossing is an unconditional,
# start-independent proof that inf f <= -50, i.e. genuine primal infeasibility
# (never observed finite Delta_dual in this codebase's real data exceeds
# ~0.35, nowhere near 50). What is NOT start-independent is whether a given
# trajectory reaches f<=-50 within its iteration/time budget at all -- a weak
# start might terminate a different way (maxit-exceeded, or a false
# "converged" report) without ever tripping the clip. Confirmation with a
# materially different start is the guard against exactly that residual risk;
# see the audit doc for the direct empirical evidence (phaseF_primal_feasibility_lp.jl:
# 10/10 KNITRO -300 failures independently HiGHS-LP-certified infeasible, 0
# false; check_multistart_warmstart_rescue.jl: 8/8 failures survived a smooth
# 20-step warm-started path from a known-feasible point; this audit's own live
# D=20/delta=5 multi-start test, see negcache_audit_experiment.jl results).
# ============================================================================

using Dates: DateTime, now

const FEASIBLE_CODES_NC = (0, -100, -101, -103)   # mirrors FEASIBLE_CODES / oracle.jl's is_cacheable_result's positive set

abstract type CachedInnerResult end

"A genuine feasible inner solve. Never a caching decision by itself -- just the
classification label; storage still goes through oracle.jl's own SafeExactCache."
struct SolvedInnerResult <: CachedInnerResult
    inner_status::Int
end

"An exact mathematical infeasibility certificate from a screen (pairwise/
witness/winner-scan/envelope/winning-range/moment-range safety net --
inner_status <= -9000, see infeasibility_screen.jl/fast_range_screen.jl's
`infeasible_result`/`infeasible_result_ranged`). Independent of solver
tolerances/version BY CONSTRUCTION -- screens never call KNITRO."
struct CertifiedInfeasibleResult <: CachedInnerResult
    inner_status::Int
    screen_status::Symbol
end

"""
    ConfirmedNegativeResult

A raw solver failure (typically KNITRO -300) that has been CONFIRMED: a
second solve attempt from a materially different starting dual (different
warm-start LABEL, not just a repeat of the same start) at the IDENTICAL exact
point also returned a compatible non-feasible status. Only these are ever
eligible for negative caching under Policy B.

Fields carry everything Section 6/7 of the audit brief asks a confirmed
negative entry to record.
"""
struct ConfirmedNegativeResult <: CachedInnerResult
    first_status::Int
    first_label::Symbol
    confirm_status::Int
    confirm_label::Symbol
    diagnostic::NamedTuple      # e.g. (first_Delta=, confirm_Delta=, first_wall=, confirm_wall=, confirm_dual_norm=)
    code_version::String        # git short-sha at time of caching (task 7: confirmed-numerical negatives may be solver/version-sensitive)
    timestamp::DateTime
end

"""
    TransientFailureResult

A raw solver failure observed exactly once (no confirmation attempt has run
yet, or the confirmation attempt itself has not completed). MUST NEVER be
served from a cache as if it were authoritative -- this type exists so that
outcome is structurally distinguishable from `ConfirmedNegativeResult` at the
type level, not just by convention. Callers must re-solve on every repeat
occurrence of a `TransientFailureResult`.
"""
struct TransientFailureResult <: CachedInnerResult
    inner_status::Int
end

"""
    classify_result(result, screen_status=nothing) -> CachedInnerResult

Single source of truth for turning an `evaluate_fullA*`-shaped NamedTuple (as
returned by `evaluate_fullA`/`evaluate_fullA_fast`/`evaluate_fullA_screened_ranged`)
into one of the four typed outcomes above. Does not itself decide
cacheability -- see `confirm_and_maybe_cache_negative!` for that.
"""
function classify_result(result::NamedTuple; screen_status::Union{Nothing,Symbol} = nothing)
    s = result.inner_status
    if s in FEASIBLE_CODES_NC
        return SolvedInnerResult(s)
    elseif s <= -9000
        return CertifiedInfeasibleResult(s, screen_status === nothing ? :unknown_screen : screen_status)
    else
        return TransientFailureResult(s)
    end
end

"""
    compatible_failure(first_status, confirm_status) -> Bool

Two raw failure statuses "agree" for confirmation purposes if they are
literally the same KNITRO code (the common and expected case: both -300), OR
both fall in KNITRO's documented family of definitive non-recoverable
outcomes for this convex problem (-300 KN_RC_UNBOUNDED, -301
KN_RC_UNBOUNDED_OR_INFEAS). Does NOT treat a resource-limit code
(KN_RC_ITER_LIMIT_FEAS/-400s, KN_RC_TIME_LIMIT/-400s family, or any status
outside this short list) as compatible with a -300 -- a maxit/maxtime
exhaustion is a DIFFERENT, non-cacheable outcome (see the taxonomy in the
audit doc's Section 1: RESOURCE_LIMIT is its own bucket, never promoted to
ConfirmedNegativeResult) and must fall through to remaining a
TransientFailureResult so the caller retries with a larger budget rather than
trusting an inconclusive timeout as a mathematical certificate.
"""
function compatible_failure(first_status::Int, confirm_status::Int)::Bool
    # Remediation task Part E (finding F11): renamed from `unbounded_family` -- (-300, -301) are
    # KNITRO INFEASIBILITY codes (KN_RC_INFEASIBLE / KN_RC_INFEAS_XTOL), not "unbounded" codes;
    # the old name recreated the exact -300-vocabulary confusion this codebase's own methodology
    # doc warns about elsewhere. The policy itself (two independent-start confirmed failures with
    # matching status) is unchanged.
    infeasible_family = (-300, -301)
    return first_status in infeasible_family && confirm_status in infeasible_family
end

"""
    SafeNegativeCache{K}

Lock-guarded negative-only cache, structurally separate from oracle.jl's
`SafeExactCache{K}` (which only ever holds `SolvedInnerResult`/
`CertifiedInfeasibleResult`-classified entries). Kept as its own object
(never merged into the positive cache's `Dict`) so a bug in negative-cache
wiring cannot corrupt or shadow the existing, already-validated positive
cache -- a lookup miss here always falls through to a real solve, by
construction (see `negcache_lookup`).
"""
struct SafeNegativeCache{K}
    d::Dict{K, ConfirmedNegativeResult}
    lock::ReentrantLock
end
SafeNegativeCache{K}() where {K} = SafeNegativeCache{K}(Dict{K, ConfirmedNegativeResult}(), ReentrantLock())

Base.length(c::SafeNegativeCache) = lock(() -> length(c.d), c.lock)

"Lookup: returns the stored `ConfirmedNegativeResult` or `nothing`. Never mutates."
negcache_lookup(cache::SafeNegativeCache{K}, key::K) where {K} = lock(() -> get(cache.d, key, nothing), cache.lock)

"Store: only ever called with an ALREADY-classified `ConfirmedNegativeResult` -- there is
no code path in this file that stores a `TransientFailureResult`."
negcache_store!(cache::SafeNegativeCache{K}, key::K, entry::ConfirmedNegativeResult) where {K} =
    (lock(() -> (cache.d[key] = entry), cache.lock); nothing)

"""
    negative_result_namedtuple(x_free, template, entry::ConfirmedNegativeResult) -> NamedTuple

Builds a result NamedTuple with the SAME FIELD SET as `evaluate_fullA`'s
failure branch (so downstream code that destructures by field name keeps
working identically to an uncached repeat failure), stamped with
`inner_status = entry.confirm_status` and `cache_hit = true`. `template` is
any prior full result NamedTuple from this same context (used only for its
field names/types via `merge`, not its values).
"""
function negative_result_namedtuple(x_free::AbstractVector{Float64}, template::NamedTuple, entry::ConfirmedNegativeResult)
    base = merge(template, (x_free = collect(x_free), inner_status = entry.confirm_status,
        Delta_dual = NaN, Delta_primal = NaN, cache_hit = true,
        error_reason = "negative_cache hit: confirmed $(entry.first_label)->$(entry.confirm_label) " *
                        "($(entry.first_status)/$(entry.confirm_status)) at $(entry.timestamp), code_version=$(entry.code_version)"))
    return base
end

"""
    confirm_and_maybe_cache_negative!(neg_cache, key, xf, first_result, first_label,
                                       confirm_fn; code_version) -> (result, cached::Bool)

Given a FIRST non-feasible, non-certified result (`classify_result` returned
`TransientFailureResult`), runs ONE confirmation solve via `confirm_fn()`
(caller-supplied closure that performs a solve from a materially different
start and returns an `evaluate_fullA*`-shaped NamedTuple + a Symbol label) and:
  - if the confirmation is ALSO feasible: returns the confirmation's own
    (feasible) result, caches NOTHING here (this is oracle.jl's existing
    `is_cacheable_result` positive-cache territory -- the caller is expected
    to store the feasible confirmation result in the ordinary `SafeExactCache`
    itself, exactly as run_polish_checkpointed's cold-retry already does for
    the VALUE, just not yet for the cache -- see the audit doc's confirmed
    driver-side gap).
  - if the confirmation fails COMPATIBLY (`compatible_failure`): builds and
    stores a `ConfirmedNegativeResult`, returns the confirmation result with
    `cached=true`.
  - if the confirmation fails INCOMPATIBLY (e.g. a resource-limit code) or is
    itself inconclusive: caches NOTHING, returns the confirmation result with
    `cached=false` -- the point remains a `TransientFailureResult` forever
    (every future call re-solves), which is the deliberately conservative
    fallback for anything this function cannot positively confirm.
"""
function confirm_and_maybe_cache_negative!(neg_cache::SafeNegativeCache{K}, key::K,
        xf::AbstractVector{Float64}, first_result::NamedTuple, first_label::Symbol,
        confirm_fn; code_version::String) where {K}
    confirm_result, confirm_label = confirm_fn()
    if confirm_result.inner_status in FEASIBLE_CODES_NC
        return confirm_result, false
    end
    if compatible_failure(first_result.inner_status, confirm_result.inner_status)
        diag = (first_Delta = get(first_result, :Delta_dual, NaN), confirm_Delta = get(confirm_result, :Delta_dual, NaN),
                first_status = first_result.inner_status, confirm_status = confirm_result.inner_status)
        entry = ConfirmedNegativeResult(first_result.inner_status, first_label,
                                         confirm_result.inner_status, confirm_label,
                                         diag, code_version, now())
        negcache_store!(neg_cache, key, entry)
        return confirm_result, true
    end
    return confirm_result, false
end
