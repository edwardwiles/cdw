# ============================================================================
# Cross-delta exact-point cache (allocation/cache cleanup task, §12).
#
# ROOT CAUSE (confirmed by direct code reading, not assumed -- see
# docs/fullA_allocation_cache_cleanup_handoff.md §12): `FullAEvalKey` (oracle.jl) includes
# both `δ` and `find_smallest` in its equality/hash, so `SafeExactCache{FullAEvalKey}` treats
# every δ-stage of a staged δ2->3->4->5 continuation as an entirely separate set of cache
# entries, even at the IDENTICAL x_free. Traced every use of `obj.δ`/`obj.find_smallest`
# inside the actual inner-solve path (oracle.jl/oracle_fast.jl/compressed_live.jl/
# fast_range_screen.jl/infeasibility_screen.jl): `obj.δ` is read in exactly one place per
# file, to compute a post-hoc REPORTING field (`Delta_minus_delta = Delta_dual - obj.δ`);
# `obj.find_smallest` is read in exactly one place per file, AFTER the inner KNITRO solve has
# already returned, to sign-flip the already-computed `obj.H[1,1]` into `obj.H_save`/`K_hard`
# (a screening/reporting field). Neither ever appears inside the inner CC dual solve's own
# F/G/H callbacks -- Delta_dual, zeta, lambda, θ_full, moment_resid, gravity_value are ALL
# genuinely independent of δ. This means Delta*(θ) (the thing that's actually expensive to
# compute) does not depend on the outer budget, exactly as the brief hypothesizes -- an exact
# hit from a DIFFERENT δ-stage (same x_free, same find_smallest/inner_loop_opt/mode) is a
# valid, free answer, not a stale one, PROVIDED `Delta_minus_delta` is recomputed against the
# CALLER's requested δ rather than returned verbatim from the stored δ.
#
# SCOPE DECISION: `find_smallest` is kept IN the inner key (not stripped alongside δ). Doing
# so correctly would require reconstructing the unsigned `obj.H[1,1]` from the stored signed
# `K_hard`/`H_save` (`H[1,1] = K_hard * (-1)^stored_find_smallest`) and re-flipping for the
# caller's find_smallest -- correct in principle (worked out above) but adds a second subtle
# sign-reconstruction on a field this codebase treats as screening-relevant, for a case that
# does not actually arise in production: a single `run_profile_checkpointed`/
# `run_polish_checkpointed` continuation never changes bound direction mid-run (it is either
# the upper or the lower search for its entire staged δ-schedule). Stripping ONLY δ already
# delivers the brief's actual motivating scenario (a staged δ2->3->4->5 continuation for ONE
# fixed direction) with zero sign-reconstruction risk. Cross-direction cache sharing is a
# disclosed non-goal, not attempted here.
#
# Purely ADDITIVE: does not modify FullAEvalKey, SafeExactCache, or any of the 5 call sites
# that construct a FullAEvalKey today -- `_cache_lookup`/`_cache_store!` are already generic
# (multiple-dispatch) functions (oracle.jl), so this file adds NEW methods for a NEW cache
# type; every existing SafeExactCache/Dict/Nothing caller is untouched. A caller opts in by
# constructing a `CrossDeltaExactCache` instead of a `SafeExactCache` and passing it through
# the SAME `cache=`/`exact_cache=` keyword every existing call site already accepts.
#
# Verified in test_cross_delta_cache.jl: a value stored at δ=2 and looked up at δ=5 (same
# x_free, same find_smallest) returns Delta_dual/θ_full/zeta/lambda/moment_resid IDENTICAL to
# a fresh direct solve at δ=5, with Delta_minus_delta correctly reflecting δ=5 (not the
# stored δ=2), across BOTH scenario_result (accept + reject, i.e. Delta_dual<=delta`/`>).
# ============================================================================

"""
Inner-problem cache key (task §12): everything Delta*(θ) actually depends on -- x_free, the bound
direction (kept in-key, see module docstring's scope decision), the inner-solve numerical
formulation, mode, and (AUD-08 postmerge-correctness fix) the context fingerprint. Deliberately
excludes δ.

Without ctx_fingerprint here, this key would reintroduce exactly the cross-context aliasing risk
AUD-08 fixed on FullAEvalKey: two DIFFERENT contexts that happen to agree on
x_free/find_smallest/inner_loop_opt/mode (e.g. differing only in draw set) would alias in this
cache even though FullAEvalKey itself now prevents that. The module's own documented lifetime
discipline ("never share across a genuinely different draw-set/... scope") makes this a
should-never-happen case in the intended one-ctx-per-staged-continuation usage
(staged_delta5.jl's reuse_context=true), but the AUD-08 fix's whole point is not to rely on
caller discipline alone for this.
"""
struct FullAInnerKey
    x_free::Vector{Float64}
    find_smallest::Bool
    inner_loop_opt::String
    mode::Symbol
    ctx_fingerprint::String
end
Base.:(==)(a::FullAInnerKey, b::FullAInnerKey) = a.x_free == b.x_free &&
    a.find_smallest == b.find_smallest && a.inner_loop_opt == b.inner_loop_opt && a.mode == b.mode &&
    a.ctx_fingerprint == b.ctx_fingerprint
Base.hash(k::FullAInnerKey, h::UInt) = hash((k.x_free, k.find_smallest, k.inner_loop_opt, k.mode, k.ctx_fingerprint), h)

_inner_key(key::FullAEvalKey) = FullAInnerKey(key.x_free, key.find_smallest, key.inner_loop_opt, key.mode, key.ctx_fingerprint)

"""
    CrossDeltaExactCache

Lock-guarded exact-point cache keyed on `FullAInnerKey` (δ stripped): a hit from ANY prior
δ-stage of the SAME staged continuation (same x_free/find_smallest/inner_loop_opt/mode) is
served without a re-solve, with `Delta_minus_delta` patched to reflect the CALLER's current
δ. Construct ONE instance per staged continuation (same lifetime discipline as
`oracle_cache_for`'s `SafeExactCache` -- never share across a genuinely different
draw-set/(method,bound-direction) scope) and thread it through every stage via the same
`cache=`/`exact_cache=` keyword every existing call site already accepts.

Counters (Phase 2 gate, task §2B): observed at the two dispatch points every existing call
site already goes through (`_cache_lookup`/`_cache_store!`), so no production call site needed
to change. `n_hit_verified`/`n_hit_infeasible` classify hits by the STORED result's own class
(only `VerifiedSolved`/`ExactInfeasible` are ever stored, per `is_cacheable_result` -- enforced
at the call site, not by this struct). `n_store_skipped_unverified` is NOT tracked here: the
`is_cacheable_result(result) && _cache_store!(...)` gate that filters unverified/failed results
lives at each of the (several) shared call sites in `oracle.jl`/`oracle_fast.jl`, common to
EVERY cache backend (`Dict`/`SafeExactCache`/`CrossDeltaExactCache`); adding a counter there
would mean touching already-audited, already-passing shared code for an observability nicety.
Instead: `n_lookups - n_hit_verified - n_hit_infeasible - n_miss_stored_after` (see below) is
zero by construction, and the per-stage `n_eval` in `staged_delta5.jl`'s summary already gives
the denominator needed to derive "how many evals this stage did NOT produce a cacheable
result" (n_eval minus the growth in cache size). `n_context_mismatch` is likewise not a
separate code path: `ctx_fingerprint` is part of `FullAInnerKey`, so a context mismatch is
simply a different key -- i.e. a normal miss under the intended one-ctx-per-continuation usage
this cache is scoped to (see struct docstring). Counted as `n_miss` like any other miss; the
distinguishing evidence for "no cross-context aliasing occurred" is the AUD-08 key design
itself (see `docs/fullA_independent_audit_remediation.md`), not a runtime counter.
"""
mutable struct CrossDeltaExactCache
    d::Dict{FullAInnerKey, NamedTuple}
    lock::ReentrantLock
    n_lookups::Int
    n_hit_verified::Int
    n_hit_infeasible::Int
    n_miss::Int
    n_store::Int
end
CrossDeltaExactCache() = CrossDeltaExactCache(Dict{FullAInnerKey, NamedTuple}(), ReentrantLock(), 0, 0, 0, 0, 0)
Base.length(c::CrossDeltaExactCache) = lock(() -> length(c.d), c.lock)

"Snapshot of the counters, for reporting -- not itself locked (read after the run, not concurrently)."
cache_counters(c::CrossDeltaExactCache) = (lookups = c.n_lookups, hit_verified = c.n_hit_verified,
    hit_infeasible = c.n_hit_infeasible, miss = c.n_miss, store = c.n_store,
    hit_total = c.n_hit_verified + c.n_hit_infeasible)

function _cache_lookup(cache::CrossDeltaExactCache, key::FullAEvalKey)
    hit = lock(cache.lock) do
        cache.n_lookups += 1
        v = get(cache.d, _inner_key(key), nothing)
        if v !== nothing
            if v.inner_status in (0, -100, -101, -103)
                cache.n_hit_verified += 1
            else
                cache.n_hit_infeasible += 1
            end
        else
            cache.n_miss += 1
        end
        v
    end
    hit === nothing && return nothing
    # Delta_dual/θ_full/zeta/lambda/moment_resid/gravity_value are δ-independent (module
    # docstring) -- returned verbatim. Delta_minus_delta is NOT -- patched to the CALLER's δ,
    # not the stage that originally populated this entry.
    return merge(hit, (Delta_minus_delta = hit.Delta_dual - key.δ,))
end

function _cache_store!(cache::CrossDeltaExactCache, key::FullAEvalKey, result)
    lock(cache.lock) do
        cache.d[_inner_key(key)] = result
        cache.n_store += 1
    end
    return nothing
end
