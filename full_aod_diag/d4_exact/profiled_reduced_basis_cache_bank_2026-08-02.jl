# ============================================================================
# Phase 12 (integration/phase12-13-runner-checkpoints-2026-08-02), item 2: separate
# exact-cache and dual-bank compatibility keys for the reduced/profiled economic basis --
# a cache/bank entry built under `:profiled_destination_scales` must never be silently
# reused for a `:full_gamma_normalized` request or vice versa.
#
# WHY a NEW type (not a new field bolted onto CMProductionEvalKey): `CMProductionEvalKey`
# (cm_exact_cache_production.jl) is a plain positional struct with exactly two non-test call
# sites (cm_checkpoint.jl:1111, cm_originzc_checkpoint.jl:761), both constructing it
# positionally with the CURRENT 11-field shape. Adding a field would be a breaking change to
# both of those PRODUCTION call sites for a cache dimension they don't need (they only ever
# build dense/full keys) -- exactly the kind of unnecessary risk to a file that's "pre-existing
# infrastructure to EXTEND, not fork". A brand-new, structurally distinct key type is the
# lower-risk, more idiomatic choice here, AND a strictly stronger safety property than a shared
# struct with a discriminator field: `SafeExactCache{CMProductionEvalKey}` and
# `SafeExactCache{ProfiledCMProductionEvalKey}` are different Julia TYPES, so a `Dict`/cache
# keyed on one can never even syntactically accept the other -- the type system enforces the
# separation the task asks for, not just a runtime tag comparison that a future caller could
# forget to check. Additionally, keying on `w_profiled` (the SHORT reduced outer coordinate)
# rather than `x_free`/`xf` (the full free-parameter vector `CMProductionEvalKey` uses) means a
# reduced-path cache entry is not even numerically comparable to a dense one -- different
# VECTOR LENGTH in the general case (`length(w_profiled) = 1 + n_retained(spec)-1` vs
# `length(x_free) = D*Ddest+3` for the dense side), so accidentally constructing one key type
# from data meant for the other throws a DimensionMismatch/BoundsError immediately downstream
# rather than silently producing a wrong-shape hit.
#
# ADDITIVE ONLY -- does not modify cm_exact_cache_production.jl / dual_bank.jl /
# cm_dual_bank_production.jl.
# ============================================================================

isdefined(Main, :SafeExactCache) || error("profiled_reduced_basis_cache_bank_2026-08-02.jl requires oracle.jl to be included first.")
isdefined(Main, :DualBank) || error("profiled_reduced_basis_cache_bank_2026-08-02.jl requires dual_bank.jl to be included first.")

# ----------------------------------------------------------------------------
# Exact-point cache
# ----------------------------------------------------------------------------

"""
    ProfiledCMProductionEvalKey

Exact-point cache key for the profiled/reduced-basis family evaluators
(`evaluate_profiled_originzc_point`/`evaluate_profiled_cmzc_point`/`evaluate_profiled_flexcm_point`/
`evaluate_profiled_frechet_point`), keyed on the REDUCED outer coordinate `w_profiled` -- never on
the full `x_free`/`xf` `CMProductionEvalKey` uses. Every field that changes the mathematical value
at a point is included, same discipline as `CMProductionEvalKey`/`FullAEvalKey`.
`economic_parameterization` is validated at construction: this key type exists ONLY for
`:profiled_destination_scales` -- passing anything else is a caller bug, not a legitimate
discriminator value (unlike a shared-struct design, there is no `:full_gamma_normalized` case
this type is ever correctly used for), so the inner constructor throws immediately rather than
silently accepting a mislabeled key.
`outer_layout_digest` (`stable_layout_digest(fctx)`, profiled_stable_layout_digest_2026-08-01.jl)
is included so a cache entry from one anchor-spec/gravity-pivot layout can never collide with one
from a structurally different layout, even at coincidentally-equal `w_profiled` values.
"""
struct ProfiledCMProductionEvalKey
    economic_parameterization::Symbol
    w_profiled::Vector{Float64}
    nu::Vector{Float64}
    delta::Float64
    find_smallest::Bool
    family_tag::Symbol
    outer_layout_digest::String
    ctx_fingerprint::String
    function ProfiledCMProductionEvalKey(economic_parameterization::Symbol, w_profiled::Vector{Float64},
            nu::Vector{Float64}, delta::Float64, find_smallest::Bool, family_tag::Symbol,
            outer_layout_digest::AbstractString, ctx_fingerprint::AbstractString)
        economic_parameterization == :profiled_destination_scales ||
            error("ProfiledCMProductionEvalKey: economic_parameterization must be :profiled_destination_scales, got :$economic_parameterization -- a :full_gamma_normalized point belongs in CMProductionEvalKey (cm_exact_cache_production.jl), not this type")
        return new(economic_parameterization, w_profiled, nu, delta, find_smallest, family_tag,
            String(outer_layout_digest), String(ctx_fingerprint))
    end
end
Base.:(==)(a::ProfiledCMProductionEvalKey, b::ProfiledCMProductionEvalKey) =
    a.economic_parameterization == b.economic_parameterization && a.w_profiled == b.w_profiled &&
    a.nu == b.nu && a.delta == b.delta && a.find_smallest == b.find_smallest &&
    a.family_tag == b.family_tag && a.outer_layout_digest == b.outer_layout_digest &&
    a.ctx_fingerprint == b.ctx_fingerprint
Base.hash(k::ProfiledCMProductionEvalKey, h::UInt) = hash((k.economic_parameterization, k.w_profiled, k.nu,
    k.delta, k.find_smallest, k.family_tag, k.outer_layout_digest, k.ctx_fingerprint), h)

"Fresh, empty exact-point cache for the profiled/reduced-basis family evaluators -- SEPARATE Julia type from cm_production_exact_cache(), one per run."
profiled_cm_production_exact_cache() = SafeExactCache{ProfiledCMProductionEvalKey}()

"Separate counters from CM_EXACT_CACHE_COUNTERS (cm_exact_cache_production.jl) -- a profiled-path hit/miss must never be silently folded into the dense path's own reported hit rate."
mutable struct ProfiledCMExactCacheCounters
    lookups::Int
    hits::Int
    misses::Int
    store_rejections::Int
end
ProfiledCMExactCacheCounters() = ProfiledCMExactCacheCounters(0, 0, 0, 0)
const PROFILED_CM_EXACT_CACHE_COUNTERS = Ref(ProfiledCMExactCacheCounters())
reset_profiled_cm_exact_cache_counters!() = (PROFILED_CM_EXACT_CACHE_COUNTERS[] = ProfiledCMExactCacheCounters())

function print_profiled_cm_exact_cache_counters(c::ProfiledCMExactCacheCounters = PROFILED_CM_EXACT_CACHE_COUNTERS[])
    println("[profiled-cm-exact-cache] lookups=", c.lookups, " hits=", c.hits, " misses=", c.misses,
            " store_rejections=", c.store_rejections,
            " hit_rate=", c.lookups == 0 ? "n/a" : round(c.hits / c.lookups, digits = 4))
end

"""
    profiled_cm_cache_lookup_or_compute!(cache, key, compute_fn) -> (base, verify)

Reduced-basis sibling of `cm_cache_lookup_or_compute!` (cm_exact_cache_production.jl). Same
discipline exactly (only cache a genuinely feasible/verified result), but type-restricted to
`Union{Nothing,ProfiledCMProductionEvalKey}` -- calling this with a `CMProductionEvalKey` (or vice
versa, calling `cm_cache_lookup_or_compute!` with a `ProfiledCMProductionEvalKey`) is a Julia
MethodError, not a silent cross-parameterization hit.
"""
function profiled_cm_cache_lookup_or_compute!(cache, key::Union{Nothing,ProfiledCMProductionEvalKey}, compute_fn)
    if cache === nothing || key === nothing
        return compute_fn()
    end
    PROFILED_CM_EXACT_CACHE_COUNTERS[].lookups += 1
    hit = _cache_lookup(cache, key)
    if hit !== nothing
        PROFILED_CM_EXACT_CACHE_COUNTERS[].hits += 1
        return hit.base, hit.verify
    end
    PROFILED_CM_EXACT_CACHE_COUNTERS[].misses += 1
    base, verify = compute_fn()
    if verify.inner_status in (0, -100, -101, -103)
        _cache_store!(cache, key, (base = base, verify = verify))
    else
        PROFILED_CM_EXACT_CACHE_COUNTERS[].store_rejections += 1
    end
    return base, verify
end

# ----------------------------------------------------------------------------
# Dual-bank warm-start
# ----------------------------------------------------------------------------

"""
    ProfiledRestrictedDualBank

Reduced-basis sibling of `RestrictedDualBank` (cm_dual_bank_production.jl), wrapping the SAME
generic `DualBank` (dual_bank.jl, reused unchanged) with an explicit, immutable
`economic_parameterization` tag fixed to `:profiled_destination_scales` at construction.
`RestrictedDualBank`/`DualBank` are already in-process, ephemeral, per-run objects (never
persisted/keyed by a generic string on disk), so there is no PRE-EXISTING collision risk in the
current codebase -- this wrapper exists so a future caller that DOES start persisting or sharing
a bank across runs cannot silently mix a reduced-basis bank into a dense-formulation query path
(or vice versa): `assert_bank_parameterization` throws immediately if ever queried against a
mismatched expectation, rather than relying on every future call site remembering to check by
convention.
"""
mutable struct ProfiledRestrictedDualBank
    economic_parameterization::Symbol
    bank::RestrictedDualBank
    function ProfiledRestrictedDualBank(maxsize::Int = 8)
        return new(:profiled_destination_scales, RestrictedDualBank(maxsize))
    end
end

"""
    assert_bank_parameterization(bank_tag::Symbol, expected::Symbol) -> Nothing

Throws unless `bank_tag == expected`. Call at every profiled-runner call site that consumes a
`ProfiledRestrictedDualBank`/`ProfiledCMProductionEvalKey` cache, passing
`expected=:profiled_destination_scales`, so a future refactor that accidentally threads a
dense-formulation bank/cache into the reduced-path runner (or vice versa) fails loudly at the
FIRST call, not via a silently-wrong warm start or cache hit.
"""
function assert_bank_parameterization(bank_tag::Symbol, expected::Symbol)
    bank_tag == expected ||
        error("assert_bank_parameterization: bank tagged :$bank_tag, expected :$expected -- refusing to use a mismatched-parameterization dual bank")
    return nothing
end

record_success_profiled!(pb::ProfiledRestrictedDualBank, eval_id::Int, w_profiled::AbstractVector{Float64}, x_solved::AbstractVector{Float64}) =
    record_success_restricted!(pb.bank, eval_id, w_profiled, x_solved)
