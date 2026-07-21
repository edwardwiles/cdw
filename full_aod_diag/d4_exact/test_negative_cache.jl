# ============================================================================
# Unit tests for negative_cache.jl (negative-cache audit, Policy B). Fast,
# no KNITRO/ctx build required -- exercises classify_result/compatible_failure/
# SafeNegativeCache/confirm_and_maybe_cache_negative! directly against
# synthetic NamedTuples, matching test_safe_exact_cache.jl's own lightweight
# style for its equivalent `is_cacheable_result` unit checks (§2 there).
#
# The END-TO-END live validation (screened_eval's neg_cache lookup/store path
# against REAL D=20/W=80,000/delta=5 organic -300 points, confirmed never
# rescued by any of 7 alternative starts across 9 fresh organic candidates) is
# `negcache_audit_experiment.jl`'s own live run -- see
# docs/fullA_negative_cache_audit.md Sections 3/4 for that data; not repeated
# here since it requires an ~80s D=20 context build this fast unit-test file
# deliberately avoids, matching this investigation's existing convention of
# keeping unit tests of pure logic separate from expensive live D=20 checks.
#
# Run standalone:
#   julia --project=. full_aod_diag/d4_exact/test_negative_cache.jl
# ============================================================================
include(joinpath(@__DIR__, "negative_cache.jl"))

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1
        println("  PASS: ", name)
    else
        n_fail += 1
        println("  FAIL: ", name)
    end
end

println("== 1. classify_result ==")
feasible_r = (inner_status = 0, Delta_dual = 0.05)
check("feasible (0) classifies as SolvedInnerResult", classify_result(feasible_r) isa SolvedInnerResult)
for s in (-100, -101, -103)
    check("feasible-family ($s) classifies as SolvedInnerResult", classify_result((inner_status = s,)) isa SolvedInnerResult)
end
check("-102 (excluded from FEASIBLE_CODES by design) is NOT SolvedInnerResult",
      !(classify_result((inner_status = -102,)) isa SolvedInnerResult))
check("screen sentinel (-9001) classifies as CertifiedInfeasibleResult",
      classify_result((inner_status = -9001,); screen_status = :pairwise_certified_infeasible) isa CertifiedInfeasibleResult)
check("raw -300 (unconfirmed) classifies as TransientFailureResult", classify_result((inner_status = -300,)) isa TransientFailureResult)
check("raw -301 (unconfirmed) classifies as TransientFailureResult", classify_result((inner_status = -301,)) isa TransientFailureResult)
check("resource-limit -401 classifies as TransientFailureResult (never a certificate)", classify_result((inner_status = -401,)) isa TransientFailureResult)

println("\n== 2. compatible_failure ==")
check("-300/-300 compatible", compatible_failure(-300, -300))
check("-300/-301 compatible (same unbounded family)", compatible_failure(-300, -301))
check("-301/-301 compatible", compatible_failure(-301, -301))
check("-300/-401 (resource limit) NOT compatible", !compatible_failure(-300, -401))
check("-300/0 (feasible) NOT compatible", !compatible_failure(-300, 0))
check("-401/-401 (two resource limits) NOT compatible -- neither is in the unbounded family",
      !compatible_failure(-401, -401))

println("\n== 3. SafeNegativeCache lookup/store round-trip ==")
struct DummyKey
    x::Float64
end
neg = SafeNegativeCache{DummyKey}()
k = DummyKey(1.23)
check("fresh cache: miss", negcache_lookup(neg, k) === nothing)
entry = ConfirmedNegativeResult(-300, :warm, -300, :cold, (a = 1,), "testcommit", now())
negcache_store!(neg, k, entry)
hit = negcache_lookup(neg, k)
check("after store: hit", hit !== nothing)
check("hit is the exact stored entry", hit === entry)
check("length reflects 1 entry", length(neg) == 1)
k2 = DummyKey(4.56)
check("different key: still a miss (no cross-key contamination)", negcache_lookup(neg, k2) === nothing)

println("\n== 4. negative_result_namedtuple field-shape preservation ==")
template = (x_free = [1.0, 2.0], inner_status = 0, Delta_dual = 0.01, cache_hit = false, foo = "bar")
nt = negative_result_namedtuple([9.0, 9.0], template, entry)
check("preserves unrelated template fields (foo)", nt.foo == "bar")
check("overrides inner_status to the CONFIRM status, not the first", nt.inner_status == entry.confirm_status)
check("cache_hit = true", nt.cache_hit == true)
check("x_free updated to the queried point, not the template's", nt.x_free == [9.0, 9.0])
check("Delta_dual is NaN (never a real value for a negative-cache hit)", isnan(nt.Delta_dual))

println("\n== 5. confirm_and_maybe_cache_negative! (mocked confirm_fn -- no KNITRO) ==")
neg2 = SafeNegativeCache{DummyKey}()
k3 = DummyKey(7.0)
first_r = (inner_status = -300, Delta_dual = NaN)

# 5a: confirmation ALSO fails compatibly -> cached
r5a, cached5a = confirm_and_maybe_cache_negative!(neg2, k3, [7.0], first_r, :warm_slot,
    () -> ((inner_status = -300, Delta_dual = NaN), :cold_neutral); code_version = "test")
check("5a: compatible double-failure -> cached=true", cached5a)
check("5a: entry actually present in cache", negcache_lookup(neg2, k3) !== nothing)

# 5b: confirmation is FEASIBLE -> never cached as negative (positive cache's job instead)
neg2b = SafeNegativeCache{DummyKey}()
r5b, cached5b = confirm_and_maybe_cache_negative!(neg2b, k3, [7.0], first_r, :warm_slot,
    () -> ((inner_status = 0, Delta_dual = 0.02), :cold_neutral); code_version = "test")
check("5b: confirmation feasible -> cached=false", !cached5b)
check("5b: nothing stored", negcache_lookup(neg2b, k3) === nothing)
check("5b: returns the FEASIBLE confirmation result", r5b.inner_status == 0)

# 5c: confirmation is a resource limit -> inconclusive, never cached
neg2c = SafeNegativeCache{DummyKey}()
r5c, cached5c = confirm_and_maybe_cache_negative!(neg2c, k3, [7.0], first_r, :warm_slot,
    () -> ((inner_status = -401, Delta_dual = NaN), :cold_neutral); code_version = "test")
check("5c: incompatible (resource-limit) confirmation -> cached=false", !cached5c)
check("5c: nothing stored (deliberately conservative fallback)", negcache_lookup(neg2c, k3) === nothing)

println("\n" * "="^60)
println("RESULT: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail test(s) FAILED")
