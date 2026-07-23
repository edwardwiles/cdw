# Complete-state cache: shadow-mode measurement — 2026-07-23

Part III.2 of the follow-up brief. Measures the ONE hit class Part III.1's audit identified as
both unhandled and reachable without persistence (line-search/solver-restoration revisit within a
single process), on real D=20/W=80000/L=50 CM trajectories, rather than continuing to reason about
it abstractly.

## Method

Added `shadow_stats::Union{Nothing,Dict{Symbol,Any}}=nothing` to `run_cm_upper_checkpointed`
(`cm_checkpoint.jl`) -- opt-in, additive-only instrumentation. `nothing` (the default, every
existing call site) is exactly zero behavioral/performance cost. When a caller passes a
pre-populated stats dict, `cb_F!`/`cb_G!` record the SAME point-fingerprint scheme
(`hash(round.(w, digits=12))`) the design doc's own `outer_point_key` uses, on every call, purely
as a counter -- never read to change behavior. Classifies every `cb_G!` call into exactly one of:
- **same-as-last-F**: the point matches `last_F_state[]` exactly -- already handled for free, no
  cache needed.
- **could-have-hit-cache**: the point does NOT match `last_F_state[]`, but WAS solved by some
  EARLIER `cb_F!` call in this same process -- the one class a complete-state cache could add value
  for without persistence.
- **genuine-miss**: the point has never been seen before in this process -- no cache helps.

Ran three real, independent trajectories (`cm_cache_shadow_measurement.jl`), same problem instance
as the real production campaign (W=80000, L=50, `:orthonormal` contrasts, `pseudorandom` draws,
seed 20260719), starting from the calibration point, `cm_gradient_backend=:reference`,
`maxtime_real=300s` each, at `delta ∈ {0.1, 1.0, 2.0}` -- exactly the three deltas the brief asks
for. Run on cores disjoint from all other active work on the machine at the time (verified via
`ps`/`taskset` immediately before launch; two other real jobs were active,
`sensitivity_w100k_sobol` on 20-39/60-79 and 3 unrestricted-affinity processes from other sessions
-- system load average was ~39/208 cores at launch, comfortable headroom).

## Results

| delta | wall | cb_F! calls | unique F points | REPEAT F-calls | cb_G! calls | same-as-last-F | could-have-hit-cache | genuine-miss |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 387.6s | 6 | 6 | **0** | 1 | 1 (100%) | **0 (0%)** | 0 |
| 1.0 | 335.8s | 8 | 8 | **0** | 8 | 8 (100%) | **0 (0%)** | 0 |
| 2.0 | 452.7s | 5 | 5 | **0** | 5 | 5 (100%) | **0 (0%)** | 0 |

**Zero exact-point revisits at any delta.** Every single `cb_F!` call in all three trajectories
solved a point never seen before in that process (0 repeat F-calls out of 19 total `cb_F!` calls
across the three runs). Every single `cb_G!` call was either the already-free same-as-last-F case
(100% at every delta) or -- trivially, since there were zero F-repeats -- could not possibly have
hit a complete-state cache, because the cache's only reachable hit class (Part III.1) requires an
EARLIER `cb_F!` to have solved the exact same point, and that never happened.

This confirms Part III.1's analytical prediction (interior-point/barrier algorithms do not
naturally revisit exact prior iterates the way trust-region rejection/retry does) with real,
measured data rather than leaving it as a hypothesis.

## What this does NOT rule out

- Only `:reference` backend was measured (the shadow instrumentation is backend-agnostic by
  construction -- it counts point keys, not gradient values -- so this is not expected to differ
  under `:cplus`, but was not independently re-measured this session).
- Only 300s windows (5-8 outer evaluations each) were observed. A MUCH longer real campaign stage
  (the actual production stages ran 1-19 evaluations over up to ~19 hours per the completed
  campaign's own supervisor logs) could in principle behave differently over a longer horizon,
  though there is no structural reason to expect revisits to become MORE likely as a barrier
  method's central-path iterates continue to move monotonically toward the solution.
- This measures only the WITHIN-PROCESS hit class. Part III.1 already established that every
  CROSS-process hit class (restart, next-delta seed, cross-chain) requires persistence to be
  reachable at all, and remains unmeasured (and, per the design doc's own next-delta-seed framing,
  likely low-value even WITH persistence, since the next delta starts from a different point).

## Verdict

**Negligible measured hit rate; per the brief's own §3.2 instruction ("If the potential hit rate
is negligible or the avoidable solve wall is trivial, stop and leave the cache as a prototype. Do
not wire it merely because it is correct."), Part IV (conditional wiring) is correctly NOT
attempted this session.** The complete-state cache remains classified
`PROTOTYPE_ONLY_NEGLIGIBLE_VALUE` (see the session's final report). This is not a defect in the
prototype's own correctness (which the prior session's 32/32 D=4 tests already established) --
it is a real, measured absence of an exploitable opportunity in the actual production algorithm
configuration, consistent with (not contradicting) the process-lifecycle audit's structural
argument.
