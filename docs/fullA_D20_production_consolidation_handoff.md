# Full-A_od D=20 production consolidation, timing audit, and δ=5 improvement

Status: IN PROGRESS (written incrementally as phases complete)

Integration branch: `integration/fullA-d20-runtime-delta5`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d20-runtime-delta5`, base `98983bd`
(`diag/fullA-d4-exact`, treated as production).

## 1. Source-stream / commit integration map

| Source | Branch (worktree) | Tip | What it contributed | Status here |
|---|---|---|---|---|
| Canonical rerun | `diag/fullA-d20-canonical-rerun` (`gravity-fullA-d20-canonical-rerun`) | `98983bd` (+ uncommitted) | δ∈{0.1,1,2,5} D=20/W=80,000 frontier, pairwise-screen crash root-cause + uncommitted fix, organic infeasibility traces/checkpoints, first algorithm=2 experiment at δ=5 | Pairwise fix ported+committed here (`afe09a9`). Traces/checkpoints to be reused in Phase D from that worktree directly (not copied). Marked superseded once this branch merges. |
| Warm-start replay | `diag/fullA-d20-warmstart-replay` (`gravity-fullA-d20-warmstart-replay`) | `baec13e` (7 commits past `98983bd`, clean) | `SafeExactCache` (lock-guarded exact-point cache), KKT-proxy-scored successful-dual bank (policy P3), live δ=5 A/B (-29.5% wall, trajectory-dependent) | To be ported in Phase B. Marked superseded once merged. |
| Fast range screen integration | `integration/fullA-fast-range-screen` (`gravity-fullA-fast-range-screen-integration`) | `98983bd` (clean, = base) | Envelope/winning-range/safety-net screens, already merged into base | Base only, nothing to port. |
| QMC investigation | `diag/fullA-d20-qmc-delta1` (`gravity-fullA-d20-qmc-delta1`) | `5882c16` | Validated pseudorandom/Sobol(Cranley-Patterson-shifted)/Halton(Owen-style-digit-scrambled) draw generators for D=20/W=80,000, generator validation report, Stage A-E replicate results | To be ported as an explicit `draw_design` option in Phase A2/B2 (mid-session addendum), default unchanged. Marked superseded once merged. |

## 2. KNITRO version (resolved, no further investigation needed)

`.knitro_env.sh` in every worktree points `KNITRODIR` at `/opt/shared_sw/knitro/14.2.0`, and the
canonical-rerun handoff doc's header says "KNITRO 14.2.0" — **this is wrong for what actually
runs**. `KNITRO.jl`'s `deps/deps.jl` (`~/.julia/packages/KNITRO/LHqTK/deps/deps.jl`) hardcodes:

```julia
const libknitro = "/opt/shared_sw/knitro/13.0.1/lib/libknitro.so"
```

Confirmed live in this worktree:

```
KNITRO.jl loaded library path: /opt/shared_sw/knitro/13.0.1/lib/libknitro.so
```

regardless of `.knitro_env.sh`/module state. **13.0.1 is the real, only, always-loaded version on
this machine.** All timing comparisons in this consolidation use 13.0.1 throughout — there is no
cross-version comparison risk since nothing here has ever actually run on 14.2.0. The
warmstart-replay branch's commit `c408858` already documented this correctly; the canonical-rerun
handoff's "14.2.0" header should be read as a documentation error, not a real solver-version
difference.

## 3. Pairwise-screen crash fix (Phase A, DONE)

Commit `afe09a9` on this branch. See commit message + `test_pairwise_screen_meta_ranged.jl` for
the regression test (6/6 checks pass): constructs a genuinely pairwise-certified-infeasible D=4
point via the same adversarial recipe as `test_infeasibility_screen.jl`'s test 9, confirms
`evaluate_fullA_screened_ranged`'s `:pairwise_certified_infeasible` branch now returns
`worst_o`/`worst_d`, and that the driver's exact field-access pattern
(`(stage=:pairwise, o=screen_meta.worst_o, d=screen_meta.worst_d, ...)`) no longer throws.

## 4. Exact-point cache (Phase B, DONE)

Commit `43928e3`. `SafeExactCache` (lock-guarded, ported from
`diag/fullA-d20-warmstart-replay/safe_exact_cache.jl`) wired directly into all 5 real production
evaluation paths (`oracle.jl::evaluate_fullA`, `oracle_fast.jl::evaluate_fullA_fast`,
`compressed_live.jl::evaluate_fullA_fast_compressed`,
`infeasibility_screen.jl::evaluate_fullA_screened`,
`fast_range_screen.jl::evaluate_fullA_screened_ranged`) — not just a parallel diagnostic entry
point, the actual `cache::Union{Nothing,Dict}` slot every one of those functions already had.

New `is_cacheable_result` guard fixes a real latent bug in production's *previous* raw-Dict
behavior: it used to cache a bare `-300`/unresolved inner-solve failure unconditionally
(`cache !== nothing && (cache[key] = result)`, no status check), permanently answering that exact
point with a stale failure for the rest of the run even if a different warm start would have
succeeded. Now only `inner_status in (0,-100,-101,-103)` (genuinely solved) or `inner_status <=
-9000` (an exact screen certificate — pairwise/witness/winner-scan/envelope/winning-range/
moment-range, all inherently exact and draw-independent) get cached.

`cache` kept **untyped** (not widened to `Union{Nothing,Dict,SafeExactCache}`) after an
include-order issue: `context_real_d20.jl` includes `infeasibility_screen.jl` before `oracle.jl`,
and a type annotation referencing `SafeExactCache` needs it resolved at parse time, unlike a
function-body reference (`FullAEvalKey(...)`) which resolves at call time. Dispatch is correct
regardless via `_cache_lookup`/`_cache_store!`.

Validated (`test_safe_exact_cache.jl`, 18/18 + `test_safe_exact_cache_stress.jl`, 5/5):
same-key repeated call (cache hit, byte-identical, zero new inner solves), the cacheability
predicate across every real status/sentinel value, screen-certified-infeasible caching, a
500-trial × 8-thread synthetic lock-stress test (zero corrupted/mismatched hits, zero lost/wrong
keys, zero crashes — kept in its own standalone script, see the note below), and a real
3-concurrent-KNITRO-solve probe (no process crash regardless of KNITRO's own resource-contention
outcome).

**Real, non-obvious finding**: running the 500-trial synthetic stress test in the *same process*
right after the earlier real-KNITRO sections silently kills the Julia process (exit code 1, no
stacktrace) on this machine, while it passes cleanly (500/500, exit 0) as its own process. This
is a genuine interaction between KNITRO's internal threading/license-check state and a later
plain `Threads.@threads` use in the same process — orthogonal to `SafeExactCache`'s own
correctness, and exactly why the source branch's own `cache_threadsafety_test.jl` already
isolated its raw-vs-safe comparison into separate processes ("a real Dict data race can hard-crash
the Julia process (segfault), not just throw a catchable exception"). Resolved here by keeping
`test_safe_exact_cache_stress.jl` a separate script, matching that precedent.

**Second real finding, caught by fixing a Julia scoping bug**: `test_safe_exact_cache.jl`'s
section-3 adversarial-infeasible-point search originally left `found_infeasible`/`xf_bad` without
`global` inside a top-level `for` loop — Julia's soft-scope rules silently created loop-local
shadows, so `found_infeasible` never actually propagated `true` to the outer scope in earlier
runs (masking the bug: the `if found_infeasible` branch that uses `xf_bad` never ran). Adding
`global` correctly propagates it, which then exposed `xf_bad` having the exact same bug
(`UndefVarError`) — fixed the same way. Both fixes are in the committed test file; this was a
bug in the new test script, not in production code.

## 5. Successful-dual / KKT-scored bank (Phase B, DONE)

Commit `9f11643`. `dual_bank.jl`: small recency-bounded bank (default size 8) of
successfully-solved dual vectors, Policy P3 from `diag/fullA-d20-warmstart-replay` (candidates
`{current production last-successful slot (obj.x, if not NaN-poisoned), last-accepted, nearest-
successful-by-scaled-reduced-coordinate-distance, neutral}`, scored via `cheap_score`'s KKT-proxy
— `compressed_cc_value_grad`, no KNITRO call — lowest score wins). Wired into
`c10_d20_production_driver.jl`'s `screened_eval` (used by every `cb_F!`/`cb_G!`/`cb_newpt!` in
both `run_profile_checkpointed` and `run_polish_checkpointed`) via `use_dual_bank::Bool=true`
(`dual_bank_size::Int=8`), independently toggleable from the exact-cache's own
`use_exact_cache::Bool=true` (both wired in the same commit since they were built in the same
session, but each is revertible independently by flipping its own kwarg — see commit message).

Validated (`test_dual_bank.jl`, 9/9): history eviction at `maxsize` (recency window, not
value-based), `cheap_score`'s scoring direction (a converged dual scores at or below neutral at
its own point — confirms lower-is-better matches `select_warm_start`'s `argmin`), correct
fallback to neutral when the bank is empty and `obj.x` is NaN-poisoned, the selected candidate is
always either a genuinely-recorded success or neutral (never fabricated), and deterministic
selection at fixed bank/point state.

Not yet run against a live δ=5 trajectory with organic failures in this phase (that validation —
"same optimum/gradient, no reused failed dual, non-increased wall time" — is deferred to the
canonical post-integration A/B, Phase F/section 11, per the plan).

## 6. Draw-design port (pseudorandom / sobol_randomized / halton_scrambled) — TBD (addendum)

## 7. Timing-regression audit — TBD (Phase C)

## 8. Granular δ=1/2/5 profiling — TBD (Phase D)

## 9. Organic -300 certification — TBD (Phase D)

## 10. Staged δ=5 workflow — TBD (Phase E)

## 11. Canonical post-integration A/B — TBD (Phase F)

## 12. Final merge / rollback — TBD (Phase F)
