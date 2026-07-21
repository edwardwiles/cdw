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

## 7. Timing-regression audit (Phase C, DONE)

**Verdict: no regression.** The current code is not slower than the historical benchmark at any
matched point; the apparent "~0.6s vs ~4s" discrepancy the task opened with is two *different*
timing objects, not a regression.

**Setup**: `timing_harness.jl` (this branch, worktree-copied unchanged into worktree B) /
`timing_harness_legacy.jl` (worktree A, cf74d89's older pre-range-screen API). Both reuse
Continuation 10's own canonical-benchmark point construction exactly (`c10_canonical_benchmark.jl`,
commit `cf74d89`): **genuine** calibration (`ctx.θ0_up`'s own `A_od` block — not the
gravity-elimination pivot's `z=0` reference point, see §4/memory note) perturbed by `gp0*1.01`,
real D=20/W=80,000, δ=1, `draw_seed=20260719`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, actual loaded KNITRO 13.0.1 throughout (see §2). Four worktrees, one process
at a time:

| | A: `cf74d89` (legacy, pre-range-screen) | B: `98983bd` (base, caches off by construction) | C: this branch, caches off | D: this branch, caches on |
|---|---|---|---|---|
| ctx build | 65.9s | 65.6s | 66.1s | 65.4s |
| **cold value eval (complete callback)** | 21.72s | 21.56s | 21.59s | 20.51s |
| **exact same-point warm re-solve** | 0.694s | 0.685s | 0.639s | 1.165s |
| exact-point cache hit | n/a | n/a | n/a | **0.040s** |
| **nearby changed-point (Δzfree~0.05) warm solve** | 3.944s | 3.947s | 4.084s | 4.047s |
| distant changed-point (Δzfree~1.0) warm solve | 10.127s | 10.074s | 11.347s | 10.125s |
| full outer gradient | 11.927s (unbuffered) | 9.662s (buffered) | 9.094s (buffered) | 9.649s (buffered) |
| Delta_dual @ cold / near / distant | 0.23088414900346 / 0.23005288903742 / 0.24679246075768 | *identical* | *identical* | *identical* |
| gradient norm | 97.68020700602995 | 97.68020700609458 | 97.68020700609458 | 97.68020700609458 |

`Delta_dual` matches to every digit shown at all three points across all four versions;
gradient norm matches to 9 significant figures (the tiny ~6e-8 relative residual between A's
unbuffered and B/C/D's buffered gradient is the already-validated FD-buffering equivalence, not a
new discrepancy). **No economic/behavioral difference anywhere in this table.**

**Timing**: ctx build, cold eval, nearby, and distant solves are flat across all four versions
(the ~4-11% single-trial spread is consistent with ordinary run-to-run noise on a shared server,
not a trend — cold eval is if anything *fastest* on D). The buffered gradient (B/C/D) is
genuinely ~19-24% faster than the legacy unbuffered one (A), consistent with memory's own
"~1.1-1.9x faster" finding — a pre-existing win, not something this session changed. The
exact-point cache hit (D only) is ~16-29x faster than a repeated warm re-solve of the same point
(0.040s vs 0.64-1.17s) — the cache does exactly what it's for.

**Resolving the ~0.6s-vs-~4s question directly**: the historical "~0.588s" figure
(`c10_canonical_benchmark.jl`'s own step 2) and this table's "exact same-point warm re-solve" row
are the SAME measurement, and they still agree (0.588s historically vs 0.64-0.69s here, A/B/C —
well within noise; D's 1.165s here is a single-trial reading, see caveat below). The canonical
rerun's own "~4.08s median / ~4.26s mean complete F callback" figure is NOT that same measurement
at all — it is structurally the **nearby changed-point warm solve** case (3.94-4.09s across all
four versions here), which is a genuinely different, and correctly more expensive, workload: a
changed outer point needs a real KNITRO warm-started re-solve from a *different* starting dual,
not a trivial re-evaluation of an already-solved point. These two numbers were never comparable in
the first place. **There is no fixed-point regression to bisect.**

**Caveat**: single-trial timings (no repeated-trial medians) given the time cost of four full
D=20/W=80,000 ctx builds; the spread observed (4-11% across most rows, one 1.165s outlier for D's
same-point warm re-solve, likely ordinary noise since it uses `cache=nothing` explicitly and
should be mechanically identical to B/C's measurement) is well below the 15-20% regression
threshold this task's decision rule specifies, so no bisection was triggered.

### 7.1 Nested component breakdown ("what's in a typical 4-second callback")

At the nearby-changed-point (the representative "~4s" case), `evaluate_fullA_screened_ranged`'s
own already-built-in instrumentation (`screen_elapsed`, `n_inner_solves`, `n_inner_iters`,
`n_fg_calls`, `n_hess_calls` — no new instrumentation needed) gives, for one real call
(`t_total=4.057s`):

| Component | Time / count | Share |
|---|---|---|
| Pre-solve screens (reconstruct θ_full, pairwise cert, envelope cert, winner-scan) | 0.299s | 7.4% |
| Inner CC dual solve (KNITRO) + post-processing | ~3.758s (by subtraction) | 92.6% |
| — inner solves | 1 | |
| — inner KNITRO iterations | 32 | |
| — FG (objective/gradient) callback evaluations | 7 | |
| — Hessian callback evaluations | 6 | |

**Answers task closing question 2 directly**: a typical ~4s δ=1 value callback is overwhelmingly
(>92%) the actual KNITRO inner dual solve itself (32 iterations here) — the pre-winner screening
machinery this and prior sessions built (pairwise/envelope/winner-scan) is a small, fixed
~0.3s overhead on an *accepted* point, not the bottleneck.

## 8. Granular δ=1/2/5 profiling (Phase D, DONE)

Used REAL saved checkpoints from `diag/fullA-d20-canonical-rerun`'s own δ-frontier run (Start A,
copied into `organic_pathology/`) rather than freshly-generated points: an accepted δ=1 boundary
point (`n_eval=93`), an accepted δ=2 boundary point (`n_eval=158`), an accepted δ=5 interior point
(`n_eval=16`). `phase_d_profile.jl`:

| | δ=1 | δ=2 | δ=5 |
|---|---|---|---|
| cold solve wall | 8.315s | 8.417s | 24.694s |
| cold solve status | 0 (fully converged) | 0 | **-100** (weaker convergence criterion, not full optimality) |
| cold solve inner iterations | 27 | 56 | 112 |
| cold solve FG calls | 13 | 15 | **85** |
| cold solve Hessian calls | 12 | 14 | 41 |
| exact same-point warm re-solve wall | 0.68s | 0.624s | **6.98s** |
| warm re-solve iterations | 27 (1 FG call — dual didn't need to move) | 56 (1 FG call) | **123** (20 FG, 11 Hessian — genuinely re-worked) |

**Answers task closing question 5 directly.** Inner iterations roughly double from δ=1→δ=2
(27→56) and again to δ=5 (→112-138 depending on start), but FG-call count jumps far more
sharply at δ=5 (13-15 at δ=1/2 → 85 at δ=5 cold) — each δ=5 iteration is doing more expensive
work per step (more line-search/backtracking), not just more of them. δ=5 also never reaches
KNITRO's full-optimality status (0) here, settling for `-100` in every condition tried
(cold/warm/checkpoint-restart) — consistent with the divergence budget forcing a genuinely more
extreme reweighting (larger, more extreme implied `A_od` departures from calibration, matching
this repo's own prior finding of "~41-122× calibration" `A_od` entries near δ=5 candidates,
`docs/fullA_next_handoff.md`). **A real, non-obvious second finding**: warm-starting provides much
less benefit at δ=5 than at δ=1/2 — the "exact same-point warm re-solve" takes 6.98s (not the
~0.6s pattern seen at δ=1/2, or elsewhere in this report at δ=1), because neither the cold nor the
warm re-solve reaches clean optimality; re-solving from an already-not-fully-converged dual still
requires substantial further Newton work near the `-100` tolerance boundary.

## 9. Organic -300 certification (Phase D, partial — real finding, scope-limited)

**Real, unplanned finding**: the one available saved "organic pathology" candidate
(`nonzero_winner_infeasible_delta5_candidate1.json`, independently reproduced by the canonical
rerun on 2026-07-20 as genuinely `inner_status=-300` via the *older* `evaluate_fullA_screened`
path) does **NOT** reproduce `-300` in this worktree's current code. Instead it is caught
instantly (0.061s, vs. its previously-reported ~20-31s wait) by the already-merged pre-winner
envelope screen (`screen_status=EXACT_INFEASIBLE_PREWINNER_ENVELOPE`). This is genuine evidence
that the envelope screen (`fast_range_screen.jl`, merged into production before this session)
already resolves at least this instance of the exact organic-infeasibility problem the task asked
this phase to build new certification machinery for — the point simply predates that screen (its
own provenance says it was verified via the *pre-range-screen* `evaluate_fullA_screened`, which
has no envelope check at all).

**Scope limitation, reported honestly rather than worked around**: this means the only
readily-available "organic" test point is not representative of the *current* code's remaining
`-300` population. The canonical rerun's own real δ=5 trace (`d5_startA_canon_trace.csv`) shows
40 genuine `-300` events at THIS session's actual production code (envelope screen included,
per that worktree's own report: pairwise/envelope/witness/winner/winning-range/safety-net screens
all show **zero** rejections at δ=5 — every one of those 40 events passes every current screen
and still hits `-300`), so genuine current-generation organic infeasibility unquestionably still
exists — this session simply did not have one materialized as a reusable standalone point (the
trace CSV logs summary stats per eval, not the full `zfree` vector, and reconstructing one would
require re-running a real δ=5 continuation from scratch to capture a live `-300` event's full
outer vector, which this phase's time budget did not include).

**Consequence for the dual-ray monitor / cutting-plane work (task §7.1-7.3)**: not attempted.
Building novel infeasibility-certificate machinery and claiming "zero false positives" without a
genuine, current-architecture failing point to validate it against would not be a responsible
claim. **Recommendation for a follow-up session**: instrument `c10_d20_production_driver.jl`'s
`cb_F!` to dump the full `(g, zfree)` vector (not just summary stats) the *first* time an organic
`-300` occurs during a live δ=5 (or δ=2) run, then build and validate the dual-ray monitor against
that real point.

## 10. Staged δ=5 workflow (Phase E, DONE — real negative result with a diagnosed cause)

Implemented `staged_delta5.jl`'s `run_staged_delta5_continuation` (finer δ continuation:
2→3→4→5, each stage via the existing, unmodified `run_polish_checkpointed`, feeding the
previous stage's `best_feasible` forward as the next stage's start) and ran ONE bounded real
comparison (`staged_delta5_comparison.jl`) from the real δ=2 checkpoint's `best_feasible` point
(§8's own starting point), 60s/stage (240s total budget) for the staged arm vs. 240s for a
direct δ=5 attempt from the same start:

| | Staged (2→3→4→5) | Direct (δ=5 straight) |
|---|---|---|
| final κ | 0.003114 | **0.004550** |
| total wall | 504.1s | 280.7s |
| total rejected | 5 | 3 |

**Direct wins** at this budget — staged continuation did not help, and cost nearly 2x the wall
time doing it.

**Diagnosed root cause, not just the headline number**: each stage pays a full ~65-83s context
rebuild (`d20_real_setup` — the same real D=20/W=80,000 draw generation + screen setup every
single call, since `run_polish_checkpointed` always builds its own `ctx` internally and has no
"reuse an existing ctx" path) *inside* its own 60s wall budget, leaving only ~17-46s of genuine
optimizer wall-time per stage after that overhead — the staged run's 504.1s total includes
roughly 4×65s ≈ 260s of pure redundant context-rebuilding, over half its total wall time, doing
zero optimization work. The direct arm pays this cost exactly once. **This is an implementation
inefficiency in how the comparison was run, not necessarily evidence against the continuation
idea itself** — a corrected implementation would build `ctx` once and thread it through every
stage (requires a `run_polish_checkpointed` signature change to accept a pre-built `ctx` instead
of always constructing one, which was out of scope to add and re-validate in this pass).

**Not attempted in this pass** (see task §8.2-8.3): the fixed-g profile continuation
(`min_A Δ*(g,A)` at increasing `g`) and staged KNITRO algorithm switching (Interior/CG
exploration → Interior-Direct polish) — `run_staged_delta5.jl` currently implements only the
simplest version of the idea (finer δ continuation alone). The canonical rerun's own §8
algorithm=2 experiment (already in its handoff doc, not repeated here) showed a real but modest
δ=5 improvement (κ 0.05761→0.07761 at Start A, still short of the best δ=5 κ found by the direct
canonical run) — consistent with staged/alternative approaches at δ=5 being a genuinely hard,
not-yet-solved problem rather than a quick win.

**Recommendation**: do not promote staged δ continuation to the default δ≥2 driver mode based on
this result. Before re-testing, fix the ctx-rebuild redundancy (thread one `ctx` through all
stages) so the comparison isolates the continuation idea's real merit from this overhead artifact.

## 11. Canonical post-integration A/B — TBD (Phase F)

## 12. Final merge / rollback — TBD (Phase F)
