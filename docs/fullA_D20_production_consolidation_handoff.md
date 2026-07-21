# Full-A_od D=20 production consolidation, timing audit, and δ=5 improvement

Status: Phases A-F and the QMC draw-design addendum are all complete, independently validated,
and merged into `diag/fullA-d4-exact`.

Integration branch: `integration/fullA-d20-runtime-delta5`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d20-runtime-delta5`, base `98983bd`
(`diag/fullA-d4-exact`, treated as production).

## 1. Source-stream / commit integration map

| Source | Branch (worktree) | Tip | What it contributed | Status here |
|---|---|---|---|---|
| Canonical rerun | `diag/fullA-d20-canonical-rerun` (`gravity-fullA-d20-canonical-rerun`) | `98983bd` (+ uncommitted) | δ∈{0.1,1,2,5} D=20/W=80,000 frontier, pairwise-screen crash root-cause + uncommitted fix, organic infeasibility traces/checkpoints, first algorithm=2 experiment at δ=5 | Pairwise fix ported+committed here (`afe09a9`). Traces/checkpoints to be reused in Phase D from that worktree directly (not copied). Marked superseded once this branch merges. |
| Warm-start replay | `diag/fullA-d20-warmstart-replay` (`gravity-fullA-d20-warmstart-replay`) | `baec13e` (7 commits past `98983bd`, clean) | `SafeExactCache` (lock-guarded exact-point cache), KKT-proxy-scored successful-dual bank (policy P3), live δ=5 A/B (-29.5% wall, trajectory-dependent) | To be ported in Phase B. Marked superseded once merged. |
| Fast range screen integration | `integration/fullA-fast-range-screen` (`gravity-fullA-fast-range-screen-integration`) | `98983bd` (clean, = base) | Envelope/winning-range/safety-net screens, already merged into base | Base only, nothing to port. |
| QMC investigation | `diag/fullA-d20-qmc-delta1` (`gravity-fullA-d20-qmc-delta1`) | `5882c16` | Validated pseudorandom/Sobol(Cranley-Patterson-shifted)/Halton(Owen-style-digit-scrambled) draw generators for D=20/W=80,000, generator validation report, Stage A-E replicate results | Ported as `draw_design.jl`'s `d20_real_setup_design` (§6), merged (`6e126fe`), independently re-validated (22/22). Superseded. |

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

## 6. Draw-design port (pseudorandom / sobol_randomized / halton_scrambled) — DONE (addendum)

Ported by a background agent (3 commits: `2939c1e` add `draw_design.jl`, `6a9a765` wire the driver
+ checkpoint schema, `29b3eec` a real bug fix — see below), merged into this branch via a genuine
3-way merge (`git merge --no-ff`, no conflicts — the agent's branch and this one diverged from a
common ancestor `81cc21e` and touched different lines of the one shared file,
`c10_d20_production_driver.jl`), then independently re-validated by this session
(`test_draw_design.jl`, 22/22 — not just trusting the porting agent's own claims).

**What it is**: one new entry point, `d20_real_setup_design(; draw_design=:pseudorandom,
draw_seed=..., ...)`, a thin selector over three already-existing, already-validated code paths
from `diag/fullA-d20-qmc-delta1` (now subsumed): `:pseudorandom` (exact existing production call
sequence, bit-for-bit unchanged), `:sobol_randomized` (Sobol.jl `SobolSeq` + a Cranley-Patterson
random shift — named honestly, NOT "scrambled", since Sobol.jl doesn't implement digital/Owen
scrambling), `:halton_scrambled` (genuine Owen-style per-digit-scrambled Halton, ported from Art
B. Owen's R code). All three route through the same inverse-CDF transform, country/dimension
ordering, Frechet parameters, and `1/W` weights; nothing about the CC objective, divergence,
moments, winner rules, or tie convention is touched.

`c10_d20_production_driver.jl` gains a `draw_design::Union{Nothing,Symbol}=nothing` kwarg on both
`run_profile_checkpointed`/`run_polish_checkpointed` (`nothing` = "no opinion, inherit the
checkpoint's own design on resume" — a real sentinel distinct from explicitly passing
`:pseudorandom`, which must still hard-error against a mismatched checkpoint). `D20Checkpoint`
gains `draw_design`/two draw checksums (schema bumped 1→2, `load_checkpoint` hard-errors on an
old schema-1 file); resume hard-errors on any design or checksum mismatch; `guard_checkpoint_path`
refuses to overwrite a checkpoint built under a different design/checksum at the same path.

**A real bug caught and fixed by the porting agent's own negative test** (`29b3eec`): the first
version of the resume-mismatch guard special-cased `draw_design_in != :pseudorandom` to mean
"caller didn't specify," but that's indistinguishable from a caller *explicitly* requesting
`:pseudorandom` against a checkpoint built under a different design — which must still error. Its
own driver smoke test's negative case caught this (`:pseudorandom` explicitly requested against a
sobol/halton checkpoint silently succeeded on first pass); fixed by making the "no opinion"
sentinel a real `Union{Nothing,Symbol}` default instead of overloading `:pseudorandom` itself.

**Independent re-validation** (`test_draw_design.jl`, written by this consolidating session, not
the porting agent — 22/22): `:pseudorandom`'s `ctx.U` and a real value-eval `Delta_dual` are
bit-identical to the pre-existing `d20_real_setup` path; fresh-context checksum reproducibility
for all three designs; `:sobol_randomized`/`:halton_scrambled` build and solve successfully with
genuinely different (and mutually different) draw checksums, zero non-finite or exact-boundary
draws; negligible `log_draw_meta` overhead (~1-3% of a ~20s ctx build). One bug found in *this
session's own* validation script along the way (checked `inner_status isa Int`, but KNITRO
returns `Int32`; the underlying sobol/halton value evaluations had already succeeded with sane
results throughout — a test-assertion bug, not a port defect) — fixed and re-run clean.

**Not independently re-verified**: the porting agent's own calibration-comparison and
generator-validation scripts (`draw_design_calibration_comparison.jl`,
`draw_design_generator_validation.jl`, `draw_design_overhead_benchmark.jl`,
`draw_design_driver_smoketest.jl`) were left uncommitted on the agent's own branch when this
session took over finishing the port directly — not merged here, since this session's own
independent validation (above) already covers the addendum's core acceptance bar (pseudorandom-
default unchanged, designs build/solve/checksum-reproduce correctly, negligible overhead). The
30-replicate-style statistical comparison against the original QMC investigation's own numbers
(§7 of the original addendum spec) was not repeated — out of scope for a port-correctness check.

`diag/fullA-d20-qmc-delta1` (worktree `gravity-fullA-d20-qmc-delta1`) is now subsumed and can be
marked superseded.

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

## 11. Canonical post-integration A/B (Phase F, synthesized from Phases C-E's real runs)

Per the plan, this reuses the real runs already completed rather than repeating the full
four-point frontier:

- **δ=1** (values/gradients/candidate unchanged): §7's 4-way benchmark — `Delta_dual` identical
  to every digit shown (`0.23088414900346...`) across legacy/base/this-branch-off/this-branch-on,
  gradient norm matching to 9 significant figures. No behavior change from any of this session's
  integrations.
- **δ=2** (cache benefit + organic failures): §7's exact-point cache hit (0.040s) vs. a repeated
  warm re-solve (0.64-1.17s) — a real ~16-29x speedup on an exact-repeat point; §8's δ=2 accepted-
  boundary profile (8.417s cold / 0.624s warm, 56 inner iterations) shows the cache/bank additions
  add no measurable overhead at this δ (matches §7's B-vs-C timing parity).
- **δ=5** (staged workflow vs. canonical baseline): §10's real comparison — direct beats the
  staged continuation at a matched 240s budget (κ=0.004550 vs 0.003114), with the shortfall
  diagnosed as a redundant per-stage context-rebuild overhead in the staged implementation, not
  necessarily a verdict on the continuation idea itself.

**Cold-verification of the best reported candidate**: the δ=1 upper candidate (κ=0.0786007120,
`docs/provenance/fullA_delta1_round4_independent_verification.md`) is unchanged by anything in
this consolidation (no code path touching its calculation was modified in a way that changes
output — confirmed structurally, since §7's Delta_dual/gradient-norm parity check is exactly this
kind of regression test at a *different* point with the same code paths). Not independently
re-solved from scratch in this phase (would duplicate the already-extensive Round4 verification);
flagged here as unchanged rather than re-verified redundantly.

## 12. Final merge / rollback

**Rollback point**: tag `pre-consolidation-2026-07-21` on `98983bd` (the pre-consolidation
`diag/fullA-d4-exact` head), created in the `gravity-fullA-d4` worktree. To roll back:
`git checkout pre-consolidation-2026-07-21` or `git reset --hard pre-consolidation-2026-07-21`
(the latter is destructive to anything built on top — confirm before using).

**Merged into `diag/fullA-d4-exact`**: fast-forward `98983bd`→`afb75af` (Phases A-F, no rebase/
squash, full history preserved), then a genuine (no-conflict) merge commit bringing in the QMC
draw-design addendum's 3 commits from the porting agent's branch plus this session's own
follow-on validation/fix commits, up to the final tip:

1. `afe09a9` — pairwise-screen crash fix + regression test (§3)
2. `81cc21e` — handoff doc scaffold + KNITRO version finding (§2)
3. `43928e3` — lock-guarded exact-point cache, SafeExactCache (§4)
4. `9f11643` — successful-dual/KKT-scored bank + driver wiring for both (§5)
5. `76f240b` — handoff doc update
6. `b74bae5` — Phase C timing-regression audit (§7)
7. `f830390` — Phase D granular profiling + organic-300 finding (§8-9)
8. `38fd767` — Phase E staged-δ5 continuation result (§10)
9. `afb75af` — finalized Phase A-F handoff doc
10. `2939c1e`/`6a9a765`/`29b3eec` — QMC draw-design port + driver wiring + a real bug fix (§6),
    from the porting agent's branch, merged via a no-conflict 3-way merge
11. `6e126fe` — independent draw-design validation (22/22) (§6)

**Not merged / explicitly deferred**:
- Dual-ray monitor / cutting-plane certificate (task §7.1-7.3) — not attempted, no validated
  current-architecture organic failing point available this session (§9).
- Fixed-g profile continuation and staged KNITRO algorithm switching (task §8.2-8.3) — not
  attempted; only the simplest staged-δ-continuation variant was built and tested, and it lost to
  the direct baseline at the one bounded budget tried (§10).
- QMC draw-design port's own calibration-comparison/generator-validation/overhead-benchmark
  scripts (left uncommitted on the porting agent's branch) — not merged; superseded by this
  session's own independent validation (§6), which covers the addendum's core acceptance bar.

**Branches to mark superseded** (not deleted, per instructions):
- `diag/fullA-d20-canonical-rerun` — pairwise fix absorbed into `afe09a9`; its traces/checkpoints
  were reused directly (not copied wholesale) in §8-9.
- `diag/fullA-d20-warmstart-replay` — SafeExactCache design and KKT-bank policy P3 absorbed into
  `43928e3`/`9f11643`.
- `integration/fullA-fast-range-screen` — was already the base (`98983bd`), nothing further to
  supersede.
- `diag/fullA-d20-qmc-delta1` — generator/validation methodology absorbed into the QMC addendum
  port (once merged).

**Push**: not yet pushed to `origin` — confirm with the user before pushing, per standing git
safety practice for this session.

---

### Answers to the task's 9 closing questions

1. **Did the current code regress relative to the known historical fixed-point benchmarks? No.**
   §7's 4-way benchmark shows `Delta_dual` and gradient norm identical across the historical
   (`cf74d89`), pre-consolidation (`98983bd`), and both cache-off/cache-on variants of this
   session's code, at every timing point measured. All timings are flat within ordinary
   run-to-run noise on this shared server.

2. **What precisely does a typical 4-second δ=1 value callback contain?** ~7.4% pre-solve
   screening (θ_full reconstruction, pairwise/envelope/winner-scan certificates), ~92.6% the
   actual KNITRO inner CC-dual solve itself (32 iterations, 7 FG calls, 6 Hessian calls in the
   measured instance) — §7.1.

3. **How much do exact caching and successful-dual selection save at δ=1,2,5?** The exact-point
   cache saves ~16-29x on a literal repeated point (0.040s vs 0.64-1.17s, §7) — essentially free
   money whenever the exact same point is re-evaluated (which happens routinely across
   `cb_F!`/`cb_G!`/`cb_newpt!` in one KNITRO iterate). The successful-dual/KKT bank was validated
   for mechanism correctness (9/9, §5) but not benchmarked on a live multi-iterate δ=1/2/5
   trajectory in this session — deferred, see §11.

4. **Does an organic inner -300 erase the single warm slot, and does the new successful-dual bank
   prevent the resulting cold start?** Yes to the first (confirmed by code inspection: `obj.x .=
   NaN` on `!warm`, and `inner_loop_initial_values`'s NaN→zeros fallback) — this was the original
   motivation for the bank. The bank mechanism is built and unit-validated to never store a failed
   point and to fall back correctly (§5), but a live-trajectory demonstration that it actually
   rescues a real organic `-300` was not run this session.

5. **Why do successful inner solves become slower at larger δ?** Inner iterations roughly double
   at each step (27→56→112+); FG-call count jumps far more sharply at δ=5 (13-15→85) — each
   iteration does more work, not just more iterations; δ=5 never reaches full KNITRO optimality
   (status `-100` throughout) because the divergence budget forces a genuinely more extreme
   reweighting; warm-starting itself provides much less benefit at δ=5 (§8).

6. **Can organic joint infeasibility be certified materially before KNITRO's original -300
   termination?** Not established this session. The one available saved "organic" point no longer
   reproduces `-300` in current code (caught in 0.061s by the already-merged envelope screen
   instead) — real evidence that screen improvements already resolved this specific instance, but
   it leaves no current-architecture failing point to validate new certificate machinery against.
   Real, current `-300` events do exist (40 at δ=5, confirmed passing every current screen) but
   weren't captured as reusable standalone points this session (§9).

7. **Does fixed-g/finer-δ continuation reduce the δ=5 rejection rate?** Only the finer-δ
   continuation variant was tried (not fixed-g); at the one bounded (240s) budget tested, it did
   NOT beat a direct δ=5 attempt (κ 0.003114 vs 0.004550), but the shortfall is diagnosed as a
   redundant per-stage context-rebuild overhead (over half the staged run's wall time), not
   necessarily a verdict on the continuation idea itself (§10).

8. **Which KNITRO exploration/polish sequence gives the best exact-feasible κ per wall time?**
   Not established beyond what the canonical rerun's own §8 already found (algorithm=2 gave a
   real but modest δ=5 improvement, still short of the best δ=5 κ) — not independently re-tested
   or extended this session.

9. **Which changes were merged into production, at what commits, and how can they be rolled
   back?** See the merge list and rollback tag above.
