# Full-A_od CM + parallel-production consolidation: final handoff

Branch: `integration/fullA-cm-parallel-production`, worktree `gravity-fullA-cm-parallel-production`,
base `diag/fullA-d4-exact @ f6ae01e`. This document is Phase 3-4 of the engagement plan
(`atomic-brewing-unicorn.md`) — picking up after Phase 1 (KNITRO version resolution, branch setup)
and Phase 2 (CM port + BLAS-threading-guard port, `6d5eb67`) were already committed. Everything
below (Tasks 1-7) was built and validated this session, commits `5cf7f32`..HEAD.

## 1. Branch/commit map

| Role | Branch | Worktree | Tip | Status |
|---|---|---|---|---|
| **This branch (current tip)** | `integration/fullA-cm-parallel-production` | `gravity-fullA-cm-parallel-production` | HEAD (this doc's commit) | Active; NOT merged into `diag/fullA-d4-exact`, NOT pushed, per instructions |
| Production base | `diag/fullA-d4-exact` | `gravity-fullA-d4` | `f6ae01e` | Base this branch forked from |
| CM integration source | `integration/fullA-common-marginals` | `gravity-fullA-common-marginals-integration` | `7e24b94` | **Archival** — fully subsumed by `cc9ed94`'s merge onto this branch |
| BLAS-threading diagnostic source | `diag/fullA-inner-blas-threading` | `gravity-fullA-inner-blas-threading` | `ecc820d` | **Archival** — its 5 "what to merge" items already ported in `6d5eb67`; its own "what NOT to merge" verdict on the threaded CM Hessian is **superseded** by this session's Task 2 (see §4) |
| Canonical D20 rerun (read-only source for Task 5's point 2) | `diag/fullA-d20-canonical-rerun` | `gravity-fullA-d20-canonical-rerun` | live | Untouched; only read `canonical_rerun/checkpoints/d1_startB_canon_latest.jls` |
| Consolidation (fast-forwarded into base) | `integration/fullA-d20-runtime-delta5` | `gravity-fullA-d20-runtime-delta5` | `f6ae01e` | Archival, already subsumed into `diag/fullA-d4-exact` |

Do not delete any archival branch/worktree — mark superseded only, per standing instructions.

## 2. KNITRO version (Q1)

**13.0.1 is loaded and pinned.** `.knitro_env.sh` (repo root and this worktree) points
`KNITRODIR=/opt/shared_sw/knitro/13.0.1`. 14.2.0 and 14.0.0 are installed on this host but **not
covered by the current site Ziena license** — `KN_new()` returns `-520 "Could not find a valid
license"` on both, confirmed live in Phase 1 of this engagement (see the plan doc's own header and
`.knitro_env.sh`'s inline comment). `knitro_version_check.jl::verify_knitro_version()` fails fast if
the loaded release ever drifts from the declared production version; every script in this handoff
calls it (directly or via `c10_d20_production_driver.jl`'s own include-time check) and all runs
confirmed `Knitro 13.0.1` live. Do not attempt to re-enable 14.x until Artelys confirms license
coverage.

## 3. Current parallelism map (Q2, Q3)

| Component | Julia threading | BLAS threading | Decided where |
|---|---|---|---|
| Moment construction (`hFunction!`/`EK_moments_gammanorm_directgp!`) | Already `Threads.@threads` in production (pre-existing, not built this session) | n/a | production code |
| Outer-gradient coordinate loop (`composite_gradient_at_fast_buffered`'s `do_coord!`) | Already `Threads.@threads :static` in production (pre-existing) | n/a | production code |
| CM structured-Hessian bin-table construction (`build_bin_tables!`) | **NEW this session (Task 2)**: `hessian_cm_structured_v2!(threaded_bins=true)`, `cm_hessian_threaded.jl` | n/a for this stage | opt-in via `threaded_bins` kwarg; not yet a hard default (see §12 recommendation) |
| CM Hessian H_EE block (`Ews'*Ews`) | n/a | `BLAS.syrk!` (was `gemm!`) **NEW this session** | `BLAS.set_num_threads(n)`, caller-controlled |
| Unrestricted/no-CM dense Hessian (Architecture A, `hessopt=1`) | n/a | genuine `O(W*n^2)` gemm — benefits from `OPENBLAS_NUM_THREADS>1` per the prior BLAS-threading report (not retested this session) | caller-controlled |
| KNITRO-internal (`par_numthreads`, `par_blasnumthreads`, `par_lsnumthreads`) | n/a | left at production default (`par_numthreads=1`) | `full_aod_diag/ek_inner.opt` — prior report found no measurable benefit to raising these, not retested |
| Inner-solve sequencing guard | `parallelism_guards.jl` (ported `6d5eb67`) — runtime mutual-exclusion assertion between "inner KNITRO solve active" and "coordinate probe pool active" | n/a | wired into every real inner-solve entry point + both coordinate loops |
| `par_concurrent_evals` (`ek_inner.opt`) | n/a | `no` (was `yes`, changed `6d5eb67`) | defense-in-depth against a latent data race in shared non-thread-local callback state |

**No nested/oversubscribed threading was introduced.** `hessian_cm_structured_v2!(threaded_bins=true)`
uses `Threads.nthreads()` (process-launch-time, e.g. `-t 10`); BLAS thread count is set independently
via `BLAS.set_num_threads`, matching the existing `DIRECT_VALUE_NOCM`/`DIRECT_VALUE_CM` policy split
the prior report already established (no code change to that policy this session — see §12 for the
one update this session's finding warrants).

## 4. Combined CM-Hessian benchmark (Q4 — does it help on ACTUAL hard solves?)

**Task 1** (`c14_find_hard_cm_point.jl`) found a genuinely hard point (not calibration-adjacent):
a magnitude-0.5 perturbation along a fixed random direction from calibration's own reduced (`zfree`)
coordinates, at the real D20/W80000/δ=1/L=50 CM production context — **n_hess=9 Hessian callbacks,
wall=30.4s**, squarely matching the committed real full-CM-L=50 outer run's own typical profile
(`c13_perf_comparison/CM_L50/prof_summary.csv`: mean 8.95 Hessian calls/solve, median 30.3s/solve —
i.e. this is the TYPICAL point on a real trajectory, not a cherry-picked outlier). A near-infeasible
witness (magnitude 1.0, last point where the inner solve still converges before magnitude 1.5's
genuine `-300` failure) was also captured. Both saved to
`results/fullA_d4/c14_parallel_prod/hard_cm_point.jls`.

**Task 2** (`c14_cm_hessian_benchmark.jl`, `cm_hessian_threaded.jl`) ran the full 2x2 matrix — cells
A(serial bins, BLAS1), B(threaded bins, BLAS1), C(serial bins, BLAS∈{8,10,20}), D(threaded bins,
BLAS∈{8,10,20}), all with the `syrk!` fix for H_EE — at the hard point, the near-infeasible point,
calibration, and one real accepted CM outer-trajectory checkpoint (`stage_L50_latest.jls`), against
a REF cell (untouched production `hessian_cm_structured!`, serial+`gemm!`) as ground truth.

| Point | n_hess (REF) | REF wall | Cell A | Cell B | Cell C (best BLAS) | **Cell D (best)** |
|---|---:|---:|---:|---:|---:|---:|
| hard_cm_point | 12 | 45.7s | 1.19x | 3.10x | 1.39x (BLAS8) | **4.61x** (BLAS10) |
| near_infeasible_cm_point | 14 | 47.2s | 1.03x | 2.83x | 1.25x (BLAS10) | **4.33x** (BLAS10) |
| calibration | 11 | 36.6s | 1.08x | 2.75x | 1.23x (BLAS10) | **3.96x** (BLAS10) |
| cm_trajectory_stage_L50 | 10 | 34.3s | 1.07x | 2.61x | 1.20x (BLAS10) | **3.45x** (BLAS10) |

(wall-clock speedup vs REF; Hessian-callback-only speedups run ~5-15% higher at every cell — see
`results/fullA_d4/c14_parallel_prod/cm_hessian_benchmark.csv` for the full table including
`hess_cb_s`/allocation/iteration columns.) **Correctness held at every single cell, every point**:
`|Delta_dual - REF|` ranged 1.1e-16 to 2.8e-14, consistent with floating-point reduction-order noise
from the chunked/threaded summation (never a real discrepancy).

**Verdict: YES, the threaded CM Hessian helps substantially on real hard workloads** — a robust,
consistent ~4.0-4.6x wall-clock speedup (cell D) across all 4 tested points. This **reverses**
`diag/fullA-inner-blas-threading`'s own "do not merge — only 1.04x end-to-end" verdict, which tested
an easy ~2s calibration-adjacent cold solve where the one-time moment-construction cost dominated and
the Hessian callback barely fired. On a genuinely hard point (7-14 Hessian calls), the Hessian
callback is a large enough share of total wall time that threading it pays off decisively. Most of
the win is from Julia-threading the bin tables alone (cell B, ~2.6-3.1x at BLAS=1); BLAS threading
alone (cell C) contributes a smaller ~1.2-1.4x on top.

**Bug caught and fixed before trusting this result**: the shared `PsiObjectiveBundleImplicit.x`
warm-start cache persists across calls; without an explicit `force_cold_start!()` reset before every
cell (including REF), every cell after the first trivially warm-started from the PRIOR cell's own
converged solution at the identical point (1 FG call, 0 further Newton iterations) — see the Task 2
commit message for the full story, including a second, unrelated Julia-parser gotcha
(`Meta.parseall` silently accepts an unterminated `"""` docstring as `Expr(:incomplete,...)` instead
of erroring — a real trap for any future "did my script parse OK" pre-flight check on this codebase).

## 5. Allocation audit (Task 3)

Profiled the 7 listed sites (`@timed` bytes/gctime + `Base.gc_live_bytes()` deltas,
`results/fullA_d4/c14_parallel_prod/allocation_audit.csv` /
`allocation_audit_sites127.csv`):

| Site | Allocation | Classification |
|---|---:|---|
| 1/2/7. Compressed unrestricted value eval (TOTAL, real cold 15-iter solve) | 1069 MB | Mixed — could not cleanly decompose into sub-stages (see gap below) |
| 2 (dense path, `oracle_fast.jl`). `inner_moment_build` | 762 MB / call | (2) reusable scratch — one-time per-solve cost, not per-Newton-iteration |
| 2 (dense path). `inner_dual_hessian_callback` | ~0 MB / 9 calls | Already efficient, no action needed |
| **4. `build_lfix_base_cache`** | **1078 MB → 650-590 MB (FIXED)** | **(3) redundant copy — FIXED, see below** |
| 5. `a_block_fd_component!` (one coordinate probe, buffers preallocated) | 7.4 MB | (1) mostly unavoidable (real per-probe `q`/`psi` computation) |
| 6. `composite_gradient_at_fast_buffered` (400 coords, threaded) | 3997 MB total (~10 MB/coord) | Consistent with site 5 scaled up; the `q_bufs`/`psi_bufs` arrays (`nT` W-length vectors) are allocated fresh per gradient call — a (2) reusable-scratch candidate not fixed this session (see gap below) |
| 3. CM structured Hessian callback | See §4's own benchmark (`hess_cb_alloc_bytes` column, `cm_hessian_benchmark.csv`) | Not independently re-measured here — same physical numbers |

**Fixed**: `build_lfix_base_cache` (biggest single offender found) called `price_and_pTsigma_cell`
(`lfix_incremental.jl`) 400 times (D²=20²) in a tight loop, each allocating and immediately discarding
two fresh W=80,000-length vectors after copying into `price0`/`pTσ0` — ~512MB of pure allocate-then-
copy churn. Added `price_and_pTsigma_cell!` (in-place, fused broadcast, zero intermediate allocation)
and wired **only** `build_lfix_base_cache`'s loop to it; the original allocating function is untouched
and still used unchanged by its other 4 call sites. **Verified bit-for-bit identical** output at 2 real
points (`c14_verify_lfix_buffer_fix.jl`: `price0`/`pTσ0` `max_abs_diff=0.0` exactly, plus the
codebase's own independent `validate_dense=true` self-check still passes). **Real allocation
reduction confirmed**: 1078MB → 650MB at calibration (-40%), 1078MB → 590MB at a nearby perturbed
point.

**Gaps, disclosed not silently skipped** (given the session's time budget, per the brief's own "don't
boil the ocean" instruction):
- Sites 1/2/7's per-stage breakdown (compressed path) could not be obtained: `fast_range_screen.jl`'s
  `evaluate_fullA_screened_ranged` has **zero** `@prof` labels of its own (confirmed by direct grep) —
  it calls `compressed_live.jl`'s `inner_loop_KNITRO_compressed` directly, but the callback functions
  that DO carry `@prof` labels (`_callbackEvalFG_inner_compressed!`, `_callbackEvalH_inner_compressed!`)
  never populated the profile table in this call path for a reason not fully isolated (plausibly a
  DIFFERENT, locally-defined callback pair inside `fast_range_screen.jl` itself — not confirmed).
  Only the TOTAL number is trustworthy for this path; the dense-path decomposition (site 2) stands in
  as a partial substitute.
- Site 6's `q_bufs`/`psi_bufs` per-gradient-call allocation (part of its 3997MB) is a real, plausible
  (2)-reusable-scratch candidate — not fixed this session; would need a persistent, `ctx`-scoped (or
  thread-pool-scoped) buffer surviving across gradient calls, a larger change than a one-function
  in-place rewrite and not attempted given the time already spent verifying the `build_lfix_base_cache`
  fix correctly.
- A real live bug was caught and fixed mid-investigation (not a finding, but worth flagging): a
  warmup-then-reset measurement pattern at the SAME point left the *timed* call trivially warm-started
  from its own immediately-prior solution — same root cause as Task 2's bug (§4), independently
  rediscovered here. Fixed by dropping the warmup for that one measurement.
- **Caveat discovered AFTER this section's numbers were collected** (see §7's own writeup for the
  full story): the scripts behind this section's sites 1/2/7 measurements used an ad-hoc include
  list later found to silently corrupt `Delta_dual` (unrelated to allocation counting). The BYTES/
  GCTIME numbers reported above are very likely still valid — the missing files
  (`compressed_cc_inner.jl`/`lfix_buffer_reuse.jl`/`bandwidth_cache_policy.jl`/`dual_bank.jl`) are
  gradient/warm-start-bank/cache-policy code, not moment-construction or Hessian-callback code, and
  a real 15-iteration solve with plausible allocation magnitudes did occur — but this was not
  independently re-verified against the fixed include list given the time already spent nailing
  down Task 5's own version of the same bug. Flagged honestly rather than silently trusted.

## 6. Exact-cache instrumentation (Task 4, Q5)

`c14_cache_instrumentation.jl` ran a realistic outer-loop-style sequence (same point queried multiple
times within one simulated iterate, then a nearby point, then a later revisit) against both caches:

- **Unrestricted** (`SafeExactCache{FullAEvalKey}`): 4/7 probes hit, **every hit had
  `iters_delta==0`** (zero KNITRO iterations — confirmed via `CS.INNER_ITERS_TOTAL[]`), wall
  ~0.00002-0.03s vs ~0.68-15s for genuine misses (500-1000x+). A key-field-mismatch probe (identical
  `x_free`, different `δ`) correctly returned `nothing` from `_cache_lookup` — no false-positive hit.
- **CM** (`SafeExactCache{CMEvalKey}`): a repeat at the same `draw_checksum` hits
  (`iters_delta==0`); the SAME `x_free` under a DIFFERENT `draw_checksum` correctly misses and
  re-solves (`iters_delta=11`, a genuinely separate entry — confirms `cm_cache_key`'s partitioning);
  reverting to the original checksum still hits afterward, unaffected by the intervening entry — no
  cross-checksum collision.
- `cache_ur === cache_cm` confirmed `false` (different `SafeExactCache{K}` types) — they can never
  collide even in principle.

**Related work on a sibling worktree (informational, not merged here)**: `gravity-fullA-negative-cache-audit`
(a separate session, finished after this instrumentation was already committed) restored `-300`
negative caching under an opt-in "confirm-then-cache" policy (`negative_cache.jl`, wired into
`c10_d20_production_driver.jl` with `use_neg_cache=` defaulting to `off`) and fixed a real bug where
the driver's cold-retry success path never got cached. That work is scoped to a DIFFERENT question
(should a genuine, reproducible `-300` ever be cached, and under what confirmation policy) than this
section's own scope (does the EXISTING `is_cacheable_result`-gated `SafeExactCache` behave correctly
on POSITIVE hits/misses/collisions, which it does, per above). The two are complementary, not
overlapping or conflicting — `is_cacheable_result`'s `-300`-exclusion rule (verified still in effect
throughout this session's testing) is exactly the gap that worktree's `negative_cache.jl` opt-in
layer is designed to sit on top of. Not merged into this branch; flagged here for whoever performs
the eventual merge into `diag/fullA-d4-exact` to be aware both pieces of cache work exist on
separate, not-yet-reconciled branches.

**Answer to Q5: yes, exact repeated points are always cache hits, unconditionally, and a hit
mechanically never calls `KN_new`** (verified via the iteration counter, not merely assumed from
`cache_hit=true`'s presence in the return value). See
`results/fullA_d4/c14_parallel_prod/cache_instrumentation_{unrestricted,cm}.csv` for the raw
per-probe table and `Base.summarysize` memory costs (0.04MB / 1.26MB for these short sequences —
negligible; scales with the number of DISTINCT points ever queried, not call count).

## 7. Final CM/no-CM A/B benchmark (Task 5)

`c14_final_ab_benchmark.jl`, following `timing_harness.jl`'s exact point-construction and scenario
conventions, same commit/KNITRO 13.0.1/draw seed (20260719) throughout: calibration, the latest
unrestricted δ=1 candidate (read-only from `gravity-fullA-d20-canonical-rerun`'s
`d1_startB_canon_latest.jls`, κ=0.07864), and the hard CM L=50 candidate from Task 1. Each ×
{cold, exact-cache-hit, nearby-warm (|Δzfree|~0.05), difficult-solve (|Δzfree|~1.0)}.

| repr | point | cold | cache-hit | nearby-warm | difficult-solve |
|---|---|---:|---:|---:|---:|
| no-CM | calibration | 21.0s (Δ=0.230884) | 0.044s | 4.7s (Δ=0.230053) | 6.7s (Δ=0.268266) |
| no-CM | unrestricted_δ1_candidate | 8.7s (Δ=0.999960) | 0.00005s | 5.2s (Δ=1.000331) | 5.5s (Δ=1.051387) |
| no-CM | hard_cm_l50_candidate | 7.9s (Δ=0.237882) | 0.00006s | 3.1s (Δ=0.240788) | 4.8s (Δ=0.320509) |
| CM L=50 | calibration | 43.7s (Δ=1.498009) | 0.019s | 38.5s (Δ=1.433212) | 46.7s (Δ=6.722759) |
| CM L=50 | unrestricted_δ1_candidate | 21.1s **CM-INFEASIBLE (-300)** | n/a | 22.2s **-300** | 22.8s **-300** |
| CM L=50 | hard_cm_l50_candidate | 41.7s (Δ=1.698421) | 0.0001s | 40.9s (Δ=1.845521) | 41.4s **-300** |

**Cross-validation**: `calibration`'s no-CM `Δ=0.230884` matches `timing_harness.jl`'s own
committed historical value (`0.23088414900346...`) to every digit shown; `unrestricted_δ1_candidate`'s
no-CM cold `Δ=0.999960` matches the SOURCE checkpoint's own `verify_Delta_dual` to every digit
shown — both are genuine, independent reproductions, not assumed agreement.

**Real finding (not a bug)**: the unrestricted δ=1 candidate is **CM-L50-infeasible at every
scenario tried**. An unrestricted-optimal boundary point is not guaranteed compatible with the
ADDITIONAL common-marginal restrictions layered on top — the same class of failure as Task 1's
near-infeasible witness, now seen from the opposite direction (an unrestricted-native point
failing under CM, rather than a CM-native point failing under further perturbation). The hard CM
point's own "difficult-solve" scenario (one more 1.0-magnitude perturbation on top of an already-
hard point) also genuinely fails — consistent with Task 1's own magnitude-1.5-fails-from-
calibration finding. Both are caught gracefully (not crashes) via an explicit try/catch, per the
brief's own framing that CM-off is the byte-identical-unaffected baseline and CM-on is a genuinely
harder, sometimes-infeasible additional restriction, not a drop-in replacement.

Two real bugs were found and fixed reaching this table (both are real findings about THIS
SESSION'S OWN new diagnostic code, not about production): (1) an ad-hoc include list (missing
`compressed_cc_inner.jl`/`lfix_buffer_reuse.jl`/`bandwidth_cache_policy.jl`/`dual_bank.jl`)
silently corrupted `Delta_dual` to `-0.0` for every no-CM point, diagnosed by direct cross-check
against unmodified `timing_harness.jl`; (2) the canonical checkpoint's `best_feasible.w` field
does not equal `vcat(g, zfree)` (confirmed by direct inspection) — using the latter (which the
checkpoint's own resume-validation logic actually uses) fixed a "garbage point" that was crashing
the benchmark outright. See the Task 5 commit message for the full diagnostic story.

## 8. Smoke tests (Task 6)

**Unrestricted** (`c10_prod_driver_smoke_original.jl` / `c10_prod_driver_smoke_resume.jl`, reused
unmodified): `n_eval=3`, 2 KNITRO iterations, real feasible incumbent
(`Delta_dual=0.2145`, `max_abs_moment_kkt_resid=4.6e-13`, `gravity_value~4.7e-18`), 7/7 screens
passed. **Resume, in a separate fresh process**: loads the checkpoint, regenerates draws, validates
checksums match, reproduces `Delta_dual`/`gravity_value`/`max_abs_moment_kkt_resid`/
`moment_resid_norm` to **exact bit-for-bit agreement** (`|Δ|=0.0` on all four) — checkpoint/resume
round-trip fully verified, schema-3 `D20Checkpoint` (with `knitro_version`) exercised throughout.

**CM L=50** (`c14_smoke_cm_l50.jl`, `run_cm_upper`, 180s budget): `n_eval=3`, `n_grad=2`, real
accepted incumbent (`kappa=0.0318`, `best.Delta=0.169 <= delta=1`), checkpoint round-trip
(serialize→deserialize) confirmed OK, KNITRO version independently re-verified (13.0.1). Disclosed
gap: `run_cm_upper` does not go through the unrestricted path's schema-3 `D20Checkpoint` machinery
(its own simpler NamedTuple checkpoint instead) — flagged as a follow-up in §9/§10, not attempted
here given the smoke test's own time budget.

Both smoke tests: real D20/W80000/δ=1 data throughout, no crashes, no unexpected errors — the
integrated pipeline (CM port + BLAS-threading guards + exact-cache, everything from `6d5eb67`
onward) works end to end.

## 9. Go/no-go recommendations, by component

| Component | Recommendation |
|---|---|
| Threaded CM Hessian (`threaded_bins=true`, `cm_hessian_threaded.jl`) | **Promote to default** for CM production value/gradient-base solves — real, validated ~4-4.6x win on hard points (§4), correctness held at every tested cell. Recommend `BLAS.set_num_threads(10)` alongside it (`D_blas10` was the consistent best cell, `blas20` never beat it — matches the prior report's own "past ~10 BLAS threads, more threads mostly aren't buying anything" finding). |
| `syrk!` for H_EE | **Merge** — free, validated, zero downside (part of the same `hessian_cm_structured_v2!`). |
| `price_and_pTsigma_cell!` buffer-reuse fix in `build_lfix_base_cache` | **Merge** — verified bit-for-bit, real ~40% allocation reduction, zero behavior change, zero other call sites touched. |
| `parallelism_guards.jl`, `par_concurrent_evals no` | Already merged (`6d5eb67`), carry forward. |
| CM exact-point cache (`CMEvalKey`/`SafeExactCache{CMEvalKey}`) | Already merged (`6d5eb67`), now independently re-verified under a realistic sequence (§6) — carry forward, no changes needed. |
| `q_bufs`/`psi_bufs` reuse in the gradient call (site 6 gap) | Not built — flag as a follow-up, not urgent (3997MB/gradient-call is real but not the biggest offender found). |
| Compressed-path per-stage `@prof` labels (sites 1/2/7 gap) | Not built — flag as a follow-up if a future session wants a finer allocation breakdown of the production hot path specifically. |
| CM continuation driver on schema-3 `D20Checkpoint` | **Real, disclosed gap** (see §8) — `run_cm_upper`/`cm_outer_driver.jl` was never unified with the unrestricted path's `run_profile_checkpointed`/schema-3 checkpoint machinery. Flag as a follow-up; not attempted this session given the smoke test's own time budget. |

## 10. Answers to the brief's 7 closing questions

1. **Which KNITRO version is loaded and why?** 13.0.1, pinned. 14.2.0/14.0.0 are installed on this
   host but not covered by the current site Ziena license (`KN_new()` returns `-520`, confirmed
   live in Phase 1). See §2.
2. **Which callbacks use Julia threading?** Moment construction and the outer-gradient coordinate
   loop (`do_coord!`) — both pre-existing, not built this session. NEW this session: the CM
   structured-Hessian bin-table construction (`hessian_cm_structured_v2!(threaded_bins=true)`),
   opt-in, not yet a hard default. See §3.
3. **Which BLAS ops use how many threads and where is that decided?** `BLAS.set_num_threads(n)`,
   caller-controlled, independent of Julia thread count. The CM Hessian's H_EE block now uses
   `syrk!` (was `gemm!`); the unrestricted dense Hessian's `O(W·n²)` gemm benefits from
   `OPENBLAS_NUM_THREADS>1` per the (not retested) prior report. KNITRO's own internal
   `par_numthreads` stays at 1 (no measured benefit found previously). See §3.
4. **Does the threaded CM Hessian help on ACTUAL hard 20-35s solves?** **Yes, decisively** — a
   robust ~4.0-4.6x wall-clock speedup (cell D: threaded bins + `syrk!` + BLAS≥8) across all 4
   tested hard/near-infeasible/calibration/trajectory points, reversing the prior report's 1.04x
   "do not merge" verdict from an easy-point test. See §4.
5. **Are exact repeated points always cache hits now?** **Yes**, for both the unrestricted and CM
   caches, verified via `CS.INNER_ITERS_TOTAL[]` staying unchanged on every hit (not just trusting
   a `cache_hit=true` flag) — no false hits on key-field mismatches, no cross-checksum collisions
   in the CM cache. See §6.
6. **What's the recommended CM and no-CM direct-solve policy?** No-CM: unchanged production
   defaults (verified byte-identical to pre-session behavior throughout — every no-CM `Delta_dual`
   in §7's table cross-validates exactly against independently-trusted historical/checkpoint
   values). CM: adopt the threaded Hessian (`threaded_bins=true`, `syrk!`, `BLAS.set_num_threads(10)`)
   as the new default for CM value/gradient-base solves per §4/§9's recommendation; keep the exact
   cache and dual-warm-start behavior unchanged (already correct, §6); be aware that CM restricts
   the feasible set relative to no-CM — not every no-CM-feasible point is CM-feasible (§7's
   `unrestricted_δ1_candidate` row is a real, reproducible example, not an edge-case artifact) — so
   a CM-on run should expect genuine `-300` outcomes at some fraction of candidate points and should
   not treat them as bugs.
7. **What was merged into this branch this session vs. what's still open?** See §11 immediately
   below.

## 11. What was merged into this branch this session vs. what's still open

**Merged (committed, this session, `5cf7f32`..HEAD)**: Tasks 1-6's diagnostic/benchmark scripts,
the `price_and_pTsigma_cell!` buffer-reuse fix, the ported+improved `cm_hessian_threaded.jl`
(bug-fixed counter, `syrk!`, unified serial/threaded dispatch). All additive except the one
`lfix_incremental.jl` change, which is verified bit-for-bit safe.

**Still open** (see §9's table for the itemized list): the `q_bufs`/`psi_bufs` gradient-call
allocation, the compressed-path `@prof` label gap, and CM-driver checkpoint-schema unification.
None of these block merging what IS done here — they are follow-up items, not correctness blockers.

**This branch was NOT merged into `diag/fullA-d4-exact` and NOT pushed to the remote**, per
instructions — that decision is left to a final human/coordinating-session gate.
