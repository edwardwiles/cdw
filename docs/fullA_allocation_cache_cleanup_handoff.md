# Full-A_od allocation reduction, persistent workspaces, checkpoint schema, cross-δ cache: handoff

Branch: `perf/fullA-allocation-cache-cleanup`, worktree `gravity-fullA-alloc-cache-cleanup`,
base `diag/fullA-d4-exact @ cffc028` (the real final-production-merge head — rebased onto it
mid-session once that merge landed; see §0). **Not merged into `diag/fullA-d4-exact` and not
pushed, per the task's explicit instruction** — a coordinating/human session should review
before merging.

## 0. Coordination with the final merge

At session start, the "final production head" this task assumed did not yet exist:
`integration/fullA-cm-parallel-production @ 08b07d0` had the CM parallel Hessian, ported
BLAS/threading, and a prior allocation audit ("Task 3", see §1 below), but
`diag/fullA-driver-delta5` (gamma bounds, direction fix, `reusable_context.jl`,
`organic_failure_capture.jl`) was not yet merged in, and no other session was actively working
(confirmed: no live Julia processes, no recent file activity). The user confirmed another
Claude session was doing that merge; I did read-only prep (auditing the exact-cache key,
reviewing `organic_failure_capture.jl`) until a live process was confirmed running in
`gravity-fullA-final-production-merge`, then branched from that session's in-progress tip
(`e6e2f92`) on the user's explicit go-ahead, since my target files (`lfix_incremental.jl`,
gradient/value internals) were untouched by that session's own remaining diff.

Mid-session, the other Claude finished and the REAL final merge landed on `diag/fullA-d4-exact`
at `cffc028` (`docs/fullA_final_production_merge_handoff.md`, rollback tag
`pre-final-merge-2026-07-21` at `08b07d0`). My branch's base (`e6e2f92`) was a direct ancestor
of `cffc028` (only 3 commits apart: a direction-label bugfix, the merge itself, the handoff
doc), so `git rebase cffc028` applied with **zero conflicts**. All tests were re-run and
re-passed post-rebase (§4/§7).

That handoff doc itself flags, as its own "remaining nonblocking follow-ups": **"CM
checkpoint-schema unification for `run_cm_upper`, `q_bufs`/`psi_bufs` allocation work"** — i.e.
exactly this task's §7 and §11. This task is a direct continuation, not independent/duplicate
work.

## 1. What was already done before this session (do not re-attribute)

A prior session ("Task 3", commit `ddbe98c` on the path from `cm-parallel-production`, now part
of the merged history) already:
- Profiled 7 hot-path allocation sites at real D20/W=80000.
- Found and fixed `build_lfix_base_cache`'s biggest offender: `price_and_pTsigma_cell` (allocating)
  called 400 times in a tight loop → `price_and_pTsigma_cell!` (in-place). **1078MB → 590-650MB**,
  the exact figure this task's brief cites — that reduction is NOT this session's work.
- Explicitly flagged, as NOT built: the `q_bufs`/`psi_bufs`-adjacent allocation in
  `composite_gradient_at_fast_buffered` (this task's §7), and CM checkpoint-schema unification
  (this task's §11).

Separately, `diag/fullA-driver-delta5` (merged into the final head, see §0) already built
`reusable_context.jl` (context/pe/rsc reuse across δ-stages — related to but distinct from this
task's §12, the EXACT-CACHE key) and `organic_failure_capture.jl` (~80% of this task's §13,
see §6).

## 2. Irreducible memory floor (§3 of the brief)

At real D=20, W=80,000:

| Array | Dimensions | Approx. size | Lifetime |
|---|---|---|---|
| `ctx.U` (productivity draws, transformed) | W×D | 80000×20×8 = 12.8 MB | Persistent (ctx) |
| `price0`/`pTσ0` (LFixBaseCache) | W×D×D each | 80000×400×8 = 256 MB each, 512 MB together | Per-cache (one build_lfix_base_cache call) |
| `winner0`/`runnerup0`/`third0` + price levels | W×D each, ×6 arrays | 80000×20×8×6 ≈ 76.8 MB | Per-cache |
| `contrib0` | W×D | 12.8 MB | Per-cache |
| Dense unrestricted moment matrix (`G`/`H` in `oracle_fast.jl`'s dense path) | W×402 | 80000×402×8 ≈ 257 MB | Per-value-eval (matches the brief's own worked example almost exactly) |
| Compressed moment representation | O(W·D) not O(W·D²) | ~13-26 MB | Per-value-eval (Continuation 7's compression) |
| Hessian (dense, unrestricted) | 401×401 | 1.3 MB | Per-Hessian-callback |
| CM contingency tables (Architecture C, L=50) | O(W) bin-indexed, thread-local ×nT | ~0.64 MB × nT | Per-CM-Hessian-callback |
| `GradWorkspace` (this task, §7) | 6 × W-length | 6×0.64 MB = 3.84 MB/thread | Now PERSISTENT (pool), was per-call |
| Successful-dual-bank entries | small (zeta+lambda length) | negligible | Persistent, bounded (size=8 default) |
| Exact-cache entries | one NamedTuple/point | ~1.3 MB/entry (`Base.summarysize`, per the prior session's own measurement) | Persistent, unbounded by point count (not by call count) |

**Lower-bound estimate, one unrestricted dense value eval**: ~257 MB (moment matrix) is the
dominant single array; total realistic live memory for one eval (moment matrix + price/pTσ
scratch + Hessian + misc) is in the 300-400 MB range, consistent with `build_lfix_base_cache`'s
own measured 590-650 MB (it additionally holds price0/pTσ0/winner tables simultaneously, which
a single value-eval does not all need at once).

**One full outer gradient** (400 coordinates): before this session, ~4.49 GB allocated (§3 below);
after, ~756 MB (§3) — the floor is NOT "one gradient's worth of W-length temporaries" (that
would be tiny, ~4 MB) but the ONE-TIME `build_lfix_base_cache` cost (~590-650 MB) plus whatever
solve_base_state itself needs, since `composite_gradient_at_fast_pooled`/`_buffered` both
rebuild the cache fresh at the top of every call.

## 3. Allocation audit: the ~4GB/gradient breakdown (§4/§6 of the brief)

Measured live (`bench_grad_alloc.jl`, real D20/W=80000/calibration, warmed, `@allocated`):

| Site | Bytes | Classification |
|---|---:|---|
| `a_block_fd_component!` (one coordinate probe, buffer-reuse path, BEFORE this session's fix) | 7.72 MB | Traced to: `price_and_pTsigma_cell` (allocating, 1.28MB/call) + fresh `contrib`/`cf` `Vector{Float64}(undef,W)` (0.625MB each), called 2-4× per probe (Lp/Lm × 1-2 affected destinations + cf_contrib) |
| `dest_contrib_incremental_o1` (single call, 1 changed origin) | 1.957 MB | (3) redundant — see fix below |
| `price_and_pTsigma_cell` (allocating, called inside) | 1.28 MB | (3) redundant — in-place variant already exists, just not wired into this call site |
| `cf_contrib_at` (single call) | 1.28 MB | (3) redundant — two W-length temporaries from un-fused broadcasts |
| `composite_gradient_at_fast_buffered` (D²=400 coords, serial, BEFORE) | **4489.9 MB** (~11.2 MB/coord) | Matches the brief's own "~4GB/gradient" figure exactly |

**Root cause, precisely**: `composite_gradient_at_fast_buffered`'s own `q_bufs`/`psi_bufs`
(Continuation 10/11) are ALREADY reused across all 399 coordinates within one call — that part
was already fixed, before this session. What was NOT reused is everything `a_block_fd_component!`
calls THROUGH those buffers: `dest_contrib_incremental_o1` (the default, common-case tier,
handling the single-changed-origin case that a real D20 gradient overwhelmingly hits) still
calls the ALLOCATING `price_and_pTsigma_cell` — not the in-place `price_and_pTsigma_cell!` that
`build_lfix_base_cache` already uses — and allocates a fresh `contrib` vector every call;
`cf_contrib_at` allocates two more fresh W-length arrays on every counterfactual-column touch.

## 4. Fix: persistent per-thread gradient workspace (`gradient_workspace.jl`, §7)

Commit `5595021` (pre-rebase `1d12b33`). Adds:
- `GradWorkspace` (6 W-length buffers: q, psi, price, pTσ, contrib, cf) and `GradWorkspacePool`
  (one `GradWorkspace` per thread SLOT, sized via `Threads.maxthreadid()` — NOT `nthreads()`,
  matching the existing code's own documented task-migration bug catch).
- In-place `dest_contrib_incremental_o1!`/`cf_contrib_at!`, using `price_and_pTsigma_cell!`
  (already existed) and single fused broadcasts (`.=` over the whole expression tree).
- `composite_gradient_at_fast_pooled`: same signature/semantics/`:static`-scheduling
  requirement as `composite_gradient_at_fast_buffered`, but takes a caller-owned, PERSISTENT
  `pool` instead of allocating `q_bufs`/`psi_bufs` fresh every call.
- **Scope decision, disclosed**: the rare 2-changed-origin-in-one-destination case (both the
  direct coordinate and the gravity pivot land in the same destination) falls back to the
  existing ALLOCATING `dest_contrib_incremental_top3`/`_generic`, respecting the caller's
  `multi_method` kwarg exactly as the original does — not a new policy, just not converted to
  in-place given this is a rare path (per the codebase's own established "correctness first,
  rare paths may allocate" precedent).

**Purely additive** — `lfix_incremental.jl`/`lfix_buffer_reuse.jl`/`composite_gradient_fast.jl`
are byte-for-byte unchanged.

**Verified** (`test_gradient_workspace.jl`, 9/9 assertions): bit-for-bit identical to
`composite_gradient_at_fast_buffered` at 2 real D20/W=80000 points (calibration, a nearby
perturbation), both serial and threaded (`nthreads=10`), with threaded also matching serial.
Pool-reuse allocation: **756.5 MB vs 3713.0 MB per gradient call (4.91x reduction)** on a warm
pool (measured with `h_mode=:fixed` to isolate the workspace effect from bandwidth-cache
warming). Re-verified passing after the mid-session rebase onto the real final production head.

**Remaining ~750MB**: dominated by `build_lfix_base_cache`'s own ~590-650MB (rebuilt fresh
every gradient call, per §2/§5) plus `solve_base_state`. Reducing THAT further (e.g. skipping
the rebuild when the outer point hasn't moved) is a distinct, larger change — not attempted
this session (would change cache lifetime/invalidation semantics, a correctness-sensitive
decision better made explicitly, not as a side effect of an allocation pass).

## 5. `build_lfix_base_cache`'s remaining 590-650MB (§5 of the brief)

Not independently re-derived this session beyond what §1/§2 above already establish: it is the
REQUIRED persistent output (`price0`/`pTσ0`, 512MB combined, W×D×D each — genuinely needed by
every subsequent coordinate probe in the gradient, not scratch) plus the winner/runner-up/
third-place tables (~77MB) and smaller per-destination arrays. The prior session's own fix
(§1) already eliminated the ONE real redundant-copy pattern found (`price_and_pTsigma_cell`'s
allocate-then-discard in the D²-loop). This session's own audit (§3/§4) did not find a second
comparably-sized redundant allocation inside `build_lfix_base_cache` itself — the remaining
~600MB is real, required output, not further churn. (A genuinely different lever — not
attempted — would be NOT rebuilding this cache from scratch on every gradient call when the
outer point is unchanged since the last value-eval; flagged as a follow-up, see §9.)

## 6. Organic `-300` failure capture (§13 of the brief) — partial, disclosed

`organic_failure_capture.jl` (already on the merged base, from `diag/fullA-driver-delta5`)
already captures: exact outer vector (g, zfree), full reconstructed A (`logA_full`), draw
checksums, config hash, screen counts, decoded KNITRO status, dual before/after snapshots,
timing, checkpoint parent, and a reproduction command — bounded at `max_n=5` (matching the
brief's own suggested `save_first_n_organic_failures` default verbatim), plus a
`replay_organic_failure` one-command replay. Gaps against the brief: no residual vectors
captured, and only before/after dual snapshots rather than a full trajectory.

**This session's own investigation**:
1. **Live fire-test** (`organic_fire_test.jl`): magnitude-scanned a random direction from
   calibration at δ=5 (0.5 to 4.0 in 0.5 steps, real D20/W=80000, ~8 solves × 7-22s each) — every
   point solved (`inner_status=0`), no organic failure found in this scan. δ=5's wide budget
   makes naive random perturbations solve easily; the known real pathologies (see below) came
   from specific degenerate points on the SEQUENTIAL method's own trajectory, not generic
   perturbations.
2. **Replay of a known, previously-documented real failure**
   (`diagnostics/infeasibility_points/nonzero_winner_infeasible_delta5_candidate1.json`, D20/
   W=80000/δ=5, originally found in Continuation 11): reconstructed the exact point (gp_focal +
   full A_od from its own recorded CSV) and re-evaluated through the CURRENT production
   `screened_eval`. Result: **now caught by the envelope screen**
   (`EXACT_INFEASIBLE_PREWINNER_ENVELOPE`, `inner_status=-9004`) BEFORE reaching the real KNITRO
   inner solve at all — a genuine sign the screen stack has gotten stronger since that point was
   first found (a positive finding), but it means this specific historical point can no longer
   serve as an "organic" (post-screen) failure test case.

**Disclosed gap**: no FRESH organic `-300` was captured and replayed this session. The
capture/replay MECHANISM itself was read-reviewed (not modified) and is the clear next step for
whoever has more search budget — likely needs either the sequential method's own trajectory
(not the unrestricted full-A path this session searched) or a longer/smarter search, not a
short random-direction magnitude scan. Extending the record schema with residuals/full dual
trajectory (the two disclosed gaps above) was not attempted either, given no fresh failure was
available to design against live.

## 7. CM production checkpoint schema (§11 of the brief)

Commit `b0b2f33`. `run_cm_upper` (`cm_outer_driver.jl`) had **zero** checkpoint/resume support —
a single-shot KNITRO call. The one place CM runs WERE checkpointed
(`c13_d20_cm_upper_continuation.jl`'s `save_stage`) wrote a bare ad-hoc NamedTuple: no schema
version, no draw-checksum validation, no KNITRO-version field, no load/resume logic — write-only,
used only to seed the NEXT (coarser→finer) grid stage.

Adds `cm_checkpoint.jl`: `CMCheckpoint` (same rigor as the unrestricted path's schema-3
`D20Checkpoint` — run_id/label/branch/find_smallest/delta/W/draw_seed/draw_design/checksums/
KNITRO version/best_feasible/bandwidth_cache/n_eval/wall_elapsed/checkpoint_reason — PLUS the
CM-specific fields the brief lists: `cm_L`, `cm_probs` (exact cutpoints, not re-derived from L),
`cm_contrasts`, `cm_grid_rule`, `cm_basis`, `cm_hessian_backend`) and
`run_cm_upper_checkpointed`, a NEW wrapper — `run_cm_upper`/`cm_outer_driver.jl` are UNCHANGED.
Checkpoints on `:new_best`/`:wall_interval`/`:stage_complete` (matching
`run_polish_checkpointed`'s own discipline); resume hard-refuses (not silently proceeds) if
regenerated draw checksums don't match the checkpoint's recorded provenance.

**Verified end-to-end**, real D20/W=80000/L=10, TWO SEPARATE PROCESSES
(`test_cm_checkpoint_original.jl` → `test_cm_checkpoint_resume.jl`, mirroring the unrestricted
path's own `c10_prod_driver_smoke_original.jl`/`_resume.jl` convention):
- Original run forced to stop early (`maxtime_real=45s`) → checkpointed after 2 evals.
- Resumed run (fresh process): eval counter continued 2→4 (not reset), found a genuinely BETTER
  incumbent (`gp` 0.99754→0.99743, `Delta` 0.392→0.380, both ≤δ=1) from where it left off.
- Draw checksums: exact match on regeneration.
- Checkpoint's own recorded best-feasible Delta, independently re-verified at that exact point:
  agrees to `2.8e-15` (relative) — the same order of floating-point noise this codebase's own
  cross-warm-start-path comparisons already treat as solver noise, not a discrepancy (not a bit-
  exact match, since this crosses a cold-vs-different-warm-start KNITRO boundary — disclosed in
  the test's own docstring, not silently ignored).

**Not attempted**: migrating the OLD `c13_d20_cm_upper_continuation.jl` ad-hoc NamedTuple
checkpoints to the new schema (no migration path provided — old checkpoints simply can't be
loaded by `load_cm_checkpoint`, which is the same "start fresh, don't silently half-load" policy
`D20Checkpoint`'s own schema check uses).

## 8. Cross-δ exact-cache persistence (§12 of the brief)

Commit `e848b1a` (pre-rebase `ebce75d`). **Root cause, confirmed by tracing every use of
`obj.δ`/`obj.find_smallest` inside the actual inner-solve path** (`oracle.jl`, `oracle_fast.jl`,
`compressed_live.jl`, `fast_range_screen.jl`, `infeasibility_screen.jl`): `FullAEvalKey`
includes both in its hash/equality, but NEITHER is read by the real inner CC dual solve —
`obj.δ` only computes a post-hoc `Delta_minus_delta` reporting field; `obj.find_smallest` only
sign-flips an already-computed `obj.H[1,1]` into `K_hard`/`H_save` AFTER the solve returns.
`Delta*(θ)` is genuinely δ-independent — a staged δ=2→3→4→5 continuation was needlessly
treating every stage as a cache miss at the identical `x_free`.

Adds `cross_delta_cache.jl`: `CrossDeltaExactCache` (new `FullAInnerKey`: x_free/find_smallest/
inner_loop_opt/mode, δ stripped) as NEW `_cache_lookup`/`_cache_store!` multiple-dispatch
methods — purely additive, the 5 existing `FullAEvalKey` call sites are byte-for-byte
unchanged, existing `SafeExactCache`/`Dict`/`Nothing` callers see zero behavior change.
`Delta_minus_delta` is recomputed against the CALLER's current δ on every hit, never returned
stale.

**Scope decision, disclosed**: `find_smallest` stays IN the key (not stripped alongside δ).
Doing so correctly is possible in principle (`H[1,1] = K_hard * (-1)^stored_find_smallest`,
re-flip for the caller's `find_smallest`) but adds a second sign-reconstruction on a
screening-relevant field, for a case that doesn't arise in production — a single
`run_profile_checkpointed`/`run_polish_checkpointed` continuation never changes bound direction
mid-run. Stripping only δ delivers the brief's actual motivating scenario at zero extra risk.

**Verified**: D=4 (36 assertions — every δ-independent field, at 3 different δ values, plus a
genuinely-different-`x_free` still misses) and real D20/W=80000 (a value stored at δ=2 and
looked up at δ=5 matches a fresh 12.1s cold solve exactly on `Delta_dual`/`θ_full`/
`moment_resid`, with `Delta_minus_delta` correctly reflecting δ=5 — cache lookup 0.00017s vs
12.11s fresh, **~70,000x**). Re-verified passing after the mid-session rebase.

## 9. Benchmark tables

| Metric | Before | After | Change |
|---|---:|---:|---:|
| `build_lfix_base_cache` (calibration) | 1078 MB (pre-existing baseline) | 650 MB (prior session's fix, not this session's) | -40% (attribution: prior session) |
| Full serial gradient (400 coords, real D20/W80000) | 4489.9 MB | 756.5 MB (pooled, `h_mode=:fixed`) | **-83.2%** (4.91x) |
| Cache-hit lookup (cross-δ, same x_free, different δ) | 12.11s (fresh solve, no reuse possible) | 0.00017s | **~70,000x** |
| CM checkpoint/resume overhead | N/A (no checkpoint existed) | negligible (checkpoint write ~ms, resume adds one context rebuild ~65-83s, same as any fresh `run_*_checkpointed` start) | new capability |
| Peak RSS, one real D20/W80000 benchmark process (ctx build + `build_lfix_base_cache` + allocation-audit calls incl. one full 4.49GB-allocating serial gradient) | — | 3.33 GB (`VmHWM`, `/proc/self/status`) | first real number for this dimension this session — not independently isolated per-stage (ctx build alone vs. gradient alone), disclosed as a coarse measurement, not a full breakdown |

**Not done this session** (disclosed, not silently skipped): a full stage-by-stage GC-time/
allocation-count table (only total bytes were measured, via `@allocated`/`@timed`, not
`Base.gc_num()` deltas for GC time specifically); a systematic D=4 randomized-equivalence battery
across many random points (the D=4 correctness checks done — cross-delta cache's 36 assertions —
used one calibration-adjacent point, not a large random battery); RSS before/during/after broken
out per call rather than one coarse peak-for-the-whole-process number.

## 10. Correctness summary

| Change | Test file | Points tested | Result |
|---|---|---|---|
| Gradient workspace pool | `test_gradient_workspace.jl` | 2 real D20/W80000 (calibration, nearby perturbed), serial+threaded | 9/9 pass, bit-identical |
| Cross-δ exact cache | `test_cross_delta_cache.jl` | 1 D=4 point × 3 δ values + 1 miss check | 36/36 pass |
| Cross-δ exact cache (D20 spot check) | ad-hoc script, not committed as a test | 1 real D20/W80000 point, δ=2 store / δ=5 hit | Delta_dual/θ_full/moment_resid exact match vs fresh solve |
| CM checkpoint schema | `test_cm_checkpoint_original.jl` + `_resume.jl` | 1 real D20/W80000/L=10 interrupt+resume, 2 separate processes | 4/4 pass |
| 8 pre-existing/backfilled regression suites (parallelism guards, checkpoint schema-3, CM interval equivalence, lfix buffer bit-identity, direction bounds, incumbent seeding, KNITRO status decoder, per-solve counters) | existing files, re-run independently on this branch | — | All pass (independently re-verified, not merely trusted from the other session's own report) |

## 11. Production integration recommendation

| Change | Recommendation |
|---|---|
| `gradient_workspace.jl` (`GradWorkspace`/`GradWorkspacePool`/`composite_gradient_at_fast_pooled`) | **MERGE** — purely additive, bit-identical verified serial+threaded at 2 real D20 points, 4.91x allocation reduction with no wall-time regression observed (not independently re-timed end-to-end this session beyond the allocation comparison itself — flag for a coordinating session to add a wall-clock A/B before treating the speed claim as proven, not just the allocation claim). |
| `cross_delta_cache.jl` (`CrossDeltaExactCache`) | **MERGE** — purely additive, zero behavior change for existing callers, real ~70,000x win on the brief's own motivating staged-δ scenario, verified at D=4 and real D20. Wiring it INTO `c10_d20_production_driver.jl`'s actual staged-δ continuation scripts (replacing a fresh `SafeExactCache` per stage) is a separate, small follow-up not done this session (the cache type exists and is proven; nothing currently constructs one in the real staged-continuation driver code). |
| `cm_checkpoint.jl` (`CMCheckpoint`/`run_cm_upper_checkpointed`) | **MERGE AFTER LONG-RUN TEST** — mechanically verified end-to-end at L=10/short budget; a coordinating session should run a REAL long L=50 CM campaign with this wrapper (matching the multi-hour real production runs this repo actually does) before fully trusting it under production wall-budgets/checkpoint-interval tuning. |
| Organic-failure capture/replay extension | **KEEP DIAGNOSTIC ONLY / NOT DONE** — no fresh capture this session; the existing mechanism (from `diag/fullA-driver-delta5`) is unmodified. Flag as the clearest remaining follow-up: either search from the sequential method's own trajectory (where the known real pathologies actually came from) or accept a longer live search budget than this session used. |
| Deeper GC-time/RSS-per-stage breakdown | **KEEP DIAGNOSTIC ONLY / NOT DONE** — only one coarse whole-process peak-RSS number was collected. |

Suggested independent commits for whoever merges (already separated on this branch,
`git log --oneline` shows exactly this order): (1) `gradient_workspace.jl` + its test,
(2) `cross_delta_cache.jl` + its test, (3) `cm_checkpoint.jl` + its two-part test. No commit
bundles unrelated changes.

## 12. Rollback

`diag/fullA-d4-exact` is untouched (this branch was never merged into it). If any of this
branch's 3 commits need reverting individually after a future merge, they are cleanly
separated (see §11) — `git revert` any one of `5595021`/`e848b1a`/`b0b2f33` (or their
post-rebase hashes) without affecting the other two, since none of the three touch a
shared file.

## Answers to the brief's 10 closing questions

1. **Why did `build_lfix_base_cache` still allocate 590-650 MB?** It doesn't, redundantly — that
   figure is the prior session's own already-fixed number (was 1078MB), and this session's own
   audit found no further comparably-sized redundant copy inside it; the remainder is required
   persistent output (§5).
2. **How much of that is unavoidable persistent output?** All of it, per this session's audit —
   `price0`/`pTσ0` (512MB) + winner tables (~77MB) are genuinely read by every subsequent
   coordinate probe.
3. **Why did one gradient allocate ~4 GB?** `dest_contrib_incremental_o1`/`cf_contrib_at` (called
   through the ALREADY-reused `q_bufs`/`psi_bufs`) still called the ALLOCATING
   `price_and_pTsigma_cell` and allocated fresh `contrib`/`cf` W-length vectors every single
   coordinate probe — not the buffers themselves, but what they fed into.
4. **Which exact call sites caused most of it?** `dest_contrib_incremental_o1` (1.96MB/call) +
   `cf_contrib_at` (1.28MB/call), each called 2-4× per coordinate via `a_block_fd_component!`'s
   Lp/Lm probes — §3/§4.
5. **How much allocation was removed through persistent buffers?** 4.91x per gradient call
   (4489.9MB → 756.5MB), measured live at real D20/W=80000.
6. **Did lower allocation improve wall time and thread scaling?** Not independently re-timed
   end-to-end this session (disclosed gap, §11) — only the allocation reduction itself is
   proven bit-identical and measured; a wall-clock A/B is a flagged follow-up.
7. **Are CM runs now using the full checkpoint schema?** Yes, via a NEW opt-in
   `run_cm_upper_checkpointed` wrapper (`run_cm_upper` itself unchanged) — verified end-to-end
   with a real interrupt+resume in separate processes at L=10.
8. **Can exact inner results and successful duals safely persist across δ?** Confirmed for the
   exact-cache (this session, `CrossDeltaExactCache`, root-caused and verified). The
   successful-dual bank's own cross-δ persistence was NOT investigated this session (out of
   scope given time; `dual_bank_ab_harness.jl` exists on the merged base for a future session
   to use for exactly this benchmark).
9. **Was a real organic `-300` captured and replayed?** No fresh one this session (disclosed,
   §6) — a live fire-test found none, and a known historical pathology point is now caught by
   an improved screen before reaching the inner solve at all (a positive but orthogonal
   finding).
10. **Which changes should enter production before the next long production campaign?** The
    gradient workspace pool and cross-δ cache (both MERGE-ready per §11); the CM checkpoint
    schema after one real long L=50 run exercises it under production wall-budgets.
