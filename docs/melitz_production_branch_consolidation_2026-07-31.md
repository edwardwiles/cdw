# Melitz production-branch consolidation (2026-07-31)

Integration branch: `integration/melitz-production-consolidation-2026-07-31`
Worktree: `/bbkinghome/edav/gravity_robustness/worktrees/integration-melitz-production-consolidation-2026-07-31`
Canonical base: `6e803d43875e73deb3cf8acf3c04e8febd808e53` (`6e803d4`)
Proposed production HEAD (tip of this branch as of this writeup): see "Exact commits" below.

This document covers Phases 1-9 of the consolidation task. Phases 1-2 (branch/worktree
inventory, outstanding-commit classification) were established by a parallel investigation
before this session started; they are restated concisely here with this session's own
verification, not re-derived from scratch.

## Ancestry proof

```
$ git merge-base --is-ancestor 6e803d4 HEAD && echo OK
OK
$ git log --oneline 6e803d4..HEAD
53f98fa Fix pre-existing segfault in "bundle pool members are independently mutable" test
9ce8a18 Phase 9: checkpoint code-version enforcement (resume gate + migration)
417fbde Phase 8: regression tests for production_launcher.jl refusal/pass-through behavior
36ee669 Fix segfault in the new MelitzCCBundle sentinel-safety test (missing operator equilibration)
33a7d66 Fix Cmd+ignorestatus construction in melitz_check_ancestry
b2a396b Fix Cmd construction bug in production_launcher.jl (git shell-out)
3ccab3d Phase 8: canonical Melitz production launcher (git/ancestry/toolchain/fingerprint preflight)
d1b1215 Phase 4.2/4.3 regression tests + Phase 5 allocation-ceiling fix and group test runner
2cc370f Melitz perf audit: eliminate two real allocations, fuse mul_G! hot loop, correct docs
fc1e930 Melitz legacy-H removal audit: fix one dormant dense-fallback gap, add structural gates
```

Every commit on this branch descends from `6e803d4`; `6e803d4` is a genuine ancestor (not
merely "close to" or "equal to" it in some looser sense).

## Phase 1-2 restated (established before this session, re-verified here)

- Two Melitz-relevant branches: `melitz/fullD-delta-star` (successful overnight campaign,
  tip advanced to `20e9f2c`/`64a0e2a` after this integration's base was fixed -- those two
  commits are explicitly NOT part of what this integration builds on, per the governing
  prompt's own decision) and `audit/melitz-legacy-H-removal-2026-07-31` (tip `6b65b1d`).
- Both branches diverge from `d0904c7`. The audit branch has exactly 3 commits past
  `d0904c7`: `358d3d1` (source fix + structural gates), `68aa6be` (docs/logs only, NOT
  ported), `6b65b1d` (perf: empty-sentinel + `mul_G!` fusion).
- Re-verified this session: `git diff --stat 6e803d4 6b65b1d` confirms `358d3d1` and
  `6b65b1d` touch only `src/melitz/delta_star.jl`, `src/melitz/include_melitz.jl`,
  `src/melitz/production_structural_gate.jl` (new), `src/melitz/cc_bundle.jl`,
  `src/melitz/moment_operator.jl`, plus their own new scripts/docs -- none of these were
  touched between `d0904c7` and `6e803d4`, so cherry-picking both onto `6e803d4` produced
  zero textual conflicts (confirmed live: `git cherry-pick -x` succeeded cleanly for both).
- Two experimental branches (`experiment/melitz-full-outer-geometry-constraints-2026-07-30`,
  `experiment/melitz-profiled-q-envelope-gradient-2026-07-31`) were reviewed per the
  governing prompt's instruction; nothing in either was found to be a clearly
  production-ready item warranting explicit flagging for this integration -- both remain
  exploratory/inconclusive as characterized in the task brief, and nothing from them was
  merged.

## Phase 3: source duplication / include-structure audit

**Include structure.** `src/melitz/include_melitz.jl` is the single aggregator, 65 `include(...)`
calls in a fixed order (doubleDiff.jl must be included separately first, by convention,
before this file). `test/melitz/runtests.jl` does **not** call `include_melitz.jl` -- it
duplicates the same include list line-for-line inside the test file itself. This is a
**real, pre-existing duplication** (not introduced by this consolidation): every new file
added to `include_melitz.jl` must also be manually added to `runtests.jl`'s own copy, or it
silently gets zero test coverage (confirmed this happened at least once before: `runtests.jl`
had its own header comment noting `lfd_preserving_state.jl` was added to `include_melitz.jl`
on 2026-07-30 but not to `runtests.jl` until later). This session's own two new files
(`production_launcher.jl`, `checkpoint_versioning.jl`) were added to **both** lists
deliberately, to not repeat that gap.

**No campaign script found** that (a) includes an alternate copy of a production function,
(b) redefines a production method after `include_melitz.jl`, (c) depends on process working
directory in a way that silently resolves to the wrong tree (the one dynamic-path pattern
found, `@__DIR__`-relative `joinpath`s, is used correctly throughout src/melitz -- the one
place this went wrong was in THIS session's own new `scripts/melitz_test_group_runner_2026-07-31.jl`,
which parses `runtests.jl`'s source via `Meta.parse` rather than `include`; `@__FILE__`/`@__DIR__`
inside the parsed forms would have resolved to the RUNNER's own location rather than
`test/melitz/`'s, causing a wall of `SystemError: opening file ".../src/melitz/profiling.jl"`
-- fixed by passing `filename=RUNTESTS_PATH` to `Meta.parse`, restoring correct resolution;
this is a bug in this session's OWN tooling, not in any production script), or (d) loads
source from an output/checkpoint directory.

**Duplicate top-level function/type definitions**, checked via a static scan of every
top-level (column-0) `function`/`struct` definition across `src/melitz/*.jl`: several names
ARE defined in two files (`_base_arg0!`, `_direct_coordinate_grad`, `_direct_coordinate_grad_sorted`,
`_direct_coordinate_grad_touched_row`, `fstar_equal_weight_moments`, `melitz_recover_lfd`,
`melitz_recover_lfd_from_solution`) -- in every case this is legitimate Julia multiple
dispatch (different argument type signatures, e.g. `melitz_recover_lfd(obj::MelitzCCBundle, ...)`
in `cc_bundle.jl` vs `melitz_recover_lfd(obj, ...)` -- generic, for the legacy dense bundle --
in `delta_star.jl`), not an accidental redefinition/shadowing bug. A handful of other
same-named matches (`cb_F!`, `cb_G!`, `build_state`) are indented (4-space) local closures
nested inside different enclosing functions in different files -- not top-level at all, no
conflict.

**Runtime `methods()`/`which()` audit** (actual Julia method table for a normal campaign
process, not a grep guess -- `src/melitz/production_launcher.jl`'s own
`melitz_source_location_manifest` now derives this same information programmatically for
every campaign launch):

| Entry point | file:line |
|---|---|
| `melitz_recover_lfd` (matrix-free) | `src/melitz/cc_bundle.jl:910` |
| `melitz_recover_lfd` (dense, generic) | `src/melitz/delta_star.jl:904` |
| `melitz_recover_lfd_from_solution` (matrix-free) | `src/melitz/cc_bundle.jl:827` |
| `melitz_recover_lfd_from_solution` (dense) | `src/melitz/delta_star.jl:808` |
| `mul_G!` | `src/melitz/moment_operator.jl:280` |
| `mul_Gt!` | `src/melitz/moment_operator.jl:343` |
| `melitz_full_weighted_gram!` (structured Hessian, serial) | `src/melitz/moment_operator.jl:609` |
| `melitz_full_weighted_gram_parallel!` (structured Hessian, parallel) | `src/melitz/moment_operator.jl:718` |
| `melitz_fixed_q_middle_constraint_system` (fixed-q A middle solve) | `src/melitz/fixed_q_a_middle_loop.jl:265` |
| `melitz_middle_objective_and_gradient!` | `src/melitz/fixed_q_a_middle_loop.jl:445` |
| `melitz_run_welfare_plus_a_sequential_search` (profiled welfare continuation) | `src/melitz/matched_effort_controller.jl:34` |
| `melitz_reduced_q_propose_direction` (reduced-q direction builder) | `src/melitz/reduced_q_subspace.jl:409` |
| `melitz_reduced_q_propose_direction_threaded` | `src/melitz/reduced_q_threaded_direction.jl:93` |
| `melitz_assert_production_bundle!` (production verifier) | `src/melitz/production_structural_gate.jl:92` |
| `melitz_live_backend_manifest` | `src/melitz/production_structural_gate.jl:28` |
| `MelitzCCBundle` functor (`obj(x)`, the KNITRO inner callback) | `src/melitz/cc_bundle.jl:284` |
| `build_melitz_psi_bundle` | `src/melitz/delta_star.jl:603` |
| `build_melitz_cc_bundle` | `src/melitz/cc_bundle.jl:243` |
| `evaluate_melitz_delta` | `src/melitz/delta_star.jl:997` |
| `evaluate_melitz_delta_from_solution` | `src/melitz/delta_star.jl:1098` |

## Phase 4: integration diff summary

1. **`358d3d1`** cherry-picked clean (commit `fc1e930`). Fixes
   `evaluate_melitz_delta_from_solution`'s `store_G=true` branch (was a guaranteed
   `FieldError` for `MelitzCCBundle`; now dispatches through `melitz_bundle_dense_G_at_theta`
   like its sibling `evaluate_melitz_delta`). Its own regression test
   (`scripts/audit_phase3_storeG_fix_regression_2026-07-31.jl`) run live: **PASS** (`max|G_from_fix
   - G_from_known_diagnostic_path| = 0.0`, bit-identical).

2. **Empty-sentinel safety proof** (before integrating `6b65b1d`'s cc_bundle.jl half): grepped
   every call site of the `MelitzCCBundle` functor across `src/`, `scripts/`, `test/` (~40
   matches). Every real call site either omits `g`/`h`/`constr`/`jac` entirely (using the
   default) or passes a caller-owned buffer explicitly (`obj(x, g; h=H)`, `obj(x,
   constr=localc)`) -- confirmed by reading the functor body itself (`cc_bundle.jl:284-360`):
   every read/write of `g`/`h`/`constr` is gated behind `length(...) > 0` BEFORE any access;
   the `θ` argument is only ever length-checked (`length(θ)==0 || error(...)`), never
   written. No call site anywhere captures the sentinel and mutates it later. `6b65b1d`
   cherry-picked clean (commit `2cc370f`).

3. **New regression tests added** (commit `d1b1215`):
   - `mul_G!` loop-fusion direct-kernel equivalence, D=4, wired into `test/melitz/runtests.jl`
     (the ported commit's own `scripts/perf_audit_mulG_fusion_verify_2026-07-31.jl` only ever
     ran as a standalone real-D20 script, never as part of the regression suite). A literal
     reimplementation of the ORIGINAL two-pass algorithm is run against the fused version
     across 20 random trials; asserts `isapprox(...; atol=1e-9, rtol=1e-9)` per trial and a
     worst-case bound `< 1e-8` across all trials (headroom above the ~1e-14-scale
     reassociation difference documented at real D=20/W=80,000).
   - `MelitzCCBundle` shared empty-sentinel mutation-regression guard: exercises every real
     call pattern (200×2 calls omitting kwargs in every combination), asserts the sentinels
     remain length-0 and the SAME object (`objectid` unchanged) afterward; a real (nonempty)
     buffer call is confirmed to still write correctly and is confirmed NOT to be the
     sentinel object; a direct attempted in-place mutation (`_MELITZ_CC_EMPTY_VEC[1] = 1.0`)
     is confirmed to raise `BoundsError` (fails loudly, never silently corrupts) --
     deliberately does NOT call `push!` on the real shared const (would permanently corrupt
     it for the rest of the process; demonstrated on a `copy()` instead).

4. `68aa6be` (docs/logs only) was **not merged** -- no source dependency, per instructions.

## Phase 5: test harness

**Allocation-ceiling investigation** (own live measurement, not assumed from the prior
audit's report, though it corroborates it): on this Julia 1.12.6 install,
`Threads.@threads :static` itself allocates real per-call scheduler machinery. Measured via
`@allocated` + `Profile.Allocs` at `nthreads` = 4, 8, 20, for two DIFFERENT parallel
gradient-backend kernels (`make_melitz_gradient_delta_direct_parallel`,
`make_melitz_gradient_delta_direct_sorted_parallel`) at TWO problem sizes each (D=4/W=20,000
fixture and real D=20/W=80,000):

| nthreads | plain-parallel (D=4 or D=20, same value) | sorted-parallel (D=4 or D=20, same value) |
|---|---|---|
| 4 | 5,296 B | 5,968 B |
| 8 | 8,912 B | 9,968 B |
| 20 | 19,760 B | 21,968 B |

For a FIXED kernel, bytes/call is **exactly identical** across a >25x change in problem size
(confirmed by running the same function object on both fixture sizes at nthreads=8: 8,912 B
both ways for the plain-parallel kernel, 9,968 B both ways for the sorted-parallel kernel) --
i.e. genuinely nthreads-scaling, not D/W/moment-count-scaling. `Profile.Allocs` at nthreads=4
confirms the allocation is exactly `4× Base.Threads.SpinLock`, `4× Base.GenericCondition`,
`4×` the per-region closure object, `4× Task`, `4× Base.IntrusiveLinkedList{Task}` -- scheduler
objects, not a workspace/boxing allocation. The 8,912-byte figure at nthreads=8 matches the
prior audit's own reported number exactly.

Fix (commit `d1b1215`): replaced both exact-`bytes==0` assertions with
`melitz_threads_alloc_ceiling(nthreads) = 1200*nthreads + 4000` bytes, derived from a linear
fit to the measured points (slope ~900-1000 B/thread, intercept ~1700-2000 B) with >=2,800
bytes of headroom at every measured point. Serial kernels
(`make_melitz_gradient_delta_direct_serial`, `make_melitz_gradient_delta_direct_sorted_serial`,
every other hot-path routine) remain held to strict, exact `bytes==0` -- confirmed unchanged
live (0 bytes at both D=4 and D=20 for both serial functions).

**Group-by-group test runner** (`scripts/melitz_test_group_runner_2026-07-31.jl`, commit
`d1b1215`): splits `runtests.jl`'s source into top-level forms via `Meta.parse`, evaluates
each independently under its own `try`/`catch`, so one failing/erroring top-level `@testset`
(which throws `Test.TestSetException` on `finish()` and would otherwise abort the entire file
-- confirmed this IS what was happening: Julia's `Test` stdlib throws at the outermost
testset in a chain if it recorded any failure/error, and nothing at file-top-level catches
that) no longer blocks every subsequent group. Documented granularity caveat: operates at the
level of top-level FORMS, not every individual `@testset` -- a handful of top-level `if
KNITRO_AVAILABLE ... end` blocks lexically contain more than one `@testset`; if the first one
inside such a block fails, siblings inside the SAME `if` block do not run that pass (but
every other top-level form in the file, the overwhelming majority, is unaffected).

**Two real segfaults found and fixed while shaking this runner out** (both are the SAME
underlying failure mode, found independently twice):
- **Own new test** (sentinel-safety test, Phase 4): built a fresh `MelitzCCBundle` and called
  its functor with random `x` WITHOUT first calling `melitz_update_operator_at_theta!`.
  `build_melitz_moment_operator` zero-initializes `order` (among other fields);
  `mul_G!`'s `@inbounds` loop then reads `order[m,o]==0` and indexes `trade_index[o, 0]` --
  an out-of-bounds index 0 that `@inbounds` does not catch, corrupting memory instead of
  throwing `BoundsError`. Fixed (commit `36ee669`) by equilibrating the operator first;
  verified in isolation (200 iterations, no crash) before relaunching.
- **Pre-existing test** (NOT touched by this consolidation, present on `6e803d4` too): the
  "Gate 2/3 matched-effort + threaded direction infrastructure" testset's own "bundle pool
  members are independently mutable" sub-test built a 2-member bundle pool via
  `melitz_build_thread_bundle_pool` and called `pool3[2](ge_x0)` without ever equilibrating
  `pool3[2].op` (`melitz_build_thread_bundle_pool`'s own docstring: the factory only
  constructs, the CALLER equilibrates). The test's own assertion
  (`Dp_2_unperturbed ≈ ge_lfd0.Delta`) had therefore been silently comparing against
  undefined-behavior garbage whenever it happened not to crash outright. Fixed (commit
  `53f98fa`) by equilibrating `pool3[2].op` at the shared anchor `ge_theta0` before
  `pool3[1]` is perturbed away from it; verified in isolation (values now match exactly, no
  crash) before relaunching. **This is a genuinely pre-existing bug, unrelated to any ported
  commit, that this session found and fixed as a byproduct of getting the full suite to
  actually complete** -- flagged explicitly for the orchestrating session's awareness (see
  "Draft Phase 10 input" below), since fixing a pre-existing test bug was not explicitly
  requested by the governing prompt but was necessary to produce a complete group matrix.

**Group-by-group matrix**: see "Test-group matrix results" in the final report (this
document is being finalized while the 4th full-suite background run is still in flight;
the orchestrating session should re-run
`julia -t 20 --project=. scripts/melitz_test_group_runner_2026-07-31.jl` if a completed run
is not yet available in `docs/key_results/melitz_test_group_matrix_2026-07-31.txt` --
partial results through 3 of 4 attempts show 100% pass rate on every group reached, the two
segfaults above being the only failures found, both now fixed).

**Flaky/solver-trajectory-sensitive tests**: no test comparing exact optimizer path/iteration
count was found beyond what already existed pre-session (this consolidation did not add
any); the pre-existing tests already compare typed classification, Delta/objective values, or
economic state within tolerance (e.g. `isapprox(...; rtol=1e-8)`), not raw iteration counts.
The new tests added this session follow the same convention (see Phase 6 gates below, which
explicitly do NOT require identical KNITRO iteration counts).

## Phase 6: integration correctness gates

All gates run identically against a clean checkout of the canonical base
(`melitz-consolidation-baseline-6e803d4` worktree, `6e803d4`) and this integration branch
(tip at time of each gate run noted below), same process/thread configuration each time.

1. **`mul_G!` kernel equivalence**: covered by the new D=4 unit test (Phase 4, passes) and
   the pre-existing real-D20 `scripts/perf_audit_mulG_fusion_verify_2026-07-31.jl` (max abs
   diff 1.4e-14, max relative 4.9e-16, already verified by the ported commit itself, not
   re-derived here).

2. **Objective/gradient/Hessian callback equivalence** (D=4 fixture and real D=20/W=80,000,
   at the recovered LFD dual point):

   | quantity | D=4 baseline | D=4 integration | D=20 baseline | D=20 integration |
   |---|---|---|---|---|
   | objective-only `f` | -7.554508757e-6 | -7.554508757e-6 (identical) | -4.07206993064e-4 | -4.07206993064e-4 (identical) |
   | `\|g\|` (obj+grad) | 4e-15 | 4e-15 | 3.5e-14 | 3.3e-14 (ULP-level) |
   | `\|h\|` (obj+hess) | 2.096321025057869 | 2.096321025057869 (identical) | 6.691752839987524 | 6.691752839987522 (ULP-level, ~2e-15 relative) |
   | `sum(h)` | 11.631733965203878 | 11.631733965203878 (identical) | 42.926235323263619 | 42.926235323263619 (identical) |

3. **D=4 complete inner solve**: `Delta = 0.000007554509`, `lfd_ok=true`, `nStatus=0` --
   identical on both branches.

4. **Real D=20 complete inner solve** (anchor point, `real_data/noah_D20`, same recipe as
   `scripts/audit_phase1_2_live_bundle_2026-07-31.jl`): `Delta = 0.000407206993`, `lfd_ok=true`,
   `nStatus=0` -- identical on both branches.

5. **Representative fixed-q A middle solve**: `Delta_incumbent = Delta_start_verified =
   0.000407206993064`, `unique_A_points=12`, `unique_inner_solves=12` -- **bit-identical**
   on both branches.

6. **Cold verification of the current best upper and lower (delta=0.5) incumbents**
   (`phase6_upper_m1`/`phase6_lower_m-1`, the campaign's own final-verified points, GT=
   8.849341%/0.272266% per `docs/melitz_d20_profiledA_overnight_qpoll_delta0p5_2026-07-31.md`),
   using the existing `scripts/melitz_overnight_final_verify_2026-07-31.jl` (parameterized
   copies pointed at each worktree):

   | | upper (`phase6_upper_m1`) | lower (`phase6_lower_m-1`) |
   |---|---|---|
   | classification | `FiniteSolved` both | `FiniteSolved` both |
   | GT | 8.84934137951131% both | 0.2722655222019532% both |
   | Delta* (fresh, cold) | 0.4981225098057109 (int.) / 0.49812250980571005 (base) | 0.1137638312270617 (int.) / 0.11376383122706066 (base) |
   | q/f reconstruction drift | 0.0 both | 0.0 both |
   | A-gravity residual | 1.3322676295501878e-15 both | 7.771561172376096e-16 both |
   | nStatus | 0 both | 0 both |

   Delta* for the FiniteSolved incumbents agrees to ~1e-16 relative (ULP-level, matching the
   documented `mul_G!` reassociation difference). The "fixed original-calibration A/f" and
   "nearest-anchor profiled A/f" comparison points (both `AboveEvaluationCap`) differ by
   ~1e-8 to ~1e-9 absolute (e.g. 10.882049075542717 vs 10.882048996197465) -- LARGER than
   pure ULP noise but both branches still agree to 8-9 significant digits and both classify
   identically as `AboveEvaluationCap`. This is a **changed optimization trajectory**
   (the `AboveEvaluationCap` certified bound is `-f` at the LAST evaluated point before the
   cap fires, which is path-dependent given the reassociated `mul_G!` arithmetic changes
   which exact intermediate iterate gets capped) -- reported separately from the verified
   `FiniteSolved` results per the governing prompt's own instruction, not treated as a
   correctness regression.

7. **Short profiled welfare continuation segment** (Method A, D=4 fixture, 2 stages, capped):
   `incumbent.objective = -0.042387309873331`, `n_stages=2` -- bit-identical on both branches.

8. **Checkpoint/resume**: this is a NEW capability introduced on this integration branch
   (Phase 9); the base commit has no equivalent versioned-checkpoint mechanism to compare
   against. Verified via Phase 9's own dedicated regression suite (6 test groups against a
   real, disposable scratch worktree -- see Phase 9 below), not a base-vs-integration
   behavioral diff.

**No KNITRO iteration-count equality was required or claimed** anywhere in this gate battery
-- every comparison above is against typed classification / Delta* / moments / A-gravity
residual / GT, matching the governing prompt's own tolerance rules.

## Phase 7: performance gates

D=20, W=80,000, same process/thread configuration (`-t 20`, `BLAS.set_num_threads(1)`),
median + dispersion over multiple repetitions, canonical base vs integration branch:

| kernel | reps | baseline median | integration median | note |
|---|---|---|---|---|
| `mul_G!` (tight loop, 500 calls/rep) | 10 | 3.337e-3 s/call | 3.659e-3 s/call | see caveat below |
| objective callback (`obj(x)`) | 200 | 5.564e-3 s | 4.120e-3 s | ~26% faster, consistent w/ sentinel fix |
| gradient callback (`obj(x,g)`) | 200 | 9.360e-3 s | 8.159e-3 s | ~13% faster, consistent w/ sentinel fix |
| Hessian callback (`obj(x,;h=h)`) | 200 | 1.826e-2 s | 1.793e-2 s | ~2% faster, within noise (dominated by O(D^2) construction) |
| **complete inner solve (whole-solve)** | 10 | 2.125e-1 s | 1.997e-1 s | ~6% faster |
| representative middle solve (max_evals=10) | 3 | 2.757 s (std 3.64) | 2.942 s (std 3.55) | inconclusive, n=3, one rep per branch hit a benign KNITRO eval-error retry |

**`mul_G!` caveat, reported honestly rather than papered over**: this session's own
re-measurement of the isolated kernel (both a per-call `@elapsed` loop and a tight
500-calls-per-`@elapsed` loop matching the ORIGINAL audit script's own methodology) did
**not** cleanly reproduce the previously-documented 1.41x speedup at real D=20/W=80,000 --
medians came out ~10% SLOWER on the integration branch in both methodologies, but with wide,
overlapping dispersion (integration range 2.74-4.94ms vs baseline range 3.10-3.53ms across
different runs) consistent with host-level scheduling/thermal noise on this shared
208-core machine dominating a true microsecond-scale kernel difference measured in
millisecond-granularity wall-clock. The prior audit's own correctness-focused measurement
(`scripts/perf_audit_mulG_fusion_verify_2026-07-31.jl`, run once per process rather than
across separate OS processes) is taken as the more reliable source for the ISOLATED kernel
claim; this session's own numbers should be read as "no confident directional claim
recoverable from repeated-process wall-clock at this scale," not as a contradiction proven to
a similar level of confidence.

**Whole-solve vs isolated-kernel, reported separately as required**: the isolated mul_G!
kernel shows no confidently-recoverable speedup by this session's own measurement (above),
while the WHOLE inner solve is measurably faster (~6%, complete real-D20 `melitz_recover_lfd`
solves, 10 reps each, medians 0.1997s vs 0.2125s) and the isolated objective/gradient
callbacks show a clearer, more plausible improvement (13-26%, consistent with the empty-
sentinel allocation elimination on the single hottest call site in the codebase, which this
whole-solve number partially reflects). **No whole-solve speedup is claimed to derive solely
from the isolated `mul_G!` kernel number** -- the two are reported and reasoned about
separately, per the governing prompt's explicit instruction.

## Phase 8: canonical production launcher

`src/melitz/production_launcher.jl` (new) + `scripts/melitz_production_launcher.jl` (thin
runnable entry point). `melitz_production_preflight!`:

1. Resolves+prints git common dir, worktree path, branch, commit SHA, tracked-tree dirty
   status (shells out to `git`; untracked files explicitly excluded via
   `--untracked-files=no`).
2. Refuses to launch (`MelitzLaunchRefusal`, not a generic error) if TRACKED files are
   modified, unless `allow_dirty=true` is passed explicitly.
3. Verifies current HEAD descends from an approved production base commit via `git
   merge-base --is-ancestor`, read from a small git-tracked config file
   (`melitz_production_approved_base.txt`, currently `6e803d4` -- this integration's own
   canonical base) or an env override -- NOT a hardcoded SHA in source.
4. Derives file:line source locations for the same production entry points audited in Phase
   3, via the real Julia method table.
5. Records Julia version, live KNITRO version (`KN_get_release`), Julia thread count, BLAS
   thread count, and SHA-256 content fingerprints of the real-data CSVs and `.opt` files in
   use.
6. Packages all of this into a `MelitzProductionManifest`, writable to any checkpoint/result
   directory (`write_manifest_to` kwarg).

**Live end-to-end smoke test** (against this exact worktree, clean tree): prints a full
manifest, `PREFLIGHT PASSED`. Two real bugs were found and fixed while shaking this out
(`Cmd(::Vector{Any})` has no valid constructor when mixing a backtick literal with strings in
an array; `Cmd(::Vector{String}; ignorestatus=...)` is similarly invalid -- both fixed to
build a plain `Cmd` first, then wrap for `ignorestatus`) -- both caught by actually RUNNING
the launcher against a real repo, not merely reading the code.

**Regression tests** (`scripts/melitz_test_production_launcher_2026-07-31.jl`), against a
real, disposable `git worktree add -d` scratch worktree (removed at the end regardless of
outcome) -- **not** a mocked git layer:
- `melitz_check_ancestry` unit-tested against real commit pairs (`d0904c7`->`6e803d4` is a
  real ancestor relationship; `91f5ec2`, a real unrelated old Ricardian diagnostic-branch
  commit, confirmed NOT an ancestor either direction via direct `git merge-base
  --is-ancestor` checks both ways).
- Pass-through on a clean scratch worktree.
- Untracked file does NOT block launch.
- Dirty TRACKED file causes refusal; `allow_dirty=true` passes through and is recorded
  (`diagnostic_override_used=true`).
- Wrong ancestry (env override to `91f5ec2`) causes refusal.
- `write_manifest_to` writes a real, readable manifest file.

All 5 test groups (17 `@test` assertions) **PASS** live.

## Phase 9: checkpoint code-version enforcement

`src/melitz/checkpoint_versioning.jl` (new). `MelitzCheckpointVersionInfo` records exact
source commit, a tracked-diff hash (`git diff HEAD`, distinct from Phase 8's dirty boolean --
distinguishes two dirty runs at the SAME commit from each other), the full Phase-8 manifest,
and data/options fingerprints. `melitz_save_checkpoint!` runs the Phase-8 preflight and
serializes `MelitzCheckpoint(version_info, payload)`. `melitz_resume_checkpoint` is the
resume gate:

- Same commit as current HEAD -> returns `(payload, version_info)` silently.
- Different commit, `migrate=false` (default) -> throws `MelitzCheckpointVersionMismatch`.
- Different commit, `migrate=true` -> requires a `cold_reverify(payload)::Bool` callback,
  called under CURRENT code. A `false` result, a thrown exception, or a non-`Bool` return all
  refuse migration (never silently trusts a stale checkpoint). A `true` result returns
  `(payload, migration_info)` recording BOTH old and new commits plus the reverification
  outcome/timestamp.

**Regression tests** (`scripts/melitz_test_checkpoint_versioning_2026-07-31.jl`), against a
real scratch worktree with a REAL second commit created on top (not a mocked commit
difference) -- 6 test groups, all **PASS** live:
1. Same-commit resume passes silently.
2. Different-commit resume without `migrate=true` refuses.
3. Different-commit resume with `migrate=true`, `cold_reverify` returns `true` -> migrates,
   records both old and new commit SHAs correctly.
4. `cold_reverify` returns `false` -> migration refuses.
5. `cold_reverify` throws -> migration refuses (not silently trusted).
6. `migrate=true` with no `cold_reverify` callback at all -> refuses.
Plus `melitz_tracked_diff_hash` determinism (stable on a clean tree, changes when tracked
content changes).

## Exact commits made this session

```
53f98fa Fix pre-existing segfault in "bundle pool members are independently mutable" test
9ce8a18 Phase 9: checkpoint code-version enforcement (resume gate + migration)
417fbde Phase 8: regression tests for production_launcher.jl refusal/pass-through behavior
36ee669 Fix segfault in the new MelitzCCBundle sentinel-safety test (missing operator equilibration)
33a7d66 Fix Cmd+ignorestatus construction in melitz_check_ancestry
b2a396b Fix Cmd construction bug in production_launcher.jl (git shell-out)
3ccab3d Phase 8: canonical Melitz production launcher (git/ancestry/toolchain/fingerprint preflight)
d1b1215 Phase 4.2/4.3 regression tests + Phase 5 allocation-ceiling fix and group test runner
2cc370f Melitz perf audit: eliminate two real allocations, fuse mul_G! hot loop, correct docs
fc1e930 Melitz legacy-H removal audit: fix one dormant dense-fallback gap, add structural gates
```

Proposed production HEAD = tip of `integration/melitz-production-consolidation-2026-07-31` at
the time this document is finalized (see the orchestrating session's own final commit list
for the exact SHA, since this document itself is committed as part of the branch and its own
commit is the true tip).

---

# Draft Phase 10 input (raw findings only -- not a decision)

This section is deliberately NOT a go/no-go recommendation. It lists blockers hit, surprises,
and anything not fully verified, for the orchestrating session to weigh.

**Go-supporting evidence:**
- Every Phase 6 correctness gate that completed shows bit-identical or ULP-level-identical
  results between the canonical base and this integration branch, at both D=4 and real
  D=20/W=80,000, including the campaign's own two final production incumbents
  (GT=8.849341%/0.272266%).
- Both ported commits' own regression tests pass live, and this session's own additional
  tests (mul_G! fusion equivalence, sentinel mutation guard) pass live.
- Phase 8/9 (new capabilities, not present on base) both have real, passing regression test
  suites exercising actual git plumbing, not mocks.

**Surprises / things that were NOT expected going in:**
- Two independent real segfaults (not mere test failures) were found while shaking out the
  Phase 5 group-test runner -- both the exact same failure mode (calling a `MelitzCCBundle`
  functor against an un-equilibrated `MelitzMomentOperator`, whose zero-initialized `order`
  field causes `mul_G!`'s `@inbounds` indexing to read `trade_index[o, 0]`, an out-of-bounds
  index that corrupts memory instead of throwing `BoundsError`). One was in this session's
  own new test (fixed). **The other was in PRE-EXISTING test code, unrelated to any ported
  commit, present on the canonical base commit `6e803d4` itself** -- this is a genuine latent
  memory-safety landmine in the existing test suite (not the production code path itself,
  since production code always equilibrates before calling the functor as far as this session
  found) that this session fixed as a byproduct of needing the suite to complete, but did not
  go looking for and has not exhaustively searched for other instances of. **The orchestrating
  session should treat "are there other un-equilibrated-functor-call landmines elsewhere in
  the test suite or in any campaign script" as an open question**, not resolved by this
  session's two fixes.
- The previously-documented `mul_G!` isolated-kernel 1.41x speedup did not cleanly reproduce
  under this session's own repeated-process wall-clock re-measurement (see Phase 7) -- traced
  to probable host-level noise on a shared machine rather than a real regression (the
  correctness-focused reassociation-diff measurement, which does not depend on wall-clock
  precision, still stands and was not re-litigated), but this is a genuine discrepancy
  between two independent measurement sessions that the orchestrating session should be aware
  of rather than assume resolved.
- `test/melitz/runtests.jl` duplicates `include_melitz.jl`'s own include list by hand rather
  than calling it — a pre-existing structural fragility (confirmed to have already caused at
  least one prior silent test-coverage gap) that this session worked around (added new files
  to both lists) rather than fixed structurally. A future session collapsing this into a
  single include call would remove an entire class of "forgot to add it to the test file"
  bugs, but was out of scope for this consolidation.

**What could NOT be fully completed/verified by this session:**
- **The Phase 5 full-suite group-by-group matrix was still running in the background
  (4th attempt, after fixing both segfaults) at the time this document was drafted.** Three
  prior attempts each got substantially further than the last before hitting one of the two
  segfaults (now both fixed); the 4th attempt's live progress at time of writing had passed
  every group reached with a 100% pass rate and no further crashes, but had not yet reached
  the end of the (8,374-line) test file. The orchestrating session should check
  `docs/key_results/melitz_test_group_matrix_2026-07-31.txt` for the completed matrix (written
  automatically when the runner finishes) and re-run
  `julia -t 20 --project=. scripts/melitz_test_group_runner_2026-07-31.jl` from this worktree
  if that file is stale/absent.
- Phase 7's middle-solve performance comparison is based on only 3 repetitions per branch
  (deliberately small, since each middle solve costs several seconds and one repetition per
  branch hit a benign KNITRO "grad_callback returned -502" evaluation-error retry that adds
  substantial, non-representative variance) -- treated as inconclusive rather than a claimed
  result, not padded out with more repetitions given this session's time budget.
- This session did not attempt to search src/melitz for OTHER `@inbounds`-guarded indexing
  sites that could exhibit the same "reads zero-initialized state as a valid index" failure
  mode as `mul_G!`'s `order[m,o]` -- the two instances found were both in test code
  (un-equilibrated operator misuse), not in the indexing logic itself, and fixing the
  underlying `@inbounds` lack of bounds-checking in production code was judged out of scope
  (would be a production hot-path change, not requested by the governing prompt, and the
  ACTUAL production code paths always equilibrate before calling, as far as this session's
  audit found).
