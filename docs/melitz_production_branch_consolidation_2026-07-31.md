# Melitz production-branch consolidation (2026-07-31)

Integration branch: `integration/melitz-production-consolidation-2026-07-31`
Canonical worktree (as of this document's finalization): `/bbkinghome/edav/cdw/melitz/integration-production-consolidation-2026-07-31`
(built by the background agent at `/bbkinghome/edav/gravity_robustness/worktrees/integration-melitz-production-consolidation-2026-07-31`,
then relocated — see "Repository migration" below — this file's own path history reflects
the same journey.)
Canonical base: `6e803d43875e73deb3cf8acf3c04e8febd808e53` (`6e803d4`)
**Proposed production HEAD: `00c1435f1af6f11147d4f3ce0f4fc07498678738` (`00c1435`)** — the
current tip of this branch, pushed to `origin` (`git@github.com:edwardwiles/cdw.git`).

This document covers Phases 1-9 of the consolidation task, plus this session's own Phase 10
decision. Phases 1-2 (branch/worktree inventory, outstanding-commit classification) were
established by a parallel investigation before the background agent's session started; they
are restated concisely with verification, not re-derived from scratch. Everything below the
"Repository migration" and "Phase 10" headings was added by the orchestrating session after
the background agent's own work (which stopped at commit `99f4cf2`) completed.

## Repository migration (orchestrating session, after the agent's own work)

Per an explicit user decision made mid-session, `cdw` (not `trade_robustness_modular`) is now
the sole go-forward home for Melitz work, under a new `cdw/melitz/` worktree directory. This
branch, along with `melitz/fullD-delta-star` and `audit/melitz-legacy-H-removal-2026-07-31`,
was pushed from `trade_robustness_modular` to the shared GitHub remote
(`git@github.com:edwardwiles/cdw.git`, already configured as a second remote in
`trade_robustness_modular` under the name `cdw` — both directories were always independent
local clones of the same remote, just badly out of sync: `cdw`'s own local copy of
`melitz/fullD-delta-star` was a stale ref more than 50 commits behind), then fetched into
`cdw` and checked out as a new worktree at `cdw/melitz/integration-production-consolidation-2026-07-31`.
File layout inside the repo is unchanged (same commit graph, same `src/melitz/` tree) — only
the working-copy location changed. `melitz/fullD-delta-star` itself was NOT re-checked-out in
`cdw` yet: it is still live-checked-out in `trade_robustness_modular`'s main worktree (git
does not allow the same branch checked out in two worktrees at once), which also still has a
5+ day old orphaned diagnostic process running in it (unrelated to this consolidation,
deliberately left untouched all session — see Phase 0). The `cdw`-side local branch ref for
`melitz/fullD-delta-star` has been fast-forwarded to match (`64a0e2a`) so it is at least
readable/diffable from `cdw` even though not checked out there.

Two bugs were found and fixed by the orchestrating session as a direct result of this
relocation, both now committed on this branch (`da7ae7f`, `00c1435`):
1. `scripts/melitz_test_production_launcher_2026-07-31.jl`'s wrong-ancestry test cases
   hardcoded commit `91f5ec2`, a real commit from an unrelated old branch in
   `trade_robustness_modular`'s history. That object was never reachable from any pushed
   branch tip, so it doesn't exist in the `cdw` clone — both sub-tests referencing it broke
   immediately (`fatal: Not a valid object name 91f5ec2`) the moment this branch was relocated.
   Fixed by fabricating a guaranteed-non-ancestor commit locally (`git commit-tree` on the
   empty tree, no parent) instead of depending on repo-specific history. Re-verified live,
   standalone: all 5 test groups / 17 assertions PASS.
2. The full test-group matrix committed by the background agent
   (`docs/key_results/melitz_test_group_matrix_2026-07-31.txt`, generated `06:25:31`) was
   stale: it was generated **one minute before** the agent's own missing-include fix
   (`99f4cf2`, `06:27:03`) landed, and still reflected the resulting single failure. The
   orchestrating session found the agent's own leftover rerun process (`group_runner_full5.log`,
   started `06:27` under a background PID that outlived the agent's own conversational turn —
   see `feedback-compaction-can-leave-orphaned-duplicate-background-jobs` memory for this
   exact pattern) still running, let it complete rather than duplicate the work (an
   accidentally-launched duplicate rerun was found and killed within a minute of starting it),
   and committed the clean result: **53/53 groups pass, 0 fail, 0 error.**

## Ancestry proof

```
$ git merge-base --is-ancestor 6e803d4 HEAD && echo OK
OK
$ git log --oneline 6e803d4..HEAD
00c1435 Record clean full-suite test-group matrix (post include-fix): 53/53 pass
da7ae7f Fix repo-portability bug in production_launcher regression test
99f4cf2 Fix pre-existing missing include in runtests.jl (reduced_q_switch_geometry.jl)
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
merely "close to" or "equal to" it in some looser sense). Separately: `merge-base(HEAD,
melitz/fullD-delta-star) == 6e803d4` exactly — this branch and the live campaign branch have
each added their own commits on top of the same shared point, a clean two-way divergence with
no rebase needed to reunite them (see Phase 10's proposed merge command).

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

**Group-by-group matrix — FINAL, clean**: `docs/key_results/melitz_test_group_matrix_2026-07-31.txt`,
regenerated by the orchestrating session after the missing-include fix (`99f4cf2`) landed
(the version originally committed here was generated one minute *before* that fix and still
showed its resulting failure — see "Repository migration" above for the full story):

```
SUMMARY	pass=53	fail=0	error=0
```

**53 of 53 testset-bearing top-level forms pass. Zero failures, zero errors.** Every group
reached across the full 8,000+-line suite, including the long real-KNITRO groups (a 313s
real-D20 CC inner-loop test, a 136s matrix-free nuisance-profile port test, etc.), completed
cleanly. The two segfaults documented above (own new test, and the pre-existing
"bundle pool members are independently mutable" test) were the only failures found across
this and prior attempts, and both are fixed and included in this clean run.

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

## Exact commits on this branch (6e803d4..HEAD)

```
00c1435 Record clean full-suite test-group matrix (post include-fix): 53/53 pass
da7ae7f Fix repo-portability bug in production_launcher regression test
99f4cf2 Fix pre-existing missing include in runtests.jl (reduced_q_switch_geometry.jl)
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

**Proposed production HEAD: `00c1435f1af6f11147d4f3ce0f4fc07498678738`**, on branch
`integration/melitz-production-consolidation-2026-07-31`, pushed to `origin`
(`git@github.com:edwardwiles/cdw.git`), worktree `/bbkinghome/edav/cdw/melitz/integration-production-consolidation-2026-07-31`.

---

# Phase 10: Consolidation decision

## Verdict: A. Safe integration complete

All selected fixes (`358d3d1`, `6b65b1d` — both audit-branch source commits) are integrated
cleanly with zero textual conflicts, all correctness gates pass at bit-identical or
ULP-level-identical precision including the campaign's own two final delta=0.5 incumbents, the
full test suite passes 53/53 groups with zero failures after fixing two pre-existing bugs
(both confirmed present on the canonical base itself, unrelated to anything ported), and the
new Phase 8/9 capabilities (launcher, checkpoint versioning) both have real, independently
re-verified, passing regression suites. **The integration branch is suitable to become the
canonical production branch.**

This verdict was reached only after independent verification, not by accepting the background
agent's self-report: I re-ran the Phase 8/9 regression scripts myself and found (and fixed) a
real bug the agent's own testing had not caught — a hardcoded foreign commit SHA in the
launcher test that broke the instant the branch was relocated to a different clone (see
"Repository migration" above) — and I re-ran the full test-group matrix myself after
discovering the committed one was generated one minute before its own fix landed.

## Known open items (carried forward, not blockers to this verdict)

1. **Un-equilibrated-functor memory-safety landmine, not exhaustively searched.** Calling a
   `MelitzCCBundle` functor against a `MelitzMomentOperator` that was never equilibrated at any
   theta causes `mul_G!`'s `@inbounds` loop to read a zero-initialized `order` entry and index
   `trade_index[o, 0]` — an out-of-bounds index `@inbounds` does not catch, corrupting memory /
   segfaulting instead of throwing. Two instances were found and fixed (one in a new test added
   this session, one pre-existing in the test suite, present on `6e803d4` itself, unrelated to
   any ported commit). No production code path was found to have this issue (production always
   equilibrates before calling), but neither this session nor the background agent
   exhaustively searched for other un-equilibrated-call sites in campaign scripts. **Treat as
   an open question for a future audit, not resolved.**
2. **`mul_G!` isolated-kernel wall-clock speedup did not cleanly reproduce.** The audit
   branch's own correctness-focused measurement (reassociation-diff, not wall-clock) still
   stands; a repeated-process wall-clock re-measurement this session came out ~10% slower with
   wide overlapping dispersion, attributed to host-level noise on a shared 208-core machine
   rather than a real regression, but not proven either way to the same confidence as the
   correctness result. The whole-solve (~6%) and callback-level (13-26%) speedups are measured
   independently and are more confidently attributable to the sentinel-allocation fix.
3. **Middle-solve performance comparison is inconclusive** (n=3 reps/branch, one rep per branch
   hit a benign KNITRO evaluation-error retry) — not padded out further given session scope.
4. **`test/melitz/runtests.jl` still hand-duplicates `include_melitz.jl`'s include list** rather
   than calling it directly — the root cause of both the `lfd_preserving_state.jl` gap
   (2026-07-30) and this session's `reduced_q_switch_geometry.jl` gap. Worked around (both new
   files added to both lists) but not fixed structurally; a future session collapsing this to
   a single include call would remove an entire class of "forgot to add it" bugs.
5. **`melitz/fullD-delta-star` has advanced past this integration's base** (`20e9f2c` then
   `64a0e2a` — an independent adversarial economic audit and its own self-correction,
   concluding the full maintained equilibrium holds at both incumbents under a corrected
   closure). Neither commit is part of this integration by explicit decision (the audit
   finding was live and unresolved when this task began; it resolved itself via a concurrent
   session's correction commit partway through this one). They are compatible with this
   integration (no file overlap) and should be picked up in the merge below or a follow-up.
6. **The other, currently-live `sigma3_W500k` five-family campaign** (10 real KNITRO
   processes, branch `campaign/resource-layout-test-sigma3-W500k-2026-07-30`, a completely
   different worktree) is unrelated to and unaffected by this consolidation, noted here only
   so a future reader doesn't mistake its concurrent existence for something this task touched.

## Proposed merge (not performed — informational only, per task instruction not to modify other branches)

`melitz/fullD-delta-star` (tip `64a0e2a`) and this integration branch share `6e803d4` as their
exact merge-base — a clean two-way divergence, no rebase required to reunite them. Once the
live campaign no longer needs its current checkout (`trade_robustness_modular`'s main
worktree, currently occupied by both the checked-out branch and an unrelated orphaned
diagnostic process), the exact commands to adopt this integration as `melitz/fullD-delta-star`'s
own new tip are:

```
git worktree add /path/to/a/fresh/worktree melitz/fullD-delta-star
cd /path/to/a/fresh/worktree
git merge --no-ff integration/melitz-production-consolidation-2026-07-31 \
  -m "Merge legacy-H-removal + perf-audit fixes, repaired test harness, production launcher, and checkpoint-version enforcement (production-consolidation-2026-07-31)"
```

This does not touch `melitz/fullD-delta-star`'s existing worktree/checkout and can be done in
a disposable worktree first to confirm no surprises, then fast-forwarded/pushed once reviewed.

---

# Final questions (answered explicitly)

**1. Is there one Git repository or more than one?**
More than one. `/bbkinghome/edav/cdw` and `/bbkinghome/edav/gravity_robustness/trade_robustness_modular`
were two independent local clones of the same GitHub remote (`edwardwiles/cdw`), each with
their own large (~50+) worktree trees, badly out of sync (`cdw`'s local `melitz/fullD-delta-star`
ref was 50+ commits stale). Per an explicit user decision this session, `cdw` (specifically a
new `cdw/melitz/` subdirectory) is now established as the sole go-forward home for Melitz
work; `trade_robustness_modular` remains in use only because its main worktree still holds the
live checkout of `melitz/fullD-delta-star` (git disallows checking out the same branch twice)
plus an orphaned 5+ day old diagnostic process, both left untouched.

**2. Which worktree and branch should be canonical production?**
`/bbkinghome/edav/cdw/melitz/integration-production-consolidation-2026-07-31`, branch
`integration/melitz-production-consolidation-2026-07-31`, tip `00c1435`.

**3. Which current campaigns used which commits?**
The D20 delta=0.5 overnight campaign (`melitz/fullD-delta-star`) used through `d0904c7`
originally; the branch has since advanced to `6e803d4` (overnight continuation), then
`20e9f2c`/`64a0e2a` (a self-audit and its correction, done by a separate concurrent session,
not part of this integration's base). The audit branch's work (`358d3d1`/`68aa6be`/`6b65b1d`)
was a standalone audit exercise, never checked out by any live campaign. A completely separate,
currently-live `sigma3_W500k` five-family campaign runs from an unrelated worktree/branch
(`campaign/resource-layout-test-sigma3-W500k-2026-07-30`, tip `2e02c18`).

**4. Did any campaign use `6b65b1d`?**
No. It only ever existed as the audit branch's tip; no worktree had it checked out as a live
campaign; this session's background agent cherry-picked (not merged) its source changes onto
the integration branch.

**5. Which audit/performance commits should be integrated?**
Both `358d3d1` (source fix + structural gates) and `6b65b1d` (perf: empty-sentinel + `mul_G!`
fusion) — both cherry-picked cleanly with zero textual conflicts, both correctness-verified.
`68aa6be` (docs/logs only) was correctly NOT merged (no source dependency).

**6. Are the empty-sentinel and `mul_G!` changes safe under complete solver replay?**
Yes. Phase 6 gates show bit-identical or ULP-level-identical results at D=4 and real D=20
(objective, gradient, Hessian, complete inner solve, middle solve, and both cold-verified
production incumbents), and a dedicated mutation-regression test (200×2 real call-pattern
exercises, checked object identity via `objectid`) confirms the shared sentinels are never
mutated, resized, or reassigned by any real call path.

**7. Why did the old test suite fail?**
Two independent bugs, both pre-existing on the canonical base itself, unrelated to any ported
commit: (a) `runtests.jl` hand-duplicates `include_melitz.jl`'s include list and was missing
`reduced_q_switch_geometry.jl`, causing an `UndefVarError` cascade in one testset under any
full run; (b) a segfault in a pre-existing test that called a `MelitzCCBundle` functor against
an un-equilibrated operator (see "Known open items" #1). Separately, the originally-reported
blocker — `Threads.@threads` allocation tests asserting exact-zero bytes — is explained by
Julia 1.12.6's own scheduler-object allocation (`Task`/`SpinLock`/`Condition`, confirmed via
`Profile.Allocs`), which scales with `nthreads`, not problem size.

**8. Does the repaired complete test matrix pass?**
Yes: **53/53 groups pass, 0 fail, 0 error**, confirmed by an independent full rerun after the
include fix landed (the matrix originally committed in this branch's history was stale by one
minute and has been superseded — see "Repository migration").

**9. Are there duplicate or shadowed Melitz source definitions?**
No unsafe shadowing found. `include_melitz.jl` is the single 65-file aggregator;
`runtests.jl`'s hand-duplicated copy of that list is a confirmed structural fragility (caused
at least two silent test-coverage gaps now, `lfd_preserving_state.jl` and
`reduced_q_switch_geometry.jl`) but is not itself a shadowing bug. Every duplicate top-level
name found (`melitz_recover_lfd` etc.) is legitimate Julia multiple dispatch on different
concrete argument types, verified via the real runtime method table.

**10. Can a campaign accidentally launch from the wrong worktree today?**
Yes, easily, for any script not yet updated to call the new launcher — with 2 independent
repos and roughly 140 total worktrees between them (many stale, some detached HEAD, ahead/behind
states scattered), nothing previously stopped a script from running against a stale or wrong
checkout.

**11. Does the new launcher prevent that?**
Yes, for any campaign script that adopts it: `melitz_production_preflight!` refuses to proceed
unless the tracked tree is clean (or an explicit override is passed) AND the current commit
provably descends from the approved base (`melitz_production_approved_base.txt`), verified
live via real git plumbing — including surviving this session's own independent portability
fix after finding the original test hardcoded a foreign commit SHA that broke on relocation.

**12. Does checkpoint resume enforce code-version consistency?**
Yes, for any campaign adopting `melitz_save_checkpoint!`/`melitz_resume_checkpoint`:
same-commit resume is silent; different-commit resume without `migrate=true` refuses;
migration requires a `cold_reverify` callback whose failure/exception/non-`Bool` return all
refuse rather than silently trusting a stale checkpoint. 6 regression-test groups pass live
against a real second commit in a disposable scratch worktree.

**13. What exact commit should future Melitz campaigns use?**
**`00c1435f1af6f11147d4f3ce0f4fc07498678738`**, branch
`integration/melitz-production-consolidation-2026-07-31`, worktree
`/bbkinghome/edav/cdw/melitz/integration-production-consolidation-2026-07-31` — pending the
documented (not yet executed) merge into `melitz/fullD-delta-star` to become that branch's own
tip.
