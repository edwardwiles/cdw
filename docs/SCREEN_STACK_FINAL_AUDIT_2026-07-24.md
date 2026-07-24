# Screen Stack + Threshold-10 Final Audit — 2026-07-24 release

Branch: `release/fullA-screens-threshold10-now-2026-07-24`. Base: `production/fullA-exact@fd21f9f2`
(== `cdw/production/fullA-exact` at fetch time). This release **selectively ports** (does not
merge wholesale) the tested Part B+C implementation from
`release/fullA-omit-row-restore-screens-2026-07-23@acae5c7` — cherry-picked cleanly onto this
fresh branch (identical parent, zero conflicts) — then closes gaps this session's own audit found
that the 2026-07-23 session's "READY+tested" claim did not cover. The 2026-07-23 omit-ROW
worktree/branch itself was left untouched throughout (a separate Claude session is actively
working there; this release never read past its own `git show <commit>` of already-committed
history, never touched its working tree).

## Provenance

- Full production base commit: `fd21f9f2275478d79787cceb3a1fa3e22e8cd0f4` (== `production/fullA-exact` == `cdw/production/fullA-exact`)
- Julia: `1.12.6`
- KNITRO: `13.0.1` (`$KNITRODIR=/opt/shared_sw/knitro/13.0.1`, confirmed via `LD_LIBRARY_PATH`/module env, matching the task's stated installed version — NOT the also-installed 14.2.0 seen elsewhere on this host)
- Checkpoint schemas: unrestricted `CHECKPOINT_SCHEMA=3` (`c10_d20_production_driver.jl`); CM/CM+mean-ZC `CM_CHECKPOINT_SCHEMA=4` (`cm_checkpoint.jl`); origin-ZC `CMCheckpointV5` (`cm_originzc_checkpoint.jl`)
- Layout: `D20_REAL=20`, D-by-D square (every origin is also a destination) — pre-omit-ROW, as expected; Part A (omit-ROW) is separate, in-progress work this release does not depend on or block on

## Part B: screen reachability, before and after

Re-traced independently (not merely re-asserted from the 2026-07-23 audit) via the same
grep-for-actual-call-site method as `docs/SCREEN_STACK_AUDIT_2026-07-23.md`:

| Family | Screen status (2026-07-23 session) | Screen status (this release, after fixes) |
|---|---|---|
| Unrestricted | ACTIVE (`screened_eval`, sole choke point) | ACTIVE, unchanged; startup banner + threshold-config print added |
| Flexible CM | Restored via `cm_screen_bridge.jl`, wired at `cm_checkpoint.jl`'s 6 call sites | Restored; **one additional bypass found and fixed** — `cm_production_stage_runner.jl`'s own preflight check (line ~198) still called the raw unscreened `cm_production_value_verified` even after the 2026-07-23 restoration (only the mean-ZC branch of that same preflight block had been switched to the screened wrapper) |
| CM + mean/ZC | Restored, wired | Restored, wired; now also carries live screen counters |
| Origin-specific ZC | Restored, wired | Restored, wired; now also carries live screen counters |

**New finding, this release**: the 2026-07-23 session's `CMScreenCounters` struct was defined but
**no production caller ever attached one** — every `..._screened` call site ran with
`counters=nothing`, so `screen_calls`/`pairwise_hits`/`winner_hits`/`inner_solves_avoided` were
never actually recorded anywhere a real campaign would see them (Part B step 7's own text
anticipated this exact gap: "The prior implementation defined counters but did not prove that
production callers retained them. Close that gap."). Fixed: `with_screen_counters(pcx)`
(`cm_screen_bridge.jl`) attaches a live `CMScreenCounters` to `pcx` right after each
`build_*_production_context` call in `run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`
and the two stage-runner preflight blocks; every `..._screened` call in those functions now passes
`counters = pcx.screen_counters`. `print_screen_summary(pcx)` prints the accumulated counters
(`calls`, `pairwise_hits`, `hard_winner_hits`, `witness_hits`, `points_passed`,
`inner_solves_avoided`, `screen_wall_s`) once at the end of each `run_*_upper_checkpointed` call,
and the same NamedTuple is now also returned as `screen_summary` in that function's result.

## Part B step 3: active-origin/active-destination accessors

New file `cc_algo/active_layout.jl`: `active_origins(ctx)`, `active_destinations(ctx)`,
`active_od_cells(ctx)`. Falls through to `1:ctx.D` for both when `ctx` carries no explicit
`active_origins`/`active_destinations` field (today's default, pre-omit-ROW) — zero behavior
change. `cm_screen_bridge.jl`'s witness-certificate loop (previously
`for d in 1:ctx_cm.D, o in 1:ctx_cm.D`) now iterates `active_od_cells(ctx_cm)` instead. This does
**not** implement omit-ROW — `pairwise_certificate`/`screen_hard_winners` still operate on the
full D×D matrices internally, by design (that generalization is Part A's job) — it only makes the
one explicit screen-side loop already in this release's own code rectangular-ready. Verified with
a synthetic non-square-mask unit test (`test_active_layout_accessors.jl`, 11/11 pass, construction-only, no KNITRO): a `(D=4, active_destinations=[1,2,4])` synthetic context correctly yields 12 (not 16) cells and never contains the omitted destination.

## Part B: screen defaults

Explicit per the task's requirement:

- Pairwise certificate: **enabled** (production default in every restricted-family entry point)
- Hard-winner certificate: **enabled** (same)
- Witness certificate: **opt-in**, `use_witness=false` default — carried over unchanged from the
  2026-07-23 session, which never ran the bounded cost/benefit benchmark the task's own step 4
  requires before flipping it on by default. Not evaluated further this session (out of the
  bounded scope of this release); left as an explicit `use_witness` kwarg on every screened
  entry point so a future session can flip the default once that benchmark exists.
- Envelope/winning-range/safety-net screens: unrestricted-only (`c10_d20_production_driver.jl`'s
  `screened_eval`), unchanged — their validity for the CM/mean-ZC/origin-ZC restricted moment
  sets has not been established, so this release does not extend them there (matches the
  2026-07-23 audit's own scoping).

## Part B: exact-rejection semantics and startup/observability printing

`cm_screen_precheck!` raises the existing typed `CMExpectedSolveFailure` with a screen-name
(`pairwise`/`witness`/`zero-winner`) plus origin/destination in its message — unchanged from
2026-07-23, re-verified this session (`test_cm_screen_restoration.jl` Group 1/2/3, real
D=20/W=80,000, 10/10 pass). A heuristic/unresolved check never exact-rejects; the screens raise
only from the same draw-free exact certificates `infeasibility_screen.jl` already documents as
mathematically exact (unchanged from 2026-07-23's own audit of that file).

Startup banner (new this release) at the top of every `run_*_upper_checkpointed` and the two
`run_profile_checkpointed`/`run_polish_checkpointed` (unrestricted) entry points:
```
[screen-stack] mode=<mode> enabled=true
[screen-stack] ordered active screens: <list>
[threshold-config] mode=<mode> requested_delta=<d> resolved_active_threshold=<t> stored_in_objective_bundle=<t>
```

## Part C: threshold-10 mechanism

Core mechanism (`cc_algo/threshold_early_abort.jl`, `ThresholdAbortState`,
`maybe_abort_on_threshold!`, `CertifiedDivergenceLowerBound`, `resolve_threshold_for_delta`,
`threshold_permits_reject`) carried over unchanged from the 2026-07-23 session — re-verified this
session with a fresh 20/20-assertion real-KNITRO run (`test_threshold10_early_abort.jl`), including
the boundary sign tests (9.999/10.000/10.001 on both the `lower_bound` and low-level `f`
framings), the `lower_limit=-50` coexistence (unchanged, separate field, confirmed still present
and untouched in every objective-bundle constructor site touched this session), and the
cross-delta cache-reuse rule.

**New finding, this release**: every restricted-family objective-bundle **rebuild** site —
`build_cm_augmented_obj` (`common_marginals_moments.jl`), the archB rebuild inside
`build_cm_production_context` (`cm_production_bundle.jl`), `build_cm_augmented_obj_archB`
(`cm_hessian_architectures.jl`), `build_cm_augmented_obj_interval`
(`common_marginals_interval.jl`), `build_cm_augmented_obj_from_CM`
(`c12b_interval_common_marginals_moments.jl`), `build_cm_meanzc_augmented_obj`
(`cm_meanzc_moments.jl`), and `build_originzc_augmented_obj` (`cm_originzc_moments.jl`) — construct
a **fresh** `PsiObjectiveBundleImplicit` from the base `ctx.obj`'s fields, and **none of the seven
forwarded `threshold_state`**. Since `PsiObjectiveBundleImplicit`'s `threshold_state` field
defaults to `ThresholdAbortState()` (`threshold=Inf`, disabled) when not explicitly given, this
silently **disabled Part C's threshold-10 early-abort for every CM / CM+mean-ZC / origin-ZC
production entry point**, even on a base `ctx.obj` that `d20_real_setup` had correctly configured
with a finite threshold — exactly the failure mode the task's own step 10 warned about ("The
threshold must not default to Inf in replacement restricted bundles"). This was NOT caught by the
2026-07-23 session's own Part C tests because those built their context via `context.jl`'s base
`ctx` directly, never through any of these seven CM/ZC rebuild functions.

**Fixed**: all seven sites now pass `threshold_state = obj0.threshold_state` (or
`obj_cm.threshold_state` for the cm_production_bundle.jl archB rebuild, which rebuilds from the
already-CM `obj_cm` rather than the base `obj0`). Verified this session with a new
construction-only regression test (`test_threshold_propagation_regression.jl`, real D=20/W=80,000
context, no KNITRO solve needed since this is a pure field-propagation check): confirms
`pcx.ctx_cm.obj.threshold_state.threshold == 10.0` (not `Inf`) at `delta=1` for flexible CM,
CM+mean-ZC, and origin-ZC alike — ALL PASS.

## Part C: threshold activation rule, current campaigns

`resolve_threshold_for_delta(requested_delta; base_threshold=10.0, safety_margin=1.0)` returns
`base_threshold` when `requested_delta < base_threshold - safety_margin`, else `Inf`. For the
current campaign deltas (0.1, 0.5, 1, 2) this resolves to exactly `10.0` in every family,
confirmed both by the sign-test suite and the new propagation-regression test. `delta=9` disables
(within the 1.0 safety margin of 10); `delta>=10` disables.

## Real D=20/W=80,000 evidence this session (fresh reruns on this branch, not re-cited from 2026-07-23)

| Test | Result |
|---|---|
| `test_active_layout_accessors.jl` (construction-only) | 11/11 PASS |
| `test_threshold10_early_abort.jl` (real KNITRO) | 20/20 PASS |
| `test_cm_screen_restoration.jl` (real D=20/W=80,000) | 10/10 PASS |
| `test_threshold_propagation_regression.jl` (real D=20/W=80,000, construction-only) | ALL PASS (5/5) |
| D=4 no-regression batch: `test_infeasibility_screen.jl` | 72/72 PASS |
| D=4 no-regression batch: `test_cm_meanzc_d4_gates.jl` | 31/31 PASS |
| D=4 no-regression batch: `test_cm_meanzc_d4_gates.jl` | 31/31 PASS |
| D=4 no-regression batch: `test_cm_originzc_pure_moments.jl` | PASS |
| D=4 no-regression batch: `test_cm_verified_success.jl` | pre-existing failure, NOT a regression — see below |

## Part D: four-mode production supervisor smokes (real D=20/W=80,000/L=50/delta=1/C+ backend)

All four launched via the ACTUAL production entry points (not synthetic reimplementations):

| Mode | Entry point | Result |
|---|---|---|
| Unrestricted | `c10_prod_driver_smoke_original.jl` (`run_profile_checkpointed`) | PASS: startup banner+`resolved_active_threshold=10.0` printed; feasible point found (`Delta_dual=0.2145`, `inner_status=0`); checkpoint written, `reason=stage_complete`, cold-verified fields match; clean process exit, no orphan |
| Flexible CM | `cm_production_stage_runner.jl` (`run_cm_upper_checkpointed`) | PASS: startup banner (`mode=cm_flexible`, `resolved_active_threshold=10.0`) printed; feasible incumbent found (`Delta=0.771`, `kappa=0.0431`); **live screen-summary counters printed and non-null** (`calls=5 points_passed=5 inner_solves_avoided=0`), proving the counters-threading fix works end-to-end in a real production run, not just in a unit test; `STAGE_DONE` sentinel; checkpoint valid |
| CM + mean/ZC | `scripts/d20_meanzc_supervisor_smoke_test.sh` (real supervisor: setsid/pgid launch, actual production process-group mechanics) | PASS, full 8-step protocol: (1) real stage launch; (2) schema-4 checkpoint with typed `best_feasible` incumbent confirmed; (3) deliberate SIGTERM of the process group; (4) confirmed **zero** orphaned/stray processes after kill; (5) resume through the identical supervisor mechanics; (6-7) cold-verify in a fresh process, cache disabled — reported vs. cold `Delta_dual` agree to `1.67e-16` (machine precision); (8) resume correctly **refused** under 4/4 mismatched configs (K_mean, K_pair, cm_extension, cm_gradient_backend without `allow_backend_switch`). Startup banner + screen-summary counters confirmed present in both the original and resumed stage logs. |
| Origin-specific ZC | `originzc_production_stage_runner.jl` (`run_originzc_upper_checkpointed`), `DISTRIBUTION_RESTRICTION=origin_specific_moments ORIGIN_K_MEAN=1` | PASS: startup banner (`mode=origin_zc`, `resolved_active_threshold=10.0`) printed; screened preflight cold-verify passed (`Delta_dual=0.00269`); real KNITRO outer solve ran to its 90s time budget with monotonically decreasing optimality error and a feasible tracked incumbent (`gp=0.9702`, `Delta=0.847`, `kappa=0.0491`); `[screen-summary] calls=10 points_passed=10 inner_solves_avoided=0` printed with real counts; `STAGE_DONE` sentinel; checkpoint written and valid |

Two launch-config bugs found and fixed while setting these up (both **pre-existing**, not introduced by Part B/C, both in non-production or ancillary code, neither blocking the actual production entry points once given correct config):
1. `d20_originzc_shakedown.jl` (a diagnostic script, not the production entry point) never included `cm_screen_bridge.jl` — see "Second pre-existing gap" below.
2. First origin-ZC launch attempt via the correct production entry point failed with `KeyError: DISTRIBUTION_RESTRICTION` — a required, no-default `ENV` var the stage runner's own header already documents; this is expected/correct behavior (fail-fast on missing required config), not a bug — just an operator-config miss on this session's first attempt, corrected on retry.

### Second pre-existing gap found: stale include lists in non-production diagnostic scripts

While attempting a real D=20 origin-ZC smoke via `d20_originzc_shakedown.jl`, hit
`UndefVarError: with_screen_counters not defined` — that script calls
`run_originzc_upper_checkpointed` but never included `cm_screen_bridge.jl`. Confirmed this is
**not** something this session introduced: `run_originzc_upper_checkpointed` has called the
`_screened` wrapper functions (which require `cm_screen_bridge.jl`) since the 2026-07-23 session's
own `acae5c7` commit, and this script's include list was never updated then either — my new
`with_screen_counters`/banner calls just happen to be the first missing symbol hit at runtime. The
**production entry point** (`originzc_production_stage_runner.jl`) already includes
`cm_screen_bridge.jl` correctly and is unaffected. Fixed the one file needed for this release's own
smoke test (`d20_originzc_shakedown.jl`, one-line include add). A repo-wide grep found four more
non-production diagnostic scripts with the identical gap
(`c33_phase4_cm_shakedown_control.jl`, `c33_phase4_cm_shakedown_interrupt.jl`,
`c33_phase4_cm_shakedown_resume.jl`, `cm_cplus_matched_trajectory.jl`) — left unfixed as
out-of-scope for this release (none are production entry points; fixing all stale includes
repo-wide is a separate cleanup task, matching how `test_cm_verified_success.jl`'s analogous gap
was handled below).

All four production supervisor smokes pass. See individual entries above for per-mode detail.

### Pre-existing failure identified and ruled out as a regression

`test_cm_verified_success.jl` (D=4) throws `UndefVarError: meanzc_resolve_K not defined` from
inside `run_cm_upper_checkpointed` (`cm_checkpoint.jl:391`). Root cause: this test file's own
`include` list was last updated at `c6a94f9` (before `cm_meanzc_config.jl`, which defines
`meanzc_resolve_K`, was introduced at `d2a0f76`, "Port CM+moments(+ZC) mathematical core onto
CM-C+ production tip") and was never updated to include it — `cm_checkpoint.jl` itself has
unconditionally called `meanzc_resolve_K` since that commit for every caller, including the
non-mean-ZC path. **Confirmed independently on a clean, untouched
`production/fullA-exact@fd21f9f2` checkout** (`gravity-production-fullA-exact` worktree, `git
status` clean of any release-branch changes): the identical error reproduces there. This bug
predates this release, predates the 2026-07-23 screens/threshold session, and is unrelated to
screens or thresholds — out of this release's bounded scope to fix (would require auditing every
other D=4 test file's include list for the same staleness, a separate task). Not a regression
introduced by Part B or Part C.

## Final decision

**READY_AND_MERGED.**

- Exact screens reachable in all restricted modes: YES (Group 1/2/3 of `test_cm_screen_restoration.jl`, real D=20/W=80,000, 10/10; plus the second bypass this session found and fixed in the flexible-CM stage-runner preflight)
- Real screen hits prove zero inner solves: YES for the unrestricted family (`test_infeasibility_screen.jl` §9, real screen rejection with zero new KNITRO solves, 72/72 overall) and for the CM family's pathological-point case (`test_cm_screen_restoration.jl` Group 3); the four production smokes this session ran all landed on feasible calibration points by construction (0 screen hits recorded, `screen_wall_s` nonzero and counters live either way) — a genuine screen-triggering *production* point was not separately hunted for at D=20 for CM+mean-ZC/origin-ZC beyond the D=4 battery, matching the task's explicit minimal-scope instruction ("No broad battery is required" / reuse D=4 for this)
- Threshold 10 active in all actual objective bundles: YES, confirmed for all three restricted families via `test_threshold_propagation_regression.jl` (this session's own new construction-level test, closing a real gap the 2026-07-23 session's tests did not cover) plus the live `[threshold-config] ... resolved_active_threshold=10.0` banner in all four real supervisor smokes
- Typed certificates reach the outer layer and cache: YES, unchanged from 2026-07-23 (`CertifiedDivergenceLowerBound`, `threshold_abort_result`, `threshold_permits_reject`), re-verified this session (20/20 real-KNITRO sign/behavior assertions)
- Real D=20 threshold tests pass: YES (`test_threshold10_early_abort.jl` Section 2, real KNITRO inner solve, clean `KN_RC_USER_TERMINATION`)
- All four supervisor smokes pass: YES (table above)
- No regressions or process leaks: YES — D=4 battery all green except one confirmed pre-existing, unrelated failure; CM+mean-ZC supervisor smoke explicitly confirmed zero orphaned processes after a deliberate SIGTERM/SIGKILL cycle

Two real, previously-undiscovered bugs were found and fixed as part of closing this release's gates (not present in the 2026-07-23 session's own "READY+tested" claim, which this session's audit found insufficient on inspection): (1) `threshold_state` silently dropped to `Inf` on all seven restricted-family objective-bundle rebuilds, disabling Part C everywhere except the unrestricted path; (2) the flexible-CM stage-runner's own preflight check bypassed screens even after the 2026-07-23 restoration. Both fixed and verified above.
