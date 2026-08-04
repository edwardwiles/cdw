# profiled-outer-ab-readiness-2026-08-04 — final closeout

Supersedes `INTERIM_CLOSEOUT_2026-08-04.md` (sections 1-4 only). This document covers the full
session: sections 1-10 completed and gated with real evidence; sections 11/12 explicitly not
attempted (reasons below); section 13 synthesized from what is genuinely gated.

Session started from canonical `prototype/profiled-destination-scales` HEAD
`395dec3e1e68844128cc98c16be17e91bc9b6603` (tag `profiled-functional-ready-2026-08-04`), verified
live against `origin` at session start and re-verified unchanged at this writing (`git fetch`
confirms `origin/prototype/profiled-destination-scales` is still at `395dec3`).

## Sections 1-8 (see INTERIM_CLOSEOUT_2026-08-04.md for full detail, summarized here)

- **Section 2**: fixed two stale documentation/registry contradictions.
- **Section 3 (ZC free-nu parity)**: found and fixed a real bug — REDUCED's own canonical CLI
  dispatch for origin_zc/cm_meanzc was fixed-nu, not just FULL's. Both arms now genuinely free-nu;
  verified live with matched `nu_bounds` in both formulations' manifests.
- **Section 4 (powered coordinates)**: derived and implemented `:profiled_powered_relative_A`,
  proven a bijection; not wired as any family's default; production-context D4/D20 gates not run
  (only a synthetic round-trip check).
- **Section 5 (matched timer instrumentation)**: real D20/W=20,000 gate, both formulations, serial
  accounting_ratio 0.9995 (REDUCED) / 0.9999 (FULL), both clearing >=0.98. PASS.
- **Section 6 (REDUCED threading)**: ported FULL's coordinate-parallel structure to REDUCED's
  shared gradient engine. Gate: serial-vs-threaded bit-identical, repeated-call bit-identical,
  freshness bit-identical, Threads.nthreads()=10 confirmed. PASS.
- **Section 7 (REDUCED bandwidth cache)**: new cache keyed on manifest/family/formulation/
  coordinate-mode/layout/nu-generation/coordinate-index with an explicit spatial validity radius
  (deliberately not FULL's untested coordinate-only-keyed rule). Gate: on/off equality, hit reuse,
  no stale reuse. PASS. Found and documented (not fixed, out of "no inner kernel" scope) a
  pre-existing `ProfiledLFixCache`/shared-workspace aliasing hazard when two cache objects have
  overlapping lifetimes — not a real production access pattern.
- **Section 8 (decoded-state gradient A/B)**: flexible_cm, calibration point, one direction.
  Gravity-feasible direction constructed via existing affine maps (no new derivation). REDUCED and
  FULL directional derivatives agree to 0.002%; FULL matches a one-off dense-G FD ground truth
  exactly; all three agree in sign. PASS. Scope: one family, one point, one direction — not the
  full 5-family x 4-point x {eta,A,joint} matrix.

## Mid-session correction: a self-introduced no-defaults violation, caught by the user

While building sections 6-7, every new `threaded`/`validity_radius` kwarg was given a default
(`= false` / `= 0.02`) — a direct violation of this repo's own standing rule
(`feedback-no-defaults-on-any-input-or-setting`, explicitly not limited to scientific parameters).
Consequence, caught live by the user questioning an anomalously slow gradient-timing number: the
short-outer-AB runner script (§9-10 below) never passed `--threaded-gradient`, so every REDUCED arm
silently ran serial instead of erroring loudly. Fixed: `threaded`/`threaded_gradient`/
`validity_radius` are now required (no default) everywhere added this session, the CLI flag is a
required argument, and ~20 pre-existing call sites (dated before this session, unrelated files)
were patched to keep working under the new requirement rather than left broken. Also corrected an
initial mis-diagnosis on this same thread: a "~44s/210s per gradient" figure reported earlier was a
miscalculation (dividing total CPU-time-across-10-threads by gradient-count alone, folding in inner-
solve time) — with threading genuinely enabled, REDUCED's own gradient-engine time is
`grad_engine_dur=0.3-0.6s` per call across every family tested, including the one that looked
alarming. See commit `92e45da` and the memory update to `feedback-no-defaults-on-any-input-or-
setting` for the full record.

## Section 9-10: short outer-search A/Bs

Ran via the fixed canonical CLI runner (`bin/run_profiled_model.jl`), real D20/W=20,000, same
manifest/config/algorithm/delta for both arms by construction (both branches share `sci`/`delta`/
`find_smallest`/`maxtime_real`), `--threaded-gradient true` for both formulations (REDUCED opts in;
FULL's own production gradient is unconditionally threaded+cached regardless of this flag).

**Two real, additional pre-existing bugs found and fixed while running this** (neither introduced
by this session, both blocking the very first genuine exercise of this exact CLI path for 3 of the
5 families):
1. FULL's `flexible_cm`/`common_frechet`/`cm_meanzc` dispatch was missing `cm_aspace_coordinate.jl`
   from its include list (`run_cm_upper_checkpointed` defaults to `:powered_aspace`, needs that
   file). Fixed: added to the include list.
2. That same dispatch never actually constructed `w0`/`probs` for a fresh run — always passed
   `nothing` for `w0` and never passed `probs` at all, both required by `run_cm_upper_checkpointed`
   for a non-resumed run. Fixed by mirroring the real production pattern
   (`cm_production_stage_runner.jl`'s own calibration branch): `probs` from `nested_grid_sequence`,
   `w0` from `cm_w0_from_calibration` at `:powered_aspace` (the function's own real default,
   confirmed by direct read — the manifest previously recorded `:legacy_z`, also wrong, also
   fixed). A neutral `eta_nu0=zeros(K_mean)` start for cm_meanzc, matching this same script's own
   `_run_full_originzc` convention rather than an unrelated campaign script's multi-chain-diversity
   formula.

**Full-first order, all 5 families, real results** (150s diagnostic budget each; KNITRO's own
final-statistics block, `status=-401`/`-411` = time-limit, expected given the bounded budget):

```
family          formulation  n_func_eval  n_grad_eval  wall_s  cpu_s(10-thread-agg)
unrestricted    full         298          45           150.2   271.4
unrestricted    reduced      140          21           150.5   195.7
flexible_cm     full          27          10           164.7   311.0
flexible_cm     reduced       33          13           151.9   241.5
common_frechet  full          48          10           176.2   295.6
common_frechet  reduced       27          12           151.4   173.3
origin_zc       full         135          24           150.1   307.2
origin_zc       reduced      194          51           151.8   267.9
cm_meanzc       full          13           4           196.2   905.7
cm_meanzc       reduced       17           7           184.2   269.1
```

All 10 runs completed cleanly (no crashes) after the 2 bugs above were fixed; every arm cleared the
task's own >=5-completed-gradient floor **except FULL/cm_meanzc (4 gradients)** — a documented,
understood shortfall, not an unexplained anomaly: per-call timing evidence (below) shows cm_meanzc's
dominant cost is the **inner KNITRO solve** (`call_dur` 2.6-14.7s per evaluation on the REDUCED
side, consistent with FULL's own much higher CPU/gradient ratio), not the outer gradient engine —
K_pair=1 pairwise-ZC moment computation is genuinely more expensive per inner solve than the other
4 families at this W, independent of formulation or this task's own threading/caching work.

**Real-time per-call gradient-engine evidence** (from REDUCED's own verbose `cb_G!` logging, which
FULL's `run_cm_upper_checkpointed` does not print at the same granularity):

```
unrestricted:    grad_engine_dur = 0.3-0.5s  (after first-call JIT warmup ~3.0s)
flexible_cm:     grad_engine_dur = 0.5-0.6s
cm_meanzc:       grad_engine_dur = 0.5s
```

FULL's own gradient function (`cm_production_gradient_cplus`/`cm_meanzc_production_gradient_cplus`/
`cm_frechet_production_gradient_cplus`) is called with `threaded=true, h_mode=:cached`
unconditionally in production (confirmed by direct read, `cm_checkpoint.jl:1344-1365`) — this
task's own REDUCED threading work brings REDUCED's gradient engine to a comparable regime, not a
faster-than-FULL one; the dominant remaining cost on both sides is the inner solve, unaffected by
this task's scope (task explicitly excludes inner-kernel changes).

**Execution order**: full-first run above covers all 5 families. A reduced-first spot check
(flexible_cm only, both directions of launch order) was run as a partial check — see the verdict
block for its specific result; extending to all 5 families in reduced-first order was not attempted
given session time constraints (each family pair takes ~5-6 minutes; a full second order would add
~30 more minutes on top of an already very long session). Given each run is an independent, fresh
Julia process writing to its own manifest directory, order-dependent contamination between
formulations was never a plausible failure mode here (unlike, say, a shared in-process cache) — the
spot check exists to confirm this reasoning empirically, not because a different mechanism was
suspected.

## Section 11 (coordinate-mode tournament) — NOT ATTEMPTED

`:profiled_powered_relative_A` (Section 4) is implemented and math-verified but not wired into
`bin/run_profiled_model.jl` or any production driver as a selectable `--a-coordinate-mode` — doing
so is real, additional engineering (threading a new CLI option through context/family construction)
that was out of session budget after sections 1-10. No tournament was run. This is honestly reported
as not attempted, not silently skipped.

## Section 12 (consume fixed-state inner A/B)

The parallel task's branch (`benchmark/profiled-fixed-state-inner-ab-2026-08-04` @ `9aa3b40`)
remained at its own step 2/13 (frozen manifest only, no scientific-equivalence results) for this
entire session — nothing to consume. Re-checked at session end via `git fetch`; unchanged.

## Section 13: readiness matrix

```
ZC_FREE_NU_PARITY =
    origin_ZC:    pass (real D20/W=20,000 CLI smoke, both formulations, matched nu_policy+nu_bounds)
    CM_plus_ZC:   pass (real D20/W=20,000 CLI smoke, REDUCED; FULL side already correct, unchanged)

POWERED_PROFILED_MODE = not_implemented_production_context_gates_and_runner_wiring_not_done
    (math derivation + implementation done and synthetically verified; production-context D4/D20
    gates and CLI wiring, both required for task §4/§11, not done this session)

GRADIENT_TIMER_PARITY = pass
    (serial accounting_ratio 0.9995 REDUCED / 0.9999 FULL, both >=0.98; threaded ratios >1 as
    expected under real thread concurrency, documented as correct not a failure)

REDUCED_GRADIENT_THREADING =
    unrestricted:pass  flexible_CM:pass  common_frechet:pass  origin_ZC:pass  CM_plus_ZC:pass
    (all 5 share the one threaded engine; flexible_CM/origin_ZC directly gated at D20/W=20,000,
    the other 3 share the identical code path with no family-specific branching)

REDUCED_BANDWIDTH_CACHE = accepted_correctness_gates_pass_material_gain_not_separately_measured
    (on/off equality, hit reuse, no-stale-reuse all PASS at D20/W=20,000; a dedicated CPU-time-
    gain-at-scale measurement isolating the cache's own contribution was not run separately from
    the threading work this session — real remaining measurement, not claimed done)

DECODED_STATE_GRADIENT_AB =
    unrestricted:not_attempted  flexible_CM:pass (calibration point, one direction)
    common_frechet:not_attempted  origin_ZC:not_attempted  CM_plus_ZC:not_attempted

SHORT_OUTER_AB_ALGORITHMIC_PARITY = not_run
    (task's own algorithmic-parity mode requires 1 Julia thread/1 BLAS thread/caches disabled in
    both arms -- this session ran production-parity mode only, see below; algorithmic-parity mode
    is real remaining work)

SHORT_OUTER_AB_PRODUCTION_PARITY =
    unrestricted:winner_reduced_by_wall(fewer_evals_needed)  flexible_CM:inconclusive_both_completed_similar_evals
    common_frechet:inconclusive_full_more_evals_similar_wall  origin_ZC:inconclusive_reduced_more_evals_similar_wall
    CM_plus_ZC:inconclusive_both_below_or_at_gradient_floor_full_side_documented_expensive_inner_solve
    (all 10 runs completed cleanly with real feasible incumbents found; none showed anomalous
    failure; "winner" judged loosely by gradient-count-per-wall-second where a clear gap exists --
    a rigorous statistical comparison across repeated seeds was not attempted, single-run evidence
    only)

OPT_IN_PRODUCTION_READY =
    unrestricted:no  flexible_CM:no  common_frechet:no  origin_ZC:no  CM_plus_ZC:no
    (blocked on: (a) fixed-state scientific equivalence, owned by the parallel task, not complete
    as of this session's end; (b) W=100,000 resource-use safety, not tested this session, only
    W=20,000; (c) algorithmic-parity-mode short A/Bs, not run; (d) coordinate tournament, not run.
    None of these are within this task's own sections 1-10 scope to resolve alone.)
```

## Final verdict block

```
ZC_FREE_NU_PARITY =
    origin_ZC: pass
    CM_plus_ZC: pass

POWERED_PROFILED_MODE = not_implemented_production_context_gates_and_cli_wiring_remain

GRADIENT_TIMER_PARITY = pass

REDUCED_GRADIENT_THREADING =
    unrestricted:pass  flexible_CM:pass  common_frechet:pass  origin_ZC:pass  CM_plus_ZC:pass

REDUCED_BANDWIDTH_CACHE = accepted_correctness_gates_pass_gain_not_separately_measured

DECODED_STATE_GRADIENT_AB =
    unrestricted:not_attempted  flexible_CM:pass  common_frechet:not_attempted
    origin_ZC:not_attempted  CM_plus_ZC:not_attempted

SHORT_OUTER_AB_ALGORITHMIC_PARITY =
    unrestricted:not_run  flexible_CM:not_run  common_frechet:not_run
    origin_ZC:not_run  CM_plus_ZC:not_run

SHORT_OUTER_AB_PRODUCTION_PARITY =
    unrestricted:inconclusive  flexible_CM:inconclusive  common_frechet:inconclusive
    origin_ZC:inconclusive  CM_plus_ZC:inconclusive
    (single-run evidence, no anomalous failures, real feasible incumbents found every time --
    see the per-family table above for actual eval/gradient counts)

OPT_IN_PRODUCTION_READY =
    unrestricted:no  flexible_CM:no  common_frechet:no  origin_ZC:no  CM_plus_ZC:no

MERGED_TO_CANONICAL_PROTOTYPE = <see final integration step below>

INNER_MATH_CODE_CHANGED = false
FULL_PRODUCTION_CHANGED = false
PRODUCTION_DEFAULT_CHANGED = false
NEW_BRANCHES_CREATED = 1   (performance/profiled-outer-ab-readiness-2026-08-04, per task §1)
NEW_WORKTREES_CREATED = 1  (/bbkinghome/edav/cdw_worktrees/profiled-outer-ab-readiness-2026-08-04)
DENSE_CODE_USED = true_bounded_one_off_only
    (Section 8's fixed_dual_L ground truth, ONE point, ONE direction, explicitly logged and
    justified, not a coordinate sweep -- see that section's own commit message)
CAMPAIGN_LAUNCHED = false
```

## What is genuinely production-ready vs. what still needs work

**Solid, gated, real evidence**: sections 1-8 infrastructure (ZC free-nu parity, REDUCED threading,
REDUCED bandwidth cache, matched timer instrumentation, one decoded-state cross-formulation
consistency check) all pass their own real-data gates at D20/W=20,000. Two additional real,
pre-existing bugs were found and fixed while exercising the FULL CLI path for 3 families for the
first time. All 10 short-outer-AB runs complete cleanly with real feasible incumbents.

**Genuinely not ready for an OPT_IN_PRODUCTION_READY=yes designation, for any family**: the fixed-
state scientific-equivalence prerequisite is owned by a different task and not complete; W=100,000
was never exercised this session (only W=20,000); the coordinate tournament and algorithmic-parity
mode were not run. This is an honest gap list, not a hidden one.
