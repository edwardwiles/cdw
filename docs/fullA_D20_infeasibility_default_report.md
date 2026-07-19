# Continuation 10, Section 5: exact infeasibility screening wired as the production default

Branch `c10-prod-wiring`, worktree `/bbkinghome/edav/gravity_robustness/gravity-fullA-d4-c10-prod-wiring`.
Measured on `demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, real D=20 France-focal data, W=80,000. Builds directly on
Continuation 9's `docs/fullA_D20_infeasibility_screening_report.md` (the screen's math and
correctness were validated there, 0 false positives across D=4/6/8/10/20) — this task's job
was to make it the DEFAULT production evaluation order and re-confirm the WIRING itself
introduces no regression.

## 1. What changed

**`full_aod_diag/d4_exact/context_real_d20.jl`** (`d20_real_setup`): new `build_screen::Bool
= true` kwarg (default on). When true, builds `precompute_pairwise_M` and
`build_extreme_draw_witness` ONCE, right after the rest of the context, and returns them as
new `ctx.pairwise`/`ctx.witness` fields (plus `ctx.screen_setup_wall`, a
`(pairwise=, witness=)` NamedTuple of build times). This is an additive change to the
returned NamedTuple's field set — every existing caller that accesses fields by name
(the universal convention in this codebase) is unaffected.

**`full_aod_diag/d4_exact/c10_d20_production_driver.jl`** (new, see the Section 6 report for
its full role): every evaluation goes through `evaluate_fullA_screened(...; pairwise =
ctx.pairwise, witness = ctx.witness, use_witness = true)` — the exact order the brief
specifies (pairwise → witness → destination winner-scan with early exit → compressed
moments → CC inner solve). A raw, unscreened `evaluate_fullA_fast` call does not appear
anywhere in this driver's hot path (`screened_eval` wrapper, used by both `cb_F!`/`cb_G!`/
`cb_newpt!` in both the profile and polish stages).

## 2. Witness/pairwise setup cost — built ONCE, confirmed negligible

Measured live, D=20/W=80,000 (`c10_screen_wiring_validate.jl` and the production driver's
own smoke test, both independently):

| structure | build wall | (from Continuation 9's own W=80,000 report, for comparison) |
|---|---|---|
| pairwise certificate | 0.053-0.127s | ~0.05s (not separately timed there) |
| extreme-draw witness | 2.19-2.34s | 2.47s |

Total screen setup: **~2.2-2.5s**, paid ONCE per context (i.e. once per outer-loop
optimization run), vs. a single feasible-point cold dense solve costing **5.8s** (measured
this task) and a real production run evaluating hundreds to thousands of points — setup
cost is negligible relative to a full run, exactly as required.

## 3. Rejection counts and timing on real test runs

**Short real production run** (`c10_prod_driver_smoke_original.jl`, 90s-budget profile
stage, upper branch, starting AT the calibration point): **0 pairwise, 0 witness, 0
winner-scan rejections out of 17 total screened evaluations** — every point passed the
screen and reached the real solve. This is itself an honest, expected finding, not a null
result to be embarrassed about: Continuation 9's own D=20 perturbation sweep
(`docs/fullA_D20_infeasibility_screening_report.md` §4.4) already found that D=20 needs a
LARGE step (≈16 in 399-dim pivot-reduced z-space) before any genuine infeasibility
appears at all — a real KNITRO run started near the calibration point and run for 90s
never wanders that far. The screen is doing its job silently (near-zero overhead, 0
false rejections of real, near-feasible points) exactly as it should on this kind of run.

**Adversarial validation run** (`c10_screen_wiring_validate.jl`, 40 large-perturbation
trials at step sizes in [8,16] unit-norm in pivot-reduced z-space, matching Continuation
9's own construction — this time through the NEWLY WIRED `ctx.pairwise`/`ctx.witness` path,
not ad hoc per-script structures as Continuation 9's own scripts used):

| stage | rejections (of 40) |
|---|---|
| pairwise certificate | 3 |
| witness | 1 |
| winner-scan (full) | 0 |
| passed through to real solve | 36 |

**False-positive re-confirmation**: every one of the 4 rejected points was cross-checked
against the ground-truth full, unhurried winner-scan (`screen_hard_winners(...;
full_scan=true)`, no early exit) — **0/4 false positives**, i.e. the screen holds under the
new context-embedded wiring exactly as Continuation 9 found for the underlying math.
Separately, 10 feasible small-perturbation points (Class 1, step 0.5 in the same
coordinates) all correctly passed the screen (0/10 false rejections of a truly feasible
point).

**Timing, screen-reject vs. real solve** (this task's own measurement, D=20/W=80,000):

| | wall time |
|---|---|
| feasible point (screen passes, real dense solve runs) | 5.8466s |
| pairwise-certified-infeasible point (screen rejects, no solve attempted) | 0.000112s |
| **speedup** | **52,099x** |

Matches Continuation 9's own class-2 finding (">100,000x", §7 of that report) to within the
same order of magnitude — the exact rejection mechanics are unchanged, only the context
wiring is new here.

**Which destinations/cells caused rejection**: the 4 rejections in the adversarial run were
identified via `screen_meta.worst_o`/`worst_d` (pairwise) and the analogous witness fields
— logged per-rejection in `ScreenCounters.rejections` in the production driver (an
auditable `Vector{NamedTuple}` of `(stage, o, d, n_eval)`), available for inspection on any
real run; not enumerated exhaustively here since this was a synthetic adversarial sweep
(random directions), not a real optimization trajectory — a genuine Section-10 frontier
run will have its own specific rejected cells logged in its own `ScreenCounters`.

## 4. Verdict

**Adopted as the default evaluation order in the production driver** (Section 6). Zero
false positives re-confirmed under the new context-embedded wiring (4/4 checked
rejections, plus the underlying math's own 0/many from Continuation 9). Setup cost
(~2.2-2.5s) is negligible against a full run. On a real, near-feasible short production
run, the screen correctly rejects nothing (0/17) — exactly the expected, honest behavior
near a good starting point — while an adversarial sweep confirms it DOES reject genuine
infeasible points when they occur, at a >50,000x cost advantage over attempting the real
solve.

## 5. Files

Modified: `full_aod_diag/d4_exact/context_real_d20.jl` (additive `build_screen` kwarg +
`pairwise`/`witness`/`screen_setup_wall` fields; no existing field removed or changed).

New: `full_aod_diag/d4_exact/c10_screen_wiring_validate.jl` (false-positive
re-confirmation + rejection-count/timing validation, this report's source).
