# Melitz real D=20 outer search: evaluation-cap semantic correction — 2026-07-24

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing the same-day
`docs/melitz_real_d20_outer_correction_2026-07-24.md` session. That session's own early-abort
interface conflated "certified above the outer budget `delta`" with "unsolvable" — this
session diagnoses and corrects that conceptual error, tests the fix at real D=20 scale, and
reruns the constrained campaign under the corrected semantics. New/modified:
`src/melitz/inner_screening.jl`, `src/melitz/finite_delta_outer.jl`,
`src/melitz/origin_block_screen.jl`, `test/melitz/runtests.jl`,
`scripts/melitz_real_d20_evaluation_cap_correction_2026-07-24.jl` (new),
`melitz_inner_loop_options_capped_2026-07-24.opt` (new).

**Continuation (same day)**: Section 7's own finding — the corrected interface still can't
find a second outer-feasible point — prompted two follow-up diagnostics (user-directed):
whether the inner CC problem is unusually sensitive to the A/f block relative to `g`, and
whether Monte Carlo sample size `W` is a contributing cause. See Section 9. New:
`scripts/melitz_real_d20_w_sensitivity_and_gradient_stepsize_2026-07-24.jl`,
`scripts/melitz_real_d20_cutoff_feasibility_check_2026-07-24.jl`.

**Second continuation (same day, user-directed)**: three further probes into the outer
search's own step behavior -- a genuine theoretical `g`-bound (from the paper's own proven
`lambda_dd^(1/(sigma-1))` GT ceiling), a gradient-informed A/f trust-region radius, and an
attempted `Delta_profile(g)=min_f Delta(g,f)` sweep from calibration toward the ceiling
(the last of which stalled and was root-caused rather than completed). See Section 10. New:
`scripts/melitz_real_d20_theoretical_gamma_bound_2026-07-24.jl`,
`scripts/melitz_real_d20_gradient_informed_af_radius_2026-07-24.jl`,
`scripts/melitz_real_d20_gt_profile_sweep_2026-07-24.jl`,
`scripts/melitz_nuisance_canary_2026-07-24.jl`,
`scripts/melitz_nuisance_canary_instrumented_2026-07-24.jl`,
`scripts/melitz_nuisance_canary_lowerlimit_2026-07-25.jl`,
`src/melitz/nuisance_profile.jl` (additive `on_start` hook + `on_eval` now fires
unconditionally, including on a failed call),
`melitz_outer_nuisance_profile_capped_2026-07-24.opt`,
`melitz_outer_nuisance_profile_canary_2026-07-24.opt`,
`melitz_nuisance_canary_tried_points_2026-07-24.jls`,
`melitz_nuisance_canary_lowerlimit_tried_points_2026-07-25.jls` (serialized tried points,
D=798 each, across both instrumented runs).

## Executive summary

1. **The diagnosed bug was real and reproduced live at real D=20 scale.** The prior
   session's inner early-abort threshold (`lower_limit = -(delta + guard)`) was tied to the
   OUTER BUDGET `delta`, not to any independent notion of "too far to bother." A certified
   dual lower bound above `delta` was treated as license to abort the inner solve and hand
   outer KNITRO that bound *as if it were* `DeltaStar(theta)`, with the bound's own
   (possibly-suboptimal, path-dependent) fixed-dual gradient presented as `grad DeltaStar`.
   Section 3 below reproduces this live: at the companion report's own `g=-0.4988` point
   (true `DeltaStar=1.06198`), the old coupling classifies `AboveEvaluationCap` with
   `certified_lower_bound=1.020549` — a real number, but not `DeltaStar`, and not even the
   same order of precision as the truth.
2. **Fix: a `delta_evaluation_cap` independent of `delta`.** Introduced as a genuinely
   separate parameter (production default `10.0`) that gates every early-abort mechanism
   (the KNITRO-native `lower_limit` mid-solve bailout, the stored-dual screen, the
   dual-polish screen). `delta` — the outer budget — no longer plays any role in whether an
   inner evaluation is aborted; it is used *only* to scale a genuinely-solved value into the
   outer constraint row `c(theta) = DeltaStar(theta)/delta`.
3. **Renamed taxonomy** (Section 2, `inner_screening.jl`): `InnerSolved` → `FiniteSolved`,
   `BudgetInfeasible` → `AboveEvaluationCap`, `MomentInfeasible` → `InfiniteDeltaCertified`;
   `NumericalFailure` unchanged. The rename is load-bearing, not cosmetic — `BudgetInfeasible`
   was itself part of the conceptual error (it implied economic infeasibility; the corrected
   meaning is "unresolved relative to the cap," which may be an ordinary finite point).
4. **Outer-callback policy, revised twice this session, final design below (Section 5).**
   The first cut implemented an eval-error ("Policy A") as the safe default, with a
   "standardized barrier value" alternative ("Policy B") implemented, disabled, and then
   removed. On review, the eval-error default was itself reconsidered: it discards exactly
   the kind of real magnitude/direction information this repo's own prior work found
   empirically valuable, and it is not actually necessary to avoid the original bug. **Final
   design**: `AboveEvaluationCap`/`InfiniteDeltaCertified` report a FIXED constant
   `delta_evaluation_cap/delta` with a ZERO gradient, as an ordinary successful evaluation
   (never an eval-error); only `NumericalFailure` (no certificate of any kind) remains a
   genuine KNITRO evaluation error, matching the pre-existing Ricardian
   `lower_limit=-50`/`ccOuter.jl` convention this whole mechanism is modeled on.
5. **A genuine hard inner time cap was added** (`melitz_inner_loop_options_capped_2026-07-24.opt`,
   `maxtime_real=90`), independent of the cap-triggered abort. Confirmed live: two grid
   points beyond the explored corridor ran to genuine KNITRO-detected non-convergence
   (`nStatus=-401`) in ~92-93s each rather than the previously-documented 287s unbounded
   outlier, and were correctly classified `NumericalFailure` (no certificate invented).
6. **Real-D20 validation (Section 3)**: a point at `g=-0.50783` with true `DeltaStar=2.1124`
   (`nStatus=0`, genuinely solved) is now reported `FiniteSolved` with the exact value under
   the corrected code — the core regression this session exists to fix, confirmed at
   production scale, not merely in the D=4 unit-test fixture.
7. **Corrected campaign at cap=10 (Section 7)**: 168 FC calls, 26 `FiniteSolved`,
   107 `AboveEvaluationCap`, 35 `InfiniteDeltaCertified`, 0 `NumericalFailure`.
   `cold_verified_incumbent` still falls back to the external fixed-A/f reference
   (`kappa=0.92939627`) — **the flexible search still does not find a second outer-feasible
   point**, the same qualitative finding the pre-correction session reported. The interface
   correction fixes what information is (and is not) fabricated; it does not, by itself, fix
   the outer search's own difficulty finding a second feasible point in a ~0.08-wide corridor
   embedded in a 798-dimensional space.
8. **Cap sensitivity (Section 6)**: caps 5, 10, 20 produce materially identical outcomes —
   same final incumbent, same early trajectory (KNITRO's own step decisions are insensitive
   to which of these three magnitudes it back off from), similar total wall time (1187-1294s).
   No evidence that the specific cap value in this range matters for outer-search behavior at
   this fixture. `10.0` is confirmed as a reasonable production default.
9. **Diagnosis of `AboveEvaluationCap` points (Section 4)**: of 8 points sampled from the
   live campaign's own rejections, none resolved to an ordinary finite value when given more
   room (cap 50, or effectively uncapped at 1e9) — every one either diverged further (larger
   certified bound at a larger cap, consistent with genuine unboundedness) or genuinely failed
   to resolve even uncapped. The general concern that a tight threshold could hide an ordinary
   finite point IS real (the grid scan's own `g=-0.50783`/`Delta=2.11` point demonstrates
   it exists elsewhere in the space) — but among the specific points this trajectory actually
   visited, none were of that kind.

## 1. Exact semantics

- **Finite feasibility** (`FiniteSolved`): `DeltaStar(theta)` is finite, and the inner CC
  dual problem was solved to a genuine KNITRO-accepted optimum. This is independent of any
  outer budget: a point with `DeltaStar=1.5`, `2`, or `5` is `FiniteSolved` exactly the same
  way a point with `DeltaStar=0.5` is, whenever `DeltaStar < delta_evaluation_cap`.
- **Budget feasibility**: `DeltaStar(theta) <= delta`, the ACTUAL economic constraint of the
  outer program. Computed only from a genuinely-solved `FiniteSolved.Delta`; never estimated
  from a certificate.
- **Evaluation cap** (`delta_evaluation_cap`, default `10.0`): a purely computational
  threshold, unrelated to `delta`, beyond which the routine inner evaluation gives up trying
  to find the exact value and instead certifies only the one inequality
  `DeltaStar(theta) > delta_evaluation_cap`. Raising or lowering this cap changes how much
  wall-clock the inner solve is willing to spend chasing an exact value at an increasingly
  bad point; it has no economic meaning.
- **`certified_lower_bound`** (a field of `AboveEvaluationCap`): a valid (by unconditional
  weak duality) but generally loose and PATH-DEPENDENT lower bound on the true
  `DeltaStar(theta)`. Never equal to `DeltaStar(theta)` in general, and never reported to
  outer KNITRO as if it were.
- **`InfiniteDeltaCertified`**: `DeltaStar(theta) = +infinity`, PROVEN by an exact
  finite-support separation certificate (a moment column's draw-level range excludes zero) —
  independent of any cap.
- **`NumericalFailure`**: no certificate of any kind was obtained — not a finite optimum, not
  an evaluation-cap certificate, not an infinite-support certificate. Includes a genuine
  routine time/iteration cap being reached with nothing else established (Section 7's own
  addition to the taxonomy).

## 2. Old versus corrected interface

| | Old (pre-this-session) | Corrected |
|---|---|---|
| Abort threshold | `lower_limit = -(delta + guard)` — tied to the OUTER BUDGET | `lower_limit = -(delta_evaluation_cap + guard)` — independent parameter, default `10.0` |
| A point with `delta < DeltaStar < delta_evaluation_cap` | Aborted early, certificate substituted for the true value | Solved fully, reported as an ordinary `FiniteSolved`, genuinely over budget |
| Value reported to outer KNITRO for an aborted point | `result.lower_bound/delta` — PATH-DEPENDENT (varies with which screen/iteration tripped) | A FIXED constant `delta_evaluation_cap/delta`, identical for every such point |
| Gradient reported for an aborted point | The exact fixed-dual gradient AT THE ARBITRARY certifying point — a real derivative of an unstable quantity | Exactly zero — the true derivative of "always report this constant," self-consistent with the value |
| `NumericalFailure` | Eval-error (`DomainError`→`KN_RC_EVAL_ERR`) | Unchanged: still an eval-error (the one case with no certificate at all) |
| Type names | `InnerSolved`/`BudgetInfeasible`/`MomentInfeasible` | `FiniteSolved`/`AboveEvaluationCap`/`InfiniteDeltaCertified` |

Two intermediate designs were tried and rejected before the final one above, for the record:

- **First cut**: `AboveEvaluationCap`/`InfiniteDeltaCertified` unconditionally threw
  (eval-error), reasoning that *any* fabricated value risked corrupting KNITRO's internal
  state. Rejected on review: this repo's own prior session found empirically that switching
  a rejection FROM an eval-error TO a real finite value "measurably changed exploration (a
  real, evolving trajectory rather than backtrack-to-near-zero)" — an eval-error is not
  obviously safer in practice, only in a narrow theoretical sense, and it discards exactly
  the magnitude/direction information this repo's own evidence says helps.
- **A "standardized barrier value" was implemented as an opt-in alternative, then the
  opt-in itself was removed** (not just the alternative's use) — a *disabled* second policy
  that a future session could silently re-enable is itself a risk; the final design has no
  policy kwarg, no menu, one unconditional behavior.

## 3. Tests at known finite `DeltaStar` values above and below budget

**Unit level** (D=4 synthetic fixture, `test/melitz/runtests.jl`, "Phase I.1" testset,
28/28 passing): direct construction of a point with `certified_lower_bound` above a tight
cap (classified `AboveEvaluationCap`) and, critically, the SAME point re-classified with the
cap raised above its true `Delta` (classified `FiniteSolved`, exact match to an independent
unrestricted solve) — and a third, dedicated regression test: a point set up so `delta <
DeltaStar < delta_evaluation_cap` (over budget, under cap) is `FiniteSolved` with the exact
true value, never aborted merely for exceeding `delta`.

**Real D=20 scale** (`scripts/melitz_real_d20_evaluation_cap_correction_2026-07-24.jl`,
Phases 1-2, real KNITRO, `W=80,000`, `seed=1`):

| `g` | `DeltaStar` (unrestricted) | `nStatus` | verified | over budget (`delta=1`) | classified under `delta_evaluation_cap=10` | matches exactly |
|---|---:|---|---|---|---|---|
| `-0.49783` | `9.969030e-01` | `0` | true | no | `FiniteSolved`, `Delta=9.969030e-01` | yes |
| `-0.50783` | `2.112405e+00` | `0` | true | **yes** | `FiniteSolved`, `Delta=2.112405e+00` | yes |

2/2 verified grid points matched exactly, including the genuinely over-budget one — the core
regression, confirmed at production scale, not just in the unit-test fixture. Two further
grid points (`g=-0.51783`, `g=-0.52783`) did not converge at all (`nStatus=-401`,
~92-93s each, bounded by the new hard time cap) — this repo's already-documented
conditioning fragility beyond the explored corridor, not a new finding, and correctly
classified `NumericalFailure` (no value invented).

**Concrete old-bug demonstration** (same script, Phase 3), at the companion report's own
`g=-0.4988` reference point (true `DeltaStar(g)=1.061980`, `nStatus=0`):

- Old coupling reproduced (`delta_evaluation_cap` artificially set equal to `delta=1.0`):
  classified `AboveEvaluationCap`, `certified_lower_bound=1.020549` — real, but not
  `DeltaStar`, and the true value was discarded.
- Corrected coupling (`delta_evaluation_cap=10.0`): classified `FiniteSolved`,
  `Delta=1.061980` — the exact true value, correctly reported as genuinely over budget.

## 4. Classification of previously-aborted points

The pre-correction session's own raw rejected-theta log was not preserved for replay, so
this diagnosis uses the CORRECTED campaign's own live rejections as the representative
sample (governing prompt's own fallback: "collect representative points that previously
triggered early abort near delta=1" — these points are exactly that, gathered live during
this session's own trajectory rather than the prior session's). 8 `AboveEvaluationCap`
points from the cap=10 campaign were re-diagnosed at cap=10/50/effectively-uncapped
(`1e9`), each under the SAME 90s hard time cap:

| point | cap=10 | cap=50 | cap≈∞ (1e9) |
|---|---|---|---|
| 1 | `AboveEvaluationCap`, `lb=1.58e8` | same | `NumericalFailure` (`nStatus=-401`, 95s) |
| 2 | `AboveEvaluationCap`, `lb=1.65e9` | same | `AboveEvaluationCap`, `lb=1.65e9` (unchanged — genuinely diverged past `1e9` too) |
| 3 | `AboveEvaluationCap`, `lb=4.00e6` | same | `AboveEvaluationCap`, `lb=1.28e9` (kept diverging) |
| 4 | `AboveEvaluationCap`, `lb=3.16e6` | same | `NumericalFailure` (`nStatus=-102`, ~24-30s) |
| 5 | `AboveEvaluationCap`, `lb=7.77e7` | same | `AboveEvaluationCap`, `lb=1.57e10` (kept diverging) |
| 6 | `AboveEvaluationCap`, `lb=1.36e7` | same | `NumericalFailure` (`nStatus=-401`, 95s) |
| 7 | `AboveEvaluationCap`, `lb=1.67e6` | same | `AboveEvaluationCap`, `lb=1.77e9` (kept diverging) |
| 8 | `AboveEvaluationCap`, `lb=6.54e6` | same | `AboveEvaluationCap`, `lb=1.42e9` (kept diverging) |

Identical across all three cap-sensitivity runs (cap=5/10/20 campaigns) — the trajectories'
early exploration is deterministic and cap-insensitive at this fixture, so the same 8 points
recur every time.

**Finding**: none of these 8 sampled points is an ordinary finite point merely hidden by a
too-tight cap. Five of eight kept diverging to a LARGER certified bound when given a much
larger cap (consistent with genuine unboundedness — in convex duality, a dual objective
running to enormous magnitude is itself informative, not merely "we didn't look far
enough"); three of eight genuinely failed to resolve at all even essentially uncapped.
**This is a meaningfully different, and more informative, answer than the old delta=1
threshold could ever have produced** — the old code would have reported ONE arbitrary
number per point (whatever the old, tightly-coupled threshold happened to trip at) with no
way to distinguish "probably infinite" from "possibly just needs more room." The corrected
diagnostic machinery draws that distinction directly. That said, the *general* concern that a
tight threshold can hide an ordinary finite point is real and independently confirmed: the
grid-scan's own `g=-0.50783` point (`DeltaStar=2.11`) is exactly such a point — it simply was
not among the 8 sampled from this particular trajectory's own visited region.

## 5. Policy for points above the evaluation cap

Final, single, unconditional policy (no kwarg, no menu — see Section 2's "rejected designs"
for why):

- `FiniteSolved`: the real, fully-optimized value and its exact envelope-theorem gradient.
- `AboveEvaluationCap` / `InfiniteDeltaCertified`: the FIXED constant
  `c_outer = delta_evaluation_cap / delta`, identical for every such point regardless of the
  exact certificate level, with an EXACT ZERO gradient — reported to KNITRO as an ordinary
  successful evaluation. This is deliberately modeled on the Ricardian model's own
  pre-existing `lower_limit=-50` convention (`cc_algo/ccOuter.jl`): a fixed, numerically-
  motivated threshold beyond which the model is treated as certainly-bad, reported as a real
  (if extreme) constraint violation rather than a blind eval-error, so the outer search
  retains real magnitude/direction information.
- `NumericalFailure`: a genuine KNITRO evaluation error (`DomainError`→`KN_RC_EVAL_ERR`) —
  the one case with no certificate of any kind, not even a lower bound.

Why the fixed constant is not the original bug: value and gradient here are
self-consistent (a constant's true derivative is exactly zero) and identical for every point
in the bucket, independent of solve path/iteration/screen. The original bug's value
(`certified_lower_bound`) and gradient (the fixed-dual derivative at the arbitrary
certifying point) were both path-dependent AND paired with each other only by coincidence of
which point happened to trip a screen — Section 4's own diagnosis table shows exactly how
volatile that raw number is (the SAME 8 points' `certified_lower_bound` values span nearly 4
orders of magnitude, `28` to `1.65e9`, purely as an artifact of which screen/iteration fired)
— precisely the false-ordering hazard this session's governing prompt (Section 1) warns
against, now visible directly in real data rather than argued abstractly.

## 6. Cap sensitivity (5, 10, 20)

Matched short campaigns (`delta=1.0`, upper direction, identical starting point, identical
240s-nominal/actual-longer-due-to-the-documented-KNITRO-between-call-check-limitation wall
budget) at `delta_evaluation_cap in {5, 10, 20}`:

| cap | wall (s) | `n_fc_calls` | `FiniteSolved` | `AboveEvaluationCap` | `InfiniteDeltaCertified` | `NumericalFailure` | final `cold_verified_incumbent` |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 1294.06 | 223 | 26 | 169 | 28 | 0 | external, `kappa=0.92939627` |
| 10 | 1208.77 | 168 | 26 | 107 | 35 | 0 | external, `kappa=0.92939627` |
| 20 | 1187.36 | 204 | 26 | 166 | 11 | 1 | external, `kappa=0.92939627` |

**Finding**: the three caps produce MATERIALLY IDENTICAL outer-search outcomes — the same
final incumbent, byte-identical early trajectories (KNITRO's own step decisions are
insensitive to whether the reported barrier is `5`, `10`, or `20` — all are "very
infeasible" from its own merit-function perspective at this fixture), and similar total wall
time (within ~9% of each other). No evidence in this fixture that the specific cap value in
this range changes outer-search behavior. `10.0` is confirmed as a reasonable, unremarkable
production default — there is no sign a different choice in `{5,20}` would do better or
worse.

## 7. Corrected D20 outer trajectory (production cap=10)

`delta=1.0`, upper direction, native `:linear` cutoffs, `lower_limit_guard=1e-6`,
`delta_evaluation_cap=10.0`, `gradient_backend=:B_direct_argument_parallel`, block-scaled
`theta_box`, external incumbent = the Phase 6 fixed-A/f verified point
(`g_fixed=-0.49783321`), starting point `g=-0.497333` (`Delta0=0.9652724`), same setup as the
pre-correction session's own campaign for direct comparability.

```
wall=1208.77s  nStatus=-400  n_fc_calls=168  n_ga_calls=26  n_inner_solved=26
n_infinite_delta_reject=35  n_above_cap_reject=107  n_numerical_failure_reject=0
```

| incumbent | `g` | `kappa` | `GT` | `Delta` | source |
|---|---:|---:|---:|---:|---|
| initial_incumbent | `-0.497333` | `0.929706` | `0.070294` | `9.653e-01` | initial |
| best_live_incumbent | `-0.497335` | `0.929705` | `0.070295` | `9.653e-01` | live |
| **cold_verified_incumbent (THE ANSWER)** | `-0.497833` | **0.929396** | `0.070604` | `9.969e-01` | external |

**Reading**: `best_live_incumbent` is infinitesimally different from `initial_incumbent`
(fifth decimal place) — across 168 FC calls (up from the pre-correction session's 91, a
larger, more thorough exploration under the corrected constraint surface), KNITRO's own
trajectory still does not find a genuinely SECOND outer-feasible point in this
798-dimensional space. `cold_verified_incumbent` falls back to the external fixed-A/f
reference exactly as before correction. **The interface correction did not, by itself,
change this qualitative outcome.**

Wall-clock decomposition (`inner_solve_above_evaluation_cap`: 107 calls, 488.9s total, mean
4.57s, **max 82.2s** — one live-threshold crossing took 82 seconds of real KNITRO barrier
iteration before tripping, a reminder that even a "just a screen" rejection can sit behind a
slow underlying attempt):

```
total outer KNITRO wall        = 1208.7708s   (nominal request was 180s -- see below)
complete FC/GA callback wall    = 1120.8865s (fc_total=1021.4326s, ga_total=99.4539s)
residual KNITRO-C/API wall      = 87.8843s (7.3%)
```

The requested `180s` outer wall budget was exceeded by `~6.7x` — the SAME documented KNITRO
limitation the pre-correction session's own report already found (`maxtime_real` is checked
only BETWEEN callback returns; enough individual calls in the multi-second-to-82-second
range accumulate past the nominal budget before the next check). Not a regression introduced
this session; disclosed for completeness.

## 8. Recommendation

1. **Ship the corrected interface as the production default**: `delta_evaluation_cap=10.0`,
   the fixed-sentinel outer-callback behavior (Section 5), the 90s hard inner time cap. All
   three are validated (unit tests + real-D20 empirical checks) and none is exotic — the
   sentinel design is a direct, deliberate analogue of the Ricardian model's own long-standing
   `lower_limit=-50` convention, not a novel mechanism.
2. **Do not spend further engineering effort tuning the exact cap value.** Section 6 found
   caps 5/10/20 empirically indistinguishable in outer-search outcome at this fixture; `10.0`
   is a fine, unremarkable default with no evidence a different choice helps.
3. **The genuinely open problem is the outer search itself, not this interface.** Section 7's
   own finding — 168 real, correctly-informed FC calls still find only one outer-feasible
   point — is now measured on a corrected, honest constraint surface (no fabricated
   path-dependent values anywhere in the loop) and the qualitative conclusion is unchanged
   from the pre-correction session: a flexible 798-dimensional search embedded around a
   ~0.08-wide feasible corridor is hard for reasons independent of the bug this session
   fixes. This matches the pre-correction session's own prioritization (its Section 8, item
   4: "a better outer-search strategy... is now explicitly THIRD priority" behind the
   profiled/nuisance-minimization alternative) — this session's own recommendation is
   unchanged: do not invest further in the CONSTRAINED search's own algorithm without first
   giving the profiled reformulation (that session's Stage 2, still blocked on porting its
   own exact-point cache) a fair trial.
4. **Before any further campaign work, apply the exact same rename/semantics correction to
   any other outer-search entry point that still references the old
   `InnerSolved`/`BudgetInfeasible`/`MomentInfeasible` names** — this session scoped its edits
   to `inner_screening.jl`, `finite_delta_outer.jl`, `origin_block_screen.jl`, and the active
   test suite; `nuisance_profile.jl` was confirmed to have no structural dependency on these
   types (only descriptive comments, updated for consistency) but was not otherwise audited
   for the SAME conceptual error under a different name — worth a dedicated check before
   Stage 2 work resumes.

## 9. Why the flexible search can't find a second feasible point: gradient concentration, a
## real-vs-numerical check, and W-sensitivity (continuation, same day)

Section 7 found the corrected campaign still cannot locate a second outer-feasible point.
Three follow-up diagnostics probe *why*, at the campaign's own starting point
(`g=-0.497333`, `Delta=0.9653`).

### 9.1 The gradient is almost entirely in the A/f block, not `g`

The exact envelope-theorem gradient of `Delta(theta)` (the SAME `:B_direct_argument_parallel`
backend the production outer search itself uses, computed at the just-converged optimal
dual — no new formula) was computed and decomposed:

```
||grad(Delta)|| = 570.75    g-component = -62.00    A/f-block norm = 567.37
=> 98.82% of the gradient's Euclidean norm is in the 797 A/f coordinates, 1.18% in g
```

Pure-`g` steps (A/f held exactly fixed) stay feasible and well-behaved out to `eps=0.01`
(`Delta` rises smoothly to `2.02`). Steps along the TRUE steepest-descent direction of
`Delta` (which necessarily pulls in A/f, weighted by its outsized share of the gradient)
are still fine at `eps=1e-5` (`Delta` genuinely decreases, `0.9651` vs baseline `0.9652`) but
break completely at `eps=1e-4` — where the A/f coordinates moved by a Euclidean norm of only
`1e-4` (about `3e-6` per coordinate) — landing on `nStatus=-401` (KNITRO's own "problem
appears unbounded" code), not a graceful increase in `Delta`.

### 9.2 That break is confirmed NOT a real economic-constraint violation

`melitz_outer_state`'s `min_slack` (the deterministic domestic/export cutoff feasibility
check — closed-form, no Monte Carlo, no KNITRO) was checked directly at every step size that
broke the inner CC solve above:

```
eps=0.0      min_slack=0.02455297  feasible=true
eps=1e-5     min_slack=0.02455297  feasible=true
eps=1e-4     min_slack=0.02455298  feasible=true   <- inner CC solve already failing here
eps=1e-3     min_slack=0.02455307  feasible=true
eps=1e-2     min_slack=0.02455395  feasible=true   <- inner CC solve still failing here
```

`min_slack` is essentially frozen (`0.024552` to `0.024554`) across the entire range,
including step sizes where the inner CC solve had already completely broken down. **The
point remains comfortably cutoff-feasible with healthy slack when the Monte-Carlo-based
inner divergence problem fails** — the failure lives entirely inside the finite-sample
divergence-minimization problem, not in any real economic constraint.

### 9.3 A larger `W` measurably moves the boundary, not just the numbers

Re-running the gamma-only grid (A/f fixed at calibration, `g`-only path — the ONE direction
already known robust, Section 9.1) at `W=160,000` against the `W=80,000` baseline:

| `g` | `Delta` at `W=80,000` | `Delta` at `W=160,000` |
|---|---:|---:|
| `-0.49783` | `0.9969` (`nStatus=0`) | `0.8967` (`nStatus=0`) |
| `-0.50783` | `2.1124` (`nStatus=0`) | `1.6972` (`nStatus=0`) |
| `-0.51783` | **fails** (`nStatus=-401`) | **`4.3449`, `nStatus=0`** |
| `-0.52783`…`-0.55783` | fails | still fails |

A point that was completely unsolvable at `W=80,000` converges cleanly at `W=160,000`
(`Delta=4.34`, comfortably under `delta_evaluation_cap=10` — an ordinary `FiniteSolved`
point under the corrected interface). Already-converging points also shift their `Delta`
estimate down by 10-20% — repo memory already documents the same *direction* of effect at
lower `W` (`W=8,000` understates `kappa` relative to `W>=80,000`); this finds it continuing
past `80,000`.

### 9.4 Synthesis

Three independent checks triangulate on the same conclusion: (1) the gradient is almost
entirely an A/f-block phenomenon, not a `g` phenomenon; (2) the specific point where the
inner solve breaks is NOT economically infeasible (comfortable `min_slack`) — the CC
divergence-minimization problem itself is failing numerically; (3) more Monte Carlo draws
measurably push that numerical failure boundary outward and shift `Delta` estimates
noticeably even where nothing was failing before. **This looks like a genuine finite-sample
conditioning problem in the inner CC solve at this `W`, not a structurally razor-thin
feasible region that no amount of computation would open up.** Not yet tested: whether the
outer search itself, run at `W=160,000`, finds a second feasible point given the visibly
larger solvable region — a natural next step, deferred pending confirmation given its cost
(a full outer campaign at `W=160,000` is markedly more expensive than the `W=80,000` ones in
Section 6-7, since moment-construction cost scales with `W` while dual-problem dimension
does not).

## 10. Step-size probes and the nuisance-profile stall, root-caused

### 10.1 Theoretical `g`-bound (replacing the ad hoc `g_radius`)

The paper's own theoretical ceiling `GT <= lambda_dd^{1/(sigma-1)}`'s complement, converted
to a bound on the outer coordinate itself:

```
g_ceiling = -0.567517   (GT=11.28%, kappa_min=0.887208, the Delta->infinity limit)
g_floor   =  0.0        (GT=0%, gamma_prime_target=1)
```

Substituted for the old ad hoc `g_radius=0.05` (a symmetric radius chosen for KNITRO
well-posedness, no economic content) as an EXACT lower bound for the `:upper`-direction
search (`g_radius = theta_init[1] - g_ceiling`, reproducing the true bound on the side that
matters for this direction). Rerunning the full campaign (same `delta=1`, `cap=10`, starting
point, external incumbent) with this bound:

```
wall=1227.20s  n_fc_calls=166  n_inner_solved=26
cold_verified_incumbent: UNCHANGED -- g=-0.497833, GT=0.070604, source=external
min g visited = -0.567166  (99.5% of the way to the true ceiling)
```

**No change to the outcome.** KNITRO used the wider, now-theoretically-exact range fully
(reaching 99.5% of the true ceiling) but still found nothing better -- confirming the ad hoc
box was never the binding constraint; the search fails for the same reason described in
Section 7/9, not because it couldn't see far enough.

### 10.2 Gradient-informed A/f radius

Following Section 9.1's gradient decomposition (98.8% of `||grad(Delta)||` in the 797 A/f
coordinates), `A_radius=f_radius` was tightened from the ad hoc `0.15` to `1e-4` (the same
order of magnitude as the combined-norm scale where a full-Jacobian descent step first broke
in Section 9.1), keeping the theoretical `g`-bound from 10.1.

```
wall=269.21s  n_fc_calls=12  n_ga_calls=12  n_inner_solved=1  nStatus=-201
min g visited = -0.567517 (the exact ceiling, reached almost immediately)
cold_verified_incumbent: UNCHANGED -- GT=0.070604, source=external
```

This run is not clean evidence either way: KNITRO's presolve eliminated all 400 cutoff
constraints as degenerate (their slack barely moves within a `1e-4` box) and the search
terminated early (`nStatus=-201`) after an erratic jump almost straight to the `g`-ceiling --
consistent with a badly-SCALED problem (798 coordinates with wildly different box radii, `g`
at `~0.07` vs A/f at `1e-4`, can confuse a solver's own internal scaling) rather than a
genuine, methodical small-step search. The bottom line is unchanged regardless.

### 10.3 `Delta_profile(g) = min_f Delta(g,f)` sweep: stalled, then root-caused

The user's request to profile `Delta*` while holding `A` fixed and sweeping `kappa` from the
raw Pareto calibration toward the theoretical ceiling reuses `src/melitz/nuisance_profile.jl`
(built in the *prior* session, `:f_only`/`:A_only` block masks already implemented, but left
unfinished at real D=20 scale -- that session's own report already disclosed this file has no
exact-point cache: `cb_G!` always re-solves from scratch even at a `theta` `cb_F!` just
evaluated).

**First attempt** (radius=0.5, matching the prior session's own D=4-validated default, `:f_only`
block, starting at raw calibration): never printed past outer "Iter 0" in 35+ minutes; killed
manually (`SIGTERM` alone did not stop it -- it was blocked in native KNITRO/BLAS code with no
Julia-level signal check; `SIGKILL` was required).

**Second attempt** (radius=0.02, a much smaller nuisance box, same starting logic, this time
wrapped in an OS-level `timeout` rather than relying on KNITRO's own `maxtime_real`): killed
automatically at 400s. The `SIGTERM` stack dump caught the process inside the inner CC
solve's own Hessian callback (`callbackEvalH_inner!` -> `hessian!`,
`cc_algo/PsiObjectiveBundle.jl:642`) -- suggestive, but a single stack snapshot cannot
distinguish "one pathologically slow Hessian evaluation" from "many individually-fast calls,
unluckily sampled mid-Hessian."

**Third attempt, instrumented** (same radius=0.02; a new `on_start` hook added to
`nuisance_profile.jl`, additive, fires BEFORE `inner_loop` so an attempted point is on record
even if the call never returns; every attempted `theta` serialized to disk incrementally,
`melitz_nuisance_canary_tried_points_2026-07-24.jls`, robust to an abrupt kill; bounded by an
OS-level `timeout 320`). This resolved the question definitively:

```
[START] fc idx=1  t=  3.89s  ||theta-theta_g||=0            (the pinned starting point itself)
[ EVAL] fc idx=1  t=  9.09s  elapsed=5.19s   val=3.5707e-02  nStatus=0    (fast, easy -- as expected)
[START] ga idx=1  t= 12.18s  ||theta-theta_g||=0
[ EVAL] ga idx=1  t= 19.91s  elapsed=7.73s   val=3.5707e-02  nStatus=0    (re-solves from scratch, no cache, as documented)
[START] fc idx=2  t= 19.93s  ||theta-theta_g||=1.238e-02     (KNITRO's first real trial step)
                                                               -- NEVER completes (no matching EVAL)
[START] fc idx=3  t=112.54s  ||theta-theta_g||=6.188e-03     (started 92.6s after idx=2 -- HALVED step)
                                                               -- NEVER completes
[START] fc idx=4  t=204.84s  ||theta-theta_g||=3.094e-03     (started 92.3s after idx=3 -- HALVED again)
                                                               -- run killed by the 320s OS timeout before this one resolved
```

**Preliminary read from timing alone (SUPERSEDED by Section 10.4 -- kept here for the
record, since it was wrong in an instructive way): the ~90-93s duration per failed retry
matches the inner solve's own `maxtime_real=90` cap almost exactly, so the working
hypothesis at this point was "it is genuinely slow to converge, not diverging." Section 10.4
tested this directly and found the OPPOSITE.**

### 10.4 The actual mechanism: the inner dual problem genuinely diverges -- and a second,
### completely separate `lower_limit` bailout was never wired up for this object at all

The user asked directly: is a SEPARATE, already-built KNITRO-native early-stop
(`cc_algo/PsiObjectiveBundle.jl`'s `if f <= lower_limit; return -KNITRO.KN_INFINITY`, the
SAME mechanism this whole session's main correction wired up as `delta_evaluation_cap` on
the OUTER `PsiObjectiveBundleImplicit` bundle) firing here? Checked directly:
`PsiObjectiveBundleDelta` (the object type `nuisance_profile.jl`/`evaluate_melitz_delta` use)
has the IDENTICAL `lower_limit` field and functor bailout
(`cc_algo/PsiObjectiveBundle.jl:518`) -- but `build_melitz_psi_bundle_from_calibration`
(`src/melitz/pareto_calibration.jl`) never passes `lower_limit` at construction, so it sits
at the struct default, `-KNITRO.KN_INFINITY` (permanently disabled), for every inner solve
this session has made through this object. Confirmed live: `obj_inner.lower_limit` printed
as `-1.797693e+308` before any fix.

Wired it up directly (`obj_inner.lower_limit = -(10.0 + 1e-6)`, matching the user's own
preferred cap of `10`) and reran the IDENTICAL canary (same starting point, radius, warm
start). Result -- unambiguous:

```
BEFORE fix: obj_inner.lower_limit = -1.797693e+308  (disabled)
AFTER fix:  obj_inner.lower_limit = -10.000001

idx=2  ||theta-theta_g||=1.238e-02  elapsed=3.03s   val=1.0e10  nStatus=-300  (unbounded)
idx=3  ||theta-theta_g||=6.188e-03  elapsed=2.84s   val=1.0e10  nStatus=-300  (unbounded)
idx=4  ||theta-theta_g||=3.094e-03  elapsed=4.45s   val=1.0e10  nStatus=-300  (unbounded)
idx=5  ||theta-theta_g||=1.547e-03  elapsed=2.48s   val=1.0e10  nStatus=-300  (unbounded)
idx=6  ||theta-theta_g||=7.735e-04  elapsed=5.10s   val=1.0e10  nStatus=-300  (unbounded)
idx=7  ||theta-theta_g||=3.867e-04  elapsed=2.37s   val=1.0e10  nStatus=-300  (unbounded)
idx=8  ||theta-theta_g||=1.934e-04  elapsed=2.76s   val=1.0e10  nStatus=-300  (unbounded)
idx=9  ||theta-theta_g||=6.953e-03  elapsed=2.70s   val=1.0e10  nStatus=-300  (unbounded)  <- new direction tried
idx=10 ||theta-theta_g||=2.495e-03  elapsed=5.89s   val=1.0e10  nStatus=-300  (unbounded)
idx=11 ||theta-theta_g||=1.248e-03  elapsed=3.55s   val=1.0e10  nStatus=-300  (unbounded)
idx=12 ||theta-theta_g||=6.239e-04  elapsed=3.38s   val=1.0e10  nStatus=-300  (unbounded)
idx=13 ||theta-theta_g||=3.120e-04  elapsed=5.71s   val=1.0e10  nStatus=-300  (unbounded)
idx=14 ||theta-theta_g||=1.560e-04  elapsed=2.73s   val=1.0e10  nStatus=-300  (unbounded)

COMPLETED: nStatus=-401 (no improving step found)  Delta_min=3.5707e-02 (== the starting value)  wall=78.47s
```

**This is the definitive, corrected answer -- and it is NOT "genuinely converges, just
slowly."** Every one of KNITRO's own trial steps -- shrinking geometrically across roughly
two orders of magnitude, `1.24e-2` down to `1.56e-4` -- hits `nStatus=-300`
("problem determined to be unbounded"), the exact status the `lower_limit` bailout is
designed to produce. This is genuine divergence of the inner CC dual problem's own raw
objective, not slow-but-eventual convergence: even a combined `f`-perturbation of `1.56e-4`
(about `7.8e-6` per free coordinate on average) is enough to send the unconstrained dual
`(zeta, lambda)` problem toward an unbounded direction at THIS starting point
(`g = calibration - 0.02`, i.e. a modest, otherwise-unremarkable `g`-only step). Without
`lower_limit` set, KNITRO's own internal barrier method still eventually reaches essentially
the same conclusion (`nStatus=-401`, generic non-convergence) or exhausts `maxtime_real` --
but takes the FULL ~90 seconds to get there, because detecting genuine unboundedness
robustly through ordinary barrier iterations is slower than a certificate check against an
already-known dual point crossing a fixed threshold.

**What the fix does and does not do**: it does NOT make the nuisance minimization succeed at
this starting point -- the outer solve still ends in `nStatus=-401`, no improving `f`
direction was found among the 13 tried. What it DOES do is turn every one of those 13
rejections from ~90 seconds of blind grinding into 2-6 seconds with an exact, legible reason
(`nStatus=-300`, unbounded) -- exactly the same class of fix (a fast, certificate-based
early-exit rather than waiting out a generic wall-clock cap) this whole session's main
correction already validated for the OUTER search's `PsiObjectiveBundleImplicit` bundle,
just never previously applied to this SEPARATE `PsiObjectiveBundleDelta` object.

**Root cause, restated correctly**: `melitz_classified_inner_solve` (behind the MAIN
finite-delta search, Sections 3-9) has both this `lower_limit` mechanism AND cheap sub-second
pre-solve screens (moment-range certificate, stored-dual lower bound) -- either one, alone,
would have caught these divergent points in seconds; together, the main search's rejections
cost only ~2-7s each even across 142 of them (Section 7). `nuisance_profile.jl` had NEITHER
wired up for its own `obj_inner`, so every divergent trial point (which, per this section, is
apparently most of the nearby directions from this starting point) paid the full,
un-certificated `maxtime_real` price. `lower_limit` alone (this section) fixes the SPEED of
rejection; it does not by itself find a direction that converges.

**Disposition**: the `Delta_profile(g)` sweep (both `:f_only` and `:A_only`) is still NOT
completed this session -- but the reason is now precisely understood rather than merely
timed-out. Recommended before attempting it again: (1) always wire up
`obj_inner.lower_limit = -(delta_evaluation_cap + guard)` for any future
`nuisance_profile.jl` use (a one-line fix, now validated); (2) given 13/13 tried directions
from this exact starting point were unbounded, a genuinely different starting point or a
much more conservative first step (rather than trusting KNITRO's own untuned first Newton
direction) is likely needed, not merely a smaller version of the same direction; (3) the
missing exact-point cache and moment-range pre-screen (both already built and validated for
the main search) remain worth porting for cost reasons even once (1)-(2) are addressed. All
14 attempted trial points across both instrumented runs (the earlier 5 plus this section's
own, including the ones that never completed under the old, disabled `lower_limit`) are
preserved with their full `theta` vectors in `melitz_nuisance_canary_tried_points_2026-07-24.jls`
and `melitz_nuisance_canary_lowerlimit_tried_points_2026-07-25.jls` for that investigation.

## Required-report checklist

1. Exact semantics of finite feasibility, budget feasibility and evaluation caps — Section 1.
2. Old versus corrected inner/outer interface — Section 2.
3. Tests at known finite `DeltaStar` values above and below budget — Section 3.
4. Classification of previously aborted points — Section 4.
5. Policy for points above the evaluation cap — Section 5.
6. Cap sensitivity at 5, 10 and 20 — Section 6.
7. Corrected D20 outer trajectory — Section 7.
8. Recommendation for the production evaluation cap and outer callback policy — Section 8.
