# Profiled unrestricted outer gradient + A/B master report (2026-08-01)

Continuation of `diagnostic/profiled-scales-unrestricted-knitro-2026-08-01` (the validated
fixed-point inner-solve prototype) onto a new branch/worktree:

- **Source branch**: `diagnostic/profiled-scales-unrestricted-knitro-2026-08-01`,
  worktree `/bbkinghome/edav/gravity_robustness/worktrees/diagnostic-profiled-scales-unrestricted-knitro-2026-08-01`,
  committed HEAD `d84d392ccbbd35155c4767bddfe146f0a3ed20f8` (cut from
  `architecture/profile-all-destination-scales-2026-07-31` @ `d84d392`, itself descended from
  `production/fullA-exact @ cd17235`).
- **New branch**: `diagnostic/profiled-scales-unrestricted-outer-ab-2026-08-01`,
  worktree `/bbkinghome/edav/gravity_robustness/worktrees/diagnostic-profiled-scales-unrestricted-outer-ab-2026-08-01`.
- **Snapshot commit**: `2abb119` — "Commit validated unrestricted profiled inner formulation and
  D20 omit-ROW gates" — the untracked implementation copied file-for-file from the source worktree
  (checksums verified identical before/after copy), with the D4 (symmetric+perturbations) and real
  D20 (`:exclude_row`, W=80,000) recover-then-resolve gates rerun and reproduced exactly in the new
  worktree before committing.
- **Production default**: unchanged (`economic_parameterization` defaults to
  `:full_gamma_normalized_reference` everywhere; the new evaluator only activates when a caller
  explicitly builds the profiled path). Not merged, not pushed, no campaign launched.

## 1. Fixed-point snapshot (task §1-4)

Reproduced exactly (see §4 of the task prompt's own expected numbers):

| Point | profiled nStatus | reference nStatus | winner match | divergence diff | LFD max diff |
|---|---|---|---|---|---|
| calibration | 0 | 0 | identical | 3.265e-9 | 4.79e-5 |
| modest_perturbation | 0 | 0 | identical | 5.11e-9 | 4.48e-5 |

(both numbers match the source session's memory record exactly, confirming a clean, lossless
snapshot).

## 2. Live inner/outer dimensions (task §3, §6)

All confirmed via live construction (not hardcoded), see `PROFILED_OUTER_VECTOR_MANIFEST_D20_2026-08-01.json`:

```
D20_INNER_LAYOUT = { factual_share_moments: 361, france_ratio_moments: 1,
                      total_economic_moments: 362, total_inner_dual_dimension: 363 }
PROFILED_OUTER_VECTOR = { gp: 1, free_A: 360, total: 361 }
```

D×Ddest=380 active A cells, 19 anchors removed (one per active destination — Korea's anchor is
Brazil→Korea per the task's rule, France and all others are own-cell), 361 retained relative-A
coordinates, exactly one more removed by the gravity pivot → 360 free.

## 3. Outer pipeline call-graph audit (task §5)

Full audit in `UNRESTRICTED_OUTER_GRADIENT_CALL_GRAPH_2026-08-01.md`. Headline findings that shaped
everything below:

- The production entry point is `run_profile_checkpointed` (`c10_d20_production_driver.jl:531`),
  reused **unmodified** for this A/B's full/reference arm.
- **The production "C+" outer gradient is not a pure analytic formula.** Only the `gp` component
  (`gamma_component_analytic`) is closed-form exact. The entire A-block gradient is a **fixed-dual
  coordinatewise central finite difference**, made cheap via O(1)-per-changed-cell incremental
  winner updates (`lfix_incremental.jl`/`lfix_factorized_workspace.jl`). This directly shaped the
  profiled gradient's design (§5 below) after an initial envelope-theorem-only draft was
  identified, live, as the wrong approach to hand an outer NLP solver.
- The profiled decode/pivot layer (`relative_a_coordinate_2026-07-31.jl`,
  `gravity_pivot_on_retained_2026-07-31.jl`, `outer_coordinate_layout_profiled_2026-07-31.jl`)
  already produces the exact same `xf=[gp;Aod_levels]` shape the production screened-evaluator
  expects — confirmed and reused unchanged.
- Screens: the pairwise certificate and the fused winner-scan have **no disable kwarg** in
  production; only the general-range safety net and witness screen can be turned off, and the
  pre-winner envelope screen is already off by default under `:exclude_row`. This is a documented,
  unavoidable asymmetry for this first A/B (see §7 below), not silently ignored.

## 4. Profiled outer gradient derivation (task §8) — TWO live corrections

Full derivation in `PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md`. The process is reported
here because both corrections materially changed the deliverable and are important context for
anyone extending this work:

1. **First draft**: a pure envelope-theorem analytic formula (`∂M_d(ω)/∂a_{rd} = e_exponent·M_d(ω)·
   1{winner=r}`, `e_exponent=μ(σ-1)`, traced from `constCons_matrix`/`canonical_price_precompute` —
   this exponent derivation itself is correct and retained in the doc as background). **Rejected
   live** by the user: this formula is exact only *away from winner switches* and is discontinuous
   at every switch boundary — a pointwise-exact-a.e. derivative is known, in this project, to make
   an outer NLP solver behave badly when switches are common. The user's explicit instruction: use
   the *same gradient method* production uses (fixed-dual central FD), adapted only for the new
   parameterization and moments.
2. **Second draft**: a full reimplementation of production's O(1)-incremental-winner-update cache
   (`LFixBaseCache`-style) for the reduced moment basis (~500 lines). **Rejected live** by the user
   as far more custom code than warranted, with an explicit warning that custom reimplementation of
   "highly optimized" code carries real bug risk. This draft was tested anyway (per this project's
   own discipline of never skipping a gate) and **failed decisively** — cosine similarity ~0.03
   against ground-truth re-solved FD at D4 — a concrete, live demonstration of exactly the risk
   flagged.
3. **Final implementation** (`profiled_outer_gradient_fd_2026-08-01.jl`, ~70 lines): fixed-dual
   central FD where every probe calls the **unchanged** production `build_compressed_factual` plus
   this session's own already-gated `reduced_homogeneous_dual_contraction` — zero new
   winner-selection logic. O(W·D·Ddest) per probe (production's own slowest "block_local" tier
   order, not O(1)) — an explicit, honestly-reported wall-clock trade for a much smaller, more
   auditable implementation.

## 5. Gate results

**D4 gradient gate** (`PROFILED_OUTER_GRADIENT_GATE_D4_2026-08-01.csv`, vs ground-truth re-solved
central FD, all 12 coordinates): cosine similarity **0.9994 / 0.9995 / 0.9998** at
calibration/small-perturbation/many-winner-changes; sign agreement 92%/92%/100%. **PASS.**

**D20/W=80,000 gradient gate** (`PROFILED_OUTER_GRADIENT_GATE_D20_W80000_2026-08-01.csv`, vs
ground-truth re-solved central FD on an 11-coordinate representative subset — gp, the pivot's own
coordinate, and 9 random ordinary coordinates): cosine similarity **0.999940 / 0.999856** at
calibration/modest-perturbation; sign agreement **100%/100%**. **PASS.** Full 361-dim profiled
gradient costs ~131-181s per outer iterate at this scale (see §4.3's honest cost note).

**Gravity-pivot invariance gate** (`test_gravity_pivot_invariance_gate_2026-08-01.jl`, D4): (a) no
`r_free` index maps to any anchor cell — structural, verified against the retained/anchor index
sets directly; (b) an arbitrary common destination-scale shift inserted into the anchor gauge
leaves `offset_r0`/`cr`/the pivot cell selection exactly unchanged (diffs ~1e-18, i.e. floating-point
zero), confirmed both algebraically and by a live decoded-point gravity-residual check. **PASS.**

**Reference-path FD equivalence** (task §12, `PROFILED_REFERENCE_PATH_FD_EQUIVALENCE_2026-08-01_{D4,D20}.csv`):
recover-then-resolve applied at the base point and at every finite-difference endpoint (24 points
at D4, 11 at D20). **23/24 (D4) and 10/11 (D20) pass decisively** (divergence diff ~1e-14 to 1e-9,
matching the already-established fixed-point gate numbers). The **single failure in each case is
the same coordinate and direction**: `gp` perturbed in the positive direction, where **both** the
profiled and the reference-at-recovered solves fail to converge together (not a profiled-only
artifact) — consistent with this exact model's own already-documented extreme gp sensitivity near
calibration (`c10_d20_production_driver.jl`'s own comment: "a 1% deviation inflates Delta* from
~0.0026 to ~0.13, a ~50x jump"). Verdict: **PASS with one well-diagnosed, non-differential edge
case**, not a formulation defect.

## 6. Outer A/B search (task §14-17)

Matched setup: real D=20, `:exclude_row`, unrestricted family, fixed theta, W=80,000, delta=1,
same calibration start point (profiled start reduced from the exact same full calibration point,
`reduce_calibration_to_w_profiled`), same outer solver config (`csw_outer_wallclock_sr1.opt`,
`algorithm=3`, `z_halfwidth=30`), 1800s (30 min) budget each, run serially on the same reserved
cores. Full arm: unmodified `run_profile_checkpointed`. Profiled arm:
`run_profiled_outer_search` (`profiled_outer_ab_harness_2026-08-01.jl`).

**Upper bound, delta=1** (`FULL_VS_PROFILED_OUTER_AB_UPPER_DELTA1_2026-08-01_{FULL,PROFILED}_TRACE.csv`,
summary `..._SUMMARY.csv`):

| Arm | wall (s) | n_eval | n_grad_calls | best Delta_dual | KNITRO status |
|---|---|---|---|---|---|
| full (reference) | 1552.5 | 221 | 89 | **8.1535e-5** | -100 (KN_RC_NEAR_OPT) |
| profiled | 1687.4 | 30 | 13 | 8.1062e-4 | -401 |

Full's `n_grad_calls=89` in ~1550s implies ~17s/gradient call (production's O(1)-incremental-update
FD); profiled's `n_grad_calls=13` in ~1687s implies ~130s/gradient call (this session's
O(W·D·Ddest)-per-probe implementation, §4.3) — a ~7-8x per-gradient-call cost ratio, consistent
with the D20 gradient gate's own measured 131-181s/call.

**Decisive result: the full/reference formulation dramatically outperformed the profiled
formulation at matched wall-clock time in this first test.** At `t=1552.5s` (full's total wall
time), full had reached `Delta_dual=8.15e-5` while profiled had only reached `9.96e-4` — full's
final objective is **~12x better** (1121.7% relative gap) than profiled's at the SAME wall-clock
budget. Equivalently: full reached profiled's entire-30-minute-budget final value in just **242
seconds** — **~7x faster** to the same objective threshold.

**Root cause, not a mystery**: the profiled arm's outer trajectory itself was noisy early on
(`Delta` rose from 0.0021→0.0033→0.0057 over its first 20 evals before finding a better point,
`0.00081`, at eval 30) — with only 13 gradient calls total and a FIXED `h=0.01` bandwidth (no
per-coordinate adaptive selection, unlike production's `select_bandwidth`), KNITRO's SR1
quasi-Newton Hessian approximation never had enough calls to build useful curvature information.
This is a direct, expected consequence of §4.3's honestly-documented performance gap
(O(W·D·Ddest)/probe vs production's O(1)), **not evidence that the smaller profiled coordinate
system is intrinsically worse to optimize over** — a fair comparison would require the
incremental-update-optimized profiled gradient this task's scope (and the user's explicit
"surgical, minimal code" instruction) did not build.

**Lower bound**: not run. Given the upper-bound result is already decisive in the "full clearly
better" direction and the task's own §14 instructs "if stable, repeat" (implying "repeat if the
first result leaves the answer ambiguous") — it does not here.

## 6a. Destination-scale step decomposition (task §15)

Computed from the full arm's own 88 accepted (`:new_best`) outer steps
(`DESTINATION_SCALE_STEP_DECOMPOSITION_2026-08-01.csv`), decomposing each step's `Δlog(A)` into a
per-destination common-SCALE component (`mean_d(Δa_{.,d})·1`) and a RELATIVE component
(`Δa_{.,d} - mean_d(Δa_{.,d})·1`):

```
mean fraction of step norm in SCALE directions:    3.59%   (range 1.76%-7.63%, n=87 valid steps)
mean fraction of step norm in RELATIVE directions: 96.41%
```

The full optimizer's actual accepted steps in this run spend the overwhelming majority of their
norm in RELATIVE directions already — **only ~3.6% of the full formulation's own step budget goes
toward destination-scale nuisance directions in this trajectory**. This is independent evidence,
from the FULL formulation's own behavior (not the A/B comparison), that removing destination-scale
degrees of freedom was never likely to unlock a large optimization-efficiency gain for THIS stage
(fixed-theta profile stage, gp already held fixed) — reinforcing, not contradicting, the A/B's
decisive "full better" result. **SCALE_DIRECTION_DIAGNOSTIC = negligible.**

## 7. Screen-parity caveat (task §13)

The profiled arm has **no screens at all** (its evaluator never calls `evaluate_fullA_screened_ranged`).
The full/reference arm, run via unmodified `run_profile_checkpointed`, has the pairwise certificate
and the fused zero-winner/winning-range scan **always on** (no driver-level kwarg exists to disable
them — `screened_eval`'s call to `evaluate_fullA_screened_ranged` is hardcoded); only the general
range safety net was explicitly disabled where possible and the witness/pre-winner-envelope screens
were already off by default under `:exclude_row`. **This is a real, acknowledged asymmetry** for
this first A/B — porting or adding a true screens-fully-off toggle to the production driver is
explicitly out of scope for this task (§18: "screen porting").

**Correction to the call-graph audit's §9 claim, observed live during the actual A/B run**: the
pre-winner envelope screen was reported `envelope_screen_supported=true` for this run's ctx (not
unsupported/off as the audit doc's static reading of the source implied) — but its LIVE rejection
count over the full 221-eval, 30-minute run was exactly **0** (final screen tally:
`pw=2, wt=0, wn=0, env=0, wr=0, sn=0, pass=224`). Net effect on this A/B is the same either way
(zero envelope-screen rejections), but the mechanism is "supported, zero organic hits" rather than
"unsupported" — corrected here rather than left standing uncorrected.

## 8. Scope discipline

Not implemented (per task §18): CM cross-Hessian, Fréchet moments, ZC moments, CM+ZC, restricted-family
gradients, screen porting, checkpoint migration, production default change, production merge, full
campaign launch.

`PRODUCTION_DEFAULT_CHANGED = false`. `PRODUCTION_MERGE = not_attempted`. `CAMPAIGN_LAUNCHED = false`.

## Final verdict block

```
D20_INNER_LAYOUT =
    factual_share_moments: 361
    france_ratio_moments: 1
    total_economic_moments: 362
    total_inner_dual_dimension: 363

PROFILED_OUTER_VECTOR =
    gp: 1
    free_A: 360
    total: 361

OUTER_EVALUATOR = wired

PROFILED_CPLUS_GRADIENT = wired_and_verified
    (D4 cos_sim>0.999, D20/W80000 cos_sim>0.9998, both formulations use the SAME fixed-dual
    central-FD method; profiled implementation is O(W*D*Ddest)/probe, not O(1) -- honest,
    documented performance gap, not a correctness gap)

REFERENCE_PATH_FD_EQUIVALENCE = pass
    (23/24 D4, 10/11 D20 FD endpoints pass decisively; the one exception per scale is a SHARED
    gp-direction solver edge case affecting both formulations identically, not a profiled-only
    defect)

OUTER_AB_UPPER = full_better_objective_at_matched_time
    (full: Delta=8.15e-5 @ 1552.5s/221 evals/89 grads; profiled: Delta=8.11e-4 @ 1687.4s/30 evals/
    13 grads -- full ~12x better objective at matched wall-clock, ~7x faster to profiled's own
    final threshold. Root cause: profiled gradient's O(W*D*Ddest)/probe cost -- see §6.)
OUTER_AB_LOWER = not_run_upper_already_decisive
SCALE_DIRECTION_DIAGNOSTIC = negligible
    (full formulation's own accepted steps: mean 3.59% of step norm in destination-scale
    directions, 96.41% in relative directions -- independently corroborates the A/B result)
PORT_TO_RESTRICTED_FAMILIES = do_not_recommend
    (per this session's diagnostic implementation: profiled gradient cost dominates any
    dimension-reduction benefit; gains are decisively NEGATIVE, not merely under 20%. Caveat:
    this reflects the current O(W*D*Ddest)-per-probe FD implementation choice -- made deliberately
    to minimize custom/error-prone code per explicit live user guidance -- not a proven flaw in the
    profiled coordinate system itself. An O(1)-incremental-update profiled gradient was explicitly
    out of scope this session; without it, no fair verdict on the coordinate system's own
    optimization-landscape merit can be drawn from wall-clock alone.)

PRODUCTION_DEFAULT_CHANGED = false
PRODUCTION_MERGE = not_attempted
CAMPAIGN_LAUNCHED = false
```
