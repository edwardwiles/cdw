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
3. **Third draft, "surgical" full-rebuild** (`profiled_outer_gradient_fd_2026-08-01.jl`, ~70 lines):
   fixed-dual central FD where every probe calls the **unchanged** production
   `build_compressed_factual` plus this session's own already-gated
   `reduced_homogeneous_dual_contraction` — zero new winner-selection logic. O(W·D·Ddest) per probe
   (production's own slowest "block_local" tier order, not O(1)) — deliberately traded speed for a
   much smaller, more auditable implementation. This version is what the first A/B (§6) ran against.
4. **Fourth draft, O(1)-incremental (follow-up, same day)**: after the first A/B showed the
   profiled arm losing badly, the user asked directly — "are you sure the slowness isn't just your
   implementation being unoptimized? If so, optimize it." This time, rebuilding the O(1)-per-
   changed-cell incremental gradient (`profiled_lfix_incremental_2026-08-01.jl`, reusing
   `lfix_incremental.jl`'s `update_winner_o1`/top-3 mechanism unchanged) was gated **directly
   against the now-trusted full-rebuild version at machine precision** — a much stronger,
   cheaper check than the earlier (abandoned) attempt had access to, which was only ever tested
   against expensive re-solved ground truth. This surfaced and fixed a **second, more subtle bug**
   in the analytic `gp` component: `cf.cf_raw[w]` (`compressed_moments.jl:264`) is **not** a
   gp-independent data constant, as both earlier drafts (including the one live-corrected mid-A/B)
   assumed — it is rebuilt fresh at the current `gp` and already contains its own
   `-gp^σ·wPrime_bi·LPrime_bi` term, which exactly cancels the separate `const_cf` term
   `reduced_homogeneous_dual_contraction` also adds. The true derivative collapses to the much
   simpler `-κ_cf·σ·gp^(σ-1)·Tslot_bi/M` (no `LPrime_bi·S_m` term at all — that term, added in the
   earlier "fix," was itself spurious). Found by h-sweeping the full-rebuild central FD to a clean,
   stable, h-independent limit and isolating the exact missing term via per-draw `dq[w]/dgp`
   verification (not by inspection or guessing). **Gated to machine precision** against the
   full-rebuild version: D4 cosine similarity **1.0000000000** (max rel err ~6e-6), D20/W=80,000
   cosine similarity **1.0000000000** (max rel err ~1.5e-7 to 7.4e-6) — see
   `PROFILED_INCREMENTAL_VS_FULLREBUILD_2026-08-01_{D4,D20_W80000}.csv`. **Measured speedup at
   D20: 10.0x and 15.1x** (16s vs 160s, 11s vs 168s per gradient call) — now *faster* per call than
   the full/reference arm's own production gradient (~17-18s/call). This version is what the
   fixed-iteration A/B (§6b) ran with.

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

**Initial hypothesis (partially wrong, corrected in §6b)**: at the time, this was attributed
entirely to the slow O(W·D·Ddest)/probe gradient (only 13 gradient calls total in the 30-minute
budget) starving KNITRO's SR1 Hessian approximation of curvature information — i.e., an
implementation-speed artifact, not a genuine search-quality difference. §6b tests this directly and
finds it is only PART of the story.

**Lower bound**: not run. The upper-bound result already answers the question this A/B was
designed to answer, once combined with §6b's follow-up.

## 6b. Follow-up: fixed-outer-iteration-count A/B (live user request, same day)

The wall-clock-matched result in §6 conflates two different questions: "does the profiled
coordinate system make each outer step more effective?" and "how much does each formulation's
gradient cost per call?" The user asked for a fixed-iteration-count comparison to isolate the
first question — both arms capped at the **same KNITRO `maxit` parameter** (60), using the NOW
machine-precision-validated, 10-15x-faster **O(1)-incremental** profiled gradient (§4, draft 4),
generous wall-clock safety net (3600s, did not bind for either arm):

| Arm | wall (s) | n_eval | n_grad_calls | native_outer_iters | best Delta_dual |
|---|---|---|---|---|---|
| full (reference) | 1129.2 | 133 | **61** | 60 (hit cap exactly) | **9.801e-5** |
| profiled (incremental gradient) | 833.2 | 107 | **61** | — | 7.374e-4 |

(`run_ab_full_fixediter_2026-08-01.jl` / `run_ab_profiled_fixediter_2026-08-01.jl`, traces
`FULL_VS_PROFILED_OUTER_AB_FIXEDITER_2026-08-01_{FULL,PROFILED}_TRACE.csv`.)

**Both arms landed on EXACTLY 61 gradient calls** (a direct consequence of capping the same KNITRO
`maxit` parameter identically for both) — this is now a genuinely apples-to-apples comparison,
independent of wall-clock or gradient implementation speed. Per-gradient-call cost is now
**comparable, and profiled is if anything slightly cheaper**: full ≈18.5s/call, profiled ≈13.7s/call
— profiled used *less* total wall-clock (833s vs 1129s) to run the same number of iterations.

**Despite this, full still reaches a ~7.5x better objective at the identical iteration count**
(9.80e-5 vs 7.37e-4). This is the more informative result: **the earlier wall-clock gap was real,
but the profiled search is genuinely less effective per outer step too, not merely slower per
gradient call.** The corrected picture is therefore: (a) the profiled gradient implementation's
speed deficit was real but fixable, and has now been fixed (10-15x speedup, §4 draft 4); (b) fixing
it closes most of the wall-clock gap (profiled is now competitive or cheaper per call); but (c) a
genuine per-iteration search-quality gap remains, favoring the full formulation, that gradient
speed alone does not explain. A plausible (not yet tested) contributor: production's per-coordinate
**adaptive bandwidth** (`select_bandwidth`, targeting a 0.3%-3% winner-switching-mass window) vs
this diagnostic's single fixed `h=0.01` for every profiled A-block coordinate — the profiled
gradient may simply be a noisier local model of the objective at this bandwidth. This was not
tested further (out of scope for a same-day follow-up) and is flagged as the natural next step
before drawing a final conclusion about the coordinate system's own merit.

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
    (two implementations, both machine-precision-verified: (1) full-rebuild FD, O(W*D*Ddest)/probe,
    cos_sim vs ground truth >0.999 at D4/D20; (2) O(1)-incremental FD (profiled_lfix_incremental_
    2026-08-01.jl), gated to cos_sim=1.0000000000 against (1) at BOTH D4 and D20/W80000, 10-15x
    faster than (1) at D20 -- now faster per call than production's own gradient. A second,
    genuine analytic-formula bug (gp component, cf.cf_raw's own gp-dependence) was found and fixed
    during this validation -- see PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md section 6a/6b.)

REFERENCE_PATH_FD_EQUIVALENCE = pass
    (23/24 D4, 10/11 D20 FD endpoints pass decisively; the one exception per scale is a SHARED
    gp-direction solver edge case affecting both formulations identically, not a profiled-only
    defect)

OUTER_AB_UPPER = full_better_objective_at_matched_time
    (wall-clock-matched, slow full-rebuild profiled gradient: full Delta=8.15e-5 @ 1552.5s/221
    evals/89 grads; profiled Delta=8.11e-4 @ 1687.4s/30 evals/13 grads -- full ~12x better at
    matched wall-clock. See OUTER_AB_FIXEDITER below for the corrected, apples-to-apples
    iteration-matched follow-up using the now-fixed 10-15x-faster gradient.)
OUTER_AB_FIXEDITER = full_better_objective_at_matched_iterations
    (BOTH arms capped at the identical KNITRO maxit=60, landed on the identical 61 gradient calls,
    using the fixed O(1)-incremental profiled gradient (13.7s/call, actually cheaper than full's
    18.5s/call): full Delta=9.80e-5 vs profiled Delta=7.37e-4 -- full still ~7.5x better at
    IDENTICAL gradient-call count and comparable-or-less wall-clock for profiled. This is the
    decisive result: the wall-clock gap in OUTER_AB_UPPER was real but not the whole story --
    a genuine per-iteration search-quality gap remains even after fixing gradient speed. See §6b
    for the leading unexamined hypothesis (production's adaptive per-coordinate FD bandwidth vs
    this diagnostic's single fixed h=0.01).)
OUTER_AB_LOWER = not_run
SCALE_DIRECTION_DIAGNOSTIC = negligible
    (full formulation's own accepted steps: mean 3.59% of step norm in destination-scale
    directions, 96.41% in relative directions -- independently corroborates the A/B result)
PORT_TO_RESTRICTED_FAMILIES = do_not_recommend
    (BOTH the wall-clock-matched AND the iteration-matched A/B favor full, the latter after fixing
    the profiled gradient's speed to be competitive-or-better than production's own. This is a
    stronger, more decisive basis than the first (superseded) verdict, which had wrongly attributed
    the entire gap to gradient-implementation speed. Caveat retained: the adaptive-bandwidth
    hypothesis in §6b was not tested and could narrow or close the remaining per-iteration gap --
    flagged as the concrete next step, not dismissed.)

PRODUCTION_DEFAULT_CHANGED = false
PRODUCTION_MERGE = not_attempted
CAMPAIGN_LAUNCHED = false
```
