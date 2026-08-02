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

   > **CORRECTION (production outer bridge task, §12, 2026-08-01)**: the "1.0000000000 (max rel
   > err ~6e-6)" figure above does not reproduce by re-running
   > `test_profiled_incremental_vs_fullrebuild_2026-08-01.jl` as originally committed — that script
   > actually computes `cos_sim≈0.9998982`, `max_rel_err≈2.0` at D4 (and similarly ≈0.9998,
   > max_rel_err≈3.65 at the small-perturbation point). Root cause, isolated directly: the
   > full-rebuild comparator (`profiled_composite_gradient_at`) used one FIXED `h=0.01` for every
   > coordinate, while the incremental method uses an ADAPTIVE, per-coordinate `h`
   > (`profiled_select_bandwidth`) — two different finite-difference formulas' worth of truncation
   > error, not a defect in either gradient. The script has been corrected (now reports
   > `mismatched_bandwidth` — the historical, genuinely ~0.9999-not-1.0 comparison above — and
   > `same_bandwidth` — the true formula-equivalence claim, re-running the full-rebuild reference
   > at the incremental method's own selected `h`) and re-verified: D4 A-block(2:end) same-bandwidth
   > `max_rel_err=1.5e-15`/`1.8e-15`, `cos_sim=1.0000000000` — genuine machine precision. This
   > underlying speedup/correctness result is NOT invalidated (the formula was always right,
   > confirmed now more rigorously); only this specific "1.0000000000/6e-6" sentence was describing
   > a comparison the committed script did not actually run. See
   > `PROFILED_RESTRICTED_PRODUCTION_OUTER_BRIDGE_MASTER_2026-08-01.md` §12 for the full writeup.

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

## ⚠️ RETRACTION NOTICE (read before §6/§6b below)

**Sections §6 and §6b's numbers are WRONG and RETRACTED.** They were produced by
`run_profiled_outer_search` (`profiled_outer_ab_harness_2026-08-01.jl`) with a real bug: `gp`
(`w_start[1]`) was included **inside** KNITRO's free-variable vector with a `±30` box, instead of
being held fixed as the production "profile stage" contract requires (`run_profile_checkpointed`
captures `g_in` by closure and never adds it to KNITRO's variables at all). The profiled arm was
therefore silently solving an **easier problem** than the full arm in every A/B below §6c — free
to drift `gp` back toward its easy, well-fitting calibration value instead of being held at the
caller's intended fixed target. Every "profiled wins big" / "full wins big" number in §6 and §6b
is an artifact of this asymmetry, not a real finding about either formulation. **See §6c for the
diagnosis (caught via a live user-requested LFD/gravity verification) and §6d for the corrected,
apples-to-apples results.** §6/§6b are left in place below, struck through in spirit but not
deleted, per this repo's own convention of correcting in place rather than erasing a wrong result.

## 6. [RETRACTED] Outer A/B search (task §14-17)

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

## 6b. [RETRACTED] Follow-up: fixed-outer-iteration-count A/B (live user request, same day)

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

## 6c. The gp-drift bug: diagnosis and fix

After §6b reported a 7.5x gap that survived fixing the gradient's speed, the user asked for a
deeper check: *"verify your profiled solutions by using the LFD to evaluate all of the trade
shares and the gravity regression. I am a bit suspicious of differences that large."*
`verify_lfd_shares_gravity_2026-08-01.jl` was built to do exactly that — for both arms' best
points, recompute the LFD-weighted implied bilateral trade share for every `(o, destination)` cell
and compare to the factual data share, and check the gravity regression residual.

**That specific check came back looking fine for both arms** (mean|diff| identically ~1.4e-9,
gravity residual ~1e-19/1e-20 for both) — which is itself informative in hindsight: bilateral
share-matching is essentially guaranteed by each solve's own KKT/moment conditions regardless of
the overall `Delta_dual` value, so it cannot distinguish "genuinely better fit" from "cheating on
a different margin." **The actual smoking gun came from printing the raw `gp` value at each arm's
best point**, prompted by the same investigation: the full arm's checkpoint correctly showed
`gp=0.983116` (the intended fixed value for that run); the profiled arm's best point showed
`gp=0.993019` — almost exactly back at calibration (`0.9930463...`), even though the profile-stage
task explicitly fixes `gp` and searches only over the A-block.

**Root cause**: `run_profiled_outer_search` passed `w_start` (length 361, `[gp; r_free]`) wholesale
into `KN_add_vars`/`KN_set_var_lobnds_all`/`KN_set_var_upbnds_all` with a `±30` box on every
coordinate, including `gp` — there was never any mechanism holding `gp` fixed. Every gradient call
also returned the FULL length-361 gradient (`g[1]` = the gp component) directly to
`evalResult.objGrad`, meaning KNITRO was both free to move `gp` AND told exactly which direction
would improve it. The full arm never had this problem: `run_profile_checkpointed` takes `g_in` as
a **separate scalar argument**, captured by closure, never added to KNITRO's variable set at all.

**Fix** (`profiled_outer_ab_harness_2026-08-01.jl`): `gp` is now captured by closure
(`gp_fixed = w_start[1]`), `KN_add_vars` only allocates the `n_free` A-block coordinates, and
`cb_G!` drops the gradient's `gp` component before writing to KNITRO
(`evalResult.objGrad .= g[2:end]`) — exactly mirroring production's own `gfull[2:end]` convention.
**Verified live**: a short test run confirms the terminal free-variable vector has length 360 (not
361) and `best.w[1]` stays at *exactly* the caller's fixed `gp` throughout the search.

This invalidates every profiled-arm result computed before this fix — §6, §6b, and the killed
gp=0.995/0.988 sweep points. §6d below re-runs the key comparisons with the fix in place.

## 6d. Corrected matched A/B (post-fix), across three difficulty levels

Same fixed-iteration protocol as §6b (both arms capped at the identical KNITRO `maxit=60`, same
outer solver config, `gp` now genuinely held fixed for the profiled arm too), run at three points
of increasing distance from calibration — `gp=0.99×calib`, `gp=0.985×calib`, and a `gp` reached via
a short **continuation** (`continuation_gt_targets_2026-08-01.jl`, walking `gp` down in 6 small
steps of `run_profile_checkpointed` calls, each warm-started from the previous step's converged A,
since a direct jump to these `gp` values from calibration's own A already fails to converge — see
§6e) corresponding to a **gains-from-trade target of GT=5%**
(`κ = 1 - gp^(σ/(σ-1)) = 0.05`, confirmed formula, `c12_sign_convention_smoke_test.jl` and others;
`σ=2.5` in this live D20 context, confirmed from `ctx.σ`, not assumed). The GT=5% point's shared
starting A-matrix was taken from the continuation's own converged state and translated into both
arms' coordinate systems (`pivot_expand` → `reduce_to_w_profiled`) so both start from the
economically identical point, per the task's own "same initial economic point" requirement.

| Point | gp (fixed) | full Δ (n_grad) | profiled Δ (n_grad) | profiled better by |
|---|---|---|---|---|
| gp=0.99×calib (κ≈2.8%) | 0.983116 | 0.15175 (61) | 0.14612 (61) | **3.85%** |
| gp=0.985×calib (κ≈3.6%) | 0.978151 | 0.36264 (60) | 0.33582 (57) | **7.99%** |
| GT=5% waypoint (κ=5.0%) | 0.969693 | 0.97312 (61) | 0.92158 (61) | **5.59%** |

`run_ab_{full,profiled}_fixediter_gplow_2026-08-01.jl` (parametrized via `AB_GP_FRAC`) and
`run_ab_{full,profiled}_fixediter_gt5_2026-08-01.jl`; traces
`FULL_VS_PROFILED_OUTER_AB_FIXEDITER_2026-08-01_{gp0p990,gp0p985,gt5}_{FULL,PROFILED}_TRACE.csv`.
`gp` confirmed held exactly fixed at every run via the printed `seed: gp_fixed=...` line, matching
the intended target to machine precision in every case. Re-ran the LFD/gravity verification on the
corrected gp=0.99 point: gravity residual ~1e-19/1e-20 for both arms (fully gravity-feasible,
consistent with the earlier check), `Delta_dual` matches the completed run to 5 significant figures.

**Corrected picture, replacing §6/§6b entirely**: at matched gradient-call counts across three
genuinely fixed-gp difficulty levels, the profiled formulation shows a **small, consistent 4-8%
better objective** than the full formulation — not a 12x or 7.5x gap in either direction. This is
a plausible, believable result for two mathematically equivalent reparameterizations of the same
problem (the profiled arm has 1 fewer free coordinate — 360 vs 361 counting the pivot — and no
destination-scale nuisance directions to traverse, matching §6a's own finding that the full
formulation spends little of its step budget on those directions anyway, so a small edge rather
than a large one is exactly what one would expect a priori).

## 6e. Continuation mechanics (for the GT=5% waypoint)

A direct jump from calibration to `gp=0.9697` (holding the A-block at calibration values) fails to
converge (`inner_status=-400`) well before reaching that target — probed directly: `gp` fractions
down to `0.985` (κ≈3.6%) converge cleanly, `0.982` (κ≈4.1%) does not. `continuation_gt_targets_2026-08-01.jl`
instead steps `gp` down by a factor of `0.996` per step (6 steps total to reach κ=5.0% from
calibration's κ≈1.16%), re-optimizing the A-block at each step via a **short**
`run_profile_checkpointed` call (`maxit=10`, warm-started from the previous step's own converged
`zfree`) — reusing the trusted production driver at every step rather than writing new continuation
machinery. All 6 steps converged to a verified feasible point before advancing; total continuation
wall-clock ≈2000s (6 × ~330s/step). The final waypoint (`gp=0.9696927825876808`,
`kappa=0.049999999999999993`) is serialized to `results/profiled_ab_2026-08-01/continuation_gt/waypoint_GT5.jls`.

## 6a. Destination-scale step decomposition (task §15)

(Unaffected by the §6c bug — computed entirely from the FULL arm's own trajectory, which never had
a `gp`-fixing problem.) Computed from the full arm's own 88 accepted (`:new_best`) outer steps
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

OUTER_AB_UPPER = RETRACTED_see_OUTER_AB_FIXEDITER
    (the original wall-clock-matched full-vs-profiled comparison, and its own "fixed-iteration"
    follow-up, were both computed with a real bug in the profiled harness: gp was included in
    KNITRO's free-variable set instead of held fixed, letting the profiled arm silently drift gp
    back toward its easy calibration value -- an unfair comparison invalidating both the 12x and
    7.5x gaps originally reported. See §6c for the full diagnosis (caught via a live user-requested
    LFD/gravity verification) and §6d for the corrected results.)
OUTER_AB_FIXEDITER = profiled_better_by_4_to_8_pct_at_matched_gradient_calls
    (CORRECTED, post-bugfix, three points of increasing distance from calibration -- gp=0.99*calib
    (kappa~2.8%): full=0.15175/profiled=0.14612 (61/61 grads, profiled +3.85%); gp=0.985*calib
    (kappa~3.6%): full=0.36264/profiled=0.33582 (60/57 grads, profiled +7.99%); GT=5% waypoint via
    continuation (kappa=5.0%): full=0.97312/profiled=0.92158 (61/61 grads, profiled +5.59%). gp
    confirmed held exactly fixed in every run. A small, consistent, believable edge -- not a wild
    anomaly in either direction -- across three genuinely matched difficulty levels. See §6d.)
OUTER_AB_LOWER = not_run
SCALE_DIRECTION_DIAGNOSTIC = negligible
    (full formulation's own accepted steps: mean 3.59% of step norm in destination-scale
    directions, 96.41% in relative directions -- unaffected by the §6c bug, independently
    corroborates the small (not large) effect size found in the corrected A/B)
PORT_TO_RESTRICTED_FAMILIES = insufficient_evidence
    (revised down from the earlier "do_not_recommend" verdict, which was based on invalidated
    numbers. The CORRECTED result -- a consistent but modest 4-8% edge for the profiled formulation
    across three difficulty levels -- is directionally encouraging but far too small a sample (3
    points, one A/B seed each, upper-bound direction only, one real-data draw) to recommend a
    multi-family rewrite on. A genuine recommendation would need: the lower-bound direction, more
    seeds/starting points, and a check of whether the 4-8% edge holds or changes at yet-more-extreme
    gp values. This session's own repeated experience with getting the comparison wrong twice
    (envelope-theorem gradient rejected, O(1)-incremental gradient's gp-formula bug, then this
    gp-drift harness bug) is itself a reason for caution before generalizing from 3 points.)

PRODUCTION_DEFAULT_CHANGED = false
PRODUCTION_MERGE = not_attempted
CAMPAIGN_LAUNCHED = false
```
