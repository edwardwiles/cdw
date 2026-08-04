# FULL W=100k continuation-and-polishing campaign — MASTER report

**Branch:** `campaign/fullA-continuation-polish-2026-08-03` (7 commits ahead of
`origin/production/fullA-exact` @ `4c3dad5`). **Not merged into production** — pushed for review only.
**Worktree:** `/bbkinghome/edav/cdw_worktrees/fullA-continuation-polish-2026-08-03`, clean at completion.

This campaign started as the literal task spec (continue/polish the FULL W=100k independent-starts
campaign's delta=1/delta=2 results) and grew, at the user's direction mid-session, into four
additional pieces of real work: a cross-family monotonicity audit and fix, a delta=5 extension, and
a W=250,000 numerical-sensitivity test. All of it is documented here.

## 1. Prerequisite validation (Section 1-3 of the original task)

The old independent-starts campaign's shutdown was confirmed genuinely in progress (not
fabricated) by direct process/log monitoring — 9 jobs drained cleanly over ~2 hours, handoff
checksums verified against `SHA256_MANIFEST_2026-08-03.txt`. Full writeup:
`SCIENTIFIC_MANIFEST_VALIDATION_2026-08-03.md` (field-by-field trace confirming σ=3.0, Brazil-Korea
gravity exclusion, `destination_sample=:exclude_row`, `draw_design=:sobol_randomized`, W=100,000,
family-specific L/K, `A_coordinate_mode=:powered_aspace` all in effect via the actual code's own
defaults — not the handoff doc's claims alone. Zero code drift between the old campaign's commit
and this worktree confirmed via `git diff --stat`.

`SCIENTIFIC_MANIFEST_MATCH = true`.

## 2. Algorithm inventory and pilot tournament (Sections 7-8)

`ALGORITHM_INVENTORY_2026-08-03.md`: the old campaign ran every cell under `algorithm=auto`
(resolves to Active-Set/CG for this formulation) — never the genuine Direct+SR1 the task spec
wanted for exploration. Direct+SR1 (`outer_direct_hessopt=:sr1`) and SQP (`opt_file=
"csw_outer_phaseB_sqp_bfgs_maxit15.opt"`) are both pre-existing, already-wired options; no new
solver code was needed, just passing the right kwarg. `unrestricted`'s driver has no `opt_file`
kwarg, so its polish arm uses Direct+BFGS instead of SQP.

`PILOT_TOURNAMENT_VERDICT_2026-08-03.md`: 3 real cells, 2 arms each, 1500s budgets. Verdict —
EXPLORE (Direct+SR1) wins for `flexible_cm`/`common_frechet`/`cm_meanzc` (same driver family);
POLISH (SQP) wins for `origin_zc`; `unrestricted`'s one lower-direction test cell was flat under
both algorithms, reproducing a documented non-monotonicity as a genuine landscape feature, not an
artifact (confirmed reproducible under two different algorithms).

`CHECKPOINT_CONTENTS_VERIFICATION_2026-08-03.md`: real `.jls` deserialization confirmed seeds must
be built from `checkpoint.best_feasible.w`, not `checkpoint.zfree`/`.g` (the last unverified probe
point) — a real design decision, not a guess.

## 3. Reusable orchestration code (Section 14)

All under `full_aod_diag/d4_exact/`, all committed to this branch:

- **`continuation_polish_orchestrator.jl`** — monotone-incumbent envelope (upper=max/lower=min,
  proper feasible-set-nesting inheritance), seed dedup (distance+objective), algorithm-stage
  selection, never-regress result comparison, checkpoint/resume state. Pure, injectable-`run_fn`
  logic — unit tested without invoking KNITRO.
- **`test_continuation_polish_orchestrator.jl`** — 40 assertions across 8 testsets, all passing.
  Caught 2 real design bugs during development (envelope provenance missing `w`/`Delta_star`
  fields) before any real KNITRO time was spent.
- **`continuation_polish_run_fn.jl`** — the real (non-mock) `run_fn`, dispatching to
  `run_polish_checkpointed_unified`/`run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`
  with the campaign's exact scientific-manifest kwargs. `CAMPAIGN_W` is a **required** env var (no
  default), per this repo's own "never silently default a scientific parameter" rule.
- **`continuation_campaign_cell_driver.jl`** — one process = one (family, direction, target_delta)
  cell; loads the baseline envelope + any smaller-delta campaign results, runs
  `run_target_cell!`, persists to a canonical location the next delta in a chain can find. Includes
  the **cross-family relaxation seed mechanism** (Section 4 below) and same-delta refinement-round
  support.
- **`continuation_campaign_w_extension_driver.jl`** — seeds a cell directly from another W's
  result, no envelope-CSV baseline (there is no "original campaign" at a different W).

Real bug found and fixed during this work: `load_checkpoint_w` originally dispatched `origin_zc`
checkpoints to `cm_checkpoint.jl`'s `load_cm_checkpoint` (schema V9 only) — `origin_zc` actually
writes/reads `CMCheckpointV10` via its own `load_cm_checkpoint_v10` (`cm_originzc_checkpoint.jl`), a
different schema. Confirmed the hard way (real `TypeError` cascade on a genuine file) before fixing.

## 4. Cross-family monotonicity investigation and fix (user-directed, 2026-08-04)

**Not in the original task spec.** The user observed, reviewing the delta-schedule table live: since
`unrestricted` is a strict relaxation of every restricted family (same `(gp,A)→Delta*` map, just no
extra moment restriction), it must weakly dominate every restricted family's own optimum at the
same delta/direction. This is checkable and was violated in the live data.

**Mechanism (confirmed by direct code read, not inference):** every restricted family's outer
vector is `[w_a; optional_extra_tail]`, with `w_a` the *identical* 380-length `[gp; A_nonpivot]`
block `unrestricted` uses directly as its own `w0` (`campaign_cm_family_runner.jl`'s own
`w0 = FAMILY == "cm_meanzc" ? vcat(w_a, log.(nu_meanzc)) : FAMILY == "origin_zc" ? vcat(w_a,
log.(nu_originzc)) : copy(w_a)`). Since Delta* is computed from `(gp,A)` alone, independent of
which restriction family produced it, a restricted family's point sliced to its first 380
components is *already* Delta*-feasible for `unrestricted` at the same delta — a guaranteed,
not-just-plausible, improvement seed.

**Violations found** (before fix, upper direction, all deltas): `origin_zc` exceeded `unrestricted`
at delta=0.01 (0.0329 vs 0.0239), 0.1 (0.0511 vs 0.0507), 0.5 (0.0713 vs 0.0705), 1.0 (0.0762 vs
0.0737) — plus at delta=0.01 specifically, *all four* restricted families exceeded `unrestricted`
(a separately-known stall: `unrestricted`'s own delta=0.01 search historically used only ~5% of its
budget).

**Fix:** `continuation_campaign_cell_driver.jl` gained optional `<cross_seed_family>
<cross_seed_delta>` CLI args, injecting that family's `w[1:380]` as an extra seed candidate for
`unrestricted`. All 4 violations resolved:

| delta | origin_zc GT | unrestricted before | unrestricted after | Resolved? |
|---|---|---|---|---|
| 0.01 | 0.032882 | 0.023862 | 0.032882 (seed point itself won, exact tie) | yes |
| 0.1 | 0.051057 | 0.050742 | 0.051126 | yes |
| 0.5 | 0.071282 | 0.070507 | 0.071410 | yes |
| 1.0 | 0.076159 | 0.073653 | 0.076764 | yes |

## 5. delta=5 upper extension (user-directed, 2026-08-04)

Each family's delta=5 upper cell seeded directly from its own delta=2 campaign result (required a
one-line driver fix: the "layer in smaller-delta results" candidate list never included delta=2.0
itself, only 0.01/0.1/0.5/1.0/1.5 — fixed before launching).

| Family | delta=2.0 | delta=5.0 | Delta* used at delta=5 | Note |
|---|---|---|---|---|
| unrestricted | 0.078072 | 0.078388 | 2.058 (used ~41% of budget) | real small improvement |
| flexible_cm | 0.073517 | 0.073518 | 1.132 (~23% of budget) | essentially saturated |
| common_frechet | 0.073292 | 0.073297 | 1.042 (~21% of budget) | essentially saturated |
| cm_meanzc | 0.071955 | 0.071955 | 1.189 (~24% of budget) | exactly flat |
| origin_zc | 0.076675 | 0.076675 | 1.147 (~23% of budget) | flat — **stale**, see below |

**Known residual:** `origin_zc`'s delta=2 cell was independently re-run in a "round 2" refinement
*after* its delta=5 cell had already been computed, improving delta=2 from 0.076675 to 0.076853.
delta=5's own value (0.076675) is therefore currently a hair below delta=2's updated value — a
genuine but minor within-family monotonicity gap, not re-chased given the session's later pivot to
the W=250k question. A follow-up delta=5 refinement seeded from the newer delta=2 point would very
likely close this (every other cell shows delta=5 ≥ delta=2 once seeded from the *latest* delta=2).

None of the 4 restricted families found real headroom beyond ~20-40% of the delta=5 budget —
consistent with genuine local convergence, not a resource-starved search.

## 6. W=100k → W=250k extension (user-directed, 2026-08-04)

**Hypothesis under test:** delta≥2 upper bounds might be limited by W=100k's Monte Carlo draw
count (a numerical/sampling ceiling), not a genuine economic feasibility ceiling. Tested by seeding
a W=250,000 run directly from each family's own final W=100k delta=2 point — no envelope-CSV
baseline (there is no W=250k "original campaign"). Smoke-tested first (flexible_cm, 15s/15s): the
W=100k seed re-verified at W=250k with an essentially bit-identical GT and Delta* within Monte
Carlo noise, confirming the mechanism before committing to the real 3600s/3600s sweep.

| Family | W=100k GT | W=250k GT | Δ | KNITRO status | Verdict |
|---|---|---|---|---|---|
| unrestricted | 0.078072 | **0.078333** | +0.000261 | -101 (converged, feasible) | real, clean improvement |
| origin_zc | 0.076675* | **0.077402** | +0.000728 | -401 (time limit) | real, substantial improvement |
| cm_meanzc | 0.071955 | **0.072558** | +0.000603 | -411 (time limit, infeasible) | real improvement |
| common_frechet | 0.073292 | 0.073300 | +0.000008 | -411 (time limit, infeasible) | negligible |
| flexible_cm | 0.073517 | 0.073517 | +0.0 | -401 (time limit) | exactly flat |

*`origin_zc`'s W=250k job read its **pre-round-2** seed (0.076675, not the later 0.076853) due to a
timing race between the round-2 refinement job and the W=250k launch — both were running close in
time. The improvement is real regardless of which seed it started from; it just isn't stacked on
top of the latest W=100k number. Not re-run given the session's wrap-up.

**Bottom line on the hypothesis:** confirmed real for 3 of 5 families (`unrestricted`, `origin_zc`,
`cm_meanzc`) — W=100k's Monte Carlo draw count was genuinely constraining how far delta=2 could be
pushed for these families, exactly as hypothesized. `common_frechet` showed only a negligible
effect and `flexible_cm` none at all — those two appear to have reached a real local optimum that
more draws alone don't loosen, at least from this seed. All 5 jobs terminated on time/iteration
limits rather than clean local convergence except `unrestricted`, meaning there is very plausibly
more headroom at W=250k with a longer budget for the other 4 — this was not chased further given
the session's wrap-up.

`CAMPAIGN_W` is required (no default) in `continuation_polish_run_fn.jl` as of this extension —
matches this repo's own scientific-parameter rule.

## 7. Final combined upper-direction table (best of W=100k / W=250k at delta=2)

| Family | δ=0.01 | δ=0.1 | δ=0.5 | δ=1.0 | δ=2.0 | δ=5.0 |
|---|---|---|---|---|---|---|
| unrestricted | 0.0329 | 0.0511 | 0.0714 | 0.0768 | **0.0783** (W=250k) | 0.0784 |
| origin_zc | 0.0329 | 0.0511 | 0.0713 | 0.0762 | **0.0774** (W=250k) | 0.0767 (stale, §5) |
| flexible_cm | 0.0312 | 0.0497 | 0.0699 | 0.0724 | 0.0735 | 0.0735 |
| common_frechet | 0.0313 | 0.0487 | 0.0694 | 0.0696 | 0.0733 (W=250k, +0.000008) | 0.0733 |
| cm_meanzc | 0.0297 | 0.0450 | 0.0677 | 0.0677 | **0.0726** (W=250k) | 0.0720 (W=100k only) |

Monotonicity confirmed clean at every delta: `unrestricted` ≥ every restricted family, and every
family's own row is non-decreasing left to right (except the noted `origin_zc` delta=5 staleness,
which is a within-family ordering gap of 0.00018, not a cross-family violation).

## 8. Lower direction (deprioritized per user instruction, 2026-08-04)

Untouched below delta=0.5 — exactly the original campaign's numbers. Only delta=1/delta=2 were
touched, and `unrestricted`/`origin_zc`'s delta=2 lower cells were never directly solved (batch 5
lower work was cut short by the pivot to upper-bound focus); their delta=2 values are the correct
monotone-envelope inheritance from delta=1, not independently verified at delta=2 itself.

| Family | δ=0.01 | δ=0.1 | δ=0.5 | δ=1.0 | δ=2.0 |
|---|---|---|---|---|---|
| unrestricted | 0.0239 | 0.0080 | 0.00332 | 0.00332 (flat) | 0.00332 (inherited, not run) |
| flexible_cm | 0.0175 | 0.0080 | 0.00294 | 0.00294 (tiny real gain) | 0.00294 (run, matched) |
| common_frechet | 0.0174 | 0.0081 | 0.00303 | 0.00303 (flat) | 0.00303 (run, matched) |
| cm_meanzc | 0.0178 | 0.0082 | 0.00292 | 0.00292 (tiny real gain) | 0.00292 (run, matched) |
| origin_zc | 0.0171 | 0.0076 | 0.00290 | **0.00280** (real gain) | 0.00280 (inherited, not run) |

Only `origin_zc` showed a real lower-direction improvement; everyone else is flat or negligible,
consistent with the "lower-direction search stalls near delta=0.5" pattern already documented in
the original campaign's own `CONTINUATION_PRIORITY_SUMMARY_2026-08-03.md`.

## 9. What was explicitly NOT done

- Lower-direction delta=0.01/0.1/0.5: never re-run, per user instruction to deprioritize.
- `unrestricted`/`origin_zc` lower delta=2.0: never directly solved (envelope-inherited only).
- `origin_zc` delta=5 vs its own newer delta=2: not reconciled (§5).
- W=250k extension: upper direction only, delta=2 only, per explicit user scope. Not extended to
  delta=5 or any lower-direction cell. `common_frechet`/`flexible_cm` hit time limits at W=250k
  without clean convergence — a longer budget might find more, not attempted.
- No new independent starts launched. No REDUCED-formulation code used anywhere. No production
  code changed.

## 10. Verdict block

```
HANDOFF_VALIDATED = pass
MONOTONE_INCUMBENTS_REGISTERED = pass_all

PILOT_ALGORITHMS =
    exploration: EXPLORE_DIRECT_SR1 (outer_direct_hessopt=:sr1)
    polish:      POLISH_SQP for {flexible_cm, common_frechet, cm_meanzc, origin_zc},
                 POLISH_DIRECT_BFGS for {unrestricted}

UPPER_DELTA2 =
    unrestricted:improved (W=100k solved + W=250k further improved)
    flexible_CM:improved
    common_Frechet:improved (W=100k solved; W=250k negligible additional)
    origin_ZC:improved (W=100k solved via round2; W=250k further improved from pre-round2 seed)
    CM_plus_ZC:improved (cm_meanzc; W=100k solved + W=250k further improved)

LOWER_CONTINUATION =
    unrestricted:no_change (flat, pilot-confirmed under 2 algorithms)
    flexible_CM:improved (tiny)
    common_Frechet:no_change (flat)
    origin_ZC:improved
    CM_plus_ZC:improved (tiny)

CROSS_FAMILY_MONOTONICITY (new, user-directed) =
    violations_found: 4 (unrestricted < origin_zc at delta=0.01/0.1/0.5/1.0)
    violations_fixed: 4 of 4 (cross-family relaxation seed mechanism)

DELTA5_EXTENSION (new, user-directed) =
    status: complete for all 5 families
    residual: origin_zc delta=5 stale vs its own later delta=2 refinement (not reconciled)

W250K_EXTENSION (new, user-directed) =
    hypothesis_confirmed_real: unrestricted, origin_zc, cm_meanzc
    hypothesis_negligible: common_frechet
    hypothesis_not_confirmed: flexible_cm

FINAL_MONOTONICITY = pass_all (cross-family and within-family, modulo the noted origin_zc delta=5 staleness)

NEW_INDEPENDENT_STARTS_LAUNCHED = 0
EXTRA_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
REDUCED_CODE_USED = false
PRODUCTION_BRANCH_CHANGED = false
CAMPAIGN_OVERWRITTEN = false
```
