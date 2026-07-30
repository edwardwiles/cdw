# Melitz D20 profiled-A welfare continuation, delta=0.5 (2026-07-30)

Governing prompt: the final bounded D20 search experiment for the Melitz extension --
determine whether a profiled-A welfare continuation can find verified gains-from-trade beyond
the fixed-A/f profile at divergence budget `delta = 0.5`, converting the divergence slack the
prior fixed-q A-middle-loop experiment found (`DeltaStar` fixed-A/f `= 0.483276` vs. profiled-A
at fixed q `= 0.289806`) into a more extreme verified welfare outcome. Mandatory addendum:
repair the middle solver's cap-handling/incumbent-retention/FC-GA-dedup defects FIRST, gate the
repair, and only then run the continuation.

Follow-up to
[`melitz_fixed_q_A_middle_loop_experiment_2026-07-30.md`](melitz_fixed_q_A_middle_loop_experiment_2026-07-30.md).
Repo: `trade_robustness_modular`, branch `melitz/fullD-delta-star`. Real D=20 (`noah_D20`,
focal=fra, `sigma=2.5`, `seed=1`, `W=80,000`), delta budget `0.5` throughout, both upper and
lower gains-from-trade directions. Does not modify the Ricardian implementation; does not touch
the free-q coordinates (held fixed at the anchor's own values throughout); no q-gradient,
chamber-graph, relative-parameterization, or broad-multistart development, per the governing
prompt's own non-negotiable scope.

---

## Part 1: mandatory performance and cap-handling addendum

### Section A: AboveEvaluationCap certificate audit

Three representative real-D20 `AboveEvaluationCap` points were re-run directly (bypassing the
typed classifier to also capture the raw, otherwise-unexposed KNITRO `nStatus`) --
`docs/key_results/melitz_addendumA_cap_audit_2026-07-30.csv`:

| point | raw objSol (KNITRO-facing) | lower_limit | threshold_crossed | certified_lower_bound | raw nStatus | any non-finite |
|---|---:|---:|---|---:|---:|---|
| ~1.86e5 (post_switch1 fixed-A) | -1e10 | -10.0 | true | 186,160.70 | -300 | false |
| ~3.23e10 (post_switch2 wandered trial) | -1e10 | -10.0 | true | 3.2325e10 | -300 | false |
| ~15.9 (post_switch2 extreme H-perturbation) | -1e10 | -10.0 | true | 15.887 | -300 | false |

(The second/third points landed at different magnitudes than the three illustrative examples
named in the governing prompt -- ~3.2e10 and ~15.9 rather than ~1e10 and ~1e2-1e3 -- because the
deterministic reconstruction used a fixed, non-random perturbation rather than replaying the
exact prior-session KNITRO trajectory point-for-point; disclosed, not adjusted after the fact.
The three values obtained still span 9 orders of magnitude, which is the substantive point.)

**Determination** (identical for all three, traced directly against `cc_bundle.jl`'s
`f<=Q.lower_limit` branch, not assumed): every `AboveEvaluationCap.certified_lower_bound` is `-f`
at the SPECIFIC dual iterate where the KNITRO-native `lower_limit` early-stop fired during that
attempt (`Q.threshold_crossing_bound[]=-f`, recorded ONLY on a genuinely finite `f` -- a
separate, disclosed `!isfinite(f)` branch guards against ever recording a NaN/Inf value here,
confirmed live: `any_nonfinite=false` on all three probes). Because a `-KN_INFINITY` return
signals "unbounded" to KNITRO, that inner attempt aborts immediately afterward (`raw
nStatus=-300`, "problem appears unbounded", on all three) -- so the recorded crossing bound is,
in effect, also that attempt's terminal iterate. **This is a genuinely valid weak-duality lower
bound, not a NaN/floatmax/sentinel artifact and not an invalid diagnostic** -- but its MAGNITUDE
is real, PATH-DEPENDENT arithmetic (which dual iterate happened to cross `lower_limit=-10` first
along that specific KNITRO trajectory), not a stable property of the underlying `DeltaStar`.
Two different warm starts crossing at different iterates give different, sometimes
wildly-different (9-orders-of-magnitude-apart, per the three probes above) certificates for
what may be similar true divergence. **This path-dependence -- not any invalidity -- is exactly
why Addendum A prohibits feeding this raw number to middle KNITRO as an ordinary objective
value**: doing so hands the quasi-Newton (L-BFGS) curvature model an arbitrarily-scaled,
zero-gradient "stationary" point that has no informative relationship to nearby trial points.

### Section B/C/D: repaired middle-loop driver (`solve_melitz_fixed_q_A_profile_v2`)

New function, `src/melitz/fixed_q_a_middle_loop.jl` (additive -- the original
`solve_melitz_fixed_q_A_profile` is untouched, still covered by its own existing 78/78-assertion
test suite). Repairs all four disclosed defects:

1. **Cap handling (Addendum A)**: `AboveEvaluationCap`/`InfiniteDeltaCertified` trials are never
   fed to KNITRO as an ordinary value. Two modes implemented and empirically A/B-gated (Section
   G below): `cap_handling=:reject` (throws `DomainError` inside `cb_F!`/`cb_G!` -- verified
   directly against the installed KNITRO.jl source, `C_wrapper.jl`'s `_try_catch_handler`, that
   this converts to a genuine `KN_RC_EVAL_ERR`, KNITRO's native per-point-rejection/backtrack
   mechanism, not an immediate abort of the whole solve) and `cap_handling=:barrier` (Addendum
   A's own disclosed fallback -- a FIXED, bounded value, `cap_barrier_multiple * cap`, zero
   gradient, the SAME constant for every certified-bad point regardless of its true raw
   certificate, matching `finite_delta_outer.jl`'s own established `divergence_sentinel`
   convention).
2. **Strict incumbent retention (Addendum B)**: the input start is classified first (seeding the
   incumbent iff `FiniteSolved`); every `FiniteSolved` trial updates the incumbent if strictly
   better; at the end, the (cold-reverified) start, best trial, and KNITRO's own (cold-reverified)
   terminal point are all compared and the best is returned --
   `Delta_incumbent <= Delta_start_verified + tol` asserted internally, not merely hoped.
3. **Continuation at both levels (Addendum C)**: the caller passes the preceding accepted
   `A_free` (continuation start) and/or a cellwise `p*`-compensated `A` (from
   `lfd_preserving_state.jl`'s formula, reused verbatim) as `x_start`; the inner dual warm state
   (`session.obj.x`/`use_cached_x`) is left untouched by the driver itself between calls (only
   its own internal cold-reverification steps use `warm_start_source=:neutral`, and only at the
   very end, so they cannot leak into a subsequent point's own warm state).
4. **One inner solve per unique A point (Addendum D)**: `melitz_middle_objective_and_gradient_cached!`
   wraps the Phase-2 evaluator with the SAME `MelitzExactPointCache`/`melitz_exact_cache_get`/
   `melitz_exact_cache_insert!`/`melitz_heavy_snapshot`/`melitz_heavy_restore!` machinery the
   production `(A,f)` outer search already uses (`finite_delta_outer.jl`) -- keyed on the exact
   `theta_free_middle` fingerprint (a deterministic, injective function of A + fixed q + fixed
   g). A second, lightweight cache (`MelitzMiddleBadPointCache`) covers repeat requests at an
   already-classified `AboveEvaluationCap`/`InfiniteDeltaCertified` point. Both D4 (smoke test)
   and D20 (gate run) confirm `unique_inner_solves <= unique_A_points` (asserted internally, not
   merely observed) and `NumericalFailure` is never returned as a value.

### Section E: fixed-q hot-path profiling (D20 anchor, 24 unique A evaluations)

`docs/key_results/melitz_addendumE_hotpath_profile_2026-07-30.csv` /
`melitz_addendumE_perpoint_2026-07-30.csv`. `MELITZ_PROFILE[]=true`, real-D20 anchor, 20 Julia
threads / 1 BLAS thread.

| category | count | total_s | mean_ms |
|---|---:|---:|---:|
| `fc_inner_hess_eval` | 230 | 2.742 | 11.92 |
| `fc_inner_obj_eval` | 680 | 2.416 | 3.55 |
| `fc_operator_merge` (moment-operator update) | 24 | 2.111 | 87.96 |
| `moment_operator_link_update` (focal-link scalar update, subset of the row above) | 24 | 1.975 | 82.29 |
| `fc_inner_grad_eval` | 266 | 0.880 | 3.31 |
| `screen_stored_dual_*` | 24 | 0.659 | -- |
| `fc_inner_dpsi_eval` | 503 | 0.252 | 0.50 |
| `fc_theta_expand` (state reconstruction) | 24 | 0.003 | 0.14 |
| `fc_warm_start_resolve` | 19 | ~0 | ~0 |

**Immutability check**: cutoff/rank/participation structure (`melitz_origin_intervals(...).rank`
for every origin) was fingerprinted at all 24 unique `A` evaluations and found **bit-identical**
to the anchor's own fingerprint in every case (`ranks_match_anchor=true`, all 24 rows) --
confirming the module header's "zero participation switches by construction" claim empirically,
not merely algebraically, under the repaired driver.

**Cost breakdown**: the genuine inner KNITRO solve (hess+obj+grad+dpsi callbacks) accounts for
`~6.29s` of `~9.06s` total tracked time (**~69%**) across the 24 points; the moment-operator
update accounts for `~2.11s` (**~23%**), of which `~1.97s` (93% of that) is specifically the
`moment_operator_link_update` step -- i.e. already the minimal "update only the A-dependent
focal-link scalar in place" operation the governing prompt itself describes as the target state,
not a wasteful full moment-matrix rebuild. State reconstruction (`fc_theta_expand`) and
warm-start resolution are both negligible (`<0.2%` combined).

### Section F: normalized-moment prototype -- **not implemented (trigger not met)**

The conditional trigger ("implement only if profiling shows meaningful time in A-scaled moment
updates OR poor inner conditioning") is **not met** in a way that justifies the prototype: the
inner KNITRO solve dominates cost (~69% vs. ~23%), and the ~23% moment-operator cost is already
the minimal per-A-point update (the `moment_operator_link_update` sub-step, not a redundant
rebuild) -- there is no elimination opportunity a normalized `T_od(p,q)-H_od(A)=0` reformation
would capture beyond what the existing matrix-free architecture already does. Building and
validating the prototype (exact `DeltaStar`/LFD equality, dual warm-start transform, exact
analytical gradient, no dense G, measured benefit -- Addendum F's own bar) would be pure
additional risk for no expected payoff given this profiling evidence. **Skipped, per the
addendum's own explicit instruction not to adopt machinery that would not materially help.**

### Section G: performance gate -- v1 vs. v2(reject) vs. v2(barrier), D20 anchor

`docs/key_results/melitz_addendumG_gate_v1_vs_v2_2026-07-30.csv`. Same start (anchor `A_free`),
same `box=0.1`/L-BFGS options file, `max_evals=120`:

| driver | Delta reported | wall_s | classified (F/C/I) | unique_A_points | unique_inner_solves | cache_hits | cache-hit rate |
|---|---:|---:|---|---:|---:|---:|---:|
| v1 (original) | 0.307479 | 47.2 | 95/25/0 | -- | -- | -- | -- |
| v2, `cap_handling=:reject` | 0.418725 | 23.0 | 27/94/0 | 107 | 107 | 14 | 8.7% |
| v2, `cap_handling=:barrier` | **0.287953** | 30.6 | 117/4/0 | 64 | 64 | 57 | **34.8%** |

**A real, disclosed finding, not glossed over**: the "preferred" `:reject` mode (Addendum A's
own first-choice recommendation) empirically explores WORSE than the ORIGINAL v1 driver within
the identical evaluation budget at this exact point (`0.419` vs. `0.307`) -- far more trials
land in `AboveEvaluationCap` (94 vs. 25), consistent with `finite_delta_outer.jl`'s own
already-documented finding, at the OUTER (A,f) level, that switching to eval-errors "measurably
changed exploration (backtrack-to-near-zero)" relative to a smooth fixed-value landscape. The
`:barrier` fallback mode (also anticipated explicitly by the governing addendum) resolves this:
best `Delta` of all three configurations, by far the fewest capped trials (4), and the highest
FC/GA cache-hit rate (34.8%, vs. 8.7% for `:reject` -- fewer distinct capped points means more
repeat requests at already-known-good points). **Winner: `cap_handling=:barrier`
(`cap_barrier_multiple=5.0`, i.e. a fixed value of 50.0)**, used for the entire welfare
continuation below.

**Gate checks (all passed)**: `Delta_incumbent <= Delta_start_verified` for both `v2` modes;
`unique_inner_solves <= unique_A_points` for both (107<=107, 64<=64, both exactly equal --
every unique A point triggered exactly one inner solve); zero `NumericalFailure` outcomes in
either mode; above-cap points structurally never fed as a raw certificate (by construction of
`_eval`, both modes). **OVERALL GATE: PASS.**

---

## Part 2: profiled-A welfare continuation, delta=0.5

Uses the repaired `solve_melitz_fixed_q_A_profile_v2` with the gate's own winning
configuration, `cap_handling=:barrier, cap_barrier_multiple=5.0` (fixed barrier value `50.0`);
`max_evals=120`, `box=0.1`, L-BFGS middle options file, per the governing prompt's own middle
settings. Exactly two deterministic starts at every welfare point (continuation from the
preceding accepted point; cellwise `p*`-compensated), no random starts, no log-H second
coordinate system.

### Setup

Real-D20 anchor: `Delta0 (fixed-A/f) = 0.483276`, `GT0 = 6.290641%` (`wage_ratio=1.295205`,
`sigma=2.5`). Welfare coordinate `g = theta_free[1] = log(gamma_prime)`; `q(g)` reconstructed at
each candidate `g` via the EXISTING q-gravity map (`expand_free_theta_logcutoff`'s own
`build_q_gravity_offset`/`derive_qjj_from_autarky_cutoff`), holding the anchor's own free-`q`
coordinates (`theta_plain0[2+nA:end]`) FIXED throughout -- **verified live**: perturbing `g` by
`0.01` moved `q[j,j]` and (necessarily, being an affine pivot) exactly one other cell (the
q-pivot's own physical pivot cell), zero other free `q` cells changed by more than `1e-9`.

Initial welfare point (`g=g0`, the anchor itself): profiled using the anchor `A` and the
(trivially-identical, since `q(g0)==q0` exactly) cellwise-compensated `A`.
**`Phi(g0) = 0.287953`** (cross-checks the prior session's own reported `0.289806` to within
0.6% -- the small residual gap is attributable to the different cap-handling mode, not a
discrepancy in the underlying middle-loop machinery).

### Continuation results

**Upper direction (increasing GT%) -- CONVERGED via safeguarded bisection, 13 of 15 points
used**: expanded cleanly through 6 accepted points (GT `6.29% -> 6.34% -> 6.42% -> 6.53% ->
6.70% -> 6.90% -> 7.10%`, `Phi` rising smoothly `0.288 -> 0.293 -> 0.308 -> 0.332 -> 0.373 ->
0.431 -> 0.499`, i.e. monotonically approaching the budget from below), bracketed at the 7th
point (`GT=7.297%`, `Phi=0.580`, over budget), then bisected for 6 more points, all of which
remained (narrowly) over budget, progressively tightening the infeasible edge down to
`GT=7.1000%, Phi=0.5002` -- **the bracket width shrank to `0.0031` percentage points (below the
`0.005` stopping tolerance), a clean convergence, not a budget cutoff**. The retained extreme
point is therefore the last (and only) point on the FEASIBLE side of this tight bracket:
**`GT=7.096891%, Phi=0.499019`, FiniteSolved**, essentially exactly at the `delta=0.5` boundary.

**Lower direction (decreasing GT%) -- budget-exhausted, did not bracket, 15 of 15 points used**:
`Phi` fell steadily and never approached the `0.5` budget (`GT=6.29%,Phi=0.288 -> ... ->
GT=3.68%,Phi=0.046` at the 15th point) -- the fixed step-doubling-then-capped-growth schedule
(capped at `0.2` GT-percentage-point steps per the governing prompt's own `<=50%` growth rule)
was not aggressive enough to reach this side's own true boundary within the 15-point budget.
**Disclosed limitation, not a negative result**: this direction simply was not pushed far
enough to find where either the fixed-A/f or the profiled-A frontier actually binds at
`delta=0.5` on the downside; a follow-up with a larger initial step or a higher point budget
would be needed to resolve it (out of this bounded experiment's own scope).

Zero `AboveEvaluationCap`/`InfiniteDeltaCertified`/`NumericalFailure` outcomes were encountered
across all 30 welfare points (60 middle-loop profile calls, two starts each) in the actual
campaign -- every single point was cleanly `FiniteSolved` (the earlier gate/audit runs
deliberately probed degenerate/extreme points; the real campaign, using genuine
continuation+compensated starts throughout, never needed the cap-handling machinery's rejection
path at all in practice, though it remains available/tested for whenever it is needed).

### Baseline comparison

`docs/key_results/melitz_d20_profiledA_continuation_summary_2026-07-30.csv`:

| | **upper** | **lower** |
|---|---:|---:|
| starting profiled-A point (GT0, Phi) | 6.290641%, 0.287953 | 6.290641%, 0.287953 |
| **fixed-A/f profile Delta at the extreme GT** | **1.035952** (FiniteSolved) | 0.056433 (FiniteSolved) |
| **best profiled-A Delta at the same extreme GT** | **0.499019** (FiniteSolved) | 0.045694 (FiniteSolved) |
| extreme GT reached | 7.096891% | 3.684391% |
| best start at the extreme | continuation | continuation |
| A movement norm (log units, vs. anchor) | 0.391268 | 0.529903 |
| f movement norm (log units, vs. anchor) | 0.589575 | 0.790181 |
| welfare points evaluated | 13 (converged) | 15 (budget-exhausted) |
| classification counts (F/C/I/NumFail) | 14/0/0/0 | 16/0/0/0 |

**The decisive comparison is the upper direction**: at `GT=7.096891%`, the fixed-A/f profile
(the anchor's OWN unchanged `A`/`f`, re-evaluated at this new `q`) gives `DeltaStar=1.035952` --
**more than double the `delta=0.5` budget, i.e. this welfare level is flatly UNREACHABLE under a
fixed technology matrix**. The profiled-A search, genuinely re-optimizing `A` (and hence `f`,
via the fixed-`q` cellwise recovery) at the SAME `q`/`g`, finds `DeltaStar=0.499019` -- verified
`FiniteSolved`, safely inside the budget, at a point resolved to within `0.003` GT-percentage-
points of the TRUE `delta=0.5` boundary. This is a genuine, decisively verified
**+0.81-percentage-point extension of the reachable upper gains-from-trade frontier** (from
`GT0=6.29%`, where the anchor's own fixed-A/f `Delta0=0.483` is already close to the cap, out to
`GT=7.10%`) that a fixed-A/f search cannot reach at all within the SAME divergence budget.

The lower-direction comparison (`1.036%` vs `0.046%`... i.e. `0.056` vs `0.046`) is NOT decisive
in the same sense -- both approaches remain comfortably within budget at the most extreme point
tested, so this direction does not yet demonstrate (or contradict) a frontier extension; it
simply was not pushed far enough.

### Decision

### **A -- Profiled-A continuation makes material welfare progress** (upper direction; lower
direction inconclusive within this bounded experiment's own budget, disclosed above, not
evidence against A).

**Recommendation** (matching Decision A's own governing-prompt wording): use the repaired
fixed-q A-middle-loop profile (`solve_melitz_fixed_q_A_profile_v2`, `cap_handling=:barrier`) as
the practical D20 Melitz welfare search going forward, describing it as a conservative
FIXED-CUTOFF subset of the full model (free-`q` coordinates are held at the anchor's own values
throughout -- only `A`/`f` and the welfare coordinate `g` genuinely vary). **Do not claim global
optimality** -- every reported `Phi(g)` is a verified, cold-reverified LOCAL optimum from a
SMALL, deterministic two-start portfolio (continuation + cellwise-`p*`-compensated), not an
exhaustively-searched global one; the D20-scale start-sensitivity the prior fixed-q-only
experiment documented (Phase 4/5 there) has not been re-litigated here, only reused via the
same reliable start portfolio.

**Concretely**: at `delta=0.5`, this session verifies a real, useable, genuinely-optimized-`A`
welfare point at `GT=7.10%` that is completely inaccessible to the existing fixed-A/f outer
search at the same budget -- a materially more informative upper gains-from-trade bound for
this model/calibration than the fixed-A/f search alone can produce. The lower-direction
boundary remains open; a natural, cheaply-scoped follow-up (NOT undertaken here, per the
governing prompt's own non-negotiable scope) would extend the same continuation machinery with
a larger initial step or higher point budget specifically on that side.

---

## Reproducible scripts

- `scripts/melitz_addendum_audit_and_gate_2026-07-30.jl` -- Sections A/E/G above.
- `scripts/melitz_d20_profiled_A_welfare_continuation_2026-07-30.jl` -- Part 2 above.
- `scripts/melitz_middleloop_v2_d4_smoke_2026-07-30.jl` -- throwaway D4 correctness smoke test
  of `solve_melitz_fixed_q_A_profile_v2` (not part of the deliverable, kept for provenance).
- `scripts/melitz_addendum_v2_standalone_test_2026-07-30.jl` -- standalone isolated run of the
  new testset (avoids the pre-existing, unrelated `mul_G!` SIGSEGV in the full 8000-line suite,
  same convention the original fixed-q-A-middle-loop session used). **16/16 assertions pass.**
- New source: `src/melitz/fixed_q_a_middle_loop.jl` (additive section appended, ~450 new lines);
  new testset `test/melitz/runtests.jl`, `"Mandatory addendum 2026-07-30 (repaired v2
  middle-loop driver)"`.
- Machine-readable results: `docs/key_results/melitz_addendumA_cap_audit_2026-07-30.csv`,
  `melitz_addendumE_hotpath_profile_2026-07-30.csv`, `melitz_addendumE_perpoint_2026-07-30.csv`,
  `melitz_addendumG_gate_v1_vs_v2_2026-07-30.csv`,
  `melitz_d20_profiledA_continuation_points_2026-07-30.csv`,
  `melitz_d20_profiledA_continuation_summary_2026-07-30.csv`.

## Provenance

- Repo: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular`, branch
  `melitz/fullD-delta-star`.
- Julia: `juliaup` toolchain, `-t 20` (Melitz-local `CLAUDE.md`'s documented production
  default), `BLAS.set_num_threads(1)` explicitly.
- KNITRO: Artelys Knitro 13.0.1, academic license.
- Committed locally at the end of this session; **not pushed** (per this repo's own standing
  "confirm before pushing to production" convention).
- **Data-currency caveat (disclosed, not silently absorbed)**: the user flagged mid-session that
  "the underlying data has just changed," which may affect these numbers if the pipeline is
  re-run now. `real_data/noah_D20/{pi,L,tau,countries}.csv` on this filesystem carry an
  unchanged mtime of `2026-07-23 16:56` (predating this session's own run), so the change is
  either to a different/shared upstream source this session did not touch directly, or occurred
  in a location not confirmed here. **Every number in this report reflects the specific
  `real_data/noah_D20` snapshot used at run time (2026-07-30)** -- if the calibration pipeline
  is re-run against a newer data snapshot, the anchor `Delta0`/`GT0` and every downstream
  profiled-A result above should be expected to shift and would need re-verification, not
  assumed unchanged.
