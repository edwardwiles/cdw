# Continuation 8, Section 9: nested-W continuation (W=8000/20000/80000)

**Branch** `c8-nested-w`, base `2896091` (tip of `diag/fullA-d4-exact` after Wave 1, Wave 2A/2B/2C,
and Section 8's registration of `lower_v2` as the headline lower incumbent). **Machine**
`demand.mit.edu`, `JULIA_NUM_THREADS=20`. **Candidates tested**: `upper_lfixcomposite_sr1_60s`
(κ=0.17245688540655113, γ'=0.8926359584642946) and `lower_v2` (κ=0.004387827651021192,
γ'=0.9973649883022927), both from `candidate_registry.jl`.

Note on authorship: this workstream's compute (the nested-draw-pool build, the reeval/reopt grid,
and the gradient cross-check) was run by a background agent this session and the raw
CSVs/logs/A-solutions it produced are trustworthy and used as-is below. The agent stalled twice
after finishing real compute (once after the grid finished, once after the gradient cross-check
finished) without writing this report or committing — this document and the final commit were
completed directly by the coordinating session from the agent's own output artifacts, not
re-derived or guessed.

## 1. Draw-pool methodology — why a fresh seed, not a replay of the discovery draws

The registered candidates were found against draws from `AD_PARAMS.seedU=888` (via
`d4_exact_setup()`'s internal `drawU()` call). This workstream does **not** reuse that seed. Reason
(verified from source, documented in `full_aod_diag/d4_exact/c8_nestedw_context.jl`): `drawU()` fills
its `W x D` matrix in **column-major** order after a single `Random.seed!(seedU)` call, so column 1
(RNG stream positions `1:W`) is identical regardless of `W`, but column 2 starts at stream position
`W+1` — which differs between e.g. `W=8000` (starts at 8001) and `W=80000` (starts at 80001). Only
one of the four columns would coincidentally nest; the other three would not. So a literal replay of
`seedU=888` at three different `W` values would **not** give genuinely nested draws.

The fix: draw one `Wmax=80000 x D` pool once, with a **separate, fixed** seed (`NESTEDW_SEED=91234`),
using the same elementwise transform (`rand!` then `-log(1-x)`) as the production `genExpRands!`. Then
`W`-specific contexts are built normally via `d_exact_setup_scaled` and have their draw-derived arrays
(`obj.U`, `obj.γ.Ū`, `obj.γ.Uσ`) overwritten in place with `NESTEDW_POOL[1:W, :]` — a literal row-prefix,
so nesting across `W=8000/20000/80000` is exact by construction (smoke-tested by the agent before the
full run).

**Consequence for interpretation**: because this is a *fresh* draw realization (not the discovery
draws), the "reeval" rows below are not testing "what happens as the ORIGINAL sample grows" — they are
testing whether a registered candidate, tuned to fit one specific finite sample, generalizes to an
**independent** same-size (and larger) resample. That is arguably a more informative robustness check
than a literal-growth check would have been, but it is a different question and is labeled as such
throughout.

## 2. Re-evaluation grid: candidates are NOT feasible against a fresh W=8000/20000 draw set

| W | candidate | γ' | κ (fixed, unchanged) | Δ_dual | feasible | h_diag |
|---|---|---|---|---|---|---|
| 8000 | upper | 0.892636 | 0.172457 | **1.154966** | **false** | 9.89e-4 |
| 8000 | lower_v2 | 0.997365 | 0.004388 | **1.077992** | **false** | 4.28e-4 |
| 20000 | upper | 0.892636 | 0.172457 | **1.103603** | **false** | 4.21e-4 |
| 20000 | lower_v2 | 0.997365 | 0.004388 | **1.134810** | **false** | 3.61e-4 |
| 80000 | upper | 0.892636 | 0.172457 | 0.992771 | **true** | 4.13e-5 |
| 80000 | lower_v2 | 0.997365 | 0.004388 | **1.126391** | **false** | 9.68e-5 |

Both candidates' own (g,A) are held fixed here — only the draw realization changes. Every row is
infeasible (Δ_dual > δ=1) **except upper at W=80000**, which becomes feasible with real margin
(Δ_dual=0.9928). This is consistent with the law of large numbers: as W grows the sample-estimated
divergence should concentrate toward its population value, and apparently the population value near
the upper candidate's (g,A) is comfortably feasible, while near `lower_v2`'s (g,A) the population
divergence is close enough to δ=1 that even W=80000 fresh draws still read infeasible (Δ_dual=1.126,
i.e. the resampling noise at this W is still large relative to how close to the boundary `lower_v2`
sits — expected, since `lower_v2` by construction sits almost exactly on the Δ=δ boundary against its
OWN discovery draws).

**Read this as**: both candidates are real, but their exact numerical feasibility at W=8000 is
sensitive to the specific realized draws — not evidence the candidates are wrong, but a genuine
finite-sample-fragility caveat that a single fixed W=8000 draw set does not by itself certify
population-level feasibility with much of a safety margin, especially for `lower_v2`, which is (as
Section 8 already established) a boundary point almost by definition.

## 3. Re-optimization grid: both candidates recover full feasibility once A is allowed to adapt

| W | candidate | γ' (reopt) | κ (reopt) | Δ_dual | feasible | knitro_status | runtime (s) | outcome |
|---|---|---|---|---|---|---|---|---|
| 8000 | upper | 0.894765 | 0.169165 | 0.999773 | true | -103 | 19.1 | converged |
| 20000 | upper | 0.892955 | 0.171964 | 0.999614 | true | **-401** | 90.1 | **stalled at budget** |
| 80000 | upper | 0.891401 | 0.174364 | 0.999980 | true | **-401** | 400.2 | **stalled at budget** |
| 8000 | lower_v2 | 0.997288 | 0.004515 | 0.998379 | true | -103 | 7.5 | converged |
| 20000 | lower_v2 | 0.997373 | 0.004375 | 0.999238 | true | -102 | 34.0 | converged |
| 80000 | lower_v2 | 0.997197 | 0.004667 | 1.000000 | true | -101 | 56.3 | converged |

All six re-optimization runs are exact-feasible at their reported point (cold-hard-rechecked per the
`note` column in the raw CSV: `converged_best_feasible_tracked_cold_recheck` for the four genuine
convergences, `best_feasible_stalled_at_budget_cold_recheck` for the two upper-direction stalls — both
kinds are cold-verified, the "stalled" label refers only to the KNITRO termination status, not to
whether the reported point is itself trustworthy).

**Asymmetric difficulty**: `lower_v2`'s re-optimization genuinely converges (KNITRO status -101/-102/
-103, all real convergence codes) at every W, including W=80000 in 56s. The upper candidate's
re-optimization only genuinely converges at W=8000 (19s); at W=20000 and W=80000 it hits `-401`
(iteration/time-budget stall) even after 90s and a generous 400s respectively — the best-feasible point
found is still reported and is itself cold-verified feasible, but KNITRO did not certify it as a local
optimum within budget. This is consistent with the upper direction generally being the harder/slower
direction throughout this investigation (larger κ, more A-block curvature) and with per-inner-solve
cost scaling with W.

**κ trajectory**: upper reopt κ drifts mildly upward with W (0.1692 → 0.1720 → 0.1744) — plausibly
because the two higher-W points are budget-stalled, not fully converged, so these are not directly
comparable to the W=8000 converged value. lower_v2 reopt κ is non-monotonic and stays close to the
original discovery-draws value (0.004516 → 0.004375 → 0.004667 vs. the registered 0.004388) — no
systematic drift, consistent with `lower_v2` being a genuinely stable point rather than an artifact of
the specific discovery draws, once A is allowed to re-adapt to whatever draws are in view.

**Headline**: at W=8000-80000 there is no case where re-optimizing against a fresh draw pool finds a
MEANINGFULLY better candidate than what Section 8 already registered — the reopt κ values bracket the
registered numbers closely in both directions. This is a "no further improvement, and no regression
either, once you let A re-adapt" result, which is itself a useful stability finding: it is the
FIXED-A reevaluation (§2) that is fragile to resampling, not the underlying optimization problem.

## 4. Gradient cross-check: fast (`lfix_composite`) vs. slow (`delta_fd`) directions

Three points tested (upper/W=8000, upper/W=80000, lower_v2/W=80000), each against two slow-gradient
bandwidths (fixed h=0.01, and an adaptively-selected h).

| candidate | W | full_cos (h=adapt) | γ-component relerr (h=adapt) | A-block cos (h=adapt) | A-block sign-agree (h=adapt) | wall fast (s) | wall slow, h=adapt (s) |
|---|---|---|---|---|---|---|---|
| upper | 8000 | 0.99994 | 0.0043 | 0.5207 | 0.733 | 8.93* | 1.08 |
| upper | 80000 | 0.99995 | 0.0000 | 0.5898 | 0.800 | 1.80 | 9.63 |
| lower_v2 | 80000 | 0.99978 | 0.0013 | 0.1675 | 0.933 | 2.24 | 8.48 |

\* The upper/W=8000 "fast" wall time (8.93s) is a JIT-compilation artifact, not a genuine cost — it is
the first call in a fresh process to the `lfix_composite`-family callback path, which (per
`docs/fullA_algorithm_frontier_c8.md` §3, this same continuation-8 session) is known to consume most
of its first ~10-20s on compilation of the Wave-1-expanded call graph (winner-margin certificate,
top-3 cache, compressed-live wiring) rather than actual computation. The other two points, run after
that warm-up, show the expected pattern: fast is **~4-5x cheaper** than the slow reference once warm
(1.8-2.2s vs 8.5-9.6s).

**Full-vector agreement is excellent** (cosine 0.9998-0.9999, γ-component relative error at or below
0.4% using the adaptively-chosen bandwidth) — the fast gradient's overall descent direction is
trustworthy, consistent with every outer-loop run in this investigation converging correctly on the
fast gradient.

**A-block-only agreement is real but weaker, and should not be overstated**: restricted to just the
15 A_od coordinates (excluding γ'), cosine similarity ranges 0.17-0.59 and elementwise sign agreement
is only 73-93% (i.e. up to 27% of A-block coordinates disagree in SIGN between the fast and slow
gradient at a given point). This does not appear to break the outer optimization in practice (§3's
reopt runs all converge or stall gracefully, never diverge), and is plausibly explained by the A-block
being a comparatively flat/near-degenerate direction at these points (consistent with
`results/fullA_d4/bb74649/gamma_profile_nonmonotonicity_report.md`'s finding of a "flat A-valley" near
the profile minimum — many A_od perturbations barely move Δ, so small numerical differences between two
different gradient ESTIMATION methods can easily disagree on sign in near-flat directions without
either being "wrong"). Flagged honestly as an open, unresolved point of disagreement rather than
smoothed over — it is a genuine discrepancy between the fast and slow A-block sub-gradients, just one
that (so far) hasn't visibly compromised the outer solves that rely on the fast gradient.

## 5. Bottom line

- Fixed-point feasibility is **not** robust to resampling at W=8000-20000 for either candidate under an
  independent draw set of the same size; only the upper candidate stabilizes to feasible by W=80000.
- Re-optimizing (letting A re-adapt) recovers feasibility at every W tested for both candidates, with
  κ values that stay close to the registered numbers — no evidence of a better candidate lurking at
  higher W, and no evidence of regression either.
- The upper direction's re-optimization becomes harder to fully converge (budget-stalled, not
  diverged) as W grows; the lower direction's does not.
- The fast gradient's overall direction is trustworthy; its A-block sub-gradient shows real,
  unresolved sign disagreement with the slow reference in a near-flat region — worth a note for anyone
  building further tooling on `lfix_composite`, not an immediate correctness blocker given the observed
  outer-loop behavior.

## Reproduce

```bash
source .knitro_env.sh
JULIA_NUM_THREADS=20 julia --project=. full_aod_diag/d4_exact/c8_nestedw_run_grid.jl
JULIA_NUM_THREADS=20 julia --project=. full_aod_diag/d4_exact/c8_nestedw_gradcheck.jl
```

Raw data: `results/fullA_d4/2896091/c8_nestedw_grid_20260718_181252/c8_nestedw_results.csv` (12-row
grid) + `c8_nestedw_a_solutions.jld2` (A-solutions, keyed as in the `a_solution_key` column);
`results/fullA_d4/a1e7681/c8_nestedw_gradcheck_20260718_192858/c8_nestedw_gradcheck.csv` (3-row
cross-check).
