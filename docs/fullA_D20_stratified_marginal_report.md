# Per-country stratified-marginal (Latin Hypercube) sampling vs plain pseudorandom MC: does it reduce zero-winner-pair incidence at smaller W?

Continuation 10, branch `c10-stratified-marginal` (forked from `diag/fullA-d4-exact` @
`97cdd80`, which includes the three just-merged Continuation-10 workstreams: prod
wiring, chunked Hessian, QMC/importance-sampling). Worktree:
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4-c10-stratmarg`. Real D=20 data
(France focal), `demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`.

**This is a narrowly-scoped, single-question follow-up.** It does NOT re-test whether
a smarter draw design reduces gradient/kappa estimation NOISE — that question was
already asked and answered (no measurable benefit) by the prior QMC investigation
(`docs/fullA_D20_qmc_investigation_report.md`, branch `c10-qmc-is`, scrambled
Halton/Sobol). This task asks a genuinely different question: does per-country
**stratified-marginal (Latin Hypercube) sampling** fix the **tail-coverage** failure
mode `docs/fullA_D20_infeasibility_screening_report.md` sec 5 found at W=8,000 —
5 of 400 positive-share bilateral (origin, destination) pairs getting literally
**zero** winning draws on that draw support?

---

## 1. Design

`full_aod_diag/d4_exact/c10_stratmarg_draws.jl::stratified_marginal_U(W, D; seed, Q=W)`
— additive-only, follows the house style of `qmc_draws.jl`'s
`pseudorandom_U`/`halton_U`/`sobol_U` (same `(W, D; seed) -> W x D Exp(1)` signature),
reuses the model's REAL Exp(1) inverse-CDF (`exp_from_uniform01`, defined in
`qmc_context_real_d20.jl`, unmodified) — not a new ad hoc transform.

- **Q = W (finest stratification)**, chosen over a coarser Q because constructing D
  independent length-W permutations is `O(D*W)` — at most 20×80,000 = 1.6M elements,
  confirmed sub-40ms even at W=80,000 (§4) — so there is no computational reason to
  coarsen. `Q` is left as an optional keyword for future use at scales where this
  stops being cheap; it is not exercised at Q<W anywhere in this task.
- Each of the D columns (`U` is W×D, **origin-indexed only**, shared across
  destinations — confirmed in `qmc_context_real_d20.jl`'s header note) is stratified
  with an **independent** random permutation (`Random.shuffle` on a fresh assignment
  vector **per column**) — not one shared permutation across columns, which would
  induce spurious cross-country rank correlation absent from the real model.
- Standalone correctness check (not committed as a script, reproduced here): at
  W=20/D=4 and W=100/D=3/Q=10, every column's inverted strata form an exact
  permutation of `0:W-1` (Q=W case) or an exactly-balanced count per stratum (Q<W
  case), and two columns' rank orders are never identical (independence confirmed).

## 2. Test matrix

W ∈ {8000, 20000, 40000, 80000} × scheme ∈ {plain pseudorandom, stratified-marginal}
× point ∈ {calibration (gp0=0.98776, A0), delta=1 upper candidate (gamma'=0.955701,
its A, from `qmc_fixed_points/upper_candidate_w.csv`, same provenance as the QMC
report)}. The evaluation **point is held fixed** across every draw realization tested
(same experimental design the QMC investigation used) — the question is whether the
SAME outer-loop point stays winner-feasible on different draw sets, not whether a
new point should be re-derived per draw set.

Replicate count tiered by measured per-W context-build cost (same tiering principle
as `c10_phase7_qmc_precision_comparison.jl`'s `SCRAMBLE_PLAN`): **15 reps at W=8000
and 20000, 12 at W=40000, 10 at W=80000** (a regression sanity check, not the focus).
Total wall clock for the full sweep: **~27 minutes** (13:05–13:32, well under the
~60–90 min estimate that assumed no caching benefit across replicates).

Metrics computed via the EXISTING, unmodified exact screen
(`infeasibility_screen.jl`: `precompute_pairwise_M`, `compute_a_od`,
`pairwise_certificate`, `order_destinations`, `screen_hard_winners` with
`full_scan=true` for an exact total zero-win-pair count, not just first-failure).

Driver: `full_aod_diag/d4_exact/c10_stratmarg_screen_sweep.jl`. Raw CSVs:
`results/fullA_d4/97cdd80/c10_stratmarg_screen_sweep_20260719_130454/` —
`screen_sweep.csv` (960 rows), `screen_sweep_aggregate.csv`, `followup_real_solves.csv`.

## 3. Main result: zero-winner-pair incidence

| point | W | scheme | mean n_zero | std | range | frac fully-feasible (0/400 zero-win pairs) |
|---|---|---|---|---|---|---|
| calibration | 8000 | plain | 4.47 | 3.20 | [1, 10] | 0/15 |
| calibration | 8000 | **stratified** | **1.33** | 0.90 | [0, 3] | **2/15** |
| calibration | 20000 | plain | 0.40 | 0.63 | [0, 2] | 10/15 |
| calibration | 20000 | **stratified** | **0.07** | 0.26 | [0, 1] | **14/15** |
| calibration | 40000 | plain | 0.00 | 0.00 | [0, 0] | 12/12 |
| calibration | 40000 | stratified | 0.00 | 0.00 | [0, 0] | 12/12 |
| calibration | 80000 | plain | 0.00 | 0.00 | [0, 0] | 10/10 |
| calibration | 80000 | stratified | 0.00 | 0.00 | [0, 0] | 10/10 |
| upper_candidate | 8000 | plain | 3.27 | 2.05 | [0, 8] | 1/15 |
| upper_candidate | 8000 | **stratified** | **1.27** | 0.96 | [0, 3] | **3/15** |
| upper_candidate | 20000 | plain | 0.27 | 0.59 | [0, 2] | 12/15 |
| upper_candidate | 20000 | **stratified** | **0.07** | 0.26 | [0, 1] | **14/15** |
| upper_candidate | 40000 | plain | 0.00 | 0.00 | [0, 0] | 12/12 |
| upper_candidate | 40000 | stratified | 0.00 | 0.00 | [0, 0] | 12/12 |
| upper_candidate | 80000 | plain | 0.00 | 0.00 | [0, 0] | 10/10 |
| upper_candidate | 80000 | stratified | 0.00 | 0.00 | [0, 0] | 10/10 |

**Stratification gives a real, consistent, replicated reduction**: roughly a
**3–4x drop in mean zero-winner-pair count at both W=8,000 and W=20,000** for both
points, and it materially raises the fully-structurally-feasible replicate rate at
W=20,000 (calibration 67%→93%, upper 80%→93%). **W=80,000 sanity check: zero
regression** — both schemes are perfectly feasible across every replicate, as
expected.

**Stratification does NOT reach 100% reliability at either W=8,000 or W=20,000.**
Even with the finest per-country marginal stratification, some replicates still
produce a positive-share pair with zero winning draws (2/15 and 3/15 fully-feasible
at W=8,000 for calibration/upper; 14/15 at W=20,000, not 15/15). **Neither draw
scheme's `newly_feasible_via_stratification` flag (frac_fully_feasible: stratified=1.0
AND plain<1.0) ever fired** in the automated sweep — the improvement is real but
partial, not a clean flip from "always fails" to "always works."

**Incidental finding**: at W=8,000, the pairwise certificate alone (the cheap,
draw-free-at-each-outer-point sufficient condition) catches 93% of calibration's
genuinely infeasible replicates directly, but 6.7% needed the full destination-major
winner scan to detect (`frac_pairwise_infeasible=0.933` vs `frac_winner_scan_feasible
=0.0`, i.e. `n_zero>0` in 100% of replicates but not all were pairwise-caught) — the
prior screening report found the pairwise certificate catches 100% of D=20 genuine
failures in its own (much larger-perturbation) sweep at W=80,000; this smaller-W sweep
shows that guarantee does not automatically extend to every regime.

**W=40,000 is already 100% reliable for BOTH schemes across all 12 replicates
tested** — the real "safe W" threshold for this data/point sits somewhere between
20,000 and 40,000, not at the current 80,000 production default. This is a useful
side finding but not this task's main question.

## 4. Draw-generation wall time

Trivially small for both schemes, confirming no surprise: mean `t_draw` = 12.9ms
(plain) vs 24.5ms (stratified) across all 960 sweep evaluations, max 194ms (plain) /
95ms (stratified) — both a small fraction of context-build time (`t_ctx` mean
6.8s–17.7s depending on W) and negligible next to a real inner solve (seconds to
tens of seconds). Stratification's O(D·W) independent-permutation construction adds
no meaningful cost at this scale.

## 5. Follow-up: does screen-passing under stratification mean the real solve succeeds, and is the divergence unbiased?

Per the task's explicit follow-up instruction, ran targeted real inner-solve checks
(`full_aod_diag/d4_exact/c10_stratmarg_followup_w20000.jl`) at W=20,000, rep=12 — a
replicate the screen found **plain-infeasible (2 zero-win pairs) but
stratified-feasible (0 zero-win pairs)** for both points, i.e. exactly the kind of
case where stratification changed the screen's verdict:

| point | scheme | screen n_zero_win_pairs | REAL inner_status | Delta_dual | kappa |
|---|---|---|---|---|---|
| calibration | plain | 2 (infeasible) | **-300** | NaN | NaN |
| calibration | **stratified** | 0 (feasible) | **-300** | NaN | NaN |
| upper_candidate | plain | 2 (infeasible) | **-300** | NaN | NaN |
| upper_candidate | **stratified** | 0 (feasible) | **0 (solved)** | **1.8595** | 0.07274 |

**W=80,000 plain benchmark** (always run, same points): calibration `Delta_dual
=0.00255`, upper_candidate `Delta_dual=1.0868` (kappa=0.07274, matching the
upper-candidate headline value, as expected since gp is fixed at this point and
kappa is a deterministic function of gp alone).

**Two real findings, reported honestly:**

1. **Passing the exact zero-winner screen does not guarantee the real KNITRO inner
   solve succeeds** — exactly the "class 4" phenomenon `docs/fullA_D20_infeasibility_screening_report.md`
   already documented (structural feasibility ≠ numerical solvability). Stratification
   fixed the calibration point's zero-winner problem at this replicate, but the real
   solve still returned `-300` for an unrelated numerical reason the screen cannot
   catch or fix.
2. **Where it DID work** (upper_candidate), the recovered divergence is **substantially
   biased relative to the W=80,000 benchmark**: `Delta_dual=1.86` at W=20,000 vs
   `1.09` at W=80,000 — a ~71% relative gap. This confirms the task's flagged concern
   directly: fewer total draws can bias the divergence estimate materially even at a
   point the exact screen certifies feasible. Smaller W is not a free lunch even in
   the cases where stratification makes it technically solvable.

## 6. Recommendation

**Do not adopt stratified-marginal sampling as a production draw-generation default
on the strength of this result.** It gives a real, measured, replicated reduction in
zero-winner-pair incidence (~3-4x fewer at W=8,000/20,000) but:

- never reaches the 100% reliability a production default needs (still fails 80-87%
  of the time at W=8,000, ~7% at W=20,000);
- does not reliably fix the real inner-solve failure mode even when it does fix the
  structural zero-winner problem (§5.1);
- and even in the one case it made W=20,000 solvable, the resulting divergence
  estimate carries a large (~71%) bias relative to the current W=80,000 default
  (§5.2) — so it would not be a safe drop-in replacement for cost savings even there.

**What this result IS useful for**: confirming that the W=8,000/W=20,000 failure
really is dominated by per-country marginal tail coverage (stratification measurably
helps, as hypothesized) rather than some other draw-design property, and pinning
down that the genuine reliability threshold for this data/point is between 20,000
and 40,000, not 80,000 — a fact worth knowing independent of this task's
stratification question, should a future task want to explore reducing the
production W below 80,000 by other means (the current default is not shown here to
be minimal, only that stratification alone isn't a safe way to get there).

## 7. Files

New, all under `full_aod_diag/d4_exact/`, all additive (no production file touched):
- `c10_stratmarg_draws.jl` — `stratified_marginal_U`, the per-country LHS generator.
- `c10_stratmarg_screen_sweep.jl` — main test-matrix driver (§2-4).
- `c10_stratmarg_followup_w20000.jl` — targeted rep=12 real-inner-solve follow-up (§5).

Raw logs/CSVs:
`results/fullA_d4/97cdd80/c10_stratmarg_screen_sweep_20260719_130454/` (main sweep),
`results/fullA_d4/9700b4e/c10_stratmarg_followup_w20000_20260719_133400/` (follow-up).
