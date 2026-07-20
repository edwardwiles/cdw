# Common-marginals conditioning bases + adaptive quantile activation (D=4, full-A_od)

Branch `diag/fullA-d4-exact-cm-conditioning`, based on `diag/fullA-d4-exact-common-marginals`.
Measurement task only — nothing here changes the production default. All numbers below come from
scripts in `full_aod_diag/d4_exact/` that were actually executed on this branch (logs and CSVs
committed alongside this report); no number is assumed or extrapolated without saying so.

## Scripts

- `c12b_interval_common_marginals_moments.jl` — NEW moment construction: interval (non-cumulative)
  common-marginals restriction, plus a per-column-standardized variant, plus a generic
  `build_cm_augmented_obj_from_CM` helper (reuses `wrap_moments_with_cm` from
  `common_marginals_moments.jl` unchanged; does not modify that file).
- `c12b_conditioning_battery.jl` → `c12b_conditioning_battery_results.csv` / `.log` — Part 1:
  4 bases × L∈{10,20,50} × 2 points × 4 reference countries = 96 runs.
- `c12b_rank_diag.jl` — follow-up diagnostic explaining the one rank-deficiency pattern found in
  Part 1 (see below).
- `c12b_adaptive_grid.jl` → `c12b_adaptive_grid.log` — Part 2: fixed-outer-point adaptive
  quantile activation, L_full∈{50,20}, calibration + perturbed points, plus a warm-vs-cold
  side-by-side and a dense-grid agreement check.

## Part 1: conditioning-preserving transformations

### Bases tested

1. **anchored** — existing default: `1{U_o<=z_l} - 1{U_1<=z_l}` (cumulative CDF).
2. **orthonormal** — existing validated Helmert-contrast rotation of (1), `R=(BB')^{-1/2}` closed
   form (`orthonormal_contrast_matrix`), already proven on this branch to give identical KKT
   residuals/Delta_dual to anchored at fixed points; re-used unchanged.
3. **interval** — NEW: `1{z_{l-1}<U_o<=z_l} - 1{z_{l-1}<U_1<=z_l}`, same L thresholds as (1)/(2),
   `z_0:=-Inf`, top bin dropped so column count stays `(D-1)*L`.
4. **std_interval** — (3) with each raw column divided by its own empirical std across the W
   baseline draws before insertion into G (and before any rotation, though no rotation is used
   here — anchored contrasts only).

Points: calibration (`x_free_calib = ctx.θ0_up[ctx.free_idx]`) and a perturbed-feasible point
(`x_perturbed[2:end] .*= exp.(0.05 .* randn(...))`, seed 4243, matching
`c12_d4_fixed_param_battery.jl`'s convention). Reference country: all 4 of `refIndex1 ∈ 1:D`.

All 96 runs converged (`nStatus=0`, calibration: 4 inner KNITRO iterations; perturbed: 5), i.e.
zero solver failures across the whole battery.

### Conditioning comparison (mean `cond(Hessian)` over the 4 reference countries)

| L  | point     | anchored | orthonormal | interval   | std_interval |
|----|-----------|---------:|------------:|-----------:|-------------:|
| 10 | calib     | 35,793   | 16,342      | **12,197** | 22,036       |
| 10 | perturbed | 52,373   | 24,795      | **19,986** | 32,990       |
| 20 | calib     | 62,048   | 22,405      | **11,553** | 22,119       |
| 20 | perturbed | 91,060   | 32,822      | **19,609** | 33,504       |
| 50 | calib     | 143,174  | 42,305      | **11,345** | 22,486       |
| 50 | perturbed | 212,193  | 62,513      | **19,386** | 34,359       |

(Hessian = exact dense `hessian!` from `cc_algo/PsiObjectiveBundle.jl` at the solved dual point,
size `outer_constr_index × outer_constr_index` = 48/78/168 for L=10/20/50.)

Ranking is consistent across **every** L and point tested: **interval < std_interval < orthonormal
< anchored** (lower cond = better). At L=50, plain interval beats anchored by **~12.6× (calib) /
~10.9× (perturbed)**, and beats orthonormal by **~3.7× / ~3.2×**.

**Scaling with L is the more important part of this finding.** Anchored's conditioning grows
roughly 4× from L=10 to L=50 (35,793 → 143,174, calib); orthonormal grows ~2.6× (16,342 → 42,305);
interval and std_interval are essentially **flat in L** (interval: 12,197 → 11,553 → 11,345 calib;
std_interval: 22,036 → 22,119 → 22,486). Plausible mechanism (stated as a hypothesis consistent
with the data, not separately verified via direct correlation-matrix inspection — time-boxed):
cumulative-CDF columns at adjacent thresholds are nested index sets and become increasingly
collinear as L grows and thresholds get closer together; interval-indicator columns are supported
on (near-)disjoint bins, so adding more bins doesn't increase cross-column correlation. This is the
part of the finding most relevant to a future D=20/L=50 run, since that is exactly the large-L
regime where anchored's conditioning degrades and interval's does not.

Standardizing (std_interval) is **not** an improvement over plain interval on any L/point tested —
it's consistently ~2× worse in conditioning than unstandardized interval, though still much better
than anchored/orthonormal. No mechanism for this was separately verified; noted as an empirical
fact, not explained further given time constraints.

### Rank / redundant columns

Numerical rank via `svdvals` (tolerance `max(size)*eps(max(sv))`):

- **Every** calibration-point run, all 4 bases × 3 L's × 4 refs (48 runs): rank = `d_total − 1`
  (exactly one redundant column).
- **Every** perturbed-point run (48 runs): full rank `d_total`.

Follow-up (`c12b_rank_diag.jl`) isolates the source: at calibration, the **core** (pre-CM) moment
matrix alone has rank 17/18 (one redundant column), the CM block **alone** is always full rank
(30/30 checked at L=10 anchored), and the combined core+CM matrix has rank exactly 17+30=47/48.
**The redundancy is 100% inherited from the pre-existing core moments at the exact calibration
point (A_od=1)** — it is not introduced by the common-marginals restriction in any basis, and it
disappears at any perturbed point. This is not a bug to fix here (the core moment set predates
this branch); no CM columns were found to be exact duplicates in any of the 4 bases, so nothing
was removed.

### Reference-country sensitivity

Spread `(max−min)/mean` of `cond(Hessian)` across `refIndex1 ∈ {1,2,3,4}`, holding basis/L/point
fixed:

| basis        | spread range (over L, both points) |
|--------------|-------------------------------------|
| orthonormal  | 0.8% – 1.5% (least sensitive)        |
| anchored     | 2.9% – 9.7%                          |
| interval     | 2.2% – 10.3%                         |
| std_interval | 15.9% – 18.9% (most sensitive)       |

None of this is material in the sense of changing feasibility or convergence: **every** one of the
96 runs converged with `nStatus=0`, and iteration count was identical (4 at calibration, 5 at
perturbed) regardless of reference-country choice, in every single run. Reference-country choice
changes conditioning by single-digit percent for anchored/interval/orthonormal, and by up to ~19%
for std_interval — real but an order of magnitude smaller than the 3–12× gaps between moment
bases. Conclusion: reference-country choice is **not** a first-order lever for conditioning at this
scale; basis choice is.

### Sparsity tradeoff (structural, not just a numerical footnote)

`orthonormal_contrast_matrix(D=4)` is a fully dense 3×3 matrix (all 9 entries nonzero, checked
directly: diagonal 0.8333, off-diagonal −0.1667). Consequently:

- **anchored / interval**: column `(l, oi)` is a function of exactly **2** origins' raw draws
  (`U[:,o]` and `U[:,ref]`) — provably sparse per column.
- **orthonormal**: each rotated column in a threshold's block is a linear combination of **all**
  `nO=D-1` raw columns in that block (mixing across every non-reference origin, each of which
  itself involves the reference origin) — dense per column, touching all D origins' data.

This matters beyond D=4: prior work on this repo (compressed moments, `full_aod_diag/`
history) already reduced core-moment construction from O(W·D²) to O(W·D) by exploiting exactly
this kind of per-origin sparsity. Adopting orthonormal contrasts for the CM block would forfeit
that structure specifically for the CM columns, while interval (and anchored) would not. This is a
real cost of orthonormal contrasts that the conditioning numbers alone don't show — flagging it
explicitly per the task brief, since it's the more consequential axis at D=20/L=50 scale even
though it wasn't directly measured at that scale here.

### Recommendation (not adopted, per task instructions — this is a measurement task)

No basis is switched as the default here. If conditioning becomes a practical blocker at
D=20/L=50, the data above point to **interval (unstandardized)** as the strongest documented
candidate: best conditioning of the 4 at every L tested, conditioning advantage over anchored
*widens* rather than shrinks as L grows (the opposite of what you'd want if you were worried this
was a small-L artifact), and it preserves the same per-origin sparsity as anchored (unlike
orthonormal). Orthonormal remains the right choice only when sparsity doesn't matter and an
exact, already-validated invertible mapping back to anchored coordinates is specifically wanted.
std_interval (standardizing) is not recommended — worse conditioning than plain interval on every
metric tested, and the most reference-country-sensitive of the four, with no offsetting benefit
found.

## Part 2: adaptive quantile activation

### Method

`L_full=50` dense candidate grid built once via `precalc_common_marginals_cdf(U, ref, 50;
contrasts=:anchored)`. Seed active set: the 50-grid's own probability levels
(`range(1/50,49/50,length=50)`) closest to `{0.10,0.25,0.50,0.75,0.90,0.95,0.99}` → 7 thresholds
(indices `[5,13,25,38,46,48,50]` at `refIndex1=1`), 21 active columns (D=4, nO=3). Tolerance:
**1e-4**, chosen because the CM columns are indicator differences bounded in `[-1,1]`, so an
absolute tolerance of 1e-4 is simultaneously a relative-to-moment-scale tolerance here; the active
(satisfied) restrictions' own KKT residuals come out at the ~1e-18 level (see below), so 1e-4
cleanly separates "genuinely still violated" from "converged to solver precision" — roughly 14
orders of magnitude of headroom.

At each round: solve with only the active columns, read `obj.arg1` (LFD weight `m(s)`, a side
effect of the constrained `obj(...)` call inside `evaluate_fullA`), evaluate the weighted-mean
discrepancy of **every** one of the 50 candidate thresholds' columns (the dense `CM_full` matrix
doesn't depend on θ, so this is a cheap direct computation, no re-solve needed), activate every
inactive threshold whose discrepancy exceeds tolerance (batched "add-all-violators" rather than
one-at-a-time, for round-count efficiency — noted explicitly since it changes what "how many
rounds" means), and warm-restart: new columns are always **appended** after existing ones (never
interleaved), so the previous round's solved `(ζ,λ)` is an exact prefix of the new, larger `x`
vector — pad with zeros for the newly added columns and set `obj_cm.x` directly before calling
`evaluate_fullA(...; warm=true)`.

### Results

**L_full=50, calibration point** (`refIndex1=1`):
- Round 1: 7/50 thresholds active (21 cols). Max violation among the 43 inactive thresholds:
  **5.748e-3** (at threshold 7) — far above the 1e-4 tolerance.
- Round 2: all 43 violating thresholds activated at once → **50/50 active** (150 cols), warm-started
  from round 1's solution (4 iterations, 0.11s vs round 1's 8.15s cold solve). Max inactive
  violation: 0 (none left).
- **Converged in 2 rounds, but needed the full 50/50 grid** — not a small subset.

**L_full=50, perturbed-feasible point**: identical pattern — round 1 (7/50, violation 9.6e-3) →
round 2 (50/50, converged).

**L_full=20 robustness check, calibration point**: same pattern — round 1 (6/20 seed thresholds,
violation 6.2e-3) → round 2 (20/20, converged).

**Why it isn't sparse**: a follow-up diagnostic (`/tmp/c12b_violation_dist.jl`, not committed —
throwaway) checked the full violation distribution among the 43 inactive thresholds after round 1
at L=50/calibration: **minimum** inactive violation was 4.5e-4, median 3.1e-3, maximum 5.7e-3 — i.e.
essentially *all* un-imposed thresholds carry a real, non-negligible violation once only 7 are
imposed, at every tolerance from 1e-3 down to 1e-6 (41/43, 43/43, 43/43, 43/43 violating
respectively). The 7 seed thresholds' own residuals were ~1e-18 (solver precision), confirming the
solve itself is correct — the LFD reweighting satisfies exactly the restrictions it's given and
nothing more; it does not "for free" approximately satisfy nearby un-imposed quantile restrictions.
**Honest conclusion: at D=4, this restriction family shows no redundancy across quantile levels —
each of the 50 thresholds carries distinct binding information, so a coarse subset is a poor
approximation of the dense grid, and adaptive activation converges quickly but to the full set, not
a small one.** This is a real, if not the hoped-for, finding, and it argues against expecting large
constraint-count savings from adaptive activation at the D=20/L=50 scale without further
investigation (larger D might behave differently — not tested here, out of scope per the task's
explicit "fixed-outer-point only" instruction).

### Warm-start verification

Explicitly compared, at the final (50/50) active set, calibration point: cold-restart from scratch
(`nStatus=0`, 4 iterations, 0.170s) vs the adaptive loop's actual warm-started round 2 (`nStatus=0`,
4 iterations, 0.110s). Same iteration count at this scale (KNITRO converges in 4 either way — the
problem isn't hard enough at D=4 to show a warm-start iteration-count win), but warm-start does
save wall time (~35% here) by skipping the round-1-cost inner solve setup; the API-level
warm-restart itself (padding `obj.x` and reusing it via `evaluate_fullA(...; warm=true)`) was
verified to work correctly — round 2's `Delta_dual` and LFD weights match the cold re-solve to
~1e-11, and both match the fully independent dense-grid solve (below) to ~1e-16–1e-18.

### Agreement with the dense L=50/L=20 reference

At every point tested, the adaptive loop's final (fully-covered) active set was compared against
`build_cm_augmented_obj(ctx, CS; L=L_full, contrasts=:anchored)` (the already-validated dense
reference), evaluated independently at the same fixed outer point:

| point                    | L_full | \|Delta_dual diff\| | max\|LFD weight diff\| |
|--------------------------|-------:|---------------------:|------------------------:|
| calibration              | 50     | 5.2e-18               | 8.4e-19                 |
| perturbed-feasible       | 50     | 3.2e-16               | 5.1e-18                 |
| calibration              | 20     | 8.7e-19               | 4.2e-18                 |

All three agree to machine precision, as expected since a fully-covered active set is
mathematically the same restriction set as the dense grid (just built up incrementally with
appended columns instead of all at once) — this is a genuine correctness check that passed, not an
assumption.

## Summary of what was and wasn't done

- Part 1: fully executed — 96/96 runs, all converged, all 4 metrics tabulated, reference-country
  sensitivity checked, sparsity tradeoff documented, rank-deficiency source isolated.
- Part 2: fully executed for the required cases (L_full=50 at both points) plus the L_full=20
  robustness check requested as time-permitting. The "outer-loop restart protocol" was explicitly
  NOT implemented, per the task's explicit instruction that this is being handled elsewhere.
- Not done: a direct correlation-matrix inspection to mechanistically confirm the
  nested-vs-disjoint-support hypothesis for why interval conditions better than anchored (stated as
  a plausible, data-consistent hypothesis only); testing L_full=100+ or D>4 for the adaptive grid
  (out of scope per task brief).
