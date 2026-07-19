# Full-A D=20 real-data A-block `L_fix` gradient validation (Continuation 9, Phase 6)

Direct follow-up to `docs/fullA_nested_w_continuation_c8.md` sec 4's open finding
(D=4, synthetic data): the fast `lfix_composite`/`composite_gradient_at_fast`
gradient's FULL vector agrees excellently with a slow finite-difference
reference (cosine >=0.9998), but the A-block-ONLY sub-vector shows real,
unresolved sign disagreement (cosine 0.17-0.59, elementwise sign agreement
73-93%). That finding was flagged as a scientific gate: "do not launch the
final ten profile points until the A-block derivative is judged usable." This
had never been tested at D=20 or on real data before this task. Measured on
`demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, commit **`7783ad3`** (branch `c9-phase6-gradval`, forked
from `diag/fullA-d4-exact`). Context via `context_real_d20.jl`'s
`d20_real_setup(W=80000)` (France focal, natural-theta calibration — same
context as `docs/fullA_D20_W80k_microbenchmark.md`). Harness:
`full_aod_diag/d4_exact/c9_phase6_gradval_d20.jl` (main run) +
`c9_phase6_bandwidth_followup.jl` (follow-up, see sec 5). Raw logs + CSVs:
`results/fullA_d4/7783ad3/c9_phase6_gradval_d20_20260719_063832/`.

## 0. Method summary

- **3 points**, matching `docs/fullA_D20_W80k_microbenchmark.md`'s own Point
  1/3/4 exactly (same context, same offsets): calibration (natural theta,
  gamma'=0.987762), upper branch (gamma'=gp0*1.01=0.997640, A at natural
  theta), lower branch (gamma'=gp0*0.99=0.977884, A at natural theta). All 3
  reconfirmed feasible on the first try (`inner_status=0`), matching the W80k
  doc's own finding.
- **20 deterministic random "gravity-tangent A-only" directions** (seed
  `20260719`, unit vectors in the `D^2-1=399`-dim pivot-reduced A-block,
  gravity-exact by construction of `pivot_expand` — same guarantee the W80k
  doc's own Point 2 established, reused not re-derived), the SAME 20
  directions reused across all 3 points for direct comparability.
- **Fast gradient**: `composite_gradient_at_fast(...; h_mode=:cached,
  bandwidth_cache=<fresh Dict>, threaded=true)`, per
  `docs/fullA_D20_bandwidth_optimization_report.md` sec 4's explicit
  production recommendation (NOT `:quantile` — numerically fine but little
  wall-clock win; NOT `:fixed` — no per-coordinate adaptivity). A FRESH
  `Dict` per point (all-miss on first use) matches the realistic "first outer
  iterate at a genuinely new point" case. Before running: verified the
  Dict-write thread-safety fix (`ReentrantLock`, commit `a6ed25e`) and the
  `needs_outer_moment_jacobian=false` memory-safety default are both present
  in this worktree by grepping the actual source (not trusting a comment) —
  script aborts if either check fails. For each point, ONE full 400-coordinate
  gradient call gives every A-block coordinate's partial derivative at once;
  each direction's predicted directional derivative is then just
  `dot(g_fast[2:end], v)` — no extra fast-gradient calls needed per direction.
- **Slow reference**: `evaluate_fullA` (plain, fully re-solved oracle) at
  `w0 +/- h*v`, for `h in {0.02, 0.005}`, giving central/left/right secants of
  `Delta_dual`. `Delta(w0)` computed once per point and reused for both
  one-sided secants across all 20 directions/2 bandwidths (cost management).
- **Winner-switch counts / CC-weighted switch mass**: reused existing
  `winners.jl::compute_winners` + `winner_switching.jl::switch_stats`
  (compares two `winner` matrices under common draws) — the correct existing
  sibling tool for an ARBITRARY multi-coordinate direction (`count_winner_flips`
  itself only covers single/double-origin PER-COORDINATE perturbations, so it
  does not apply directly to a random direction touching all 399 coordinates
  at once; this is a deliberate, documented substitution, not a
  reimplementation of the same logic).
- **Steepest-descent-direction sanity probe** (additional, beyond the task's
  literal spec): at each point, `v_sd = -g_fast[A-block]/||.||` (the ACTUAL
  direction an optimizer would take), tested at both bandwidths — the single
  most decision-relevant check, since this is what a real outer-loop line
  search actually uses, as opposed to an arbitrary random direction.
- **Memory safety**: VmHWM after `d20_real_setup` = 2.62 GB (memcheck passed,
  threshold 5GB); final VmHWM at end of the full run (all 3 points, ~255
  solves) = **5.26 GB** — safe, consistent with prior D=20 benchmarks in this
  investigation. Total wall clock: ~30 minutes (55s setup + ~29min for the 3
  points' direction loops, slower than this task's initial 10-16 minute
  estimate because the estimate did not account for `compute_winners`'
  O(W*D^2) dense-price-array cost, paid twice per direction/bandwidth for the
  switch-mass diagnostic — reported honestly, not hidden).

## 1. Headline: A-only cosine/sign-agreement statistics per point

**Do not rely on the full-vector cosine dominated by gamma'_focal (not
computed at all in this task — the task explicitly asked for the A-only
statistic, so no full-vector number is reported here; see the D4 nested-W
doc for that comparison).**

| point | h | direction-sample cosine(pred,central) | sign agreement | mean abs err | median abs err | Delta0 |
|---|---|---|---|---|---|---|
| calibration | 0.02 | **0.7296** | 17/20 (85.0%) | 4.23e-4 | 4.04e-4 | 0.002591 |
| calibration | 0.005 | **0.4148** | 14/20 (70.0%) | 6.54e-4 | 5.98e-4 | 0.002591 |
| upper (gp0\*1.01) | 0.02 | **0.9544** | 19/20 (95.0%) | 8.79e-3 | 8.33e-3 | 0.230884 |
| upper (gp0\*1.01) | 0.005 | **0.9394** | 18/20 (90.0%) | 1.14e-2 | 1.00e-2 | 0.230884 |
| lower (gp0\*0.99) | 0.02 | **0.9955** | 19/20 (95.0%) | 1.27e-3 | 1.17e-3 | 0.129645 |
| lower (gp0\*0.99) | 0.005 | **0.9885** | 18/20 (90.0%) | 2.01e-3 | 1.66e-3 | 0.129645 |

**The finding is real but reverses direction from what was expected going in.**
At D=4, the A-block sub-gradient disagreement was uniform-ish across the
handful of points tested (cosine 0.17-0.59 across upper/lower, W=8000/80000).
At D=20/real-data, the picture is sharply point-dependent:

- **Calibration is the poor case** — cosine 0.41-0.73, sign agreement
  70-85%, WORSE at the smaller bandwidth (h=0.005) than the larger one
  (h=0.02). This is the point where `Delta_dual` is smallest (0.0026) —
  consistent with `results/fullA_d4/bb74649/gamma_profile_nonmonotonicity_report.md`'s
  finding that the benchmark Fréchet value sits almost exactly at the
  gamma-profile's interior minimum, i.e. calibration sits in a genuinely
  near-flat region of the A-block's effect on `Delta_dual`, where small
  numerical differences between two different gradient ESTIMATION methods
  can easily disagree in sign without either being "wrong" (same explanation
  the D4 doc offered, now directly corroborated by a THIRD, independent piece
  of evidence: the point with the smallest `Delta0` is exactly the point with
  the worst A-block agreement).
- **Upper and lower branches are GOOD** — cosine 0.94-0.996, sign agreement
  90-95%, at BOTH bandwidths. These are exactly the kind of points a real
  outer-loop / profile-continuation run spends most of its time at (away from
  the near-degenerate calibration point), and they are also the two points
  `docs/fullA_D20_W80k_microbenchmark.md` used as its own Point 3/Point 4.

**Consistent secondary pattern**: at every point, sign agreement is slightly
*worse* at the smaller bandwidth h=0.005 than at h=0.02 (calibration:
85%->70%; upper: 95%->90%; lower: 95%->90%). This replicates
`docs/fullA_D20_bandwidth_optimization_report.md` sec 2's D=4 finding that a
naive small-h FD estimate is noisier near winner-boundary kinks (fewer draws'
worth of switching mass to average over) — smaller h is not "more accurate"
here, it is noisier, because the true function has kinks.

## 2. Steepest-descent-direction probe: the decision-relevant check

Rather than a random direction, this additionally tests the ACTUAL direction
an outer-loop line search would take: `v_sd = -g_fast[A-block]/norm(.)`.

| point | h | pred_slope | central secant | right secant | left secant | actually decreases? |
|---|---|---|---|---|---|---|
| calibration | 0.02 | -0.014300 | -0.014141 | -0.010282 | -0.018000 | **true** |
| calibration | 0.005 | -0.014300 | -0.014245 | -0.014921 | -0.013570 | **true** |
| upper | 0.02 | -0.169003 | -0.228466 | -0.183119 | -0.273813 | **true** |
| upper | 0.005 | -0.169003 | -0.207704 | -0.259945 | -0.155462 | **true** |
| lower | 0.02 | -0.267636 | -0.275168 | -0.270313 | -0.280024 | **true** |
| lower | 0.005 | -0.267636 | -0.267904 | -0.260935 | -0.274872 | **true** |

**Every single row: `actual_decreases=true`, and the predicted slope matches
the true central secant closely** (calibration: -0.0143 predicted vs -0.0141
to -0.0142 actual, essentially exact; lower: -0.268 vs -0.267 to -0.275, also
tight; upper: -0.169 vs -0.208 to -0.228, same sign and order of magnitude but
a genuine ~20-35% magnitude gap). This is the single most important result in
this report: **even at the calibration point, where the RANDOM-direction
statistics look bad, the AGGREGATE steepest-descent direction the fast
gradient actually produces is directionally correct and quantitatively close**.
This resolves the apparent tension between sec 1's "poor at calibration" and
this investigation's own repeated observation ("hasn't visibly broken any
outer solve so far") — an outer-loop optimizer uses the FULL 399-dimensional
gradient vector as one direction, not 20 independent random probes, and
averaging over 399 coordinates cancels much of the same per-coordinate sign
noise that shows up starkly when looking at a handful of individual random
directions.

## 3. Winner-switch counts / weighted switch mass

Reused `compute_winners` + `switch_stats` (CC-weighted mass uses
`base0.m_star`, the base-point LFD weights).

| point | h | mean switches (+/-) | mean CC-weighted mass (+/-) | mean fraction of W |
|---|---|---|---|---|
| calibration | 0.02 | 231.7 / 234.2 | 0.00289 / 0.00292 | 0.29% |
| calibration | 0.005 | 63.9 / 62.9 | 0.00080 / 0.00078 | 0.08% |
| upper | 0.02 | 231.7 / 234.2 | 0.00328 / 0.00345 | 0.29% |
| upper | 0.005 | 63.9 / 62.9 | 0.00088 / 0.00089 | 0.08% |
| lower | 0.02 | 231.7 / 234.2 | 0.00264 / 0.00268 | 0.29% |
| lower | 0.005 | 63.9 / 62.9 | 0.00075 / 0.00072 | 0.08% |

**Switch counts are IDENTICAL across the 3 points at a given h/direction** —
this is expected, not a bug: A_od is held fixed at natural theta across all 3
points (only gamma'_focal differs), and `compute_winners`' underlying price
formula (`factual_prices`) depends only on the A-block, not on gamma'_focal —
so a perturbation of the SAME A-block direction produces the SAME winner
matrix regardless of which point's gamma' is in effect. This is a useful
internal cross-check that the harness is behaving exactly as the mechanics of
the model predict, not a redundant/wasted measurement. Switch mass stays
small in absolute terms (0.08-0.29% of draws) at both bandwidths — well below
the `[0.3%, 3%]` target band `select_bandwidth` aims for, consistent with
`docs/fullA_D20_bandwidth_optimization_report.md` sec 3C's finding that
`select_bandwidth`'s own floor is binding for many D=20 coordinates (a
different but related manifestation of the same "not much switching mass per
probe at this W/D" fact).

## 4. Is the disagreement explained by proximity to the profile minimum?

Yes, directly supported by this run's own data: `Delta0` at calibration
(0.0026) is 50-90x smaller than at upper/lower (0.231, 0.130) — i.e.
calibration sits in the flattest, most near-degenerate region of the 3 points
tested, exactly where two independent numerical-derivative ESTIMATORS (fast
incremental-winner FD vs. slow fully-re-solved secant) are most likely to
disagree on sign without either being wrong, since the true function's local
slope is itself close to a sign change / near-zero in many A-block directions
there. This is the SAME explanation the D4 nested-W doc offered ("A-block
being a comparatively flat/near-degenerate direction... consistent with... a
flat A-valley near the profile minimum") — now corroborated by a genuinely
independent D=20/real-data measurement rather than merely repeated.

## 5. Follow-up: does a different `L_fix` bandwidth help at the poor
   (calibration) point?

Per the standing brief's "pursue ONE follow-up if the base result is clearly
poor" allowance — picked the cheapest option (a different bandwidth for the
FAST gradient), since it reuses the SAME 20 directions and the
ALREADY-COMPUTED optimized-value central secants from sec 1 (no new
`evaluate_fullA` solves needed, only 3 extra `composite_gradient_at_fast`
calls). Script: `c9_phase6_bandwidth_followup.jl`.

| fast-gradient h_mode | vs secant h=0.02: cos / sign | vs secant h=0.005: cos / sign | \|\|g_A\|\| |
|---|---|---|---|
| `:cached`/adaptive (this report's baseline) | 0.7296 / 0.85 | 0.4148 / 0.70 | 0.01430 |
| `:fixed`, h0=0.05 | 0.7440 / 0.85 | 0.4621 / 0.70 | 0.01449 |
| `:fixed`, h0=0.01 | 0.7963 / 0.90 | 0.5563 / 0.75 | 0.01439 |
| `:fixed`, h0=0.005 | **0.8382 / 0.95** | 0.5278 / 0.70 | 0.01557 |

**A genuine, modest improvement**: a SMALLER fixed bandwidth for the fast
gradient itself (h0=0.005, matching the smaller secant probe's own h) improves
cosine from 0.73 to 0.84 and sign agreement from 85% to 95% against the
h=0.02 secant reference — the opposite bandwidth direction from what
`docs/fullA_D20_bandwidth_optimization_report.md` sec 4 recommended for
production speed (that report favored `:cached`/adaptive for wall-clock
reasons, not accuracy). This is a real but bounded fix: even the best
`h_mode` tested here (`:fixed h0=0.005`) still only reaches 0.53-0.56 cosine
against the SMALLER (h=0.005) secant reference — the calibration point's
near-flat-valley disagreement is reduced, not eliminated, by bandwidth
tuning alone. Not pursued further (a full re-tuning of `select_bandwidth`'s
target-mass band for D=20, as `docs/fullA_D20_bandwidth_optimization_report.md`
sec 5 already flagged as a separate open item, is out of this task's scope).

## 6. Go/no-go read

**GO for Phase 8, with one explicit caveat carried forward.**

- The single most decision-relevant test (sec 2: does moving along the fast
  gradient's own predicted A-block descent direction actually move `Delta_dual`
  the way it predicts) passes at **all 3 points, both bandwidths, 6/6** — this
  is the property an outer-loop optimizer or profile-continuation driver
  actually depends on, and it holds up at real D=20 scale exactly as it has
  held up (with zero observed divergent outer solves) throughout this whole
  investigation.
- The per-random-direction A-only cosine/sign statistics (sec 1) are GOOD
  (cosine 0.94-0.996, sign 90-95%) at the upper and lower branch points — the
  two points structurally most similar to where a real profile-continuation
  pilot spends its time (away from the near-degenerate calibration point).
- The one genuinely poor result is AT the calibration point specifically
  (cosine 0.41-0.73, sign 70-85%), and it is well-explained (sec 4) as a
  near-flat-valley numerical-estimator disagreement, not a sign the gradient
  is systematically wrong — the same explanation this investigation already
  offered at D=4, now independently corroborated.
- **Caveat to carry into Phase 8**: do not trust an individual A-block
  coordinate's sign in isolation near a near-degenerate point like calibration
  (`Delta0` close to zero, or more generally near a KKT/profile-minimum-like
  region) — use the FULL aggregate gradient direction (as any quasi-Newton
  outer solve already does) rather than reading off single coordinates, and
  if a future workflow needs single-coordinate-level A-block conclusions near
  such a point specifically, prefer the optimized-value secant (slow,
  trusted) over the fast gradient for that specific diagnostic, or use
  `h_mode=:fixed` with a smaller h0 per sec 5's bounded improvement.

## Files

New this session, all under `full_aod_diag/d4_exact/`:
`c9_phase6_gradval_d20.jl` (main run), `c9_phase6_bandwidth_followup.jl`
(sec 5 follow-up). Raw logs + CSVs:
`results/fullA_d4/7783ad3/c9_phase6_gradval_d20_20260719_063832/`
(`direction_secants.csv`, `steepest_descent_probe.csv`, `summary.csv`,
`harness_log.txt`).
