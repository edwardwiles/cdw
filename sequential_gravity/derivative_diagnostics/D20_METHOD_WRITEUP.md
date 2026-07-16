# Method used to compute the D=20 real-data sequential/profiled bounds

This is a clean statement of the exact method behind the D=20, W=80,000, upper-bound
results (delta = 0.1, 1.0, 2.0, 5.0) in
`sequential_gravity/batch_out_realD20_W80000_fixeddualfdfull_scaled05/`. It intentionally
does not repeat the debugging history (see `full_d2_correction_report.md` for that) --
this document states what the method IS, for someone who wants to use or cite it without
re-deriving it.

**CORRECTION (found after this document was first drafted)**: every mention below of
`scaling_power=0.5` is WRONG for these specific saved results. `run_one_bound` (the batch
loop that actually produced them) never threaded the `scaling_power` keyword through to
`outer_solve_nested_cached` at all -- only `use_var_scaling` was wired up. The `SCALING_POWER`
env var used at launch time was silently ignored, and the run used the function's own
default, **scaling_power=1.0** (full magnitude-matching), not 0.5. This has been fixed in
`run_profiled_production.jl` for future runs. The RESULTS in this document remain valid
and independently verified (see `verify_d20_deltagrid.jl` / section 5) -- only the
"scaling_power=0.5" label attached to them throughout is incorrect; read it as 1.0
everywhere it appears below. This also means the D=4 finding that "scaling_power=1.0
overcorrects and converges worse than 0.5" (`full_d2_correction_report.md` section 7.1)
did NOT reproduce at D=20 -- the D=20 run used power=1.0 the whole time and converged to a
clean, verified, genuinely-improving result. Whether 0.5 or another value does even better
at D=20 has not actually been tested yet.

## 1. The economic/statistical object being computed

For a single "focal" destination in a Ricardian (CDW-style) trade model, we compute
Christensen-Connault (2023) distribution-free bounds on the gains-from-trade statistic
kappa = 1 - (gamma'_focal)^(sigma/(sigma-1)), where gamma'_focal is a counterfactual
(autarky-comparison) price-index object. The bound is obtained by extremizing gamma'_focal
over a set of "nearby" data-generating processes -- reweightings of the observed Monte
Carlo draws, and (in the "full-A" extension used here) alternative values of the focal
destination's origin-competitiveness parameters A_{o,focal} -- subject to a divergence
budget: the reweighting/reparameterization must stay within a hybrid-divergence distance
delta of the calibrated (Frechet) baseline. The upper bound minimizes gamma'_focal (hence
maximizes kappa); this document's runs are all the upper bound.

## 2. The sequential-linearization ("profiled") method

Rather than optimizing over the full D x D matrix of origin-competitiveness parameters
(the "full-A_od" method: D^2 free parameters, expensive and separately ill-conditioned),
this method allows only the FOCAL destination's own column, A_{.,focal} (D free
parameters -- one per origin, all destined to the focal destination), to move. The other
(D-1) "omitted" destinations' implied fundamentals are backed out from their OBSERVED
trade data by inverting the model's destination-share equations, given the CURRENT
divergence-minimizing reweighting p. A gravity-consistency restriction -- that the
resulting (origin, destination) log-competitiveness matrix, after two-way (origin and
destination) demeaning, is uncorrelated with the demeaned log trade-cost matrix, the
standard cross-equation gravity identification condition -- is checked across the FULL D x
D matrix (not just the focal column) and enforced via an iteratively-updated LINEARIZED
moment (an influence-function-based first-order approximation around the current
reweighting/omitted-destination inversion), folded into the Christensen-Connault inner
divergence-minimization problem as an additional moment.

The inner problem therefore matches D+2 moments: the D focal bilateral trade shares, the
target counterfactual gamma'_focal, and the current linearized gravity-consistency moment.
At each outer iterate (a candidate A_{.,focal}), this D+2-moment inner problem is solved,
the omitted destinations are re-inverted, the gravity linearization is refreshed, and the
process iterates (a fixed number of times or until the gravity residual is within
tolerance) before the outer search evaluates the next candidate A_{.,focal}.

Code: `sequential_gravity/run_profiled_production.jl::seq_gravcol` implements one full
evaluation of this inner loop (given theta = [mu, sigma, gamma'_focal, A_{.,focal}]);
`profiled_gravity.jl` implements the destination inversion, gravity residual, and
influence-function linearization primitives it calls.

## 3. Three methodological corrections layered on top of the baseline method

The baseline sequential method (as it existed before this work) searches over
(gamma'_focal, A_{.,focal}) via a single KNITRO SQP/interior-point outer solve, subject to
the divergence-budget constraint. Three corrections were needed to make that search
actually explore A_{.,focal} and find genuine improvements over holding it fixed at the
calibrated value A*:

### 3a. Corrected winner-boundary (Dirac) outer-loop gradient

The trade-share moments depend on a HARD argmax winner rule (which origin wins each Monte
Carlo draw for the focal destination). The outer gradient of the divergence-budget
constraint with respect to A_{.,focal} was previously computed via plain ForwardDiff
through this hard argmax -- which is mathematically WRONG whenever perturbing A_{.,focal}
causes some draws' winning origin to change: it captures only the "intensive margin"
(smooth, within-current-winner-regime) sensitivity and provably omits the first-order
"extensive margin" (winner-boundary crossing) term. This was confirmed both analytically
(against the exact Frechet-benchmark moment Jacobian) and numerically (against fully
re-solved profile finite differences): the naive gradient is off by 60-230% wherever the
boundary term matters, vs 0.05-2% for the corrected method.

**The fix**: a full-(D+2)-moment "fixed-dual" finite-difference gradient. At the current
outer iterate, the inner dual variables (zeta, lambda) are held fixed at their just-solved
values, and the EXACT dual criterion (with hard winners recomputed exactly at each
perturbed A_{.,focal}, and using ALL D+2 moments including the gravity-linearized one and
its own dual multiplier -- an earlier version of this fix incorrectly dropped that
multiplier from a REDUCED (D+1)-moment approximation, which is not generally valid since
the divergence's convex conjugate is nonlinear in the full moment vector) is
central-difference differentiated with respect to each A_{o,focal} coordinate. No inner
KNITRO re-solve is needed per finite-difference evaluation -- only cheap moment/conjugate
recomputations. gamma'_focal's own gradient component continues to use plain AD, which is
exact for that coordinate (the winner rule and the price-index moment have no dependence
on gamma'_focal).

Code: `derivative_diagnostics/full_fixed_dual_criterion.jl` (the frozen-gravity full-(D+2)
moments function), `fixed_dual_criterion.jl`/`fixed_dual_fd.jl` (the dual criterion and its
finite-difference gradient, generic in the moment count), wired into production via
`gradient_method = :fixed_dual_fd_full` (`full_gradient_method_wiring.jl`).

### 3b. KNITRO variable scaling

At D=20 real-data scale, gamma'_focal's own sensitivity to the divergence-budget
constraint is ~1e8-1e9, while the entire A_{.,focal} block's sensitivity is only ~10-11,500
-- a ratio of 10^4-10^5 (vs ~160x at D=4, where the search already worked without any
scaling correction). Left uncorrected, the outer KNITRO search -- even with the corrected
gradient from 3a -- essentially never moved A_{.,focal} away from its calibrated value at
D=20 (confirmed: the resulting A_{.,focal} was bit-identical to A* under both the old and
corrected gradient methods).

**The fix**: KNITRO's native per-variable scaling (`KN_set_var_scalings_all`). A probe
gradient evaluation at the starting point gives a representative sensitivity magnitude per
free coordinate; gamma'_focal's own scale is left at 1.0 (preserving the outer objective's
natural units, since the objective's gradient is a hard-coded +-1 at gamma'_focal only and
scaling it too would corrupt the KKT balance -- an error caught and fixed during this
work), and each A_{o,focal} coordinate is rescaled by
(|d(constraint)/d(gamma'_focal)| / |d(constraint)/d(A_{o,focal})|)^scaling_power, with
scaling_power=0.5 (a square-root compromise between no correction and full
magnitude-matching -- tested at both extremes; full matching overshoots and converges
worse even where it does help, see full_d2_correction_report.md section 7.1 for the
sensitivity of this choice).

This scaling is NOT a universal default -- it helps at D=20 (where the baseline search was
completely stuck) but actively hurts D=4 (where the unscaled search already explores A
fine) -- so it is applied only for the D=20 runs described here.

Code: `use_var_scaling`/`scaling_power` kwargs on `outer_solve_nested_cached`
(`run_profiled_production.jl`), `var_scales` kwarg on `cc_algo/outer_loop_cached.jl`.

### 3c. Feasible-only warm-starting across the delta grid

The outer KNITRO search's raw endpoint is not always gravity-feasible (production
separately tracks the best point verified gravity-feasible via an independent
`seq_gravcol` check, `best_theta`/`best_kappa`, alongside the raw KNITRO result). When
solving a grid of delta budgets sequentially (warm-starting each from the previous
budget's result, ascending), warm-starting from the RAW endpoint -- rather than the
verified-feasible one -- can chain a broken starting point through the rest of the grid:
this was observed directly (delta=2.0 and delta=5.0 both found ZERO feasible points after
~50 minutes of search each, having inherited delta=1.0's gravity-infeasible raw endpoint).

**The fix**: warm-start every subsequent delta budget from the previous budget's verified
best-feasible point, never the raw endpoint (falling back to the raw endpoint, with an
explicit warning, only if no feasible point was found at all).

## 4. Exact settings for these results

- D=20, real trade/tariff/GDP data ("Noah's D=20 dataset",
  `real_data/noah_D20/{countries,L,pi,tau}.csv`, loaded via `FAKEDATA=3`).
- W = 80,000 Monte Carlo draws.
- Upper bound (gamma'_focal minimized, kappa maximized), `find_smallest=true`.
- `gradient_method = :fixed_dual_fd_full` (section 3a).
- `use_var_scaling = true`, `scaling_power = 0.5` (section 3b).
- Delta grid {0.1, 1.0, 2.0, 5.0}, ascending, warm-started from the previous delta's
  verified-feasible endpoint (section 3c); delta=0.1 (the smallest, first in the grid) is
  cold-started from theta_r0 (A = A*, gamma'_focal = the Frechet-benchmark point estimate).
- KNITRO outer-loop options: `full_aod_diag/csw_outer_1000.opt` (maxit=1000,
  maxtime_real=10,000s, ftol=1e-15/ftol_iters=5).
- Destination inversion parallelized across the D-1=19 omitted destinations
  (`PARALLEL_INVERSION=true`, `julia -t 19`).
- Sequential loop's own convergence: gravity residual tolerance 5e-4 (R_mean), maxit=20
  sequential-linearization iterations per inner evaluation (production default) / maxit=100
  in the standalone post-hoc verification/audit calls (see
  `full_d2_correction_report.md` section 4, bug 5 -- 20 is sometimes an iteration-limit
  artifact for an audit call at an arbitrary theta, not for the production search itself,
  which iterates within each outer evaluation as needed up to its own maxit=20 cap using
  warm starts across evaluations).

Driver: `sequential_gravity/run_profiled_production.jl`'s standard batch loop, invoked via

    FAKEDATA=3 DVAL=20 WVAL=80000 BOUND=upper DELTA_GRID=0.1,1.0,2.0,5.0 \
      PARALLEL_INVERSION=true GRADIENT_METHOD=fixed_dual_fd_full \
      USE_VAR_SCALING=true SCALING_POWER=0.5 \
      OUTER_OPT_FILE=<repo>/full_aod_diag/csw_outer_1000.opt \
      OUT_DIR=<repo>/sequential_gravity/batch_out_realD20_W80000_fixeddualfdfull_scaled05 \
      REAL_DATA_DIR=<repo>/real_data/noah_D20 \
      julia -t 19 --project=. sequential_gravity/run_profiled_production.jl

## 5. Results

| delta (nominal) | gamma'_focal | kappa | gravity R_mean | max trade-share error (correct rho per destination) | exact delta* (moved A) | exact delta*(A=A*, same gamma') |
|---|---|---|---|---|---|---|
| 0.1 | 0.972866 | 0.044813 | -2.26e-04 (PASS) | 9.62e-07 (PASS) | 0.100418 (ratio 1.0042) | 0.114740 (Δδ=+0.014) |
| 1.0 | 0.950260 | 0.081518 | -9.63e-05 (PASS) | 7.23e-07 (PASS) | 1.000044 (ratio 1.0000) | 1.799106 (Δδ=+0.799) |
| 2.0 | 0.944055 | 0.091491 | -8.79e-05 (PASS) | 9.18e-07 (PASS) | 2.000090 (ratio 1.0000) | **INFEASIBLE** (A* cannot reach this gamma' at all) |
| 5.0 | 0.939767 | 0.098359 | -7.66e-05 (PASS) | 7.63e-07 (PASS) | 4.986679 (ratio 0.9973) | **INFEASIBLE** (A* cannot reach this gamma' at all) |

All four points independently verified (`verify_d20_deltagrid.jl`, a completely fresh cold
re-solve at each saved `best_feasible_theta`, no warm start reused from the original run):

- **Gravity holds** at every point (R_mean well within the 5e-4 tolerance).
- **All D trade-share moments hold** -- both the focal destination (checked against the
  literal hard-argmin rule, matching `EK_moments_focal_norm_directgp!`'s own convention) and
  the D-1 omitted destinations (checked against the rho=0.002 smoothed model, matching
  `invert_destination`'s own internal convention) -- to 1e-6/1e-7, consistent with
  production's own `DEST_INV_TOL=1e-6` target. (Two false alarms were found and fixed in
  the VERIFICATION SCRIPT itself while producing this table, not in the production
  solution -- see the note below.)
- **Exact delta\* matches the nominal budget closely** (ratio 0.997-1.004) at every point --
  a fresh, independent inner-dual KNITRO solve, not merely re-reading what the original
  search reported.
- **A\* (fixed-A baseline) genuinely underperforms**: at delta=0.1 and 1.0, A* CAN reach the
  same gamma'_focal but needs strictly more divergence (+0.014 and +0.799 respectively). At
  delta=2.0 and 5.0, A\* **cannot reach that gamma' at all** -- the fixed-A inner solve fails
  outright (not merely "more expensive"), confirming moving A is not just an efficiency
  gain at these higher budgets but a genuine feasibility requirement.

An additional, stricter check was run at the user's request: re-inverting the omitted
destinations under the LITERAL hard-argmin rule (rho=0, not the production rho=0.002
smoothing), warm-started from the production solution, to see whether the smoothed
solution corresponds to (or is close to) a genuine hard-argmin equilibrium. See
`verify_d20_deltagrid_hardargmin.jl` / its run log for that result.

**Two false alarms in the verification script itself, corrected before reaching the table
above** (recorded so a future re-check doesn't repeat them): (1) an early version checked
the omitted destinations' shares under rho=0 (hard-max) when they were actually solved
under rho=0.002 (smoothed) -- showing spurious 1e-3-to-3e-3 "errors" that vanished once the
correct rho was used. (2) Having fixed that, applying rho=0.002 UNIFORMLY (including to the
focal destination) then showed a NEW spurious, delta-growing error on focal specifically --
because focal's shares are computed via a literal hard argmin in
`EK_moments_focal_norm_directgp!` (never smoothed at all; focal's u comes directly from
`focal_u(theta)`, not from `invert_destination`). The correct check uses rho=0 for focal
and rho=0.002 for the omitted destinations -- matching what each part of the model actually
solves under -- which is what produced the clean PASS row above.

**A separate, independent bug was also found and fixed while producing this table**:
`run_one_bound` (the actual batch-loop entry point that produced these four saved results)
never threaded the `scaling_power` keyword through at all -- these results were actually
computed with `scaling_power=1.0` (the function default), not 0.5 as originally documented
throughout this file (see the correction note at the top). The results themselves are
unaffected (independently verified above); only the scaling_power label was wrong.

## 6. Scope and honest limitations

- The scaling correction (3b) is empirically tuned (one probe gradient, one power
  parameter), not a theoretically principled or adaptive scheme. It was tested at exactly
  one value (scaling_power=0.5) for this delta grid.
- The winner-boundary correction (3a) has an analogous closed-form ("boundary derivative")
  alternative that was NOT extended to the full D+2-moment problem this session -- only
  the finite-difference version (3a) was used for these results.
- Post-hoc validation: an independent Claude session ran a genuinely different check (an
  unconstrained "fix gamma'_focal, minimize delta* over A_{.,focal} alone" reformulation,
  see memory entry `profiled-reformulation-fix-gp-min-deltastar`) confirming the delta=1.0
  endpoint is a real local optimum, not a scaling artifact. See section 5 above for whether
  the same check has been extended to delta=0.1/2.0/5.0.
- This is the UPPER bound only; the lower bound has not been run with this exact
  configuration at D=20 (D=4 has both bounds validated, see
  `full_d2_correction_report.md` section 5).
