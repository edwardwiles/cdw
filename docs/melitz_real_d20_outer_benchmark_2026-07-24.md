# Melitz real D=20 outer benchmark: calibration closure + upper-GT-bound campaign -- 2026-07-24

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`), starting checkpoint `787ba98`
(tip of `docs/melitz_optimization_report_2026-07-24_closure.md`), continuing the same-day
sessions `docs/melitz_pareto_data_calibration_2026-07-24.md` and
`docs/melitz_d20_rank_deficiency_mechanism_2026-07-24.md`. New/modified:
`src/melitz/pareto_calibration.jl` (calibration closure), `test/melitz/runtests.jl`
(updated real-D20 testset), `scripts/melitz_real_d20_*_2026-07-24.jl` (six new diagnostic
scripts backing every number below).

## Executive summary

1. **Real-data calibration closure (Phases 1-2)**: a genuine estimation-order bug (theta
   estimated on raw pre-policy `tau` while calibration used post-policy `tau`) collapsed the
   A-/f-gravity residuals from `~1e-4` to machine precision (`~1e-15`) once fixed
   (`theta_star=:estimate`). Wage calibration now uses an authoritative direct linear solve
   instead of an iterative method's own convergence floor. Full test suite: `1371/1371`,
   unchanged, both before and after.
2. **Seed sensitivity (Phase 3), an unanticipated finding**: at the calibrated real D=20
   point, `W=80,000` converges cleanly for only 1 of 8 QMC seeds tried -- the prior
   session's "W=80,000 resolves D=20" claim was a seed=1-specific result, not a general one.
   `W>=120,000` is more robust. This session proceeds at the governing prompt's literal
   `W=80,000`/seed=1 as instructed, with the fragility disclosed rather than hidden.
3. **Kernel profiling (Phase 4)**: BLAS>=4 saturates the inner solve (~5s); 16 Julia threads
   give ~7x speedup on the 798-dimensional direct gradient (25s -> 3.4s serial-to-parallel).
4. **Gradient quality (Phase 5)**: the direct fixed-dual gradient agrees in SIGN with
   genuinely reoptimized secants at every sparse coordinate direction tried; DENSE random
   directions can drive the reoptimized secant itself into numerical failure (not the
   gradient backend) -- a real finding for how large an outer step can safely be.
5. **Fixed-A/f benchmark (Phase 6)**: a clean, cold-verified restricted incumbent,
   `kappa_fixed=0.9294`, `GT_fixed=0.0706` -- about 3.5x the calibrated reference's own GT.
6. **Full outer campaign (Phase 7)**: the first attempt hung for 55+ minutes -- root-caused
   (not guessed) to Melitz's own `lower_limit_guard` KNITRO-native early-bailout never being
   wired up, plus an unbounded nested inner-solve `maxit`. Fixed and re-run to a clean
   completion (`157` function evals, `22` full inner solves, `600.65s`). **The flexible
   798-dimensional search made essentially ZERO improvement over its own starting point**
   (`kappa=0.947535` vs. starting `0.947560`) and did NOT reach the fixed-A/f incumbent.
7. **Interpretation (Phase 8)**: per the addendum's own decision rule, `kappa_full >
   kappa_fixed` is a SEARCH-PERFORMANCE finding, not an economic one (the restricted
   feasible set is a strict subset of the full one, so the true optimum cannot be worse).
   **Recommendation (Phase 9)**: use the fixed-A/f scalar profile as the production
   upper-bound procedure at this finite delta; the flexible search needs the
   `lower_limit_guard` fix made standard, and likely a better outer-search strategy, before
   it is a reliable routine step.

## 0. Phase 0: preserve and reproduce

- Julia `1.12.6`, KNITRO `13.0.1` (`/opt/shared_sw/knitro/13.0.1`), 208 logical CPUs, 3.0TiB
  RAM. `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` exported at every Julia launch per this
  repo's standing rule; BLAS threads set at runtime via `BLAS.set_num_threads`.
- Working tree at session start: 5 modified tracked files (`bounded_cache.jl`,
  `delta_star.jl`, `finite_delta_outer.jl`, `include_melitz.jl`, `test/melitz/runtests.jl`)
  plus `src/melitz/pareto_calibration.jl` and 3 docs untracked -- all from the same-day
  prior sessions this one continues. Archived (patch + untracked-file copies + a
  `git stash create` object, `5e1d519b...`, never applied) before any edit this session.
- Full Melitz test suite (`test/melitz/runtests.jl`) reproduced the documented **1371/1371**
  baseline exactly, unchanged, before any code change this session.
- The specific W=80,000 real D=20 fixed-point number from the companion rank-deficiency
  report (`nStatus=0`, `Delta=4.1232488e-4`, wall~16.3s) is not independently re-run
  byte-for-byte under the OLD (pre-fix) code -- it is superseded directly by Phase 3 below,
  which re-derives the same benchmark under the FIXED calibration pipeline and reports both
  the new number and its relationship to the old one.

## 1. Phase 1: real-data gravity calibration closure

### 1.1 The ordering bug, confirmed and fixed

The prior session's real-D20 test built `MelitzObservedData` from `pi.csv`/`tau.csv`, but
estimated `theta_hat` via `melitz_gravity_theta_check(lambdaData, tauData, 100.0; sigma=2.5)`
called on the **raw, pre-`MelitzObservedData`** CSV arrays -- i.e. `tau`'s diagonal still at
its raw, non-unit values (`row`'s `1.0011215579`) -- while the calibration that followed used
`observed.tau`, whose diagonal had already been snapped to exactly `1.0`. Two different `tau`
matrices, one `theta_hat`.

**Fix** (`calibrate_melitz_pareto`, `theta_star=:estimate`): `theta_star` is now estimated,
when requested, from the SAME frozen `observed.lambda`/`observed.tau` every other step of the
function reads -- `melitz_estimate_theta_hat(observed.lambda, observed.tau)`, called strictly
after `MelitzObservedData`'s construction, never before.

**Measured effect** (`real_data/noah_D20`, France focal, sigma=2.5):

| quantity | value |
|---|---|
| `theta_hat`, estimated on the FROZEN, post-policy `observed` object | `8.751773` |
| `theta_hat`, OLD ordering (estimated on raw pre-policy `tauData`) | `8.747116` |
| difference | `4.657e-03` |

A real, non-trivial (though numerically small) difference -- confirming the ordering bug was
genuine, not merely theoretical.

### 1.2 Gravity specification audit

Traced `prestep/master_prestep.jl`'s `thetaIn==0` branch line-by-line: `thetaHat =
-sum(Wlambda.*Wtau)/sum(Wtau.*Wtau)`, `Wlambda = withinTransform(lambda)`, `Wtau =
withinTransform(tau)`. `withinTransform` (`misc/doubleDiff.jl`) logs internally
(`lz = log.(z)`) and applies the standard two-way (origin+destination) fixed-effects
"within" residual -- by Frisch-Waugh-Lovell this is exactly the OLS two-way-FE gravity
coefficient. `moments/newGravityMoment!.jl`'s `UoModel==1` branch (confirmed via `grep`
the universal production setting) builds the SAME `withinTransform`-based bilinear
orthogonality moment. No mask, no weights, no ROW exclusion, no domestic-cell exclusion:
the full `D x D` matrix (every origin-destination pair, including domestic `o==d` and the
`row` aggregate) enters via SHARES (not flows), in LEVELS (the log is internal to
`withinTransform`).

`melitz_estimate_theta_hat` (`pareto_calibration.jl`) now calls this exact formula, reused
verbatim -- **not re-derived** -- so the Melitz real-data theta estimate is, by
construction, identical in specification to `production/fullA-exact`'s own canonical
gravity moment.

**Reconciliation of `8.75` vs. the `~6.8` figure appearing elsewhere in this repo's Melitz
work**: `6.8` is the SYNTHETIC D=4 fixture's own chosen/assumed `theta_star`
(`generate_fake_melitz_data(...; theta_star=6.8...)`), a convenient benchmark parameter for
a fabricated economy -- it is not a canonical empirical estimate for
`real_data/noah_D20` or for this dataset's own gravity regression. No script or config in
this repo computes a "canonical" Ricardian-side theta for `real_data/noah_D20` specifically
to reconcile against (`grep` for `noah_D20` across `production/fullA-exact` returns
nothing) -- the two numbers describe two different datasets/DGPs and are not expected to
agree. `8.751773` is simply this real dataset's own two-way-FE OLS gravity coefficient
under this repo's canonical specification.

### 1.3 Gravity residuals: from ~1e-4 to machine precision

With `theta_star=:estimate` (i.e. `theta_star == theta_hat` computed on the SAME data the
calibration itself uses), the two separate A-/f-gravity restrictions become compatible **by
construction**, not merely within a loosened tolerance:

| quantity | OLD (documented, prior session) | NEW (this session, fixed ordering) |
|---|---:|---:|
| `rhs_A` vs `rhs_f` scaled gap | `3.1e-5` (`gravity_tol=1e-4`) | `3.331e-16` |
| `gravity_residual_A` | `-1.4e-4` | `-1.092e-15` |
| `gravity_residual_f` | `2.6e-4` | `-2.810e-16` |

`gravity_tol` in `calibrate_melitz_pareto` for the real-D20 path is now `1e-6` (the D=4
synthetic default), not the previously-loosened `1e-4` -- and passes comfortably. The
residual was **entirely attributable to the estimation-order bug**, exactly as the
governing prompt's Section 1.3 anticipated -- not a genuine real-data/finite-sample
limitation, and no averaging of incompatible targets (`rhs_used = (rhs_A+rhs_f)/2`) is
doing any real work anymore (the two targets already agree to `3.3e-16` before averaging).

### 1.4 Outer-coordinate roundtrip

Derivation (addendum Section 7): under the active `:logf` outer parameterization,
`theta_free = (log(gamma_prime_j), A_free[1:D^2-1], f_free_free[1:D^2-2])`.
`expand_free_theta` reconstructs `A` from `A_free` ALONE (no `gamma` dependence) and every
free `f` cell OTHER than `f[j,j]` from `f_free_free` ALONE (via a pivot offset that depends
on `f[j,j]`, not directly on `gamma`). The ONE object that is NOT held fixed as `gamma`
varies is `f[j,j]` itself, derived from `gamma` via `derive_fjj_from_autarky_cutoff` --
exactly the theorem's own `zhat'_jj=1` requirement (main prompt Section 6 explicitly
forbids freezing `f_jj`). **Conclusion**: "fixed A/f, vary gamma" is represented EXACTLY by
holding `theta_free[2:end]` fixed at the calibrated point and letting `theta_free[1]`
alone move -- no new coordinate system is needed; this IS the addendum's restricted path.

Roundtrip check (`melitz_calibration_roundtrip_check`): reduce calibrated `(A,f,gamma')` to
`theta_free`, expand back, compare:

| quantity | value |
|---|---|
| `max_rel_A_diff` | `5.298e-15` |
| `max_rel_f_diff` | `6.191e-15` |
| `gamma_prime_diff` | `0.0` (exact) |
| `max_abs_share_diff` | `3.331e-16` |
| `focal_link_residual_reexpanded` | `-2.383e-10` (matches the calibration's own) |

Machine precision throughout -- a direct consequence of 1.3's fix (a calibration with
non-machine-precision gravity residuals would show pivot-cell drift here; this one does not).

### 1.5 Raw-data policy (addendum Sections 1-3)

`MelitzObservedData` no longer silently mutates supplied data. New explicit,
recorded policies:

- `share_policy` (default `:as_supplied`): `real_data/noah_D20/pi.csv`'s raw columns sum to
  1 with `max|colsum-1| = 7.058e-08` (machine/file precision) -- used EXACTLY as supplied,
  no forced renormalization (`observed.lambda == Matrix{Float64}(lambdaData)` byte-for-byte).
- `tau_diagonal_policy` (default `:normalize_to_one`): `MelitzPrimitives` (`types.jl`)
  hard-asserts `diag(tau)==1.0` EXACTLY (a genuine, repo-wide structural requirement, not a
  preference of this file) -- so the addendum's Section 2.2 branch applies. The raw `row`
  diagonal (`1.0011215579`, a measurement artifact) is preserved in `.tau_diagonal_raw` for
  the record; the ONE explicit snap is applied consistently before theta estimation OR
  calibration ever reads `tau` (fixing 1.1's ordering bug simultaneously).

## 2. Phase 2: authoritative direct wage solve

`melitz_solve_wages_linear` (new): replaces one row of `(I-lambda)` with the numeraire
condition `E[numeraire]=L[numeraire]` and solves the resulting rank-`D` dense linear system
directly (Gaussian elimination, no iteration). `calibrate_melitz_wages` now returns THIS as
the authoritative `w`/`E`; damped-Jacobi and an independent Perron-eigenvector solve are
retained as cross-checks only (`w_damped`, `damped_residual`, `perron_residual`,
`damped_vs_direct_residual`).

| quantity | value |
|---|---|
| direct-solve market-clearing residual | `8.517e-08` |
| damped-Jacobi residual (200,000 iterations) | `1.892e-08` |
| Perron eigenvector residual | `9.464e-06` (eigenvalue `0.9999999811`) |
| `max\|w_direct - w_damped\|/max\|w\|` | `9.517e-06` |
| wage range | `[0.0371, 2.0089]` |

**Honest reading**: the market-clearing residual sits at `~8.5e-8`, not machine precision
`~1e-15` -- this is a genuine CONDITIONING floor of `(I-lambda)` at this D=20 real-data
scale (an eigenvalue close to, but not exactly, the Perron unit eigenvalue -- confirmed
independently by the Perron solve's own `0.9999999811`), not an artifact of any particular
method's own iteration count or tolerance. All three independent methods (direct linear
solve, 200,000-iteration damped Jacobi, dense eigendecomposition) agree to within
`~1e-5` relative -- strong cross-validation that this is a genuine numerical floor of the
underlying linear-algebra problem, not a bug in any one solver. The real, substantive
improvement over the prior session is not the residual's ORDER OF MAGNITUDE (comparable)
but that it now comes from a non-iterative, authoritative, tolerance-free solve rather than
an iterative method's own convergence choice.

## 3. Phase 3: real D=20 fixed-point revalidation + seed sensitivity

At the FIXED calibration (`theta_star=:estimate`), W=80,000, `seed=1`:

| quantity | value |
|---|---|
| moment matrix rank | `401/401` |
| condition number | `1.064e+05` |
| min active draws (worst cell) | `200`, cell `(3,14)` (bra->kor, same cell flagged in the prior session) |
| `nStatus` | `0` |
| `Delta(theta*)` | `4.0720699e-04` |
| KKT opt/feas error | `1.914e-14` / `0.0` |
| max weighted moment residual | `1.915e-14` |
| primal/dual divergence | `4.0720699e-04` / `4.0720699e-04` (agree to `2.4e-15`) |
| `residual_autarky_cutoff` | `0.0` exact |
| `N_prime_diff_rel` | `5.329e-15` |
| wall | `9.02s` (BLAS=16) |
| GT_model vs GT_ACR | `0.0203028104` both, diff `1.285e-11` |

This matches the companion rank-deficiency report's own W=80,000/seed=1 finding
(`nStatus=0`, `Delta~4.1e-4`, full rank, cond~1e5) closely -- the small numeric shift
(`4.07e-4` vs. `4.12e-4`) is attributable to the theta_star correction (1.1/1.3 above), not
a regression.

### 3.1 Seed sensitivity: a real, previously unexamined fragility

**Finding, not anticipated going in**: the W=80,000 fixed point is **NOT** robust across
QMC seeds. Sweeping `seed=1..8` at the SAME calibrated point, W=80,000:

| seed | rank | condition number | `nStatus` | `Delta` | `lfd_ok` |
|---|---|---|---|---|---|
| 1 | 401/401 | 1.064e+05 | 0 | 4.0721e-04 | true |
| 2 | 400/401 | 1.369e+16 | -102 | 1.0e10 (sentinel) | false |
| 3 | 399/401 | 1.339e+16 | -103 | 3.05e+08 | false |
| 4 | 399/401 | 1.269e+16 | -102 | 1.0e10 | false |
| 5 | 401/401 | 1.430e+07 | -103 | 1.31e+08 | false |
| 6 | 400/401 | 1.259e+16 | -103 | 1.54e+08 | false |
| 7 | 401/401 | 5.289e+07 | -102 | 1.0e10 | false |
| 8 | 399/401 | 1.362e+16 | -102 | 1.0e10 | false |

**Only 1 of 8 seeds (seed=1) converges cleanly at W=80,000.** This revises the companion
report's "W=80,000 fully resolves the D=20 rank deficiency for real data" finding: that
finding was correct FOR SEED=1, but W=80,000 sits at a fragile threshold, not a robust
resolution -- most seeds still land in the double-precision-floor conditioning regime
(cond `~1e16`, rank 399-400/401) that afflicts small `W`.

Raising `W` further resolves the fragility directly: seed=2 (failed at W=80,000) converges
cleanly at both `W=120,000` (rank 401/401, cond `1.13e5`, `nStatus=0`, `Delta=1.98e-4`) and
`W=150,000` (rank 401/401, cond `1.12e5`, `nStatus=0`, `Delta=1.06e-4`). **Revised
recommendation**: real D=20 work should treat `W=80,000` as marginal, not safely resolved
-- `W>=120,000` is the more robust practical threshold for THIS calibration
(scripts/`melitz_real_d20_seed_sensitivity_2026-07-24.jl`). Per this session's own scope
(a delta=1 upper-bound benchmark, explicitly not a new W-sweep campaign), Phases 4-7 below
proceed at the governing prompt's literal `W=80,000` using the one seed (`seed=1`) known to
converge cleanly.

## 4. Phase 4: kernel profiling at the calibrated reference (W=80,000, seed=1)

(`scripts/melitz_real_d20_kernel_profile_2026-07-24.jl`)

### 4.1 Inner successful solve: BLAS thread sweep (Julia coordinate threading inactive)

| BLAS threads | wall | nStatus | bytes allocated |
|---|---:|---|---:|
| 1 | 12.2-14.9s | 0 | ~1.16GB |
| 4 | 5.0-5.5s | 0 | ~518MB |
| 8 | 4.8-5.6s | 0 | ~518MB |
| 16 | 4.7-5.0s | 0 | ~518MB |
| 20 | 4.8-5.3s | 0 | ~518MB |

BLAS threads beyond 4 give no further benefit (a ~2.5x speedup from 1->4, flat after) --
consistent across 5 independent process runs. Recommendation: **BLAS=4** is sufficient for
the inner solve; the production default of 16 costs nothing extra but is not needed either.

### 4.2 Rejection classification

Two real, distinct rejection paths, deliberately NOT conflated:

- **Cheap deterministic cutoff/export-selection screen** (`melitz_outer_state`, Section
  1.3, no Monte Carlo, no KNITRO): at a wildly displaced point (`g -= 5.0`\*, cutoff-
  infeasible, `min_slack=-3.31`), wall = **142 microseconds**. This is the genuinely fast
  rejection path.
- **A cold, isolated (no warm dual-bank context) full inner solve** at a "moderately"
  displaced point (`g += 0.5` from calibration): this point turned out to ALSO be
  cutoff-infeasible (`min_slack=-0.31`) -- illustrating that even a "moderate" 0.5 log-gamma
  displacement already breaks export-selection at this real calibration (a genuinely narrow
  feasible corridor around the reference point, corroborating the Phase 6 finding below that
  the whole `Delta<=1` upper-bound region sits within `|g-g_calib|<~0.1`). The raw solve
  (maxit capped at 2000 for this diagnostic, vs. production 10,000) took **115.0s** to hit
  the iteration limit (`nStatus=-400`), a genuine `NumericalFailure`, not a certified
  `BudgetInfeasible`.

**Correct reading**: the `melitz_classified_inner_solve` machinery's stored-dual and
live-threshold screens (main prompt's "fast rejection") require a WARM dual bank populated
from nearby ALREADY-SOLVED trajectory points -- they cannot be exercised meaningfully by a
single isolated cold probe, only by a live trajectory (Section 7 below reports their actual
hit counts/timing from the real campaign). What IS confirmed here: (a) the CHEAP
range/cutoff-style screens that need no KNITRO at all are, as expected, orders of magnitude
faster than any KNITRO attempt; (b) a genuinely difficult point's raw KNITRO solve is NOT
fast even when eventually classified as a failure -- consistent with this repo's own
documented `-400`/`-102` D=20 findings elsewhere.

### 4.3 Full outer gradient: Julia thread sweep (BLAS=1), `:B_direct_argument_serial`/`_parallel`

`n_theta` (free outer coordinates) = **798** (matches `2*20^2-2`).

| Julia threads | serial wall | parallel wall | per-coord (parallel) | max serial/parallel disagreement | gradients possible in 600s (parallel) |
|---|---:|---:|---:|---:|---:|
| 1 | 25.6s | 25.8s | 0.0323s | 0 | 23.3 |
| 4 | 24.9s | 8.5s | 0.0106s | 0 | 70.8 |
| 8 | 25.0s | 5.2s | 0.0066s | 0 | 114.8 |
| 16 | 25.0s | 3.4s | 0.0043s | 0 | 174.2 |
| 20 | 24.8s | 3.7s | 0.0046s | 0 | 163.7 |

Serial/parallel backends agree EXACTLY (max relative disagreement `0.0` -- both compute the
identical finite-bandwidth secant, just with/without `Threads.@threads`). Near-linear
speedup through 16 threads (25s -> 3.4s, ~7.3x), a slight REGRESSION at 20 (3.7s, likely
oversubscription/contention on this 208-core shared box against other concurrent jobs, not
a genuine algorithmic ceiling). **Recommendation: 16 Julia threads, BLAS=1, for the outer
gradient phase** (matching the main prompt's own phase-specific-threading instruction) --
affords ~174 complete 798-dimensional gradients in a 600-second budget, i.e. a full
gradient is cheap relative to the 10-minute campaign budget; the binding constraint on how
many OUTER ITERATIONS a 10-minute run completes is much more likely to be the NUMBER OF
INNER SOLVES (FC calls) KNITRO's own line search/trust region needs per accepted step, not
gradient wall-clock itself.

## 5. Phase 5: gradient-quality diagnostics (real D=20/W=80,000/seed=1)

(`scripts/melitz_real_d20_gradient_quality_2026-07-24.jl`). Compares the `:B_direct_argument_serial`
fixed-dual secant (`direct_deriv = dot(grad, v)`) against a genuinely REOPTIMIZED
(independently re-solved) central-difference secant (`reopt_secant = (Delta(theta+hv) -
Delta(theta-hv))/2h`, both endpoints a real cold `evaluate_melitz_delta` call), at 8
representative directions, `h=1e-4` (production default). Scoping decision (documented,
following this repo's own established practice of a representative subset over an
exhaustive grid once informative): `h=1e-4` only for the main comparison (not the full
`{3e-5,1e-4,3e-4,1e-3}` sweep -- 8 directions x 4 bandwidths x 2 signs = 64 real cold
KNITRO solves near a fixture already shown, Section 3.1, to be conditioning-fragile, would
cost far more wall-clock than the marginal information justifies), plus `h=1e-3` as a
single-order-of-magnitude robustness check on 2 directions.

| direction | `direct_deriv` | `reopt_secant` | sign match | rel. err | switches |
|---|---:|---:|---|---:|---:|
| A_ordinary | -4.749e-02 | -6.986e-02 | true | 0.320 | 10 |
| A_pivot_sensitive | -1.300e-04 | -2.540e-04 | true | 0.488 | 5 |
| f_ordinary | 2.559e-02 | 3.168e-02 | true | 0.192 | 7 |
| f_pivot_sensitive | -2.457e-04 | -2.608e-04 | true | 0.058 | 12 |
| gamma | 3.355e-03 | 3.904e-03 | true | 0.140 | 79 |
| focal_origin | 1.487e-02 | 1.537e-02 | true | 0.033 | 14 |
| random_block_1 | 2.027e-01 | **0.000e+00** | true\* | 2.0e9 | 222 |
| random_block_2 | -1.446e-01 | **-4.99e+13** | true\* | 1.000 | 199 |

`h=1e-3` robustness check: `gamma` (`direct=3.355e-03`, `reopt=7.339e-03`, sign match);
`A_pivot_sensitive` (`direct=-1.300e-04`, `reopt=-4.130e-05`, sign match).

**Reading, honestly**: every SPARSE basis-coordinate direction (the 6 non-random rows)
agrees in SIGN with the direct fixed-dual secant, at relative errors `3%-49%` -- large
relative errors are EXPECTED at a finite `h` through a discontinuous-active-set map
(exactly this repo's own documented, pre-existing finding for the D=4 fixture, closure
report Section 3: neither estimator is the "classical derivative" at a participation-
boundary jump), not a sign of a broken gradient. The two DENSE random-block directions are
the important, previously-untested finding: at `h=1e-4`, ONE side of each random-block
central difference (`Delta(theta+hv)` or `Delta(theta-hv)`) landed at the `Delta=1e10`
numerical-failure sentinel (touching 199-222 participation switches simultaneously, vs.
5-79 for the sparse directions) -- the reoptimized secant is then either a spurious
near-zero (both sides saturate near the sentinel, canceling) or an enormous, meaningless
number, NOT a real measurement of the true derivative. **The direct fixed-dual gradient's
own coordinate-by-coordinate construction (perturbing ONE `theta_free` coordinate touched
per direction) never has this failure mode** -- it is the reoptimized secant, not the
gradient backend, that breaks down under a dense simultaneous perturbation touching hundreds
of cells at once. This is informative for Phase 7: a full 798-dimensional KNITRO step
(effectively a dense direction) risks the SAME failure mode if its trial step is too large,
reinforcing the importance of KNITRO's own trust-region/step-size control and the
`theta_box` bound.

**Final registered-Jacobian audit**: the SAME 8 rows, read as
`registered=direct_deriv/delta_budget`, `FD-of-registered=reopt_secant/delta_budget`
(`delta_budget=1`, this session's own `delta=1` upper-bound budget) -- identical numbers to
the table above (division by 1 is a no-op), confirming the closure-audit session's own B1
finding (registered Jacobian scaling is correct) continues to hold at this real D=20 point,
for every SPARSE direction. **Production bandwidth selected: `h=1e-4`** (the existing
default) -- no evidence from this diagnostic favors a different bandwidth, and the random-
direction pathology is a KNITRO-trust-region concern, not a bandwidth concern.

## 6. Phase 6: fixed-A/f scalar delta=1 profile

(`scripts/melitz_real_d20_fixed_af_profile_2026-07-24.jl`)

### 6.0 Derivation: "fixed A/f" under the active `:logf` outer parameterization (addendum Section 7)

Answered concretely in Section 1.4 above: `theta_free = (log(gamma_prime_j),
A_free[1:D^2-1], f_free_free[1:D^2-2])`; `A` is reconstructed from `A_free` ALONE, every
free `f` cell except `f[j,j]` from `f_free_free` ALONE; only `f[j,j]` depends on `gamma`
(via the `zhat'_jj=1` autarky normalization, `derive_fjj_from_autarky_cutoff`) -- exactly
the object the main prompt explicitly says must NOT be frozen. **"Fixed A/f, vary gamma" =
hold `theta_free[2:end]` fixed at the calibrated point, let `theta_free[1]` (`g =
log(gamma_prime_j)`) alone move.** No new coordinate system needed; the existing outer
parameterization already IS this restricted path.

### 6.1/6.2 Coarse grid + bracket + bisection (`W=80,000`, `seed=1`, warm-started sequentially)

| `g` | `gamma_prime` | `kappa` | `GT` | `Delta` | `nStatus` | verified | `min_slack` |
|---|---:|---:|---:|---:|---|---|---:|
| -0.4188 (calib) | 0.657855 | 0.979697 | 0.020303 | 4.072e-04 | 0 | true | 0.0205 |
| -0.4388 | 0.644829 | 0.966721 | 0.033279 | 3.571e-02 | 0 | true | 0.0246 |
| -0.4688 | 0.625771 | 0.947579 | 0.052421 | 2.184e-01 | 0 | true | 0.0246 |
| -0.4988 | 0.607277 | 0.928816 | 0.071184 | 1.060e+00 | 0 | true | 0.0246 |

Bracket found after 4 grid points (early-stop, avoiding an unnecessarily deep grid scan
into a region already shown, Section 3.1/4.2, to be conditioning-fragile). Bisection
(`Roots.jl`, `xatol=1e-3` -- loosened from a naive `1e-6`: each evaluation here is a REAL
cold KNITRO solve near the fragile `Delta~1` boundary, not a closed-form function, so a
tight `xatol` would cost many more real solves for benchmark-irrelevant extra precision):

| quantity | value |
|---|---|
| `g_fixed` | `-0.49783321` |
| `gamma_prime_fixed` | `0.60784631` |
| `kappa_fixed` | `0.92939627` |
| `GT_fixed` | `0.07060373` |
| `Delta(g_fixed)` | `9.969031e-01` |
| `nStatus` | `0` |
| `min_slack` | `0.024553` |
| gravity residual A/f | `-3.757e-15` / `7.086e-16` (machine precision) |
| cold/full-value re-verification | `Delta=9.969031e-01`, `nStatus=0`, `lfd_ok=true` -- MATCHES the warm bisection result exactly |

**This is the Phase 6 incumbent, saved as the verified restricted benchmark**: at the real
D=20/W=80,000/seed=1 calibration, holding every genuine A/f primitive fixed at its
calibrated level and moving only the autarky price power `gamma_prime_j`, the maximal
divergence-`<=1`-feasible upper-bound welfare gain is `GT_fixed = 0.0706` (`kappa_fixed =
0.9294`) -- roughly **3.5x** the calibrated reference's own `GT = 0.0203`.

A striking, concretely-measured feature of this fixture: the ENTIRE `Delta<=1` feasible
region along the gamma-only restricted path is `g in [-0.4988, g_calib]`, a window of only
`~0.08` in log-gamma (`gamma_prime in [0.607, 0.658]`) -- Section 4.2's "moderate" `g+=0.5`
displacement is `>6x` wider than this ENTIRE feasible corridor, explaining why it landed
cutoff-infeasible.

## 7. Phase 7: ten-minute full A/f outer campaign at delta=1

(`scripts/melitz_real_d20_full_campaign_2026-07-24.jl`)

### 7.0 Two real bugs found and fixed live during this phase (not speculation -- both root-caused and verified)

The first campaign attempt (`theta_box=2.0`, `g_start=-0.495`, no other changes) hung for
**55+ minutes** before being killed manually -- far past the intended `maxtime_real=600`
KNITRO option. Root-caused, not guessed:

1. **`lower_limit_guard` was never wired up for Melitz.** `build_melitz_implicit_bundle`
   (this file) exposes a `lower_limit_guard` kwarg that sets the shared `cc_algo`
   functor's KNITRO-native mid-solve early-bailout (`if f <= lower_limit; return
   -KNITRO.KN_INFINITY`, an objective-threshold stop DURING the barrier method, not merely
   a pre-solve screen). The Ricardian model hardcodes `lower_limit=-50`
   (`cc_algo/ccInner.jl`/`ccOuter.jl`) so a barrel-method iterate that has clearly wandered
   far past any usable divergence budget bails out immediately. Melitz's own construction
   left this at its struct default (`nothing` -> `-KNITRO.KN_INFINITY`, i.e. PERMANENTLY
   DISABLED) -- so a single nested inner CC solve inside one outer callback had no fast-exit
   and ran all the way to `maxit`/convergence-tolerance criteria, which for a genuinely
   divergent/ill-conditioned trial point can be extremely slow. **Fixed**: this session's
   campaign passes `lower_limit_guard=49.0` (`lower_limit = -(delta+guard) = -(1+49) = -50`,
   matching the Ricardian convention exactly) to `solve_melitz_finite_delta_bound`.
2. **`maxit` on the inner solve was uncapped for a nested-solve context.** The production
   `melitz_inner_loop_options.opt` (`maxit=10000`) is appropriate for a STANDALONE inner
   solve but, combined with (1)'s missing early-exit, meant a single outer callback could in
   principle run for the ~700s-order non-convergence times this repo has documented
   elsewhere for D=20. **Fixed**: the campaign uses a maxit=1000-capped copy of the inner
   option file for every nested solve (including the initial cold evaluation), bounding the
   worst case per callback regardless of (1).
3. Additionally confirmed (per direct user question, not assumed): `par_concurrent_evals`
   is `yes` in the production `melitz_inner_loop_options.opt` (matching this repo's own
   documented history that `par_concurrent_evals=no` previously caused a DIFFERENT
   outer-nests-inner deadlock, AUD-02, and was reverted -- `cc_algo/PsiObjectiveBundle.jl`'s
   own header comment). The base `melitz_outer_finite_delta.opt` left it UNSET (KNITRO
   default) -- the campaign's derived outer opt file now sets it explicitly to `yes` to
   remove that ambiguity (KNITRO 13.0.1 also warns this option name is deprecated in favor
   of `concurrent_evals`, noted but not required for this session's fix).

With BOTH fixes, a second attempt (`theta_box=2.0`, `g_start=-0.4688` -- Phase 6's own
fast/reliable grid point, chosen as a safety margin over the more fragile `g=-0.495`) ran
to a clean, correct completion: `EXIT: Time limit reached. Current point is feasible.`

### 7.1 Campaign configuration

`W=80,000`, `seed=1`, `delta=1.0`, `direction=:upper`, `gradient_backend=
:B_direct_argument_parallel` (Phase 4/5's own recommendation), `h=1e-4`, `theta_box=2.0`,
`lower_limit_guard=49.0`, `dual_bank_max_size=8`, Julia threads=16 (BLAS=1, matching Phase
4.3), `maxtime_real=600`, inner `maxit=1000` (campaign-only cap, see 7.0). Starting point
`g=-0.4688` (`kappa0=0.947560`, `GT0=0.052440`, `Delta0=0.2187`) -- an INTERIOR point on the
Phase 6 grid, not the `0.90-0.95`-`Delta` point the main prompt's own default guidance
suggests, because Phase 6/4.2 jointly established the entire `Delta<=1` feasible corridor
along the gamma-only path is only `~0.08` wide and fragile near its edge -- starting exactly
at that edge risked exactly the kind of intractable cold solve Section 7.0 already
documents. The full 798-dimensional search still has the ENTIRE feasible region to explore
from this interior start; it is not artificially constrained to the fixed-A/f path.

### 7.2 Is the outer solver actually searching? Yes -- and it does not improve on its own starting point

KNITRO's own termination: `nStatus=-401` (time limit), `157` function evaluations, `22`
gradient evaluations, `21` outer iterations, feasible at exit (feasibility error `0.0`).
Reconstructed directly from this session's own `[LIVE]` trajectory log (the structured
`MelitzFiniteDeltaOuterResult` summary itself was lost to a trivial post-hoc script bug --
a wrong field name in a `@printf` call, `n_inner_eval_failures` vs. the actual
`inner_eval_failures` -- fixed in the script for the record, but re-running the 600s
campaign a third time purely to reprint an already-fully-logged summary was not worth the
wall-clock; every number below is independently recovered from the raw per-event log):

| quantity | value |
|---|---|
| total FC (function) calls | 157 |
| total GA (gradient) calls | 22 |
| classified `InnerSolved` (genuine converged inner solves) | 22 |
| classified `BudgetInfeasible` (Delta>1, certified/rejected) | 133 |
| classified `MomentInfeasible` | 2 (exact finite-support certificate, no KNITRO needed) |
| sum (= total FC calls) | 157 |
| accepted (outer-feasible: `InnerSolved` AND `Delta<=1`) | 22 (100% of `InnerSolved` calls -- EVERY converged inner solve at THIS fixture satisfied the delta=1 budget) |
| **distinct `g` values among the 22 ACCEPTED points** | **effectively ONE**: `-0.4688`/`-0.46883`/`-0.46884` (floating-point-level noise around the STARTING point itself) |
| best accepted `kappa` (min, over the whole 600s run) | `0.947535` (at `n_fc=51`, `t=233.81s`) |
| starting `kappa0` | `0.947560` |
| **improvement over the starting point** | `0.000025` absolute (`0.0026%` relative) -- economically and numerically negligible |

**KNITRO's own trajectory explored a wide range of `g`** (`BudgetInfeasible` rejections
span `g in [-0.758, -0.469]`, i.e. it genuinely probed far outside the narrow feasible
corridor, consistent with a real line search/trust-region exploring the 798-dimensional
space) -- but **every single point it ever accepted as outer-feasible landed back at
essentially the exact starting point**. KNITRO's own final objective
(`-4.68840593075633e-01`) matches the starting `g` to 5 decimal places. This is not "the
search barely moved" -- by every accepted-point metric available, **the search did not
move the ACCEPTED incumbent at all** within the 600-second budget, despite exploring widely
along the way.

### 7.3 Wall-clock accounting

KNITRO's own native accounting: `Total program time = 600.65s` (`2385.03s` CPU time across
16 threads), `Time spent in evaluations = 559.45s` (**93.1%** of total wall-clock), leaving
only `~41.2s` (**6.9%**) as KNITRO's own outer-algorithm overhead (barrier-method
bookkeeping, line search, gradient assembly outside the callback itself). This session's own
`[LIVE]` timestamps, classified by outcome (inter-event wall-clock gap, a direct proxy for
each callback's own cost):

| classification | count | total wall | mean | min | max |
|---|---:|---:|---:|---:|---:|
| `InnerSolved` | 22 | 135.8s | 6.17s | 4.33s | 34.24s (first, cold) |
| `BudgetInfeasible` | 133 | 475.9s | 3.58s | 1.59s | 23.27s |

Sums to `611.6s`, matching the log's own last timestamp (small difference from KNITRO's own
600.65s is this session's logger clock starting slightly before `KN_solve` itself, at
bundle construction). **`BudgetInfeasible` rejections are NOT free/instant** at this
fixture -- averaging `3.58s` each, they still require substantial callback work (moment
reconstruction plus whatever partial dual-solve activity the stored-dual/live-threshold
screens perform before certifying rejection), materially more than Section 4.2's
microsecond-scale CHEAP deterministic cutoff screen. `InnerSolved` calls (genuine converged
solves) average `6.17s` when warm-started (vs. `34.24s` cold for the very first one) --
consistent with Phase 4.1's own BLAS-threaded inner-solve benchmark (`~5s` at BLAS>=4) plus
warm-start savings.

## 8. Phase 8: fixed-A/f vs. full flexible comparison

| Quantity | Fixed A/f (Phase 6) | Flexible A/f (Phase 7, best accepted) |
|---|---:|---:|
| `kappa` | `0.92939627` | `0.947535` |
| `GT=1-kappa` | `0.07060373` | `0.052465` |
| `Delta` | `0.9969031` | `0.21903` |
| budget slack (`1-Delta`) | `0.0031` | `0.781` |
| min cutoff slack | `0.024553` | `0.0246` |
| gravity residual A/f | `-3.757e-15` / `7.086e-16` | machine precision (pivot-exact by construction) |
| inner verification status | `nStatus=0`, cold-verified | `nStatus=0` (22 independent verified solves) |
| total wall time | few minutes (grid+bisection, several cold solves) | 600.65s (hard cap) |
| inner solves | ~9 (grid+bisection+cold-verify, per cache stats) | 22 |
| gradients | 0 (scalar bisection, no gradient needed) | 22 |

**`kappa_full_best (0.947535) > kappa_fixed (0.92939627)`** -- the flexible 798-dimensional
search's BEST accepted point is materially WORSE (higher kappa, lower GT) than the 1-
dimensional restricted benchmark. Per the addendum's own decision rule (Section 6/10):

> "Full search remains worse than the fixed benchmark: this means the high-dimensional
> numerical search has not recovered the known restricted feasible incumbent. Continue
> reporting the restricted solution as the best incumbent. This is a search failure, not an
> economic result."

**This is exactly that case, cleanly diagnosed, not ambiguous.** The restricted feasible
set (theta_free[2:end] fixed) is a strict SUBSET of the full feasible set (theta_free[1]
alone, at Phase 6's own optimum, is one specific point already reachable by the flexible
search too) -- so the theoretical global optimum of the flexible problem can never be
worse than the fixed one. The observed ordering (`kappa_full_best > kappa_fixed`) is
therefore ATTRIBUTABLE ENTIRELY to the flexible search's own numerical performance within
its 600-second budget, not to any property of the model, the theorem, or A/f flexibility
itself being economically harmful.

**Diagnosing WHY the search failed to move (concrete, not speculative), from the evidence
already gathered**:

1. **The starting point itself is already very close to a local optimum along the easy
   gamma direction** -- Phase 6 showed the ENTIRE gamma-only feasible corridor is only
   `~0.08` wide, and this campaign started well inside it (`Delta0=0.219`, comfortable
   slack). A local optimizer sitting near even a LOCAL optimum of a highly nonlinear,
   near-`Delta=1`-active-constraint region may need many small, well-conditioned steps to
   make progress -- and this problem's own `A_ordinary`/`f_ordinary` gradient-quality
   relative errors (Section 5, `19%-49%`) show the finite-bandwidth secant gradient is a
   genuinely NOISY/approximate signal near hard participation boundaries, which a
   quasi-Newton outer method (`hessopt=2`, no exact Hessian) can struggle to use for fine
   local refinement.
2. **`maxit=1000` per inner solve (this session's own safety cap, Section 7.0) may itself
   be truncating some genuine progress** -- if KNITRO's own trust region occasionally
   proposed a promising but not-yet-converged step, a capped inner solve could report
   `NumericalFailure` or an inaccurate `Delta` rather than a clean accept, discouraging
   further exploration in that direction. This is a genuine, disclosed TRADE-OFF this
   session made deliberately (Section 7.0: without the cap, the campaign does not run at
   all) -- not evidence the underlying model/search is unsalvageable, only that safely
   running this search within a bounded wall-clock budget currently costs some of its own
   effectiveness.
3. **21 outer iterations in 600s is a modest budget** for a 798-dimensional NLP with a
   noisy gradient -- Phase 4.3 showed a single COMPLETE gradient costs `3.4s` at 16 threads
   in isolation, but under this campaign's real, contended conditions (nested nonconverged
   nonlinear solves, nested cold-vs-warm variability, nested screening), each outer
   iteration cost roughly `600/21 approx 28.6s`, an order of magnitude more than the
   isolated gradient-only benchmark -- confirming Section 4.3's own prediction that
   NUMBER OF INNER SOLVES per accepted step, not gradient wall-clock, is the binding
   constraint on outer-iteration throughput.

**A/f movement**: given every accepted point sits at the identical `g`, `theta_free[2:end]`
(every A/f coordinate) also never moved at any ACCEPTED point -- the flexible search never
found a reason (an improving, outer-feasible direction) to move away from holding A/f fixed
at THIS starting point, at least within its budget. This is consistent with, though does not
by itself prove, the theorem's qualitative prediction that A/f flexibility's VALUE may be
concentrated at more extreme welfare targets (closer to the delta-infinity limit) rather
than at this comparatively modest finite delta=1 budget -- but the direct, honest reading
of THIS run is a search-performance finding, not an economic one.

## 9. Phase 9: what to optimize next

Ranked by the wall-clock evidence actually gathered this session (not intuition):

1. **Highest priority: make the inner-solve safety cap (`maxit`) unnecessary, or raise it,
   by fixing the ROOT conditioning problem** -- `lower_limit_guard` now gives a fast,
   correct native bailout for genuinely far/infeasible points (Section 7.0), which should
   let a FUTURE campaign safely use the production `maxit=10000` (removing this session's
   own `maxit=1000` cap and its associated risk of truncating genuine progress, Section
   8 point 2) as long as `lower_limit_guard` is always wired up going forward -- this is a
   ONE-LINE production fix (the bug was purely "never called with this kwarg"), not a
   design change, and should be the default for any future Melitz outer campaign.
2. **Outer-search strategy/parameterization** -- 21 outer iterations for zero net movement
   suggests the quasi-Newton line search (`hessopt=2`) is not making effective use of even
   the SIGN-correct-but-noisy gradient near this fixture's narrow feasible corridor.
   `docs/melitz_optimization_report_2026-07-24_closure.md`'s own prior finding (Method B3/B4
   comparisons still open) and this session's Phase 5 gradient-quality diagnostic (large
   relative errors at basis directions, catastrophic failure at dense random directions) both
   point the same way: before spending more wall-clock on a longer campaign, a smaller
   theta_box, a scaled/preconditioned step, or an exact-Hessian outer solver (if a
   Hessian-vector product for `Delta(theta)` is ever made available) may matter more than
   raw compute.
3. **Inner exact-Hessian/BLAS**: NOT the binding constraint here (Phase 4.1: BLAS>=4 already
   saturates at ~5s per solve; Phase 4.3: 16 Julia threads gives ~174 parallel gradients per
   600s in isolation) -- deprioritize further optimization here.
4. **Moment construction / screening**: `BudgetInfeasible` rejections average `3.58s`, not
   free -- a cheaper mid-tier screen (between the microsecond cutoff check and a full/
   partial inner solve) could shrink the `475.9s` (78% of this campaign's total wall-clock)
   spent on rejections, but this is a secondary optimization relative to point 1/2's more
   fundamental "the search isn't finding improving directions at all" finding.
5. **Recommendation for the production upper-bound procedure**: given the fixed-A/f scalar
   profile (Phase 6) already obtains a verified, cold-checked, gravity-exact incumbent in a
   handful of cheap evaluations, and the full flexible search -- even after fixing two real
   bugs that were blocking it from running at all -- did not beat that incumbent within a
   600-second budget, **the fixed-A/f restricted procedure is the more reliable, far
   cheaper production path for this upper-bound question at this finite delta**. The
   flexible search remains theoretically necessary only for confirming/improving on the
   restricted bound at LARGER delta (approaching the theorem's own limiting regime) or with
   more generous compute/robustness fixes (point 1/2 above) -- not recommended as a routine
   10-minute production step until those fixes are in place and independently re-validated.
