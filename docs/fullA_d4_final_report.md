# Full-A D=4 exact formulation: findings report (INTERIM)

Branch `diag/fullA-d4-exact`, worktree `../gravity-fullA-d4`, off production commit `53ffb58`.
All artifacts under `results/fullA_d4/<commit>/`. This report is being written incrementally as
the investigation proceeds — see the "Status of this report" section at the end for exactly what
is and isn't covered yet. It supersedes nothing; `docs/fullA_d4_code_audit.md` remains the
authoritative record of the code-level findings this report summarizes and builds on.

## 1. Exact free-parameter vector (task §6)

At D=4: **l_full = 23** (μ, σ, 4 inert γ_θ slots, γ'_focal, 16 A_od entries), **n_free = 17**
(γ'_focal + all 16 A_od entries — every entry, not D²−D). This resolves the ambiguity the
methodology PDF itself flags: the code generalizes the PDF's focal-only `γ_focal≡1` gauge to
`γ_d≡1` for *every* destination, removing the `A[1,d]=1` pin from every column, not just the focal
one. Independently verified (not just read off the code): pack/unpack round-trip exact, every free
coordinate has a measurable nonzero effect on the moment system, the 4 γ_θ slots are functionally
inert (byte-identical output when perturbed), duplicate/gap detection works. See
`results/fullA_d4/642dfe3/parameter_map.csv` and `full_aod_diag/d4_exact/parameter_table.jl`.

## 2. Formulation implemented

Gravity is currently a **second explicit KNITRO equality constraint** (closed-form analytic
gradient, validated to relerr 2.5e-10 against finite differences) rather than eliminated. This
investigation additionally built and validated **exact gravity elimination** in log(A_od)
coordinates, where gravity is provably affine (verified to 1.7e-18, not assumed): both a sparse
pivot (drop the largest-|coefficient| entry) and an orthonormal nullspace parameterization achieve
machine-precision gravity feasibility (~1e-18) at random points, with correct rank (D²−1=15) and
exact pack/unpack round trips. **Not yet resolved**: transformed box bounds under either
parameterization (task's own explicit warning that these aren't independent per-coordinate);
the short solver runs (§7 below) use generously wide, non-rigorous bounds as a pragmatic stand-in.

## 3. The central finding: the winner-boundary derivative bug is real, present, and now quantified exactly

A prior session diagnosed (at D=5, in a different formulation) that ForwardDiff through
`hFunction!`'s hard `MinInd!` branch silently drops the winner-boundary/Dirac term. This
investigation **independently re-confirmed it for this exact D=4/D²-free-A path** and, going
further, **quantified it precisely** using a closed-form (not linearized) exact tie-threshold
formula built for this purpose (`winner_switching.jl::exact_tie_thresholds`, validated: the
predicted threshold produces exactly 1 real winner switch at a ±1e-7 straddle):

- Along a representative direction, naive pathwise AD (`Method A`, exactly reproducing production's
  existing Method-B gradient) matches a central finite-difference secant to **~2e-11** for any `h`
  below the nearest tie threshold (`t*=1.82e-4`), then the gap jumps to **0.0114** (~16% of the
  0.0685 slope) for `h` just above `t*` — a clean, textbook confirmation that AD misses exactly the
  switching contribution, nothing else. (`results/fullA_d4/6b6ff4a/method_A_pathwise_ad_check.csv`)
- The finite-draw objective is **genuinely kinked**, not just smoothly curved: mean(G) and the
  frozen-adjoint scalar both show a jump ~1e5x larger at the exact predicted threshold than at a
  matched control point straddled by the same tiny (1e-9) delta.
  (`results/fullA_d4/45ac6c6/winner_switching_summary.csv`)
- Derivative-only smoothing (`Method E`, wiring the codebase's own existing but unused
  `smoothMinIndNew!` into a diagnostic-only moment function) genuinely removes the kink: the same
  1e-9-straddle jump shrinks from 8.27e-6 (hard) to 1.52e-11 (smoothed, tuner=-100) — a ~5e5x
  reduction — and converges cleanly back to the hard value as the smoothing temperature sharpens.
  (`results/fullA_d4/9e03706/smoothing_check.csv`)

**No fix for this bug is wired into full-A's actual outer-loop gradient in this investigation** —
the sequential/profiled method's existing `fixed_dual_fd_full` correction does not transfer
(different moment count, different free-parameter vector; see code audit §6). Building and wiring
an analogous corrected gradient for the true full-A problem is the largest piece of unfinished
work this investigation identifies.

## 4. Three-way derivative distinction (task §4/§9/§12A-D)

Implemented and validated for the true full-A `d=18`-moment problem (the sequential method's
`D+2`-moment machinery does not apply as-is). All three objects (frozen-adjoint `Q_adj`,
fixed-dual `L_fix`, fully-optimized `Delta`) are **exactly equal at the base point** (diff ~1e-17),
a real identity forced by the inner KKT conditions plus strong duality, not a coincidence — derived
by hand and confirmed numerically. Away from the base point:

- `Q_adj` and `L_fix` track each other closely across the entire h-grid tested (0.2 down to
  0.00625; diff 2e-5 to 1.3e-3).
- The **fully re-optimized value's slope diverges sharply from both at the h=0.1 benchmark step**:
  6.7x the value it converges to by h≈0.0125, and the optimized value is outright **NaN
  (inner-solve infeasible) at h=0.2**. `frozen_adjoint_Q`/`fixed_dual_L` show no such failure at
  any tested h. (`results/fullA_d4/1bdb1cc/h_sweep.csv`)

**Two sign-convention bugs were caught and fixed while building this** (both confirmed by hand
derivation before being accepted as bugs, not assumed from a mismatch alone): `Delta_dual` was
initially computed with the wrong sign (`Delta(theta) = -f`, not `f`, per KNITRO's own constraint
convention in the actual callable code); and the task brief's own schematic `Q_adj` formula, taken
literally, has a derivative exactly opposite in sign to `L_fix`'s under this codebase's convention.

## 5. Inner CC dual (task §8)

Validated independently (own direct re-evaluation of the dual scalar, bypassing the production
callable, agrees to 7.6e-18) and found genuinely, non-trivially **ill-conditioned**: the inner
dual's own Hessian (`H_y = mean(Psi''(q_s)[1;G_s][1;G_s]')`, computed directly, not read off
KNITRO) is **full rank (18/18) but has condition number ~1.1e4** at the calibration point —
exactly the trap the task brief warns about ("do not infer good conditioning from full rank").
Cold/warm and loose/tight-tolerance reproducibility both confirmed clean (spread 0.0 in both
cases). (`results/fullA_d4/b048943/inner_checks.csv`)

## 6. KNITRO options (task §16) — a second, independent real finding

The production outer opt file (`csw_outer_25.opt`) requests `hessopt=4` (BFGS) but also sets
`eval_fcga=yes`, which KNITRO **silently downgrades to L-BFGS** — confirmed not from the opt file
alone but from KNITRO's own log output on the real economic problem: `"Option hessopt=4 not valid
when eval_fcga=1. Changing hessopt to 6 (LBFGS)."` Setting `eval_fcga=no` restores genuine BFGS
(confirmed by the ABSENCE of that log line). **This means every full-A result in this codebase's
history that used the default opt file unmodified ran on L-BFGS, not the Hessian mode it
requested and reported** — including the previously-reported "best-conditioned" D=4/D=10 numbers
from an earlier session. (`results/fullA_d4/bf00b00/knitro_default.log`,
`knitro_fcga_no.log`)

## 7. Short D=4 solver runs (task §18) — headline result: a verified stationary point exists

A from-scratch outer-loop KNITRO driver (`run_d4_optimized_fd.jl`) was built: pivot-eliminated
reduced coordinates (gravity dropped as an explicit constraint, satisfied exactly by construction),
optimized-value central finite differences (h=0.01, chosen from the h-sweep finding that h=0.1 is
unsafe) as the gradient method, `eval_fcga=no` (genuine BFGS, per §6 above), maxit=15,
best-feasible-incumbent tracking, and an exact fresh cold recheck of both the raw terminal point
and the tracked best-feasible point.

Two bugs were found and fixed while getting this running (both in the new driver, not production
code): a feasibility-gate that checked the wrong (unweighted, not LFD-weighted) moment residual,
and a missing guard against a non-finite FD probe crashing the entire KNITRO solve
(`ERROR: Jacobian element jac[4]... is undefined at the current point`, KNITRO status -502,
reproduced identically twice before the fix — see commits `6b38f02`, next one). After both fixes:

- **Upper direction (maxit=15)**: hit the iteration cap (KNITRO status -400, not internally
  converged — `opt_err=0.0016`) but the tracked best-feasible point is **exactly feasible on a
  fresh cold recheck** (gravity_value≈1.6e-18, max_abs_moment_kkt_resid≈1.7e-16, mean_m_resid≈0,
  Delta−delta=−0.0011, i.e. essentially binding) and **passes the external KKT stationarity check
  (task §22) cleanly**: constraint multiplier eta=0.0083>0, KKT residual 0.14% relative to the
  objective gradient's own norm, complementary slackness ≈−9e-6, zero active bounds, zero
  non-finite gradient probes. κ=0.1706. **This is a genuine `VERIFIED_STATIONARY_FEASIBLE_CANDIDATE`**
  per every criterion task §25 lists — the first point in this investigation (and, as far as this
  investigation's audit found, in this codebase's full-A history) checked this rigorously.
  (`results/fullA_d4/9e03706/optfd_upper_20260717_182444/`, `stationarity_check_upper.txt`)
- **Lower direction (maxit=15, after the fix)**: completed cleanly (no crash) and is exactly
  feasible (gravity≈3.4e-18, KKT≈6e-16), but Delta−delta=−0.101 — **far from binding**, meaning the
  search simply ran out of iterations before using its divergence budget. The stationarity check
  correctly reports this as **NOT stationary** (residual=1.0, eta≈0, since a fixed nonzero
  objective gradient cannot be a KKT point at an inactive constraint) — an honest
  `BEST_FEASIBLE_STALLED` result, not evidence against the method for this direction. A longer
  maxit=40 upper run was launched to check whether the upper result was genuinely converging or
  coincidentally near-stationary at maxit=15; not yet complete at the time of writing.
  (`results/fullA_d4/9e03706/optfd_lower_20260717_190831/`, `stationarity_check_lower.txt`)

## 8. Status labels applied so far

- **Upper direction, κ=0.1706**: `VERIFIED_STATIONARY_FEASIBLE_CANDIDATE` (task §25's strongest
  label — exact hard gravity, exact hard primal divergence, all full moments, inner KKT, external
  outer stationarity, and a fresh cold re-evaluation all pass).
- **Lower direction, κ=0.0107 (partial)**: `BEST_FEASIBLE_STALLED` — exact feasible point found,
  external stationarity fails, consistent with simply running out of the artificially short
  maxit=15 budget rather than a numerical breakdown.

## 9. Status of this report / what remains

**Done and reported above**: task §4, §6, §8, §9, §10, §11, §12 (A/B/C/D/E; F/G partial — F not
attempted, G implemented as a policy function only), §13, §16, §18 (one direction verified, one
stalled), §22 (applied to both §18 candidates).

**Not yet done**: the full derivative benchmark suite across all required point classes (§14);
scaling/conditioning experiments across the 5 required transformations (§15); the inner-tolerance
schedule as a genuine adaptive rule (§17, currently only two fixed tolerances tested); a longer/
tuned lower-direction run to see whether it also reaches a verified stationary point given enough
iterations; continuation from the sequential solution (§19); the gamma-profile (§20);
derivative-free cross-checks and multistart (§21, §23).

**Preliminary reading toward task §26's recommendation** (still provisional — one verified point in
one direction is encouraging but not sufficient for a confident overall call): the exact full-A
method **can** reach a genuinely verified stationary feasible point at D=4, provided (a) gravity is
eliminated rather than left as an explicit constraint, (b) the gradient is computed via optimized-
value finite differences at a safe step size (not h=0.1, and not the uncorrected AD/Method-B
gradient, which is confirmed biased), and (c) the KNITRO Hessian-mode misconfiguration is fixed
(`eval_fcga=no`). This is closer to "(b) viable only with a hybrid/corrected-gradient method" than
"(d) not currently viable" — but confirming the lower direction, multistart, and D=10+ scaling
behavior (all outside this D=4 investigation's scope) remain necessary before that reading is
solid.
