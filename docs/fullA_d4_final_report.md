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

## 7. Short D=4 solver runs (task §18) — IN PROGRESS

A from-scratch outer-loop KNITRO driver (`run_d4_optimized_fd.jl`) was built: pivot-eliminated
reduced coordinates (gravity dropped as an explicit constraint, satisfied exactly by construction),
optimized-value central finite differences (h=0.01, chosen from the h-sweep finding that h=0.1 is
unsafe) as the gradient method, `eval_fcga=no` (genuine BFGS, per §6 above), maxit=15,
best-feasible-incumbent tracking, and an exact fresh cold recheck of both the raw terminal point
and the tracked best-feasible point.

First attempt (before a feasibility-gate bug fix, see below) at maxit=15/timeout=400s:
- **upper** direction did not finish within 400s (still mid-search).
- **lower** direction terminated at KNITRO status **-502** after only 4 outer iterations
  (opt_err=0.051, not converged) — not yet diagnosed further.

A real bug was caught while reviewing this output: the feasibility gate used the raw, UNWEIGHTED
moment residual (expected to be large away from the base point — not a feasibility criterion) 
instead of the LFD-weighted KKT residual the inner dual actually targets. Fixed, and both
directions re-launched with a 600s timeout. **This section will be updated with final numbers once
those complete** — do not treat the two data points above as representative of the method's
viability; they reflect an unfinished driver, not the method itself.

## 8. Status labels applied so far

Per task §25's taxonomy, nothing in this investigation yet qualifies as
`VERIFIED_STATIONARY_FEASIBLE_CANDIDATE` (external KKT stationarity, task §22, has not been run).
The two completed short-run attempts are best characterized as
`INFEASIBLE_OR_NUMERICALLY_UNRESOLVED` under the corrected feasibility gate — but with the caveat
that this reflects the current (unfinished, un-tuned) driver rather than a considered judgment
about the method.

## 9. Status of this report / what remains

**Done and reported above**: task §4, §6, §8, §9, §10, §11, §12 (A/B/C/D/E; F/G partial — F not
attempted, G implemented as a policy function only), §13, §16.

**Not yet done** (see the working todo list in-session for current state): the full derivative
benchmark suite across all required point classes (§14); scaling/conditioning experiments across
the 5 required transformations (§15); the inner-tolerance schedule as a genuine adaptive rule
(§17, currently only two fixed tolerances tested); completing and interpreting the short solver
runs (§18); continuation from the sequential solution (§19); the gamma-profile (§20);
derivative-free cross-checks and multistart (§21, §23); the external KKT stationarity check (§22).

**No overall recommendation (task §26's a/b/c/d) is offered yet** — it depends materially on
whether a corrected gradient can be built and whether the short solver runs, once complete and
tuned, converge to genuinely stationary points. The evidence so far is consistent with either
"viable only with a hybrid/corrected-gradient method" or "not currently viable due to the
uncorrected winner-boundary bias" — distinguishing between them requires the unfinished work above.
