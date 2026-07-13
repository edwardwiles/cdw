# Gravity-seeded initial CC solve: evaluation

## The question

The sequential-linearized profiled-gravity method's first step, at every θ-evaluation, solves a
min-divergence CC problem matching ONLY the focal moments (D+1 of them) — it does not know
anything about gravity-feasibility. Only AFTER that first solve (and the resulting destination
inversions) is the exact gravity residual computed, and only in the SUBSEQUENT sequential
iterations does the method start correcting `p` toward gravity-feasibility, via a linearized
gravity moment appended to the CC solve.

The concern motivating this experiment: does ignoring gravity in that very first solve risk
"drift" — landing on a `p` (and hence destination inversions) that's far from what's needed, in a
way the linearized correction can't cleanly fix?

## The variant tested

`seq_gravcol_gravityseed` (in `run_real_outer_comparison.jl`): if a previous θ's converged
`(p, umat)` is available (the outer-loop's own warm-start chain), the FIRST CC solve at the new θ
is augmented with the linearized gravity moment evaluated at that PRIOR `(p, umat)` — i.e. instead
of a blind D+1-moment solve, it's a D+2-moment solve seeded with an approximate gravity
constraint from the last point visited. Falls back to the blind solve when no prior F exists
(first-ever evaluation). Everything else (the sequential refinement loop, damping, exact
convergence check) is untouched and identical to baseline.

## An important confound found and fixed along the way

The FIRST attempt at this comparison (at `delta=10`, the same divergence budget that produced
genuinely degenerate KNITRO-own points in trade_robustness_modular's own recent history — commit
`60ca63d`, "Sequential upper delta=10: KNITRO-own point is ALSO degenerate (R_mean=1.1e+107)")
immediately reproduced that exact failure mode: `max|R_mean|` up to `4.8e+137`. Root-caused (see
the git log of `sequential_gravity/profiled_gravity.jl` on this branch) to a catastrophic Newton
overshoot in `invert_destination`'s damped-Newton solver when a destination's target share becomes
numerically unreachable under an ill-conditioned Hessian (observed `cond(H)` up to `4.7e20`) —
fixed with Levenberg-Marquardt damping + a hard reject on oversized trial steps. **This fix is
orthogonal to the gravity-seeding question** (it lives entirely inside `invert_destination`,
independent of how the LFD weights `p` were obtained) and was validated separately before redoing
this comparison. Confirmed: it benefits BOTH the baseline and gravity-seeded methods equally
(`max|R_mean|` is ~1.7-1.8e-03 for both after the fix, vs astronomical before) — so it does not
explain the difference between the two methods below.

## Result (delta=10, upper bound, D=10, W=8000, outer maxit=15, LM fix active on both)

| | baseline (blind first solve) | gravity-seeded |
|---|---|---|
| θ-evaluations | 76 | 81 |
| feasible | 16 | 14 |
| infeasible | 60 | 67 |
| final KNITRO-own point | γ'=0.6781, κ=0.4766, R_mean=−4.9e-04, **gravity_ok=true** | γ'=0.6726, κ=0.4837, R_mean=−6.2e-04, **gravity_ok=false** |
| max\|R_mean\| across all evals | 1.73e-03 | 1.78e-03 |
| wall time | 841s | 2690s (3.2x slower) |

Raw data: `real_outer_comparison_results.csv` (includes earlier maxit=3 smoke-test rows from
before the LM fix, kept for the record — those show the `4.8e137`/`2.1e128` pre-fix blow-ups).

## Conclusion: gravity-seeding the first solve is NOT recommended

It is worse on every measured axis in this comparison: substantially slower (expected — it solves
a harder D+2-moment problem at every evaluation instead of D+1), finds fewer feasible points, and
— most importantly — **the point the outer solve ultimately settles on is itself
gravity-infeasible**, whereas baseline's final point is feasible.

Best interpretation: the seed moment is computed from the PREVIOUS θ's `(p, umat)`, which is only
a valid local linearization near that prior point. At a large divergence budget (δ=10 lets the
outer search take large steps), the new θ can be far enough from the previous one that this
"gravity-seed" moment is a poor, even actively misleading, approximation — it doesn't push `p`
toward gravity-feasibility at the NEW θ so much as toward satisfying a constraint that was only
sensible at the OLD θ. So the original drift concern (the blind first solve ignores gravity
entirely) cuts the other way here: no information turned out to be safer than stale information.

**Caveat**: single run each, one bound direction, one δ value — not repeated across multiple outer
search trajectories or random seeds. The gap (3.2x slower, flips feasible→infeasible at the final
point) is large enough that it doesn't look like noise, but this is not an exhaustive sweep. If
useful, natural extensions would be: the lower bound direction, a smaller δ (where the outer
search stays closer to the calibrated point and the seed's staleness matters less), or blending
the seed with the current point's own moments instead of using the prior point's raw linearization.
None of these were run, given the clear direction of the result already in hand.

## Files

- `setup_and_variant.jl` — pipeline setup + both `seq_gravcol_baseline`/`seq_gravcol_gravityseed`
  as standalone functions (used by the synthetic-perturbation script below; superseded by the
  real-outer-loop comparison for the actual conclusion).
- `compare_drift.jl` — an earlier, synthetic-theta-perturbation version of this comparison,
  superseded per the user's request to use a real KNITRO outer search instead. Kept for reference;
  not the basis for the conclusion above.
- `run_real_outer_comparison.jl` — the actual comparison: a real `outer_solve_nested_cached` KNITRO
  search with either method wired into `make_stateful_moments`, env-configurable
  (`GRAVITY_SEED`, `DELTA`, `BOUND`, `OUTER_MAXIT`, `WVAL`, `DVAL`). This is what produced the
  result above and the LM-fix diagnostic data.
- `real_outer_comparison_results.csv` — every run's summary row (method, config, result, timing).
- `csw_outer_*_driftcheck.opt` — reduced-maxit KNITRO outer option files auto-generated by the
  script (byte copies of production's `csw_outer_25.opt` with only `maxit` changed).

Reproduce:
```bash
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
DVAL=10 WVAL=8000 GRAVITY_SEED=false DELTA=10 BOUND=upper OUTER_MAXIT=15 julia --project=. full_aod_diag/gravity_seeded_initial_solve/run_real_outer_comparison.jl
DVAL=10 WVAL=8000 GRAVITY_SEED=true  DELTA=10 BOUND=upper OUTER_MAXIT=15 julia --project=. full_aod_diag/gravity_seeded_initial_solve/run_real_outer_comparison.jl
```
Must run on `demand.mit.edu` (KNITRO license is machine-locked there).
