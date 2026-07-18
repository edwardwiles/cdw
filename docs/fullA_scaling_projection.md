# Full-A D/W computational scaling projection

Phase 1C deliverable. **Computational benchmark only** — every D≠4 economy here uses fresh random
draws/parameters (via `context_scaled.jl`'s `d_exact_setup_scaled`); no kappa/Delta value is ever
compared across D. All timings are wall-clock, single Julia process, `Threads.nthreads()=1`, on
`demand.mit.edu`, commit `cd8adc2`. Raw data:
`results/fullA_d4/cd8adc2/profile_D_W_scaling/profile_D_W_scaling.csv`.

**Caveat repeated throughout, not just here**: these are local empirical fits from 4 (D) and 3 (W)
points respectively — enough to see the right order of magnitude and rule out wildly wrong
assumptions, not enough for a tight confidence interval on the exponent. The D=10 `eval_warm`
data point is visibly noisy (lower than D=8's, breaking monotonicity) — flagged explicitly below,
not smoothed over.

## D scaling at fixed W=8000

| D | n_free | exact eval (warm, median) | `Delta_FD` full gradient | `L_fix_FD` full gradient | `Q_adj_FD` full gradient |
|---|---|---|---|---|---|
| 4 | 17 | 43ms | 3.69s | 0.66s | 0.62s |
| 6 | 37 | 162ms | 14.92s | 3.04s | 2.80s |
| 8 | 65 | 409ms | 38.11s | 7.55s | 10.26s |
| 10 | 101 | 234ms *(noisy — see caveat)* | 98.51s | 20.96s | 17.85s |

Fitted log-log slopes (power-law exponent in `cost ~ D^k`):

- `n_free`: **k=1.94** — matches the exact structural fact `n_free = D²+1` (slope→2 as D grows); this
  one is not really an empirical fit, it's a consistency check that the harness is measuring what it
  claims to.
- exact eval (warm): **k≈2.11**, but low confidence given the D=10 non-monotonicity noted above —
  plausibly consistent with `n_free`'s own D² growth dominating per-eval cost, but this specific
  number should not be trusted to more than one significant figure.
- `Delta_FD` gradient: **k≈3.54**
- `L_fix_FD` gradient: **k≈3.70**
- `Q_adj_FD` gradient: **k≈3.78**

All three full-gradient methods cluster around **D^3.5-3.8** — steeper than `n_free`'s D² alone,
consistent with a genuine second factor (per-FD-probe cost itself growing with D, on top of needing
more probes) rather than a single clean mechanism. This compounding is the central scaling risk this
investigation faces, not the per-evaluation cost in isolation.

### Projections (D^k fit extrapolated — explicitly NOT validated beyond D=10)

| metric | D=10 (fit vs actual) | D=20 (projected) |
|---|---|---|
| `Delta_FD` gradient | 91.7s fit vs 98.5s actual (close) | **~1064s (~18 minutes) for ONE gradient** |
| `L_fix_FD` gradient | 19.4s fit vs 21.0s actual (close) | **~254s (~4.2 minutes) for ONE gradient** |
| `Q_adj_FD` gradient | 20.0s fit vs 17.9s actual (close) | ~274s |
| `n_free` | 100.5 fit vs 101 actual (exact) | 387 (exact: `D²+1=401`, fit is close) |

**Reading this projection**: a single D=20 outer-loop iteration needing even one `Delta_FD` gradient
would cost ~18 minutes; the existing D=4 short runs needed 15-40 outer iterations to reach a
`BEST_FEASIBLE` candidate (`optfd_upper_20260717_190946`: 40 iterations, 583s total at D=4). If outer-
iteration counts scale similarly at D=20 (untested, but not obviously wrong given the underlying
optimization landscape's kink structure doesn't change with D), an `Delta_FD`-only D=20 run is
plausibly **many hours to days**, before any block-locality or parallelization work (Phase 2, not
attempted this continuation). `L_fix_FD`'s ~4x cost advantage at D=20 (254s vs 1064s per gradient) is
therefore considerably MORE valuable at D=20 than the ~3x advantage measured at D=4 — directly
supporting Phase 4/6's finding that a hybrid `L_fix`-primary scheme is the right direction, now with a
concrete cost projection for why it matters more as D grows, not just a qualitative argument.

## W scaling at fixed D=4

| W | exact eval (warm, median) | `Delta_FD` full gradient |
|---|---|---|
| 8,000 | 43ms | 3.69s |
| 20,000 | 84ms | 2.76s *(noisy — see caveat)* |
| 80,000 | 366ms | 14.39s |

Fitted slopes: exact eval **k≈0.94** (essentially linear in W, sensible for a Monte-Carlo-average
cost dominated by per-draw work). `Delta_FD` gradient **k≈0.64** (sub-linear — but this fit is built
from only 3 points, one of which — W=20,000's gradient time being LOWER than W=8,000's — is itself
noisy/non-monotonic; do not treat 0.64 as a trustworthy exponent, only as evidence the growth is not
worse than roughly linear). **This is genuinely good news if it holds**: it means the D-scaling
projection above is the primary risk, not a combined D×W blowup — W=80,000 (the paper's actual
production setting) does not appear to multiply the D-driven cost problem by another order of
magnitude, though this needs a cleaner (more repetitions, D>4 cross-points) confirmation before being
relied on.

## What this rules in and out for Phase 8 (staged D scaling)

- **D=6/D=8 computational pilot**: cost is real but tractable — `Delta_FD` gradients of 15-40s are
  compatible with a short (tens of outer iterations) pilot run within a normal working session.
- **D=10 computational pilot**: `Delta_FD` gradients of ~100s each mean a 15-40-iteration outer run
  using `Delta_FD` alone would take 25-70 minutes just in gradient cost, before KNITRO's own step-
  taking overhead — feasible for one supervised run, not for casual iteration/debugging at that D.
- **D=20 full optimization**: NOT attempted, and this projection is exactly why — per the task's own
  gating criteria, this should wait for Phase 2 (block-locality reuse, parallel FD, reducing the
  `moments_recompute` redundancy documented in `docs/fullA_performance_profile.md`) and for the
  `L_fix`-hybrid scheme (Phase 4/6) to be wired into a live solver, not attempted with the current
  `Delta_FD`-only, no-block-reuse implementation.
