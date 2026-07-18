# Continuation 4, Phase 4: wall-clock algorithm frontier (minimum-bar comparison)

Supersedes nothing (`docs/fullA_algorithm_frontier.md` was scoped-not-attempted in continuation 3;
this is the first real run). Scope: the task's own stated **minimum bar** — "at least ONE wall-clock-
matched comparison between the existing product-FD/optimized-value control and hybrid-`L_fix`+SR1-
or-LBFGS" — plus the free intermediate points that cost nothing extra once the harness exists
(`run_phase4_frontier.jl`).

## 1. What "hybrid" means here, and a real finding that changed the comparison design

Phase 3 (`docs/fullA_next_handoff.md`'s newest section, `composite_gradient.jl`) built two new
gradient methods:

- **`lfix_composite`**: ALWAYS the cheap gradient (gamma component analytically exact, A-block via
  adaptive-bandwidth central FD over the O(1)-incremental `L_fix` tier). One inner dual solve per
  outer iterate (to build the base cache), vs. `delta_fd`'s 2×n_free=32 full inner re-solves.
- **`hybrid`**: switches between `lfix_composite` and the full `delta_fd` gradient per outer iterate,
  via a policy (`HybridGradientPolicy`) wired to `derivative_methods.jl::should_refresh`.

**Found this session, before trusting `hybrid` for anything**: gradient-SOURCE switching across
outer iterates is not safe to combine with a quasi-Newton Hessian approximation (SR1/BFGS/L-BFGS) as
KNITRO implements them here — a live run showed KNITRO terminating after only 2 outer iterations
(status `-102`) instead of running to the iteration/time limit, plausibly because those methods'
curvature updates assume a CONSISTENT gradient source across the secant pairs they build, and
alternating cheap/expensive breaks that assumption. `lfix_composite` (no source-switching at all)
does not exhibit this. **`lfix_composite` is therefore the primary new-method comparator below**;
`hybrid` configs are still run and reported (not hidden) so the instability is visible in the table.

## 2. Setup

- D=4, W=8000, δ=1, upper direction (`find_smallest=true`), common start `w0` (from `ctx.θ0_up`, same
  for every config — `d4_exact_setup`'s own deterministic construction).
- Wall-clock budget: **60 seconds** per config (`D4X_MAXTIME_REAL=60`, `maxit` bumped to 1e6 so wall
  time is genuinely binding, not `maxit` — verified via `KN_set_param_by_name`, see
  `run_d4_optimized_fd.jl`).
- 7 configs, all from `run_phase4_frontier.jl`:

| label | gradient | hessopt |
|---|---|---|
| `control_deltafd_productfd` | delta_fd (original, unchanged) | productfd (hessopt=4, the historical control) |
| `deltafd_sr1` | delta_fd | sr1 |
| `deltafd_lbfgs` | delta_fd | lbfgs |
| `lfixcomposite_sr1` | lfix_composite | sr1 |
| `lfixcomposite_lbfgs` | lfix_composite | lbfgs |
| `hybrid_sr1` | hybrid | sr1 |
| `hybrid_lbfgs` | hybrid | lbfgs |

- Each run does a mandatory exact-fresh-cold recheck of both the raw terminal iterate and the
  tracked best-feasible incumbent (unchanged scaffold behavior) — the numbers below are read from
  each run's own `summary.txt`, not re-derived.

## 3. Results

All numbers are the tracked **best-feasible** exact-cold-recheck κ (never the raw terminal iterate),
read from each run's own `summary.txt`. `wall_wrapper` is the full subprocess wall time (Julia
startup+compile+solve); `total_inner_solves` is `CS.INNER_SOLVE_COUNT[]` consumed inside gradient
calls only (excludes the function-value evaluations, which are identical/cheap across all configs).

| config | best-feasible κ | KNITRO status | outer iters | gradient calls | cheap calls | inner solves (gradients) | wall (wrapper) |
|---|---|---|---|---|---|---|---|
| `control_deltafd_productfd` | 0.156552 | -401 (iter/time limit) | 2 | 56 | 0 | 1792 | 111.1s |
| `deltafd_sr1` | 0.169596 | -411 | 50 | 51 | 0 | 1632 | 110.5s |
| `deltafd_lbfgs` | 0.171774 | -401 | 46 | 47 | 0 | 1504 | 111.9s |
| **`lfixcomposite_sr1`** | **0.172457** | -103 (converged, stopped at 42.7s of its own 60s budget) | 113 | 114 | 114 | **114** | 93.7s |
| **`lfixcomposite_lbfgs`** | **0.172345** | -101 (converged, stopped at 29.4s of its own 60s budget) | 60 | 61 | 61 | **61** | 80.2s |
| `hybrid_sr1` | 0.168874 | -411 | 46 | 47 | 7 | 1327 | 111.4s |
| `hybrid_lbfgs` | 0.169572 | -411 | 41 | 42 | 2 | 1329 | 112.0s |

## 4. Interpretation

**The minimum bar is met, decisively, in favor of the new composite gradient — via `lfix_composite`,
not `hybrid`** (see §1 for why `hybrid` is not the right vehicle for this comparison this session):

- **`lfixcomposite_sr1` is the best config by κ** (0.172457), beating every `delta_fd` variant
  including the historical control (`control_deltafd_productfd`, 0.156552 — a **+10.2% relative**
  improvement) and the best `delta_fd` variant found here (`deltafd_lbfgs`, 0.171774 — a **+0.4%**
  improvement, small but real and in the SAME wall-clock budget class).
- **It got there using 114 total inner solves vs. 1504-1792** for the `delta_fd` variants — a
  **13-16x reduction** in the expensive operation, and it didn't even need the full 60s budget
  (KNITRO's own `-103` convergence stop came at 42.7s of internal solve time). `lfixcomposite_lbfgs`
  is even more extreme: 61 inner solves total (**25-29x fewer** than `delta_fd`), converged
  (`-101`) using only 29.4s of its 60s budget, and still reached κ=0.172345 — within 0.06% of the
  best `delta_fd` config found in a FULL 60s budget.
- **113 and 60 outer iterations** for the two `lfix_composite` configs vs. 2-50 for `delta_fd` — the
  cheap gradient lets KNITRO take far more steps per unit wall time, which is exactly the mechanism
  Phase 2/3 were built to exploit.
- **`hybrid` (both Hessian modes) underperforms every `lfix_composite` config and even underperforms
  `deltafd_lbfgs`** (0.1686-0.1696 vs. 0.1723-0.1725) here, despite taking MORE wall time (111-112s)
  than `lfix_composite` needed. With the disagreement-trigger fix (§ Phase 3 handoff notes), only
  7/47 and 2/42 calls were cheap — the policy is still triggering an expensive refresh on the large
  majority of calls, so `hybrid` pays nearly `delta_fd`'s full inner-solve cost (1327/1329 vs. 114/61)
  without matching `lfix_composite`'s κ. **This is a real, reportable finding, not hidden**: as
  currently tuned, `hybrid`'s refresh policy is too conservative to realize the savings
  `lfix_composite` demonstrates are available; either the policy needs retuning (larger `gap_tol`,
  fewer triggers) toward something closer to always-cheap, or — given `lfix_composite` alone already
  clears the minimum bar decisively — `hybrid`'s extra complexity may not be worth pursuing further
  for this problem size.
- KNITRO's own reported statuses vary genuinely across configs (-401 iteration/time limit, -411 a
  relative-step-size stall, -103/-101 genuine convergence) — read as intended (not ranked by
  "KNITRO's own reported optimality," per the task's explicit instruction), the κ column is what
  matters and it tells a consistent, one-directional story.

**Bottom line**: at D=4/W=8000, upper direction, 60s wall-clock budget, common start — the composite
gradient (`lfix_composite`) beats the historical optimized-value-FD control by +10.2% relative κ using
13-29x fewer inner CC-dual solves, and modestly beats the best-tuned `delta_fd` variant found here too.
This is the clearest, most decisive finding of this continuation.

## 5. What this does and does not establish

- This is ONE wall-clock budget (60s), ONE starting point, ONE direction (upper). The task's full
  Phase 4 spec asks for multiple budgets (30/60/180/~324s) — not run this session given time spent on
  Phase 3's derivation/debugging; flagged as the natural next step, not silently skipped.
- `hybrid`'s instability (§1) is a genuine finding, not a tuning failure to paper over — a future
  session should either (a) skip Hessian updates / reset the quasi-Newton approximation on a
  gradient-source switch (if KNITRO's API exposes a way to signal that), or (b) restrict `hybrid` to
  Hessian modes that don't rely on a consistent secant history (e.g. `productfd`, which recomputes
  Hessian-vector products fresh each time rather than accumulating curvature across iterates).
- Rankings below are read from each config's OWN best-feasible exact-cold-recheck κ, per the task's
  explicit instruction not to rank by KNITRO's own reported optimality alone.
