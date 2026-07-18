# Full-A D=4: algorithm performance frontier

Phase 7 deliverable. **Status: NOT ATTEMPTED this continuation.** Recorded explicitly, per the task's
transparency requirements, rather than reusing the prior continuation's Phase B numbers as if they
answered this phase's question — they don't, and the distinction matters.

## Why the existing Phase B data does not substitute for this phase

The prior continuation's Phase B (`docs/fullA_d4_final_report.md` §2, `results/fullA_d4/1b2a3a0/
phaseB_hessian_matrix/`) compared genuine BFGS/SR1/L-BFGS/product-findiff/SQP configurations, but **at
a fixed outer-iteration count (maxit=15)**, not fixed wall-clock time. That comparison already showed
the product-findiff control reaches a better kappa (0.171 vs ~0.168-0.169) but takes ~15-20x longer per
iteration (324s vs 17-24s) — meaning the iteration-matched comparison structurally favors whichever
config does the LEAST work per outer step, not whichever config is actually most efficient. The task
explicitly calls this out: "The previous Phase B comparison was iteration-matched, not wall-clock-
matched... Re-run a fair frontier."

## What this phase requires that was not done

1. Common starts and wall-clock budgets (15s/30s/60s/180s/~324s) across barrier-direct+SR1,
   barrier-direct+L-BFGS, barrier-direct+BFGS, product-FD as an explicit control (not default),
   any viable trust-region/active-set config, and an exact-value derivative-free local polish
   comparator.
2. Crossing (selectively) with `optimized-value FD`, `L_fix FD` with separate gamma/A rescaling
   (validated in Phase 6 this continuation — the rescaled variant is directly available in
   `full_aod_diag/d4_exact/phase6_blockwise_gradient_check.jl`'s `g_Lfix_rescaled` construction, not
   yet wired into an actual KNITRO outer-loop driver), and a hybrid refresh policy (the decision rule
   already exists — `derivative_methods.jl::should_refresh` — but has never been wired into a live
   solve, per that function's own docstring).
3. A trajectory (indexed by both wall time and exact-hard-value evaluation count) of best-exact-
   feasible-kappa-so-far, for every configuration, to build the requested Pareto frontier.

## What exists to build on

- `full_aod_diag/d4_exact/phaseB_hessian_algorithm_matrix.jl` is a working, validated KNITRO-driver
  template (confirmed to correctly report requested-vs-effective Hessian mode, fallback detection,
  exact fresh rechecks) — the natural base to extend with a wall-clock stopping criterion (KNITRO
  `maxtime_cpu`/`maxtime_real` options, not `maxit`) and periodic best-feasible-kappa snapshots rather
  than only a single terminal summary.
- Phase 1's cost data (`docs/fullA_performance_profile.md`, `docs/fullA_scaling_projection.md`) gives
  the per-gradient-evaluation costs needed to interpret any frontier found: an `L_fix_FD`-based config
  should complete roughly 3x more outer iterations than a `Delta_FD`-based config in the same wall-
  clock budget at D=4, growing to a much larger ratio at higher D per the scaling projection — this is
  a testable, falsifiable prediction the frontier run would directly check.

## Recommendation for when this is picked up

Do not run this combinatorially (the task explicitly says "selectively rather than combinatorially").
Given Phase 6's finding that `L_fix_FD` is the only cheap method that survives blockwise scrutiny, the
highest-value single run is: barrier-direct+SR1 (cheapest genuine Hessian mode found in the prior
Phase B) crossed with `L_fix_FD`-primary/`Delta_FD`-refresh hybrid, at the ~324s wall-clock budget that
matches the existing product-findiff baseline exactly — one apples-to-apples run, not a full matrix.
