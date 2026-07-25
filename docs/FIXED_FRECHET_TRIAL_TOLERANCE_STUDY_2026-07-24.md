# Trial-Tolerance Study — CDF-Only Fixed Fréchet — 2026-07-24

Real D=20/W=80,000/L=50/`:exclude_row`, 20-thread Hessian, KNITRO nt=1 (the selected backend per
`KNITRO_THREAD_LINEAR_SOLVER_BENCHMARK_2026-07-24.md`). `opttol`/`opttol_abs` and `ftol` varied
together (`ftol` scaled proportionally: 1e-9/1e-11/1e-13/1e-15 alongside opttol
1e-6/1e-8/1e-10/1e-12); `feastol`/`feastol_abs` left at the production 1e-12 throughout — only
optimality tolerance is loosened, never feasibility.

| Trial tolerance | P1(cold) wall | P1 Δ_dual | P1 \|dual-primal gap\| | P1 max KKT resid | P2(warm) wall | P2 Δ_dual | P2 \|gap\| | P2 max KKT resid |
|---|---|---|---|---|---|---|---|---|
| 1e-6 | 25.08s | 0.0089201808 | 1.677e-09 | 3.960e-07 | 9.99s | 0.0089789031 | 3.366e-10 | 1.423e-09 |
| 1e-8 | 17.25s | 0.0089201808 | 3.202e-11 | 2.482e-10 | 9.29s | 0.0089789031 | 3.367e-10 | 1.424e-09 |
| 1e-10 | 19.35s | 0.0089201808 | 5.574e-13 | 4.321e-12 | 9.55s | 0.0089789031 | 6.136e-12 | 2.587e-11 |
| 1e-12 (current production) | 20.08s | 0.0089201808 | 9.786e-15 | 7.523e-14 | 14.14s | 0.0089789031 | 1.117e-13 | 4.705e-13 |

## Findings

**Δ_dual is stable to 8 significant figures across the entire 1e-6→1e-12 tolerance range at both
P1 and P2** — the loosest trial tolerance tested (1e-6) already gives an outer-constraint-relevant
quantity indistinguishable, to the precision reported, from the strict production tolerance. The
KKT/gap residuals scale down cleanly and monotonically as tolerance tightens (1.7e-9→9.8e-15 at
P1), confirming the solves are behaving as expected under tolerance changes, not hitting an
unrelated stopping criterion.

**Wall-time differences across tolerance levels are within run-to-run noise** on this shared,
contended machine (17-25s at P1, 9-14s at P2 — not monotonic in tolerance, e.g. 1e-8 was faster
than the looser 1e-6 in this run). This is itself an informative finding: because CDF-only solves
are already fast at the *strict* production tolerance (see the threading benchmark doc), loosening
the trial tolerance does not deliver the large timing win it would for a genuinely slow solve (the
`:cdf_power` regime this lever was originally designed for) — the dominant cost driver here is
KNITRO's fixed per-iteration overhead at a small handful of iterations (5-9), not marginal
convergence-tail iterations that a looser tolerance would skip.

## Policy recommendation

Given Δ_dual's demonstrated 8-significant-figure stability, **1e-8 is recommended as the justified
ordinary-trial tolerance** — safely loose enough to have zero measurable effect on any outer
feasibility decision at delta budgets of 0.1/1/2 (Δ_dual differences are ~10⁻⁹, six orders of
magnitude below any of those budgets), while still meaningfully tighter than the loosest tested
level. Per the task's two-tier policy: ordinary trial points may use 1e-8; accepted/new-record
candidates must still be cold-verified under the existing strict 1e-12 tolerance (unchanged,
`cm_frechet_verified_state`/`_threaded`'s cold-verify path already always uses the caller's
`obj.inner_loop_opt`, which the shakedown driver leaves at the production strict file for verify
calls). No arbitrary short time limit is introduced anywhere in this branch.
