# Flexible theta post-C+ matched comparison — 2026-07-26

Task §13 deliverable: does the theta-derivative speedup restore flexible theta's practical
competitiveness against fixed transformed-A? Same methodology as
`FLEXIBLE_THETA_MATCHED_COMPARISON_2026-07-25.md` (real D=20/W=80,000/seed=20260719, genuine
calibrated start, current production Hessian backend, `run_polish_checkpointed_unified`, 20 Julia
threads, algorithm=auto, 600s per arm), rerun with the reconciled driver now unconditionally
routing flexible-mode theta derivatives through `theta_cplus_secant`.

## Results (raw: `docs/key_results/release_b_postcplus_matched_600s_2026-07-26.csv`)

| Arm | κ (pre-C+, 2026-07-25) | κ (post-C+, 2026-07-26) | n_eval (pre → post) | cold-verify |
|---|---|---|---|---|
| fixed, δ=1 | 0.075535 | 0.075535 | 83 → 86 | agrees to 10 sig figs |
| flexible, δ=1 | 0.073310 | 0.073310 | 58 → **67** | agrees to 10 sig figs |
| fixed, δ=2 | 0.081785 | 0.081785 | 48 → 52 | agrees to 10 sig figs |
| flexible, δ=2 | 0.0738247 | 0.0738247 | 35 → **35** | bit-identical |

**Fixed-mode results are essentially unchanged**, as expected — `theta_cplus` only touches
flexible-mode code paths. Confirms the reconciliation introduced no cross-mode side effects.

**Flexible-mode kappa is essentially unchanged too — matching to 8-12 significant figures at both
deltas — despite the theta-block being 6.43x faster and δ=1 getting 9 MORE evaluations (67 vs 58)
in the identical 600s budget.** At δ=2, the evaluation count didn't even increase (35 both times),
though live diagnostics confirm the theta-derivative cost itself dropped sharply (`theta_wall_total
= 18.752s` across 31 secant calls in the post-C+ δ=2 run — roughly what 31 calls of the OLD
~4.26s/call theta block alone would have cost in *under 5 calls*).

**Fixed transformed-A still beats flexible theta by essentially the same margins as the pre-C+
comparison**: 3.0% at δ=1 (0.075535 vs 0.073310), 11.4% at δ=2 (0.081785 vs 0.073825).

## Interpretation — the hypothesis this task set out to test is not confirmed

`FLEXIBLE_THETA_MATCHED_COMPARISON_2026-07-25.md` and
`FLEXIBLE_THETA_DERIVATIVE_PERFORMANCE_ANALYSIS_2026-07-26.md` hypothesized that flexible theta's
underperformance was *driven by* the brute-force theta-gradient's wall-clock cost starving the
outer search of evaluations at a fixed time budget. This task fixed that cost (6.43x
theta-block speedup, confirmed live in production runs, not just isolated benchmarks) and reran
the identical matched comparison. **The final answer quality (kappa) did not improve** — at δ=1,
more evaluations were available and used, but converged to essentially the same point; at δ=2,
the evaluation count didn't even increase, and the outcome was unchanged to 12 significant
figures.

This is evidence **against** the wall-clock-starvation hypothesis, not for it. A plausible
alternative explanation, consistent with everything observed: KNITRO's outer optimization
trajectory for this problem is not primarily bottlenecked by how many evaluations fit in 600
seconds, but by something structural to the search itself — e.g. the added theta dimension
changing the local optimization landscape's conditioning or step-acceptance behavior in a way
that converges to a similar-quality stationary point regardless of extra evaluation budget. This
task's speedup was a genuine, real, well-verified engineering improvement (see the correctness
and allocation-profile docs) — it just does not appear to be the binding constraint on flexible
theta's practical value at this budget. Determining the *actual* binding constraint (search
algorithm, step acceptance, an intrinsic property of the added dimension) is outside this task's
scope.

## Verdict fields (task §14)

```
FLEXIBLE_THETA_POST_OPTIMIZATION = still_not_faster
```

Not `practical_gain_restored` (kappa did not improve at either delta) and not `inconclusive`
(the comparison ran cleanly, cold-verified, at both deltas, with a clear and consistent null
result — this is a definite finding, not an ambiguous one).
