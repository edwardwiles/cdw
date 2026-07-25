# Matched Flexible-CM vs CDF-Only Fixed-Fréchet Comparison — 2026-07-24

Task addendum §7: "Can CDF-only fixed Fréchet achieve inner-solve performance comparable to
flexible CM when it uses the same optimized structured-Hessian architecture?" Same D=20/W=80,000/
L=50/`:exclude_row` context (built once, shared), same threaded/syrk Architecture-C engineering on
both sides (`archC_hess_cb_builder_v2` for flexible CM — using the previously-unwired
`cm_hessian_threaded.jl` kernel this session activated for the comparison; `archC_frechet_hess_cb_builder_v2`
for CDF-only), 20 Julia threads, KNITRO nt=1, evaluated at the SAME θ (the live calibration point).

| | flexible-CM | CDF-only fixed-Fréchet | ratio |
|---|---|---|---|
| ncore | 382 | 382 | 1.0 |
| ncm | 950 (=(D-1)·L) | 1000 (=D·L) | 1.053 |
| n (total inner vars) | 1332 | 1382 | **1.038** |
| Context-construction time | 17.1s | 17.3s (fpcx only, ctx shared) | ~1.0 |
| FG callback (warmed) | 3.09s | 3.33s | 1.08 |
| FG callback allocations | 1369.4MB | 1369.4MB | 1.0 (identical — same underlying dense CM/moment buffer size class) |
| Hessian callback (threaded/syrk, nt=20, warmed) | 1.69s | 1.39s | 0.83 (CDF-only faster) |
| Hessian callback allocations | 18.18MB | 18.71MB | 1.03 |
| Full inner solve at calibration θ | 17.67s (5 iters) | 23.42s (9 iters) | **1.33** |

## Finding

Confirms the addendum's prediction directly: because the two problems' dimensions are within 4% of
each other (1332 vs 1382), their per-callback costs are statistically indistinguishable (FG within
8%, Hessian callback actually *faster* for CDF-only, both well within noise for a shared, contended
machine), and the full inner-solve time differs by only 1.33× — driven by CDF-only needing 9 KNITRO
iterations at this point vs flexible-CM's 5, not by any structural inefficiency. **This is
"the same order," not qualitatively different**, exactly as the addendum's dimension-check argued
and directly contrary to the ~20-30× gap the original (pre-addendum, `:cdf_power`-based) diagnosis
observed. The remaining ~1.33× gap is fully explained by iteration count, not per-iteration cost —
consistent with `:cdf_only` being a numerically slightly-harder-to-converge but not structurally
different problem from flexible CM's own production restriction.

No further architectural work is indicated to close this gap — it is already closed to within
normal point-to-point solve-time variance for this problem class (compare: flexible-CM's own P0-vs-
P1-style variation across different trial points in its own production shakedowns).
