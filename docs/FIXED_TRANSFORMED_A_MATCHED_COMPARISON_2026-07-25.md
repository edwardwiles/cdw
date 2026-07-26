# Fixed transformed-A matched comparison — production port, 2026-07-25

Task §17 deliverable. Post-rebase/reconciliation (Phase 1), post-gate (Phase 2) real matched
comparison through the actual public driver (`run_polish_checkpointed_unified`), current
production Hessian backend, D=20/D_dest=19/W=80,000/seed=20260719, genuine calibrated start,
20 Julia threads, `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` (project hard cap). 300s per arm,
`price_cache_backend=:cplus`, algorithm=auto (no `pin_outer_algorithm` pin — matching this task's
own default, not the separate matched-algorithm-A/B methodology from the concurrent Hessian task).

## Results (raw log/CSV: `docs/key_results/release_a_matched_300s_2026-07-25.csv`)

| Coordinate mode | delta | kappa | n_eval | wall (s) | KNITRO status | cold-verify |
|---|---|---|---|---|---|---|
| legacy_z | 1 | 0.067725 | 37 | 302.8 | -411 (time limit, infeasible-side) | Delta_dual agrees to 9 sig figs |
| powered_aspace | 1 | 0.073706 | 40 | 335.5 | -401 (time limit, feasible-side) | agrees to 12 sig figs |
| legacy_z | 2 | 0.070179 | 21 | 304.4 | -411 | agrees to 11 sig figs |
| powered_aspace | 2 | 0.071181 | 20 | 304.7 | -411 | agrees to 12 sig figs |

**Transformed-A beats legacy-z at both budgets: +8.8% at delta=1 (0.073706 vs 0.067725), +1.4% at
delta=2 (0.071181 vs 0.070179).**

**All 4 arms finished cleanly — zero KNITRO hangs.** This is a materially better outcome than
every prior attempt at this exact comparison on the source branch (`docs/
UNIFIED_COORDINATE_LAYOUT_ADDENDUM_2026-07-25.md` §6: 2 of 4 runs hit an unresolved hang-past-
timeout there). Not claimed as a root-cause fix (no code change targeted the hang specifically —
see `FIXED_TRANSFORMED_A_PRODUCTION_PORT_2026-07-25.md`'s KNITRO-reliability section for the
honest framing: not reproduced in this session's 8/8 clean runs, cause not independently
confirmed absent).

Both `powered_aspace` numbers land close to (delta=1: +13.9% higher) or exactly at (delta=2:
0.071181 vs the source branch's own `0.0711808...` — effectively the same point, both cold-
verified) the source branch's prior measurements — consistent with this being the same underlying
optimization landscape explored with a full, unconfounded budget rather than a different result
from the reconciliation itself. The delta=1 `powered_aspace` number here (0.073706) is materially
higher than the source branch's confounded 9-eval delta=1 legacy_z run or its own 20-eval
`powered_aspace` delta=1 run (0.0710) — expected, since this run got a genuinely full, unconfounded
300s (n_eval=40 vs 20).

## Verdict

Equivalence gates (Phase 2): ALL PASS at machine precision. Public-driver gates: ALL PASS.
Checkpoint/resume: ALL PASS. Matched comparison: transformed-A **beats** legacy-z at both budgets,
non-confounded, cold-verified, both arms completed without truncation-by-hang. This clears the
brief's §17 bar ("verified progress improves or is within 5% of legacy... because it is an exact
reparameterization, do not require proof of a different final optimum") with room to spare — this
is not a marginal 5%-tolerance pass, it is a clean win at both tested budgets.

**FIXED_TRANSFORMED_A = PORT_READY_NOT_MERGED** (rebased onto canonical production, all gates
pass, matched comparison favorable — held back only pending the explicit merge confirmation this
task's standing instructions require before any push to `cdw/production/fullA-exact`, not for any
technical gap).
