# Flexible-theta matched comparison — production port, 2026-07-25

Task §18 deliverable. Same methodology as
`FIXED_TRANSFORMED_A_MATCHED_COMPARISON_2026-07-25.md` (real D=20/W=80,000/seed=20260719,
genuine calibrated start, current production Hessian backend, `run_polish_checkpointed_unified`,
20 Julia threads, algorithm=auto), 600s per arm, comparing `fixed_aspace` (theta held at
calibration) vs `flexible_aspace` (theta searched, box `[2(σ-1)·1.05, 3·theta_star]`).

## Results (raw log/CSV: `docs/key_results/release_b_matched_600s_2026-07-25.csv`)

| Mode | delta | kappa | n_eval | wall (s) | KNITRO status | cold-verify |
|---|---|---|---|---|---|---|
| fixed | 1 | 0.075535 | 83 | 605.7 | -411 | agrees to 10 sig figs |
| flexible | 1 | 0.073310 | 58 | 613.2 | -411 | agrees to 10 sig figs |
| fixed | 2 | 0.081785 | 48 | 607.0 | -401 | agrees to 10 sig figs |
| flexible | 2 | 0.073825 | 35 | 610.7 | -411 | bit-identical |

**This reverses the source branch's own finding and the practical value this task's brief cites
in its "Established result" table.** In this clean, non-confounded, cold-verified,
zero-KNITRO-hang, full-600s-budget run: **fixed transformed-A beats flexible theta at both
budgets** — by 3.0% at delta=1 (0.075535 vs 0.073310) and by 11.4% at delta=2 (0.081785 vs
0.073825). This is stated plainly, not minimized: the brief's own cited flexible-theta numbers
(0.06588/0.07382) were single, session-specific matched runs on a stale driver, one of which was
itself confounded by a KNITRO hang (§ below); this run is the first clean, fully-budgeted,
hang-free, reconciled-driver measurement of this specific comparison, and it points the other way
at both deltas tested.

Notably, `flexible` delta=2's kappa here (0.073825) is essentially identical to the source
branch's own confounded delta=2 flexible number (0.073825 there too, per
`FLEXIBLE_THETA_THREE_ARM_FOLLOWUP_2026-07-25.md`) — flexible mode's result did not change from
the reconciliation. What changed is `fixed`'s own throughput: **n_eval=48 here vs the source
branch's un-reconciled runs typically in the 20-40 range at this budget** (see e.g. Release A's
own 300s `fixed_aspace` runs at n_eval=20/40) — consistent with Phase 1's workspace-reuse
hardening (`CompressedFactualWorkspace`/canonical-price-precompute/hard-score-B, ported forward
from the concurrent allocation/Hessian production release) disproportionately benefiting fixed
mode, which pays that per-eval cost far more times per unit wall-clock than flexible mode does.

**Why flexible mode gets systematically fewer evaluations per unit wall-clock, independent of the
result above:** every flexible-mode `cb_G!` call does the fixed-dual central-difference theta
secant (two extra `theta_fixed_dual_delta_pivot_A` recomputations of `ctx.obj.H`, §6 of
`TRANSFORMED_A_COORDINATE_MATHEMATICS_2026-07-25.md`) on top of the shared C+ gradient kernel
every mode pays for — a genuine, structural per-callback cost that scales with wall-clock budget,
not iteration count. At matched WALL-CLOCK budgets (as this task specifies, not matched
evaluation-count budgets), this cost is a real headwind for flexible theta that fixed transformed-A
does not pay.

## KNITRO reliability

**Zero hangs in this 8-run campaign** (4 Release A + 4 Release B, including both flexible arms).
Not claimed as a fix — no code change in this port specifically targeted the hang the source
branch observed (`gravity-robustness-knitro-hang-past-timeout.md`; also reproduced in fixed-mode
legacy_z vs powered_aspace runs on the source branch, so it is not flexible-theta-specific). Stated
honestly: not reproduced in this session's runs, root cause not independently re-confirmed absent.

## Decision on the brief's optional 2-hour delta=2 follow-up

**Not run.** The brief's own conditional is explicit: "If favorable and stable, run one longer
two-hour delta=2 comparison." This result is neither — flexible loses at delta=2 by 11.4% in a
clean run, the opposite of favorable. Spending a 2-hour compute budget to further confirm an
already-unfavorable result would not change the verdict below and was judged not worth the
compute cost; the honest, timely reporting of a real reversal is more valuable than a longer
run defending a conclusion the fresh data does not currently support.

## Verdict

Theta derivative validation, cache/dual-bank fix, checkpoint/resume: all pass (Phase 1/2 — see
`FLEXIBLE_THETA_PRODUCTION_PORT_2026-07-25.md`). The matched comparison, however, does **not**
show non-regression: fixed transformed-A wins cleanly at both tested budgets in this session's
non-confounded runs. Per the brief's own Release B merge rule ("matched comparisons show
non-regression or practical gain"), this specific evidence does not clear that bar right now.

**FLEXIBLE_THETA = PORT_READY_NOT_MERGED, with an unfavorable matched-comparison result that
should block promoting it to default/recommended status until re-examined** (not a
`FAILED_THETA_DERIVATIVE`/`FAILED_CACHE_CHECKPOINT` failure — those all pass — but the practical
value case this release exists to make is not currently supported by this session's cleanest
available data). Recommend: either accept flexible theta as a documented opt-in with unproven
current practical benefit, or investigate the per-callback cost gap (e.g. a cheaper theta
derivative, or a coarser secant step schedule) before recommending it for real campaigns. Not
recommended for promotion to any default configuration.
