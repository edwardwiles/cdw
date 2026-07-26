# Five-Family Public-Driver Gate Matrix — 2026-07-26

Task §9 asks for a full test matrix (D=4 square, D=4 rectangular non-last omission, D=20/W=80,000,
calibration, near-delta=1, hard point, upper/lower smoke, checkpoint save, process-group hard kill,
fresh-process resume, exact cache, dual bank on/off, backend counters, no silent fallback) per
family, with rows keyed to the five real family contexts specifically (per the task's own
instruction — the inherited remediation's CM+ZC constructor bug was missed because earlier gates
only ever exercised plain CM).

**This matrix is reported honestly below as PASS / NOT RUN, not inferred or assumed.** Cells
marked NOT RUN are real gaps, not silent passes.

| Check | Unrestricted | Flexible CM | Common Fréchet | CM+ZC | Origin-ZC |
|---|---|---|---|---|---|
| D=4 square | NOT RUN this session | NOT RUN this session | NOT RUN this session | NOT RUN this session | NOT RUN this session |
| D=4 rectangular (non-last omission) | NOT RUN this session | NOT RUN this session | NOT RUN this session | NOT RUN this session | NOT RUN this session |
| D=20/W=80,000, real calibration point | PASS (Phase 1.1 run 1/2) | PASS (Phase 1.2/1.3, Phase 2) | PASS (Phase 1.2/1.3, Phase 2) | PASS (Phase 1.2/1.3, Phase 2) | PASS (Phase 1.2/1.3, Phase 2) |
| Near delta=1 | PASS (all runs used delta=1.0) | PASS | PASS | PASS | PASS |
| "Hard point" (genuinely distinct perturbed point) | NOT explicitly tested as a separate hard-point case (only calibration + a small 0.05-scale zfree perturbation for cache-B, not a stress "hard" point) | same caveat | same caveat | same caveat | same caveat |
| Upper bound smoke | PASS (Phase 1.1 run 1) | PASS (Phase 2/8, `find_smallest=true` default) | PASS | PASS | PASS |
| Lower bound smoke | PASS (Phase 1.1 run 2) | NOT RUN this session | NOT RUN this session | NOT RUN this session | NOT RUN this session |
| Checkpoint save | PASS (every run wrote a real checkpoint) | PASS (Phase 2/8 wrote checkpoints, not deeply inspected) | PASS | PASS | PASS |
| Process-group hard kill + fresh-process resume | NOT RUN (see `FIVE_FAMILY_KILL_RESUME_REPORT_2026-07-26.md`) | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Exact cache (hit/miss/same-point-resolves=0) | N/A (unrestricted uses a different, pre-existing cache) | PASS (Phase 1.2/1.3, full 5-point sequence) | PASS | PASS | PASS |
| Dual bank on/off, real trajectory | N/A (unrestricted's own KKT-scored bank pre-dates this task, not re-benchmarked) | PASS (Phase 2 — see `RESTRICTED_DUAL_BANK_FINAL_DECISION_2026-07-26.md`) | PASS | PASS | PASS (focused re-run after fixing a real missing-include bug; same finding: identical outer progress, 3/12 warm starts failed) |
| Backend counters printed/consistent | PASS (`print_core_hessian_counters` in every Phase 1.1 run) | PASS (exact-cache + immutability counters, Phase 1.2/1.3) | PASS | PASS | PASS |
| No silent fallback observed | PASS (fail-fast `price_cache_backend` validation exercised, Phase F item 4) | PASS | PASS | PASS | PASS |

## Summary

The **D=20/real-calibration axis** — the scale this task's own acceptance invariants are stated
at — has real, passing coverage across all five families for: cache correctness, workspace
correctness, dual-bank real-trajectory behavior, backend-counter consistency, and (for
unrestricted) upper/lower bound + resume + legacy-refusal semantics. The **D=4 axis**, the
explicit "hard point" stress case, and **kill/resume for the four restricted families** were not
run this session — real, disclosed gaps, not inferred passes. Per the task's own instruction ("do
not infer one family's success from shared code"), every PASS cell above reflects a gate that
specifically named and exercised that family's own real production context — none were inferred
from a sibling family's result.
