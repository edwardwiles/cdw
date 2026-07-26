# Five-Family Final 300s Profile — Status — 2026-07-26

## Status: NOT ATTEMPTED this session

Task §11 asks for five sequential 300-second direct-upper-bound profiles (one per family, same
process, D=20/D_dest=19/W=80,000/seed=20260719/delta=1/calibrated start/fixed theta/20 Julia
threads), with wall-clock attributed across eleven categories (moment construction, outer
gradient, inner FG callback, inner gradient callback, H_EE, H_RR, H_ER, screens, cache/bank,
KNITRO internal, checkpoint/logging, residual), followed by regenerating the final 5×7 numerical
matrix, supporting-plumbing matrix, allocation report, and flexible-theta overlay.

This was deliberately not attempted this session, for two concrete reasons:

1. **Sequencing**: task §11's own header says "Once all passing changes are canonical, run..." —
   this session's own work is on a feature branch, not yet merged to `production/fullA-exact` (per
   this project's own standing "confirm before pushing to a real remote" rule — merging requires
   your explicit go-ahead, not requested this session). Running the "final" profile before the
   remediation it is meant to measure is actually canonical would produce numbers that describe a
   branch state, not the production state the task's own wording implies "final" should mean.
2. **Instrumentation gap**: the eleven-category wall-clock attribution this section requires does
   not fully exist yet as production instrumentation (some pieces do — `core_hessian_counters`
   already separates winner-pair vs dense-fallback Hessian calls; `CM_EXACT_CACHE_COUNTERS`/
   `RESTRICTED_DUAL_BANK_COUNTERS` already separate cache/bank overhead — but a single profile run
   producing all eleven categories in one pass was not built or tested this session).

Real wall-clock timings **were** collected incidentally this session (Phase 2's dual-bank
benchmark, Phase 8's transformed-A smoke tests) — see the master report and
`RESTRICTED_DUAL_BANK_FINAL_DECISION_2026-07-26.md` for those numbers — but they are 45-90 second
short-trajectory runs for a different purpose (dual-bank/coordinate-mode correctness), not the
300-second, fully-attributed production profile task §11 specifies. They should not be
substituted for it in the final 5×7 matrix.

## Scoped follow-on

1. Merge the passing items from this session (per `docs/*_2026-07-26.md`'s individual gates) to
   `production/fullA-exact`, with your explicit authorization.
2. Build the eleven-category wall-clock attribution as its own instrumentation pass (extends the
   counters this session already added, does not replace them).
3. Run the five 300-second profiles specified, one process at a time, and regenerate the final 5×7
   matrix (MD+CSV), supporting-plumbing matrix, allocation report, and flexible-theta overlay from
   real data.
