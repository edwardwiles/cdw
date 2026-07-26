# Five-Family Kill/Resume Report — 2026-07-26

## Status: partial — unrestricted verified end-to-end; restricted families NOT tested this session

## Unrestricted family — verified (Phase 1.1, run 4)

`unrestricted_stage_runner.jl` MODE=resume against its own run 1 checkpoint (`D20CheckpointUnified`)
succeeded: exit=0, no migration error, real continuation of the outer solve. This is a **graceful**
resume (the run completing its budget and a fresh process picking the checkpoint back up), not a
hard process-group kill mid-solve. A genuine `kill -9`-equivalent mid-run test (the task's own
"process-group hard kill" + "fresh-process resume" requirement) was **not performed** this session
for any family — every run this session used its own `maxtime_real` budget to end naturally rather
than being killed externally.

Separately, run 5/6 (Phase 1.1) exercised the **refusal path** for a genuinely pre-existing legacy
V4 checkpoint (real hard-kill artifact from a prior, unrelated 2026-07-24 session) — confirmed a
precise migration-refusal message, not a silent misinterpretation. See the master report's Phase
1.1 section for the run 6 finding (a pre-existing, out-of-scope bug in the frozen legacy driver's
own error-printing path, triggered by that stale checkpoint).

## Restricted families — NOT tested this session

None of the four restricted families (flexible CM, common Fréchet, CM+ZC, origin-ZC) were tested
for checkpoint/resume or kill/resume this session. `run_cm_upper_checkpointed`/
`run_originzc_upper_checkpointed` both have `resume_from` kwargs and `checkpoint_interval_s`
machinery (used with a long, effectively-disabled interval — `1000.0` — in this session's Phase
2/8 benchmark scripts specifically to avoid writing checkpoints mid-benchmark, which means those
scripts incidentally prove nothing about resume). A real kill/resume gate for these four families
— true process-group hard kill mid-solve, then a fresh process resuming from whatever checkpoint
was on disk at that moment — was not attempted.

## Scoped follow-on

1. For each of the five families: launch a real campaign with a short `checkpoint_interval_s`,
   hard-kill the process group mid-solve (`kill -TERM` at the process-group level, or `kill -9` if
   `-TERM` is caught cleanly and exits gracefully — the task wants a genuine "process died with no
   chance to clean up" scenario, not a graceful shutdown), then launch a fresh process with
   `resume_from` pointing at the last on-disk checkpoint and confirm it picks back up correctly
   (same outer point, same screen/counter state where applicable).
2. This is bounded, mechanical work (no new numerical kernels), lower risk than Phases 5-7, and
   could reasonably be completed in a following session focused specifically on it.
