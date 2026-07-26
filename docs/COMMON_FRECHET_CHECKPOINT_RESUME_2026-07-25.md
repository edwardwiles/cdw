# Common-Fréchet checkpoint/resume — 2026-07-25/26 (Part V)

New checkpoint schema `CMCheckpointV8` — full design in
`COMMON_FRECHET_CM_DRIVER_PORT_2026-07-25.md`. This document is the resume-gate validation record.

## Test (`smoke_frechet_resume.jl`), real D=20/W=80,000 checkpoint from the live smoke run

| Check | Result |
|---|---|
| Resume under mismatched `marginal_restriction` (`:common_flexible` vs the checkpoint's `:common_frechet`) | **PASS** — correctly refused, error message: `"marginal_restriction MISMATCH on resume ... refusing to resume under a different restriction family"` |
| Resume under matched `marginal_restriction` | **PASS** — real KNITRO resume, real continued execution |
| `n_eval` continues from checkpoint state | **PASS** — `10 → 12` (not reset to 0) |
| `n_grad` continues from checkpoint state | **PASS** — `6 → 8` |
| Resumed checkpoint still reports correct mode | **PASS** — `marginal_restriction=common_frechet` after resume+re-save |
| Resumed checkpoint schema | **PASS** — `schema=8` (`CM_CHECKPOINT_SCHEMA`) |

6/6 PASS.

## What this validates vs. what it doesn't

This is a **graceful resume** test (load an already-written checkpoint into a fresh
`run_cm_upper_checkpointed` call) — it exercises the real serialize/deserialize path
(`save_cm_checkpoint`/`load_cm_checkpoint`), the schema-upgrade dispatch (confirmed reachable, not
exercised against an actual pre-schema-8 file this session since none exists yet for
`:common_frechet`), and the mismatch-refusal guard, all for real.

**Not run this session**: a genuine mid-run `SIGKILL`/process-group-kill test (the more rigorous
form the winner-pair release's own checkpoint gate used — see
`shared-winner-pair-production-merge-2026-07-26` memory for that precedent). The graceful-resume
test above gives strong evidence the mechanism works (same underlying `do_checkpoint`/
`save_cm_checkpoint` machinery flexible CM already uses in production, extended by exactly one new
field), but a true kill-mid-write test would additionally confirm the atomic-write discipline
(`.tmp` file + `mv`) protects `:common_frechet` checkpoints the same way it already does for plain
CM. Disclosed follow-up, not attempted given the session's wall-clock budget.

## Verdict

`CHECKPOINT_RESUME = pass_graceful_not_kill_tested`.
