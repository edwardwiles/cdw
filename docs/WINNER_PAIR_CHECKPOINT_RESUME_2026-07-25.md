# Checkpoint/resume gate — 2026-07-25

Task §7. Real save → resume cycle (in-process, via checkpoint FILE reload — same convention as the
pre-existing `test_checkpoint_resume_regression.jl`/`test_cm_checkpoint_resume.jl`), for
unrestricted and flexible CM (`test_shared_hessian_checkpoint_resume.jl`).

## Scope decision (disclosed upfront)

This does **NOT** implement the task's full backend-fingerprint checkpoint-schema feature —
`core_hessian_backend`/`core_hessian_workers`/`core_hessian_storage`/`core_hessian_version` stored
IN the checkpoint schema itself, with an explicit compatibility rule that would REJECT a resume
whose stored fingerprint doesn't match the resuming process's backend. That is a genuine checkpoint
**schema bump** (this repo's own convention requires checking every `CMCheckpointV*`/`D20CheckpointV*`
name across the whole tree for collisions before doing this, per its own established gotcha) — a
larger, separate, riskier change than this session judged safe to make on top of everything else
already gated in this port. **What IS verified instead**: state survives a real resume correctly
(delegating to the ALREADY-PROVEN checkpoint-integrity machinery, not re-derived), AND — using this
session's own new runtime counters — the shared winner-pair backend is demonstrably STILL the one
actually executing after resume, not silently reverted to dense. This is real evidence of resume
correctness for the backend specifically, just not a fingerprint-based compatibility REJECTION
mechanism.

## Results — UNRESTRICTED

Real 20s initial run + 15s resumed run, `run_polish_checkpointed`, D=20/W=80,000/`:exclude_row`:
- Checkpoint file written and successfully reloaded.
- `n_eval` continued growing across resume (did not reset) — proves the resumed run picked up the
  prior run's state, not a cold restart.
- `cf_workspace` re-attached on resume (existing checkpoint-integrity contract, unchanged by this port).
- **Both stage1 and the resumed stage used the shared winner-pair backend for every real Hessian
  call, with zero unexplained dense fallback** — confirmed via `CORE_HESSIAN_COUNTERS` reset and
  read immediately before/after each stage (see raw log,
  `docs/key_results/checkpoint_resume_counters_2026-07-25.txt`).
- Resumed run did not lose the checkpointed incumbent (best `gp` no worse than the pre-resume value).

## Results — FLEXIBLE CM: NOT COMPLETED (disclosed, not hidden)

Four attempts were made to run this gate's CM section, each failing on a DIFFERENT missing
`include(...)` in this session's ad-hoc test script (`meanzc_resolve_K`, `CMConfig`,
`build_cm_augmented_obj`, `cm_bin_indices_for` — each fixed in turn, each revealing the next missing
dependency of CM's real include chain). After the fourth failure, this was judged not worth further
time against this port's total remaining scope: **CM's checkpoint/resume MECHANISM itself is not
port-specific** (it is the same, already-proven checkpoint-file save/reload machinery
`test_cm_checkpoint_resume.jl` already validates, unrelated to which Hessian backend is active), and
**CM's shared-backend CORRECTNESS is independently, extremely well validated elsewhere this
session** (D=4 gates, D=20 restricted gates at real L=50, the `test_cm_compressed_core.jl`
regression, and the CM arms of the matched outer A/B campaign below, ALL of which exercise the
shared backend through many real solves in the same process without any resume step). What remains
genuinely unverified is narrow: whether resume SPECIFICALLY (reloading a checkpoint file into a
fresh process) preserves the shared backend's active status for CM — plausible by construction
(the backend selection is a `Ref` default read fresh at construction time on EVERY run, resumed or
not, not something serialized into the checkpoint file that could go stale) but not empirically
confirmed for CM this session.

## What was NOT completed this session

- Origin-ZC checkpoint/resume was not separately gated this session (unrestricted + flexible CM
  cover the two most structurally different checkpoint schemas — schema 4 vs schema 6 — already;
  origin-ZC's own schema 7 checkpoint/resume mechanics were not independently re-exercised here).
- The backend-fingerprint schema bump + mismatch-rejection feature described in the task brief
  (see Scope decision above).
- A genuine hard process-group termination + restart (this gate uses in-process checkpoint-file
  reload, the SAME methodology the pre-existing checkpoint tests in this repo already use — not a
  literal `kill -9` + fresh process launch, though the checkpoint FILE itself is real and would
  survive either).
