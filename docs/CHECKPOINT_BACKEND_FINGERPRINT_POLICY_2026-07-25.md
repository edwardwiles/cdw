# Checkpoint backend-fingerprint policy (task §9)

## Current state

No CM/CM+meanZC/origin-ZC checkpoint schema (`CMCheckpointV4`/`V6`/`V7`) currently persists
`core_hessian_backend`, `core_hessian_workers`, or a `core_hessian_version` field. Verified by
grep across `cm_checkpoint.jl`, `cm_originzc_checkpoint.jl`, `cm_checkpoint_fingerprint.jl` —
zero occurrences of any of the three field names in this release branch.

## Decision: do not migrate the checkpoint schema for this release

Task §9 is explicit that dense-BLAS and the shared exact winner-pair kernel are **two exact
numerical backends for the identical mathematical H_EE**, not two different scientific contexts —
a checkpoint resumed under one backend and continued under the other must (and, per the D=4/D=20
correctness gates in this release, does) reach bit-for-bit-equivalent-up-to-floating-point results.
There is therefore no scientific-validity reason to reject a resume across a backend change, and
no schema bump is added here.

This is consistent with [[checkpoint-schema-bump-collision-check]] and with the general schema
discipline already documented in `cm_checkpoint.jl`/`cm_originzc_checkpoint.jl` (V4/V5/V6/V7 bumps
are reserved for changes that alter what a checkpoint's fields *mean* scientifically — e.g.
`destination_sample`, `distribution_restriction` — not for swapping which exact code path computed
an already-defined block).

## What IS worth logging (not gating)

`core_hessian_backend`/`core_hessian_workers` are already visible at runtime via
`resolve_core_hessian_counters_manifest()`/`print_core_hessian_counters()` (task §2's runtime
counters) and via the startup manifest's `core_hessian_backend`/`core_hessian_workers`/
`core_hessian_worker_policy` fields (task §3). A future session MAY add these as informational,
non-validated fields to a future checkpoint schema bump that is independently motivated by an
actual scientific-context change — not as a reason to bump the schema on its own. Not done in this
release, per task §9's explicit instruction not to delay the merge for a "cosmetic
backend-fingerprint migration."

## Verified in this release's fresh-process checkpoint/resume gates

The flexible-CM and origin-ZC checkpoint/resume gates (see
`FLEXIBLE_CM_WINNER_PAIR_CHECKPOINT_RESUME_2026-07-25.md` /
`ORIGIN_ZC_WINNER_PAIR_CHECKPOINT_RESUME_2026-07-25.md`) confirm directly that a checkpoint saved
while the winner-pair backend was active resumes cleanly in a fresh process with the winner-pair
backend still active and zero dense fallback — the absence of a backend fingerprint field does not
cause any resume failure, mismatch, or silent backend change.
