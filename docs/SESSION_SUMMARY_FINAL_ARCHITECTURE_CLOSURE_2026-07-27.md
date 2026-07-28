# Session Summary — Final Architecture Closure — 2026-07-27/28

## Scope and status at a glance

```
Task: final architecture closure for the "full A_od" gravity-elimination estimator (5 families) --
default flips, dense-composite-G elimination, CM+ZC block-partition refactor, counter
instrumentation, decisive gates, and canonical production merge.

STATUS: 7 of the task's numbered goals complete and merged into
        release/final-architecture-closure-and-production-merge-2026-07-27
        (HEAD 47fb98e as of this doc). Goal 10 complete but NOT YET merged into that branch
        (lives on agent/skip-cm-fill-ref-removal-2026-07-27, HEAD bc6ffdb) -- held back per
        explicit user direction to document first. Canonical production merge NOT YET performed.
```

## Branch/commit map

| Branch | HEAD | Status |
|---|---|---|
| `production/fullA-exact` (canonical) | `f1fa8e7` | Unchanged all session (confirmed via `git fetch`, both local and `origin`) |
| `release/final-architecture-closure-and-production-merge-2026-07-27` | `47fb98e` | Integration branch: reconciliation + Goals 3/4/5/6/7/8/9/11/12, all merged and gated |
| `agent/skip-cm-fill-ref-removal-2026-07-27` | `bc6ffdb` | Goal 10, complete, gated, NOT YET merged into the integration branch |

## What was done, goal by goal

### Reconciliation (task §2)
Confirmed canonical `production/fullA-exact` had not advanced beyond the inherited base `f1fa8e7`
(both local and `origin` fetched and checked). Found and deliberately did NOT merge a sibling
branch (`feature/shared-economic-moment-state-builder-2026-07-27`) with real file-overlap risk for
a marginal allocation win — documented as a deferred item, not silently dropped.
(`docs/FINAL_ARCHITECTURE_CLOSURE_RECONCILIATION_2026-07-27.md`)

### Goal 3 — common-Fréchet operator FG default flip
`CM_FRECHET_INNER_FG_BACKEND_DEFAULT` flipped `:dense_reference` → `:cm_frechet_lookup`.
D=20/W=80,000/L=50, anchored + orthonormal, calibration + perturbed point, all PASS. Dispatched to
a background agent; result reviewed and integrated by the main session.
(`docs/COMMON_FRECHET_OPERATOR_FG_DEFAULT_FLIP_2026-07-27.md`)

### Goal 4 — unrestricted shared A-gradient default flip
`resolve_price_cache_backend`'s no-kwarg default changed `:cplus` → `:shared`
(`economic_A_gradient!`, the same shared entry point the other 4 families already used). D=4 + real
D=20 equivalence bit-identical; real KNITRO driver smoke 9/9 PASS, including a self-caught and
fixed flaw in the gate's own original equivalence check (a maxtime-bounded KNITRO trajectory is not
a reliable oracle across sequential runs — replaced with a deterministic unit-style check).
(`docs/UNRESTRICTED_SHARED_A_GRADIENT_DEFAULT_FLIP_2026-07-27.md`)

### Goals 5–6 — explicit `moment_representation` dispatch
New `MOMENT_REPRESENTATION::Ref{Symbol}` selector (`:operator`/`:dense_reference`) gating whether
generic inner-solve setup fills family-specific restriction columns nothing downstream reads.
Wired for flexible-CM (safe); explicitly NOT wired for common-Fréchet at the time (correctly
flagged as unsafe given the Hessian dependency — see Goal 10 below, which fully explains why).
(`docs/OPERATOR_MODE_NO_COMPOSITE_G_SETUP_2026-07-27.md`)

### Goal 7 — CM+ZC E/C/Z block partition + new `H_CZ` primitive
Completed the `G=[E|C|Z]` partition CM+ZC's Hessian was missing. New `bin_zc_cross_hessian_fill!`
(`H_CZ = C'SZ`, bin-index-keyed, never reads dense CM columns). D=4 (K1/K2, both contrasts) and
real D=20 (both contrasts, calib + perturbed, serial + threaded) ALL PASS at machine precision
(max|ΔH| 6.4e-13 to 9.0e-13). Isolated production-only counter check confirms
`dense_cross_hessian_calls=0` at CM+ZC's real production default.
(`docs/CM_MEANZC_BLOCK_PARTITION_AND_HCZ_RELEASE_2026-07-27.md`)

### Goal 8 — shared direct `H_ZZ` (CM+ZC and ZC-only)
New `zc_restriction_gram!` (`H_ZZ = Z'SZ` from raw ZC feature state, not `obj.H` columns), one
routine shared by both families. Same gate suite, same machine-precision results.
(`docs/SHARED_ZC_HRR_DIRECT_RELEASE_2026-07-27.md`)

### Goal 9 — unrestricted dense-reporting removal
`evaluate_fullA_fast_compressed` unconditionally materialized dense `obj.H` + `select_G_from_H` +
a full constraint-evaluation call solely to serve three reporting-only fields
(`gravity_raw`/`benchmark_unweighted_moment_mean`/`max_abs_moment_resid`) not consumed by any
admission/verification logic. New `dense_reference_diagnostics::Bool=false` kwarg skips that block
by default. Verification-critical fields proven identical regardless of the flag (dedicated test);
pre-existing 25-comparison dense-vs-compressed parity suite re-run in full with the flag threaded
through, ALL PASS.
(`docs/UNRESTRICTED_DENSE_REPORTING_REMOVAL_2026-07-27.md`)

### Goal 10 — `skip_cm_fill_ref` removal — see dedicated report
**This is the goal with a real, substantive finding, not just a mechanical refactor.** Full report:
`agent/skip-cm-fill-ref-removal-2026-07-27`'s own
`docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md` (also pushed to Dropbox alongside this
summary — see below). Short version: the mutable-`Ref` anti-pattern IS removed (Goal 10's literal
ask). A separate hypothesis — that the *underlying skip optimization* was now safe to re-enable —
was tested empirically at real D=20 scale and found **wrong** for both flexible-CM and
common-Fréchet (a real `nStatus=-400` KNITRO solve failure at non-calibration points, previously
undetected for flexible-CM because it had only ever been tested at the calibration point). Root
cause fully traced to `_archC_prep_for_hessian!`'s unconditional dense recompute of a shared
per-draw Hessian weight vector, and confirmed — by direct comparison against the unrestricted
family's own production Hessian callback, which does NOT do this — that the dependency is
architecturally unnecessary, not inherent. Both families reverted to the always-safe, always-fill
behavior (bit-identical to pre-investigation behavior, confirmed via a final real D=20 re-run). The
real fix (a lookup-aware `_archC_prep_for_hessian!` variant) is fully specified in that document but
NOT implemented this session, by explicit user direction, in favor of a complete, precise writeup.

### Goal 11 — counter instrumentation
Added `direct_restriction_hessian_calls` (visibility counter for family-owned structured H_RR
computations) and `no_dense_g_report_task_names()` (task's own exact 13 counter names, same
underlying state as the existing `no_dense_g_report()`, no repo-wide rename). Verified against a
real D=4 5-family run.

### Goal 12 — final 5×7 architecture matrix
`docs/FINAL_FIVE_BY_SEVEN_ARCHITECTURE_MATRIX_2026-07-27.md`/`.csv`. **Note: written before the
Goal 10 correction landed** — its `H_RR` row for flexible-CM/common-Fréchet should be read
alongside the Goal 10 report above; it does not yet reflect the corrected, more precise
characterization of the CM/level fill as "currently necessary, real fix specified but not
implemented" rather than "unchanged, pre-existing." Refreshing this matrix is a listed next step,
not done as part of this push (see below).

## What genuinely changed production behavior this session (the actual diffs, not process)

1. Common-Fréchet: `:dense_reference` → `:cm_frechet_lookup` FG default.
2. Unrestricted: `:cplus` → `:shared` A-gradient default.
3. CM+ZC: new `H_CZ`, new shared `H_ZZ` — both now `dense_cross_hessian_calls=0` in production
   (previously nonzero via a dense fallback).
4. ZC-only: `H_ZZ` now via the same shared `zc_restriction_gram!` (previously its own dense calc
   reading `obj.H` columns).
5. Unrestricted: three reporting-only fields no longer force a dense materialization by default.
6. `skip_cm_fill_ref` (mutable `Ref`) removed everywhere; the CM/level dense fill it used to
   conditionally skip remains unconditional (i.e. **no net behavior change** here versus the
   session's starting state — the investigation concluded "no", not "yes", after a real test).

## What was investigated and found NOT safe to change (equally important to record)

- Flexible-CM's and common-Fréchet's CM/level moment-column fill cannot currently be skipped ahead
  of the Hessian callback without corrupting the shared per-draw weight vector — see Goal 10.
- A sibling branch with an already-built shared economic moment-state builder was found but not
  merged, due to real file-overlap conflict risk for a marginal allocation gain (reconciliation §).

## Explicitly NOT done this session (by user direction, stopped mid-plan)

- The lookup-aware `_archC_prep_for_hessian!` fix specified in the Goal 10 report.
- Merging `agent/skip-cm-fill-ref-removal-2026-07-27` into the integration branch.
- Refreshing the 5×7 matrix to reflect the Goal 10 correction.
- The remaining decisive D=4/D=20 release gates consolidation (task §13).
- The canonical merge to `production/fullA-exact`, tagging, push, and post-merge smoke (task §14).
- The user-requested five parallel δ ∈ {0.1, 0.5, 1, 2} production chains from the calibrated A*
  starting point.
- The final master report / SHA256 manifest / final verdict block the task's own deliverable list
  asks for.

All of the above remain open, tracked, and ready to resume from this exact point.

## Provenance

- Working directory: `/bbkinghome/edav/gravity_robustness` (worktrees under `worktrees/`).
- `release/final-architecture-closure-and-production-merge-2026-07-27` @ `47fb98e`.
- `agent/skip-cm-fill-ref-removal-2026-07-27` @ `bc6ffdb`.
- Canonical `production/fullA-exact` @ `f1fa8e7` (unchanged, confirmed against `origin`).
- Julia 1.12.6 via juliaup; real KNITRO throughout (no simulated/mocked solves at any point this
  session); `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` set for every run.
