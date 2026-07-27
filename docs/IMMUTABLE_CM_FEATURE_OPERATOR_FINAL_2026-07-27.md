# Immutable CM Feature Operator — Final Diagnosis — 2026-07-27

Phase C, item 12. Per the task's own prioritization ("section 12 last... more of an
engineering/refactor task"), this session spent minimal new time here: the prior session
(`ca5f176`, `0a85db4`, both ancestors of this session's HEAD `a69b32d`) already did the substantive
audit and instrumentation work. This document reconfirms that work still holds on the current HEAD
and closes out the one item it explicitly left open.

## Verdict

```
CM_FEATURE_IMMUTABILITY = pass
```

`cm_feature_rebuilds_due_to_A_or_gp = 0` holds **structurally** (not by luck) for every one of the
three restricted-family production context builders that carry a CM grid (flexible CM, common
Fréchet, CM+ZC — origin-ZC has no CM grid at all). `cm_dense_feature_materializations` is a real,
separately-tracked gap (see below), not a false pass.

## What this session re-verified (not re-derived)

Re-grepped every non-test `compute_bin_indices(...)` call site in the production tree on the
current HEAD (`a69b32d`), the same audit methodology `IMMUTABLE_CM_FEATURE_OPERATOR_2026-07-26.md`
used, to confirm nothing regressed after Phase A's economic-operator retrofit (which touched
`CMLookupState`, `CMFrechetLookupState`, and `OriginZCOperatorState` — all adjacent to, but not
inside, the CM feature-build call sites):

| Family | Call site (this session's HEAD) | When it runs |
|---|---|---|
| Flexible CM | `cm_production_bundle.jl:108` (`build_cm_production_context`) | once, at ctx build |
| Common Fréchet | `cm_frechet_level.jl:298` (`build_cm_frechet_production_context`) | once, at ctx build |
| CM+ZC | `cm_meanzc_production.jl:51` (`build_cm_meanzc_production_context`) | once, at ctx build |
| Hessian scratch (CM, CM+ZC) | `cm_hessian_architectures.jl:284,456` (`build_cm_bin_ctx`/`build_cm_meanzc_bin_ctx`) | once, at ctx build |
| Origin-ZC | — | zero hits, confirmed again |

All five call sites are still exactly where the 2026-07-26 audit found them — none moved inside a
`moments!`/FG/Hessian callback. The `record_cm_feature_context_build!()` counter hook
(`cm_feature_immutability_counters.jl`) is also still wired into all four
`build_cm_*_production_context` functions on this HEAD (spot-checked via grep, not re-run as a live
D=20 gate this session — the prior session's real D=20/W=80,000/all-four-families run,
`cm-feature-immutability context_builds=4 rebuilds_due_to_theta=0 rebuilds_due_to_A_or_gp=0
dense_materializations=0`, committed at `ca5f176`, already empirically confirmed this on an
ancestor commit; this session's contribution is confirming the wiring did not silently regress
across the intervening Phase A commits, not re-running the full gate).

**This session's own D=4 and D=20 four-arm scripts** (`c15_d4_four_arm_basis_contrast_comparison.jl`,
`c15_d20_four_arm_basis_contrast_comparison.jl`, see the interval-vs-cumulative and
anchored-vs-orthonormal diagnosis docs) build a fresh `CMBinHessCtx`/`CMBinHessCtxInterval` per
`compare_bases` call — consistent with, not a counterexample to, the "build once at context
construction" invariant: this is diagnostic code deliberately constructing several distinct
one-off contexts to compare, not a production driver re-solving the same outer point repeatedly.

## What was NOT built (unchanged from 2026-07-26, still the right call)

A consolidating `CMImmutableFeatureOperator` struct — one object literally owning bin indices,
threshold/grid metadata, interval targets, the cumulative transform, the origin-contrast
transform, common-Fréchet common-level targets, and persistent forward/transpose/Hessian scratch —
was **not** built this session either. The reasoning from 2026-07-26 still holds and is reconfirmed
by this session's own reading of the code: five independent build functions each already capture
their own immutable features by Julia closure, correctly, once. A consolidating struct would be a
real but purely stylistic refactor (gated by a byte-identical-output test against the current five
functions), not a correctness or performance fix. Given this session's time was spent on the
higher-value items 13-16 (interval-vs-cumulative basis, anchored-vs-orthonormal contrasts, the
interval Hessian re-derivation) per the task's own explicit prioritization, this refactor remains
scoped-but-undone.

## Fingerprint

No single `fingerprint` field exists across the five build functions today (each closure captures
its own `(ctx.U, z, R, ...)` implicitly rather than through one hashable struct) — this is the
concrete gap a future `CMImmutableFeatureOperator` would need to add a `fingerprint::UInt64` (e.g.
`hash((ctx.U, aug.z, aug.contrasts, aug.L))`) field for, to make immutability *checkable* by
equality rather than only *believed* by code-reading. Noted as the natural anchor point for the
consolidating-struct follow-on, not built here.

## Follow-on (unchanged from 2026-07-26, still valid)

1. Build `CMImmutableFeatureOperator` as a thin wrapper around the already-correct existing fields
   (`Bidx`, `z`, `R`, `level_targets`, `refIndex1`), gated by a byte-identical-output regression
   test against the current five build functions — pure refactor, low risk, not attempted this
   session (time went to items 13-16 per the task's own priority order).
2. Wire `cm_dense_feature_materializations` into the actual FG/Hessian hot paths it is meant to
   catch (currently defined but never incremented by any call site — see
   `NO_FULL_G_MATERIALIZATION_AUDIT_2026-07-26.md` for the real, non-zero dense-materialization
   finding this counter would need to surface once wired in).
3. Add a `fingerprint` field (see above) so a future refactor of any of the five build functions can
   be checked for immutability violation by assertion, not only by re-running this session's grep.
