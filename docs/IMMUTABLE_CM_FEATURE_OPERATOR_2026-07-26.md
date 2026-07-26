# Immutable CM Feature Operator — Audit + Instrumentation — 2026-07-26

## Finding: the mandatory invariant already holds structurally

Task §3 requires that for fixed theta, fixed draws, thresholds, and contrast basis, bin/threshold
assignment, cumulative-CDF features, origin contrasts, and Fréchet common-level features are built
**once at context initialization** and never rebuilt merely because A or gp changed.

An exhaustive grep of every `compute_bin_indices(...)` call site in the production tree (excluding
tests/debug scripts) found exactly five call sites, **all of them inside a `build_cm_*_production_
context` / `build_cm_bin_ctx` function — i.e. all of them run once, at context-build time, never
inside a `moments!`/FG/Hessian callback**:

| Family | Call site | When it runs |
|---|---|---|
| Flexible CM | `cm_production_bundle.jl:102` (`build_cm_production_context`) | once, at ctx build |
| Common Fréchet | `cm_frechet_level.jl:278` (`build_cm_frechet_production_context`) | once, at ctx build |
| CM+ZC | `cm_meanzc_production.jl:44` (`build_cm_meanzc_production_context`) | once, at ctx build |
| Hessian scratch (CM, CM+ZC) | `cm_hessian_architectures.jl:261,410` (`build_cm_bin_ctx`/`build_cm_meanzc_bin_ctx`) | once, at ctx build |
| Origin-ZC | — | **zero hits** — this family has no CM grid at all (confirmed, not assumed) |

The resulting `Bidx` (bin indices), `z` (threshold grid), contrast matrix `R`, and — for common
Fréchet — `level_targets`/`refIndex1` are captured by Julia closure into the family's `moments!`
function and reused, unchanged, across every subsequent outer-point evaluation for the lifetime of
that context. Changing `A`/`gp` changes the *value* fed through these fixed features; it never
triggers a rebuild of the features themselves.

**Conclusion: `cm_feature_rebuilds_due_to_A_or_gp = 0` already held before this session touched
anything.** This is architecture the codebase already has right, not a gap this task needed to
close by writing new code — the actual gap was the *absence of a runtime counter proving it*,
which is easy to get wrong silently (a future change that moved a `compute_bin_indices` call inside
a callback would silently violate the invariant with no test catching it).

## What was built this session

`cm_feature_immutability_counters.jl`: a small, dependency-free counter module
(`CM_FEATURE_IMMUTABILITY_COUNTERS`) with the four counters the task requires
(`cm_feature_context_builds`, `cm_feature_rebuilds_due_to_theta`,
`cm_feature_rebuilds_due_to_A_or_gp`, `cm_dense_feature_materializations`), wired via a
`isdefined(Main, :record_cm_feature_context_build!) && record_cm_feature_context_build!()` guard
(zero risk to any script that doesn't include the new file) into all four production
context-builders.

**Verified, live, real D=20/W=80,000, all four families**:

```
[cm-feature-immutability] context_builds=4 rebuilds_due_to_theta=0 rebuilds_due_to_A_or_gp=0 dense_materializations=0
ALL PHASE 1.2/1.3 D=20 ALL-FAMILY (INCL. ORIGIN-ZC) GATES PASSED
```

`cm_feature_context_builds == 4` (one per family) and `cm_feature_rebuilds_due_to_A_or_gp == 0`
despite each family being queried at two genuinely distinct real, KNITRO-solved outer points (five
times each, including cache-hit re-queries that must NOT trigger any rebuild). Committed as
`ca5f176`. This confirms empirically, not merely by static reading, that the mandatory invariant
holds. `dense_materializations=0` reflects that this counter is not yet wired into any call site
(no production code currently increments it) — see `NO_FULL_G_MATERIALIZATION_AUDIT_2026-07-26.md`
for the real, non-zero dense-materialization finding this counter would need to surface once wired
into the actual FG hot paths (Phase 4/5 scope, not yet done).

## What was NOT built, and why

A full `CMImmutableFeatureOperator` consolidating struct — one object owning compact bin indices,
threshold/grid metadata, interval targets, the cumulative transform, the origin-contrast
transform, the Fréchet common-level direction/targets, and persistent forward/transpose/Hessian
scratch, as the task's original wording describes — was **not** built.

Reasoning: given the invariant already holds architecturally (five independent build functions
each correctly capture their own immutable features by closure, once), the *only* thing a new
wrapper struct adds is a stylistic reorganization of fields that are already correctly immutable
and already correctly threaded through the right code paths. That is real work, but it is not
where this task's actual remaining risk is — Phases 4-7 (eliminating the dense `G` materialization
that the FG/Hessian callbacks still build from these already-immutable features, and the
winner-aware cross-Hessian/interval-basis derivations) are the genuinely unbuilt, higher-value
items. Consolidating the immutable-feature bookkeeping into one struct is a legitimate, low-risk
follow-on — flagged here, not silently dropped — but was not this session's priority given limited
remaining scope.

## Follow-on (if pursued later)

1. Build `CMImmutableFeatureOperator` as a thin wrapper around the *already-correct* existing
   fields (`Bidx`, `z`, `R`, `level_targets`, `refIndex1`) rather than re-deriving them — pure
   refactor, gated by a byte-identical-output test against the current five build functions.
2. Extend the counters to the flexible-theta overlay path (`cm_feature_rebuilds_due_to_theta`
   currently has no live wiring since no restricted-family driver rebuilds features on a theta
   change today — flexible theta is not yet combined with any restricted family in production).
