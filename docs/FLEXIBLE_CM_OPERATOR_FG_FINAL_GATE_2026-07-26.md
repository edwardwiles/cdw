# Flexible CM Operator FG — Final Gate — 2026-07-26

This branch did not modify flexible CM's own inner FG kernel (`CMLookupState`,
`cm_lookup_kernels.jl`/`cm_lookup_production.jl`) — inherited unchanged from
`port/finish-five-family-optimization-stack-2026-07-26@217e91b` per this branch's own commit-0
classification (`ADOPT`, D=4+D=20 ALL PASS at adoption time, allocation at exact parity with dense,
1.108x-1.616x faster than `:dense_reference` at every tested thread count 1/4/8/10/20). Already
`CM_INNER_FG_BACKEND_DEFAULT[] = :cm_lookup` before this branch started.

## What this branch did NOT do (explicit scope boundary, see
`SHARED_ECONOMIC_FG_OPERATOR_DESIGN_2026-07-26.md` §5)

`CMLookupState`'s own economic-core (E) block is **still dense** — it computes `E*λ_core` via
`BLAS.gemv!` against `obj.H`'s core columns, not the shared `economic_forward!`/
`economic_transpose!` operator this branch built. Retrofitting it was explicitly deferred: the
inherited handoff's own priority order says the full-unification "harder ask" should wait "until B
is real for at least one restricted family" and warns it "is real, correctness-sensitive
refactoring work... should not be rushed" — this branch instead built the shared operator on the
two families that had **zero** prior FG alternative (origin-ZC, CM+ZC), establishing the pattern
without touching flexible CM's own already-shipped-as-default kernel.

## Validation this session (regression check only, no new capability)

`test_phaseB1_cmlookup_production_correctness.jl d4` re-run at the end of this session (after
layering on every other change this branch made) — **ALL PASS**, confirming no regression.

## Backend status (unchanged from the inherited port)

```
ECONOMIC_FG_BACKEND[flexible_cm] = dense (unchanged this branch)
RESTRICTION_FG_BACKEND[flexible_cm] = cm_lookup (CM_INNER_FG_BACKEND_DEFAULT[], inherited default)
VERIFICATION_BACKEND[flexible_cm] = dense_reference (skip_cm_fill_ref toggle, unchanged this branch)
```

## Recommended follow-on (not started this session)

Retrofit `CMLookupState`'s own E-block to `economic_forward!`/`economic_transpose!` against
`cctx.core_cf_ref[]` (the SAME `cf` the shared H_EE Hessian backend and this branch's own
origin-ZC/CM+ZC operators already consume). This is the addendum's "harder ask" (task §4 in the
priority order this branch followed) — real, correctness-sensitive work on already-trusted
production code, needing its own dedicated D=4+D=20 gate before any flip, explicitly out of this
session's scope.
