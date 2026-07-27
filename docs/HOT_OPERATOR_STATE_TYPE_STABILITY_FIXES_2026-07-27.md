# Hot Operator State Type-Stability Fixes — 2026-07-27

## Status: NOT ATTEMPTED this pass

Task §10 asks to audit/benchmark `OriginZCOperatorState`/`CMMeanZCOperatorState` (and, per this
task's own addendum note, to read the comments a sibling session already left on `CMLookupState`/
`CMFrechetLookupState`'s own `econ_ws`/`econ_ws_for`/`core_cf_ref` fields before "fixing" anything).

This was explicitly deprioritized by the task's own guidance: "Do NOT attempt section 10's
Hessian-allocation work or broad type-stability changes unless 1-4 are already solid with real
evidence — a rushed change with no before/after measurement is worse than leaving it alone and
reporting it honestly as not attempted." Sections 1-4 are only PARTIALLY solid this pass (1, 2, 3
have real evidence; 4 was not attempted at all, see `PERSISTENT_LFIX_BASE_WORKSPACE_2026-07-27.md`)
— given that, and the two unplanned bug investigations that consumed a large share of this
session's time, a type-stability change (which the task itself flags as needing `@code_warntype`
plus before/after runtime measurement, not a stylistic edit) was not attempted.

## What the prior audit already established (read, not re-derived)

From `HOT_PATH_TYPE_STABILITY_AUDIT_2026-07-26.md` (the sibling allocation audit's own deliverable,
read as part of this task's required background reading): `OriginZCOperatorState`
(`cm_originzc_lookup_kernels.jl:35-49`) has four `Any`-typed fields (`obj::Any`, `layout::Any`,
`core_cf_ref::Ref{Any}`, `econ_ws_for::Any`), read on every FG callback of every ZC-only/CM+ZC
`:operator`-backend inner KNITRO solve. That audit explicitly marked the field TYPES as
`CONFIRMED` (static read) but their dynamic-dispatch COST as `UNVERIFIED` (no `@code_warntype` or
`@allocated` isolation was run against them, in that audit or this one).

This task's own addendum note additionally flags that a sibling session (the one whose in-progress
work this worktree branched from, commit `a69b32d`) has since added SIMILAR `econ_ws`/`econ_ws_for`/
`core_cf_ref` fields to `CMLookupState`/`CMFrechetLookupState` using the same `::Any` idiom
**deliberately**, per that session's own in-file comments, for load-order-safety reasons (these
structs are defined before the concrete types they'd otherwise reference are available). This is a
real, disclosed design tradeoff in the codebase already — not an oversight this task should
"fix" without first reading those comments and understanding why `Any` was chosen there. This task
did not open those specific files to re-verify the sibling session's own reasoning (out of scope;
this task's changes never touched `CMLookupState`/`CMFrechetLookupState`), consistent with not
touching type-stability at all this pass.

## Recommended next step (not started)

1. `@code_warntype` on the `:operator`-backend FG callback that reads `OriginZCOperatorState`'s
   `Any` fields, to see whether Julia's dynamic dispatch actually shows up as a red/yellow flag at
   the specific call sites (not assumed from the field type alone).
2. A before/after `@allocated`/wall-clock comparison of the SAME FG callback with a
   parameterized/concretely-typed version of the struct (feasible only where load order allows a
   concrete type reference — the sibling session's own `Any`-field precedent suggests this may NOT
   be safe for every field without restructuring include order, which is itself a nontrivial,
   separately-risky change).
3. Only apply a change if step 2 shows a real, non-negligible effect — per the task's own explicit
   instruction not to change fields "for stylistic reasons alone."

No code for this was written or benchmarked this session.
