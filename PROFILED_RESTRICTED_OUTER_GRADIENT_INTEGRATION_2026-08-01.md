# Profiled restricted outer-gradient: integration status (2026-08-01)

## Status: NOT YET INTEGRATED

The inner workstream has not yet reached the point this branch needs to integrate against. This is
expected — the two workstreams were explicitly run in parallel, and integration was always deferred
until the inner branch exposes live typed accessors for at least one restricted family's D4 context.

## Inner branch state observed at the time of this check

```
architecture/profiled-restricted-inner-endtoend-2026-08-01
tip at outer-gradient branch time: f1f969b (Fix H_EF/H_EZ pi_vec-vs-Lam_homog formula bug)
tip at this integration check:     d48dbbc (Phase 1 complete: record 12/13 gate reproduction
                                             + full threaded-test diagnosis/fix)
3 commits ahead of the outer-gradient branch's own base point (f1f969b), none of them adding
profiled_economic_layout/economic_dual_range/restriction_dual_ranges/profiled_anchor_spec/
profiled_outer_coordinate_layout (grep confirmed empty over full_aod_diag/ at d48dbbc).
```

`profiled_restricted_family_base_2026-08-01.jl` (read-only reference on the inner branch, not
modified by this workstream) confirms the economic-layout-sharing design premise
(`build_reduced_base_obj_for_family` reuses `ProfiledEconomicMomentLayout` unchanged for every
family's width bookkeeping) but does not itself expose any of the five contract accessors, and the
inner branch's own current work (per its commit history) is still at the H_EE/H_EC/H_EF/H_EZ
formula-correctness and D4-gate-reproduction stage, not yet at "family inner context ready to
evaluate a point end-to-end."

## What is blocking integration

A real, live family inner context/evaluator (equivalent in shape to `evaluate_profiled_point`'s own
return NamedTuple) for at least one restricted family, exposing the five accessors in
`profiled_outer_gradient_layout_contract_2026-08-01.jl` (or an adapter this branch can wrap around
whatever the inner branch does expose).

## What is NOT blocking integration (already done, ready to consume real contexts)

- The shared economic A/gp gradient engine (`profiled_shared_economic_gradient_engine_2026-08-01.jl`)
  is contract-driven, not unrestricted-specific — verified via `MockRestrictedFamilyCtx` (four mock
  families) to machine precision against an independent full-rebuild reference
  (`PROFILED_RESTRICTION_MOCK_FAMILY_GATE_2026-08-01.csv`).
- The unrestricted regression (real KNITRO, D4 and real D20/W=20,000) confirms the refactor changed
  nothing about the unrestricted family's own numbers — bit-identical to the pre-refactor code.
- The outer A/B harness (`PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl`) is family-generic;
  its unrestricted arm is smoke-tested and working; its four restricted-family arms are explicit,
  loud "not ready" placeholders (`restricted_family_evaluator_not_ready`) rather than silent stubs.

## Plan once the inner branch is ready

1. Rebase or cherry-pick this branch onto the inner branch's tip at that time (record the exact
   commit here).
2. Replace `build_mock_restricted_family_ctx`'s four call sites (in whatever test/integration code
   consumes them) with real adapters implementing the five accessors over the inner branch's own
   context type per family — this is expected to be a small amount of glue code, since the mock
   adapters already establish the exact shape needed
   (`full_aod_diag/d4_exact/profiled_family_adapters_2026-08-01.jl`).
3. Re-run every gate in this package (D4, D20-small-W, restriction-parameter regression, performance)
   against the real contexts, replacing every `BLOCKED_PENDING_INNER` row in the CSVs with a real
   result.
4. Do not resolve any merge conflict by overwriting inner Hessian/FG code — this branch's edits are
   confined to `profiled_lfix_incremental_2026-08-01.jl` (an extraction/refactor, not a formula
   change, see the master doc) plus wholly new files; conflicts, if any, should be mechanical.
