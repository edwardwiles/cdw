# Dead-code removal manifest — flexCM/Fréchet FG harmonization (2026-07-29)

## Code removed (replaced by shared functions, zero remaining live references)

1. The duplicated (E)-block forward computation previously inlined separately in
   `CMLookupState.dual_index!` (`cm_lookup_kernels.jl`) and `CMFrechetLookupState.dual_index!`
   (`cm_frechet_lookup_kernels.jl`) — both now call `economic_forward_into_arg0!` (new, single
   definition in `cm_lookup_kernels.jl`).
2. The duplicated (C)-block forward computation in the same two methods — both now call
   `cm_forward_contribution!` (new, single definition).
3. The duplicated (E)-block backward/transpose computation previously inlined separately in both
   FG functors — both now call `economic_transpose_into_g1_and_gE!` (new, single definition).
4. The duplicated (C)-block backward/transpose computation in the same two functors — both now
   call `cm_transpose_into_g!` (new, single definition).
5. The verbatim-copied CM-block computation inlined at the top of
   `frechet_cm_level_fixed_contribution` (`cm_frechet_cplus.jl`) — replaced by a call to
   `cm_fixed_value_contribution` (new, single definition in `lfix_cm_aware.jl`), which
   `cm_fixed_contribution` (flexible CM) also now calls.

## Verification: zero remaining live references to the removed inline code

The removed code was never a separately-named function (it was inlined directly in
`dual_index!`/the FG functors/`frechet_cm_level_fixed_contribution`), so there is no dangling
function name to grep for — the removal IS the diff to those three files. What can be verified is
that the four new shared functions are the ONLY forward/backward implementations left:

```
$ grep -n "function dual_index!\|function (st::CM" full_aod_diag/d4_exact/cm_lookup_kernels.jl full_aod_diag/d4_exact/cm_frechet_lookup_kernels.jl
```

confirms each `dual_index!`/functor body is now under 10 lines and contains no inlined
economic/CM math — every remaining line is either a slice (`@view`), a call to one of the four
shared functions, or the family-specific `[F]` extension.

## What was intentionally NOT removed (see duplication audit for rationale)

- `CMLookupState`/`CMFrechetLookupState` struct definitions (kept separate — 50+ ad hoc call sites
  depend on `CMLookupState`'s exact type/constructor; no runtime cost to the field duplication).
- `inner_loop_KNITRO_cmlookup_production`/`inner_loop_KNITRO_cmfrechetlookup_production` and
  `inner_loop_internal_cmlookup_production`/`inner_loop_internal_cmfrechetlookup_production` (KNITRO
  wiring layer — left as parallel structure, not hot-path, flagged as the top remaining item).
- No test files or family-specific formula tests were deleted, because no NEW test files were made
  obsolete by this refactor — the pre-existing `c12i_validate_lookup_fg.jl` and
  `bench_frechet_operator_fg_default_gate_2026-07-27.jl` gates remain valid AND now double as the
  refactor's own regression tests (re-run against the new code in this task, both still pass).

## Imports/exports

No module `export` lists exist in this codebase (flat `include()`-based script composition, no
`module`/`export` blocks) — nothing to update on that front. No `using`/`include` statements needed
to change: the new shared functions live in `cm_lookup_kernels.jl`, which every caller already
included before `cm_frechet_lookup_kernels.jl`/`cm_frechet_cplus.jl` (verified in
`FLEXCM_FRECHET_METHOD_REUSE_PROOF_2026-07-29.md`).
