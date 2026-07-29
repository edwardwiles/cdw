# FG-callback duplication audit — flexible_cm vs common_frechet (2026-07-29)

Full call-graph table: `FLEXCM_FRECHET_CURRENT_FG_CALL_GRAPH_2026-07-29.md`. This doc summarizes
the verdict and the concrete duplicate pairs found (all pre-refactor line numbers, i.e. as of
`production/fullA-exact @ 8aae735`, before this task's edits).

## Verdict

**DUPLICATION_FOUND = broader_shared_state_duplication** (economic + CM state/orchestration, not
just one block) — but concentrated at the FG-callback layer specifically, NOT the Hessian layer
(already harmonized) or the low-level numeric kernels (already shared).

The Hessian side (`hessian_cm_structured!`/`_v2!`, `fill_cm_HCC!`, `pack_upper_cm_hessian!`) and the
verification side (`_verify_inner_solution_operator_cm_core`) were already fully harmonized via an
`extension::Any=nothing` / `level_targets::Union{Nothing,Vector}` optional-argument pattern — ONE
real implementation per block, common Fréchet's own function names are one-line wrappers. The
FG-callback layer (state structs, `dual_index!`, the FG functor, KNITRO wiring) had NOT reached the
same standard: the low-level numeric kernels (`apply_contrast!`, `suffix_sums!`,
`cumulative_forward_contribution!`, `build_weighted_histogram!`, `prefix_sums!`,
`cumulative_backward_gradient_from_prefix!`, `economic_forward!`, `economic_transpose!`) were
already literally shared, but the code that COMPOSES them per callback — `CMLookupState`'s and
`CMFrechetLookupState`'s own `dual_index!` methods and FG functors — each inlined its own copy of
that composition, rather than calling one shared composition function the way the Hessian side did.

## Duplicate pairs found (pre-refactor)

1. **`dual_index!(::CMLookupState, x)`** (`cm_lookup_kernels.jl:401-437`, pre-refactor) vs
   **`dual_index!(::CMFrechetLookupState, x)`** (`cm_frechet_lookup_kernels.jl:187-228`,
   pre-refactor). The (E)-block forward (lines 402-423 / 188-210) and (C)-block forward (lines
   425-436 / 212-216) were reproduced verbatim between the two — differing only in the trailing
   (F) level-block extension in the second. Dispatched only by Julia multi-dispatch on struct type,
   not by an explicit "extension" argument the way `hessian_cm_structured!` does it.

2. **`(st::CMLookupState)(x,g)`** (`cm_lookup_kernels.jl:445-489`, pre-refactor) vs
   **`(st::CMFrechetLookupState)(x,g)`** (`cm_frechet_lookup_kernels.jl:235-278`, pre-refactor).
   Same structure line-for-line: economic-transpose branch (`cm_lookup_kernels.jl:467-473` ≈
   `cm_frechet_lookup_kernels.jl:255-261`), histogram+backward-gradient block (`475-482` ≈
   `263-271`) — only the level block differed.

3. **`CMLookupState`** (`cm_lookup_kernels.jl:323-372`) vs **`CMFrechetLookupState`**
   (`cm_frechet_lookup_kernels.jl:107-155`) — every field of `CMLookupState` reappears verbatim in
   `CMFrechetLookupState`, which adds only `ncm_cm`/`ncm_level`/`D`/`level_targets`/`invsqrtD`/
   `level_contrib`/`P_level`/`g_level`. NOT merged in this task (see "what was deliberately NOT
   changed" below) — the field-level duplication is structural, not a live maintenance-burden
   duplicate the way the two `dual_index!`/functor bodies were.

4. **`cm_fixed_contribution`** (`lfix_cm_aware.jl:83-95`, pre-refactor) vs the CM-block computation
   inlined at the top of **`frechet_cm_level_fixed_contribution`** (`cm_frechet_cplus.jl:64-87`,
   pre-refactor, specifically lines 70-76). The Fréchet function's own docstring admitted this was
   inlined "not calling that function directly, because it hardcodes `aug.ncm` as its own tail
   length" — a known, self-acknowledged duplicate blocked only by a hardcoded field-length
   assumption, not a real math difference.

5. **`inner_loop_KNITRO_cmlookup_production`** (`cm_lookup_production.jl:55-91`) vs
   **`inner_loop_KNITRO_cmfrechetlookup_production`** (`cm_frechet_lookup_production.jl:12-48`) and
   **`inner_loop_internal_cmlookup_production`** (`cm_lookup_production.jl:111-176`) vs
   **`inner_loop_internal_cmfrechetlookup_production`** (`cm_frechet_lookup_production.jl:92-129`)
   — KNITRO-plumbing/orchestration duplication (callback registration, counters, priming dispatch).
   Called ONCE per inner solve (not the hot per-callback path).

## What this task fixed

Pairs 1, 2, and 4 (the hot per-FG-callback duplication and the acknowledged fixed-contribution
duplication) — see `FLEXCM_FRECHET_METHOD_REUSE_PROOF_2026-07-29.md` for the concrete shared
functions introduced and the equivalence proof.

## What this task deliberately did NOT change, and why

- **Pair 3 (struct field duplication)**: `CMLookupState`/`CMFrechetLookupState` were NOT merged
  into one parametric struct. `cm_lookup_kernels.jl`'s own header comment documents that
  `CMLookupState` is constructed directly (via its positional-field outer constructor) by 50+ ad
  hoc scripts across the repo; merging the two struct definitions would change `CMLookupState`'s
  type identity and risk breaking that entire surface for a purely cosmetic field-list gain, with
  no runtime duplication cost (unlike the FG-functor code, unused struct fields cost nothing at
  runtime). The functions that operate on the shared field subset are now literally shared (see
  method-reuse proof); the two struct definitions remain separate, which is the lower-risk choice
  given the blast radius.
- **Pair 5 (KNITRO wiring)**: left as parallel, near-duplicate structure. This code runs once per
  inner solve (KNITRO callback registration, priming, counters), not once per FG callback (which
  can fire hundreds of times per inner solve) — the duplication cost here is maintenance burden, not
  runtime cost, and the two functions differ in real ways (different cctx/state types, an extra
  `level_targets` argument). Flagged as the top remaining structural-duplication item for a future
  task, not fixed here to keep this task's diff scoped to the genuinely hot path and its one
  self-acknowledged duplicate.
- **The `moment_representation` asymmetry** (flexible_cm defaults to the true no-H `OperatorPsiBundle`
  path; common_frechet is hardcoded to `:dense_reference` because of an unresolved, previously
  reproduced `nStatus=-400` failure when it was tried) is NOT a duplication issue and is out of
  scope for this task — it is a genuine, currently-necessary asymmetry pending separate root-cause
  work. Flagged in the call-graph doc and the master report's `HIGHEST_PRIORITY_REMAINING_GAP`.
