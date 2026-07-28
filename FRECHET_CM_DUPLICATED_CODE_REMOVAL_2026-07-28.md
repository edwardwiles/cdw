# Common Fréchet / flexible-CM harmonization: duplicated-code removal (2026-07-28)

## What was removed / consolidated (7 commits)

| # | Removed/consolidated | Replacement |
|---|---|---|
| 1 | 3 verbatim copies of the H_CC raw+congruence loop | `fill_cm_HCC!` (1 shared function) |
| 2 | Common Fréchet's own inline blanket-averaging packing loop (serial + threaded, 2 copies) | shared `pack_upper_cm_hessian!` |
| 3 | Common Fréchet's per-Hessian-callback `zeros(...)`/`Vector{Float64}(undef,...)` scratch reallocation (`Wtab`/`T1`/`Esum_wb`/`colsum`/`Hraw_cmlevel`, serial + threaded) | `CMFrechetExtension`, built once, persistent |
| 3b | Dead function `archC_frechet_hess_cb_builder_v2` (confirmed zero call sites; also called the legacy dense-only prep function, another sign of staleness) | deleted |
| 4-5 | `hessian_cm_frechet_structured!`/`_v2!` (2 entire separate copies of H_EE+H_EC+H_CC, each ~150-180 lines) | merged into the one shared `hessian_cm_structured!`/`_v2!` (`extension` parameter) + `_fill_frechet_level_blocks!` (the genuinely Fréchet-only tail, moved not duplicated) |
| 6 | `cumulative_backward_gradient_from_prefix!` (duplicate 3-line formula inside `cumulative_backward_gradient!`) | `cumulative_backward_gradient!` now delegates to the (moved, not copied) shared function |
| 7 | `verify_inner_solution_operator_cm_frechet!`'s entire body (near-duplicate of `verify_inner_solution_operator_cm!`) | both now thin wrappers around shared `_verify_inner_solution_operator_cm_core` |

Net line count across the 7 commits (`git diff --shortstat 7a185ec..HEAD`, verified, not
estimated): **504 insertions, 473 deletions** -- a small net **increase** of +31 lines. This is
expected, not a sign duplication wasn't actually removed: the increase is almost entirely new
docstrings (each consolidated function got a "harmonization task" explanation of what it now
does for both families, generally longer than the terse comments the two separate copies had),
the new `CMFrechetExtension` type definition + constructor, and the design-doc-quality comments
on the two bugs found while wiring this (§ above). The actual *computational* logic removed
(2 entire Hessian-fill-body copies, 3 H_CC copies down to 1, 2 verification-function bodies down
to 1 core) is real and large; it's just outweighed line-for-line by documentation, which this
project's own convention favors over terse or absent comments on non-obvious consolidation
decisions.

## What was intentionally KEPT (not deleted), and why

Several historical diagnostic/gate scripts from earlier sessions (2026-07-25/26/27) call the OLD
function names directly with the OLD argument signatures, not through
`archC_frechet_hess_cb_builder`:

- `test_cm_frechet_threaded_hessian_gates.jl`
- `test_frechet_winner_bin_her_wiring_d4.jl` / `_d20.jl`
- `test_frechet_d20_gates.jl` / `_L50.jl`
- `test_frechet_hessian_structured_vs_dense_d4.jl`
- `diag_frechet_hardpoint_2026-07-27.jl`

Rather than break these (or hunt down and rewrite every one, expanding this task's scope well
beyond the harmonization itself), `hessian_cm_frechet_structured!` and
`hessian_cm_frechet_structured_v2!` are kept as **one-line backward-compatibility wrappers**:

```julia
function hessian_cm_frechet_structured!(h, obj, cctx::CMBinHessCtx, level_targets::Vector{Float64})
    return hessian_cm_structured!(h, obj, cctx, _resolve_frechet_ext!(cctx, level_targets))
end
```

These are not "duplicated logic" in the sense this task is concerned with -- they contain zero
independent computation, just a resolve-and-delegate call. `test_cm_frechet_threaded_hessian_gates.jl`
was actually exercised (not just left untouched) as part of this task's own gating (see
`FRECHET_CM_HESSIAN_EQUIVALENCE_GATE_2026-07-28.md`) and passed cleanly through this wrapper path.

## What was intentionally NOT unified (named honestly, not hidden)

Per `FRECHET_CM_SHARED_DISPATCH_PROOF_2026-07-28.md`'s own "not yet unified" section:
`inner_loop_internal_cmlookup_production`/`_cmfrechetlookup_production` (FG entry points, build
genuinely different state types), their KNITRO-driver wrappers and FG callbacks (differ only in
a registered symbol -- trivial but not part of the task's explicit §15 checklist), and
`archC_hess_cb_builder` vs `archC_frechet_hess_cb_builder` (both still separate top-level KNITRO
callback builders, each wrapping the now-shared fill functions with its own family's dispatch).
These were judged lower-value / proportionally higher mechanical-edit risk relative to the task's
explicit source-reuse checklist, and were not attempted.

## Verdict

```
DUPLICATED_LOGIC_REMOVED = fill_cm_HCC (3->1), H_EC_packing (2->1), level_block_scratch
    (reallocated->persistent), hessian_fill_body (2->1 + genuine extension), CM_transpose (2->1),
    CM_verification (2->1)
BACKWARD_COMPAT_WRAPPERS_KEPT = hessian_cm_frechet_structured!, hessian_cm_frechet_structured_v2!
    (both 1-line delegates, zero independent logic, needed by 7 pre-existing historical scripts)
DEAD_CODE_REMOVED = archC_frechet_hess_cb_builder_v2 (confirmed zero call sites)
NOT_UNIFIED_BY_CHOICE = FG entry point pair, KNITRO driver pair, FG callback pair,
    Hessian-callback-builder pair -- named, not part of task §15 checklist, judged
    lower-value/higher-risk
NET_LINE_CHANGE = +31 (504 insertions, 473 deletions, verified via git diff --shortstat --
    net increase from added docstrings/CMFrechetExtension type, not evidence against real
    duplicated-logic removal, see explanation above)
```
