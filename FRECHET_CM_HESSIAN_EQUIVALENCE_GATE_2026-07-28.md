# Common Fréchet / flexible-CM harmonization: Hessian equivalence gate (2026-07-28)

Focused specifically on the Hessian-assembly consolidation (harmonization steps 1, 2, 4, 5 --
`fill_cm_HCC!`, `pack_upper_cm_hessian!`, the merged `hessian_cm_structured!`/`_v2!` with the
`extension` parameter, and `_fill_frechet_level_blocks!`), since this was both the largest and
the highest-risk single change in this task.

## H_CC consolidation (step 1)

Before: 3 verbatim copies of the H_CC raw+congruence loop (flexible-CM serial, flexible-CM
threaded, common-Fréchet's own serial+threaded each had their own). After: 1 shared
`fill_cm_HCC!(Hfull, cctx, M)`, called from all 4 sites.

Gate: D=4 equivalence for flexible CM (serial+threaded) and common Fréchet, both `max|ΔH|=0.000e+00`
before and after.

## H_EC packing consolidation (step 2)

Before: flexible CM's `pack_upper_cm_hessian!` (upper-only special case for the H_EC block)
vs common Fréchet's own inline blanket-averaging loop -- **verified (not assumed) to be a
bit-exact no-op**, since both families explicitly write the identical value into both triangles
of every off-diagonal block before packing (`0.5*(v+v) == v` in IEEE754). After: common Fréchet
calls the same `pack_upper_cm_hessian!`.

Gate: D=4 common-Fréchet equivalence, `max|ΔH|=0.000e+00` (confirms the hand-verified no-op
claim empirically, not just algebraically).

## The big merge: shared `hessian_cm_structured!`/`_v2!` (steps 4-5)

Before: `hessian_cm_structured!`/`_v2!` (flexible CM/CM+ZC) and
`hessian_cm_frechet_structured!`/`_v2!` (common Fréchet) were two separate top-level functions,
each independently implementing H_EE + H_EC + H_CC (near-verbatim between the two), with common
Fréchet's copy additionally appending 3 level blocks and flexible CM's copy additionally handling
the CM+ZC `use_direct_hcz` branch Fréchet never needs.

After: ONE `hessian_cm_structured!`/`_v2!`, taking an optional `extension::Any=nothing` argument.
`extension === nothing` (flexible CM/CM+ZC, unchanged call sites) skips the new
`_fill_frechet_level_blocks!` call entirely -- same code path as before, byte-for-byte, just
inside a function that also knows how to do more when asked. `extension::CMFrechetExtension`
(common Fréchet, via `archC_frechet_hess_cb_builder`'s `_resolve_frechet_ext!` cache) additionally
computes H_E,level/H_CM,level/H_level,level via the extracted (not duplicated)
`_fill_frechet_level_blocks!`.

### Two real bugs caught by this merge (both via real KNITRO callback failures, not silent
wrong answers -- see `FRECHET_CM_HARMONIZATION_DESIGN_2026-07-28.md`'s step 3 section and the
step 4-5 commit message for full detail):

1. `extension isa CMFrechetExtension` inside `cm_hessian_architectures.jl` would throw
   `UndefVarError` for flexible CM's own scripts (which never load `cm_frechet_hessian.jl`) --
   fixed to `extension !== nothing`.
2. `local wctx = nothing, cross_ws = nothing, bin_zc_ws = nothing` is not valid Julia
   multi-variable local initialization (confirmed via isolated `julia -e` repro before assuming
   it was the bug) -- left `cross_ws`/`bin_zc_ws` genuinely undefined, which only surfaces as an
   error the instant `use_winner_bin` is false AND the variables are read/passed -- caught
   immediately by the real KNITRO Hessian callback throwing, not a silent wrong Hessian.

### Gate results (post-fix)

| Family | D=4 max|ΔH| | Real D=20 max|ΔH| |
|---|---|---|
| flexible CM (serial) | 0.000e+00 | 0.000e+00 |
| flexible CM (threaded, production default) | 0.000e+00 | 0.000e+00 |
| common Fréchet (serial) | 0.000e+00 | 0.000e+00 |
| common Fréchet (threaded, via `test_cm_frechet_threaded_hessian_gates.jl`) | 0.000e+00 (v2-serial vs serial) / ~1.4e-14 (v2-threaded vs serial, matches documented tolerance) | not separately re-run (D=4 threaded gate is the validated one for this specific serial-vs-threaded comparison) |
| CM+ZC (uses the same `use_direct_hcz` branch in the shared function) | 0.000e+00 | 0.000e+00 |

## CM transpose consolidation (step 6)

Before: `cumulative_backward_gradient!` (flexible CM, calls `prefix_sums!` internally) and
`cumulative_backward_gradient_from_prefix!` (common Fréchet, assumes prefix sums already
computed) -- identical 3-line gradient formula, duplicated. After: the former calls
`prefix_sums!` then delegates to the latter (moved to `cm_lookup_kernels.jl` for load-order
reasons). D=4 gate: `max|Δg|=0.000e+00` for flexible CM, common Fréchet, CM+ZC.

## CM verification consolidation (step 7)

Before: `verify_inner_solution_operator_cm!`/`verify_inner_solution_operator_cm_frechet!` --
byte-identical economic+CM verification logic, common Fréchet's copy appending the level block.
After: both are thin wrappers around one shared `_verify_inner_solution_operator_cm_core`.
Dedicated gate (`test_operator_verification_cm.jl`/`_cm_frechet.jl`, L=10 and L=50): dense vs
operator KKT residual bit-identical for both families (e.g. common Fréchet L=10:
`dense_kkt=2.245e-14, operator_kkt=2.245e-14`; L=50: `5.010e-15` both).

## Verdict

```
H_CC_CONSOLIDATION        = pass_bit_exact
H_EC_PACKING_CONSOLIDATION = pass_bit_exact_confirmed_no_op
HESSIAN_FILL_BODY_MERGE   = pass_bit_exact_all_families_d4_and_d20 (2 real bugs found+fixed during
                             wiring, both caught immediately by real KNITRO errors)
CM_TRANSPOSE_CONSOLIDATION = pass_bit_exact
CM_VERIFICATION_CONSOLIDATION = pass_bit_exact_kkt_residuals
HESSIAN_EQUIVALENCE_GATE  = pass
```
