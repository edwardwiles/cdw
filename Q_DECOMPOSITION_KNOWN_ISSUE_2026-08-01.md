# `q_decomposition` known unresolved discrepancy (2026-08-01)

## Status

`profiled_restricted_accessors_2026-08-01.jl`'s **structural** accessors are solid, verified, and
safe to use: `profiled_economic_layout`, `profiled_anchor_spec`, `profiled_outer_coordinate_layout`,
`economic_dual_range`, `restriction_dual_ranges`, `gravity_dual_index`, `stable_inner_layout_fields`,
`restriction_outer_parameter_layout`, `verify_dual_ranges_partition`. All verified against real built
reduced contexts for flexible CM, common Fréchet, and ZC-only — `test_profiled_restricted_accessors_
2026-08-01.jl` exits 0, every structural check PASSes.

`q_decomposition` (and its helper `_q_gravity`) is **NOT** fully verified. Its construction identity
(`q_economic + q_gravity + q_restriction == q_total`) holds to floating-point precision by
construction (trivially true, since `q_restriction` is defined as the residual). But the substantive
claim the outer bridge actually needs — that `q_restriction` is genuinely **A/gp-independent** when
theta is perturbed at a fixed restriction dual — has a robust, reproducible discrepancy: under a
genuine theta perturbation, `q_restriction` changes by almost exactly **2x** `q_economic`'s own
change, not zero.

**Do not build outer-gradient code against `q_decomposition`'s A/gp-independence claim until this is
resolved.**

## What was ruled out (three real bugs found and fixed along the way)

1. **Economic/gravity boundary confusion**: `economic_dual_range` originally included gravity's own
   dual slot (off-by-one from a wrong reading of `NCORE`'s composition). Fixed: economic block is
   `G`'s columns `1:pregrav` (dual indices `2:NCORE`), gravity is the SEPARATE column `NCORE` (dual
   index `NCORE+1`), confirmed directly from `materialize_homogeneous_dense_G_reduced!`'s own fill
   target (`Gtmp[:,1:pregrav]`) and every `wrap_moments_with_*` family's own `fill_gravity_column_
   into!(@view(Gtmp[:, ncore_full]), ...)` call (same pattern, all three families).
2. **`q_economic`'s own beta slicing**: was originally skipping the wrong element (`λstar[first(er):
   (last(er)-1)]`, a leftover from before the economic/gravity split was understood correctly). Fixed
   to a direct `λstar[er .- 1]` once `er` itself only spans the true economic columns.
3. **Test harness moments! call**: the FIRST attempt at a "genuine theta perturbation" test called
   `obj.moments!` with fresh throwaway `K`/`G` arrays instead of writing into `obj.H`'s own views
   (`CS.select_G_from_H(obj, obj.H)`, the exact convention `inner_loop_internal_archgeneric` itself
   uses) — silently leaving `obj.H` (and therefore `q_total`) completely stale. Fixed.

Each fix moved the observed ratio (exact equality between `Δq_economic` and `Δq_restriction`, then
exact `2x`) but did not resolve it to zero.

## What was also ruled out as a cause

- **Invalid/inconsistent perturbed `theta_full`**: retested with a PROPER `x_free`-based perturbation
  (`x_free_pert[1] *= 1.01; θ_full_pert = CS.reconstruct_full(x_free_pert, ctx.m)`, the same machinery
  every real production call site uses) instead of directly poking one entry of the already-
  reconstructed `theta_full_calib` array. Confirmed the perturbation changes exactly 1 of 23
  `theta_full` entries (no hidden symmetry/tie issue). **Same exact 2x ratio reproduced** under this
  independently-constructed perturbation — rules out an inconsistent/invalid perturbed point as the
  cause.
- **Gravity itself drifting under the perturbation**: `_q_gravity`'s own output (read directly from
  `obj.H`'s gravity column) was confirmed BIT-IDENTICAL between the calibration and perturbed points
  (`max|Δq_gravity| = 0.0` exactly) — gravity is not the source of the extra factor.

## Leading hypothesis for whoever picks this up next

`materialize_homogeneous_dense_G_reduced!` constructs `G`'s economic columns via
`Gview[:,j] = reduced_homogeneous_dual_contraction(e_j, cf, ctx, θ_full, layout)` for unit vectors
`e_j` (linearity argument — this is the exact same "safe by construction" pattern this whole session
used for H_EC/H_EF/H_EZ). By that same linearity, `obj.H`'s economic columns dotted with any `β`
should exactly equal `reduced_homogeneous_dual_contraction(β, cf, ctx, θ_full, layout)` directly — the
two were never directly cross-checked column-by-column in this investigation (only end-to-end through
`q_total`/`q_economic`, which obscures WHERE a factor-of-2-ish discrepancy enters). The single highest-
value next step: build `G_check = materialize_homogeneous_dense_G_reduced!`'s own output (or read it
straight off `obj.H` post-`moments!`) and directly compare `G_check[:, 1:pregrav] * β_econ` against
`reduced_homogeneous_dual_contraction(β_econ, cf, ctx, θ_full, layout)` at the SAME `cf`/`θ_full`/`β`
— if these two disagree, the bug is inside `q_economic`'s own formula call (a stale `layout`, a
`cctx.econ_ctx` vs `ctx` identity mismatch, or similar); if they agree exactly, the bug is instead in
how `_prep_dual_index_for_archC!`/`obj.arg0`'s own dense reconstruction combines ALL of `H`'s columns
(possibly reading a WIDER slice than `2:1+outer_constr_index` actually should span, double-counting
something).

## Repro

`full_aod_diag/d4_exact/test_profiled_restricted_accessors_2026-08-01.jl`, the
`check_known_issue("flexible CM (synthetic nonzero dual): q_restriction is A/gp-INDEPENDENT under a
GENUINE theta perturbation ...")` line — reported but does not fail the gate (`ALL_PASS` unaffected,
by design, so the solid structural checks stay visibly green while this one open item stays visible
too, not silently dropped).
