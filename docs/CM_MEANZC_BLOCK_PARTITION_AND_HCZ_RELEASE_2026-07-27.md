# CM+ZC E/C/Z Block-Partition + H_CZ Release — 2026-07-27

## What changed

CM+ZC's Hessian previously treated its economic (E) and mean/pairwise-ZC-restriction (Z) columns
as ONE widened "core" block for the purposes of the CM-grid cross Hessian (`H_EC`, the common-
marginals bin/threshold restriction block, `C`) -- `hessian_cm_structured!`/`_v2!`
(`cm_hessian_architectures.jl` / `cm_hessian_threaded.jl`) always read `E = H[:, 2:1+NCORE]`
(`NCORE = ncore_core + n_restr`, the FULL widened width) when building `H_EC`, and the existing
winner-aware `:winner_bin` CM-grid backend (`winner_pair_cross_hessian_cm_block!`, Section 2 of the
prior winner-aware-H_ER phase) was explicitly gated OFF for CM+ZC (`_cm_cross_hessian_wants_
winner_bin`'s old `ncore_core == NCORE` guard) because relaxing it would have silently left the
widened Z-rows of `H_EC` uncomputed -- a real correctness gap, not a knife edge (see
`CM_MEANZC_WINNER_AWARE_HER_RELEASE_2026-07-27.md`'s own "Investigation" section, which flagged
this as future work requiring "a genuinely new cross primitive: mean/pair-restriction x
CM-grid-restriction, `Z'S*C`").

This session implements that primitive (`H_CZ`) and completes the genuine three-way `G = [E | C | Z]`
partition CM+ZC's Hessian was missing:

- **`H_EE`, `H_EC`, `H_EZ`, `H_CC`**: unchanged -- `H_EC` now genuinely only reads the TRUE
  economic sub-block (`E[:, 1:ncore_core]`, via the pre-existing `wctx`, which was ALREADY exactly
  `ncore_core`-wide and unaffected by CM+ZC's own widening -- confirmed by reading
  `_fill_cm_HEE!`'s own `build_core_exact_hessian_workspace(cf)` call, built from the plain
  economic-only `CompressedFactual`, not the widened `E`).
- **`H_CZ` (NEW)**: `bin_zc_cross_hessian_fill!`/`_block!` (`winner_pair_cross_hessian.jl`) -- a
  bin-index-keyed analogue of `winner_pair_cross_hessian_cm_block!`, with the ZC-restriction's
  already-centered, already-`S`-weighted `ZcS` (see the companion doc,
  `SHARED_ZC_HRR_DIRECT_RELEASE_2026-07-27.md`) playing the role the winner-selected economic
  column plays there. No winner-selection at all here (`Z`'s columns are plain per-draw feature
  values, not a winner-argmin outcome) -- just a bin-membership accumulation, considerably simpler
  than the primitive it mirrors. `C` itself is reconstructed from `Bidx`/bin membership directly
  (exactly as the dense path already did), never read from a materialized `obj.H` CM-grid block
  (which under `:cm_lookup`/`:operator` inner FG backends may not even be filled, see
  `skip_cm_fill_ref`).
- **Guard relaxation**: `_cm_cross_hessian_wants_winner_bin` (`cm_hessian_architectures.jl`) no
  longer requires `ncore_core == NCORE` -- it now accepts CM+ZC's widened case too, PROVIDED
  `cctx.hzz_zc_op !== nothing` (i.e. CM+ZC's own raw-ZC-feature state is available, guaranteeing
  the companion `H_CZ` fill will cover the rows the relaxed guard would otherwise leave
  uncomputed). A new decision function, `_cm_cross_hessian_wants_direct_hcz`, governs exactly when
  that companion fill runs. Both `hessian_cm_structured!` (serial) and `hessian_cm_structured_v2!`
  (threaded, the real production default) got the identical treatment -- this codebase's own
  established convention for this pair of files is to keep the H_EC/H_CC block textually mirrored
  rather than factored into one shared function, so the new H_CZ block was added to both call sites
  following that same discipline.

See the companion doc, `SHARED_ZC_HRR_DIRECT_RELEASE_2026-07-27.md`, for the new shared `H_ZZ`
primitive (CM+ZC's own `HMM` block, and origin-ZC's `HRR`) -- implemented alongside this session's
work, in the same commits, since both use the same new `ZCCenteredScratch`/`ZcS` state.

## New default

`CM_MEANZC_CM_CROSS_HESSIAN_BACKEND_DEFAULT` (`core_exact_hessian.jl`, new Ref, CM+ZC's own
analogue of `CM_CROSS_HESSIAN_BACKEND_DEFAULT`/`CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT`) is
`:winner_bin` from creation -- `build_cm_meanzc_bin_ctx`'s own `cm_cross_hessian_backend` kwarg now
defaults to it (previously hardcoded `:dense_reference`). Flipped only after the gates below passed
to machine precision.

## Gates (D=4 and real D=20, both PASS)

**D=4** (`test_cm_meanzc_hcz_hzz_direct_d4.jl`): `K_mean=1/K_pair=1` and `K_mean=2/K_pair=2`, both
contrast conventions (anchored, orthonormal), calibration + 2 perturbed points, both serial
`hessian_cm_structured!` and threaded-production `hessian_cm_structured_v2!`, complete real KNITRO
inner solve per (config,contrasts) pair -- `cm_cross_hessian_backend=:winner_bin` AND
`zc_cross_hessian_backend=:winner_bin` together (the combination that exercises the relaxed guard +
NEW `H_CZ` fill) against the fully dense-reference construction (both backends left at
`:dense_reference`, i.e. the untouched pre-refactor path). **ALL PASS**:

| config | contrasts | max\|ΔH\| (complete packed Hessian) | max\|ΔH\| (Z-rows sub-block only) |
|---|---|---|---|
| K1 (K_mean=1,K_pair=1) | anchored | 9.99e-16 -- 1.83e-15 | same |
| K1 | orthonormal | 1.44e-15 -- 3.00e-15 | same |
| K2 (K_mean=2,K_pair=2) | anchored | 1.42e-14 -- 4.55e-13 | same |
| K2 | orthonormal | 2.84e-14 -- 4.55e-13 | same |

(Full per-point breakdown in the test's own stdout; committed as run output below.) Complete inner
solve status matches (`nStatus=0` throughout) and dual point agrees to `<1e-8`. Persistent-scratch
object identity (`hzz_centered`, `bin_zc_cross`) confirmed stable across repeated warm calls (no
resize).

**Real D=20/W=80,000/L=50** (`test_cm_meanzc_hcz_hzz_direct_d20.jl`,
`destination_sample=:exclude_row`, `K_mean=1/K_pair=1`, matching `d20_meanzc_release_gates.jl`'s own
Point A config): both contrasts, calibration + a near-delta=1 perturbed point, both serial and
threaded_v2 architectures, against the fully dense-reference construction. **ALL PASS**:

| contrasts | point | arch | max\|ΔH\| complete packed | max\|ΔH\| Z-rows sub-block | scale |
|---|---|---|---|---|---|
| anchored | calib | serial | 6.395e-13 | 6.395e-13 | 3.970e+03 |
| anchored | calib | threaded_v2 | 6.395e-13 | 6.395e-13 | 3.970e+03 |
| anchored | near_delta1_perturbed | serial | 8.882e-13 | 8.882e-13 | 5.649e+03 |
| anchored | near_delta1_perturbed | threaded_v2 | 8.882e-13 | 8.882e-13 | 5.649e+03 |
| orthonormal | calib | serial | 9.024e-13 | 9.024e-13 | 3.970e+03 |
| orthonormal | calib | threaded_v2 | 9.024e-13 | 9.024e-13 | 3.970e+03 |
| orthonormal | near_delta1_perturbed | serial | 6.750e-13 | 6.750e-13 | 6.861e+03 |
| orthonormal | near_delta1_perturbed | threaded_v2 | 6.750e-13 | 6.750e-13 | 6.861e+03 |

Complete inner solve status matches (`nStatus=0` both backends, both contrasts) and dual point
agrees to `<5e-13`. `hzz_centered`/`bin_zc_cross` object identity confirmed stable across repeated
warm calls (no resize), both contrasts. Warm `hessian_cm_structured_v2!` (direct H_CZ/H_ZZ)
allocates ~10.0 MB/call.

**`K_pair=0` mean-only smoke** (real D=20, contrasts=:orthonormal): `hzz_zc_op` built, inner solve
feasible (`status=0`), Hessian call completed with no NaN/Inf. PASS.

## Production-only invariant check (isolated, no dense-reference call in the same process)

The gate script above deliberately runs BOTH `:dense_reference` and `:winner_bin` backends in one
process for A/B comparison, so its own cumulative counter dump is not a clean read of "production
alone" (`dense_cross_hessian_calls` there is nonzero, entirely from the intentional dense-reference
comparison calls). A dedicated isolated check
(`verify_production_only_zero_dense_cross_hessian_2026-07-27.jl`, K_mean=1/K_pair=1, default
backends only, never touching `:dense_reference`) confirms the actual production invariant:

```
cm_cross_hessian_backend default = winner_bin
zc_cross_hessian_backend default = winner_bin
    full_G_materializations = 0
    dense_economic_G_materializations = 0
    dense_CM_G_materializations = 0
    dense_ZC_G_materializations = 0
    dense_Frechet_G_materializations = 0
    generic_dense_FG_calls = 0
    operator_FG_calls = 7
    operator_verification_calls = 1
    dense_reference_verification_calls = 0
    dense_cross_hessian_calls = 0
    operator_cross_hessian_calls = 20
    winner_cross_hessian_calls = 20
>>> PASS: production-default CM+ZC (K_mean=1,K_pair=1) shows dense_cross_hessian_calls=0,
    winner_cross_hessian_calls=20
```

`dense_cross_hessian_calls == 0` confirmed in a real run at real D=20 scale, not a static code read
— this is the task's Section 7/13 required invariant, closed.

## Counters

Full cumulative dump from the A/B comparison gate (both backends exercised, both contrasts, 4
points, K_pair=0 smoke) — `dense_cross_hessian_calls=54` here reflects the intentional
`:dense_reference` comparison arm, not a production leak (see isolated check above for the clean
production-only read):

```
full_G_materializations = 0, dense_economic_G_materializations = 0, dense_CM_G_materializations = 0,
dense_ZC_G_materializations = 0, dense_Frechet_G_materializations = 0, generic_dense_FG_calls = 0,
operator_FG_calls = 30, operator_forward_calls = 30, operator_transpose_calls = 30,
operator_economic_FG_calls = 0, operator_restriction_FG_calls = 0, operator_verification_calls = 0,
dense_reference_verification_calls = 0, dense_cross_hessian_calls = 54, operator_cross_hessian_calls = 112,
winner_cross_hessian_calls = 112
```

## Not done / left for a future session

- `H_RR` / `H_CC` themselves stay dense BLAS unconditionally, unchanged -- explicitly out of scope.
- Rectangular (`D != Ddest`) D=4 configurations were not separately exercised for this NEW block
  (same gap the prior winner-aware-H_ER phase already disclosed -- no rectangular D=4 CM context
  builder exists in this repository).
- `K_pair=0` (mean-only) was exercised only as a status/no-crash smoke at real D=20 (task's own
  gate-3 scope), not a full residual comparison against dense-reference at that width.
