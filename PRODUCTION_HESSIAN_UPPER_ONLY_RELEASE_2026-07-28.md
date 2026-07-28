# Production Hessian upper-only cleanup — release doc — 2026-07-28

Branch: `cleanup/remove-legacy-CC-H-G-storage-2026-07-28` (committed separately from the
legacy-`obj.H` storage-removal work on this same branch, per task instruction — either change can
be reverted independently). Base: `campaign/five-family-bounds-2026-07-28@93f26df7`.

## What changed

Three edits, all in `full_aod_diag/d4_exact/`, all gated by the existing D=4 regression suite
before and after:

1. **`cm_hessian_architectures.jl`** — deleted a provably dead mirror write in origin-ZC's
   `archA_partitioned_hess_cb_builder`: `@views ∂∂f_∂∂x[NCORE+1:n, 1:NCORE] .= transpose(HER)`.
   This family's packing loop is a plain upper-only copy with no downstream averaging step —
   the deleted line wrote into a strictly-lower-triangular region nothing ever read.
2. **`cm_hessian_architectures.jl` + `cm_hessian_threaded.jl`** — extracted the two files'
   previously-independent, byte-identical final packing loops into one shared function,
   `pack_upper_cm_hessian!(h, Hfull, NCORE, n)` (defined once, in `cm_hessian_architectures.jl`,
   called from both). This removes the "must be mirrored here by hand" duplication risk
   `cm_hessian_threaded.jl`'s own header comment already flagged. While extracting it, changed the
   packing formula for the H_EC region (`i<=NCORE<j`) from `0.5*(Hfull[i,j]+Hfull[j,i])` to a
   direct `Hfull[i,j]` read — that region is filled by a mechanical mirror (never independently
   re-derived), so the average was a provable no-op there, at real per-callback cost. H_EE
   (`i,j<=NCORE`) and H_CC (`i,j>NCORE`) both keep full averaging — see rationale below.
3. **New sentinel test**: `full_aod_diag/d4_exact/test_hessian_upper_only_sentinel_d4.jl`.

This affects **flexible-CM, CM+ZC** (both share `hessian_cm_structured!`/`_v2!`, confirmed by the
constructor-audit doc) and **origin-ZC** (its own separate dead-mirror fix). **Common-Fréchet was
audited (its 4 analogous sites documented in
`PRODUCTION_HESSIAN_UPPER_ONLY_AUDIT_2026-07-28.md`) but not fixed this session** — an honest,
named gap, not an oversight; the fix is structurally identical to what was just done for
flexible-CM/CM+ZC and is the natural next step. **Unrestricted was already upper-only with no
mirror pattern at all** (confirmed by audit, no change needed).

## Why H_EE and H_CC were NOT touched (conservative choice, not an oversight)

- **H_EE**: can be filled via a dual-write winner-pair path (exactly mirrored, in principle
  eligible for the same optimization) OR the dense BLAS `gemm!` fallback (`:tied_winner`/
  `:dense_reference`/`:compressed_state_unavailable`), whose bit-exact cross-diagonal symmetry is
  a BLAS-implementation property, not a language guarantee. Special-casing H_EE's packing would
  require confirming that guarantee empirically first — not attempted this session, kept
  conservative.
- **H_CC**: the `(l,lp)` grid loop genuinely visits both `(l,lp)` and `(lp,l)` orderings
  independently (`Hraw_CC[oi,pi] = (CT[o,p,l,lp] - CT[o,ref,l,lp] - CT[ref,p,l,lp] +
  CT[ref,ref,l,lp])/M`, a fresh prefix-sum evaluation each time, not a copy) — the averaging here
  absorbs real, if small, floating-point path-order noise and must stay.
- **The mirror *writes* themselves** (e.g. `Hfull[cols,1:NCORE] .= transpose(block_ec)`) were left
  in place, not deleted, even for the now-non-averaged H_EC region — `cctx.Hfull` is read directly,
  both triangles, by real diagnostic/test scripts outside the two edited files
  (`diag_frechet_hardpoint_2026-07-27.jl`, `test_frechet_winner_bin_her_wiring_d4.jl`, confirmed by
  `grep -rln "\.Hfull\b"`), so removing the write would change `Hfull`'s public contract for those
  consumers. Only the *packing* step (which never sees `Hfull` from any consumer's perspective, only
  `h`/`evalResult.hess`) was changed.

## Verification

All runs on this branch, real KNITRO, D=4 (`d4_exact_setup(δ=1.0, find_smallest=true)`):

| Run | Test | Result |
|---|---|---|
| Baseline (pre-edit) | `test_shared_core_hessian_d4_gates.jl` | 40 PASS / 0 FAIL |
| After origin-ZC dead-mirror deletion | same | 40 PASS / 0 FAIL |
| After CM/CM+ZC pack-loop optimization (both files) | same | 40 PASS / 0 FAIL |
| After extracting the shared `pack_upper_cm_hessian!` function | same | 40 PASS / 0 FAIL |
| Common-Fréchet (untouched, confirming no collateral effect) | `test_frechet_hessian_structured_vs_dense_d4.jl` | 20 PASS / 0 FAIL |
| New sentinel test | `test_hessian_upper_only_sentinel_d4.jl` | see below |

Every one of these tests exercises real KNITRO inner solves (not synthetic points only) — Section
A/B/C/D of `test_shared_core_hessian_d4_gates.jl` each include a "real solved x*" comparison arm in
addition to `x=0` and random points.

### Sentinel test result

`test_hessian_upper_only_sentinel_d4.jl` builds a real flexible-CM production context, solves to a
real dual point, runs the (now-shared) production Hessian callback once for a reference packed
result, then **directly poisons `cctx.Hfull[NCORE+1:n, 1:NCORE]` (the H_EC mirror sub-block) with
`NaN`** and re-packs via `pack_upper_cm_hessian!` alone (not a fresh callback call, which would
just overwrite the poison before packing runs). It requires the re-packed result to be (a) free of
`NaN` and (b) bit-identical to the unpoisoned reference. A negative control separately poisons one
column of the H_CC region and requires the poison **to** propagate — proving the test itself isn't
vacuous (i.e., confirming `pack_upper_cm_hessian!` does genuinely still read *some* lower-triangle
entries, just not the H_EC ones).

```
PASS  CM: inner solve feasible
  poisoned-region NaN count in packed result: 0   max|Δ| vs unpoisoned pack: 0.0
PASS  CM: packed upper-triangle result has zero NaN after poisoning the H_EC mirror region
PASS  CM: packed upper-triangle result unchanged after poisoning the H_EC mirror region
  H_CC-region poison propagated into packed result (expected true): true
PASS  CM: negative control -- H_CC poison DOES propagate (sentinel is not vacuous)
PASS  originZC: dead-mirror removal verified by Section D of test_shared_core_hessian_d4_gates.jl
ALL HESSIAN UPPER-ONLY SENTINEL TESTS PASSED   (5/5, real KNITRO inner solve, D=4)
```

## Scope note: full task-spec sentinel not yet achievable

The task's own §5 spec describes poisoning **every** strict-lower-triangle entry and requiring the
packed result unchanged everywhere — that is not true of the current architecture for the H_CC
region (genuinely independently accumulated, not mirrored, per above), so a poison-everything
sentinel would correctly fail there today. This session's sentinel is deliberately scoped to the
region the packing-loop fix actually stopped depending on (H_EC), plus a negative control proving
it isn't vacuous. Reaching the full task-spec sentinel would require also restructuring H_CC's own
accumulation to be genuinely upper-only (fold the `(lp,l)` contribution into the `(l,lp)` entry
directly rather than computing both) — a larger, independent change not attempted this session.

## Counters (task §10)

```
production_hessian_upper_entries_written     = wired (unchanged -- every h[k] write is upper-only
                                                 by construction in all 5 families, before and
                                                 after this session's changes)
production_hessian_lower_entries_written     > 0   (Hfull's own mirror writes, e.g.
                                                 `Hfull[cols,1:NCORE].=transpose(block_ec)`, remain
                                                 -- deliberately not removed, see rationale above;
                                                 the origin-ZC dead mirror is the one exception,
                                                 now 0 there)
production_hessian_symmetrization_passes     = reduced, not zero (H_EC's averaging pass removed
                                                 for flexible-CM/CM+ZC; H_EE/H_CC/common-Fréchet's
                                                 4 analogous sites still average)
production_hessian_lower_triangle_reads      = reduced, not zero (same scope as above)
```

## Verdict

```
PRODUCTION_HESSIAN_ASSEMBLY = remaining_symmetrization_H_EE_H_CC_and_all_common_frechet_blocks
PRODUCTION_HESSIAN_LOWER_ENTRIES_WRITTEN = >0 (origin-ZC's one dead site fixed; CM/CM+ZC/common-
                                                Frechet's mirror WRITES intentionally retained,
                                                see rationale; only unnecessary READS removed)
PRODUCTION_HESSIAN_SYMMETRIZATION_PASSES = >0 (H_EC removed for flexible-CM/CM+ZC; H_EE/H_CC/all
                                                common-Frechet sites retained)
PRODUCTION_HESSIAN_LOWER_TRIANGLE_READS = >0 (same scope)
PACKED_UPPER_SENTINEL_TEST = pass_flexible_cm_and_cmzc_H_EC_region_scoped
                              (not pass_all_families -- common-Fréchet not attempted; H_EE/H_CC
                               genuinely still read the lower triangle by design, not a gap)
```

This is a genuine, gated, real reduction in wasted per-callback compute for 3 of 5 production
families (unrestricted needed no change; common-Fréchet is the clear next target), not the full
task-spec end state. Reported honestly as partial, verified progress rather than claimed complete.
