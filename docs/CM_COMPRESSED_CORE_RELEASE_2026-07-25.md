# CM compressed-core allocation release — 2026-07-25

Task §1.2/§4/§5. Status: **MERGED** (ancestor of `production/fullA-exact`, tagged
`cm-compressed-core-production-ready-2026-07-25`).

Cherry-picked from the source feature branch unchanged (commits `23f9377`, `0015e45`, `c5259d7`),
re-verified against real licensed KNITRO this session (11/11 public entry-point assertions, the
CM threaded-Hessian P0/P1/P2 sweep, the matched 300s before/after run).

## What was fixed

- **`bview*R` → `mul!`, persistent `Hraw_EC`/`block_ec` Hessian scratch** (`23f9377`):
  38.32 MB → 0.3 MB/call (`fill_cm_columns_from_bins!`); Hessian scratch 130,112 → 240 bytes/call.
- **Compressed winner-form core moments** (`0015e45`) — the largest single item in the whole
  port: CM's core (non-CM) moment columns ported from the dense
  `EK_moments_gammanorm_directgp!` path onto the same compressed winner-form representation the
  unrestricted family already uses. **760.76 MB → 65.02 MB per `moments!` call (91.45%
  reduction)**. 22/22 tests on the source branch: bit-identical `K`, core/gravity columns to
  FP-noise level (~2.3e-13), full inner-solve agreement (ζ\*, λ\*, Δ\*, KKT residual), all three
  Hessian blocks agreeing to ~1e-12–1e-13, C+ outer gradient agreeing to ~1.9e-14.
- **`Hraw_CC`/`RtHraw_CC`/`block_cc`** (`c5259d7`) — a *second*, larger allocation bug found live
  while porting the threaded Hessian (section 6.1), missed by the original section-4.2 fix (which
  only named the smaller `H_EC` sibling): `Hraw_CC`'s R-congruence product was allocating **~1.28
  GB per single Hessian callback** (L²=2,500 iterations at L=50). Fixed the same way: 6.28 MB/callback
  (99.5%). Applied to both the serial and threaded Hessian implementations.

## Preserved (per task §1.2's explicit requirements)

Cumulative CDF contrast basis, bin-index internal storage, orthonormal origin contrasts, exact
Architecture-C Hessian algebra, current C+ outer gradient — all confirmed by the 22/22
correctness suite above and by this session's own 11/11 public entry-point assertions
(`cm.cm_restriction_basis=cumulative`, `cm.cm_internal_storage=bin_index`).

## Re-verification this session

- Public entry-point assertions confirm `run_cm_upper_checkpointed` (both `:cm_only` and
  `:cm_plus_equal_means`) reaches `core_moment_representation=compressed_winner_form`.
- Matched 300s before/after: **41.702 GB → 19.652 GB user allocation (−52.9%)**, GC time
  −28.3%, cold-verify exact (diff=0.0). Outer-progress metrics in that same 300s window are
  confounded by a simultaneous, disclosed outer-algorithm change — see
  `MATCHED_300S_BEFORE_AFTER_2026-07-25.md` for the full discussion; the allocation numbers above
  are not affected by that confound.
