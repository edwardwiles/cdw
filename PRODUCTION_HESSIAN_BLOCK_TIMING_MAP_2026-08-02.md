# Production Hessian Block Timing Map (2026-08-02)

Measured via the canonical harness's `AUDIT_BLOCK_TIMING=1` mode: enables the existing
`@cmhess_prof` sub-block instrumentation (`cm_hessian_subblock_profiling.jl`, already wired for
flexible_cm/common_frechet/cm_meanzc/origin_zc; **not wired for unrestricted** -- see gap note
below), clears the profiling buffers immediately after the true-cold solve, then captures 20
repeated frozen-state Hessian callbacks (D=20, W=20,000, real calibration point, post-fix code).
Raw per-label CSVs: `results/production_hessian_audit_2026-08-02/*_block_timing_*.csv`.

## Coverage requirement (task brief: sum of block timers / total callback time >= 0.99)

The harness's naive `sum(all labels)/coarse` ratio initially came out **above 1.0** for 3 of 4
families (flexible_cm 106.1%, common_frechet 104.4%, cm_meanzc 139.9%) -- not a bug, but exactly
the documented, intentional behavior of this codebase's own profiler: `@cmhess_prof`'s docstring
states nested blocks double-count inclusive time by design. Tracing the actual call structure
confirms `"H_EE"` (recorded at the `hessian_cm_structured_v2!` call site,
`@cmhess_prof "H_EE" _fill_cm_HEE!(...)`) is a **parent** label whose own body contains the nested
child labels `"H_EE_core"` (all 3 families) plus, for cm_meanzc only, `"H_ER_prep"`/`"H_ER"`/
`"H_ZZ"` -- confirmed numerically (`H_EE`'s total_s ≈ sum of its own children's total_s to within
~0.5%, the wrapper's own negligible overhead). Excluding the redundant parent `"H_EE"` label (origin_zc
has no such parent/child pair -- its own labels are already a flat sibling set) gives the corrected,
non-double-counted coverage:

| family | corrected coverage |
|---|---|
| cm_meanzc | **99.78%** |
| common_frechet | **99.95%** |
| flexible_cm | **99.96%** |
| origin_zc | **99.98%** |

All 4 instrumented families meet the >=99% requirement.

## Hot-path breakdown (share of total Hessian-callback wall time, leaf labels only)

### cm_meanzc (n=1962, W=20,000)

| block | share | note |
|---|---|---|
| H_CZ_prep | **28.9%** | `hcz_prep_dispatch!`, `draw_chunk_reordered` backend -- the single largest block |
| H_ZZ | 19.5% | `zc_gram_dispatch!`, `blas_syrk` backend |
| bintables_prep | 16.2% | `build_bin_tables_threaded!`/`prefix_sum_tables_threaded!` |
| H_ER | 11.9% | `winner_pair_cross_hessian_zc_block_drawmajor_v2!` (the H_EM/H_EZ fill) |
| H_ER_prep | 6.8% | ZC-target refresh + centered-scratch prep |
| H_EC_prep | 5.8% | winner-bin cross-Hessian prep (core x CM-grid) |
| H_EE_core | 1.7% | winner-pair core (the block this audit's other allocation fix does NOT touch -- unaffected) |
| (remaining: H_CC, H_EC_asm, packing, misc, ddpsi) | ~9.2% combined | |

H_CZ_prep + H_ZZ + H_ER + H_ER_prep together = **67.1%** of the callback -- the ZC-specific
machinery (present only in cm_meanzc among the CM families) dominates cm_meanzc's wall time, not
the shared H_EE/H_EC/H_CC core that flexible_cm/common_frechet also run.

### common_frechet (n=1382, W=20,000) / flexible_cm (n=1332, W=20,000)

| block | common_frechet | flexible_cm |
|---|---|---|
| bintables_prep | **54.8%** | **55.6%** |
| H_EC_prep | 15.6% | 18.6% |
| H_CC | 7.8% | 9.6% |
| H_EE_core | 4.5% | 6.1% |
| packing | 4.6% | 4.6% |
| H_EC_asm | 3.4% | 3.5% |
| level blocks (H_EF+H_CF+H_FF+level_table_prep, common_frechet only) | 2.1% | n/a |

`bintables_prep` -- the O(W) bin-contingency table build shared by every consumer block (H_EC,
H_CC, and for common_frechet the level blocks too) -- **dominates both families at ~55% of total
callback time**, more than 3x the next-largest block. This is expected given the audit's own
allocation findings: this step does the actual O(W*(D*NCORE+D^2)) scan every callback, and neither
of this audit's two accepted fixes touched it (both fixes were in the FILL step that reads the
already-built tables, not the table-build step itself). Notably, common_frechet's level blocks
(H_EF/H_CF/H_FF/level_table_prep combined) are a small ~2.1% of wall time despite being where this
audit's allocation fix lived -- consistent with Gate 4's finding that the allocation fix barely
moved common_frechet's total callback wall time (2.8%): the fixed loop's own arithmetic, not its
now-eliminated allocation, was always the dominant cost within that already-small block.

### origin_zc (n=1012, W=20,000)

| block | share |
|---|---|
| originZC_H_ZZ_gram | **55.5%** |
| originZC_H_EZ_fill | 38.9% |
| originZC_H_EE_core | 3.7% |
| originZC_pack | 1.6% |
| (misc, H_EZ_prep, H_ZZ_weight) | ~0.3% combined |

H_ZZ_gram + H_EZ_fill = **94.4%** of origin_zc's callback -- unlike the CM families, origin_zc has
no shared bin-table-build step at all (no CM-grid block exists for this family), so its cost is
concentrated almost entirely in the two ZC-specific blocks.

## unrestricted: instrumentation gap (confirmed, not fixed in this pass)

`_callbackEvalH_inner_compressed!` (`compressed_live.jl:257`) has only the coarse
`@prof "inner_dual_hessian_callback_compressed"` label -- no `@cmhess_prof` sub-block breakdown
exists for this family at all (confirmed by `grep`, not assumed). At W=100,000 this family's
mean callback is only ~30ms (vs 250ms-1.4s for the other 4 families) and its packed Hessian **is**
the whole H_EE block with no CM/ZC/level extension -- i.e. there is only one real block to time,
so the practical value of adding fine-grained sub-block timing here is much lower than for the
other 4 families. Not added in this pass (would require modifying
`hessian_core_winner_pair!`/`winner_pair_hessian!` internals, a larger change than this audit's
allocation-focused scope); flagged as a real, confirmed gap for the regression-safeguards phase.

## Method note: W dependence

Timings above are absolute at W=20,000; PERCENTAGE SHARES (the numbers actually reported here)
are expected to be stable across W since every one of these blocks scales as O(W) or O(W*small)
uniformly (confirmed indirectly by the allocation-baseline CSV's own finding that per-call BYTES
are W-independent across all 5 families -- the same underlying kernels, run at different W, with
no W-dependent control-flow branch found anywhere in this audit's source reading). Not
independently re-verified at W=100,000 in this pass.
