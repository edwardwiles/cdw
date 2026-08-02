# Production Hessian Workspace-Lifecycle Audit (2026-08-02)

Classification per the task brief: `CONTEXT_STATIC` (built once at context construction) /
`OUTER_POINT_STATIC` (rebuilt only when the outer point changes) / `INNER_DUAL_DYNAMIC` (updated
once per Hessian callback) / `CALLBACK_TEMPORARY` (should not exist as a large allocation).

Methodology: direct source inspection of every large object touched inside the 5 families'
Hessian callbacks, cross-checked against the two allocation hotspots already found and fixed
(commit `8f1151e`) via `Profile.Allocs` -- i.e. this audit is validated against real measured
allocation data, not a purely static read.

## CONTEXT_STATIC (built once per campaign/context, correctly never rebuilt per callback)

| object | family | built in | evidence |
|---|---|---|---|
| `cctx.Bidx` (bin indices) | flexible_cm/common_frechet/cm_meanzc | `build_cm_bin_ctx`/`build_cm_meanzc_bin_ctx` | theta-independent (draws U fixed once context built) |
| `cctx.CScum`/`cctx.CT` (prefix-summed bin-contingency tables) | same 3 | rebuilt only by `build_bin_tables!`/`prefix_sum_tables!`, called EVERY callback (see INNER_DUAL_DYNAMIC below -- these ARE per-callback, not context-static; corrected from an earlier draft of this audit that mis-classified them) | `hessian_cm_structured_v2!` calls these unconditionally each call |
| `cctx.tls::ThreadLocalBinScratch` (`Ttab`/`Stab` per-thread scratch) | flexible_cm/common_frechet/cm_meanzc | `build_thread_local_scratch(cctx)`, called once in the production context builder | `cm_hessian_threaded.jl:build_thread_local_scratch` -- confirmed NOT called inside the callback |
| `cctx.Hraw_EC`/`cctx.block_ec`/`cctx.Hraw_CC`/`cctx.RtHraw_CC`/`cctx.block_cc` | flexible_cm/common_frechet/cm_meanzc | `CMBinHessCtx` construction | comment at `cm_hessian_architectures.jl:459-467` documents this was the fix for a real prior per-call allocation bug |
| `ext.Wtab`/`ext.T1`/`ext.Esum_wb`/`ext.colsum`/`ext.Hraw_cmlevel`/`ext.block_cmlevel` (NEW) | common_frechet | `CMFrechetExtension` construction, cached on `cctx.frechet_ext_cache` via `_resolve_frechet_ext!` | `cm_frechet_hessian.jl:48-71`; `block_cmlevel` added by this audit (commit `8f1151e`) |
| `octx.` raw/gram/drawmajor scratch (origin_zc) | origin_zc | `build_originzc_core_hess_ctx` | mirrors the CM-family `ensure_*_scratch!` idiom |
| `archA_partitioned_hess_cb_builder`'s `scratch_full` | origin_zc | inside the BUILDER (closure construction), not inside the returned closure | `cm_hessian_architectures.jl:1618-1627` -- comment explicitly notes "built once per KNITRO solve, when this builder is constructed -- not once per Hessian callback invocation" |
| `st.core_ws`/`st.cf` (unrestricted) | unrestricted | `inner_loop_internal_compressed`, lazily on first callback (`st.core_ws === nothing` guard) | `compressed_live.jl:282-285` |

## OUTER_POINT_STATIC

None of the 5 families' Hessian machinery has an object that changes with the outer point but not
every callback -- at fixed W/L/K (this audit's scope), the only thing that varies call-to-call
within a single inner KNITRO solve is the dual iterate itself. Objects that depend on theta (the
outer point) are rebuilt once per `archC_base_state`/`archOZ_base_state`/etc. call (i.e. once per
inner solve, which IS "when the outer point changes" for this codebase's actual calling
convention) -- e.g. `cctx.core_cf_ref[]` (rebuilt inside `_fill_cm_HEE!` only when `cf` identity
changes, guarded by `cctx.core_ws_for !== cf`).

## INNER_DUAL_DYNAMIC (correctly updated once per callback, not more)

| object | update site | note |
|---|---|---|
| `cctx.Hfull` | `fill!(Hfull, 0.0)` at top of `hessian_cm_structured_v2!`, then filled block-by-block | persistent buffer, only `fill!`ed (no reallocation) |
| `cctx.CScum`/`cctx.CT`/`tls.Ttab`/`tls.Stab` | `build_bin_tables_threaded!`/`prefix_sum_tables_threaded!`, called once per callback | correctly re-derived from the CURRENT dual-weighted `w=arg2`, since the bin-contingency weighting depends on the dual point |
| `cctx.hzz_centered`/`cctx.zc_cross_scratch`/`cctx.raw_zc_ws` (ZC families) | `ensure_*_scratch!` (size-checked, ONLY reallocates on genuine size change) + `refresh_*!` (in-place value update) each callback | verified: `_ensure_cm_cross_scratch!`/`_ensure_zc_cross_scratch!`/`ensure_bin_zc_cross_scratch!` all follow the identical "if wrong size, rebuild; else return unchanged" idiom -- confirmed NOT the source of either allocation hotspot found in this audit (both hotspots were in the FILL step, not the ensure/scratch-sizing step) |

## CALLBACK_TEMPORARY (should not exist as large allocation) -- FINDINGS

Two real instances found and fixed in this audit pass (commit `8f1151e`):

1. **cm_meanzc**, `cm_hessian_architectures.jl` (pre-fix) `_fill_cm_HEE!`: `@views HEE[...] .=
   transpose(HEM)` -- a genuine `CALLBACK_TEMPORARY` (1.93MB/callback), now eliminated (explicit
   loop, zero allocation, bit-identical output).
2. **common_frechet**, `cm_frechet_hessian.jl` (pre-fix) `_fill_frechet_level_blocks!`:
   `cctx.R' * Hraw_cmlevel` inside a 2500-iteration loop -- a genuine `CALLBACK_TEMPORARY`
   (500KB/callback, 2500 small allocations), now eliminated (persistent `ext.block_cmlevel` +
   `mul!`, bit-identical output).

Both fixes reclassify their respective allocations from `CALLBACK_TEMPORARY` to
`INNER_DUAL_DYNAMIC` (a persistent buffer refreshed in place) / eliminated entirely (the explicit
loop needs no buffer at all).

## Remaining small `CALLBACK_TEMPORARY` items (not yet investigated for elimination)

Per the `Profile.Allocs` breakdowns already captured (see
`PRODUCTION_HESSIAN_ALLOCATION_BASELINE_2026-08-02.csv` and the diagnostic logs), the residual
~40-70KB/callback across all 5 families is dominated by:

- `Threads.@spawn`-related per-task overhead (`Task`, `Base.IntrusiveLinkedList{Task}`,
  `Base.GenericCondition{SpinLock}`, `SpinLock` objects -- a few hundred bytes each x
  `Threads.nthreads()` workers, recurring at every `Threads.@spawn` call site in
  `core_exact_hessian.jl`/`threaded_cross_hessian.jl`/`hez_drawmajor_v2_candidate_2026-08-01.jl`).
  This is Julia's OWN per-task bookkeeping overhead, not application-level scratch -- eliminating
  it would require restructuring the parallel-task dispatch itself (e.g. a persistent thread-pool
  pattern), which is a materially larger change than this audit's "surgical, one-change-at-a-time"
  policy scopes for without a much stronger measured justification (a few KB out of tens-of-KB is
  not currently a material fraction of total Hessian-callback wall time -- see the block-timing
  map for the wall-time side of this question, still in progress).
- `cm_originzc_target_layout.jl:85/99` (`mean_targets`/`pair_targets`) and
  `cm_meanzc_moments.jl:68` (`packed_pair_index`) -- small (hundreds of bytes to ~18KB), called a
  handful of times per callback; not yet traced to confirm whether these are genuinely
  recomputed every callback or could be hoisted to context-static. Flagged as a follow-up
  candidate, not yet designed or fixed in this pass.

No further large (>100KB) `CALLBACK_TEMPORARY` allocation was found in any of the 5 families'
frozen-state Hessian callbacks at W=20,000/100,000 after the two fixes above.
