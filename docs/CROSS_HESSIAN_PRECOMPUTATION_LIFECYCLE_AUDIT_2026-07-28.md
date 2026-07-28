# Cross-Hessian precomputation lifecycle audit (2026-07-28 ADDENDUM)

Classifying every input/operation touched by the four cross-Hessian kernels in this task by
lifecycle, per the addendum's own `CONTEXT_STATIC` / `OUTER_POINT_STATIC` / `INNER_DUAL_DYNAMIC`
taxonomy.

## 1. Context-static (built once per fixed-theta context, never rebuilt during an inner solve)

- CM bin identities (`cctx.Bidx`, built once in `build_cm_bin_ctx`/`build_cm_meanzc_bin_ctx` from
  `ctx.U`/thresholds — theta-independent since `U` is the fixed draw matrix).
- Cumulative-basis / origin-contrast transforms (`cctx.R`, `cctx.CScum`/`CT` SHAPE — the prefix-sum
  ARRAYS are refreshed per Hessian callback, but their transform LOGIC is static).
- Raw ZC features `Φ` (`ZCRestrictionOperator.Zraw_all`/`Zpairraw_all`, and this task's own
  `ZCRawWeightedWorkspace.Phi`, built ONCE in `build_zc_raw_weighted_workspace` — confirmed by
  direct code reading, never reassigned after construction).
- Persistent thread workspaces: `ws.tasks_ec`/`tasks_ez`/`tasks_cz`, `ws.thread_scratch_ez` (all
  four new threaded kernels), `raw_ws.tasks` (H_ZZ threaded_packed) — sized to `Threads.nthreads()`
  at construction, never resized per callback.

`context-static rebuilds during inner solve = 0` — verified by code reading: `Bidx`/`Φ`/`R` are
never reassigned inside any Hessian callback path touched by this task; the persistent scratch
structs (`WinnerBinCrossScratch`, `WinnerZCCrossScratch`, `BinZCrossScratch`, `ZCCenteredScratch`,
`ZCRawWeightedWorkspace`) are only rebuilt by their own `ensure_*!` functions on a genuine SIZE
change (campaign-lifetime constant in practice — confirmed the same "campaign-lifetime constant"
invariant this codebase's OTHER `ensure_*_scratch!` functions already documented, not a new claim).

## 2. Outer-point-static (rebuilt once per new outer point, i.e. once per KNITRO inner solve)

- Winner identities/values (`wctx.winner`/`wctx.y`, owned by `core_ws`/`CompressedFactual`, rebuilt
  by `_fill_cm_HEE!`/`archA_partitioned_hess_cb_builder`'s own `core_ws_for !== cf` check — NOT
  rebuilt every Hessian callback, only when the compressed-factual identity changes, which happens
  at most once per outer-point's worth of dual iterations in practice per this codebase's own
  documented `core_cf_ref` discipline).
- Economic empirical-target vectors (`wctx.pi_vec`, part of `core_ws`, same lifecycle as above).
- ZC centering targets (`ZCRestrictionWorkspace.targets_mean`/`targets_pair`, refreshed via
  `refresh_zc_targets!` — called once per inner solve by `_fill_cm_HEE!`/
  `archA_partitioned_hess_cb_builder`, confirmed by direct code reading, not per-callback).
- `raw_zc_ws.tvec` (this task's flat target vector, `refresh_zc_raw_target_vector!`) — same
  lifecycle, called once alongside `refresh_zc_targets!`.

`outer-point-static rebuilds per inner iteration = 0` for the pieces above — confirmed by the
existing `core_ws_for !== cf` / `refresh_zc_targets!`'s own call-site placement (outside the
Hessian-callback-frequency loop). One CAVEAT found by this audit, not newly introduced by this
task: `refresh_zc_centered!` (which builds `cs.Zc`, needed by H_EZ's `Z` argument) is called ONCE
PER HESSIAN CALLBACK (dual-dynamic, correctly so — `Zc` does not depend on `S`, only on the
outer-point's fixed targets, so it is technically OUTER-POINT-static and is being recomputed more
often than strictly necessary), but this task did NOT change that call frequency (out of scope —
changing it would require caching `Zc` across Hessian callbacks within the same inner solve, a
larger architectural change than this task's kernel-threading scope). Flagged as
`HIGHEST_PRIORITY_REMAINING_GAP` candidate in the master report.

## 3. Inner-dual-dynamic (must change every Hessian callback — genuinely `S`-dependent)

- `S = Ψ''(r)` itself (`obj.arg2` after `ddPsi!`).
- All four kernels' own weighted sufficient statistics: `QTab`/`NuTab`/`SOnlyTab`/`QCfTab` (H_EC),
  `Snu`/`v` (H_EZ), `ZBinTab` from `ZcS` (H_CZ — `ZcS` itself is `S`-weighted, genuinely dynamic),
  `RW`/`u`/`s0`/raw Gram (H_ZZ's new candidates).

This is the irreducible per-callback work this task's threading targets — nothing here can be
cached across callbacks within the same inner solve (a full `KN_solve` typically fires the Hessian
callback once per KNITRO iteration, each at a genuinely different dual point).

## 4. Exact same-point cache

**Not implemented by this task** — no `(outer-context generation, exact inner dual vector,
valid flag)` cache was added. Rationale: the sub-block profile (see
`CROSS_HESSIAN_STATIC_VS_DYNAMIC_COSTS_2026-07-28.csv`) found the dominant cost is the raw-table
FILL itself (a genuine function of the current `S`, changes every callback in practice — KNITRO's
own trust-region/line-search rarely revisits an EXACT prior dual point within one solve), not
redundant recomputation at an unchanged point. Adding same-point caching would add complexity
without addressing the measured bottleneck — flagged as considered-and-deprioritized, not
overlooked.

## Static map rebuilds inside the Hessian callback

`static_map_rebuilds_inside_hessian = 0` for every kernel this task threaded — confirmed directly:
none of `winner_pair_cross_hessian_fill_threaded!`, `winner_pair_cross_hessian_zc_block_threaded!`,
`bin_zc_cross_hessian_fill_threaded!`, or any of the three new H_ZZ candidates ever reconstruct
`Bidx`, winner/bin group membership, or output indexing — all consumed as read-only inputs built
by the (context- or outer-point-static) callers described above.
