# Section 5 — ProfiledLFixCache / shared-workspace aliasing hazard

## The hazard, traced to its root cause

`ProfiledLFixCache` (`profiled_lfix_incremental_2026-08-01.jl:38`) is built by
`build_price_winner_base_cache` (same file), which calls
`ensure_shared_profiled_lfix_ws!(D, Ddest, W)` to get a persistent `LFixFactorizedWorkspace`, then
`build_winner_ref!(ws, x_free0, ctx)` (`lfix_factorized_workspace.jl`). `build_winner_ref!`'s own
docstring states plainly: "writes into `ws`'s persistent `logCC0`/`mulU`/winner-rank arrays instead
of allocating fresh. Returns a `WinnerRefCache` whose array fields alias `ws`'s buffers." Before
this fix, `build_price_winner_base_cache` propagated those ALIASES directly into the returned
`ProfiledLFixCache` (`logCC0 = ref.logCC0`, `winner0 = ref.winner`, etc. — no `copy`).

`ws` itself was owned by a single **module-level global** `SHARED_PROFILED_LFIX_WS_REF ::
Ref{Union{Nothing,LFixFactorizedWorkspace}}`, comment: "only one family's outer search is ever
active per process." That assumption held for the real single-threaded, single-cache-in-flight
production access pattern, but is violated by two things this repo's own outer-readiness work
requires:

1. **Concurrent cache builds on different threads** (e.g. two outer-gradient calls or two A/B arms
   evaluated in parallel) would have every thread mutating the SAME `ws.logCC0`/`ws.winner`/etc
   arrays in place — a genuine data race, not just a staleness risk.
2. **Same-thread overlapping cache lifetimes** (any code that holds onto an earlier
   `ProfiledLFixCache` — e.g. a decoded-state A/B script comparing point A and point B, or an
   `xA -> xB -> xA` sweep) would see cache A's `winner0`/`logCC0`/`mulU`/... silently mutate to
   cache B's values the moment cache B is built, because cache A never owned its own copy.

This is exactly the two properties task §5 asked to trace ownership/lifetimes for. Confirmed live
(not just read): `objectid(cacheA.logCC0)` and the corresponding array inside the shared workspace
were the SAME object before this fix (verified by direct read of `build_winner_ref!`'s return
values, which are literal field accesses on `ws`, not `copy(...)` calls).

## Trace through the layers task §5 asked for

- **Canonical runner** (`bin/run_profiled_model.jl`): calls `shared_family_outer_gradient` once per
  KNITRO gradient callback, single-threaded at the outer level (KNITRO itself is not
  multi-threaded across evaluations) — the real production pattern never had two gradient calls
  "in flight" simultaneously, so this specific hazard was latent, not yet triggered, in production.
- **Outer evaluator / threaded gradient engine**
  (`profiled_shared_economic_gradient_engine_2026-08-01.jl`): builds the cache ONCE, serially,
  BEFORE `profiled_composite_gradient_from_cache(...; threaded=...)`'s own per-coordinate loop —
  the threaded coordinate loop only READS `cache.winner0`/`cache.mulU`/etc (see
  `dest_contrib_reduced_o1`), never mutates them, so the existing `threaded=true` path was already
  safe from THIS specific hazard (it doesn't rebuild the cache mid-loop). The hazard is entirely at
  the cache-BUILD boundary, not inside a single gradient call's own coordinate loop.
- **Bandwidth cache** (`profiled_reduced_bandwidth_cache_2026-08-04.jl`): keys on
  manifest/family/formulation/coordinate-mode/layout/point, independent of `ProfiledLFixCache` —
  not implicated in this hazard (confirmed by grep: `ReducedBandwidthCache` never touches
  `SHARED_PROFILED_LFIX_WS_POOL`/`ProfiledLFixCache` fields directly, only stores/reuses bandwidth
  scalars).
- **Checkpoint/resume**: `CMCheckpointV11` never serializes a `ProfiledLFixCache` (checked by grep
  — no `ProfiledLFixCache` reference in `cm_checkpoint.jl`), so resume never risks reloading a
  stale-aliased cache across process boundaries. Not implicated.
- **Family-specific adapters** (`profiled_family_adapters_2026-08-01.jl`,
  `profiled_originzc_family_adapter_2026-08-02.jl`, `profiled_cmzc_family_adapter_2026-08-02.jl`):
  all route through the one shared `build_shared_profiled_lfix_cache`/`shared_family_outer_gradient`
  pair — no family-specific cache-build path exists to audit separately.

## Fix chosen: **B — owned workspace per build, no aliasing possible**

Two changes, both confined to `profiled_lfix_incremental_2026-08-01.jl` (a REDUCED-only file; does
not touch `lfix_factorized_workspace.jl`/`build_winner_ref!` themselves, which FULL production also
uses via `cm_production_gradient_cplus` — leaving those files' documented alias-return behavior
unchanged for FULL's own tightly-scoped single-call usage, per the task's "do not modify FULL
production" and "no inner kernel changes" constraints):

1. **Thread-indexed workspace pool.** `SHARED_PROFILED_LFIX_WS_POOL ::
   Vector{Ref{Union{Nothing,LFixFactorizedWorkspace}}}`, one slot per `Threads.threadid()`
   (mirroring the SAME `pool.slots[tid]` idiom FULL's own `GradWorkspacePool`
   (`lfix_base_workspace_pooled.jl`) already establishes as this codebase's precedent for
   exactly this problem — not a new pattern). Concurrent builds on different threads now never
   touch the same underlying arrays.
2. **Copy-out in `build_price_winner_base_cache`.** The returned NamedTuple now `copy`s
   `logCC0`/`mulU`/`winner0`/`winner_price0`/`runnerup0`/`runnerup_price0`/`third0`/`third_price0`
   instead of aliasing `ref`'s fields. Every `ProfiledLFixCache` now owns independent data,
   immune to a later cache build (on the same thread) reusing and overwriting the pool slot.
   Cost: copying compact `D x Ddest` / `W x D` tables, NOT the `W x D x Ddest` dense tensor the
   2026-08-02 performance closeout eliminated — negligible next to the O(W*D*Ddest) winner scan
   that already dominates `build_winner_ref!`.

## Required tests (task §5)

New file `test_cache_workspace_ownership_2026-08-04.jl` (real D4 KNITRO context, no mocks):

1. **Two-cache overlapping-lifetime test**: build cache A, snapshot its arrays, build cache B at a
   different point (same D/Ddest/W, forcing pool-slot reuse), confirm cache A's arrays are
   bit-identical to the snapshot.
2. **xA -> xB -> xA test**: rebuild at point A again after B; confirm the rebuild matches the
   original A snapshot exactly (no leaked state from B). (The higher-level gradient-vector version
   of this check also exists at real D20/W=20,000 —
   `test_threaded_gradient_gate_2026-08-04.jl`, re-run this session against this same fix:
   `xA->xB->xA freshness max abs diff (gA1 vs gA2) = 0.0`, PASS.)
3. **Parallel-thread cache test**: `Threads.@threads` builds a cache concurrently on every thread
   at a distinct point; every thread's `q0` is compared against an independently-computed serial
   reference for that same point.
4. **Different-(D,Ddest,W) rejection test**: forces a workspace-shape change on the same thread
   (`ensure_shared_profiled_lfix_ws!` with `W=37` then `W=41` then `W=37` again), confirms the pool
   slot rebuilds (new `objectid`) on a genuine shape change rather than silently returning
   wrong-shaped data.

Result: see `key_results/section5_cache_ownership.log` for the real run output. All four checks
PASS (`CACHE_WORKSPACE_OWNERSHIP_GATE: PASS`).

## Verdict

`CACHE_WORKSPACE_SAFETY = pass` — resolved via option B (owned workspace per thread + copy-out per
cache), not merely asserted safe by construction. Bandwidth caching (task §7-8) may now be
considered free of this specific hazard as a precondition; its own material-benefit measurement is
still a separate, not-yet-closed question (task §8).
