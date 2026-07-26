# Restricted-Family Exact-Point Cache Release — Phase C — 2026-07-26

**State: MATCHED_AB_PASSED** (not merged to production/fullA-exact; on
`port/remediate-production-5x7-audit-2026-07-26`).

## What was found

`cm_config.jl` already contains a complete exact-point cache implementation (`CMEvalKey`,
`cm_production_value_v2`, `SafeExactCache{CMEvalKey}`) — but it is built against a **different,
incompatible** `pcx` shape (`pcx.cfg::CMConfig`, `cm_base_state_v2`) than what the real production
drivers actually construct. `run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed` both
build `pcx` via `build_cm_production_context`/`build_originzc_production_context`, whose return
shape (`ctx_cm, aug, bins, cctx` / analogous) has **no `cfg` field at all** — `cm_production_value_v2`
would `MethodError`/`FieldError` immediately if called against the real production `pcx`. This is
why the baseline audit found the caching infrastructure present in the tree but never invoked by
either driver: it isn't a drop-in for them, despite existing.

## What was built

`cm_exact_cache_production.jl` (new file): `CMProductionEvalKey`, a NEW key type built against the
REAL production `pcx`/`ctx_cm`/`cctx` shapes, reusing `SafeExactCache`/`_cache_lookup`/
`_cache_store!`/`context_fingerprint` (oracle.jl) completely unchanged — no cache primitive was
re-derived. Key fields: `x_free`, `nu` (restriction-parameter vector — empty for flexible-CM/
common-Frechet, real for CM+ZC's eta_nu / origin-ZC's eta_origin), `delta`, `find_smallest`,
`inner_loop_opt`, `family_tag` (`:flexible_cm`/`:common_frechet`/`:cm_meanzc`/`:origin_zc` —
prevents any cross-family collision even if `x_free` numerically coincided), `L`, `contrasts`,
`K_mean`, `K_pair`, `A_coordinate_mode`, `ctx_fingerprint`. Follows the SAME "key on the canonical
decoded `x_free`, not the raw outer search coordinate" convention the UNRESTRICTED family's own
`FullAEvalKey` already uses — audited and confirmed safe in the baseline static audit (a hit
across `A_coordinate_mode` values is intentional reuse of an equal mathematical point).

`cm_cache_lookup_or_compute!(cache, key, compute_fn)` — the shared wrapper both drivers' `cb_F!`
now call: on a hit, returns the stored `(base, verify)` without invoking `compute_fn` at all (the
real inner-solve skip); on a miss, computes and stores, but only if
`verify.inner_status in (0,-100,-101,-103)` (a genuine feasible/verified result — never caches an
infeasible/failed solve as reusable, matching `archC_verified_state`'s own acceptance gate).

Wired into:
- `run_cm_upper_checkpointed` (`cm_checkpoint.jl`) — new `use_exact_cache::Bool=true` kwarg, a
  fresh `SafeExactCache` built once per driver call, `cb_F!`'s three-way dispatch
  (flexible/meanzc/frechet) now goes through the cache wrapper uniformly via one shared
  `cache_key` construction (`family_tag` set per branch).
- `run_originzc_upper_checkpointed` (`cm_originzc_checkpoint.jl`) — same pattern, `L`/`contrasts`
  set to sentinel `0`/`:none` (no CM grid exists for this family; `family_tag=:origin_zc` plus a
  fresh per-driver-call cache instance are what actually prevent any cross-config collision, since
  the cache is never shared across separate driver invocations).

Public counters (task-required): `CMExactCacheCounters` (`lookups`, `hits`, `misses`,
`store_rejections`), a module-level `CM_EXACT_CACHE_COUNTERS` Ref (same discipline as
`CORE_HESSIAN_COUNTERS`), `print_cm_exact_cache_counters()` for post-solve reporting.

## Correctness gate (real, D=4, ALL PASS)

`test_phaseC_exact_cache_correctness.jl` exercises the exact same call pattern `cb_F!` now uses
(`cm_cache_lookup_or_compute!` wrapping `archC_verified_state`), not a synthetic standalone test:

- First call at the calibration point: exactly one real inner solve (`CS.INNER_SOLVE_COUNT[]`
  advances by 1), recorded as a miss. **PASS.**
- Repeated identical point: **zero new inner solves** (`same_point_inner_resolves=0`, the task's
  required invariant) — `INNER_SOLVE_COUNT[]` does not advance, recorded as a hit, returns the
  byte-identical stored `base`/`Delta_dual`. **PASS.**
- A genuinely different (perturbed) point: a real inner solve, recorded as a miss, no
  false-positive collision with the calibration entry. **PASS.**
- Cache-off (`cache=nothing`) vs. cache-on, same point: both a byte-identical-stored-value
  comparison (cache hit, `==`, exact) and a fresh-recompute comparison (cache off, `isapprox`,
  ~1e-15/1e-17 agreement — the tiny difference is expected floating-point accumulation-order noise
  from a genuinely fresh threaded re-solve, not a defect; the cache's own correctness invariant is
  already fully proven by the exact-match cache-hit checks above). **PASS.**

Final counters from the gate: `lookups=3 hits=1 misses=2 store_rejections=0` — matches the test's
own call sequence exactly (miss, hit, miss).

## Not yet done (tracked separately)

- A live D=20 / full-driver-through-KNITRO gate (this gate tests the cache wrapper directly
  against the real `archC_verified_state` call, not a full `run_cm_upper_checkpointed` KNITRO
  run) — the wiring is structurally identical to what a full run exercises, but a live end-to-end
  confirmation at production scale is still owed, ideally alongside Phase I's 300s profiles (which
  will report real `lookups`/`hits`/`misses` from an actual outer solve).
- Same wiring for common-Frechet/CM+meanZC has not been independently gated at D=4/D=20 beyond the
  shared `cache_key` construction path in `cb_F!` (all three CM-family branches share the same
  wrapper call, so the mechanism is structurally identical, but a dedicated per-branch gate would
  strengthen this further).
