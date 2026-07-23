# Complete-state exact/cross-delta cache — design and mutability audit — 2026-07-22

## 1. What `solve_base_state`/`archC_verified_state` actually produce, field by field

Traced by direct code read (`three_way_derivatives.jl:14-30`, `cm_production_bundle.jl:159-192`).

`BaseDualState` (`three_way_derivatives.jl:14-20`):

| Field | Type | Source | Mutable/aliased scratch? |
|---|---|---|---|
| `x_free0` | `Vector{Float64}` | `collect(x_free0)` | NO — fresh `collect`, owned |
| `θ_full0` | `Vector{Float64}` | `CS.reconstruct_full(...)` | NO — fresh allocation |
| `ζstar` | `Float64` | scalar | trivially immutable |
| `λstar` | `Vector{Float64}` | `collect(inner_x[2:end])` | NO — fresh `collect` |
| `m_star` | `Vector{Float64}` | `copy(obj.arg1)` | NO — explicit `copy`, NOT the live `obj.arg1` buffer itself (which IS mutable scratch, overwritten by the next inner solve — the `copy` is exactly what breaks that alias) |
| `inner_status` | `Int` | scalar | trivially immutable |

`verify` NamedTuple (`archC_verified_state`, `cm_production_bundle.jl:186-190`): `inner_status, Delta_dual, Delta_primal, primal_dual_gap, weight_norm_resid, mean_m_resid, max_abs_moment_kkt_resid, m_mean, m_min, m_max` — **all plain `Int`/`Float64` scalars**, no arrays, trivially safe to retain.

**Conclusion: `BaseDualState` + `verify` are ALREADY fully-owned, non-aliased immutable-by-convention data** (every `Vector` field is populated via `collect`/`copy` of a fresh allocation, never a workspace/scratch buffer reference) — this is not an assumption, it is provable from the four lines above and is already exploited by existing production code: `cm_checkpoint.jl`'s own `last_F_state[] = (w = copy(w), base = base)` already retains a `base::BaseDualState` across the `cb_F!`→`cb_G!` callback boundary (a different KNITRO callback invocation, with arbitrary other work — including other inner solves on other threads — potentially happening in between). If `BaseDualState` were NOT safe to retain past the call that produced it, that existing pattern would already be broken. This materially simplifies the cache: **no defensive re-copying is needed when storing or restoring these two objects** — they can be stored and returned by direct reference.

## 2. What the gradient callback needs from a restored state

`cm_production_gradient(x_free0, pcx, ctx, pe; base=nothing, ...)` / `cm_production_gradient_cplus(...)`: **only `base::BaseDualState`** (`cm_production_bundle.jl:202-207`, `lfix_cm_cplus.jl`'s `cm_production_gradient_cplus`). `cb_F!` needs `verify` too (to report `Delta_dual` and gate `is_verified_success`). Nothing downstream needs the `pcx`/`ctx`/`cctx` objects to be part of the cached entry — those are reconstructed once per process/run from the (already-fingerprint-covered) config, not per outer point.

## 3. Field classification (per the task brief's three categories)

- **Intrinsic to Δ*(θ), reusable across δ** (the entire cached entry): `base` (all 6 fields), `verify` (all 10 fields) — none of these depend on δ. Confirmed structurally: `archC_verified_state(x_free0, ctx_cm, cctx)` takes no `δ` argument at all; δ only enters via the KNITRO outer constraint bound (`KN_set_con_upbnd(kc, cIndices[1], delta)`, `cm_checkpoint.jl:223`), never the inner solve.
- **Caller-dependent, must be recomputed at lookup, NEVER cached**: `feasible = isfinite(Δ) && Δ <= delta + 1e-6` (`cm_checkpoint.jl:298`) and anything derived from comparing `Delta_dual` to the CALLER's `delta` — recomputed fresh from `verify.Delta_dual` (cached) and the caller's own `delta` (not cached) at every lookup. This is a one-line comparison, not stored.
- **Direction/backend-dependent, must enter the fingerprint**: `find_smallest` (currently always `true` in production, but still part of the fingerprint below in case that ever changes), `cm_hessian_backend` (the INNER-solve Hessian architecture — changing it changes the converged `(ζ*,λ*)` itself, unlike the OUTER gradient backend selector `cm_gradient_backend`, which does NOT change `base`/`verify` at all and is correctly excluded from the fingerprint — a cache entry solved under `:reference` is exactly as valid for a `:cplus` lookup and vice versa, since both consume the identical `BaseDualState`).

## 4. Cache key / context fingerprint

A cache entry is keyed by `(context_fingerprint, outer_point_key)`:

- `outer_point_key`: `hash(round.(x_free0, digits=12))` — the outer point IS the natural cache key (this cache serves the "cb_F! and cb_G! for the SAME KNITRO-reported point" and "a later stage/restart revisits a point already solved" cases, not a nearest-neighbor/interpolation cache).
- `context_fingerprint` covers, per the brief's own required list: `ctx.draw_meta.checksum_uniform`, `ctx.draw_meta.checksum_transformed` (draws), `W`, `D`, `ctx.σ` (`σ` is fixed at context-build time; `μHat` varies per-point via `θ_full0[1]` and is part of the outer point, not the fingerprint), `L`, `contrasts`, `cm_basis` (`:cumulative`, currently hardcoded but included explicitly rather than assumed), the exact `probs` vector, `cm_hessian_backend`, `find_smallest`, the `.opt` file's own content hash (not just its filename — a filename collision across two differently-edited `.opt` files must not silently share entries), and the KNITRO release string (`KNITRO.KN_get_release()`, already computed once per run in `cm_checkpoint.jl`). A schema-version constant for this cache itself is also folded in, so a future field-list change cannot silently reuse stale entries across a code upgrade.

`cm_gradient_backend` (`:reference`/`:cplus`) is deliberately **excluded** from the fingerprint (see §3) — both backends may read the SAME cached `base`, and doing so is itself one of the required combination gates (§4 of the task brief: "cache hit + CM-C+", "cache hit + CM-Reference" using the identical stored state).

## 5. What must NEVER be stored

Only a result with `is_verified_success(verify)` true (`oracle.jl`'s existing `classify_inner_result`/`is_verified_success`, reused not re-derived — the SAME gate `cb_F!`/`cm_cold_verify.jl` already use to decide incumbent-worthiness) may be stored. A `CMExpectedSolveFailure` (infeasible/unbounded/failed inner dual solve), a bare numerical-failure (`inner_status` outside `(0,-100,-101,-103)`), or an `ExactInfeasible` classification (`inner_status <= -9000`, `classify_inner_result`, `oracle.jl:281`) must never produce a cache entry — there is no `BaseDualState` to store in those cases (the exception is thrown before one is constructed), so this is enforced structurally by only ever calling the cache's store path from inside the SAME try/catch that already narrows to `CMExpectedSolveFailure` at the real call sites, not by an extra runtime check that could itself have a gap.

## 6. Bounded LRU design

No new external dependency is introduced (`DataStructures.jl`/`OrderedDict` are not in this project's `Project.toml`). A `Dict{UInt64,CompleteStateEntry}` plus a monotonic `access_counter::Int` stamped onto each entry on insert/hit gives LRU semantics via an O(n) linear scan for the minimum-counter entry on eviction — acceptable because `max_entries` is expected to be small (tens, not thousands: one campaign's own accepted-point/checkpoint/restart working set, not a general memoization cache). See `complete_state_cache.jl`.

## 7. Instrumentation

`enabled, max_entries, hits, misses, evictions, inner_solves_avoided, bytes_current, bytes_peak, hit_restore_wall, fresh_base_solve_wall_counterfactual` — all as plain mutable-struct fields on `CompleteStateCache`, matching the brief's required list exactly. `bytes` per entry is computed once at insert (`Base.summarysize` on the stored `base`+`verify`, cheap relative to an inner solve).

## 8. Status

Implemented as an opt-in prototype in `complete_state_cache.jl` (default `enabled=false`, zero behavior change when omitted) and tested at D=4 (A/B/A restore-vs-cold-solve equality, context-mismatch rejection, draw-checksum-mismatch rejection, LRU eviction, never-store-on-failure). **Not wired into `cm_checkpoint.jl`'s `cb_F!`/`cb_G!` and not measured at D=20** in this session — see the final report for why (time/resource budget, and per the task brief's own explicit allowance: "It is acceptable to finish CM-C+ and leave the complete-state cache as a documented prototype").
