# Full-A_od exact solver: current state, 2026-07-22 (end of finalization session)

Canonical as-of-now snapshot, written at the END of the correctness-first finalization task
(Phase 0-7). Supersedes prior status claims in `docs/fullA_independent_audit_remediation.md`
(AUD-02 row only — see banner in that file), `docs/fullA_price_tensor_elimination_report.md`
(D=20 benchmark numbers only — see banner in that file), and this document's own earlier
mid-session revision (§2-6 below reflect the FINAL state, not the state at session start —
see `docs/fullA_ALLOCATION_CROSSDELTA_KB_GATE_2026-07-22.md` for the full session record of how
it got here). `docs/fullA_postmerge_allocation_productionization.md`/
`docs/fullA_factorized_price_production_gate.md` remain accurate as historical records of their
own session's state, not rewritten.

Written from `perf/fullA-factorized-price-production`, now several commits past `204c266` (the
tip at session start) — see `docs/fullA_REPO_MAP_2026-07-22.md` for the full branch/commit
picture and `git log --oneline 204c266..HEAD` for the itemized list.

## 1. Active production call path (verified in source)

Entry points: `run_profile_checkpointed` and `run_polish_checkpointed`, both in
`full_aod_diag/d4_exact/c10_d20_production_driver.jl`. Both load the outer KNITRO options via

```julia
KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
```

with `hessopt_tag::String = "sr1"` by default → **`csw_outer_wallclock_sr1.opt`**.

`cb_G!` in both entry points now dispatches on `resolved_backend = resolve_price_cache_backend(
label, use_pooled_gradient, price_cache_backend)`:

```julia
if resolved_backend == :pooled
    gfull, meta = composite_gradient_at_fast_pooled(xf, ctx, pe, grad_pool; base, threaded = true, ...)
elseif resolved_backend == :aplus
    gfull, meta = composite_gradient_at_Aplus(xf, ctx, pe, grad_pool, lfix_ws; base, threaded = true, ...)
elseif resolved_backend == :cplus
    gfull, meta = composite_gradient_at_Cplus(xf, ctx, pe, grad_pool, lfix_c_ws; base, threaded = true, ...)
elseif resolved_backend == :kbplus
    gfull, meta = composite_gradient_at_KBplus(xf, ctx, pe, grad_pool, lfix_kb_ws; base, threaded = true, ...)
else
    gfull, meta = composite_gradient_at_fast_buffered(xf, ctx, pe; base, threaded = true, ...)
end
```

`cb_F!` in both entry points calls `screened_eval(...; exact_cache = exact_cache, ...)`, where
`exact_cache = exact_cache_override !== nothing ? exact_cache_override : (use_exact_cache ? SafeExactCache() : nothing)`.
`exact_cache_override` accepts a `SafeExactCache` or a `CrossDeltaExactCache` (this session
fixed a real bug here — see §4). `run_staged_delta5_continuation` (`staged_delta5.jl`) threads a
`CrossDeltaExactCache` across stages via this parameter, opt-in via `cross_delta::Bool`, and now
also threads `price_cache_backend`/`use_pooled_gradient` through to every stage (another
this-session fix — see §4).

## 2. Current defaults (verified directly in source, post-session)

| Setting | Default | Changed this session? |
|---|---|---|
| `price_cache_backend` | **`:cplus`** (was: kwarg did not exist) | **Yes — new selector, new default** |
| `use_pooled_gradient` | `nothing`-sentinel; resolves to `:cplus` unless explicitly set (was `false::Bool`) | Yes — widened to `Union{Nothing,Bool}` so "not specified" is distinguishable from "explicitly false"; explicit `false`/`true` still mean literally `:buffered`/`:pooled`, never silently reinterpreted |
| `cross_delta` (staged continuation) | `false` (unchanged) | No — but now actually usable (see §4) |
| `maxit_override` | `nothing` (unchanged behavior: `maxit=1_000_000`) | New kwarg, opt-in |
| Outer `par_concurrent_evals` | `yes`, every actively-loaded outer `.opt` file | No (verified unchanged) |
| Application-level thread guard | Present, `PsiObjectiveBundle.jl` | No (verified unchanged) |

**Backward compatibility, explicitly verified** (`test_default_backend_dispatch.jl`, 9/9 pass):
a caller passing neither `use_pooled_gradient` nor `price_cache_backend` now gets `:cplus`
(the new default); a caller still passing the old `use_pooled_gradient::Bool` explicitly gets
EXACTLY the old behavior (`false`→`:buffered`, `true`→`:pooled`), never the new default.

## 3. Source / merged / active / default / tested matrix (final)

| Component | Source | Merged into tip | Called by driver | Default-on | Tested |
|---|---|---|---|---|---|
| `GradWorkspacePool` / `:pooled` | `gradient_workspace.jl` | Yes | Yes, via selector | No | Fn-level 9/9 + real driver 4/4 + 5-backend smoke 12/12 |
| `LFixBaseWorkspace` / `:aplus` | `lfix_base_workspace*.jl` | Yes | **Yes, via selector** (was: no) | No | Real driver smoke 12/12 |
| Factorized `:cplus` | `lfix_factorized*.jl` | Yes | **Yes, via selector** (was: no) | **Yes — new default** | D=4/D=20 exhaustive (prior sessions) + real driver smoke 12/12 + independent directional check 2/2 + cross_delta integration 2/2 + checkpoint/resume 2/2 + δ=1/δ=2 trajectories — **all gates closed this session (8/8, `c23_cplus_gate.jl`)** |
| `:kbplus` (NEW) | `lfix_kbplus*.jl` | Yes | Yes, via selector | No (correct but ~15-17% slower than C+, measured — see §6) | D=4 116/116, D=20 fixed-point 3/3, real trajectory 3/3, independent directional check 2/2 (11/11 total, `c21_kbplus_d20_gate.jl`) |
| `CrossDeltaExactCache` | `cross_delta_cache.jl` | Yes | Yes, opt-in | No | **Now real-driver-verified**: 19/19 (`c19_cross_delta_gate.jl`) — live-fire, A/B/A regression, staged continuation (eval- and wall-clock-matched, real hits observed), cold reverify |
| Exact-point cache / verified-success gate (AUD-04) | `oracle.jl` | Yes | Yes | Yes | Yes |
| Checkpoint/resume verified-success gates (AUD-09/10) | `c10_d20_production_driver.jl` | Yes | Yes | Yes | Yes; CM's own D=20/L=50 campaign still not run (pre-existing gap) |

## 4. Real bugs found and fixed this session (not pre-existing findings — discovered live)

1. **`screened_eval`'s `exact_cache` keyword was typed `Union{Nothing,SafeExactCache}`**, one
   type narrower than `exact_cache_override`'s own `Union{Nothing,SafeExactCache,CrossDeltaExactCache}`.
   Julia enforces keyword-argument types at the call site (confirmed via a minimal repro:
   `TypeError`, not silent widening) — every `cb_F!`/`cb_G!` call forwarding `exact_cache=
   exact_cache` would `TypeError` the instant it held a `CrossDeltaExactCache`, i.e. the instant
   `cross_delta=true` was ever used through the real driver. This is why no staged
   `cross_delta=true` continuation had ever been observed to complete through production before
   this session — it could not have. Fixed: widened the type.
2. **`run_staged_delta5_continuation` had no `price_cache_backend`/`use_pooled_gradient` kwargs
   at all** — `cross_delta` combined with a non-default gradient backend had never been
   reachable through the staged-continuation entry point. Fixed: threaded through, matching the
   existing `maxit_override` pattern.
3. A latent test-only bug (`test_driver_pooled_gradient_wiring.jl`'s `knitro_status isa Int`,
   which can never pass for KNITRO's native `Int32` status on a 64-bit build — `isa Integer`).

## 5. AUD-02 supersession (critical — do not undo)

Unchanged from earlier in this session: outer `par_concurrent_evals` must stay `yes`. See
`docs/fullA_nested_knitro_solve_hang_rootcause.md` for the full root-cause writeup. **Do not set
it to `no` again without re-reading that document in full.**

## 6. Backend adoption decision (Phase 6, final)

Fair benchmark (`c22_phase6_fair_benchmark.jl`, warm-compiled, median of 3, real D=20/W=80,000,
correctness cross-checked against the actual production default before every timing run):

| Backend | Speedup vs prior default | Memory vs prior default | Correctness vs Reference |
|---|---:|---:|---:|
| pooled | 1.04-1.06x | 4.93x less | bit-exact |
| A+ | 1.08-1.16x | 26.96x less | bit-exact |
| **C+** | **4.01-4.20x** | **66.77x less** | 4.3e-17 |
| kbplus | 3.58-3.71x | 61.13x less | 7.1e-17 |

`:kbplus` (the reference-aligned, no-W-scale-exp factorization this session built per the
finalization brief's own Phase 4 request) is fully correct but **measured ~15-17% slower than
C+**, despite eliminating every W-scale `exp` call from the coordinate-probe hot path — the
per-probe `constConsσ = constCons.^(1-σ)` power operation it needs instead costs more than the
single scalar `exp()` C+ already has in hand from its ranking scan. A real, measured, honest
negative result — not hidden, not tuned away.

Per the brief's own adoption rule, C+ is the winner: **`price_cache_backend` now defaults to
`:cplus`**, with all of C+'s own remaining gates (independent directional check,
cross_delta+backend integration, checkpoint/resume, δ=1/δ=2 trajectories) closed this session
(`c23_cplus_gate.jl`, 8/8). `:kbplus`, `:aplus`, `:pooled`, and `:buffered` remain fully
available and correct as explicit opt-ins.

## 7. Cache policy and workspace lifetime (final)

- `GradWorkspacePool`/`LFixBaseWorkspace`(A+)/`LFixFactorizedWorkspace`(C+)/
  `LFixKBPlusWorkspace`(:kbplus): each built ONCE per `ctx` (driver-function scope), refilled in
  place per gradient call, matching the pattern all four already shared before this session —
  now all four are actually reachable from the driver (not just built).
- `CrossDeltaExactCache`: one per staged continuation, keyed on
  `(x_free, find_smallest, inner_loop_opt, mode, ctx_fingerprint)`, δ deliberately excluded
  (`Delta*(theta)` is budget-independent). Now carries `n_lookups`/`n_hit_verified`/
  `n_hit_infeasible`/`n_miss`/`n_store` counters (new this session), observed at its existing
  dispatch points, no production call site touched.
- `SafeExactCache`: unchanged, one per single-stage run.

## 8. What remains (honest, disclosed gaps — see the gate doc's own §8 for the full list)

- `:kbplus`'s D=4 correctness suite is a reduced (2-point) pattern vs A+/C+'s own full
  136-case exhaustive sweep.
- `cross_delta`'s real hit-rate evidence is from one short staged run.
- The CM interrupted/resume production campaign (D=20/W=80,000/L=50/δ=1) from the original
  productionization brief was never run (pre-existing gap, not attempted this session).
- Independent optimized-value directional checks (both backends) covered a small number of
  coordinates/bandwidths, bounded by real per-point KNITRO cold-solve cost.
- This branch has not been pushed to `origin` or merged back into
  `integration/fullA-final-production-merge` — see `docs/fullA_REPO_MAP_2026-07-22.md` §7 for
  the proposed (not executed) consolidation plan.
