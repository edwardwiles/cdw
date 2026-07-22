# Full-A_od exact solver: current state, 2026-07-22

Canonical as-of-now snapshot. Supersedes prior status claims in
`docs/fullA_independent_audit_remediation.md` (AUD-02 row only — see banner in that file),
`docs/fullA_price_tensor_elimination_report.md` (D=20 benchmark numbers only — see banner in
that file), and `docs/fullA_postmerge_allocation_productionization.md`/
`docs/fullA_factorized_price_production_gate.md` (both still accurate as of this writing, not
superseded, just consolidated here).

Written from `perf/fullA-factorized-price-production` @ `204c266`, the tip of a fully linear
chain (`integration/fullA-final-production-merge` @ `2620097` → `perf/fullA-allocation-cache-cleanup`
@ `1279ed6` → `audit/fullA-postmerge-correctness` @ `dc3196c` → `perf/fullA-factorized-price-production`
@ `204c266`; verified via `git merge-base`/`rev-list --left-right`, 0 commits flow backward at any
step). This branch/commit has NOT been merged back into `integration/fullA-final-production-merge`.

## 1. Active production call path (verified in source, not from a doc)

Entry points: `run_profile_checkpointed` and `run_polish_checkpointed`, both in
`full_aod_diag/d4_exact/c10_d20_production_driver.jl`. Both load the outer KNITRO options via

```julia
KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
```

with `hessopt_tag::String = "sr1"` by default → **`csw_outer_wallclock_sr1.opt`**.

`cb_G!` in both entry points (lines 617-628 and 1025-1033) dispatches identically:

```julia
if use_pooled_gradient
    gfull, meta = composite_gradient_at_fast_pooled(xf, ctx, pe, grad_pool; base, threaded = true, ...)
else
    gfull, meta = composite_gradient_at_fast_buffered(xf, ctx, pe; base, threaded = true, ...)
end
```

`grad_pool = use_pooled_gradient ? build_grad_workspace_pool(W) : nothing` is built once per `ctx`
(driver-function scope, lines 434/803), not per callback — confirmed persistent when the pooled
path is active.

`cb_F!` in both entry points calls `screened_eval(...; exact_cache = exact_cache, ...)`, where
`exact_cache = exact_cache_override !== nothing ? exact_cache_override : (use_exact_cache ? SafeExactCache() : nothing)`
(line 480). `exact_cache_override` accepts either a `SafeExactCache` or a `CrossDeltaExactCache`;
only `run_staged_delta5_continuation` (in `staged_delta5.jl`) threads a `CrossDeltaExactCache`
through multiple stages via this parameter, opt-in via its own `cross_delta::Bool` kwarg.

## 2. Current defaults (verified directly in source)

| Setting | Default | Verified at |
|---|---|---|
| `use_pooled_gradient` | **`false`** | `c10_d20_production_driver.jl:387,742` |
| `cross_delta` (staged continuation) | **`false`** (unused unless passed) | `c10_d20_production_driver.jl:86` comment + `staged_delta5.jl` |
| `price_cache_backend` selector | **does not exist** — no such kwarg anywhere in `c10_d20_production_driver.jl`, `gradient_workspace.jl`, `lfix_base_workspace_pooled.jl`, or `lfix_factorized_workspace.jl` (`grep -rln price_cache_backend` returns nothing) | — |
| Outer `par_concurrent_evals` | **`yes`**, in every actively-loaded outer `.opt` file (`csw_outer_wallclock_{sr1,lbfgs,productfd}.opt` and all other `csw_outer*.opt`) | `grep -rn par_concurrent_evals *.opt` |
| Application-level thread guard | Present — `cc_algo/PsiObjectiveBundle.jl`'s `_enter_callback!`/`_exit_callback!`, tracks owning thread ID per bundle instance, allows same-thread nesting, rejects cross-thread reentry | added in `62dde8f`, untouched by the `dc3196c` revert |

## 3. Source / merged / active / default / tested matrix

| Component | Source | Merged into tip | Called by driver | Default-on | Fn-level tested | Real outer-solve tested |
|---|---|---|---|---|---|---|
| `GradWorkspacePool` / `composite_gradient_at_fast_pooled` | `gradient_workspace.jl` | Yes | Yes, opt-in | No | `test_gradient_workspace.jl` 9/9 | Yes — post-hang-fix, `use_pooled_gradient=false/true` reach identical `kappa=0.0203135174923732` |
| `LFixBaseWorkspace` / Backend A+ | `lfix_base_workspace.jl`, `lfix_base_workspace_pooled.jl` | Yes | **No** | No | `test_lfix_base_workspace_pooled.jl` 4/4, `test_production_gate_exhaustive_d4.jl` 28/28 (A+ rows) | Only via standalone `c18_short_trajectory_comparison.jl`, not the real driver |
| Factorized C+ | `lfix_factorized.jl`, `lfix_factorized_workspace.jl` | Yes | **No** | No | `test_lfix_factorized_workspace.jl` 20/20, `test_production_gate_exhaustive_d4.jl` 28/28 (C+ rows) | Same standalone-only caveat; §12 (independent directional checks) and §14 (cache/concurrency/checkpoint with C+ active) explicitly not run |
| `CrossDeltaExactCache` | `cross_delta_cache.jl` | Yes | Yes, opt-in | No | `test_cross_delta_cache.jl` 36/36 | **No** — never run through a real staged δ-continuation since the hang fix |
| Exact-point cache / verified-success gate (AUD-04) | `oracle.jl` (`VerifiedSuccessTolerances`, `InnerResultClass`, `is_cacheable_result`, `is_verified_success`) | Yes | Yes | Yes (correctness gate) | Yes | Yes |
| Checkpoint/resume verified-success gates (AUD-09/10) | `c10_d20_production_driver.jl` (`ResumeTolerances`, `check_resume_tolerances!`, `:stage_complete_unverified`) | Yes | Yes | Yes | Yes | Partially — the brief's own D=20/W=80,000/L=50/δ=1 CM interrupted/resume campaign has never been run |

## 4. AUD-02 supersession (critical — do not undo)

`docs/fullA_independent_audit_remediation.md`'s AUD-02 row describes the *original* fix: outer
`par_concurrent_evals: yes→no` plus the thread-aware `PsiObjectiveBundle` guard. **The `.opt`
half of that fix was itself a regression**, discovered and reverted at `dc3196c`
(`docs/fullA_nested_knitro_solve_hang_rootcause.md`, full root-cause writeup, git-bisected A/B
verified bit-identical against the pre-regression commit `3855430`). Current, correct state:

- Outer `par_concurrent_evals`: **`yes`** (this codebase's core architecture relies on the outer
  callback synchronously nesting a second `KN_solve` for the inner dual problem — a same-thread,
  same-call-stack pattern that KNITRO's own `par_concurrent_evals=no` enforcement deadlocks on,
  via a non-reentrant OpenMP critical section).
- The `PsiObjectiveBundle.jl` thread-aware guard (added in the same original commit `62dde8f`,
  untouched by the revert) remains the actual defense against genuine cross-thread reentry, and
  correctly allows the legitimate same-thread nesting.

**Do not set outer `par_concurrent_evals` to `no` again without re-reading
`docs/fullA_nested_knitro_solve_hang_rootcause.md` in full.**

## 5. Cache policy and workspace lifetime (as currently wired)

- `GradWorkspacePool`: one per `ctx`, built at driver-function scope, reused across every
  gradient callback for that context's lifetime. Opt-in.
- `LFixBaseWorkspace` (A+) / `LFixFactorizedWorkspace` (C+): both designed for one-per-context,
  refill-in-place lifecycles (mirrors `GradWorkspacePool`'s pattern), validated at the function
  level, but **neither is instantiated anywhere in the driver** — both still call the allocating
  `build_lfix_base_cache`/equivalent inside `composite_gradient_at_fast_buffered`/`_pooled`.
- `CrossDeltaExactCache`: one per staged continuation (`run_staged_delta5_continuation`), keyed
  on `(x_free, find_smallest, inner_loop_opt, mode, ctx_fingerprint)` — deliberately excludes
  `δ`/`find_smallest`'s budget direction because `Delta*(theta)` is budget-independent (only the
  *feasibility check against* a budget is; the cache stores the raw solved `Delta*`, not a
  feasibility verdict). Only `VerifiedSolved`/`ExactInfeasible` results are cacheable (AUD-04).
- `SafeExactCache`: one per single-stage run (`run_profile_checkpointed`/`run_polish_checkpointed`),
  same cacheability gate.

## 6. What Phase 2/3/4 of the current finalization task must still close

1. Re-verify `use_pooled_gradient`'s driver-level evidence is still current on this exact commit
   (last confirmed post-hang-fix, not re-run since).
2. Run a real staged δ=2→3→4→5 continuation with `cross_delta=true` vs `false`, eval-matched and
   wall-clock-matched, with hit/miss counters — never done.
3. Wire a `price_cache_backend::Symbol` selector (`:buffered`/`:pooled`/`:aplus`/`:cplus`) into
   `cb_G!` in both entry points, replacing/extending the current `use_pooled_gradient::Bool`
   dispatch without creating two contradictory ways to pick a backend.
4. Close C+'s two explicitly-disclosed gates: independent optimized-value directional checks,
   and cache/concurrency/checkpoint tests with C+ actually wired into the driver.
5. Evaluate the reference-aligned ratio factorization (`constCons/UPow`, no `exp((1-σ)·...)`) as
   a `:kbplus` candidate alongside C+.
