# Allocating `build_compressed_factual(` call-site audit — 2026-07-27

Every call site of the always-allocating `build_compressed_factual(θ_full, ctx; ...)` (i.e. NOT
`build_compressed_factual!`) found by `grep -rn "build_compressed_factual(" full_aod_diag/d4_exact`
at the start of this session, classified per the addendum's taxonomy. "Included by the production
driver?" was checked directly: `grep -rl 'include(joinpath(@__DIR__, "<file>"))'
c10_d20_production_driver.jl c10_d20_production_driver_unified.jl` for every candidate file —
`NONE` for every bench/diagnostic/test file listed below (confirmed live this session, not assumed).

## PRODUCTION_HOT_PATH — found and FIXED this session

| File:line (before fix) | Family | Status |
|---|---|---|
| `compressed_live.jl:319` (`inner_loop_internal_compressed`) | Unrestricted | **FIXED** — now calls `build_economic_moment_state!`, dispatches to the in-place builder whenever `ctx.cf_workspace` is attached (always true in a real driver run). This is THE per-inner-solve moment build under `moment_representation=:compressed`, the production driver's default mode — genuinely called once per KNITRO inner dual solve, for the lifetime of every unrestricted-family production run. |
| `c10_d20_production_driver.jl:406` (dual-bank warm-start scoring build) | Unrestricted (driver-level) | Was already workspace-aware via a hand-inlined ternary (2026-07-25 §3.1) — **simplified** to call `build_economic_moment_state!` directly (behavior unchanged, removes a duplicate re-implementation of the same dispatch rule). |

After this fix: **0** call sites of the allocating `build_compressed_factual` remain on any path
reachable from `c10_d20_production_driver.jl`/`c10_d20_production_driver_unified.jl`'s own include
chain during a live campaign. Verified directly via the runtime counter
`ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS`, which stayed at 0 across repeated `evaluate_fullA_
fast(...; moment_representation=:compressed)` calls at D=4 and real D=20/W=80,000 with a workspace
attached (see `FIVE_FAMILY_INPLACE_COMPRESSED_FACTUAL_GATE_2026-07-27.md`).

## PRODUCTION_HOT_PATH-adjacent — updated for consistency (were EXPLICIT_REFERENCE, not reachable)

| File:line | Classification | Reason |
|---|---|---|
| `compressed_live_v2.jl:98` (`inner_loop_internal_compressed_v2`) | EXPLICIT_REFERENCE (benchmark) | Module header states explicitly: "Not wired into evaluate_fullA_fast's moment_representation dispatch". Confirmed: `oracle_fast.jl`'s `moment_representation` dispatch only recognizes `:dense`/`:compressed`, never anything routing to the `_v2` kernels. Only caller: `c9_phase4_v2_e2e_d20_bench.jl` (a standalone benchmark script, not included by either driver). Updated to `build_economic_moment_state!` anyway for consistency; zero behavior change for its own ctx (no workspace attached there). |
| `compressed_inner_alt_solvers.jl:204` (`inner_loop_internal_compressed_variant`) | EXPLICIT_REFERENCE (benchmark) | Phase 3C alt-solver comparison harness (`:qn`/`:denseaccum`/`:hvp`), never wired into `moment_representation` dispatch either. Only callers: `c9_phase3c_d20_bench.jl`, `c9_phase3c_correctness_d4.jl` (standalone scripts). Updated to `build_economic_moment_state!` anyway. |

## SAFETY_FALLBACK — intentional, inside the shared builder itself

| File:line | Classification |
|---|---|
| `compressed_factual_buffer_reuse.jl` (`build_economic_moment_state!`'s own `else` branch, `cf_build`'s inherited fallback) | Not a separate call site — this IS the shared builder's documented, addendum-§5-sanctioned exception ("If the allocating wrapper internally creates a workspace and calls the mutating function, that is acceptable only outside repeated production execution" — here read in reverse: the shared in-place function itself falls back to the allocating one only when no workspace was ever attached, i.e. never in a real production driver run, per the runtime counter evidence). |

## PRODUCTION_CONTEXT_INITIALISATION

None found — `build_compressed_factual_workspace`/`attach_compressed_factual_workspace` (the
context-initialisation-time allocation) call `Matrix{Int}(undef,...)`/`Matrix{Float64}(undef,...)`
directly, never the allocating `build_compressed_factual` function itself.

## EXPLICIT_REFERENCE / TEST_ONLY — unchanged, all confirmed NOT included by either driver

Standalone benchmark scripts (measure the allocating builder's own cost deliberately, as a
baseline for the workspace-reusing alternative, or benchmark an unrelated code path that happens
to build a `CompressedFactual` once for setup):
`bench_partA_economic_fg_allocation.jl`, `benchmark_compressed_live.jl`, `bench_winner_
certificate.jl`, `bench_unrestricted_20thread_worker_selection.jl`, `bench_shared_core_hessian_
unrestricted_d20.jl`, `benchmark_compressed.jl`, `c9_phase4_kernels_v2_d20_bench.jl`,
`c10_structured_moment_bench_d20.jl`.

Standalone diagnostic/verification/audit scripts (one-off, run manually, not part of any repeated
campaign): `c10_finalize_equivalence_probe.jl`, `c10_structured_moment_verify.jl`,
`negcache_audit_experiment.jl`.

Test files (`test_*.jl`): `test_compressed_live_integration.jl`, `test_compressed_cc_kernels_v2.jl`,
`test_dual_bank.jl`, `test_compressed_factual_buffer_reuse.jl` (deliberately compares the
allocating reference against the in-place builder — the allocating call IS the test's reference
fixture), `test_exclude_row_unrestricted_gateB_screens.jl`, `test_compressed_cc_inner.jl`,
`test_compressed_moments.jl`, `test_winner_certificate.jl`,
`test_exclude_row_unrestricted_gateA_layout.jl`, `test_shared_core_hessian_d4_gates.jl`.

Plus this session's own new gate/bench scripts (deliberately call the allocating reference for
comparison): `test_shared_economic_moment_state_builder_2026-07-27.jl`,
`test_shared_economic_moment_state_builder_d20_2026-07-27.jl`,
`bench_shared_economic_moment_state_builder_2026-07-27.jl`.

## DEAD_CODE

None found — every call site above is reachable from at least one runnable script.

## Verdict

```
PRODUCTION_HOT_PATH calls (reachable from either production driver's include chain) = 0
```

confirmed both by static classification (this audit) and dynamically by the
`ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS` runtime counter across D=4 and real D=20/W=80,000 runs
with a workspace attached (see gate doc).
