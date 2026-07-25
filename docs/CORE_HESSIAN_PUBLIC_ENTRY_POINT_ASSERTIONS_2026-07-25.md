# Public entry-point assertions — 2026-07-25

Task §5's "Add public-entry-point tests that invoke every family and assert the literal resolved
backend."

## What exists and was exercised this session

`test_shared_core_hessian_d4_gates.jl` (task §7's correctness gates) invokes every family through
functions one layer below the checkpointed public drivers but sharing their exact inner-solve
machinery:
- Unrestricted: `inner_loop_KNITRO_compressed` (the same function `evaluate_fullA_fast_compressed`
  → `_callbackEvalH_inner_compressed!` calls from `run_polish_checkpointed`/`run_profile_checkpointed`).
- Flexible CM: `archC_base_state` (the same function `run_cm_upper_checkpointed`'s outer loop calls).
- CM+mean/ZC: `archC_meanzc_base_state`.
- Origin-ZC: `archOZ_base_state`.

Each of these runs a REAL KNITRO inner dual solve, through the REAL Hessian callback (not a
hand-invoked standalone kernel call) — this is a genuine "does the actual production wiring
resolve to the shared backend and produce a correct answer" check, not merely a unit test of the
kernel in isolation. All 40 assertions across these four call paths PASS (see
`docs/full_correctness_log_2026-07-25.txt`).

## What was found but NOT independently re-verified this session

Pre-existing tests `test_backend_manifest_unrestricted.jl`/`test_backend_manifest_cm_originzc.jl`
(added under the prior "allocation/Hessian port task §3") already call the REAL public checkpointed
drivers (`run_polish_checkpointed`, `run_cm_upper_checkpointed` ×2, `run_originzc_upper_
checkpointed`) and assert on their own `"[backend-manifest] ..."` stdout lines. This session:
- Extended `resolve_unrestricted_manifest`/`resolve_flexible_cm_manifest`/`resolve_origin_zc_
  manifest` (`production_backend_manifest.jl`) with the new granular fields (`core_hessian_backend`,
  `core_hessian_workers`, `core_hessian_storage`, `cross_hessian_backend`,
  `restriction_hessian_backend`, `full_hessian_assembly`, `knitro_hessian_format`), reading them
  live off each family's own state (`UNRESTRICTED_CORE_HESSIAN_BACKEND[]`/`cctx.core_hessian_
  backend`/`octx.core_hessian_backend`) rather than hardcoding a value.
- Did **NOT** run `test_backend_manifest_unrestricted.jl`/`test_backend_manifest_cm_originzc.jl`
  themselves this session (each needs a `JULIA_NUM_THREADS=20` real KNITRO run through the full
  checkpointed-driver stack, a real-time cost this session's budget did not extend to alongside
  everything else completed).
- Did **NOT** independently confirm whether `resolve_*_manifest`/`print_production_backend_
  manifest` are actually CALLED from inside `run_polish_checkpointed`/`run_cm_upper_checkpointed`/
  `run_originzc_upper_checkpointed` themselves (an open question already flagged by this session's
  own initial architecture-mapping research, prior to any of this port's code changes — i.e. a
  pre-existing wiring-completeness question, not something this port broke or fixed).

## Follow-up required before a `merged_all_families` verdict

Run `test_backend_manifest_unrestricted.jl` and `test_backend_manifest_cm_originzc.jl` (or their
task-§5 update, if the manifest resolvers turn out not to be called from the real drivers yet) and
confirm each family's printed manifest literally reports `core_hessian_backend =
exact_winner_pair_parallel` (or the appropriate family-specific value from
`CORE_HESSIAN_FAMILY_COVERAGE_TABLE_2026-07-25.md`'s expected-manifest examples) through the real
checkpointed driver, not just through the lower-level functions this session's own gates exercised.
