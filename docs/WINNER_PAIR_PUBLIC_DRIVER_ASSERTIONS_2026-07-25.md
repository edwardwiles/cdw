# Public driver assertions — 2026-07-25

Task §4. Real public checkpointed drivers (`run_polish_checkpointed`, `run_profile_checkpointed`,
`run_cm_upper_checkpointed` ×2 for `:cm_only`/`:cm_plus_equal_means`, `run_originzc_upper_checkpointed`),
short real budgets, asserting on the literal `[backend-manifest] ...` stdout each driver actually
prints — `test_backend_manifest_unrestricted.jl` + `test_backend_manifest_cm_originzc.jl`, both
extended this session. **Both fully PASS after real bugs were found and fixed.**

## Real bugs found and fixed (not merely "assertions written")

1. **Unrestricted** (`c10_d20_production_driver.jl`, both `run_polish_checkpointed` and
   `run_profile_checkpointed` manifest-print call sites): hardcoded
   `resolve_unrestricted_manifest(; hessian_backend = :dense_exact, ...)` — the driver's OWN
   startup banner claimed `dense_exact` regardless of what `UNRESTRICTED_CORE_HESSIAN_BACKEND[]`
   actually resolved to. Fixed: removed the hardcoded kwarg so it reads the live Ref default.
2. **Origin-ZC** (`cm_originzc_checkpoint.jl`): `resolve_origin_zc_manifest(; blas_threads=...)` was
   called with NO `octx` at all, so it always fell back to reporting the pre-port monolithic dense
   Architecture A. Fixed: passes the real `pcx.octx` the driver just built.
3. **Design flaw, all three restricted-family manifests** (`production_backend_manifest.jl`): the
   original (prior-session) `resolve_flexible_cm_manifest`/`resolve_origin_zc_manifest` gated the
   reported `core_hessian_backend` on whether a `CompressedFactual` had ALREADY been built
   (`core_cf_ref[] isa CompressedFactual`) — but the manifest prints at driver STARTUP, before any
   inner solve has run, so that condition is structurally always false at print time, regardless of
   what backend will actually be used from the first real solve onward. Fixed: report the
   CONFIGURED backend directly (what will run), leaving "did it actually run" to the separate
   runtime counters (task §2, checked AFTER a solve).

Flexible CM's driver (`cm_checkpoint.jl`) was ALREADY correctly wired (passes `cctx = pcx.cctx`
live) — no fix needed there.

## Assertions (after fixes), all PASS

**Unrestricted** (`run_polish_checkpointed`, `run_profile_checkpointed`):
```
[backend-manifest] family=unrestricted
[backend-manifest]   hessian_backend=exact_winner_pair_parallel
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   core_hessian_workers=10
[backend-manifest]   core_hessian_storage=full_stride
[backend-manifest]   core_moment_representation=compressed_winner_form
[backend-manifest]   checkpoint_schema=4
```
`run_profile_checkpointed` confirmed to also print `family=unrestricted` (same manifest resolver,
same driver-level call site pattern).

**Flexible CM** (`run_cm_upper_checkpointed`, `cm_extension=:cm_only`):
```
[backend-manifest] family=flexible_cm
[backend-manifest]   hessian_backend=threaded_architecture_c_with_winner_pair_core
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   core_hessian_workers=10
[backend-manifest]   threaded_bins=true
[backend-manifest]   core_moment_representation=compressed_winner_form
[backend-manifest]   cm_restriction_basis=cumulative
[backend-manifest]   checkpoint_schema=6
```

**CM+mean/ZC** (`run_cm_upper_checkpointed`, `cm_extension=:cm_plus_equal_means`, K_mean=1):
```
[backend-manifest] family=cm_meanzc
[backend-manifest]   hessian_backend=threaded_architecture_c_with_winner_pair_core
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   K_mean=1
```

**Origin-ZC** (`run_originzc_upper_checkpointed`, K_mean=1/K_pair=1):
```
[backend-manifest] family=origin_zc
[backend-manifest]   hessian_backend=partitioned_winner_pair_core_dense_restriction
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   cross_hessian_backend=dense_exact
[backend-manifest]   checkpoint_schema=7
```

All four families' `core_hessian_backend` literally reports `exact_winner_pair_parallel` through
the real public driver, matching task §4's explicit expected-output example exactly.

## What was NOT separately re-verified this session

`run_profile_checkpointed`'s manifest text was confirmed to print `family=unrestricted` but the
FULL set of `core_hessian_backend=...` assertions was only run against `run_polish_checkpointed`'s
output (both call the same underlying `resolve_unrestricted_manifest(; blas_threads=...)` with no
family-specific branching, so this is a very low-risk gap, not a genuine untested code path — but
disclosed rather than silently assumed). Flexible CM's "lower-bound smoke if separately exposed"
(task §4's optional bullet) was not run — no separate lower-bound entry point beyond
`run_polish_checkpointed`'s own `find_smallest` toggle (already covered) was identified as CM-specific.
