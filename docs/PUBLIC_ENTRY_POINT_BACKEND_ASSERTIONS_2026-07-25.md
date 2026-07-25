# Public entry-point backend assertions — 2026-07-25

Task §3: "Do not validate only helper functions... prove that the public driver actually reaches
the intended implementation." Full detail/commit: `26a1718`. Test files:
`test_backend_manifest_unrestricted.jl`, `test_backend_manifest_cm_originzc.jl`.

## Method

Both files call the actual public checkpointed drivers — not `resolve_*_manifest` in isolation —
at real D=20/W=80,000 scale with a short (5s) `maxtime_real`, capture stdout, and assert on the
literal `[backend-manifest] ...` lines those drivers print via `production_backend_manifest.jl`
(wired in commit `bce2013`, right after each driver's existing `[winner-engine]`/`[screen-stack]`/
`[threshold-config]` banners). Real licensed KNITRO (Artelys Knitro 13.0.1) throughout.

## Results: 11/11 PASS

| Entry point | Assertion | Result |
|---|---|---|
| `run_polish_checkpointed` | `family=unrestricted` | PASS |
| `run_polish_checkpointed` | `hessian_backend=dense_exact` | PASS |
| `run_polish_checkpointed` | `checkpoint_schema=4` | PASS |
| `run_polish_checkpointed` | `core_moment_representation=compressed` | PASS |
| `run_polish_checkpointed` | 5-screen stack (pairwise/hard-winner/envelope/winning-range/safety-net) | PASS |
| `run_profile_checkpointed` | also prints `family=unrestricted` | PASS |
| `run_cm_upper_checkpointed` (`:cm_only`) | `family=flexible_cm`, `hessian_backend=threaded_architecture_c`, `threaded_bins=true`, `checkpoint_schema=6`, `cm_restriction_basis=cumulative`, `core_moment_representation=compressed_winner_form` | PASS (6/6) |
| `run_cm_upper_checkpointed` (`:cm_plus_equal_means`, K_mean=1) | `family=cm_meanzc`, `hessian_backend=threaded_architecture_c`, `K_mean=1` | PASS (3/3) |
| `run_originzc_upper_checkpointed` (K_mean=1) | `family=origin_zc`, `hessian_backend=dense_architecture_a` (**not** `threaded_architecture_c` — confirms §7's "do not automatically replace" is honored), `checkpoint_schema=7` | PASS (3/3) |

Real driver names used throughout, matching the task's "if the live source names differ, use the
real names" instruction:

```text
unrestricted.hessian_backend      = dense_exact
unrestricted.core_moment_representation = compressed
unrestricted.blas_threads         = <ambient, opt-in via blas_threads kwarg>

cm.hessian_backend                = threaded_architecture_c
cm.threaded_bins                  = true
cm.core_moment_representation     = compressed_winner_form
cm.blas_threads                   = <ambient, opt-in>

cm_meanzc.hessian_backend         = threaded_architecture_c  (shares CMBinHessCtx/archC_hess_cb_builder with cm)

origin_zc.hessian_backend         = dense_architecture_a
origin_zc.blas_threads            = <ambient, opt-in via new blas_threads kwarg added this session>
```

An `originzc_upper_checkpointed`/`cm_upper_checkpointed` **lower-bound direction is not wired**
in either family (`find_smallest=true` hardcoded in both) — a pre-existing production limitation
predating this release, out of scope, reported as-is rather than silently "fixed."
