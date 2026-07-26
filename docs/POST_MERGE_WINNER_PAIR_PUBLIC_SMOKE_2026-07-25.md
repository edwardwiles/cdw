# Post-merge public-driver smokes, all four families (task §11 step 5–7)

Run immediately after fast-forwarding `production/fullA-exact` to `b40e0f4` and pushing (verified:
the worktree used for these smokes is checked out exactly at `b40e0f4`, the same commit now at
`origin/production/fullA-exact`). Each smoke: real public stage-runner CLI, calibration mode,
45s budget, D=20 real data, W=80,000, seed 20260719, `:exclude_row`, `JULIA_NUM_THREADS=20`,
`OPENBLAS_NUM_THREADS=1`, one process per family (all four run concurrently — independent OS
processes, no shared KNITRO/thread state, per this codebase's own documented concurrency
discipline).

## Unrestricted (`unrestricted_stage_runner.jl`)

```
[backend-manifest] family=unrestricted
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   core_hessian_workers=20
[backend-manifest]   core_hessian_worker_policy=ge20_threads_use_20
>>> STAGE result: knitro_status=-401 wall=46.0 n_eval=20 best=Delta_dual=0.0007287992409600393 n_eval=20
[core-hessian-counters] winner_pair_hessian_calls=122
[core-hessian-counters]   winner_pair_parallel_calls=122
[core-hessian-counters] dense_core_fallback_calls=0
>>> STAGE_DONE
```

## Flexible CM (`cm_production_stage_runner.jl`, `cm_extension=:cm_only`)

```
[backend-manifest]   hessian_backend=threaded_architecture_c_with_winner_pair_core
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   core_hessian_workers=20
>>> STAGE result: knitro_status=-401 wall=48.9 n_eval=3 n_grad=3 best=gp=0.972624964048607 Delta=0.509493354697917 kappa=0.04520745265885873
[core-hessian-counters] winner_pair_hessian_calls=38
[core-hessian-counters] dense_core_fallback_calls=0
>>> STAGE_DONE
```

## CM+mean/ZC (`cm_production_stage_runner.jl`, `cm_extension=:cm_plus_equal_means_zero_covariance`)

```
[backend-manifest]   hessian_backend=threaded_architecture_c_with_winner_pair_core
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   core_hessian_workers=20
>>> STAGE result: knitro_status=-401 wall=79.0 n_eval=3 n_grad=2 best=gp=0.9808352140139313 Delta=0.22586074862606484 kappa=0.031736823212207166
[core-hessian-counters] winner_pair_hessian_calls=38
[core-hessian-counters] dense_core_fallback_calls=0
>>> STAGE_DONE
```

## Origin-ZC (`originzc_production_stage_runner.jl`, K_mean=K_pair=1)

```
[backend-manifest]   hessian_backend=partitioned_winner_pair_core_dense_restriction
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   core_hessian_workers=20
>>> STAGE result: knitro_status=-401 wall=46.3 n_eval=4 n_grad=4 best=gp=0.9728314490157915 Delta=0.5426586729538228 kappa=0.04486959674280533
[core-hessian-counters] winner_pair_hessian_calls=42
[core-hessian-counters] dense_core_fallback_calls=0
>>> STAGE_DONE
```

## Verification

`dense_core_fallback_calls = 0` in all four ordinary post-merge smokes — the required condition
for an ordinary run. All four report `core_hessian_backend=exact_winner_pair_parallel`,
`core_hessian_workers=20`, `core_hessian_worker_policy=ge20_threads_use_20` — the shared winner-pair
backend and the new dynamic worker policy are both live, by default, through the real production
CLIs, on the exact commit now at `origin/production/fullA-exact`. All four reached `STAGE_DONE`
with no errors.

**`POST_MERGE_SMOKE = pass`**
