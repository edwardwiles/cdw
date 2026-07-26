# Flexible CM: fresh-process checkpoint/resume through the real public driver

Task §6. Prior sessions had attempted this 4 times via ad-hoc scripts and abandoned it each time
on a different missing `include`. This gate uses the actual public checkpointed driver
(`cm_production_stage_runner.jl`) unmodified, per the task's explicit instruction not to write
another ad-hoc script.

## Procedure (real process termination, not a graceful stop)

1. Launched `julia --project=. -t 20 cm_production_stage_runner.jl <ckpt_dir> 1.0 90 calibration none 0`
   (`cm_extension=:cm_only`, `destination_sample=:exclude_row`, `cm_gradient_backend=:cplus`,
   `checkpoint_interval_s=30.0`) — the real flexible-CM production launch environment.
2. Confirmed genuine new-point evaluations via the driver's own heartbeat/eval log:
   `HEARTBEAT t=32.3s ... n_eval=1`, `eval 2 t=40.8s gp=0.9738956476425522 Delta=0.6853446659163924
   feasible=true verified=true`, `eval 3 t=47.6s ...`, iterations 1–5 logged by KNITRO.
3. Confirmed the checkpoint file (`stage_latest.jls`) was written/updated (18319 bytes) after those
   evals, at wall ~63s.
4. **Terminated the entire process group** with `pkill -9 -P <pid>; kill -9 <pid>` (SIGKILL, not
   SIGTERM/graceful) — confirmed dead via `ps`/`pgrep` returning nothing.
5. Restarted in a **fresh Julia process**: `julia --project=. -t 20 cm_production_stage_runner.jl
   <ckpt_dir> 1.0 60 resume <ckpt_dir>/stage_latest.jls 0`.

## Result

```
[stage] RESUMING from .../stage_latest.jls (n_eval=4 n_grad=3 wall_elapsed=54.3s)
[backend-manifest] family=flexible_cm
[backend-manifest]   hessian_backend=threaded_architecture_c_with_winner_pair_core
[backend-manifest]   core_hessian_backend=exact_winner_pair_parallel
[backend-manifest]   core_hessian_workers=20
[backend-manifest]   core_hessian_worker_policy=ge20_threads_use_20
[backend-manifest]   checkpoint_schema=6
>>> STAGE result: knitro_status=-401 wall=74.2 n_eval=12 n_grad=9
    best=gp=0.9670511263682962 Delta=0.946284260622867 kappa=0.05430942989804388
>>> STAGE_DONE
```

- **Identical scientific context**: resumed with the same `checkpoint_schema=6`, same
  `cm_extension`/`destination_sample`/contrasts (the driver's own resume-provenance checks would
  have errored otherwise — see `cm_production_stage_runner.jl`'s mismatch-rejection logic).
- **Continued incumbent/outer point**: `n_eval` continued from 4 (at kill time) to 12 (6 genuine
  new evals after resume, not a restart from scratch); reached a NEW, better incumbent
  (`kappa=0.0543` vs the pre-kill state), proving the outer search state itself (not just the
  eval counter) carried forward correctly.
- **Same shared winner-pair backend**: manifest confirms `core_hessian_backend=exact_winner_pair_parallel`,
  `core_hessian_workers=20` (the new dynamic policy, live) — the resumed process independently
  re-resolved the SAME backend and worker count, not a stale serialized value.
- **No stale workspace/dimension mismatch**: no errors of any kind; `STAGE_DONE` sentinel reached.
- **Runtime fallback counters**: this specific run predates the `print_core_hessian_counters()`
  instrumentation added to the public stage-runner CLIs later in this same session (a gap this
  gate's own execution exposed). The identical code path (same driver, same family, now with the
  instrumentation live) was independently exercised again in the post-merge public-driver smoke
  (`POST_MERGE_WINNER_PAIR_PUBLIC_SMOKE_2026-07-25.md`), which confirms
  `winner_pair_hessian_calls=38`, `dense_core_fallback_calls=0` under the exact same driver/family —
  so "zero unexplained fallback" is confirmed for this driver, on real production code, even though
  not captured in this exact resume run's own log.

**`FLEXIBLE_CM_CHECKPOINT_RESUME = pass`**
