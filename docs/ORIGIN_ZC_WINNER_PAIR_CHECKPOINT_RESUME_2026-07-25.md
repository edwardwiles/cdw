# Origin-ZC: fresh-process checkpoint/resume through the real public driver

Task §7. Same procedure as the flexible-CM gate, through `originzc_production_stage_runner.jl`
unmodified, `DISTRIBUTION_RESTRICTION=origin_specific_moments_zero_covariance`, `ORIGIN_K_MEAN=1`,
`ORIGIN_K_PAIR=1`.

## Procedure

1. Launched `julia --project=. -t 20 originzc_production_stage_runner.jl <ckpt_dir> 1.0 90
   calibration none 0`.
2. Confirmed a genuine new-point evaluation: `eval 1 t=8.0s gp=0.9877737671465882
   Delta=0.0035842868145595634 feasible=true verified=true`.
3. Confirmed the checkpoint file (`stage_latest.jls`) written after that eval; manifest at launch
   showed `checkpoint_schema=7` (`CMCheckpointV7`, the origin-ZC/destination_sample schema).
4. **Terminated the entire process group** with `pkill -9 -P <pid>; kill -9 <pid>` — confirmed dead.
5. Restarted in a **fresh Julia process**: `resume` mode pointed at the same `stage_latest.jls`.

## Result

```
[stage] RESUMING from .../stage_latest.jls (n_eval=1 n_grad=0 wall_elapsed=7.9s)
  eval 2 t=15.5s gp=0.9877737671465882 Delta=0.0035842868145595634 feasible=true verified=true
  eval 3 t=41.8s gp=0.9738966198497984 Delta=0.48821397021685353 feasible=true verified=true
>>> STAGE result: knitro_status=-401 wall=64.0 n_eval=7 n_grad=5
    best=gp=0.970318272646217 Delta=0.8460550137406035 kappa=0.04897846802396799
>>> checkpoint: .../stage_latest.jls
[core-hessian-counters] winner_pair_hessian_calls=51
[core-hessian-counters]   winner_pair_serial_calls=0
[core-hessian-counters]   winner_pair_parallel_calls=51
[core-hessian-counters] dense_core_fallback_calls=0
[core-hessian-counters] compressed_core_rebuilds=8
>>> STAGE_DONE
```

- **Core winner-pair backend preserved**: `winner_pair_hessian_calls=51`, all parallel,
  **`dense_core_fallback_calls=0`** — captured directly in this resume run (the counter print was
  already live for this gate).
- **Dense cross/restriction backend preserved**: `H_ER`/`H_RR` remain the existing dense Architecture
  A blocks (unchanged by this port, per task §1) — no separate counter needed since that backend
  never had a winner-pair alternative in scope.
- **Complete restriction layout preserved**: resumed under the identical
  `distribution_restriction=:origin_specific_moments_zero_covariance`/K_mean=1/K_pair=1/
  `power_target_layout=:origin_by_power` — the driver's own resume-provenance checks
  (`cm_originzc_checkpoint.jl`) would have errored on any mismatch here.
- **Incumbent/outer point carried forward**: `n_eval` continued from 1 (at kill time) to 7 (6 genuine
  new evals after resume); final `kappa=0.04897846802396799` and `best_Delta=0.8460550137406035`
  are the SAME values independently reached by the matched outer A/B's own winner-pair arm
  (`ORIGIN_ZC_WINNER_PAIR_OUTER_AB_2026-07-25.md`) — an incidental cross-check that the resumed
  search reaches the same real optimum as a from-scratch run.
- **No stale workspace/dimension mismatch**: no errors; `STAGE_DONE` sentinel reached cleanly.

**`ORIGIN_ZC_CHECKPOINT_RESUME = pass`**
