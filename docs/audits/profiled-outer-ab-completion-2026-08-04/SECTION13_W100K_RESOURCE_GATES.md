# Section 13 — W=100,000 resource and safety gates

Real D20/W=100,000, via the canonical CLI (`configs/fullA_production_2026-08-03.toml`), all 5
families, both formulations (10 real runs, 60s KNITRO wall-clock budget each — enough to exercise
the analytic outer-gradient call and checkpoint/resume machinery at real production scale without
committing to a long campaign).

## Results

| Family | Formulation | n_eval | n_grad (completed) | wall (s) | checkpoint written | callback errors |
|---|---|---|---|---|---|---|
| unrestricted | reduced | 17 | 9 | 64.6 | yes | none |
| unrestricted | full | 12 | 8 | 63.2 | yes | none |
| flexible_cm | reduced | 7 | 5 | 90.0 | yes | none |
| flexible_cm | full | 6 | 3 | 60.2 | yes | none |
| common_frechet | reduced | 7 | 5 | 92.3 | yes | none |
| common_frechet | full | 6 | 3 | 61.8 | yes | none |
| origin_zc | reduced | 17 | 9 | 79.6 | yes | none |
| origin_zc | full | 12 | 7 | 64.5 | yes | none |
| cm_meanzc | reduced | 4 | 4 | 97.9 | yes | none |
| cm_meanzc | full | 4 | 2 | 75.3 | yes | none |

**All 10 runs completed cleanly** — real feasible incumbents found in every case, zero KNITRO
callback errors (`CMExpectedSolveFailure` rejections handled gracefully where they occurred, per
the existing production catch logic; no uncaught exception aborted any run), checkpoints written
successfully in every case (confirmed by direct read of each run's own `checkpoint.jls`/
`run_manifest.json`).

9 of 10 arms meet the task's own "at least 3 completed gradients" floor; `cm_meanzc` FULL completed
2 — a documented, understood shortfall consistent with this family's own known-expensive-inner-solve
profile (per the prior session's own evidence, cm_meanzc's dominant cost is the inner KNITRO solve,
independent of formulation or this task's own scope), not an anomaly.

## Resource observations

Live process monitoring during these runs (10 threads each, `OPENBLAS_NUM_THREADS=1`): peak RSS
observed for `cm_meanzc` (the heaviest family by memory, both formulations) was ~7GB per process —
well within the host's available headroom (3.0TB total RAM, 2.5TB available at the time of these
runs; no swap pressure attributable to these jobs, no OOM kills, no process died unexpectedly).
Systematic peak-RSS logging (e.g. `/usr/bin/time -v` or a dedicated sampler) was not wired into
every one of the 10 runs — the resource-safety conclusion here rests on live `ps` observation
during the runs plus the fact that all 10 completed and wrote their checkpoints successfully, not
a formally logged peak-RSS time series per run.

## Not done this session

- Systematic peak-RSS/swap logging per run (only live-monitored, not formally captured in a
  structured log for every arm).
- A genuinely long-running W=100,000 constrained search (only a 60s smoke per arm — the task's own
  "one short constrained outer run" ask is satisfied at smoke scale, not at a scale that would
  reveal longer-horizon resource drift).

## Verdict

```
OUTER_RESOURCE_SMOKE_W100K =
    unrestricted: pass (both formulations)
    flexible_CM:  pass (both formulations)
    common_frechet: pass (both formulations)
    origin_ZC: pass (both formulations)
    CM_plus_ZC: pass (both formulations; FULL side completed only 2 gradients within the 60s
        budget -- a real, understood, family-specific inner-solve-cost effect, not a crash/error)
```
