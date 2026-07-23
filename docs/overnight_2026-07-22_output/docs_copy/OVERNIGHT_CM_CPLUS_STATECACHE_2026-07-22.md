# Overnight CM-C+ / complete-state cache — baseline record — 2026-07-22

## Base commit

- Exact commit: `22683016b5a927d4952e52049145ad8d1f5a2b87`
- = local `production/fullA-exact` (`gravity-production-fullA-exact` worktree, clean, `git status --short` empty)
- = `cdw/production/fullA-exact` (fetched fresh this session; `git merge-base --is-ancestor` confirms
  equality both directions)
- Tag at this commit: `cm-production-ready-2026-07-22-r3`
- This is the exact commit the live CM smoke-test supervisor logged as its own
  (`production_runs/cm_smoketest_2026-07-22/part1_clean/supervisor.log`:
  `commit=22683016b5a927d4952e52049145ad8d1f5a2b87`).

## Live campaign observed at session start (2026-07-22 ~20:34)

- Process tree rooted at bash pid 170719 (chain 91), running
  `scripts/cm_production_supervisor.sh 91 production_runs/cm_smoketest_2026-07-22/part1_clean` from
  worktree `gravity-production-fullA-exact`, `JULIA_NUM_THREADS=20`, `taskset -c 0-19`. Julia worker
  pid 170763 running `full_aod_diag/d4_exact/cm_production_stage_runner.jl ... delta_1.0 1.0 3600
  calibration 91` (a smoke-test / calibration stage, delta=1.0, 3600s budget), actively consuming
  ~126% CPU, NOT idle.
- This is a **smoke test** (`cm_smoketest_2026-07-22`), not the full 3-chain campaign described in
  `docs/CM_PRODUCTION_LAUNCHER_2026-07-22.md` (`production_runs/cm_campaign_2026-07-22/chain{1,2,3}`)
  -- no `cm_campaign_2026-07-22` directory exists yet on this host. Treated with full production-priority
  caution regardless, per the task brief's explicit instruction not to interfere with "the running CM
  campaign."
- Server is a large (208-core, 3TB RAM) shared host with many other users' unrelated jobs consuming
  most of the `load average: 95` figure (Levi/pep_arch/gnome-shell processes, not ours). The CM
  smoke-test process itself is pinned to cores 0-19 via `taskset` and uses ~1.3 cores of active CPU at
  observation time (not obviously thread-saturated at the `calibration` stage's current phase).
- A second, unrelated Julia process (`c40_section9_d20_outer_trial.jl`, pid 118961, `-t 20`, ~1hr
  elapsed) is also running under this same repo tree, from what appears to be a different concurrent
  session — not touched, not identified as part of the CM campaign brief.
- Several `screen` sessions (`task1_upper`, `task1_lower`, `task2_upper`, `task3_upper`,
  `task5_cm_upper`, `fullA_lower_v2`, `fullA_upper_v2`, `seq_upper_delta20_fix`) are multi-day-old
  detached sessions from earlier head-to-head work; not touched.

## Experimental branch/worktree

- Branch: `perf/fullA-cm-cplus-statecache-overnight-2026-07-22`
- Worktree: `/bbkinghome/edav/gravity_robustness/gravity-perf-fullA-cm-cplus-statecache-overnight`
- Created via `git worktree add -b <branch> <path> 22683016b5a927d4952e52049145ad8d1f5a2b87` from the
  shared `.git` (common dir under `trade_robustness_modular`). Exactly one branch, exactly one
  worktree, per the brief.

## Resource policy for this session

- Never touch `gravity-production-fullA-exact` (working directory or its process tree).
- Never `taskset -c 0-19` for any of this session's own Julia runs (reserved for the live campaign);
  use higher core ranges (e.g. `taskset -c 20-39`) for any D=4/D=20 work launched from this branch.
- D=4 correctness work only, until the live smoke test/campaign is confirmed finished or resources are
  independently confirmed abundant; D=20 gates deferred/resource-gated per the brief's own priority
  order.
