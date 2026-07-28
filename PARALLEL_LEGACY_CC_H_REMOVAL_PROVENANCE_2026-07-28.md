# Parallel legacy CC-H removal — provenance freeze — 2026-07-28

## Purpose

This document freezes the exact starting state for the "delete legacy CC H/G/K storage from all
production operator families" cleanup task, run **in parallel** with the active
`campaign/five-family-bounds-2026-07-28` production-bounds campaign. It exists to make campaign
isolation auditable after the fact.

## Repository / worktree topology

The whole project (`cdw`, `github.com:edwardwiles/cdw.git`) is one Git repository with its
git-common-dir rooted at `/bbkinghome/edav/gravity_robustness/trade_robustness_modular/.git`, used
through ~40 parallel worktrees under `/bbkinghome/edav/gravity_robustness/` and
`/bbkinghome/edav/gravity_robustness/worktrees/`. `git worktree list` from any worktree enumerates
all of them.

## Campaign identification (as of 2026-07-28 07:28 EDT)

- **Campaign worktree**: `/bbkinghome/edav/gravity_robustness/worktrees/campaign-five-family-bounds-2026-07-28`
  (`git worktree list` shows it `locked`).
- **Campaign branch**: `campaign/five-family-bounds-2026-07-28`
- **Campaign HEAD SHA (base SHA for this cleanup)**: `93f26df7dfba96dbc0be92223b5d829c357b8a61`
  — "Session handoff doc + delta=1 upper/lower production smoke-test scripts (not yet run)"
- **Live-process check**: no `julia`/`knitro` process, and no process of any kind, had that
  worktree directory as its `cwd` at the time this cleanup started (checked via `/proc/*/cwd` over
  all `edav`-owned PIDs, and via cmdline substring match). The campaign's own handoff doc
  (`docs/SESSION_HANDOFF_2026-07-28.md` at that commit) confirms its 5 delta=1 smoke scripts were
  written but explicitly **not yet run** — consistent with no live process found. The campaign may
  resume in that worktree at any time; this cleanup does not assume it will stay idle.
- `git status --short` in the campaign worktree was clean (no uncommitted changes) at inspection
  time.

## Isolation actions taken

1. Base SHA `93f26df7dfba96dbc0be92223b5d829c357b8a61` recorded above, independently verified via
   `git log -1 --format="%H %s" 93f26df7...` before branching.
2. New worktree created via explicit `git worktree add`, **not** the Agent-tool `isolation:"worktree"`
   parameter (this repo's established practice — see project memory
   `feedback-agent-tool-worktree-isolation-unreliable`, which found that mechanism unreliable in a
   multi-worktree repo like this one):

   ```
   git worktree add -b cleanup/remove-legacy-CC-H-G-storage-2026-07-28 \
     /bbkinghome/edav/gravity_robustness/worktrees/cleanup-remove-legacy-CC-H-G-storage-2026-07-28 \
     93f26df7dfba96dbc0be92223b5d829c357b8a61
   ```

3. **New cleanup worktree**: `/bbkinghome/edav/gravity_robustness/worktrees/cleanup-remove-legacy-CC-H-G-storage-2026-07-28`
   **New cleanup branch**: `cleanup/remove-legacy-CC-H-G-storage-2026-07-28`
4. Verified via `/proc/*/cwd` scan that no process (campaign or otherwise) has the new cleanup
   worktree as its `cwd` — only this session's own shell subprocesses.
5. All logs, temporary run artifacts, and result directories for this cleanup task will be written
   inside the cleanup worktree (or under this session's scratchpad,
   `/tmp/claude-181517/-bbkinghome-edav-gravity-robustness/2b1854d3-6f36-410a-9dfc-96461574a8ba/scratchpad`),
   never into the campaign worktree, its `results/`, checkpoints, or draw-file directories.
6. No commands were run, and no files were written, edited, or deleted, inside
   `/bbkinghome/edav/gravity_robustness/worktrees/campaign-five-family-bounds-2026-07-28` during
   this task.

## Environment

```
julia version 1.12.6   (via /bbkinghome/edav/.juliaup/bin, not /opt/shared_sw — see project memory
                         julia-toolchain-use-juliaup-not-shared-sw)
host: demand.mit.edu, Linux 4.18.0-553.137.1.el8_10.x86_64
```

## Merge policy for this task (per task spec §14)

- Do **not** alter the campaign's branch or worktree.
- If the campaign is still active (not confirmed finished) when this cleanup's gates pass:
  `PRODUCTION_MERGE = port_ready_waiting_for_campaign`. Push the cleanup branch; do not move
  `production/fullA-exact`.
- Final merge (only once the campaign is confirmed finished and all correctness/performance gates
  in this cleanup pass) requires: rebase onto current canonical `production/fullA-exact`, rerun
  concise D=4 + real D=20/W=100,000 gates, merge, push, tag
  `five-family-no-legacy-CC-H-G-storage-release-2026-07-28`, run post-merge smokes.

This document will not be edited after initial write except to append a closing status line once
the task concludes.
