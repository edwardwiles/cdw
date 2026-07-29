# Provenance — flexCM/Fréchet inner-FG harmonization (2026-07-29)

## Source

- Canonical production branch: `production/fullA-exact`, remote `cdw` (`git@github.com:edwardwiles/cdw.git`)
- Fetched SHA: `8aae735d1b0e3d198a70a3fb585b7c6ac5d8fe77`
  ("Fix unrestricted campaign driver: use w_a directly, not a hand-rolled transform")
- Nearest tag: `selective-structured-hessian-release-2026-07-28` (this task's HEAD is 2 commits ahead
  of that tag: `2f98763` hardening commit, then `8aae735`)
- No-(H) operator-bundle release tag reachable in history: `five-family-true-operator-no-H-release-2026-07-28`
- CM/Fréchet Hessian harmonization commits (already on this branch, pre-existing):
  ```
  4fbd720 Harmonization step 1: extract shared fill_cm_HCC! (H_CC block)
  a2a867a Harmonization step 2: common Fréchet uses the shared pack_upper_cm_hessian!
  19adac1 Harmonization step 3: CMFrechetExtension with persistent level-block scratch
  d9db756 Harmonization steps 4-5: common Frechet's Hessian fill is now the shared hessian_cm_structured!
  680b0f9 Harmonization step 6: shared cumulative_backward_gradient_from_prefix! (CM transpose)
  13d3aca Harmonization step 7: shared _verify_inner_solution_operator_cm_core (CM verification)
  ```

## Isolated worktree/branch (this task)

- `git worktree add /bbkinghome/edav/gravity_robustness/worktrees/refactor-harmonize-flexCM-frechet-inner-FG-2026-07-29 -b refactor/harmonize-flexCM-frechet-inner-FG-2026-07-29 cdw/production/fullA-exact`
- Confirmed clean (`git status` empty) immediately after worktree creation, before any edits.
- The overnight five-family campaign (`worktrees/five-family-overnight-2026-07-28`, PIDs 2087475/
  2151107/2152551/2152609/2167343, `campaign_cm_family_runner.jl`/`campaign_unrestricted_runner.jl`,
  started ~10:21-10:42) and a separate H_ZZ backend bakeoff (PID 2188296,
  `hzz_backend_direct_bakeoff_2026-07-29.jl`) were both confirmed RUNNING at task start and were
  never touched — no files under `worktrees/five-family-overnight-2026-07-28` or that job's
  checkout were read, edited, or executed against by this task.

## Environment

- Julia: `julia version 1.12.6` (via `juliaup`, not `/opt/shared_sw` — broken copy per standing
  project memory)
- KNITRO: `14.2.0` (`KNITRODIR=/opt/shared_sw/knitro/14.2.0`, `ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt`)
- Host: 208 logical CPUs, 3.0 TiB RAM; load average ~62-96 at task start (from the concurrent
  overnight campaign + other users' jobs) — all gates in this task were run with modest thread
  counts (`-t 4` to `-t 8`) to avoid materially disturbing that campaign.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for every Julia invocation in this task (project
  standing requirement).

## Sobol / draw-design checksum

The equivalence gates in this task use `d4_exact_setup` (D=4, synthetic) and `d20_real_setup(W =
80_000, δ = 1.0, destination_sample = :exclude_row)` (real D=20 data) — the same construction calls
used by the pre-existing production gate scripts this task reuses/extends
(`bench_frechet_operator_fg_default_gate_2026-07-27.jl`, `c12i_validate_lookup_fg.jl`). Both draw
designs are deterministic given those arguments (fixed RNG seed inside `draw_design.jl`); this
task did not change `draw_design.jl`, `context.jl`, or `context_real_d20.jl`, so the draw design is
byte-identical to what every other gate in this branch's history already used. No separate
checksum artifact was generated beyond the equivalence gates' own PASS/FAIL comparisons against the
untouched dense reference callable, which is the sharper test (any draw-design drift would show up
as a real f/g mismatch, not just a different checksum string).

## Clean status

No uncommitted changes at task start. All work in this task lands as new commits on
`refactor/harmonize-flexCM-frechet-inner-FG-2026-07-29`, never by editing history on
`production/fullA-exact` directly. The overnight campaign was still running at the point this task
reached a port-ready state — see the master report for the merge-timing decision.
