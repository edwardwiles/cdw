# Full-A next-continuation handoff

Written at the end of the "performance profiling, D=4 completion, and staged scaling" continuation.
Read this first if picking up this investigation again.

## 1. Where everything is

- Repo: `git@github.com:habibiscoding/Trade-Model-Robustness.git`, branch `diag/fullA-d4-exact`,
  branched from production `53ffb58` (branch `sequential-profiled-gravity` in
  `trade_robustness_modular`).
- Worktree: `/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`.
- **Current HEAD at the end of this continuation**: run `git log --oneline -1` in the worktree —
  the last commit of this session is titled "Required deliverables: docs/fullA_d4_W_stability.md..."
  or later if more commits followed. Pushed to origin throughout; verify with `git fetch origin
  diag/fullA-d4-exact && git log origin/diag/fullA-d4-exact -1`.
- New code: `full_aod_diag/d4_exact/` (unchanged discipline from prior sessions) plus, this
  continuation, one read-only-adjacent artifact in the **production** worktree
  (`trade_robustness_modular/sequential_gravity/batch_out_d4x_continuation/` — untracked output from
  running the existing `run_profiled_production.jl` script with `OUT_DIR` overridden; no production
  file was modified, only a new output directory written).
- Results: `results/fullA_d4/<commit>/<run_id>/`, same convention as before.

## 2. What this continuation completed (commits `1b2a3a0`..`46b461c` and later)

1. **Phase 0**: state verification (`docs/fullA_next_resume_audit.md`) — HEAD matched the task's
   expected `8ae56d0` exactly, no drift.
2. **Phase 1 (mandatory, complete)**: instrumentation (`instrumentation.jl`, `oracle_profiled.jl`,
   equivalence-tested), component profiling (`profile_components.jl` →
   `docs/fullA_performance_profile.md`), gradient-method cost with corrected counters
   (`profile_gradient_methods.jl` — resolves the "n_inner_solves=34" mislabel from the prior
   continuation's Phase D: `Q_adj_FD`/`L_fix_FD` do **zero** real inner solves, confirmed via
   `CS.INNER_SOLVE_COUNT[]` diffs), and a complete D/W baseline scaling matrix
   (`profile_D_W_scaling.jl` → `docs/fullA_scaling_projection.md`, D∈{4,6,8,10} at W=8000, D=4 at
   W∈{8000,20000,80000}).
3. **Phase 6**: blockwise gradient re-check (`phase6_blockwise_gradient_check.jl`) — the single most
   important correction this continuation made: `L_fix_FD` is the only cheap method that survives
   scrutiny in the gravity-tangent A-block; hard pathwise AD's A-block gradient is anti-correlated
   with truth near the optimum despite a deceptively good full-vector cosine.
4. **Phase 5 (mandatory external validity check)**: located and ran a fresh sequential/profiled
   production solve on the identical synthetic economy (confirmed via `kappa_point_estimate` match,
   not filename trust — two candidate result files found first were actually stale D=10 runs).
   Full-A's upper candidate (κ=0.1718) beats the sequential method's own genuinely-feasible number
   (κ=0.0779) by >2x; caveated (one start, not fully converged).
5. **Phase 3 (partial)**: W-stability cost scaling complete; one candidate feasibility check at
   W=20,000 (independent, not nested, draws) — reassuring, not exhaustive.
6. **Documentation**: all 9 required deliverables exist (`docs/fullA_next_resume_audit.md`,
   `docs/fullA_performance_profile.md`, `docs/fullA_d4_W_stability.md` (partial),
   `docs/fullA_d4_profile_and_bounds.md` (scoped, not attempted),
   `docs/fullA_algorithm_frontier.md` (scoped, not attempted),
   `docs/fullA_scaling_projection.md`, updated `docs/fullA_d4_final_report.md` §9 and
   `docs/fullA_d4_recommendation.md` (with the required final decision labels), this handoff).

## 2a. CORRECTION (post-hoc, flagged by the sequential method's domain owner): the Phase 5
    full-A-vs-sequential comparison was wrong and is retracted

The original version of this handoff (and §9.3 of `docs/fullA_d4_final_report.md`) reported the
sequential run's terminal κ_upper=0.0779 as a valid comparison point and concluded full-A "beats"
sequential by >2x. **This was an error**: 0.0779 is below this investigation's own fixed-A benchmark
(κ_fixedA=0.1440, `results/fullA_d4/a377fff/movement_and_fixedA_check.txt`), which a properly-run
sequential search should never fall below (it optimizes a strict superset of what "fixed-A" allows).
Root cause, confirmed not speculated: the run used the *default* opt file
(`full_aod_diag/csw_outer_25.opt`, `maxit=25`, `eval_fcga=yes`+`hessopt=4`→silently downgraded to
L-BFGS) and one start, not the production 5-start multistart `sequential_methodology.tex` §9 requires.
**To get a reliable comparison next time**: re-run `run_profiled_production.jl` with a much larger
`maxit`/`maxtime_real`, `eval_fcga=no`, and genuine multistart (5 starts per the production spec) —
none of which the retracted run used. Both `docs/fullA_d4_final_report.md` and
`docs/fullA_d4_recommendation.md` have been corrected; treat any other document or memory referencing
"full-A beats sequential" or "κ=0.0779" from this investigation as stale until a proper comparison
exists.

## 3. What was NOT attempted, and why (do not silently assume it's done)

- **Phase 2** (block-locality optimization, incremental moment updates, parallel FD, allocation
  reduction): profiling identified two concrete targets (redundant `moments_recompute` — ~34% of
  wall time; `winner_compute`'s per-column `sort()` allocation pattern) but neither was implemented.
  This is the highest-value next step given it's now evidence-backed, not speculative.
- **Phase 4** (gamma-profile, upper polish, lower-direction completion): not started at all. Scoped
  in `docs/fullA_d4_profile_and_bounds.md` with exactly what exists to build on.
- **Phase 7** (wall-clock-matched algorithm frontier): not started. The existing Phase B data from
  the PRIOR continuation is iteration-matched and explicitly flagged as not substituting.
- **Phase 8** (staged D pilots beyond the Phase 1C cost benchmark): no actual D=6/8/10 optimization
  pilot was run, only per-evaluation/per-gradient cost benchmarking. `docs/fullA_scaling_projection.md`
  gives the cost basis for deciding whether/when to attempt one.
- **W-stability breadth** (Phase 3): only the maxit=40 upper candidate was checked, only at
  W=20,000, only with independent (not nested) draws, only via a same-theta recheck (no
  re-optimization). The lower candidate, poll-improved points, W=80,000, and nested draws are all
  outstanding.
- **Full A-matrix reconstruction of the sequential solution** (Phase 5's fuller requirement): only
  the gauge-invariant kappa comparison was done. Reconstructing the full competitiveness matrix
  requires re-running the sequential method's destination-share inversion for the non-focal columns
  and converting between the two gauges (`sequential_methodology.tex` §2.3) — not attempted.

## 4. Operational notes for whoever continues

- **Background jobs in this environment took much longer than expected but generally DID complete
  successfully** — several jobs that appeared to have "died" (no process found via `ps`, log file not
  growing for several minutes) turned out to still be running and completed correctly 10-20+ minutes
  later with a delayed completion notification. Do not conclude a background job has failed just
  because `ps` shows nothing and the log looks stale after a few minutes — wait for the actual
  completion notification before relaunching, or you will get duplicate/overlapping runs (which
  happened this session with the sequential production run — harmlessly, because
  `run_profiled_production.jl` has its own resume-skip logic keyed on result files already present in
  `OUT_DIR`, but do not rely on every script having that safety net).
- `profile_D_W_scaling.jl` now supports a `CONFIGS_SUBSET` env var (e.g. `CONFIGS_SUBSET=D8` or
  `CONFIGS_SUBSET=D10,W80000`) for resuming one config at a time — use this if a full-matrix run is
  interrupted again, rather than re-running everything.
- The sequential production run lives in the **production worktree**
  (`trade_robustness_modular/sequential_gravity/run_profiled_production.jl`), invoked with
  `DVAL=<D> DELTA_GRID=<comma-list> BOUND=<upper|lower|both> OUT_DIR=<path> julia --project=.
  sequential_gravity/run_profiled_production.jl`. Always set a **fresh, dedicated `OUT_DIR`** — the
  default (`sequential_gravity/batch_out`) has already been overwritten at least once across
  different D values by earlier sessions, which is exactly how this continuation nearly used a stale
  D=10 result under a D=4-looking filename (caught by checking the `D` field in the loaded JLD2, not
  the filename — always do this).

## 5. Next command to run

Given the priority ordering established in `docs/fullA_next_resume_audit.md` and reaffirmed by this
continuation's findings (Phase 1's redundant-moments finding is now the highest-value, lowest-risk
target):

```
cd /bbkinghome/edav/gravity_robustness/gravity-fullA-d4
julia --project=. full_aod_diag/d4_exact/test_oracle_profiled.jl   # confirm nothing has drifted
```

then start Phase 2 by implementing the `moments_recompute` redundancy fix (cache `inner_loop_internal`'s
own internal `obj.moments!` output rather than recomputing in `oracle.jl`'s `evaluate_fullA`) as a new,
additive, equivalence-tested variant — following exactly the pattern `oracle_profiled.jl` already
established (mirror, don't modify, equivalence-test before trusting).
