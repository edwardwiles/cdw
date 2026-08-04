# Outer algorithm inventory — what's already wired vs. what the old campaign used

Per task Section 7: "First inventory the actual current FULL outer algorithms and option files
already supported by the repository. Do not invent a new optimizer wrapper if existing drivers
support the required algorithms." Verified directly (file:line reads), not from memory.

## What the old 10x10 campaign actually ran under

Every cell in the old campaign ran with `algorithm=auto` (`csw_outer_wallclock_sr1.opt`, the
default `hessopt_tag="sr1"` file) and no `outer_direct_hessopt`/`opt_file` override — confirmed in
`CURRENT_CAMPAIGN_RUNNER_AND_SOLVER_SETTINGS_2026-08-03.md` and independently re-confirmed by
reading `campaign_unrestricted_runner.jl`/`campaign_cm_family_runner.jl` call sites directly (no
`outer_direct_hessopt=`/`opt_file=` at either). `algorithm=auto` resolves to Active-Set/CG for this
unconstrained-profile formulation (documented in-repo, `c10_d20_production_driver_unified.jl:176-183`).

## Exploration arm: Direct + SR1

Already wired, identically, on all 3 driver families actually used by this campaign:

| Family | Function | Kwarg |
|---|---|---|
| unrestricted | `run_polish_checkpointed_unified` | `outer_direct_hessopt=:sr1` |
| flexible_cm / common_frechet / cm_meanzc | `run_cm_upper_checkpointed` / `run_cm_lower_checkpointed` | `outer_direct_hessopt=:sr1` |
| origin_zc | `run_originzc_upper_checkpointed` / `run_originzc_lower_checkpointed` | `outer_direct_hessopt=:sr1` |

Forces genuine `algorithm=direct`(1) + `hessopt=SR1`(3), full `maxit` (effectively unbounded),
bounded only by `maxtime_real`. No new code required — just pass the kwarg the old campaign never
passed. `c10_d20_production_driver_unified.jl:38,325-328,507-508`; `cm_checkpoint.jl:1201-1204`;
`cm_originzc_checkpoint.jl:704-706`.

## Polish arm: SQP (CM/origin-ZC), Direct+BFGS fallback (unrestricted)

- **CM-family + origin-ZC**: `cm_checkpoint.jl` and `cm_originzc_checkpoint.jl` both take a
  free-form `opt_file::String` kwarg (default `"csw_outer_wallclock_sr1.opt"`) loaded verbatim via
  `KNITRO.KN_load_param_file` (`cm_checkpoint.jl:1035`). A genuine SQP option file already exists
  and is pre-built for a short local-refinement budget:
  `csw_outer_phaseB_sqp_bfgs_maxit15.opt` → `algorithm sqp` (KNITRO algorithm=4, Active-Set SQP),
  `hessopt=2` (BFGS shown in filename; file itself sets `hessopt` numerically), `maxit=15`. Pass
  `opt_file="csw_outer_phaseB_sqp_bfgs_maxit15.opt"` to `run_cm_upper_checkpointed`/
  `run_cm_lower_checkpointed`/`run_originzc_upper_checkpointed`/`run_originzc_lower_checkpointed`.
  Zero new code.
- **unrestricted**: `run_polish_checkpointed_unified` has **no** `opt_file` kwarg — only
  `hessopt_tag` (constrained to the fixed `csw_outer_wallclock_$(tag).opt` naming pattern; only
  `sr1`/`lbfgs`/`productfd` exist under that prefix, all `algorithm=auto`) and
  `outer_direct_hessopt` (`:sr1`/`:bfgs` → Direct only). **SQP is not reachable for this driver
  without new code.** Per task Section 7's own fallback instruction ("if unavailable... test the
  next established local alternative rather than guessing"): use `outer_direct_hessopt=:bfgs`
  (Direct+BFGS) as unrestricted's polish arm — a real, already-wired, more local-convergence-prone
  method than Active-Set/CG, distinct from the SR1 exploration arm, with the polish framing applied
  via a short `maxtime_real` budget at the orchestration level (this driver has no built-in
  `maxit=15`-style cap to inherit).

## Also available, not selected

- `csw_outer_phaseB_direct_bfgs_maxit15.opt` (`algorithm direct`, `hessopt=2`, `maxit=15`) — same
  mechanism as the SQP file, reachable via `opt_file` on CM/origin-ZC drivers only. Kept as the
  named third pilot arm ("one other currently supported algorithm only if necessary", Section 8)
  in case SQP underperforms in the pilot.
- No `KN_set_var_scalings*` call exists anywhere in the FULL driver chain — outer variables are
  not explicitly rescaled; whatever KNITRO's automatic scaling does is what's in effect for every
  arm (unchanged across arms, so not a confound between exploration/polish).
- `outer_algorithm_override` (mentioned in an unrelated same-day diagnostic session as added to
  `run_polish_checkpointed`, the **non**-unified driver) does not exist anywhere on
  `production/fullA-exact` or in this worktree — confirmed via `git log --all -S`. Not used here;
  `run_polish_checkpointed_unified`'s own `outer_direct_hessopt` (a different, pre-existing kwarg)
  covers the same need for the unified driver this campaign actually uses.
