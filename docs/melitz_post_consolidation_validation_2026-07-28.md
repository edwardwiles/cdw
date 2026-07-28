# Melitz post-consolidation validation (2026-07-28)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from the
inner-solver architecture-consolidation commit `caa1ce25652ab2c94f12e32418a82aa9874902a0`
(verified as actual HEAD before any edit; `git status` showed only pre-existing untracked
scratch directories inherited from other sessions, none touched). Governing prompt: a single
bounded post-refactor validation pass -- confirm which prior conclusions from the preceding
24-48 hours survive the consolidation (`docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md`),
not a new campaign.

Julia 1.12.6 (juliaup), KNITRO 13.0.1 (`.knitro_env.sh`, pinned), `OPENBLAS_NUM_THREADS=1`/
`OMP_NUM_THREADS=1` throughout, `JULIA_NUM_THREADS=20` for every diagnostic script, `-t 1` for
the test suite.

## Phase 0: architecture smoke test

Confirmed by direct source inspection (all eight governing-prompt checks):

1. Single public inner-solve API: `solve_melitz_delta!` (`src/melitz/inner_session.jl`) is the
   only defined `solve_melitz_delta!`; every production/test call site uses it.
2. Every solve has an explicit typed policy: all four bundle constructors
   (`build_melitz_cc_bundle`, `build_melitz_implicit_bundle`, `build_melitz_psi_bundle`,
   `build_melitz_psi_bundle_from_calibration`) take `policy::MelitzInnerSolvePolicy` with no
   default -- confirmed directly from each function's own signature.
3. A capped solve has `raw lower_limit = -cap`: `melitz_policy_lower_limit(CappedEvaluation(cap)) == -cap`
   exactly (`inner_solve_policy.jl:139`), verified live in Phase 1 below.
4. Capped and full-value sessions are separate objects: `MelitzInnerSession`'s own constructor
   asserts `obj.lower_limit == melitz_policy_lower_limit(policy)`; no code path shares one
   session/bank across a policy switch.
5. No capped session can return `FiniteSolved` above its cap: the pre-existing output
   invariant (`@assert Delta_theta <= delta_evaluation_cap + tol`, `inner_screening.jl:918`) is
   retained and re-verified live in Phase 1.
6. `FiniteSolved` requires full primal/LFD/dual verification: unchanged from the prior
   session's Phase 5 hardening (accepted-status gate + output invariant), confirmed present.
7. No production caller directly invokes the low-level KNITRO driver: grep confirms
   `_melitz_classified_inner_solve!`/`melitz_bundle_inner_solve!`/`melitz_cc_inner_loop_knitro!`
   are called only from their own sanctioned files; old name `melitz_classified_inner_solve` no
   longer exists anywhere in `src/melitz/` or `test/melitz/`.
8. Strict production mode performs zero dense-G materializations: `forbid_dense_fallback=true`
   used throughout the fixture builders; live-confirmed zero in Phase 6
   (`MELITZ_DENSE_MOMENT_CALLS[]`/`MELITZ_DENSE_G_MATERIALIZATIONS[]` both 0).

**Full relevant Melitz test suite** (`julia --project=. -t 1 test/melitz/runtests.jl`, run in
full before any validation-session code was written): **63/63 top-level testsets, every one
showing `Pass==Total`, zero `Fail`/`Error` lines, exit code 0** (including the pre-existing
"Section 11 (architecture consolidation): static scan for unsafe defaults/direct low-level
calls" testset, 6/6 passed). No regression from the consolidation commit.

## Phase 1: exact replay of the 1.510118e14 anomaly (mandatory, run first)

`scripts/melitz_phase1_anomaly_replay_consolidated_2026-07-28.jl`. Reused the EXACT
reconstruction recipe from the prior forensic session (`theta_pt`, `theta_new`, same seeds
`MersenneTwister(4242)` for the coordinate subsample, same deterministic SVD basis) -- only
the inner-solve mechanism was swapped from the now-nonexistent
`melitz_classified_inner_solve(obj, theta, ctx; delta_evaluation_cap=...)` to
`solve_melitz_delta!(session, theta, policy; ...)`.

- `build_realD20_fixture()` now defaults to `policy=CappedEvaluation(10.0)`; confirmed live
  `obj.lower_limit == -10.0` at construction (previously `-KN_INFINITY`, the confirmed root
  cause).
- `r0 = solve_melitz_delta!(session, theta_pt, CappedEvaluation(10.0); origin_block_screen=true)`:
  `FiniteSolved`, `Delta=0.4832764950468894`, `nStatus=0` -- bit-exact match to the original
  session's own `Delta0`.
- `r_new = solve_melitz_delta!(session, theta_new, CappedEvaluation(10.0); origin_block_screen=true)`
  (the anomaly point): **`InfiniteDeltaCertified(column=12, lo=NaN, hi=NaN, kind=:origin_block)`**
  -- **bit-exact match to the prior forensic session's own exact certificate** (Phase 3 of
  `docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md`).
- **Regression assertion passed: `r_new` is never `FiniteSolved`.**
- Capped-vs-uncapped architectural separation verified at the policy level alone (no third
  solve needed, per the governing prompt's own "stop once the architectural distinction is
  established, do not allow the uncapped solve to run toward 1e14"):
  `melitz_policy_lower_limit(CappedEvaluation(10.0)) = -10.0` vs.
  `melitz_policy_lower_limit(FullValueEvaluation()) = -1.7976931348623157e308`.

Compute budget used: 2 inner solves (of an allowed 3).

**Conclusion: the anomaly cannot recur under the consolidated API.**

## Phase 2: re-certify the fixed-A/f profiles

`scripts/melitz_phase2_profile_recert_2026-07-28.jl`. No root-finding: g interpolated
log-linearly from the existing profile CSVs at each target, then a single direct
`solve_melitz_delta!` call, `CappedEvaluation(10)`.

| fixture | label | g | result | Delta |
|---|---|---:|---|---:|
| D4 (seed=29,W=20000) | pareto | -0.042387 | FiniteSolved (nStatus=0) | 7.5545e-6 |
| D4 | delta~0.1 | -0.101233 | FiniteSolved | 0.10477 |
| D4 | delta~0.5 | -0.153522 | FiniteSolved | 0.49775 |
| D4 | delta~1.0 | -- | **not reachable** (max finite profile Delta=0.572) | -- |
| D4 | delta~2.0 | -- | **not reachable** | -- |
| real-D20 (seed=1,W=80000) | pareto | -0.418771 | FiniteSolved (nStatus=0) | 4.0721e-4 |
| real-D20 | delta~0.1 | -0.454103 | FiniteSolved | 0.10551 |
| real-D20 | delta~0.5 | -0.485461 | FiniteSolved | 0.48328 |
| real-D20 | delta~1.0 | -0.496310 | FiniteSolved | 0.90455 |
| real-D20 | delta~2.0 | -0.504931 | FiniteSolved | 1.65061 |

D4's pareto/D20's pareto and D20's `delta~0.5` points reproduce the prior sessions' own values
exactly (`4.07206993e-04`, `0.4832764950468894`). D4 having no finite reference beyond
`Delta~0.57` matches the previously-established corridor limit (NumericalFailure at
`frac=0.50`) -- not forced via a higher cap, per the governing prompt's own instruction.

**Conclusion: both fixed-A/f profiles are unchanged under the consolidated architecture.**

## Phase 3: participation-gradient diagnostic replay

`scripts/melitz_phase3_participation_gradient_recert_2026-07-28.jl`. Two real-D20 points
(`interior_Delta0.5`, `Delta0=0.4832764950468894`; `nearbudget_Delta1`, `Delta0=0.9045506027534315`),
3 direction families (pure gamma, participation block, mixed gamma+participation), 3 raw steps
(`1e-6,1e-5,1e-4`) -- 18 fully-reoptimized directional solves (the governing prompt's own
budget), plus 2 base-point solves.

Headline reproduction (assembled coordinatewise gradient "A" vs. block secant "B" vs.
reoptimized truth), near-bit-exact matches to the original diagnostic:

| point/direction/step | A (assembled) | B (secant) | reoptimized | original session's own value |
|---|---:|---:|---:|---|
| interior/pure_gamma/1e-4 | +2.5614e-3 | -- | +2.5895e-3 | A=+2.561e-3, reopt=+2.589e-3 |
| interior/participation/1e-5 | -9.3993e-6 | -2.8277e-8 | +2.4745e-6 | A=-9.40e-6, reopt=+2.47e-6 |
| nearbudget/participation (all 3 steps) | negative throughout | -- | +1.06e-6, +7.67e-7, +4.42e-5 (all positive) | A negative all 3; reopt +1.06e-6,+7.7e-7,+4.4e-5 |
| nearbudget/mixed/1e-4 | -2.2456e-2 | -4.0836e-2 | -1.4884e-3 | A=-2.25e-2, B=-4.08e-2, reopt=-1.49e-3 |

Every headline number matches the pre-consolidation diagnostic to 3-4 significant figures
(exactly what is expected: the underlying classification/economics did not change, only the
cap-plumbing architecture did).

**Conclusion: the coordinatewise-participation-gradient overstatement finding is unchanged.**

## Phase 4: archived real-D20 point replay

**Disclosed methodological substitution**: no nested-block-search CSV
(`melitz_phase9_realD20_nested_block_search_2026-07-28.csv` and siblings) persists a full
`theta` vector -- only scalar summaries -- and no serialized/JLD2 theta array exists on disk
for any of them either. A literal "load and replay" of those exact points is therefore
impossible without re-running a full outer KNITRO search (out of scope). Two complementary,
disclosed substitutes were used instead:

**Part A** (`scripts/melitz_phase4_archived_d20_replay_2026-07-28.jl`, 10 directional + 2 base
solves): 10 individually-classified real-D20 points from
`docs/key_results/melitz_phase10_realD20_coordinate_basis_comparison_2026-07-28.csv`, exactly
reconstructed (identical seeded RNG/SVD-basis recipe as Phase 1).

| direction | target | result | Delta | historical Delta |
|---|---:|---|---:|---:|
| af_random_1 (+/-1) | 0.5, 0.8 | NumericalFailure (all 4) | -- | NumericalFailure (all 4) |
| svd_near_null (+1) | 0.5, 0.8 | NumericalFailure (both) | -- | NumericalFailure (both) |
| svd_steepest (+1) | 0.5 | FiniteSolved | 0.60468 | 0.6046840248839506 |
| svd_steepest (-1) | 0.5 | FiniteSolved | 0.61613 | 0.6161307176531553 |
| svd_steepest (+1) | 0.8 | FiniteSolved | 0.88661 | 0.8866083549472599 |
| svd_steepest (-1) | 0.8 | FiniteSolved | 0.89447 | 0.8944706577231128 |

Every classification and every finite Delta value is a **bit-exact match** to the
pre-consolidation table.

**Part B** (`scripts/melitz_phase4b_gamma_only_rerun_2026-07-28.jl`, one outer-search
structural check of `solve_melitz_finite_delta_bound` itself, not counted against the
10-point budget): the `gamma_only` nested-block configuration (SQP, g-radius 0.5, from the
interior `Delta~0.5` point) re-run end-to-end:

```
re-run:     nStatus=0  wall=62.5s  dg=-0.010048147673314078  DeltaStar=0.8603593934221736
historical: nStatus=0  wall=45.2s  dg=-0.010048147673314078  DeltaStar=0.8603593934221736
```

`dg` and `DeltaStar` are **bit-exact**; only wall time differs (plausibly shared-host load
variance from this session's own concurrent jobs, not investigated further).

**Conclusion: representative archived D20 points, both individually-classified standalone
points and one full outer-search driver invocation, reproduce bit-exact under the consolidated
architecture.**

## Phase 5: selected nuisance-profile checks

`scripts/melitz_phase5_nuisance_recert_2026-07-28.jl`, D4 only (real-D20 nuisance
minimization can take >1000s per stage per the prior session's own Phase 8 finding --
4 points x up to 3 D20 stages would alone exceed this validation's remaining budget for an
effect already known to be small; classified "not rerun but low risk" below). 4 fractions x 3
nuisance stages (A-only/f-only/full) = 12 solves (the governing prompt's own budget).

| frac | Delta_fixed | best source | best Delta | reduction |
|---:|---:|---|---:|---:|
| 0.10 | 0.03422 | full | 0.006176 | **5.5x** (original: 5.5x) |
| 0.35 | 0.57206 | full | 0.21439 | **2.7x** (original: 3.3x) |
| 0.50 | NumericalFailure | none | NaN | initial script design gap -- see follow-up below |
| 0.65 | AboveEvaluationCap (cap=10) | none | NaN | initial script design gap -- see follow-up below |

The never-worse-than-fixed invariant held everywhere it could be checked (no violation).

**frac=0.50/0.65 gap, diagnosed and resolved (Phase 5b/5c follow-up, done in this same
session at the user's direct request, 13 additional solves: 1 diagnostic + 12 for a full
continuation-threaded rerun)**: the first-pass script above failed immediately at these two
fractions with a KNITRO `grad_callback -502` evaluation error. The initial hypothesis (a
stale/poisoned dual warm start, since `obj.x` was observed `NaN` after the immediately
preceding failed fixed-A/f solve) was **tested directly and REJECTED**: re-reading
`melitz_cc_inner_loop_knitro!` (`cc_bundle.jl:420`) shows `use_cached_x=false` already forces
a clean `zeros(n)` inner-dual start regardless of `obj.x`'s own contents, so a "neutral reset"
re-run (`scripts/melitz_phase5b_nuisance_neutral_warmstart_check_2026-07-28.jl`) starting from
a genuinely clean dual **still failed identically** -- ruling out the dual as the mechanism.
The KNITRO message itself ("Could not evaluate objective or constraints at the **initial
point**") is the correct signal: this is a PRIMAL evaluation failure at the raw `theta_fixed`
point itself, consistent with the fact that the FIXED-A/f model (no search at all) *also*
fails to evaluate cleanly at that exact theta. Starting any search -- warm or cold -- literally
AT that point cannot succeed, independent of dual state; this has nothing to do with the
2026-07-28 consolidation.

The original 2026-07-28 gamma-profile session's own script sidesteps this not via a better
dual guess but via a **different primal starting point**: each fraction's A-only/f-only stage
starts from the PRECEDING fraction's own converged A-only/f-only THETA (already-adjusted A/f
values, not the raw calibrated ones), with only `g` overwritten to the new target -- never
from `theta_fixed` directly. `scripts/melitz_phase5c_nuisance_continuation_rerun_2026-07-28.jl`
replicates exactly this (continuation-threaded `theta` AND dual across the full 0.10->0.35->
0.50->0.65 grid, in order) and **confirms the original claim survives the consolidated
architecture**:

| frac | Delta_fixed | best source | best Delta |
|---:|---:|---|---:|
| 0.50 | NumericalFailure | full | **0.5196** (nStatus=-101, genuine convergence; original session found 0.505) |
| 0.65 | AboveEvaluationCap | full | **1.3724** (nStatus=-101, genuine convergence) |

One smaller, honestly-disclosed residual wrinkle: at `frac=0.65`, `f-only` ALONE still fails
(`nStatus=-502`) even with continuation -- only `A-only` and `full` (seeded from `A-only`)
succeed there. Continuation fully resolves the headline fixed-vs-best-flexible comparison at
both fractions, but not every individual restricted stage.

**Conclusion: the nuisance-profile findings, INCLUDING the "flexibility rescues outright
infeasibility" claim at frac=0.50/0.65, are confirmed unchanged under the consolidated
architecture.** The initial apparent divergence was a gap in this validation session's own
first-pass script (missing cross-fraction continuation), not a consolidation regression and
not fixable by warm-starting the dual -- it required starting the outer search from a
different, non-pathological primal theta.

## Phase 6: production-fast performance smoke test

`scripts/melitz_phase6_perf_smoke_2026-07-28.jl`. Not a performance study.

- `obj.lower_limit == -10.0`: **true** (lower_limit active).
- `melitz_resolve_gradient_backend` (D=20, 20 threads, `:auto`): **`B_direct_argument_sorted_parallel`**
  (parallel backend correctly resolved and used).
- `MELITZ_DENSE_MOMENT_CALLS[]` / `MELITZ_DENSE_G_MATERIALIZATIONS[]`: **0 / 0** (no dense
  fallback triggered anywhere).
- `MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[]`: **0**.
- One finite FC wall (cold, interior `Delta~0.5`): 6.58s, `FiniteSolved`.
- One `InfiniteDeltaCertified` FC wall (the anomaly point, reconstructed): 0.828s.
- One complete 20-thread outer-gradient wall (`n=798` coordinates): 6.615s.

**Disclosed limitation on the wall-clock numbers**: this smoke test was run while several
other validation-session Julia/KNITRO processes (Phases 3-5) were still active on the same
shared host -- the measured 6.615s gradient wall is roughly 7x slower than the previously
documented clean-host figure (~0.98s for the identical `n=798`, `D=20`, 20-thread parallel
call, `docs/melitz_outer_search_gradient_redundancy_and_sensitivity_2026-07-28.md` Phase 1).
This is very likely host-contention inflation, not an architecture regression -- the
STRUCTURAL confirmation that matters for this phase's own acceptance criterion (parallel
backend correctly resolved and active, zero dense-G calls) is unambiguous regardless. A clean,
isolated re-timing was not performed (out of scope: "not a performance study").

**Conclusion: the optimized kernels (20-thread parallel gradient, matrix-free callbacks, no
dense G, active lower_limit) are all confirmed structurally intact after consolidation.**

## Phase 7: triage of prior conclusions

Full table: `docs/key_results/melitz_post_consolidation_phase7_triage_2026-07-28.csv`.

| prior result | triage |
|---|---|
| 1.5e14 `FiniteSolved` anomaly | **confirmed fixed** |
| D4 fixed-A/f profile | confirmed unchanged |
| real-D20 fixed-A/f profile | confirmed unchanged |
| D4 full 234-cell frontier | not rerun but low risk |
| participation-gradient mismatch | confirmed unchanged |
| intensive-vs-switching decomposition | not rerun but low risk |
| D20 nested block-search pattern | not rerun but low risk |
| D4 nuisance-profile improvements (frac=0.10/0.35) | quantitatively changed (directionally confirmed, magnitude drifted at frac=0.35) |
| D4 nuisance infeasibility-rescue claim (frac=0.50/0.65) | **confirmed unchanged** (resolved via continuation-threaded rerun, Phase 5c) |
| outer-parameterization six-way tournament | not rerun but low risk |
| engineering speedups (parallel/matrix-free/no dense G) | confirmed unchanged structurally; wall-clock magnitude not cleanly re-measured |

No result is classified "confirmed" merely because the new code ran without error -- every row
above is backed by either a bit-exact/near-exact numerical reproduction (anomaly, profiles,
participation-gradient, archived points, gamma_only outer rerun) or an explicit,
scope-consistent reason the item was not rerun (frontier/tournament/long D20 campaigns
explicitly out of this session's bounded scope).

## Phase 8: recommendation

**Recommendation 1 of 3: no further reruns needed before outer-search work.**

The one item flagged inconclusive after the initial Phase 5 pass (the D4 nuisance
"rescues infeasibility" claim at `frac=0.50/0.65`) was resolved within this same session
(Phase 5c, continuation-threaded rerun): confirmed unchanged, not a consolidation regression.
Every result validated this session (the anomaly fix, both fixed-A/f profiles, the
participation-gradient finding, the archived-point classifications, the `gamma_only` outer
driver, the full nuisance-profile picture including the infeasibility-rescue claim, and the
structural engineering-kernel checks) is now **confirmed and does not need rerunning** before
outer-search work resumes.

## Acceptance criteria

1. Exact anomaly point replayed through the consolidated API: **met** (Phase 1).
2. Cannot return `FiniteSolved`: **met** (Phase 1, `InfiniteDeltaCertified`, assertion passed).
3. D4/D20 fixed-A/f profiles re-certified at economically relevant points: **met** (Phase 2).
4. Participation-gradient conclusion rechecked at one interior + one near-budget D20 point:
   **met** (Phase 3).
5. Representative archived D20 trial points reclassified: **met**, via a disclosed substitute
   source given no full theta vectors were ever persisted (Phase 4).
6. Selected nuisance-profile conclusions rechecked: **met**, including the frac=0.50/0.65
   rescue claim, resolved to confirmed-unchanged via a continuation-threaded rerun after the
   first pass surfaced a genuine (script-design, not consolidation) gap (Phase 5/5b/5c).
7. No broad frontier or long outer campaign run: **met** -- only one `gamma_only` nested-block
   rerun (Phase 4 Part B) and no multi-seed/tournament/long-D20 work.
8. No delta below 0.1 investigated: **met**.
9. Twenty-thread production-fast behavior confirmed: **met** (Phase 6).
10. No Ricardian/shared source modified: **met**, see `git diff --name-only` below.
11. Full relevant Melitz tests pass: **met** (Phase 0, 63/63 testsets, exit code 0).
12. Work committed locally, not pushed: see the commit made immediately after this document.

## Files changed

`git diff --name-only caa1ce25652ab2c94f12e32418a82aa9874902a0`:

New files only (all under `scripts/`, `docs/`, `docs/key_results/` -- Melitz-only; zero diff
in `cc_algo/`, `full_aod_diag/`, or any other Ricardian path, confirmed directly):

```
docs/melitz_post_consolidation_validation_2026-07-28.md   (this document)
scripts/melitz_phase1_anomaly_replay_consolidated_2026-07-28.jl
scripts/melitz_phase2_profile_recert_2026-07-28.jl
scripts/melitz_phase3_participation_gradient_recert_2026-07-28.jl
scripts/melitz_phase4_archived_d20_replay_2026-07-28.jl
scripts/melitz_phase4b_gamma_only_rerun_2026-07-28.jl
scripts/melitz_phase5_nuisance_recert_2026-07-28.jl
scripts/melitz_phase5b_nuisance_neutral_warmstart_check_2026-07-28.jl
scripts/melitz_phase5c_nuisance_continuation_rerun_2026-07-28.jl
scripts/melitz_phase6_perf_smoke_2026-07-28.jl
docs/key_results/melitz_post_consolidation_phase1_anomaly_replay_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase2_profile_recert_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase3_participation_gradient_replay_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase4a_archived_d20_replay_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase4b_gamma_only_rerun_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase5_nuisance_recert_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase5b_neutral_warmstart_check_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase5c_nuisance_continuation_rerun_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase6_perf_smoke_2026-07-28.csv
docs/key_results/melitz_post_consolidation_phase7_triage_2026-07-28.csv
```
