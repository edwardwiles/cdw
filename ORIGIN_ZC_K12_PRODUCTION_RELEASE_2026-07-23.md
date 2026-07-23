# Origin-specific-ZC K<=2 production release (2026-07-23)

**Release branch:** `release/fullA-origin-zc-K12-2026-07-23`
**Base:** `production/fullA-exact` @ `d886d1ddde69eec1382fbee5ab6313a79c7d1393` (cdw remote, confirmed current)
**Final release commit:** `72947b5ce361d8370b8f562643b1a4b431725788`
**Release tag (pending push):** `origin-zc-K12-production-ready-2026-07-23`

---

## 1. Provenance

- Old experiment tip: `experiment/fullA-origin-specific-zc-K12-2026-07-23` @ `9e31a06` (tested code tip `fdfe5b1`, `9e31a06` adds only the experiment's own integration report; forked from `d886d1d`).
- Current production base: `production/fullA-exact` local and `cdw/production/fullA-exact` both resolve to `d886d1d` -- **zero commits of drift** since the experiment's fork point. Verified `822b2b1` (CM-C+ promotion) and `a00dc2d` (CM+meanzc promotion) are both ancestors of `d886d1d`, so the "clean linear descendant" precondition is trivially satisfied.
- Because production had not moved, porting reduced to a clean `git merge --no-ff` of the experiment branch onto a new release branch cut from `production/fullA-exact` -- no reconciliation conflicts.

## 2. Files ported (all additive, 16 files / 2480 insertions / 1 line from the merge)

```
full_aod_diag/d4_exact/cm_originzc_target_layout.jl
full_aod_diag/d4_exact/cm_originzc_config.jl
full_aod_diag/d4_exact/cm_originzc_moments.jl
full_aod_diag/d4_exact/cm_originzc_cplus.jl
full_aod_diag/d4_exact/cm_originzc_production.jl
full_aod_diag/d4_exact/cm_originzc_checkpoint.jl        (CMCheckpointV5 + run_originzc_upper_checkpointed)
full_aod_diag/d4_exact/cm_originzc_profile.jl            (diagnostic-only, Optim-based)
full_aod_diag/d4_exact/d20_originzc_fixedpoint_gates.jl   (diagnostic-only)
full_aod_diag/d4_exact/d20_originzc_shakedown.jl          (diagnostic-only)
full_aod_diag/d4_exact/test_cm_originzc_checkpoint.jl
full_aod_diag/d4_exact/test_cm_originzc_cplus_equivalence.jl
full_aod_diag/d4_exact/test_cm_originzc_pure_moments.jl
docs/ORIGIN_SPECIFIC_ZC_MATH_NOTE_2026-07-23.md
ORIGIN_SPECIFIC_ZC_K12_INTEGRATION_REPORT_2026-07-23.md   (experiment's own report, kept as history)
```

Then, release-only additions on top of the merge:

```
full_aod_diag/d4_exact/originzc_production_stage_runner.jl   (NEW: production entry point, off by default)
full_aod_diag/d4_exact/originzc_cold_verify.jl                (NEW: cold verifier for origin-ZC checkpoints)
full_aod_diag/d4_exact/originzc_release_gateB_d20.jl           (NEW: this release's Gate B script)
scripts/cm_production_supervisor.sh                            (edited: parameterized, see section 3)
full_aod_diag/d4_exact/cm_originzc_checkpoint.jl                (edited: abspath fix, section 4)
full_aod_diag/d4_exact/cm_checkpoint.jl                         (edited: same abspath fix, shared entry point)
full_aod_diag/d4_exact/cm_production_stage_runner.jl            (edited: same abspath fix)
full_aod_diag/d4_exact/test_cm_originzc_checkpoint.jl            (edited: +1 regression test, section 4)
```

## 3. Schema reconciliation

Production's highest schema at fork point was `CMCheckpointV4` (schema=4). The experiment's `CMCheckpointV5` (schema=5) is exactly the correct successor -- no collision, no renumbering needed. `upgrade_schema4` (V4->V5) ported unchanged. Frozen struct definitions (`CMCheckpoint`, `CMCheckpointV3`, `CMCheckpointV4`) were not touched.

## 4. Rough edges fixed

**4.1 Absolute checkpoint paths.** `ckpt_dir = abspath(ckpt_dir)` now runs, before `mkpath`/any model setup, at the top of:
- `run_originzc_upper_checkpointed` (`cm_originzc_checkpoint.jl`)
- `run_cm_upper_checkpointed` (`cm_checkpoint.jl`, the shared CM-family entry point -- same hazard, same fix)
- `cm_production_stage_runner.jl`'s `CKPT_DIR` capture
- `cm_production_supervisor.sh`'s `CKPT_ROOT` capture (`main()`, before `mkdir -p`)

New regression test in `test_cm_originzc_checkpoint.jl` ("abspath(ckpt_dir) is immune to a working-directory change after capture") reproduces the exact hazard: capture a relative `ckpt_dir`, `cd()` elsewhere (simulating the real-data setup's own `cd()`), then save/load a checkpoint and confirm it landed under the launch-time directory. 6/6 assertions pass.

**4.2 Optim.jl dependency.** Confirmed Optim is not a canonical production dependency anywhere else in this repo (no `using Optim` / `NewtonTrustRegion` outside `legacy/*` comments at the production tip). Project.toml/Manifest.toml were reverted to byte-identical to production's own versions -- Optim was **not** added to the committed environment. `cm_originzc_profile.jl` (the only `using Optim` file) is reachable only from `d20_originzc_fixedpoint_gates.jl`, a diagnostic script never included by any production entry point; a doc comment documents the local `] add Optim` requirement to run it.

## 5. Production entry point (off by default)

`run_originzc_upper_checkpointed(...)` (`cm_originzc_checkpoint.jl`) is a separate function from `run_cm_upper_checkpointed` -- never silently routed through the CM driver. It is now included by a documented canonical entry point:

```julia
run_originzc_upper_checkpointed(w0;
    W = 80_000, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    distribution_restriction = :origin_specific_moments_zero_covariance,   # or :origin_specific_moments
    K_mean = 1, K_pair = 1,                                                 # analogously K=2
    power_target_layout = :origin_by_power, meanzc_basis = :direct,
    maxtime_real = 3600.0, ckpt_dir = "production_runs/<campaign>", label = "originzc_upper",
    cm_gradient_backend = :cplus)                                          # :reference = fallback
```

Unrestricted and CM-family defaults are unchanged (`run_cm_upper_checkpointed` itself was not edited except for the abspath fix in section 4.1).

## 6. Production supervisor

`scripts/cm_production_supervisor.sh` now accepts:

```
STAGE_RUNNER_SCRIPT   default full_aod_diag/d4_exact/cm_production_stage_runner.jl
COLD_VERIFY_SCRIPT    default full_aod_diag/d4_exact/cm_cold_verify.jl
DISTRIBUTION_RESTRICTION / ORIGIN_K_MEAN / ORIGIN_K_PAIR / POWER_TARGET_LAYOUT   (empty by default -- off)
```

Origin-ZC opts in by pointing `STAGE_RUNNER_SCRIPT`/`COLD_VERIFY_SCRIPT` at the new `originzc_production_stage_runner.jl` / `originzc_cold_verify.jl` and setting `DISTRIBUTION_RESTRICTION`. These are recorded into each stage's `run_meta.txt`. No second shell watchdog was written -- the existing process-group-safe watchdog/state-machine is reused unchanged (wall-budget termination, stall detection, process-group kill via `setsid`/pgid file, all unmodified).

## 7. Initialization

`nu_{o,k}^{(0)} = (1/W) sum_s z_{so}^k`, `eta_{o,k}^{(0)} = log(nu_{o,k}^{(0)})`, computed independently per origin (never a shared value) from the exact frozen production draws (`draw_design=:pseudorandom, draw_seed=20260719`). The full initial nu matrix and a checksum are recorded in `w0_used.jls` on every fresh calibration-mode run.

## 8. Release gates

### Gate A -- D=4 regression (real KNITRO)

| Suite | Result |
|---|---|
| `test_cm_originzc_checkpoint.jl` (incl. new abspath test) | 28/28 PASS |
| `test_cm_originzc_pure_moments.jl` (K=1, K=2) | 36/36 PASS |
| `test_cm_originzc_cplus_equivalence.jl` (K=1,K=2 x mean-only/ZC + K=3/2 structural) | 55/55 PASS, econ cosine=1.0, eta blocks bit-identical |
| `test_cm_meanzc_d4_gates.jl` (existing production CM+mean/ZC suite, unchanged) | full pass (Hessian eq. 24/24, inner-solve eq. 90/90, nesting 8/8+3/3, outer gradient 24/24, CM-only regression 7/7) |

### Gate B -- two real D=20 cold points (`originzc_release_gateB_d20.jl`)

Cache-disabled (no exact-point cache exists on this arm) cold inner solve + full C+ vs Reference gradient, nu initialized from actual draw moments. Same two points as the experiment's own D=20 gates -- no new points.

| Point | K | nStatus | verified | Delta_dual | primal-dual gap | KKT resid | C+ vs Ref cosine | econ max&#124;diff&#124; | eta max&#124;diff&#124; | eta finite | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (benchmark A*, gp=0.9877618976237339) | 1 | 0 | true | 0.003685 | 8.7e-19 | 1.8e-16 | 1.0000000000000002 | 1.30e-16 | 0.0 | yes | 5094 MB |
| B (unrestricted delta=1.0 incumbent, gp=0.9516647191604432) | 2 | 0 | true | 1.702216 | 5.6e-14 | 2.0e-14 | 1.0000000000000000 | 1.06e-13 | 0.0 | yes | 6421 MB |

`RELEASE GATE B: PASS`.

### Gate C -- supervisor smoke test

Production-shaped K=1 stage: D=20, W=80000, delta=0.1, start=A*, K_mean=1, K_pair=1, backend=:cplus, through the actual `cm_production_supervisor.sh`.

- Attempt 1 (120s budget): wall-budget hit before KNITRO's first callback returned (cold JIT compile on a host also running 3 concurrent production chains) -- **no checkpoint existed, so the supervisor correctly failed the stage rather than false-succeeding**; process group confirmed fully terminated, no orphans. Not a code defect -- a smoke-test budget too tight for this host's current load.
- Attempt 2 (300s budget, chain id `9`): reached KNITRO eval 1 at t=8.9s (feasible + verified immediately, since Point A's Delta<<0.1), checkpoint written (`:new_best`), ran to 10 evals, wall-budget terminated cleanly at 300s (SIGTERM, 30s grace, one SIGKILL escalation, confirmed no process-group members remain afterward). Absolute checkpoint path confirmed: `.../gravity-release-originzc-K12-2026-07-23/production_runs/2026-07-23_release_gate_smoke/delta_0.1/stage_latest.jls`.
- Supervisor's own cold-verify step (`originzc_cold_verify.jl`) initially **failed** on a real script bug (missing `bandwidth_cache` kwarg for the default `h_mode=:cached`, same class of bug already fixed in the Gate B script but missed here) -- fixed (commit `72947b5`), rerun directly against the same checkpoint:
  - `verified_success=true feasible=true`, cold `Delta_dual=0.08847148038566205` vs reported `0.08847148038566219` (`|diff|=1.4e-16`)
  - all 20 `eta_nu` coordinates present and finite
  - C+ vs Reference gradient: `cosine=0.9999999999999998`, `|diff|=4.2e-14`

`GATE C: PASS` (after the one fix above).

## 9. C+ confirmation

Every gate (A, B, C) independently confirms C+ vs Reference agreement at or near float64 roundoff (cosine 1.0 to 10-16 digits, econ-block diffs 1e-13..1e-16, eta blocks bit-identical at D=4 and max|diff|=0.0 at D=20). No sign mismatches anywhere.

## 10. Memory

Peak one-process RSS: 5.09 GB (D=20, K=1), 6.42 GB (D=20, K=2). Both well under 7.5 GB.

## 11. API and launch commands

**Production entry point:** see section 5.

**Supervisor (K=1 example):**
```bash
export STAGE_RUNNER_SCRIPT=full_aod_diag/d4_exact/originzc_production_stage_runner.jl
export COLD_VERIFY_SCRIPT=full_aod_diag/d4_exact/originzc_cold_verify.jl
export DISTRIBUTION_RESTRICTION=origin_specific_moments_zero_covariance
export ORIGIN_K_MEAN=1 ORIGIN_K_PAIR=1 POWER_TARGET_LAYOUT=origin_by_power
export CM_GRADIENT_BACKEND=cplus
export JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bash scripts/cm_production_supervisor.sh <chain_id> <ckpt_root_dir>
```

### First-look K=1 campaign (D=20, W=80000, France, 10 min/delta, four independent jobs)

Not part of the merge gate -- **not launched automatically**.

```bash
for chain in 0 1 2 3; do
  ( export STAGE_RUNNER_SCRIPT=full_aod_diag/d4_exact/originzc_production_stage_runner.jl
    export COLD_VERIFY_SCRIPT=full_aod_diag/d4_exact/originzc_cold_verify.jl
    export DISTRIBUTION_RESTRICTION=origin_specific_moments_zero_covariance
    export ORIGIN_K_MEAN=1 ORIGIN_K_PAIR=1 POWER_TARGET_LAYOUT=origin_by_power
    export CM_GRADIENT_BACKEND=cplus
    export DELTAS_OVERRIDE="0.1 0.5 1.0 2.0"
    export STAGE_WALL_S=600   # 10 minutes per delta
    export JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    bash scripts/cm_production_supervisor.sh "$chain" \
      "production_runs/2026-07-23_originzc_k1/chain${chain}" \
      > "production_runs/2026-07-23_originzc_k1/chain${chain}.wrapper.log" 2>&1 & )
done
```
Cold-verify after each job: `julia --project=. full_aod_diag/d4_exact/originzc_cold_verify.jl <ckpt> <out>` (the supervisor already does this between stages).

### K=2 command

Same as above with `ORIGIN_K_MEAN=2 ORIGIN_K_PAIR=2`. **Memory-safe concurrency**: measured single-process peak was 7.4 GB at K=2 (integration report) / 6.42 GB (this release's Gate B). With headroom, budget **no more than `floor(host_RAM_GB * 0.8 / 8)` simultaneous K=2 processes** -- on this 208-core host, confirm available RAM before choosing a chain count; do not assume core count implies safe process count for this memory-bound workload.

## 12. GO / NO-GO

**GO** for internal K=1 and K=2 production runs of the origin-specific-zero-covariance restriction via `run_originzc_upper_checkpointed` / `originzc_production_stage_runner.jl` under `scripts/cm_production_supervisor.sh`.

Gates A, B, and C all pass on the release branch. Pending: fast-forward `production/fullA-exact` to `72947b5`, push to `cdw`, tag `origin-zc-K12-production-ready-2026-07-23`.
