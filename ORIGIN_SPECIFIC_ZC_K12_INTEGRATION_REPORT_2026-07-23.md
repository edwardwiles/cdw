# Origin-specific pairwise-zero-covariance restriction: integration report (2026-07-23)

**Branch:** `experiment/fullA-origin-specific-zc-K12-2026-07-23`
**Worktree:** `gravity-experiment-fullA-origin-specific-zc-K12-2026-07-23`
**Merge recommendation: READY_TO_MERGE** (see Section 8)

---

## 1. Exact Git provenance

```
$ git fetch cdw --prune
$ git rev-parse production/fullA-exact         # d886d1ddde69eec1382fbee5ab6313a79c7d1393
$ git rev-parse cdw/production/fullA-exact      # d886d1ddde69eec1382fbee5ab6313a79c7d1393  (identical)
```

`production/fullA-exact` is one commit past the required tag:

```
d886d1d (HEAD -> production/fullA-exact) Fix nested-KN_solve deadlock reintroduced via ek_inner.opt
a00dc2d (tag: cm-meanzc-production-ready-2026-07-23) Fix two real launcher bugs found during the D=20 supervisor gate
```

This is a linear advance (confirmed via `git log --oneline production/fullA-exact`), so per the task instructions the branch was created from the current canonical tip, `d886d1d`:

```
git worktree add -b experiment/fullA-origin-specific-zc-K12-2026-07-23 \
  ../gravity-experiment-fullA-origin-specific-zc-K12-2026-07-23 d886d1d
```

An existing worktree, `gravity-experiment-fullA-cm-pairwise-zero-cov` (branch `archive/fullA-cm-mean-zc-prototype-2026-07-22`), was identified as the "old prototype" referenced in the task brief. **It was not read for implementation and no code was ported from it** — this integration reuses the *current production* `cm_meanzc_*.jl` machinery (post `cm-meanzc-production-ready-2026-07-23`) as its base, per the task's explicit instruction.

Six commits were made on the experiment branch, one per logical unit (target-layout+math note; moment construction+gradient; C+ equivalence gate; checkpoint/config; profiling helper+shakedown launcher+robustness fix). **Zero commits touch `production/fullA-exact`.**

---

## 2. Math / derivative note

Full derivation: `docs/ORIGIN_SPECIFIC_ZC_MATH_NOTE_2026-07-23.md` (committed, 159 lines). Summary:

- Outer parameters `nu_{o,k} = E_F[z_o(omega)^k]`, one per (origin, power), `eta_{o,k} = log(nu_{o,k})`.
- Mean-defining moments `E_F[z_o^k - nu_{o,k}] = 0` (one per origin, per level `k<=K_mean`).
- Pairwise zero-covariance moments `E_F[z_o^k z_p^k - nu_{o,k} nu_{p,k}] = 0` (one per unordered pair `o<p`, per level `k<=K_pair`).
- **Analytic envelope derivative** (derived from the current canonical `Delta_dual = -(mean(Psi(q*))+zeta*)` sign convention, not inferred from any prior report):

```
d(Delta_dual)/d(nu_{o,k}) = -mean_m * ( lambda_mean,o,k*  +  sum_{p != o} nu_{p,k} * lambda_pair,op,k* )
d(Delta_dual)/d(eta_{o,k}) = nu_{o,k} * d(Delta_dual)/d(nu_{o,k})          (chain rule, nu = exp(eta))
```

- **Consistency check (not assumed):** summing this per-origin formula over all D origins under `nu_{o,k} = nu_{p,k} = nu_k` (the shared-target special case) reproduces the *existing* production `d_delta_dual_d_nu_vec` formula (`cm_meanzc_moments.jl`) bit-for-bit, using the double-counting identity for unordered pairs. This was verified both algebraically (math note Section 4) and empirically (Section 4 below).

---

## 3. Target-layout abstraction (task brief Section 4)

New file `cm_originzc_target_layout.jl` (157 lines). Immutable `MeanZCTargetLayout` abstract type with two concrete subtypes:

- `SharedByPowerLayout(K_mean, K_pair)` — the existing production convention (one `nu_k` per level, shared across all D origins). `n_eta = K_mean`.
- `OriginByPowerLayout(D, K_mean, K_pair)` — the new convention. `n_eta = K_mean*D`, level-major/origin-minor ordering (`eta_{1,1},...,eta_{D,1},eta_{1,2},...`).

Provided capabilities (per the task's explicit list): `n_eta`, `target_index(layout,o,k)`, `mean_targets`/`pair_targets` (dispatched through `target_index`, one implementation for both layouts), `layout_checkpoint_meta`, `layout_fingerprint`. No mutable `Ref`; every function takes the complete eta/nu vector as an explicit argument.

The `:anchored` moment basis is a reparameterization that only makes sense when every origin shares one target (math note Section 3); `OriginByPowerLayout` supports `:direct` only and hard-errors on `:anchored` — verified in `test_cm_originzc_checkpoint.jl`.

---

## 4. D=4 correctness gates

All run against **real KNITRO** (`test_cm_originzc_pure_moments.jl`, `test_cm_originzc_cplus_equivalence.jl`, `test_cm_originzc_checkpoint.jl`), `julia --project=. <file>`, K_mean=K_pair in {1,2}:

| Gate | Result |
|---|---|
| Moment-column construction vs dense reference | PASS, max&#124;diff&#124; < 1e-10 |
| Unordered-pair indexing/dims | PASS (npair = D(D-1)/2 = 6 at D=4) |
| Inner solve + canonical Delta_dual (4 configs: (1,0)/(1,1)/(2,0)/(2,2)) | PASS, nStatus=0, KKT resid 1e-14..1e-17 |
| Mean-only-arm implementation-equivalence (profiled) | PASS: `Delta_profiled = 0.00100299...` vs `Delta_unrestricted = 0.00100299...`, diff = **-1.08e-17** |
| ZC nesting (`Delta_mean_only <= Delta_zc`) | PASS, K=1: +4.0e-4, K=2: +6.5e-4 |
| Analytic eta_{o,k} derivative vs reoptimized FD | PASS: K=1 max&#124;diff&#124;=5.5e-9 cosine=1.0; K=2 max&#124;diff&#124;=2.7e-7 cosine=1.0 (single bandwidth h=1e-4, no second bandwidth needed) |
| C+ vs Reference full gradient (4 configs + (3,2) structural smoke) | PASS: econ cosine=1.0 (12 digits), eta blocks **bit-identical** (max&#124;diff&#124;=0.0) |
| Checkpoint schema-4→5 upgrade (cm_only case, meanzc case) | PASS |
| CMCheckpointV5 round-trip | PASS |
| Mixed-family refusal (`cm_extension` + `distribution_restriction` both active) | PASS (hard error) |
| Config resolve/validate refusal (bad K/K_pair/basis combos) | PASS (8/8 refused correctly) |
| Existing CM+meanzc D=4 regression suite (`test_cm_meanzc_d4_gates.jl`) | **PASS, unchanged** — Hessian equivalence 24/24, inner-solve equivalence 90/90, fixed-point nesting 8/8+3/3, outer gradient 24/24, CM-only regression 7/7 |

No production file was modified, so this regression pass is definitional (byte-identical code path) — re-run anyway per task brief Section 11.1, confirmed live.

---

## 5. D=20 real fixed-point gates (task brief Section 11.2)

Real KNITRO, W=80000, `draw_design=:pseudorandom, draw_seed=20260719` (production convention). Two economic points, cold-verified under the plain unrestricted problem before use:

- **Point A** (calibrated benchmark A*): `gp = 0.9877618976237339`, `Delta_unrestricted = 0.0025908618751122` (nStatus=0)
- **Point B** (existing cold-verified unrestricted incumbent, `production_runs/2026-07-22/fullA_exact_unrestricted_670eac4/chain_A/stage3_d1.0`, delta=1.0): `gp = 0.9516647191604432`, `Delta_unrestricted = 0.9999584535526898` (nStatus=0; checkpoint-recorded value was `0.9999584535526901`, reproduced to `3e-15`)

At each (point, K), nu initialized from the actual frozen draw moments, profiled over eta at fixed economic point via `Optim.LBFGS` + analytic gradient (`cm_originzc_profile.jl` — test/diagnostic-only, never referenced by production; see Section 7's note on a robustness fix found live during this run).

| Point | K | n_inner | n_eta | pcx wall | profile wall (iters) | grad wall | Delta_dual (profiled) | max mean resid | max pair resid | C+ vs Ref cosine | C+ vs Ref max&#124;diff&#124; econ | eta max&#124;diff&#124; | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 612 | 20 | ~1.3s | 82.2s (7) | 15.9s | 0.003583078 | 9.8e-15 | 1.1e-14 | 1.0000000000000002 | 4.6e-17 | 0.0 | 5624.6 MB |
| A | 2 | 822 | 40 | 1.31s | 518.7s (27) | 5.6s | 0.005091582 | 1.8e-15 | 3.1e-15 | 1.0000000000000004 | 4.3e-17 | 0.0 | 7316.4 MB |
| B | 1 | 612 | 20 | 1.68s | 236.7s (10) | 15.6s | 1.013731130 | 5.1e-14 | 6.4e-14 | 1.0000000000000004 | 5.0e-14 | 0.0 | 5726.3 MB |
| B | 2 | 822 | 40 | 1.38s | 1065.0s (45) | 6.2s | 1.023879312 | 9.8e-15 | 4.1e-14 | 0.9999999999999994 | 5.8e-14 | 0.0 | 7379.8 MB |

Dimension counts (`n_inner = 402 economic + n_mean + n_pair`, `n_mean=K*D`, `n_pair=K*D(D-1)/2`) match the task brief's predicted table exactly (D=20: K=1 → 210 new moments, 20 eta; K=2 → 420 new moments, 40 eta).

**Reoptimized FD spot-checks on individual eta_{o,k} derivatives**, done AT the profiled (near-stationary) point per the task's own gate list — both the analytic and FD values are themselves near zero at a converged optimum (first-order condition), so this is a *consistency* check (both agree the gradient is ~0 there), not a repeat of the rigorous away-from-optimum FD validation already done at D=4 (Section 4, cosine=1.0 on clearly non-zero values):

- A, K=1: eta[1] analytic=7.9e-10, FD=-1.4e-9 (both ~1e-9, consistent with stationarity)
- A, K=2: eta[1] analytic=-6.5e-10 FD=2.4e-7; eta[22] (origin 2, level 2) analytic=-1.7e-10 FD=-3.0e-8 (both pairs ~1e-7..1e-10, consistent with stationarity)
- B, K=1: eta[1] analytic=2.4e-9 FD=2.8e-10
- B, K=2: eta[1] analytic=5.8e-9 FD=2.2e-7; eta[22] analytic=1.2e-9 FD=-3.9e-8

**All four (point, K) combinations pass every listed check.** Memory is operationally acceptable for a single process (peak 7.4 GB at K=2, vs the existing production benchmark's ~2.6 GB for the plain economic-only D=20 inner solve at n~=402 — a real, expected increase from the widened dense Hessian, not a leak or pathology).

---

## 6. Short outer shakedowns (task brief Section 11.3)

Direct joint constrained search over `(gp, zfree, all eta_{o,k})` via `run_originzc_upper_checkpointed` (real KNITRO outer loop, not a fixed-gp profile), from the calibrated benchmark A*, `cm_gradient_backend=:cplus`, one process per job (concurrent, resource-bounded to <=20 cores each via `JULIA_NUM_THREADS=20`+`OPENBLAS_NUM_THREADS=1`+`OMP_NUM_THREADS=1`):

| Job | Budget | KNITRO status | n_eval / n_grad | best gp | best Delta | kappa | Cold-verify &#124;diff&#124; | Verified |
|---|---|---|---|---|---|---|---|---|
| K=1, delta=0.1 | 600s (602.7s wall) | -401 (time limit, feasible) | 76 / 55 | 0.9755868082395418 | 0.0999668276242073 | 0.04036 | 3.19e-16 | VerifiedSolved |
| K=1, delta=1.0 | 600s (662.4s wall) | -401 (time limit, feasible) | 39 / 16 | 0.9629060030997255 | 0.9946835436340931 | 0.06106 | 7.77e-16 | VerifiedSolved |
| K=2, delta=1.0 | 600s (607.8s wall) | -411 (time limit, infeasible terminal — best_feasible found earlier, at n_eval=6) | 29 / 22 | 0.9699181182818445 | 0.9148241833281107 | 0.04963 | 1.11e-16 | VerifiedSolved |

**All three shakedowns produced a cold-verified feasible incumbent** (exceeding the task's minimum bar of "at least K=1"). These are performance/integration shakedowns, not final bounds, per the task brief.

---

## 7. Robustness finding (fixed live, not deferred)

During the D=20 fixed-point gate run at economic Point B, `Optim.LBFGS`'s first line-search step (starting from `eta0` derived at the calibration-point scale) pushed some trial `nu` coordinates far outside the region where the inner CC dual problem is feasible, throwing `CMExpectedSolveFailure` — which the profiling helper (`cm_originzc_profile.jl`) did not catch, unlike the actual production KNITRO callback, which already handles exactly this via `reject_point`. **Fixed**: the helper now catches `CMExpectedSolveFailure` inside its `fg!` and returns `Inf`/zero-gradient (the standard reject-this-point signal), matching the production pattern. Confirmed working: the rerun of Point B (both K=1 and K=2) completed cleanly after the fix. This bug was isolated to the test/diagnostic profiling helper — no production code was affected, and no analytic math changed.

A second issue was caught and self-corrected during this session: the first attempt at launching D=20 background jobs did not bound `OPENBLAS_NUM_THREADS`, causing each process to spawn 100+ OS threads and consume 10-19 cores (observed live, flagged by the user). Fixed by setting `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` alongside `JULIA_NUM_THREADS=20`, matching the existing production supervisor's own documented convention (`scripts/cm_production_supervisor.sh:85-86`). All subsequent runs (including the four D=20 gate combinations and three shakedowns reported above) ran resource-bounded.

A third, purely bookkeeping issue: the three shakedown checkpoints landed at the worktree root (`shakedown_ckpts/`) rather than the `full_aod_diag/d4_exact/shakedown_ckpts/` path passed on the command line — one of the included D=20 setup files silently changes Julia's working directory during real-data loading (a previously-documented pattern in this codebase, see the D=20 driver's own "setwd() into OTHER worktree" fix history), so the shakedown launcher's *relative* `ckpt_dir` argument resolved against that changed cwd rather than the launch directory. This affects only where the `.jls` artifact was written, not any computed value — all reported numbers were read directly from the run's own stdout, and the checkpoints (recovered via `find`) round-trip correctly (Section 6). A production launcher should pass an absolute `ckpt_dir` to avoid this ambiguity.

---

## 8. Final assessment

### Why D*K outer targets are needed
Without common marginals, each origin's population moment is a free, origin-specific quantity — there is no single shared value that could describe every origin's distribution simultaneously (that IS the common-marginals assumption, deliberately not imposed here). Representing "the mean of origin o's k-th power" therefore requires one parameter per (origin, power), not one shared parameter per power.

### Why the mean-only arm is definitional
Section 4 confirms, by direct profiled minimization (not assumption), that `Delta^unrestricted == min_nu Delta*_origin_moments` to `1e-17` at D=4 — because nu is free to match any distribution's own moments, the mean-defining constraint alone never restricts the feasible set. The real new restriction is the pairwise product condition, confirmed to weakly increase Delta relative to mean-only at both K=1 and K=2 (D=4 nesting test, Section 4).

### Computational cost: enlarged outer loop vs. smaller no-CM inner problem
The inner dimension (612 at K=1, 822 at K=2 for D=20) is *smaller* than CM+meanzc would be at a comparable L (CM+meanzc adds an L~50 grid block on top of the same economic base, i.e. roughly `402 + (D-1)*50 ≈ 1352` columns at L=50 alone, before any mean/pair block) — confirming the task brief's hypothesis that removing the CM grid more than offsets the larger per-origin outer parameter count. The dominant real cost observed is the **profiling wall time** at K=2 (up to ~1065s / 45 LBFGS iterations for one fixed-point evaluation) — this is a diagnostic-only cost (profiling isn't part of the actual joint outer optimization, which converges its own eta coordinates jointly with gp/zfree inside a single KNITRO run, as the shakedowns demonstrate finishing full outer searches, including many eta coordinates, within the same 600s budget as the existing unrestricted/CM arms).

### Are K=1 and K=2 operationally viable?
**Yes.** Both dimensions solve cleanly at D=20/W=80000 production scale: inner solves converge to machine-precision KKT residuals, C+ and Reference backends agree to float64 roundoff, memory stays in the single-digit-GB range for one process, and the KNITRO outer loop (the actual production optimizer) finds and cold-verifies feasible incumbents within a 10-minute budget at both K values.

### Merge recommendation

**READY_TO_MERGE.** Every gate in the task's decision rule (Section 14) passes:
- D=4 gates pass for K=1 and K=2 — yes (Section 4)
- Existing shared-target CM+ZC regressions pass — yes, unchanged (Section 4)
- Both D=20 fixed-point gates pass for K=1 and K=2 — yes, at BOTH tested economic points (Section 5, exceeding the two-point requirement is not extra scope, it is the two points the brief specified)
- C+ and Reference agree under existing release tolerances — yes, cosine~1.0, diffs 1e-14..1e-17, eta blocks bit-identical (Sections 4-5)
- Analytic eta derivatives pass — yes (D=4 rigorous FD validation, Section 4; D=20 stationarity consistency check, Section 5)
- Checkpoint/configuration tests pass — yes (Section 4)
- At least the K=1 short outer shakedown produces a cold-verified feasible incumbent — yes, and so did K=2 (Section 6)
- Memory is operationally acceptable for one D=20 process — yes, peak 7.4 GB (Section 5)

No further generic edge-case testing is opened per the task's own instruction once these gates pass.

---

## 9. API examples

**Configuration:**
```julia
cfg = OriginZCConfig(distribution_restriction = :origin_specific_moments_zero_covariance,
                      K_mean = 2, K_pair = 2, power_target_layout = :origin_by_power)
K_mean, K_pair = originzc_resolve_K(cfg)
layout = originzc_make_layout(cfg, D)   # OriginByPowerLayout(D, 2, 2)
```

**Production launch (fresh run, explicit opt-in — separate function from the CM-family `run_cm_upper_checkpointed`, which is untouched):**
```julia
gp0 = frechet_benchmark_gp(ctx)
zfree0 = pivot_reduce(log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)), pe)
nu0 = [mean(ctx.U[:,o].^k) for k in 1:K_mean for o in 1:D]   # actual frozen draw moments
w0 = vcat(gp0, zfree0, log.(nu0))

run_originzc_upper_checkpointed(w0; W = 80000, delta = 1.0,
    distribution_restriction = :origin_specific_moments_zero_covariance,
    K_mean = 2, K_pair = 2, power_target_layout = :origin_by_power,
    maxtime_real = 600.0, ckpt_dir = "production_runs/originzc_k2", label = "originzc_k2_d1.0",
    cm_gradient_backend = :cplus)
```

**Resume:**
```julia
run_originzc_upper_checkpointed(nothing; resume_from = "production_runs/originzc_k2/originzc_k2_d1.0_latest.jls",
    distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 2, K_pair = 2,
    power_target_layout = :origin_by_power, maxtime_real = 600.0,
    ckpt_dir = "production_runs/originzc_k2", label = "originzc_k2_d1.0", cm_gradient_backend = :cplus)
```

A resume hard-refuses on any mismatch in `distribution_restriction`/`K_mean`/`K_pair`/`power_target_layout`/`origin_D`/draw checksum/`origin_moment_layout_version`.

---

## 10. Archive contents

See `origin_zc_k12_integration_archive_2026-07-23/` (built alongside this report) for: this report; the math note; a source diff (`source.diff`); D=4/D=20/shakedown raw logs; the checkpoint test output; a `MANIFEST.sha256` covering every file in the archive.
