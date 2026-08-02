# Phase 9 gate report: inner-solve truth gates, all 5 families (production dims)

Branch: `integration/profiled-all-five-production-closeout-2026-08-02`
Base commit for this phase: `41ad62b` (Merge Phase 2b/6/11).
Commits added this session: `e14d8ac`, `50a4dad`, `d80fc67`, `9bf2021` (see `git log`).

`PRODUCTION_DEFAULT_CHANGED = false`, `CAMPAIGN_LAUNCHED = false`, no push to any remote, no
reserved file touched (git diff --stat against the reserved list is empty throughout).

## D4 gates -- ALL PASS (8/8 confirm-runs, 0 failures)

| Family | operator FG vs FD | Hessian vs FD | Hessian vs dense-truth G'Diag(S)G |
|---|---|---|---|
| flexible_CM | PASS (existing) | PASS (existing, err~9e-15) | PASS (**new**, err~1e-14) |
| common_Frechet | PASS (existing) | PASS (existing, err~9e-15) | PASS (**new**, err~2e-14) |
| origin-ZC | PASS (existing) | PASS (existing) | PASS (existing, symmetry-audit PART B) |
| CM+ZC | PASS (existing) | PASS (existing) | PASS (existing, symmetry-audit PART A) |

New files: `test_flexcm_hessian_dense_truth_audit_2026-08-02.jl`,
`test_frechet_hessian_dense_truth_audit_2026-08-02.jl` (4 random points each, machine-precision
agreement with both ForwardDiff and an independent `G'*Diag(S)*G` dense-truth construction).

## D20/W20k gates -- ALL 5 FAMILIES PASS

| Family | real KNITRO solve | operator verification | recover-then-resolve |
|---|---|---|---|
| origin-ZC | nStatus=0 | kkt=1.31e-13 | Delta_primal diff 4.31e-08, LFD diff 4.10e-05 |
| CM+ZC | nStatus=0 | kkt(via brute-force)=n/a, Delta_dual/primal agree to 1e-10 | Delta_primal diff 4.43e-08, LFD diff 4.17e-05 |
| flexible_CM | nStatus=0 | kkt=1.63e-13 | Delta_primal diff 4.63e-08, LFD diff 6.38e-05 |
| common_Frechet | nStatus=0 | kkt=1.32e-13 | Delta_primal diff 4.48e-08, LFD diff 6.53e-05 |
| unrestricted | nStatus=0 | max\|dr\|=7.1e-15, max\|dg\|=1.57e-12 | n/a (no restriction to recover) |

New files: `test_flexcm_recover_resolve_d20_2026-08-02.jl`, `test_frechet_recover_resolve_d20_2026-08-02.jl`,
`test_unrestricted_operator_verification_d20_configurable_w_2026-08-02.jl`. origin-ZC/CM+ZC gates
already existed (`test_zc_lane_{originzc,cmzc}_recover_resolve_d20_2026-08-02.jl`) and re-ran
unchanged.

## W80k gates -- ALL 5 FAMILIES PASS

| Family | nStatus | kkt_resid (reduced) | Delta_primal recover-resolve diff | LFD diff |
|---|---|---|---|---|
| origin-ZC | 0 | -- | 4.87e-09 | 4.21e-05 |
| CM+ZC | 0 | 1.26e-16 | 4.63e-09 | 4.15e-05 |
| flexible_CM | 0 | 3.34e-13 | 3.60e-09 | 6.43e-05 |
| common_Frechet | 0 | 6.29e-13 | 5.08e-09 | 6.42e-05 |
| unrestricted | -- | max\|dr\|=9.4e-16, max\|dg\|=2.45e-12 | n/a | n/a |

Same 5 gate scripts as D20/W20k, re-run with `*_D20_W=80000` (or the existing canonical
`test_operator_verification_unrestricted.jl d20`, which is itself hardcoded to W=80,000).

## W100k production-dims genuine-cold solve (D=20, Ddest=19, L=50, K_mean=3, K_pair=3, W=100,000)

**The single most important deliverable of this task.** Each run is a completely fresh `julia`
process (new PID, no warm-start reuse, `obj.x` never touched from a prior process) that: builds the
real D=20 context at W=100,000 from scratch, builds the reduced/zero-dense family objects at
PRODUCTION dimensions, and performs exactly one cold KNITRO solve.

| Family | Status | nStatus | wall (solve) | wall (total) | Dense-G materializations at solve time |
|---|---|---|---|---|---|
| unrestricted | **DONE, PASS** | 0 | 5.17s | 92.3s | 0 |
| flexible_CM | **DONE, PASS** | 0 | 14.4s | 126.5s | 0 (econ), 0 (CM) |
| common_Frechet | **DONE, PASS** | 0 | 19.5s | 146.1s | 0 (econ), 0 (CM) |
| origin-ZC | **IN PROGRESS** at report time (PID 2552725, ~10min CPU, steadily progressing, not hung) | -- | -- | -- | -- |
| CM+ZC | **IN PROGRESS** at report time (PID 2547114, ~14min CPU, steadily progressing, not hung) | -- | -- | -- | -- |

Detail for the 3 completed:
```
unrestricted: nStatus=0  Delta_dual=Delta_primal=0.001915624515  n_fg=6  n_hess=5  dense_econ=0
flexible_CM:  nStatus=0  zeta*=-0.0070649514  kkt_resid(verify)=6.596e-13  dense_econ=0 dense_CM=0
common_Frechet: nStatus=0  zeta*=-0.0132281932 kkt_resid(verify)=7.138e-13  dense_econ=0 dense_CM=0
```
origin-ZC/CM+ZC were still executing their cold KNITRO solve when this report was finalized; logs
so far show `context build`, `layout build`, `reduced object/octx build` completed cleanly and
"Starting COLD KNITRO inner solve ..." printed, with steady CPU consumption (verified live via
`ps`, no error/stack trace in the log, no stall). Whoever picks this up next should check
`/tmp/.../scratchpad/logs/w100k_originzc_v3.log` and `w100k_cmzc_v2.log` (or their process's
successor) for the final `SOLVE:`/`VERIFY:`/`RESULT:` lines.

## Real bugs found + fixed this session

1. **`build_cm_augmented_obj_archB`/`build_cm_frechet_augmented_obj_archB` do not accept a
   `moment_representation` kwarg** (only the ZC-family builders do) -- an initial version of the
   flexCM/Frechet D20 gates passed it anyway, causing a `MethodError`. Fixed by dropping the kwarg
   for these two builders' "FULL" object construction.

2. **`verify_inner_solution_operator_cm!`/`_cm_frechet!` hardcode `ncore1 = cf.oci-1` as the
   economic-block width** -- correct only on the FULL/dense path; on the REDUCED path the economic
   block is `layout.total_reduced_economic_moments` (a smaller, different number -- `cf.oci-1`
   always describes the full economic moment space regardless of family/layout). Calling the
   generic verifier on a REDUCED solve throws `length(lambda) != expected`. Fixed (in the D20 gates
   and the W100k cold-solve drivers) by switching to the same direct `brute_force_verify`
   (`obj.moments!`/`Psi!`/`dPsi!`) the CM+ZC/origin-ZC gates already used successfully for exactly
   this reason.

3. **`NO_DENSE_G_COUNTERS[]` returns the live mutable counters struct, not a snapshot.** Re-reading
   its fields *after* `brute_force_verify` runs (which deliberately calls the dense `obj.moments!`
   for an independent cross-check) falsely attributes verify's own intentional dense recompute to
   the SOLVE step, producing a false FAIL even when the solve itself was genuinely zero-dense
   (observed live for flexible-CM: printed `dense_economic_G_materializations=0` immediately after
   solve, then failed the final check because the same live struct had since been mutated). Fixed
   by snapshotting the counter values as plain integers immediately after solve, in all three
   affected W100k cold-solve drivers (flexCM, common-Frechet, CM+ZC).

4. **`OriginByPowerLayout`'s `target_index(o,k) = (k-1)*D+o` requires `nu_full` to have length
   `K_mean*D`, not `D`.** Every prior origin-ZC D20 gate used `K_mean=1` (where `K_mean*D == D`
   coincidentally), masking this. At this task's mandated `K_mean=3`, `fill(1.0, D)` caused a
   `BoundsError` inside `refresh_zc_targets!`. Fixed to `fill(1.0, K_MEAN*D)`.

5. Missing include: `profiled_reduced_originzc_lookup_kernels_2026-08-02.jl`/
   `profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl` both require
   `profiled_reduced_lookup_kernels_2026-08-02.jl` included first (an `isdefined` guard on
   `economic_forward_into_arg0_reduced!`); the initial W100k origin-ZC/CM+ZC drivers omitted it.

## Files added (all additive, no reserved file touched)

D4: `test_flexcm_hessian_dense_truth_audit_2026-08-02.jl`, `test_frechet_hessian_dense_truth_audit_2026-08-02.jl`
D20/W80k: `test_flexcm_recover_resolve_d20_2026-08-02.jl`, `test_frechet_recover_resolve_d20_2026-08-02.jl`,
`test_unrestricted_operator_verification_d20_configurable_w_2026-08-02.jl`
W100k production-dims: `run_coldsolve_{flexcm,frechet,originzc,cmzc,unrestricted}_w100k_2026-08-02.jl`

## Commits (this session)
- `e14d8ac` D4 dense-truth Hessian audit (flexCM + common-Frechet)
- `50a4dad` unrestricted D20 operator verification at configurable W
- `d80fc67` flexCM/Frechet D20/W20k combined gate + real verifier bug fix
- `9bf2021` W80k all-5-families PASS + 2 real W100k cold-solve bugs fixed
