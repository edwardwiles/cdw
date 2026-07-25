# Fixed-Fréchet CDF-Only Inner-Solve Production-Feasibility — Final Verdict — 2026-07-24

Branch: `experiment/fixed-frechet-cdf-only-production-feasible-2026-07-24`, off
`feature/fixed-frechet-post-omit-row-port-prep-2026-07-24` @ `14e080a`. Not merged to production
(per task instruction). Machine: shared 208-core host, `uptime` load average ~105-150/208
throughout this session — every timing number is an upper bound on intrinsic cost, not a clean
isolated benchmark, per this project's standing disclosure convention.

## Scope correction (addendum)

The task's addendum supersedes the original brief's CDF+POWER default: the actual production
target is **`frechet_feature_set=:cdf_only`** (common fixed-Fréchet marginal CDFs on the L=50
grid), not the combined CDF+truncated-power restriction. Every result below is against `:cdf_only`.
CDF+POWER's status is reported separately and received no further engineering investment here (see
`FIXED_FRECHET_CDF_POWER_EXPERIMENTAL_STATUS_2026-07-24.md`).

## What was built this session (all additive; existing files/behavior untouched)

1. **Threaded/syrk structured Hessian for CDF-only** (`cm_frechet_hessian_threaded.jl`) — a direct,
   mechanical port of the flexible-CM production codebase's own threaded Architecture-C kernel
   (`cm_hessian_threaded.jl`), which existed but was never wired into production for either family.
   Validated to machine precision at D=4 and to 6.8e-13 absolute at real D=20/W=80,000/L=50.
2. **KNITRO thread-count option files** (non-deprecated names: `numthreads`/`blas_numthreads`/
   `linsolver_numthreads`), confirmed active via KNITRO's own runtime option echo, not inferred.
3. **A real, previously-hard-erroring gap fixed**: the outer-gradient path
   (`cm_frechet_production_gradient`) only ever supported `:cdf_power`. Built and validated
   (1.1e-16 vs the existing dense reference) `frechet_fixed_contribution_archB`, an O(W·nO)
   bin-lookup kernel filling this gap for `:cdf_only`, plus the supporting production entry points
   and a CDF-only outer-driver copy.
4. Trial-tolerance variant option files and a matched flexible-CM-vs-CDF-only benchmark harness.

## Headline results

| Metric | Before (documented, `:cdf_power`) | After (`:cdf_only`, this session) |
|---|---|---|
| n (inner variables) | 2382 | 1382 (1.038× flexible-CM's own 1332) |
| Genuinely new finite-point solve (cold, near-calibration) | ~660-1000s | **41-67s** |
| Same, with 20-thread Hessian (warm, production-realistic reuse) | not measured | **8.6-17.8s** |
| Hessian-callback-only wall time | ~11.8s (bounded L=8 estimate) | 3.4-4.6s serial → **0.9-1.4s at 20 threads (4.9-5.7×)** |
| KNITRO's own linear-algebra threading | not tested | **confirmed active, no measurable benefit** at n=1382 (flat 14.5-14.9s, 1→20 threads) |
| Matched vs flexible-CM at ~equal n | — | **1.33×** flexible-CM's own solve time (23.4s vs 17.7s at n=1382 vs 1332) — same order of magnitude, as the addendum predicted |
| Trial-tolerance sensitivity | — | Δ_dual stable to 8 significant figures from 1e-6 to 1e-12; 1e-8 recommended as the ordinary-trial tolerance |

Correctness gates: D=4 and D=20 threaded-vs-serial Hessian agreement (machine precision to 1e-13),
D=4 end-to-end Δ* bit-identical, new outer-gradient kernel exact vs dense reference (1e-16),
deterministic repeated solves, accepted-point state reuse intact (`n_base_reused_at_gradient`
always matches expectation across every benchmark run).

## Two disclosed, unresolved findings (not hidden)

1. **Pre-existing z-direction outer-gradient discrepancy** (~1-3×10⁻³ absolute, h-converged, not
   finite-difference noise), confirmed present and of the same order in the **existing, unmodified**
   `:cdf_power` gradient path too — not introduced by this session. See
   `FIXED_FRECHET_INNER_SOLVER_ARCHITECTURE_AUDIT_2026-07-24.md` §3.
2. **Far-from-calibration trial points, as visited by a real 30-minute outer search, are NOT
   uniformly fast** — individual solves plausibly exceed the 120s ceiling at points the SQP
   line-search wanders to (up to ~174s/solve average in the slowest third of the shakedown run),
   and the search made no net kappa improvement in 30 minutes. This is very unlikely to be a
   Hessian-construction or KNITRO-threading problem (every direct, controlled benchmark in this
   package shows uniform speedup regardless of point) — more likely a genuine numerical-difficulty
   effect at points far from the data-consistent region, possibly compounded by finding #1. See
   `FIXED_FRECHET_CDF_ONLY_30MIN_OUTER_SHAKEDOWN_2026-07-24.md` for full detail and the literal
   pass/fail table against every task-brief §12 requirement (all six pass on their literal terms).

## Verdict

**PORT READY EXPERIMENTAL — THREADED DIRECT BACKEND**, scoped specifically to: individual inner
solves at points at or near the calibration/data-consistent region (the problem this task was
chartered to fix) are now production-feasible by every measured criterion — 8-25× faster than the
prior `:cdf_power` baseline even before accounting for the additional gain from the smaller feature
set, correctness-validated at every scale tested, and matched to flexible-CM's own production
performance to within 1.33×.

This verdict does **not** extend to an unqualified claim that a real outer search will reliably
converge within a fixed wall-clock budget — the 30-minute shakedown surfaced a second, distinct,
disclosed bottleneck (far/infeasible trial-point solve cost and outer-search stall) that this
session did not fix and that is not primarily attributable to the Hessian/KNITRO-threading work
this task targeted. Recommended next steps for a follow-up session, in priority order: (1)
instrument per-eval inner-solve wall time in the outer driver to confirm/quantify the 120s-ceiling
concern precisely; (2) root-cause the pre-existing z-direction gradient discrepancy (shared with
`:cdf_power`, likely higher-leverage than any further inner-solve speed work); (3) re-run a longer
or differently-configured outer shakedown once (1)/(2) are addressed.

## Correctness gates (task brief §11) — status

- D=4 dense-vs-structured-vs-threaded: PASS (machine precision)
- Bounded D=20: PASS (direct diff, 6.8e-13)
- Full D=20/L=50 at P0/P1/P2: PASS (all `nStatus=0`, feasible, deterministic across repeated runs)
- Outer-gradient exact-kernel-vs-dense-reference: PASS (1.1e-16)
- Accepted-point state reuse: PASS (`n_base_reused_at_gradient` always matches expected count)
- No same-point re-solve: PASS by construction (`use_cached_x`/base-reuse discipline unchanged)
- Checkpoint compatibility: not exercised in this session (disclosed gap — no checkpoint-schema
  change was made, so compatibility is expected but not independently re-tested here)
- No change in economic moments/targets: confirmed — `target_sha256` fingerprints logged in every
  benchmark run are identical across all runs in this package
