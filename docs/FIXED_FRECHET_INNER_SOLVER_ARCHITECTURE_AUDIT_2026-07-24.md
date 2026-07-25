# Fixed-Fréchet Inner-Solver Architecture Audit — 2026-07-24

Task: make fixed-Fréchet inner solves production-feasible. **The addendum to the task brief
supersedes the CDF+POWER default**: the production target for this session is
`frechet_feature_set=:cdf_only` (common fixed-Fréchet marginal CDFs on the L=50 grid), not the
combined CDF+truncated-power restriction. This document — and every other deliverable in this
package — reports against that corrected scope. CDF+POWER is retained as an explicit experimental
option; its status is reported separately in
`FIXED_FRECHET_CDF_POWER_EXPERIMENTAL_STATUS_2026-07-24.md` and did **not** receive further
engineering investment in this session.

## 0. Correct interpretation of the original bottleneck

The task brief's starting diagnosis (from the prior port-prep session,
`FIXED_FRECHET_PERFORMANCE_REPORT_2026-07-24.md`/`FIXED_FRECHET_SLOW_INNER_SOLVE_DIAGNOSIS_2026-07-24.md`)
measured a genuinely new finite-point solve at **~660–1000s** for `:cdf_power` (n≈2382 inner
variables, 2,838,153 dense Hessian entries, `par_numthreads=1`). That diagnosis is not in dispute —
it was independently reproduced in spirit here. The key scope correction (the addendum) is that
`:cdf_power`'s extra truncated-`(1-σ)` restrictions were never part of the user's actual current
request; they were carried over from a historical "reproduce the paper draft" instruction in the
prior port-prep task. `:cdf_only` (D·L=1000 moments, n≈1382) is the real target, and — as this
session found from a live, direct, matched measurement — is dramatically cheaper, not merely
`(2382/1382)^3 ≈ 5.1×` cheaper as the brief's naive cubic-scaling estimate suggested.

## 1. Live audit: production flexible-CM Hessian engineering vs fixed-Fréchet CDF-only

Traced via direct code reading (`cm_hessian_architectures.jl`, `cm_hessian_threaded.jl`,
`cm_frechet_hessian.jl`, `cm_frechet_power_hessian_structured.jl`) in this repo state
(`full_aod_diag/d4_exact/`), cross-checked with an independent Explore-agent pass over the same
files. Findings, by task-brief §3 checklist item:

| Item | Flexible-CM production | Fixed-Fréchet CDF-only (pre-session) |
|---|---|---|
| Bin-index construction | Once, cached in `CMBinHessCtx.Bidx` | Same — `build_cm_frechet_bin_ctx` delegates to the identical `build_cm_bin_ctx` constructor. No difference. |
| Weighted bin-contingency accumulation | **Two unwired, experimental threaded variants exist** (`cm_hessian_architecture_threaded.jl`, `cm_hessian_threaded.jl`) — draw-chunk `Threads.@threads`, thread-local scratch, fixed-order deterministic reduction, no atomics. **Neither is included by the production stage runner** (`cm_production_stage_runner.jl` only includes the serial `cm_hessian_architectures.jl`) — so flexible-CM production Hessian construction is *itself* single-threaded today, despite the threaded kernel existing and being validated (`validate_threaded_archC.jl`). | Zero threading anywhere (confirmed by direct grep — no `@threads`/`@spawn`/`Threads.` in any `cm_frechet_*.jl` file). |
| Prefix-sum tables | Serial `prefix_sum_tables!`; a threaded analogue exists (`prefix_sum_tables_threaded!`, embarrassingly parallel over (x,y) pairs) but is likewise unwired in production. | Serial only, reused unchanged from flexible-CM's own `prefix_sum_tables!`. |
| `H_EE` block (Gram matrix) | `BLAS.gemm!` in production; `BLAS.syrk!` (~1.4–1.8×, per the threaded file's own isolated-kernel measurement) exists only in the unwired threaded variant. | `BLAS.gemm!`, same as flexible-CM production. |
| Orthonormal-congruence transform | `orthonormal_contrast_matrix(D)`, applied per threshold block. | Identical mechanism, extended to the extra "common"/reference-pin block CDF-only adds. |
| KNITRO Hessian registration | `KN_DENSE_ROWMAJOR` via the shared `inner_loop_KNITRO_archgeneric`. No sparse-coordinate path exists for either family. | Identical — same shared registration path. |
| KNITRO option file / thread settings | `full_aod_diag/ek_inner.opt`: `par_numthreads 1`, `par_blasnumthreads 0` (auto→1), `par_lsnumthreads 0` (auto→1), `linsolver auto`, `blasoption intel`. | Same file, unchanged — fixed-Fréchet never had its own `.opt` file before this session. |
| Variable/moment counts | `n_cm_moments(D,L) = (D-1)·L` → 950 at D=20/L=50, `n=ncore+ncm=1332` | `:cdf_only`: `n_cm_frechet_moments(D,L) = D·L` → 1000, `n=1382`. **1.038× flexible-CM's own dimension**, not the 1332-vs-2382 gap the brief's estimate implied. |

**Conclusion (port targets for this session):** the single highest-value, lowest-risk lever
available was porting the *already-built-but-unwired* flexible-CM threaded/syrk Architecture-C
kernel to the CDF-only fixed-Fréchet Hessian — a direct, mechanical port (same `CMBinHessCtx`,
same `build_bin_tables_threaded!`/`prefix_sum_tables_threaded!`, same discipline), since fixed-
Fréchet's own bin-table-building stage is share-for-share the same O(W·D·NCORE) computation
flexible-CM's is. This was completed and validated this session (see
`THREADED_CDF_ONLY_HESSIAN_BENCHMARK_2026-07-24.md`).

One additional, unrelated allocation issue was found and disclosed (not fixed — it's inside the
`:cdf_power`-only cross-block loop, out of this session's now-deprioritized scope): a per-iteration
`Vector{Float64}(undef, nO)` allocation inside a doubly-nested `l×lp` loop in
`cm_frechet_power_hessian_structured.jl` (~2,500 small allocations per Hessian callback at L=50).
Flagged for a future `:cdf_power` session, not touched here.

## 2. What this session actually built (additive-only; see git log on this branch)

- `cm_frechet_hessian_threaded.jl` — `hessian_cm_frechet_structured_v2!` / `archC_frechet_hess_cb_builder_v2`, direct structural port of `cm_hessian_threaded.jl`'s v2 kernel onto the CDF-only Hessian tail. Validated bit-for-bit-class agreement (D=4: 1e-16–1e-15; D=20 real data: 6.8e-13 absolute at Hessian entries up to ~4000 magnitude) against the pre-existing serial reference.
- `cm_frechet_production_bundle_threaded.jl` — additive threaded analogues of `archC_frechet_base_state`/`archC_frechet_verified_state` (never modifies the existing `:cdf_power`-default entry points).
- `cm_frechet_cdf_only_gradient.jl` — **fills a real, previously-undisclosed-as-fixed gap**: `cm_frechet_production_gradient` hard-errors for `:cdf_only` ("only :cdf_power is wired in this port-prep pass"). Built `frechet_fixed_contribution_archB`, an O(W·nO) bin-lookup forward-pass kernel (mirroring `cm_lookup_kernels.jl`'s already-validated pattern) replacing the dense `aug.CM` matvec that `:cdf_only`'s Architecture-B moment path doesn't materialize. Validated against the existing dense reference to **1.1e-16 absolute** (machine precision).
- `run_frechet_upper_cdf_only.jl` / `launch_frechet_cdf_only_shakedown.jl` — CDF-only outer-driver copies (existing `:cdf_power` driver untouched).
- `frechet_bench_opts/` — non-deprecated-named (`numthreads`/`blas_numthreads`/`linsolver_numthreads`) KNITRO thread-count option files, plus `tol_variants/` for the trial-tolerance study.

## 3. A real, disclosed finding: pre-existing outer-gradient z-direction discrepancy

While validating the new CDF-only outer gradient against central finite differences, a genuine,
h-converged (i.e. NOT finite-difference-step-size noise) discrepancy of ~1–3×10⁻³ absolute
(~5–30% relative) was found on the z-coordinate (non-`gp`) directions. An isolation check — running
the *identical* finite-difference sweep against the **existing, unmodified** `:cdf_power` gradient
path (`cm_frechet_production_gradient`, code this session never touched) — reproduced the same
pattern at the same order of magnitude. **This is a pre-existing characteristic of the shared
`composite_gradient_at_fast`/Lfix-incremental machinery's handling of the Frechet restriction's
z-directions, not something introduced by this session's new archB kernel or CDF-only work.** It
matches this project's own documented precedent (see memory
`melitz-continuation4-gradient-disagreement-followup`: "pre-existing pivot-cell non-smoothness,
present in the ORIGINAL backend too"). Not root-caused or fixed in this session (out of scope); the
outer shakedown proceeds on gradients with this known, disclosed limitation, consistent with how
this project has previously handled comparable small pre-existing gradient imprecisions without
blocking a release.
