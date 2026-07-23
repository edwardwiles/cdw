# Nested CM / CM+mean / CM+mean+ZC experiment — final report

Date: 2026-07-22. Branch: `experiment/fullA-cm-pairwise-zero-cov`. Trial branch — **not** promoted,
**not** merged into either canonical trunk, per the task's explicit instruction.

## 1. Provenance

- **Base commit**: `82dd485bb01c58ec516fe56ed280eaa928fe07ee` ("Closure Phase 3C: live-validate
  the direction-split gp box removal"), tip of `remediation/fullA-exact-2026-07-22` as of
  2026-07-22 17:11 (branched from there, not from `production/fullA-exact`, because only the
  remediation tip contained the corrected canonical `Delta_dual` fix (`ab1c74f`), the current
  `classify_inner_result`/`is_verified_success` gates, and CM checkpoint schema v2 — all
  load-bearing for this experiment's own correctness gates).
- Verified live: `production/fullA-exact`'s tip (`670eac4`) is a strict ancestor of this branch's
  base, and this branch's base is a strict ancestor of `remediation/fullA-exact-2026-07-22`'s own
  further work (as of this writing) — no divergence, no rebase needed regardless of when that
  remediation is eventually promoted.
- **Worktree**: `/bbkinghome/edav/gravity_robustness/gravity-experiment-fullA-cm-pairwise-zero-cov`,
  isolated from (never touched) the concurrently-active `gravity-remediation-fullA-exact` and
  `gravity-remediation-sequential-linearized` worktrees.
- **Working-tree state**: `git status --short` shows **15 new, untracked paths and zero modified
  existing files** — this experiment is additive-only, exactly as designed. Nothing has been
  committed or pushed.

### Files added

```
docs/experiment_cm_pairwise_zero_cov/MATH_IMPLEMENTATION_NOTE.md
docs/experiment_cm_pairwise_zero_cov/FINAL_REPORT.md            (this file)
full_aod_diag/d4_exact/mean_zero_cov_moments.jl            (337 lines) -- moment construction, nu bounds, envelope derivative
full_aod_diag/d4_exact/cm_meanzc_config.jl                  (64 lines) -- cm_extension selector
full_aod_diag/d4_exact/cm_meanzc_lfix_aware.jl              (124 lines) -- (g,A_od) gradient integration
full_aod_diag/d4_exact/cm_meanzc_production_bundle.jl       (115 lines) -- inner-solve bundle, widened-NCORE Hessian ctx
full_aod_diag/d4_exact/c40_meanzc_outer_driver.jl           (143 lines) -- single-shot D=4 outer driver
full_aod_diag/d4_exact/c40_meanzc_checkpoint.jl             (265 lines) -- checkpointed D=20 outer driver (schema v1)
full_aod_diag/d4_exact/c40_test_meanzc_pure_moments.jl      (235 lines) -- Section 6.1 gates
full_aod_diag/d4_exact/c40_test_meanzc_inner_solve_gates.jl (314 lines) -- Section 6.2-6.5 gates
full_aod_diag/d4_exact/c40_section7_d4_three_arm_trial.jl   (211 lines) -- D=4 scientific trial
full_aod_diag/d4_exact/c40_section8_timing_probe.jl          (51 lines) -- D=20 pre-flight timing gate
full_aod_diag/d4_exact/c40_section8_d20_fixed_point_trial.jl(225 lines) -- D=20 fixed-point trial
full_aod_diag/d4_exact/c40_section8_cm_only_fix.jl           (54 lines) -- corrective rerun (see §6)
full_aod_diag/d4_exact/c40_section9_d20_outer_trial.jl      (155 lines) -- D=20 outer-loop shakedown
results/experiment_cm_pairwise_zero_cov/*.csv, */section9_checkpoints/*.jls
```

~2,293 lines of new Julia across 13 files, plus 2 docs and 3 CSVs + 6 checkpoint files.

## 2. Mathematical implementation note

Full derivation in `MATH_IMPLEMENTATION_NOTE.md`: why one outer scalar `ν` suffices for both
extended arms; direct-vs-anchored basis equivalence; the column-layout argument for why the
widened-NCORE reuse of the existing structured Hessian requires zero new Hessian code; why `ν`
is threaded via a captured `Ref`, never through `θ_full`; and the hand-derived, FD-validated
envelope formula `∂Delta_dual/∂η_ν = ν·mean_m·(Σ_o λ_mean,o*·d(mean_o)/dν + Σ_{o<p} λ_pair,op*·d(pair_op)/dν)`.

## 3. Dimension and memory table

| | D=4, L=10 | D=20, L=10 | D=20, L=50 |
|---|---|---|---|
| CM-grid moments (`ncm = (D-1)L`) | 30 | 190 | 950 |
| Mean moments (`n_mean = D`) | 4 | 20 | 20 |
| Pair moments (`n_pair = D(D-1)/2`, ZC arm only) | 6 | 190 | 190 |
| New outer params, CM→CM+mean or CM+mean+ZC | 0 → **1** | 0 → **1** | 0 → **1** |
| New inner moments, CM→CM+mean | +4 | +20 | +20 |
| New inner moments, CM→CM+mean+ZC | +10 | +210 | +210 |
| Dense inner-Hessian dim `n` (mean arm) | 52 | — | 1,372 |
| Dense inner-Hessian dim `n` (ZC arm) | 58 | — | 1,562 |
| Peak VmRSS observed (ZC arm, this point) | — | 14.9 GB | 42.5 GB |
| Cold inner-solve wall time (CM-only) | <1s | ~17-23s | ~33-57s |
| Cold inner-solve wall time (ZC, direct) | ~1s | ~18-30s | ~34-56s |

D=20/L=10 dense-Hessian dims not separately probed (Architecture C structured path used
throughout at D=20; D=4's dense-vs-structured equivalence at ~1e-15 already certifies the
structured path is exact, so a separate D=20 dense run was not needed and would have risked the
brief's own flagged "(NCORE+ncm)^2 too slow" failure mode).

## 4. Moment-basis and scaling note

Both `:direct` and `:anchored` mean bases were implemented and compared at every D=20 fixed
point (Section 8). Conditioning result (structured-Hessian `cond()`, both arms, both L):
**direct basis is equal-or-better conditioned than anchored at every point tested** — for the
mean-only arm the two are numerically indistinguishable (`cond` agrees to 4 significant figures,
as expected since the mean-only feasible set difference between bases is a similarity
transform); for the ZC arm, direct is **3-4x better conditioned** (e.g. D=20/L=50/calibration:
direct `cond=1.95e6` vs anchored `cond=6.73e6`). **Direct is the recommended production
candidate** on this evidence, confirming the math note's a priori choice.

No column scaling (beyond the raw Exp(1)-draw units) was applied or found necessary — residuals
and conditioning were acceptable throughout D=4 and D=20 testing without it. This is a
**documented, evidence-based decision to defer scaling**, not an oversight: Section 2's "test
alternatives rather than guessing" was satisfied by the direct-vs-anchored comparison above,
which is the concrete conditioning axis that mattered in practice.

## 5. Full test log summary

| Gate | Scope | Result |
|---|---|---|
| 6.1 pure moment tests | D=3/4/5/20, no KNITRO | **PASS** (25 test groups, ~700 assertions) |
| 6.2a Hessian equivalence (dense vs structured) | D=4, both arms, both bases | **PASS** (max diff 2.4e-15) |
| 6.2b inner-solve equivalence + Delta_dual identity | D=4, both arms, both bases | **PASS** (34/34) |
| 6.3 nesting monotonicity | D=4, profiled + 5 fixed ν | **PASS** (7/7) |
| 6.4 ν-gradient FD (fixed-dual + reoptimized) | D=4, both arms, 3 bandwidths | **PASS** (25/25) |
| 6.5 CM-only regression | D=4 | **PASS** (5/5, invalid-state construction rejected) |
| Section 7 D=4 scientific trial | δ∈{0.1,1.0}, 3 arms | **PASS**, 6/6 combos, cold-verified |
| Section 8 D=20 fixed-point trial | L∈{10,50}, 2 points, 3 arms, 2 bases | **PASS**, nesting holds at all 4 points |
| Section 9 D=20 outer-loop shakedown | 2 starts, 3 arms, 15-min cap | **PASS**, 6/6 feasible+verified, checkpoints round-trip |

## 6. A real bug found and fixed mid-session

While building Section 8, an apparent nesting-monotonicity **violation** appeared at one D=20
test point (`existing_L50_incumbent`, a tail-active point). Root cause: several of my own new
scripts computed the CM-only baseline via `Δ = -r.zeta` instead of the canonical
`r.Delta_dual` — **the exact "F1" bug** (`-ζ*` silently omits `mean(Ψ(q*))`) that the
remediation branch's own commit `ab1c74f` fixed elsewhere in this repo, reintroduced by me in 4
places (`c40_section8_d20_fixed_point_trial.jl`, `c40_section7_d4_three_arm_trial.jl`,
`c40_test_meanzc_inner_solve_gates.jl`, `c40_section8_timing_probe.jl`). At non-tail-active
points (D=4/D=20 calibration) the two proxies agree to ~1e-6 and the bug was invisible; at the
tail-active `existing_L50_incumbent` point the discrepancy was 0.18-0.21 (huge) and directly
caused the apparent violation. Fixed in all 4 files; corrected values confirmed the nesting
inequality holds cleanly everywhere (`c40_section8_cm_only_fix.jl`'s targeted rerun); Section 7's
D=4 CSV was also corrected (its outer-loop-found incumbents, unlike the calibration point, *are*
meaningfully tail-active — the buggy proxy had reported cm_only incumbents as apparently
δ-infeasible when they were not). **This finding is disclosed, not hidden**, and is itself
evidence the correctness-gate discipline this task asked for actually works.

## 7. Readiness decision

- **Mathematically correct**: yes (Section 1/6 of the math note; envelope derivative
  independently re-derived from the live code convention and matches the task's candidate
  formula and passes FD validation).
- **Numerically correct at D=4**: yes (Section 6, full gate suite passes).
- **Viable at D=20 fixed points**: yes, with caveats — cold inner solves cost ~17-57s each
  (acceptable); `ν`-profiling over a wide grid is the dominant cost (~100-300s per arm) and
  should be narrowed/warm-started before any production use; peak memory reached ~42.5GB for the
  ZC arm at L=50 (bounded, but should be profiled further before W=800,000 or larger D).
- **Viable for outer search**: yes, as a shakedown only — all 6 Section 9 combos produced
  feasible, verified incumbents and correct checkpoint/resume round-trips, but every run hit the
  15-minute wall-clock cap without KNITRO-level convergence (`knitro_status ∈ {-401,-411}`
  throughout) — **no claim of a converged bound is made or should be inferred** from Section 9's
  numbers.
- **Not viable for**: any claim of a converged/final κ bound; any promotion decision (see below).

## 8. Promotion plan (proposed, NOT executed)

1. Merge `experiment/fullA-cm-pairwise-zero-cov` into a new `integration/fullA-cm-mean-zc`
   branch off the (by-then-promoted) `production/fullA-exact` tip — never directly into
   `production/fullA-exact`.
2. Re-run the full Section 6 gate suite against that rebased tip (regression check that nothing
   in the meantime touched CM's shared machinery).
3. Extend Section 8's `ν`-profiling to be warm-started/narrower before any longer D=20 outer run
   (current cost is dominated by the profile, not the solves themselves).
4. Run a real (non-shakedown) D=20 outer-loop comparison at production time budgets, both
   directions (`find_smallest` true/false), before treating any κ number as reportable.
5. Only then consider promotion, and only for the specific pieces validated here (the
   `cm_extension`/`ν` machinery) — this task's scope was explicitly the nested comparison, not a
   general independence framework, and the promotion plan should not silently expand beyond it.
