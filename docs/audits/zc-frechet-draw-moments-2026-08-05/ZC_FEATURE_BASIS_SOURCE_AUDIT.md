# ZC feature basis source audit — 2026-08-05

Scope: correct the ZC/CM+ZC restriction feature basis from powers of the raw Exp(1) draw
(`U^k`) to powers of the Fréchet productivity draw (`z^k = U^{-μk}`), per the paper's stated
restriction `z_o(ω) = U_o(ω)^{-μ}`, `U_o(ω) ~ Exp(1)`.

## 1. Exponent mapping (verified, not assumed)

- `ctx.U` (`prepare_cc/genRands.jl::genExpRands!` → `transform_unit01_to_exp1!`) is the raw
  `Exp(1)` draw, `W x D`.
- The Fréchet productivity draw is `z = U .^ (-μ)`. The exponent is literally named `μ`
  (`θ[1]` in the structural parameter vector; `ctx.μHat` on every real-data context), **not**
  `theta` directly. `μ = 1/θ`, where `θ` is the gravity/trade elasticity (confirmed at
  `full_aod_diag/d4_exact/flexible_theta.jl:27`, `unrestricted_stage_runner.jl:143`
  `theta_star = 1.0 / ctx0.μHat`).
- The economic/gravity moment code already computes this transform correctly:
  `prepare_cc/createUDerivatives!.jl:14` (`U .= U .^ (-μHat)`, **gated on `θConstant==1`**),
  `moments/moments!.jl:82,131-132` (`AodPow`, `UPow`), `full_aod_diag/d4_exact/moments_fast.jl:30`.
  The ZC/meanZC builders (`cm_meanzc_moments.jl`, `cm_originzc_moments.jl`) never applied this
  transform — they consumed `ctx.U` (or `ctx.U.^k`) directly, which is the raw draw, not `z`.
- **Verified invariant, not an assumption**: production's `θConstant` is hardcoded to `0` in
  `full_aod_diag/ad_benchmark/setup_context.jl:19` (`AD_PARAMS`), the single parameter set every
  real-D20 context (`build_ad_context_real_d20`/`d20_real_setup`, used by all 5 families) merges
  into, and `d20_real_setup` exposes no `θConstant` override. So `createUDerivatives!`'s
  `U .= U.^(-μHat)` in-place mutation branch is unreachable from any real production entry
  point — `ctx.U` (`prep_output.U` at `master_prepare_cc.jl:288`, unpacked at
  `context_real_d20.jl:167`) is confirmed to remain the raw, untouched `Exp(1)` draw through the
  entire pipeline. (This was checked live mid-session after a direct question raised the
  possibility of double-transformation — see the baseline-moment validation test in
  section 11 of the task, which would have caught it empirically if wrong.)

## 2. Numeric values (real D=20 production data)

- `μHat ≈ 0.114212` (`θ_star ≈ 8.75566`, `docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md:18`).
- Production hardcodes `K_mean = K_pair = 1` (`campaign_cm_family_runner.jl:70-71`); the K=3
  campaign infrastructure (`CAMPAIGN_MEANZC_K`/`CAMPAIGN_ORIGINZC_K` env vars,
  `continuation_polish_run_fn.jl`) lives on the separate, unmerged
  `campaign/fullA-continuation-polish-2026-08-03` branch (not yet in `production/fullA-exact`).
- **Finiteness**: `E[z^k] = Γ(1-μk)` is finite for `μk < 1`. At `μ≈0.1142`: finite through
  `k=8` (`μ·8≈0.9137`), only diverging at `k=9` (`μ·9≈1.028`). K=1 (current production) and
  K=3 (in-progress campaign infra) are both far from this boundary (`μ·3≈0.343`). No scientific
  blocker.

## 3. Canonical feature builder (single shared choke point)

`build_raw_mean_pair_matrices`/`build_raw_mean_pair_matrix_levels` in
`full_aod_diag/d4_exact/cm_meanzc_moments.jl` is the **sole** place every ZC/meanZC path builds
its raw feature matrices — origin-ZC (FULL and REDUCED) and CM+ZC (FULL and REDUCED) all call
into it (or into `build_cm_meanzc_augmented_obj`/`build_originzc_augmented_obj`, which call it),
never reimplementing the power transform independently. Hessian blocks
(`zc_restriction_operator.jl`) and cold-verify code are pure consumers of its output
(`Zraw_all`/`Zpairraw_all`) — they never touch `ctx.U` directly, so fixing the builder fixes them
by construction.

This made the fix narrow: two new functions
(`frechet_productivity_from_exponential(U,μ)`, `frechet_power_feature(U,k,μ)`, both requiring
`μ::Float64` with no default — this repo's rule for any parameter that changes what economic
problem is being solved) plus three signature changes, all in `cm_meanzc_moments.jl`:

| Function | Old | New |
|---|---|---|
| `build_raw_mean_pair_matrices(U,k;want_pair)` | `Uk = k==1 ? U : U.^k` | `Zk = frechet_power_feature(U,k,μ)`; `μ` now required kwarg |
| `build_raw_mean_pair_matrix_levels(U,K_mean,K_pair)` | loops the above | same, `μ` now required kwarg |
| `nu_feasible_interval(U,k)` | bounds on `U.^k` | bounds on `frechet_power_feature(U,k,μ)`; `μ` now required kwarg |

## 4. Call-site map (every consumer, and whether it needed a change)

| # | Path | File:line | Old basis | Fix applied |
|---|---|---|---|---|
| 1 | FULL origin-ZC | `cm_originzc_moments.jl:234` `build_originzc_augmented_obj` | `U^k` (via shared builder) | ✅ passes `μ=ctx.μHat` |
| 2 | FULL CM+ZC | `cm_meanzc_moments.jl:485` `build_cm_meanzc_augmented_obj` | `U^k` (via shared builder) | ✅ passes `μ=ctx.μHat` |
| 3 | REDUCED origin-ZC | `prototype_worktree/.../profiled_originzc_family_adapter_2026-08-02.jl` → calls #1's function | inherited | ✅ inherited automatically (same function, no reduced-side duplicate) |
| 4 | REDUCED CM+ZC | `prototype_worktree/.../profiled_cmzc_family_adapter_2026-08-02.jl` → calls #2's function | inherited | ✅ inherited automatically |
| 5 | Fixed-nu wrappers | `wrap_moments_with_originzc`/CM+ZC analog | pure consumers of `Zraw_all` | ✅ inherited (no independent draw-power code) |
| 6a | Free-nu / eta_nu box bounds (CM+ZC, production) | `cm_meanzc_config.jl:104-108` `meanzc_default_nu_bounds` → `nu_feasible_interval` | `U^k` | ✅ fixed (passes `μ=ctx.μHat`) |
| 6b | Free-nu / eta_nu box bounds (origin-ZC, production) | `cm_originzc_config.jl:146-158` `originzc_default_nu_bounds` | independent `Uk = ctx.U .^ k` | ✅ fixed (now `frechet_power_feature(ctx.U,k,ctx.μHat)`) |
| 6c | Free-nu diagnostic scripts (~13 files, non-production gates/smokes) | e.g. `gate2_...jl:115`, `zc_centering_d20_gate_2026-07-28.jl:72`, `run_prodscale_full_vs_reduced_ab_2026-08-02.jl:109` (listed in full by the research pass) | independent `mean(@view (ctx.U.^k)[:,o])` for an ad hoc `nu0` reference | ⚠️ **not touched** — see §6 below |
| 7 | Verification/cold-verify | `cm_cold_verify.jl`, `originzc_cold_verify.jl` | inherited via production path | ✅ inherited automatically |
| 8 | Hessian blocks (H_ZZ/H_CZ/H_EZ) | `zc_restriction_operator.jl` | pure consumer of `Zraw_all`/`Zpairraw_all` | ✅ inherited automatically, no Hessian-side code touches `ctx.U` |
| 9 | eta_nu outer-gradient target derivative | `cm_meanzc_moments.jl:514-537` (`d_delta_dual_d_nu_vec`, `d_delta_dual_d_eta_nu_vec`) | formula-only (`-1`, `-2ν`), data-independent | ✅ no change needed — correct with either feature basis, only reads column layout, not `ctx.U` |
| 10 | Checkpoint/manifest hashing | `cm_checkpoint_fingerprint.jl` hashes raw `U`, not derived features; `origin_moment_layout_version`/`meanzc_K_mean` version the *layout*, not the *formula* | **gap**: no version tag distinguished "features from U^k" vs "features from z^k" before this fix | ✅ new manifest fields added (§8 below) |
| 11 | Test reference builders | `test_cm_meanzc_pure_moments.jl`, `test_cm_originzc_pure_moments.jl`, `test_cm_meanzc_d4_gates.jl` | independently asserted against `U^k`/`k!` (own reference, not a call into #1/#2) | ✅ updated (see below) |

## 5. Residual `U^k` sites deliberately NOT touched (scope boundary)

A repo-wide search found the `nu0 = mean(@view (ctx.U .^ k)[:,o])`-style ad hoc reference
pattern (and the near-identical `nu0vec(K) = [factorial(k) for k in 1:K]` — the theoretical
`E[U^k]=k!` moment of the *raw* draw, mislabeled as `E[z^k]`) duplicated across roughly
**36 files**, all one-off gate/diagnostic/benchmark scripts under `full_aod_diag/d4_exact/`
(e.g. `diag_hzz_backend_benchmark_2026-07-28.jl`, `final_four_family_gate_2026-07-28.jl`,
`gate2/3/5_*.jl`, `phase2_originzc_dual_bank_only.jl`, `production_all_hessian_audit_harness_2026-08-02.jl`,
`zc_centering_d20_gate_2026-07-28.jl`, `smoke_delta1_originzc.jl`, and ~28 more `test_*`/`gate*`
files listed by the research pass).

**These were deliberately not hand-edited.** Rewriting ~36 diagnostic scripts is scope creep
beyond "correct the ZC restriction" — none of them are on the production/REDUCED critical path
(§4 confirms every production and REDUCED entry point routes through the single fixed builder).
Running any of them as-is post-fix will show an expected moment mismatch (their own hardcoded
`nu0` no longer matches the corrected `Delta_dual`/moment values) — **this is the correct
signal, not a regression**: those scripts encode the old, wrong `E[U^k]=k!` assumption and
should be treated as scientifically obsolete reference values, exactly like the campaign
checkpoints marked `SCIENTIFICALLY_OBSOLETE_ZC_BASIS_EXPONENTIAL`. Two of the most-central ones
(`test_cm_meanzc_d4_gates.jl`, `test_cm_originzc_pure_moments.jl`) were fixed as part of this
task since they are this task's own D4 exhaustive gate evidence (task §10); the rest are flagged
here for a future, separate cleanup pass.

## 6. Derivative implications (task §6)

- The corrected `Q`/`Z` operator is a fixed function of `(ctx.U, μHat)` only — `μHat` is not an
  outer coordinate on any ZC-family production or REDUCED path (verified: `flexible_theta.jl`'s
  free-θ/free-μ experimental machinery never calls `build_originzc_augmented_obj`/
  `build_cm_meanzc_augmented_obj`, confirmed by direct grep — no shared builder). So `∂Q/∂(A,gp,η_ν) = 0`
  exactly, on every reachable path; no derivative formula needed updating, only the *values*
  flowing into the existing formulas (`d_delta_dual_d_nu_vec` etc., item 9 above) change, because
  `Zraw_all` itself changed.
- Hessian blocks (`H_ZZ`/`H_CZ`/`H_EZ`) are structurally unaffected (item 8) — same reasoning.

## 7. Direct feature tests and D4 gates (task §9-10)

See `docs/audits/zc-frechet-draw-moments-2026-08-05/MASTER.md` for the executed test results.
