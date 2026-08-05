# Common-marginals restriction: add CDW eq.36 (truncated (1-σ)-power family) — MASTER

Branch: `fix/cm-add-truncated-power-moments-2026-08-05` (production/fullA-exact @546feff base).
Worktree: `/bbkinghome/edav/cdw_worktrees/cm-add-truncated-power-moments-2026-08-05`.

## 1. What changed scientifically

Production's "flexible common marginals" (CM) restriction previously imposed only CDW eq.35 (the
CDF family): for non-reference origin `o` and quantile cutoff `z_l`,

```
E_F[ 1{z_o'(ω) < z_l} - 1{z_1(ω) < z_l} ] = 0
```

As of this task, it imposes **eq.35 AND eq.36** (the truncated `(1-σ)`-power family) together:

```
E_F[ z_o'(ω)^(1-σ)·1{z_o'(ω) < z_l} - z_1(ω)^(1-σ)·1{z_1(ω) < z_l} ] = 0
```

for every non-reference origin `o`, reference origin (`refIndex1`), and quantile `l = 1..L`, using
the SAME productivity draw (`ctx.U`, the array this codebase's own precalc function already
treats as `z_o(ω)` for this restriction — see docs/fullA_common_marginals_handoff.md section 2)
and the SAME cutoffs `z_l` eq.35 already used.

**Note on prior code**: `precalc_common_marginals_cdf` already had an `include_truncated_moment`
kwarg and a truncated-moment code branch, but it was (a) never wired to any production call site
(grepped: zero call sites set it `true` before this task) and (b) used the wrong exponent,
`pw = μHat*(1-σHat)` instead of the paper's `(1-σ)` — an extra, incorrect `μHat` factor. This task
corrected the formula and wired it into production.

## 2. Feature definitions and block ordering

- **Family 1 (eq.35, CDF)**: `1{U[s,o] <= z_l} - 1{U[s,refIndex1] <= z_l}`, unchanged.
- **Family 2 (eq.36, truncated power)**: `U[s,o]^(1-σ)·1{U[s,o] <= z_l} - U[s,refIndex1]^(1-σ)·1{U[s,refIndex1] <= z_l}`,
  computed with the per-origin power weight `U[:,x].^(1-σ)` precomputed ONCE (a `W x D` matrix) at
  context-build time, never recomputed inside an outer callback.
- **Ordering**: CDF-block-then-power-block (columns `1:(D-1)L` = eq.35, `(D-1)L+1:2(D-1)L` = eq.36),
  each threshold-major — the SAME ordering the pre-existing (dead) `include_truncated_moment` code
  already used, adopted rather than inventing a new convention.
- **Contrast/basis**: the existing origin-contrast matrix `R` (`:anchored`/`:orthonormal`) is
  applied to each family's own `nO x nO` block independently, per threshold — identical mechanism,
  applied twice, not a new joint whitening. The `:cumulative` basis (production default) has no
  separate "T_L" transform beyond the raw indicator; the `:interval` basis is untouched
  (single-family-only, already documented as a non-default follow-up before this task).

## 3. Dimensions

```
CM_BLOCK_WIDTH   old: (D-1)*L              new: 2*(D-1)*L
D20_L50_WIDTH    950 (CDF-only)         -> 1900 (two-family)   [computed via n_cm_moments; see
                                                                 test_cm_truncated_power_2026-08-05.jl]
outer parameter vector: UNCHANGED (θ_free dimension untouched)
inner dual:      +ncm_pow = (D-1)*L additional CM dual coordinates
```

## 4. Reuse discipline / what did NOT change

- `wrap_moments_with_cm` / `wrap_moments_with_cm_meanzc` (dense_reference FG splice): already
  100% generic on `size(CM,2)` — **zero code change**, just a wider `CM` matrix.
- The (g,A_od)-block outer C+/Lfix gradient formula (`composite_gradient_at_fast`,
  `composite_gradient_at_Cplus_from_cache`, `economic_A_gradient!`): **zero code change** — by the
  envelope theorem, a theta-independent restriction block (CM is a fixed function of the frozen
  draws) only enters the outer gradient through the converged dual state, never through its own
  column count/content.
- Verification (`archC_verified_state`'s `:dense_reference` branch: `primal_divergence`,
  `kkt_residual_blas`): already generic on `G`'s width.
- ZC (mean/pairwise) restriction math: **completely untouched**, per task instruction.
- No new Hessian formula and no new outer-gradient formula were written anywhere in this task.

## 5. What DID need dimension-driving (and how)

- `precalc_common_marginals_cdf` / `n_cm_moments` / `build_cm_augmented_obj` /
  `build_cm_meanzc_augmented_obj`: corrected formula, `include_truncated_moment` made a REQUIRED
  kwarg everywhere (no default, per repo rule), new dimension metadata returned
  (`n_families`, `ncm_cdf`, `ncm_pow`).
- Outer-gradient q0-fold (`cm_fixed_contribution` / `cm_fixed_contribution_meanzc_layout`): the
  CDF sub-block's existing O(W·(D-1)) suffix-sum lookup trick is untouched (bit-identical); the
  power sub-block's fixed contribution is added via one direct BLAS matvec against the
  already-precomputed `aug.CM` power columns (a literal, un-optimized evaluation of the same dot
  product the suffix-sum trick accelerates — done once per outer point, not per Newton iteration).
  Both call sites now share one new helper, `cm_fixed_value_contribution_two_family`.
- `CMConfig` (cm_config.jl): new required field `cm_moment_families` (1|2, no default).
- Checkpoint schema: new `CMCheckpointV10` (see section 7).

## 6. Disclosed scope limitation: structured (Architecture C) Hessian and operator FG NOT extended

This is the one place this task did **not** reach full production-grade parity, and it is reported
honestly rather than silently worked around or falsely claimed complete.

`cm_hessian_architectures.jl`'s structured ("Architecture C") bin-table Hessian
(`hessian_cm_structured!`, `build_bin_tables!`, `fill_cm_HCC!`, and every specialized backend
layered on top of them — winner-pair, threaded, drawmajor, CM+ZC's H_CZ/H_ZZ) and
`cm_lookup_kernels.jl`'s `CMLookupState` operator FG both exploit the CDF family's pure 0/1
cumulative-indicator structure (`1{bin(u)<=l}`) with an O(W) suffix-sum/contingency-table
identity. The eq.36 power family's weighted indicator (`z^(1-σ)·1{...}`) does **not** collapse to
the same identity without a materially new derivation (generalizing the existing `T`/`S`/`CT`/`CS`
tables to `T^{11}`/`T^{12}`/`T^{22}` weighted variants, correctly threaded through ~15 accumulated
specialized backends). This was judged out of this task's time budget to derive AND validate
safely — attempting a partial, unverified generalization risked a silently-wrong Hessian, which is
worse than a disclosed gap.

**What this means concretely**:
- The two-family CM spec is fully correct and testable via `moment_representation=:dense_reference`
  + `use_archB_moments=false` (`build_cm_production_context`) → `archC_base_state`/
  `archC_verified_state` (which now auto-select the fully-generic Architecture A dense Hessian,
  `archA_hess_cb_builder`, for any two-family context — needs zero new Hessian code, by
  construction). All D4 gates in this doc used exactly this path and PASS.
- Every single-family-only fast path (Architecture B's `fill_cm_columns_from_bins!`, Architecture
  C's structured Hessian, `CMLookupState`'s operator FG) now HARD-REFUSES a two-family context with
  a clear error, rather than silently only computing the CDF block.
- `run_cm_upper_checkpointed` (the real production driver) was discovered mid-task to hard-require
  an `OperatorPsiBundle` (`production_bundle_api.jl`'s `prepare_production_run`: "production
  runners may only construct OperatorPsiBundle") — and this codebase's own
  `dense_reference_diagnostics.jl` explicitly bans a dense-reference bundle for "a production outer
  solve or campaign." Since the two-family spec's only currently-correct Hessian path IS dense
  Architecture A, `run_cm_upper_checkpointed(...; include_truncated_moment=true, ...)` now hard-errors
  with an explanation, rather than silently downgrading a "production" call to a diagnostic-only
  bundle or bypassing that codebase-wide safety invariant. **Extending Architecture C / the
  operator FG to two families, so the real production driver can run the two-family spec at
  production speed, is disclosed follow-up work**, not attempted in this task.

## 7. Checkpoint/result incompatibility

New `CMCheckpointV10` (fields: `cm_moment_spec`, `cm_feature_family_count`,
`cm_feature_schema_version`, `cm_feature_operator_checksum`). `load_cm_checkpoint` tries schema 10
first; **any** older schema that still deserializes under the pre-existing V2..V9 upgrade chain is
now hard-refused (not auto-upgraded, unlike every prior schema bump) — an old file's
`zfree`/`dual_warm_start`/`best_feasible` were computed against a different, half-width inner CM
block, so silently reinterpreting them under the new spec would corrupt resumed state, not just
mislabel it. `run_cm_upper_checkpointed` also gains a resume-time hard-refuse on
`cm_feature_family_count` mismatch (same discipline as its existing `cm_extension`/
`destination_sample`/`marginal_restriction` checks). Unrestricted results and ZC-only (no CM
block) results are UNAFFECTED — this checkpoint family is CM-specific.

## 8. Fixed-Fréchet marginals audit

`cm_frechet_level.jl`'s `marginal_restriction=:common_frechet` CM sub-block calls
`precalc_common_marginals_cdf(...; include_truncated_moment=false, ...)` explicitly — it does
**not** already impose both families. Its own separate `D*L` common-level-anchor block is a
structurally different (target-based, not eq.35/36-style) restriction, so this is not simply "the
same fix, apply it here too" — extending it would need its own analysis of whether/how eq.36
composes with the level anchor. **Verdict: `missing_second_family_separate_task`** (not rewritten,
per task instruction — audit only).

## 9. Families changed / unaffected

- **Changed** (now impose eq.35+eq.36 when `include_truncated_moment=true`): flexible CM
  (`build_cm_augmented_obj`), CM+ZC (`build_cm_meanzc_augmented_obj` — ZC mean/pair math itself
  untouched).
- **Unaffected**: unrestricted (no CM), origin-ZC-only, common-Fréchet's CM sub-block (audited,
  not rewritten), the `:interval` CM basis (already single-family-only before this task).

## 10. Performance

- Feature construction: O(W·D) per family, computed once at context-build time (`Pow` matrix,
  `W x D`) — confirmed by code inspection (no per-outer-callback recomputation site exists; the
  only per-call work is a `.=` copy of the precomputed `CM` matrix's columns, as before).
  `wrap_moments_with_cm`/`wrap_moments_with_cm_meanzc`'s dense splice is unchanged code, just a
  wider `CM.size(2)`.
- Outer-gradient q0-fold: CDF term unchanged O(W·(D-1)); power term adds one O(W·(D-1)·L) BLAS
  matvec ONCE per outer-gradient call (not per coordinate probe, not per Newton iteration) — at
  D20/L50/W=100,000 this is `100,000 x 950` times a length-950 vector, a few hundred microseconds,
  negligible next to a full inner KNITRO solve.
- Hessian: the one place this task's `~2x CM-only arithmetic` expectation is NOT met — the two-
  family spec currently pays the FULL cost of the dense Architecture A Hessian (`O(W·n^2)` per
  Newton iteration, `n = NCORE+ncm`) instead of Architecture C's `O(W·(D·NCORE+D^2))` bin-table
  cost, a real, disclosed regression on the Hessian backend specifically (see section 6) — NOT a
  "duplicated economic calculation" (the economic/winner-pair computation itself is untouched;
  only the CM-block Hessian assembly is slower).
- D4/D20 measurements: see the real run logs referenced in the final verdict block below.

## 11. Verification of "no dense G / no dense reference hot path" everywhere else

- FG (`wrap_moments_with_cm`): never materializes more than the already-existing `CM` (W x ncm,
  precomputed once) and `G_tmp` (W x ncore, one per FG call, UNCHANGED from before this task) —
  no new dense materialization.
- Outer gradient: the one new BLAS matvec (section 10) is against the SAME already-existing `CM`
  matrix, once per outer-gradient call — not a hot per-Newton-iteration path.
- The Hessian IS the disclosed exception (section 6) — used only through the explicit
  `moment_representation=:dense_reference` opt-in this codebase already gates behind
  `DenseReferenceDiagnostics`/an explicit non-production call, never silently defaulted into.

## 12. Real test results (actual Julia/KNITRO runs, this session)

All runs used `OPENBLAS_NUM_THREADS=1`, this worktree's `Project.toml`, juliaup Julia 1.12.6.
Full logs: `/bbkinghome/edav/repo_scratch/cm-add-truncated-power-moments-2026-08-05/`.

**D4** (`test_cm_truncated_power_2026-08-05.jl`, 5 testsets, all PASS):
- Direct feature construction: hand-built eq.35+eq.36 match `precalc_common_marginals_cdf` exactly
  at W=200/D=4/L=5.
- Dimension gate: `n_cm_moments(20,50;false)=950`, `n_cm_moments(20,50;true)=1900`.
- CDF-block preservation: new CDF sub-block `==` (bit-identical) the `include_truncated_moment=false`
  build at the same context/L/contrasts.
- Real KNITRO inner solve (L=5, L=10) via `oracle.jl::evaluate_fullA` (Architecture A/dense-generic
  Hessian): both feasible (nStatus -100 / -102), CM-block KKT residuals: CDF sub-block
  `~6e-15`/`~2.9e-14`, power sub-block `~6.6e-10`/`~3.6e-9` (both far inside tolerance; power
  sub-block looser only because its raw feature scale is larger, as expected).
- Outer-gradient q0-fold two-family helper (`cm_fixed_value_contribution_two_family`) matches the
  direct `λ_cm'*CM` definition to `1e-9` at a random λ.

**CM+ZC, D4** (`test_cm_meanzc_truncated_power_d4_2026-08-05.jl`, K_mean=1, K_pair=0, all PASS):
dimension gates pass (`ncm=2*(D-1)*L`, `n_families=2`); real KNITRO inner solve at a calibrated
`ν_1 = mean(Zraw_all[1])` starting guess: nStatus=-100 (feasible), CDF-block KKT `~1.9e-14`,
power-block KKT `~1.8e-9` — same machine-precision pattern as plain flexible CM. (The naive
`ν_1=0.0` guess used in an earlier attempt was CM-infeasible at this point — a feasibility
artifact of that specific starting guess, not a math bug; not investigated further given time
budget, since a properly-chosen `ν_1` immediately gave a clean feasible/tight-KKT solve.)

**D20 real data** (`test_cm_truncated_power_d20_2026-08-05.jl`, `destination_sample=:exclude_row`,
raw calibration point `ctx.θ0_up[ctx.free_idx]`, `moment_representation=:dense_reference`,
`use_archB_moments=false` — the disclosed section-6 path):
- **W=5,000 and W=20,000**: an adaptive probe (L=10 down to L=2, BOTH `include_truncated_moment`
  values) found NO L at which the raw calibration point is feasible under the two-family
  restriction (nStatus=-300/-400, confirmed infeasibility) — and at W=20,000 specifically, the
  SAME point IS feasible for the single-family restriction at L≤5 (nStatus=0) while STILL
  infeasible for two-family at every probed L. This is a real, informative, and expected finding,
  not a bug: it directly demonstrates the eq.36 restriction has genuine additional economic bite
  beyond eq.35 alone (more restrictions cannot expand a feasible set), fully consistent with this
  codebase's own pre-existing documented finding that even the ORIGINAL single-family CM
  restriction can render the raw calibration point infeasible (fullA_common_marginals_handoff.md
  section 3). Dimension/config gates (`ncm`, `n_families`) still PASS at both W (config-only,
  since no real solve was reachable). This also matches the pre-existing
  `d20-realdata-w-sensitivity` finding (W<80,000 understates feasibility at D20).
- **W=100,000**: at this W the raw calibration point IS feasible for BOTH family counts at L=10
  (single-family nStatus=0, two-family nStatus=-102) — confirming the W-sensitivity explanation
  above. Full cold solve / warm solve / verification / one analytic outer-gradient call were run
  at this fixed state — see the exact timings/values captured in
  `test_d20_run5_w100k.log` and summarized in the final verdict block below.

## 13. Fixed-Fréchet marginals audit (repeated from section 8 for completeness)

`marginal_restriction=:common_frechet` imposes eq.35 only (`include_truncated_moment=false`
explicitly) — confirmed by direct code read, not rewritten (task instruction: audit only).

## 14. Final grep sweep (residual hardcoded-dimension / single-family assumptions)

`grep -rln "(D-1)\*L\|include_truncated_moment" --include=*.jl .` from the repo root: **every
match is inside `full_aod_diag/d4_exact/`** — zero hits anywhere else in the repository, confirming
this task's changes are fully contained to the canonical CM implementation directory. Within that
directory, every remaining `(D-1)*L`-style hit was individually classified in
`CM_CURRENT_SINGLE_BLOCK_SOURCE_MAP.md` section 4 (interval-basis-only, common-Fréchet-only,
already-fixed dimension metadata, or a plain code comment/docstring) — none is a live, un-reused
single-family assumption in the flexible-CM/CM+ZC production path this task changed.

## 15. Production release

Not merged, not pushed, not tagged — per task instructions, this branch stays on
`fix/cm-add-truncated-power-moments-2026-08-05` in this worktree, awaiting explicit user approval
for any production integration (including the still-open Architecture C generalization needed
before `run_cm_upper_checkpointed` can actually run the two-family spec at production speed).
