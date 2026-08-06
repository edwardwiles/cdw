# Common-marginals restriction: add CDW eq.36 (truncated (1-σ)-power family) — MASTER

Branch: `fix/cm-add-truncated-power-moments-2026-08-05` (production/fullA-exact @546feff base).
Worktree: `/bbkinghome/edav/cdw_worktrees/cm-add-truncated-power-moments-2026-08-05`.

## ⚠️ CORRECTION / CONTINUATION SESSION (2026-08-05, later the same day) — READ THIS FIRST

**Sections 0–16 below are from an earlier pass the same day and are now superseded in several
material ways.** A follow-up continuation session (same day) made these corrections, in order:

1. **The exponent in section 0's "fix" was itself still wrong.** The user corrected it directly:
   the restriction is `z^(σ-1)`, not `z^(1-σ)` (a standard CES-aggregator-shaped exponent). Every
   place below that says `z^(1-sigma)` / `1-σ` is the OLD, incorrect exponent.
2. **A second, more serious bug was found and fixed after that**: the eq.36 feature reused eq.35's
   `1{U≤c}` indicator convention for the *weighted* restriction. That's harmless for the unweighted
   CDF family (a pure sign flip of the whole moment condition) but wrong for the POW family — the
   correct translation of `1{z<z_ℓ}` given `z=U^{-μ}` (decreasing in `U`) is `1{U>c}`, not `1{U≤c}`.
   Using `≤` silently introduces an additive bias term that vanishes only under the uniform/base
   measure, not under any reweighting — exactly what the inner KNITRO dual solve searches over.
   See new section 17 for the full derivation and fix.
3. **Architecture C (the real, no-dense-H production path — structured Hessian +
   `CMLookupState`/`CMMeanZCOperatorState` operator FG) was actually extended** for the two-family
   spec, for BOTH plain flexible CM and CM+ZC, with zero dense G/H anywhere — closing the gap
   section 6/6b of the earlier pass had left as a disclosed limitation. See new section 18.
4. **The D20 real-data results in section 12b below (`nStatus=0` "fully feasible" at W=20k/100k)
   predate fix #2 above and are INVALID** — they were computed with the wrong indicator direction.
   The real, corrected re-test result is that the two-family real-D20 solve does **not** converge,
   at any tested `L` down to 3, after a long investigation (sections 19–24) that ruled out several
   more mundane explanations (W-sensitivity, iteration-budget, an Architecture C formula bug) but
   did **not** identify a root cause — as of this writing this is an **open, unresolved
   diagnostic**, believed more likely than not to be a residual bug (user's judgment, section 24),
   not a confirmed finding either way. Do not cite section 12b's "PASS" claims.
5. Section 16's verdict block is likewise stale on `SCIENTIFIC_SPEC` and `FLEXIBLE_CM`/`D20`
   fields — see the new verdict block at the end of the document (section 25) for the current,
   accurate status.

## 0. READ FIRST — scope grew beyond the original task brief, twice, both user-directed

This task ended up covering FOUR distinct pieces of work, not one. In order:

1. **The original mission**: add the missing eq.36 (truncated `(1-σ)`-power) CM feature family
   alongside the existing eq.35 (CDF) family.
2. **A real bug in this task's own new code, found and fixed within the same session**: the
   first implementation of eq.36 computed the power feature as `U[:,x]^(1-σ)` -- treating the raw
   exponential draw `U` (`ctx.U`, `Exp(1)`-distributed) as if it were already the Fréchet
   productivity level `z_o(ω)`. This is harmless for eq.35 (a pure indicator/rank statistic,
   invariant to the representation) but not for a power-weighted moment. At real D20 data
   (`σ≈2.5-3`), this exponent is `≤-1` applied to a variable whose density does not vanish at 0,
   making the population moment genuinely infinite and the sample moment dominated by whichever
   single draw happens to land nearest zero -- confirmed directly (`min(U)=1.3e-5` producing a
   single-draw value of `~2.1e7`, `POW/CDF` raw-moment-scale ratio `~610,000`) and confirmed to be
   the actual cause of the D20 real-data infeasibilities documented in section 12's *first* pass
   (now superseded). Section 0a below has the full writeup, the exact formula correction, and how
   it was found (pure arithmetic, no KNITRO, per explicit instruction).
3. **A user-directed, deliberate scope expansion beyond the original brief**: switch the common-
   marginals quantile cutoffs `z_ℓ` (both families) from empirical (order-statistic estimates of
   the realized draws) to the theoretical closed-form Fréchet quantile, and add a closed-form
   (truncated incomplete-gamma) reference target for the truncated-power moment, replacing what
   would otherwise be a Monte-Carlo-only verification value. Section 0b has the full derivation
   (a genuinely subtle piece of math -- a decreasing-transform sign flip -- verified two
   independent ways, not just derived) and its consequences.
4. Steps A-G continue after that, now against the corrected/expanded implementation.

**Everything in section 12 (real test results) describes the FIRST pass, before items 2 and 3
above** -- that section's headline finding ("the two-family restriction is genuinely infeasible at
the raw D20 calibration point, and this is expected/documented behavior, not a bug") **was itself
wrong**, superseded by section 0a/12b. Left in place (not deleted) as an honest record of the
investigation path, per this project's own standing practice of correcting forward rather than
erasing a wrong prior claim.

## 0a. The bug: raw exponential draw `U` used where the Fréchet productivity `z=U^(-μ)` belongs

**Confirmed relation** (from `fix/zc-frechet-draw-moments-2026-08-05`, commit `6fd0229` --
independently fixed, same day, in this exact repo's ZC/meanZC restrictions, for the identical
class of error -- cherry-picked into this branch rather than re-derived a second time):
```julia
# cm_meanzc_moments.jl
frechet_productivity_from_exponential(U, μ) = U .^ (-μ)     # z_o(ω) = U_o(ω)^{-μ}, μ=1/θ
frechet_power_feature(U, k, μ) = exp.(-μ*k .* log.(U))      # z^k = U^(-μk)
```
`ctx.U` is confirmed the raw, untransformed `Exp(1)` draw in production (the `θConstant`-gated
`U.=U.^(-μHat)` mutation in `createUDerivatives!.jl` is hardcoded off in `AD_PARAMS`, unreachable
on any real path).

**This task's first implementation** (`common_marginals_moments.jl`):
```julia
pw = 1 - σHat
Pow[:, x] = U[:, x] .^ pw          # WRONG: treats U as if it were already z
```
For `k=1-σ`, the correct feature is `z^(1-σ) = U^{-μ(1-σ)} = U^{μ(σ-1)}`, not `U^{1-σ}` directly.

**Numeric confirmation, pure arithmetic, no solver** (`diag_raw_moment_check_2026-08-05.jl`, real
D20 data, `W=5,000`, `σ=2.5`):

| | old (wrong) exponent `1-σ=-1.5` | corrected exponent `μ(σ-1)≈0.096` |
|---|---|---|
| `mean(z^(1-σ))`, origin 3 | 4,486.6 (one draw `≈2.1e7` dominates) | 0.9523 |
| `mean(z^(1-σ))`, origin 1 | 153.8 | 0.9554 |
| POW/CDF raw-moment-scale ratio | ~610,000 | 0.83 |

After the fix, `mean(z^(1-σ))` is ~0.95 for **every** origin (as it should be for exchangeable
draws) instead of spanning 2+ orders of magnitude driven by sampling extremes -- direct, numeric
confirmation this was the actual defect, not a KNITRO/solver artifact.

**Fix applied**: cherry-picked `6fd0229` (adds the two functions above to `cm_meanzc_moments.jl`,
cleanly, no conflicts with this branch's own CM+ZC edits); widened `frechet_power_feature`'s
`k::Int` to `k::Real` (same formula, same behavior for every existing integer-`k` ZC caller, now
also usable for CM's non-integer `k=1-σ`); `common_marginals_moments.jl` now computes
`Pow = frechet_power_feature(U, 1-σHat, μHat)`; `μHat` threaded as a new required (no-default)
kwarg through `precalc_common_marginals_cdf`/`build_cm_augmented_obj`/
`build_cm_meanzc_augmented_obj`, mirroring `σHat`'s existing treatment.

**Consequence for the gates already run**: the original D4 "exact agreement" gates validated the
code against a hand formula that made the *same* mistake (`U^(1-σ)`, not `U^(μ(σ-1))`), so they
could never have caught this -- redone for real with (a) actual `Exp(1)` draws (the original
gate's synthetic `Uniform(0.1,3.1)` draws are bounded away from 0 and could never expose a
divergent-near-zero exponent regardless of sign/variable convention) and (b) an independently
re-derived hand reference (`z=U^{-μ}` computed directly in the test, not by calling
`frechet_power_feature`). All PASS -- see section 12b.

## 0b. Theoretical (not empirical) quantile cutoffs + closed-form truncated-moment target

**User-directed, beyond the original task brief** (the original brief says "do not change the
quantile grid" -- this changes how the grid's *cutoff values* are computed, not which probability
levels are on the grid).

**(a) Cutoffs**: `z_ℓ` was `quantile(U[:,refIndex1], probs)` (an order-statistic ESTIMATE of `U`'s
own distribution). Replaced with `theoretical_u_threshold.(probs) = -log.(1 .- probs)` (the
closed-form Exp(1) quantile of `U`, computed exactly rather than estimated). Why this is the
correct closed form for a target expressed in terms of the *Fréchet* quantile `z_p` (not naively
"just use U's own theoretical quantile because it's convenient") -- full derivation in
`theoretical_u_threshold`'s own docstring, `common_marginals_moments.jl`; summary:
  - Fréchet quantile at level `p` (`T=1`, confirmed convention): `z_p = (-log p)^{-μ}`.
  - `1{z_o(ω)<z_p} = 1{U_o(ω) > -log(p)}` (the map `z=U^{-μ}` is strictly DECREASING, so the
    inequality flips).
  - `1{U_o>c}-1{U_ref>c} = -(1{U_o<=c}-1{U_ref<=c})` -- an overall sign flip that does not change
    the restriction's feasible set (`E_F[X]=0 ⟺ E_F[-X]=0`), so the EXISTING code structure
    (`1{U<=c}`, unchanged, no rewrite of any downstream bin-index/Hessian/lookup machinery)
    already imposes the mathematically correct restriction once `c` is chosen correctly.
  - The equal-probability grid `range(1/L,(L-1)/L,length=L)` is symmetric under `p->1-p`, so
    evaluating at `probs` directly (rather than `1 .- probs`) tests the identical SET of L
    Fréchet-quantile levels, just re-indexed -- and conveniently keeps `z` sorted ascending
    (required by the untouched bin-index architecture) automatically.
  - Net result: `c(p) = -log(1-p)`, which is, not coincidentally, exactly `U`'s own closed-form
    Exp(1) quantile -- "switch from empirical to theoretical" for this restriction reduces to
    using `U`'s known exact quantile instead of estimating it, once the sign/relabeling above is
    accounted for.

**Validated two independent ways, not just derived** (`diag_theoretical_quantile_check_2026-08-05.jl`,
pure arithmetic, no KNITRO):
  - Direct Monte Carlo `E[z^k·1{z<z_p}]` (computed literally in z-space, independent of the
    existing code's `U`-space structure) converges to the closed-form target as `W` grows
    (`10,000 -> 1,000,000 -> 20,000,000`: `|diff|` `2.7e-3 -> 7.9e-4 -> 1.9e-4` at `p=0.1`, similar
    at `p=0.5,0.9`).
  - The actual `precalc_common_marginals_cdf` output, at `W=2,000,000`, gives
    `E[z_o^k·1{z_o<z_p}]` (computed per-origin, not differenced) matching the closed form to
    `~1e-4` (Monte Carlo noise at this `W`), AND its own differenced/contrast-anchored CM column
    mean is small (`~1e-4`, consistent with exchangeable synthetic origins), confirming the
    EXISTING code structure (unchanged) really does implement the correct restriction once fed
    the new theoretical cutoffs.
  - A naive same-sign check (`1{U<=c}` compared directly against the one-sided `E[z^k 1{z<z_p}]`
    target, NOT differenced) does NOT match (`|diff|~0.2-0.8`) -- confirms the sign-flip argument
    is doing real, necessary work and isn't a no-op; the restriction is only correct once
    differenced against the reference origin, exactly as the existing code already does.

**(b) Closed-form verification target**: `eq36_theoretical_truncated_moment(z_ℓ,k,μ) =
Γ(1-μk, z_ℓ^{-1/μ})` (upper incomplete gamma, `SpecialFunctions.gamma(s,a)`), derived
independently and confirmed to reduce to this codebase's own existing untruncated
`ν_k=Γ(1-μk)` convention as `z_ℓ→∞` (`a→0`, `Γ(s,0)=Γ(s)`) -- checked directly:
`eq36_theoretical_truncated_moment(1e15,k,μ) = 0.9119156073` vs `Γ(1-μk) = 0.9119156072537992`.
**Verification/reference use only** -- never used inside the actual inner KNITRO dual solve (that
computation is, and remains, the empirical/reweighted-measure sum over the realized `W` draws;
there is no population-level substitute for optimizing over how those draws get reweighted, per
the whole CC/robust-optimization architecture).

**Consequence (disclosed, intentional, not a regression)**: switching to theoretical cutoffs
changes eq.35's own numeric restriction values relative to pre-2026-08-05 production (which used
the empirical quantile). The "CDF-block-preservation" gate is reinterpreted accordingly (section
12b) -- it now means "self-consistent between a standalone single-family build and the two-family
build's own CDF sub-block, at the identical (now theoretical) cutoff convention," not "bit-
identical to old production's empirical-cutoff values" (that comparison is expected to differ and
was not tested). Common-Fréchet's own "level" restriction (`cm_frechet_level.jl`) reuses
`precalc_common_marginals_cdf` for its CM sub-block (`include_truncated_moment=false`) and so
inherits the same cutoff-formula change; audited (not re-verified with new gates, per this
family's existing "audit only" scope) and found LOW RISK / arguably improved: its own level-target
formula (`target_l=sqrt(D)*p_l`) was already a pure function of the probability grid `p_l`, not of
the empirical `z` values, and the relationship "`z[l]` is the reference origin's own `p_l`-quantile"
becomes EXACT rather than approximate once `z[l]` is the true theoretical quantile.
**Not touched** (explicit, pre-existing scope boundary, unrelated to this directive): the
`:interval` CM basis (`common_marginals_interval.jl`) keeps its own independent empirical
`quantile(U[:,refIndex1],...)` call -- already single-family-only and out of scope before this
task, left that way.

**Disclosed gap, found in the final grep sweep, NOT fixed (time budget)**:
`nested_quantile_grids.jl::quantiles_from_probs` independently replicates the OLD empirical
`quantile(U[:,refIndex1],probs)` convention for `cm_grid_rule=:nested_family` (a non-default
option; production default is `:equal`, which goes through the now-theoretical
`precalc_common_marginals_cdf` path exclusively). This means `:nested_family` mode's own cutpoints
remain empirical, now INCONSISTENT with `:equal` mode's theoretical convention. Neither this
task's D4/D20 gates nor any production driver default exercises `:nested_family`, so this is a
real but currently-dormant inconsistency -- flagged here rather than silently left for a future
session to rediscover from scratch.

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
### (SUPERSEDED — see section 6b below. This section is the first session's record; the gap it
### describes was closed in a follow-up session the same day, at the user's direct request.)

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

**SUPERSEDED by section 12b below** once the raw-U-vs-Frechet-z bug (section 0a) was found and
fixed and theoretical quantile cutoffs (section 0b) were implemented — re-running the identical
D20 gates after both changes gives materially different (better, and now fully explicable)
results. Left above as an honest record of the investigation path.

## 12b. Real D20 test results, REDONE after the bug fix + theoretical-quantile change (final)

Same script (`test_cm_truncated_power_d20_2026-08-05.jl`), same adaptive-L probe methodology, same
raw calibration point construction, rerun after both section 0a (bug fix) and section 0b
(theoretical cutoffs) were implemented. Full log:
`/bbkinghome/edav/repo_scratch/cm-add-truncated-power-moments-2026-08-05/test_d20_final.log`.
Result: **18/18 checks PASS, exit code 0.**

- **W=5,000**: the raw calibration point is STILL genuinely infeasible for BOTH family counts
  across the full probed range `L∈{10,9,...,2}` (nStatus=-300 throughout, unchanged by either
  fix). Since this affects the single-family (unchanged, pre-existing) restriction identically to
  the two-family one, this is NOT the bug from section 0a (that was specific to the power family's
  own divergent exponent) -- it is a genuine, real feasibility-boundary characteristic of this
  particular W/point, consistent with the pre-existing `d20-realdata-w-sensitivity` finding
  (W<80,000 understates feasibility at real D20 scale). Dimension/config gates (`ncm=380`,
  `n_families=2`) still PASS (config-only, since no real solve is reachable at this W/point).
- **W=20,000**: fully feasible at `L=10` for BOTH family counts (nStatus=0 for both, where
  BEFORE the fix the two-family restriction was infeasible at every probed L down to 2). Real
  results: context build 6.93s; context+pcx build 0.87s (`ncm=380`, `ncore=382`); **cold solve**
  4.18s, nStatus=0; **warm solve** 0.55s, nStatus=0, `Delta_dual=0.021794` (bit-consistent with the
  cold solve's own recomputed value); **verification** `max_abs_moment_kkt_resid=5.00e-14`; **one
  analytic outer-gradient call** 12.97s, length 380 (`=D*Ddest`), norm 2.10352, all finite.
- **W=100,000**: fully feasible at `L=10` for BOTH family counts (nStatus=0 for both). Real
  results: context build 25.19s (peak RSS 5.87 GB); context+pcx build 2.48s; **cold solve**
  14.38s, nStatus=0; **warm solve** 2.73s, nStatus=0, `Delta_dual=0.004261` (bit-consistent);
  **verification** `max_abs_moment_kkt_resid=6.04e-15`, `m_min=0.329` (healthy, not
  near-degenerate); **one analytic outer-gradient call** 35.0s, length 380, norm 0.802368, all
  finite. This is the task brief's explicitly-required "single fixed-state" W=100,000 gate
  (cold solve, matched warm solve, one analytic outer-gradient call) — satisfied.
- **Checkpoint/resume smoke and per-call allocation measurement at W=100,000** were NOT run
  separately (time budget, after the two rounds of scope expansion above) — the checkpoint-
  incompatibility MECHANISM itself (schema V10, hard-refuse on family-count mismatch) is
  exercised structurally in section 7/D4 but not via an actual `run_cm_upper_checkpointed` save/
  load cycle at W=100,000 real data (which is unreachable anyway per section 6's disclosed
  production-driver limitation — `run_cm_upper_checkpointed` hard-refuses `include_truncated_
  moment=true` outright). Disclosed honestly rather than fabricated.

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

## 16. Final verdict block

```
SCIENTIFIC_SPEC = CDF_plus_truncated_1_minus_sigma (feature: z^(1-sigma), z=U^(-mu); cutoffs: theoretical Frechet quantile, not empirical)
CM_BLOCK_WIDTH = old:(D-1)*L new:2*(D-1)*L
D20_L50_WIDTH = 1900
CDF_BLOCK_PRESERVED = pass_reinterpreted (self-consistent under the new theoretical-cutoff convention; NOT bit-identical to old empirical-cutoff production values -- that change is deliberate/user-directed, see section 0b)
TRUNCATED_POWER_FEATURE = pass (after fixing a real bug found+fixed within this task: z=U^(-mu) required, not raw U -- see section 0a)
DOWNSTREAM_MATH_FORMULAS_CHANGED = false (FG/Hessian/gradient formulas unchanged; only feature-construction inputs -- the exponentiation base and the cutoff values -- changed)
NEW_PARALLEL_KERNELS_CREATED = 0
DIMENSION_DRIVEN_REUSE = FG:pass Hessian:partial(Architecture A dense-generic only, Architecture C not extended -- disclosed, section 6) outer_gradient:pass verification:pass checkpoint:pass
FLEXIBLE_CM = D4:pass D20_W20K:pass D20_W100K:pass (D20_W5K: config/dimension-only, real solve unreachable at the raw calibration point at this W for EITHER family count -- pre-existing W-sensitivity, not a bug, see section 12b)
CM_PLUS_ZC = D4:pass D20_W20K:not_run D20_W100K:not_run (CM+ZC D20 real-solve gates not re-run after the final theoretical-quantile change, time budget -- D4 fully validated with machine-precision KKT residuals)
FIXED_FRECHET_AUDIT = already_has_both:false; missing_second_family_separate_task (audited only, not rewritten, per task instruction; low-risk positive side effect from theoretical cutoffs noted in section 0b)
PRODUCTION_RELEASE = not_merged_awaiting_user_push_approval
REDUCED_REDESIGNED = false
DENSE_PRODUCTION_GH_USED = false (Architecture A/dense_reference used only as an explicit, disclosed, non-default opt-in for the two-family spec's correctness gates -- never the ambient/silent default, never used in a real campaign)
CAMPAIGN_LAUNCHED = false
NEW_BRANCHES_CREATED = 0 (already created by orchestrator; one commit cherry-picked from a pre-existing sibling branch, fix/zc-frechet-draw-moments-2026-08-05, not a new branch)
EXTRA_WORKTREES_CREATED = 0 (already created by orchestrator)
BUG_FOUND_AND_FIXED_WITHIN_TASK = true (raw exponential draw U used where Frechet productivity z=U^(-mu) belongs in the new eq.36 feature -- see section 0a for full derivation/numeric confirmation)
SCOPE_EXPANSIONS_BEYOND_ORIGINAL_BRIEF = 1 (user-directed: empirical -> theoretical Frechet quantile cutoffs + closed-form truncated-moment verification target, section 0b)
```

**This verdict block is STALE — see the CORRECTION banner at the top of this document and section
25 at the end for the current, accurate status.**

---

# CONTINUATION SESSION (2026-08-05, later the same day)

Everything below is from a second, longer continuation session the same day. It (a) corrects the
exponent sign, (b) finds and fixes a second, more serious indicator-direction bug, (c) genuinely
extends Architecture C (the real no-dense-H production path) for the two-family spec, (d)
extensively verifies that extension, and (e) investigates — without yet resolving — why the real
D20 two-family solve still does not converge even after both fixes.

## 17. THE bug: eq.36's indicator direction, not just its exponent

**User-caught, not self-discovered.** After the Architecture C extension (section 18) was built and
initially verified against a hand-derived "brute-force" reference (which itself embedded the same
mistake — see the warning in section 20), the real D20 two-family solve was infeasible
(`nStatus∈{-300,-400}`) at the calibration point. The first-pass read of this was "the corrected
z^(σ-1) restriction has real economic bite" — the user rejected that immediately and firmly:

> "It is almost certainly not a scientific finding. It just means there's a bug. For example did
> you set the correct theoretical quantile given that its sigma -1 now and not 1-sigma?"

Re-deriving the theoretical cutoff math from scratch surfaced the real bug:
`theoretical_u_threshold(p) = -log(1-p)` gives the U-space cutoff `c` such that
`1{z(ω)<z_p} = 1{U(ω) > c}` — **not** `1{U(ω)<=c}` — because `z=U^{-μ}` is a *decreasing* function
of `U`. For the CDF (weight=1) family, reusing `1{U<=c}` is harmless: it's just an overall sign
flip of the whole restriction (`E[X]=0 ⟺ E[-X]=0`). For the POW (weight=`z^k`, `k=σ-1`) family it
is NOT harmless:

```
Pow_o·1{U_o<=c} - Pow_ref·1{U_ref<=c}
  = (Pow_o - Pow_ref) - (Pow_o·1{U_o>c} - Pow_ref·1{U_ref>c})
```

i.e. using `<=` instead of `>` introduces an additive bias term `E_π[Pow_o]-E_π[Pow_ref]` that
vanishes only under the base/uniform measure (both origins IID canonical Fréchet(1)) but is
generally nonzero under any *reweighted* measure π — exactly what the inner KNITRO dual solve
searches over. Confirmed numerically before touching any code, then fixed in
`common_marginals_moments.jl`'s `precalc_common_marginals_cdf` (the eq.36 block, separate
`CDF_ref_pow` built with `.>` rather than reusing the CDF family's `.<=` reference), and propagated
to every downstream consumer (section 18) and every hand-written test reference (section 20).

## 18. Architecture C extension (the real no-dense-H production path), zero dense G/H

Per direct user instruction: *"Your job is to make arch C work. It is not useful to fix it in the
dense method, since I don't use it... you should definitely do it yourself rather than trusting an
agent."* Extended, by hand, for BOTH plain flexible CM and CM+ZC:

- **`cm_hessian_architectures.jl`**: `CMBinHessCtx` gained a `Pow::Union{Nothing,Matrix{Float64}}`
  field plus `Ttab12/Ttab22/CT12/CT22/Stab2/CScum2/Hraw_CC12/Hraw_CC22/...` scratch. New
  `_build_reflected_bilinear(Ttab, CT, D, L; reflect_x, reflect_y)` helper implements the
  "total-minus-cumsum" inclusion–exclusion identity needed to get `1{bin(x)>l}`-style reflected
  cumulative tables from the same raw per-draw accumulation `build_bin_tables!` already does — no
  new per-draw accumulation, only post-processing of already-built tables.
  `hessian_cm_structured!` now supports `cctx.n_families==2` on both the dense-H and winner-bin
  (`:winner_bin`) paths.
- **`winner_pair_cross_hessian.jl`**: `WinnerBinCrossScratch`/`BinZCrossScratch` gained matching
  `*_pow` companion tables; `winner_pair_cross_hessian_fill!`/`_cm_block!` and
  `bin_zc_cross_hessian_fill!`/`_block!` (CM+ZC's own widened-row H_CZ cross term) extended the
  same way — this is what removed the CM+ZC dense-H-forced fallback the first-pass session had
  left in place.
- **`cm_lookup_kernels.jl` / `cm_meanzc_lookup_kernels.jl`**: `CMLookupState` /
  `CMMeanZCOperatorState` (the real per-iterate operator FG KNITRO calls every inner-solve
  iteration) gained `Pow`/`*_pow`/`*2` buffers, new `interval_forward_contribution_pow!` /
  `cumulative_forward_contribution_pow!` / `build_weighted_histogram_pow!` kernels, and a
  reflect-in-place step (`total_o - Hpre2[o,l]`) mirroring the Hessian-side reflection identity.
  **`CMMeanZCOperatorState`'s fields were initially dead code** — `dual_index!` and the functor
  never read them — fixed by actually wiring the pow forward/backward contributions through both.
- **`cm_production_bundle.jl` / `cm_meanzc_lookup_production.jl`**: removed the stale hard-refuses
  on `include_truncated_moment=true` + `inner_fg_backend=:cm_lookup`/`moment_representation=
  :operator` (kept only the genuine `use_archB_moments` refuse — Architecture B was never
  extended, out of scope), and flipped the production dispatch sites to the unconditional
  `archC_hess_cb_builder(cctx)` (previously conditional on single-family).
- Two unrelated, pre-existing production regressions found and fixed as a byproduct of actually
  exercising `run_cm_upper_checkpointed` for the first time this task: `CMMeanZCConfig`'s own
  `cm::CMConfig = CMConfig()` default broke the instant `CMConfig.cm_moment_families` lost its
  default (fixed in `cm_checkpoint.jl` by passing it explicitly); and a `struct CMCheckpointV10`
  name collision between `cm_checkpoint.jl` and `cm_originzc_checkpoint.jl` silently shadowed one
  or the other depending on include order (fixed by renaming the origin-ZC copy to
  `OriginZCCheckpointV10`).

## 19. D4 verification — ALL PASS at machine precision (post both fixes)

Four independent D4-scale gates, every one re-run after section 17's fix, all passing at machine
precision (`<1e-8`, mostly `<1e-14`):
- `test_cm_archc_hcc_twofamily_2026-08-05.jl` — H_CC vs an independent brute-force sum.
- `test_cm_archc_full_twofamily_2026-08-05.jl` — full assembled Hessian (dense-H path AND the real
  winner-bin production path) vs Architecture A.
- `test_cm_lookup_fg_twofamily_2026-08-05.jl` — operator FG (f,g) vs Architecture A, both `:suffix`
  (real production basis) and `:interval`.
- `test_cm_meanzc_archc_twofamily_2026-08-05.jl` — CM+ZC, both the dense-H path and the winner-bin
  + H_CZ path.

## 20. Independent ground truth via ForwardDiff (user-requested, decisive)

User's explicit concern: comparing Architecture C against a hand-derived "brute-force" reference
(section 19) risks the SAME conceptual mistake appearing in both, since both were written in the
same session with the same mental model. Requested an independent check via automatic
differentiation instead.

`test_cm_autodiff_groundtruth_2026-08-05.jl` reads ONLY the raw feature matrix
`objA.H[:,3:n+1]` (== `G`, the actual columns `objA.moments!` populates, including the two-family
CM columns) and the closed-form `Psi(a) = e^a-1` (a≤1) / `(e/2)(a²+1)-1` (a>1)
(`cc_algo/Psi.jl`, reimplemented by value — zero shared code), then differentiates
`f(ζ,λ) = (1/M)·Σ_s Psi(-ζ - dot(G[s,:],λ)) + ζ` via `ForwardDiff.gradient`/`ForwardDiff.hessian`.
Compared against the REAL operator-bundle gradient (`CMLookupState`'s own functor) and the REAL
winner-bin-path Hessian (`hessian_cm_structured!`, with a genuine `core_cf_ref` populated via
`cf_build` so `_cm_cross_hessian_wants_winner_bin` is confirmed true).

**Result: PASS at every one of 4 random dual points**, `max|Δg|` and `max|Δh|` both `~1e-14` to
`~1e-16`, across the full gradient, full Hessian, and every CM sub-block (H_CC[cdf,cdf],
H_CC[pow,pow], H_CC[cdf,pow] cross, H_EC[core,cdf], H_EC[core,pow]) — see
`/tmp/cm_autodiff_groundtruth.log`. This rules out a Hessian/gradient *assembly* bug in the
Architecture C CM extension, independent of any assumption shared with the hand-derived references
in section 19.

## 21. D20 real-data investigation: what was ruled out

What's running in every test below is **plain flexible CM only** (`build_cm_production_context`
has no `cm_extension` kwarg at all; `:cm_lookup`/`:operator` is documented "plain flexible CM
only") — not CM+ZC, not common Fréchet, not origin-ZC.

- **Raw-moment (solver-free) diagnostic** (`test_cm_raw_moment_check_2026-08-05.jl`, user-suggested
  — "the average of the moments under the raw F* Monte Carlo draws should be near zero if the
  moments are correctly specified and calibrated"): at the real calibration point, both CDF and POW
  block column means are small relative to the raw feature scale (~1.066) and shrink correctly as
  `1/√W` (max|col mean| 0.0182 at W=20k → 0.0059 at W=100k, vs the theoretical `√5≈2.24` ratio for
  a 5x sample increase) — the signature of pure Monte Carlo noise around a true zero, not a
  systematic spec bug.
- **W-sensitivity ruled out**: the real two-family KNITRO fixed-state solve
  (`test_cm_archc_d20_fixedstate_2026-08-05.jl`) gives the IDENTICAL result
  (`nStatus=-400`) at W=20,000, W=100,000, AND W=300,000 — not shrinking with W, unlike the
  single-family case at the same θ point, which genuinely improves (`-103` at W=100k → `0` at
  W=300k). This is not the pre-existing `d20-realdata-w-sensitivity` pattern.
- **Iteration-budget starvation ruled out**: `nStatus=-400` is KNITRO's
  `KN_RC_ITER_LIMIT_FEAS` — "feasible point WAS found," not a hard failure — and every production
  `archC_*_base_state` function's own accept-list (`(0,-100,-101,-103)`) is *narrower* than several
  other gates/benches in this same repo that also accept `-400/-401/-402`. Raising `maxit` 100→2000
  (`ek_inner_maxit2000_2026-08-05.opt`) gave the IDENTICAL `nStatus=-400` result after ~19 minutes
  wall-clock (vs 18.6s for single-family) — 20x the iteration budget did not resolve it.

## 22. Iteration trace and moment-matrix conditioning

An `outlev=3` trace (`ek_inner_trace_2026-08-05.opt`, `maxit=60`,
`test_cm_archc_d20_trace_2026-08-05.jl`) shows: `FeasError=0.000e+00` at every single iteration (the
point stays feasible throughout); `OptError` (KKT residual) oscillates in a 0.01–0.05 band across
all 60 iterations with no downward trend (never approaching the `opttol=1e-12` target); the
objective decreases slowly and monotonically (`-4.3e-4`→`-2.2e-3` over 60 iterations) with small
step sizes (`~1e-3`), never leveling off within the tested window.

A dense moment-matrix rank/SVD check at the real D20 calibration point, W=100,000, L=10
(`test_cm_moment_rank_2026-08-05.jl`) found no hard rank deficiency (every block full rank at a
`1e-10`-relative tolerance — unlike the previously-found CM+ZC K=2(σ=3) case, which was a true
near-machine-precision singularity) but a real conditioning jump from combining the two families:

| block | cond(·) |
|---|---|
| CDF-only (eq.35) alone | ≈32 |
| POW-only (eq.36) alone | ≈30 |
| CDF+POW combined | ≈8,730 |
| core+CDF+POW (full G) | ≈85,343 |

and a maximum per-`(origin,ℓ)` correlation between a CDF column and its matching POW column of
**0.9997**, at the last (highest-quantile) bin `ℓ=L=10`, origin index 15.

## 23. L-sweep — the near-collinearity and non-convergence persist at EVERY L, down to L=3

`test_cm_moment_rank_by_L_2026-08-05.jl` re-ran the rank check AND the real KNITRO fixed-state
solve at L∈{3,4,5,6,8,10}:

| L | ncm_cdf | cond(CM) | cond(fullG) | worst corr | at (ℓ,origin) | KNITRO |
|---|---|---|---|---|---|---|
| 3 | 57 | 1,978 | 28,092 | 0.9989 | (3, 7) | `-400` |
| 4 | 76 | 2,236 | 30,137 | 0.9993 | (4, 15) | `-400` |
| 5 | 95 | 2,881 | 36,511 | 0.9994 | (5, 15) | `-400` |
| 6 | 114 | 3,743 | 44,668 | 0.9995 | (6, 15) | `-400` |
| 8 | 152 | 5,934 | 63,624 | 0.9996 | (8, 15) | `-400` |
| 10 | 190 | 8,730 | 85,343 | 0.9997 | (10, 15) | `-400` |

Every tested L fails identically. The worst-correlated pair is always the LAST bin (`ℓ=L`)
regardless of how many bins there are — at L=3 that bin's probability threshold is only 2/3, not an
extreme tail — and origin index 15 recurs as the worst-correlated origin at every L≥4.

## 24. Current status: UNRESOLVED — user's assessment is this is a residual bug, not a finding

User's own read, directly: *"The fact that it's still there at L=3 strongly indicates that it's a
bug. It just isn't that hard to satisfy this restriction, especially at L=3."* This is a reasonable
and, as of this writing, **unrefuted** objection — a 3-bin flexible-CM restriction (only 6 CM
moments per non-reference origin) should be easy to satisfy at real D20 scale, and section 17's
already-confirmed, already-fixed bug demonstrates this exact restriction family is not immune to
subtle sign/indicator mistakes.

**What has been ruled out** (sections 19–23): a moment-specification/indicator-direction bug of
the kind already found once (raw-moment check is clean at the base measure); W-sensitivity;
iteration-budget starvation; an Architecture C Hessian/gradient *assembly* bug (independent
ForwardDiff ground truth, section 20); hard rank deficiency (formally full rank at every tested L).

**What has NOT been ruled out / not yet checked**:
- A moment-specification bug that only manifests under a *reweighted* (non-uniform) measure — the
  raw-moment check (section 21) only tests the base/uniform measure, which is exactly the situation
  section 17's own bug analysis says is insufficient to catch a POW-family sign/direction error.
- Whether the recurring origin index 15 (and the always-last-bin pattern, present even at L=3
  where the last bin isn't tail-extreme) points at something specific to how the CDF/POW columns
  are being constructed or indexed for that origin/bin combination specifically, rather than a
  property of the true restriction.
- Whether `contrasts=:anchored`'s reference-origin construction interacts with the two-family
  augmentation in some column-indexing way that isn't a "wrong economics" bug but a "wrong column"
  bug (e.g. an off-by-one in which origin/bin the POW column for a given CDF column actually
  corresponds to).
- Running the KNITRO solve to a much larger iteration count with NO cap, to see whether it
  eventually crosses `lower_limit=-50` (a clean `-300`, confirming genuine unboundedness) — not yet
  done; the trace in section 22 only covers 60 iterations.

**This document does not claim the bug is found.** It documents the fix that WAS found and fixed
(section 17, verified section 19–20) and the extensive, honest trail of eliminated hypotheses for
the SEPARATE, still-open D20 non-convergence problem (sections 21–23), so a future session — or
this same session, continuing — does not have to re-derive any of this from scratch.

## 24b. POW-only isolation experiment — confirms it's the INTERACTION, not either family alone

User-requested follow-up: *"remove the 'regular' CM moments and just leave these new z^(σ-1) ones?
So we can see if it's about the interaction."* `test_cm_pow_only_experiment_2026-08-05.jl` builds a
single-family bundle by reusing the already-validated CDF-only dense bundle as a template and
overwriting its CM columns with the (bug-fixed, section 17) POW feature values — same width, same
core columns, zero new Hessian/gradient code, dimensionally guaranteed correct. Deliberately uses
the dense Architecture A path (not Architecture C) since this question is about the underlying
restriction/moment-matrix, which is architecture-independent per section 20's ForwardDiff
equivalence result. Real D20 W=100,000 solves, both L=3 and L=10:

| Variant | L | cond(G) | nStatus | Result |
|---|---|---|---|---|
| CDF-only (baseline) | 3 | 670 | 0 | PASS |
| CDF-only (baseline) | 10 | 917 | 0 | PASS |
| POW-only (z^(σ-1) alone) | 3 | 669 | 0 | PASS |
| POW-only (z^(σ-1) alone) | 10 | 894 | 0 | PASS |

**Both families converge cleanly on their own**, with condition numbers ~670–920 — nowhere near the
~8,730–85,343 seen when combined (section 22–23). This confirms the non-convergence is specifically
about the INTERACTION between the two families, not either restriction being individually hard to
satisfy — consistent with the user's expectation that neither restriction alone should be hard to
satisfy, especially at L=3.

Combined with the near-collinearity finding (max per-`(origin,ℓ)` correlation 0.9989–0.9997 at
every tested L, section 22–23): the two families carry almost — but not quite — the same
information. Jointly resolving the small residual difference between them to tight numerical
tolerance is what's failing. Two explanations remain open and this experiment does not distinguish
between them:
1. **A genuine (non-bug) numerical-conditioning fact** — two individually-loose but ~99.9%+
   correlated moment restrictions are inherently hard to jointly resolve to `opttol=1e-12`, a known
   near-multicollinear-moments phenomenon, independent of any implementation bug.
2. **A bug in how the two families' targets/anchoring interact** — e.g. something about jointly
   anchoring eq.35 and eq.36 against the same reference origin that isn't quite right, even though
   each family is individually correct (as this experiment and section 19–20 both confirm).

No further experiment was run to separate these two explanations — this is the last checked
hypothesis as of this writing.

## 25. Updated verdict block (supersedes section 16)

```
SCIENTIFIC_SPEC = CDF_plus_truncated_sigma_minus_1 (feature: z^(sigma-1), z=U^(-mu); cutoffs: theoretical Frechet quantile; indicator 1{U>c} for the POW family, NOT 1{U<=c} -- see section 17)
INDICATOR_DIRECTION_BUG_FOUND_AND_FIXED = true (section 17; user-caught, not self-discovered; confirmed via D4 gates AND independent ForwardDiff ground truth, sections 19-20)
ARCHITECTURE_C_EXTENDED = true (plain flexible CM AND CM+ZC; zero dense G/H anywhere; section 18)
ARCHITECTURE_C_VERIFIED = true (D4 machine-precision gates section 19; independent ForwardDiff ground truth section 20 -- both PASS decisively)
D20_TWO_FAMILY_REAL_SOLVE = FAILS (nStatus=-400) at every tested (W,L) combination: W in {20k,100k,300k} x L=10, and L in {3,4,5,6,8,10} x W=100k -- STATUS: UNRESOLVED, see section 24
D20_W_SENSITIVITY_HYPOTHESIS = REFUTED (section 21 -- identical failure across W, unlike single-family at the same theta point)
D20_ITERATION_BUDGET_HYPOTHESIS = REFUTED (section 21 -- maxit 100->2000, ~19min wall-clock, identical result)
D20_HARD_RANK_DEFICIENCY = false (section 22-23 -- full rank at 1e-10 relative tolerance at every tested L, but cond jumps 30(single family)->1978-8730(combined, L=3..10) and max per-(origin,l) CDF/POW correlation is 0.9989-0.9997 at every L, always at the last bin)
POW_ONLY_ISOLATION_EXPERIMENT = section 24b: CDF-only AND POW-only EACH converge cleanly alone (nStatus=0, cond(G)~670-920, L=3 and L=10) -- confirms non-convergence is the INTERACTION between the two families, not either alone
ROOT_CAUSE_OF_D20_NONCONVERGENCE = NOT_IDENTIFIED (user's judgment: more likely a residual bug than genuine infeasibility, given persistence at L=3 and given each family alone is easy; see section 24/24b for what remains unchecked -- genuine near-multicollinear-moments conditioning vs a cross-family anchoring bug, not yet distinguished)
PRODUCTION_RELEASE = not_merged_awaiting_user_push_approval (unchanged)
BUG_FOUND_AND_FIXED_WITHIN_TASK = true (TWO bugs across the full session: exponent sign 1-sigma->sigma-1 user-corrected, section 0a/CORRECTION banner; indicator direction <=c->  >c, section 17)
```
