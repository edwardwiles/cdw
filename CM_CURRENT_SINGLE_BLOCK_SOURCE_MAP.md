# CM single-block source map (pre-change), written before any code edit

Mission: add CDW eq.36 (truncated (1-sigma)-power family) alongside the already-implemented
eq.35 (CDF family) in the flexible common-marginals (CM) restriction, growing the CM block from
`(D-1)*L` to `2*(D-1)*L`, reusing all existing operator/Hessian/gradient/verification/checkpoint
machinery.

## 1. Canonical feature-construction site

`full_aod_diag/d4_exact/common_marginals_moments.jl::precalc_common_marginals_cdf` is the ONE
place raw CM features are built, from `ctx.U` (the W x D baseline draw matrix -- already treated
as "z_o(omega)" for this restriction's purposes; see `docs/fullA_common_marginals_handoff.md`
section 2's own eq.35 statement, `1{U_o<=z_l}`) and `z` (L empirical quantile cutoffs of
`U[:,refIndex1]`).

**This function ALREADY has a partial, DEAD, never-wired-to-production `include_truncated_moment`
branch** (kwarg defaults `false`, only call site setting `true` is nonexistent -- grepped, zero
hits). Its formula is WRONG for this task: `pw = muHat*(1-sigmaHat)` (an extra `muHat` factor not
in the paper's eq.36, which is literally `z_o'(omega)^(1-sigma)`). Fix: correct the exponent to
plain `(1-sigmaHat)`, drop the now-unneeded `muHat` kwarg, and make `include_truncated_moment`
into a REQUIRED (no-default) kwarg everywhere it changes the moment layout (repo rule: never
default a scientific parameter) while still allowing `cm_frechet_level.jl`'s deliberate
single-family carve-out to keep passing `false` explicitly.

`n_cm_moments(D,L; include_truncated_moment)` is ALREADY exactly `2*(D-1)*L` when true -- no
change needed, it was already dimension-correct, just never exercised.

Column layout convention (already established by the dead code, adopted as THE chosen ordering
for this task): **CDF-block-then-power-block**, each threshold-major
(`col(l,oi)=(l-1)*nO+oi`): columns `1:(D-1)*L` = eq.35, columns `(D-1)*L+1:2*(D-1)*L` = eq.36.
`cm_block_to_anchored_residuals` already decodes exactly this layout.

## 2. Call chain, traced

`build_cm_augmented_obj` (common_marginals_moments.jl) -- generic wrap via `wrap_moments_with_cm`
(splices precomputed `CM` matrix into `G` by `.=` copy, keyed only on `size(CM,2)` -- ALREADY
100% dimension-driven, zero change needed).

`build_cm_production_context` (cm_production_bundle.jl) -- production entry point. Two
`moment_representation` modes:
  - `:dense_reference`: builds `PsiObjectiveBundleImplicit` (has real dense `obj.H`). When
    `use_archB_moments=true` (default) it swaps in `wrap_moments_with_cm_archB`
    (cm_hessian_architectures.jl), which reconstructs the CM columns EVERY call from bin indices
    via `fill_cm_columns_from_bins!` -- this function hardcodes pure-indicator semantics
    (`Bidx[s,o]<=l`), NOT dimension-driven to a second, weighted feature family. Classified:
    INCORRECT DOWNSTREAM ASSUMPTION for two-family (would silently only ever produce the CDF
    block, wrong width). Decision: two-family flexible CM forces `use_archB_moments=false`, which
    leaves `obj_cm` as `aug.obj_cm` (the plain, already-generic `wrap_moments_with_cm` path) --
    zero new code needed there, just avoid the archB fast-path for now.
  - `:operator`: builds `OperatorPsiBundle` (no dense H at all), FG via `CMLookupState`
    (cm_lookup_kernels.jl) -- an O(W*(D-1)) suffix-sum lookup that is PROVABLY single-family
    (its forward/backward identities exploit `1{U_o<=z_l}` being a pure 0/1 cumulative indicator;
    a weighted indicator does not collapse the same way without a genuinely new per-origin-pair
    weighted-suffix-sum derivation). Classified: INCORRECT DOWNSTREAM ASSUMPTION for two-family.
    Decision: two-family flexible CM MUST NOT select `inner_fg_backend=:cm_lookup` /
    `moment_representation=:operator` -- hard-guarded, not silently wrong.

`cm_hessian_architectures.jl::CMBinHessCtx`/`build_cm_bin_ctx`/`hessian_cm_structured!`
(Architecture C) -- the fast production Hessian. Bin tables `Ttab`/`Stab`/`CT`/`CScum` accumulate
PURE counts/unweighted-E sums keyed on bin membership only; every entry implicitly assumes each CM
column is `1{bin(u)<=l}` (weight 1). Classified: INCORRECT DOWNSTREAM ASSUMPTION for a weighted
second family -- extending it correctly requires additional weighted tables (T12/T22, S2), a
mechanical but nontrivial generalization (same `H=(1/M)C'DC` formula, generalized table entries).
Given the ~15 accumulated specialized sub-backends in this file (winner_bin, drawmajor threaded
variants, CM+ZC H_CZ/H_ZZ), a full-fidelity extension of every backend is out of scope for this
task's time budget. Decision (disclosed limitation, see MASTER.md): two-family CM hard-refuses
`cm_hessian_backend=:structured`; production must use `:dense_reference` (Architecture A,
`cc_algo/PsiObjectiveBundle.jl::hessian!` -- fully generic dense BLAS `C'DC`, needs literally zero
code change for any G content/width) for the two-family spec. This is a real, disclosed
performance regression on the Hessian backend only, not a correctness gap.

`lfix_cm_aware.jl::cm_fixed_contribution`/`cm_fixed_value_contribution` (outer C+/Lfix gradient,
folds `lambda_C*'C_s` into the base dual scalar q0 ONCE per outer point) -- ALSO uses the same
O(W*(D-1)) suffix-sum trick (`apply_contrast`/`suffix_sums`/`cumulative_forward_contribution!`),
called REGARDLESS of `inner_fg_backend` (i.e. even the plain/dense_reference inner-solve path
still uses this fast trick for the one-shot q0 fold). Classified: INCORRECT DOWNSTREAM ASSUMPTION
for two-family (silently drops the whole power-block contribution to q0, corrupting every
per-coordinate outer-gradient probe). Fix: keep the existing suffix-sum call EXACTLY as-is for the
CDF sub-block (bit-identical preservation), ADD a plain BLAS matvec of the already-precomputed
`aug.CM`'s power sub-block against its own lambda slice (a literal, un-optimized evaluation of the
same dot-product definition -- not a new formula, and cheap: O(W*(D-1)*L) done ONCE per outer
point, not per Newton iteration, no per-outer rematerialization). Same duplicated logic, same fix,
needed in `cm_meanzc_production.jl::cm_fixed_contribution_meanzc_layout` (CM+ZC's own inlined
copy of the identical computation, pre-existing duplication, not introduced by this task).

`build_lfix_base_cache_cm`/`_meanzc` themselves, and `composite_gradient_at_fast`/
`composite_gradient_at_Cplus_from_cache` downstream: classified GENERIC-ALREADY-DIMENSION-DRIVEN.
By the envelope theorem the CM block (theta-independent) contributes to the outer gradient ONLY
through the converged `base` state (m*, lambda*) folded into q0 above -- the (g,A_od)-block
gradient formula itself never references CM column count or content. No change needed once q0 is
folded correctly.

`cm_checkpoint.jl` -- versioned schema chain `CMCheckpoint -> V3 -> V4 -> V6 -> V8 -> V9`, each
with `draw_checksum_uniform/transformed` hard-refuse-on-mismatch already, and EVERY prior schema
bump was a *safe, inferable* upgrade (old files could only ever have had one specific value for
the new field). The two-family change is NOT safely inferable this way -- an old V9 file's
`zfree`/`dual_warm_start`/`best_feasible` solved a DIFFERENT (half-width) inner problem. Decision:
add `CMCheckpointV10` (new fields: `cm_moment_spec`, `cm_feature_family_count`,
`cm_feature_schema_version`, `cm_feature_operator_checksum`) and make `load_cm_checkpoint` HARD
ERROR (not upgrade） on any schema < 10, breaking the auto-upgrade chain at this one step only --
existing schema-9-and-earlier files remain readable ONLY by an explicit, clearly-labeled
migration helper that a human must invoke deliberately (mirroring the existing schema-1
`migrate_cm_checkpoint_v1_candidate` precedent), never silently.

## 3. cm_frechet_level.jl / common Fréchet (audit only, not rewired)

Explicitly calls `precalc_common_marginals_cdf(...; include_truncated_moment=false, ...)` --
`marginal_restriction=:common_frechet`'s CM sub-block stays single-family CDF-only by design (its
own separate `D*L` level-anchor block is a structurally different target-based restriction, not a
second flexible-CM feature family in the eq.35/36 sense). Per task instructions, left untouched;
verdict reported as `missing_second_family_separate_task`, not silently claimed complete.

## 4. Classification summary of every `(D-1)*L` / `n_cm ==` / single-family hit (final grep sweep, pre-fix)

- `common_marginals_moments.jl`: feature construction (fixed in this task) + dimension metadata
  (already correct).
- `cm_config.jl`: dimension metadata / dispatch (needs the two-family hard-refuse guard added).
- `cm_lookup_kernels.jl`, `cm_frechet_lookup_kernels.jl`, `cm_frechet_lookup_production.jl`:
  single-family-only lookup FG (frechet -- out of scope, untouched; plain-CM lookup -- hard-guard
  added, not extended).
- `cm_hessian_architectures.jl`: structured Hessian, incorrect downstream assumption for weighted
  features (hard-refuse added for two-family + `:structured`, not extended -- disclosed gap).
- `cm_frechet_cplus.jl`: Fréchet-only C+ gradient (out of scope, untouched).
- `common_marginals_interval.jl`, `c12b_interval_common_marginals_moments.jl`,
  `c12i_validate_interval_equiv.jl`: `:interval` basis, already documented single-family-only,
  independent of this task (production default is `:cumulative`).
- `lfix_cm_aware.jl`: outer-gradient q0 fold (fixed in this task).
- `test_frechet_cm_config_wiring_d4.jl`, `test_frechet_cm_level_basis_d4.jl`: Fréchet tests, out of
  scope, untouched.
