# Draw-design pipeline call graph (2026-07-30)

**Branch:** `architecture/unify-random-draw-production-pipeline-2026-07-30`
**Base:** `production/fullA-exact@cd17235` (latest production commit containing the operator-bundle
hardening release, tag `production-operator-bundle-hardening-release-2026-07-30`; predates the
in-flight `campaign/prepare-sigma3-W500k-five-family-production-2026-07-30` prep commits, which
this task does not touch).
**Method:** direct `rg`/`grep` over the full working tree (not the 2026-07-30 audit zip's estimate)
plus manual reading of every file identified as a real production entry point. Call counts below
are exact grep counts at HEAD of this branch, not extrapolated.

## 1. Public production entry points (draw-related)

| Function | File | Role |
|---|---|---|
| `d20_real_setup_design` | `full_aod_diag/d4_exact/draw_design.jl` | **THE** production-facing entry point. ~100+ call sites across production drivers, campaign checkpoint/stage-runner infrastructure, and diagnostic/test scripts. |
| `d20_real_setup` | `full_aod_diag/d4_exact/context_real_d20.jl` | Pseudorandom-only setup, called internally by `d20_real_setup_design`'s `:pseudorandom` branch. Also called directly by a handful of diagnostic scripts that don't need draw-design selection. |
| `d20_real_setup_qmc` | `full_aod_diag/d4_exact/qmc_context_real_d20.jl` | QMC-only setup. Called internally by `d20_real_setup_design`'s QMC branches, **and directly** (bypassing the resolver) by 5 diagnostic scripts (§4). |
| `build_ad_context_real_d20` | `full_aod_diag/d4_exact/context_real_d20.jl` | Called only from `d20_real_setup` (1 call site). |
| `build_ad_context_real_d20_qmc` | `full_aod_diag/d4_exact/qmc_context_real_d20.jl` | Called only from `d20_real_setup_qmc` (1 call site). |
| `master_prepare_cc` | `prepare_cc/master_prepare_cc.jl` | Called only from `build_ad_context_real_d20` (1 call site). |
| `master_prepare_cc_qmc` | `full_aod_diag/d4_exact/qmc_context_real_d20.jl` | Called only from `build_ad_context_real_d20_qmc` (1 call site). |

**Key finding:** the three duplicated pairs are NOT independently reachable from most of the
codebase -- almost every real caller (production drivers, campaign checkpoints, ~100 test/bench
scripts) goes through the single resolver `d20_real_setup_design`, which already treats
`draw_design` as a dispatch symbol at exactly one point. The duplication is real but structurally
contained to 4 files plus 5 diagnostic bypasses (§4) -- it is not smeared across the repo.

## 2. Call graph (pseudorandom branch, today)

```
d20_real_setup_design(draw_design=:pseudorandom)
  -> Random.seed!(draw_seed)              [global RNG mutation -- see UNIFIED_DRAW_API_CONTRACT]
  -> d20_real_setup(...)                       [context_real_d20.jl]
       -> build_ad_context_real_d20(...)       [context_real_d20.jl]
            -> master_setup(params)             [setup/*.jl, draw-independent]
            -> master_prestep(...)              [draw-independent]
            -> master_prepare_cc(...)           [prepare_cc/master_prepare_cc.jl]
                 -> Random.seed!(seedU)
                 -> U = drawU(SamplingWeight, globalParams)   [prepare_cc/drawU.jl -> genRands.jl]
                 -> createUDerivatives!, buildObjectsForMoments, γHat  [draw-independent given U]
       -> build_theta_gammanorm, theoretical_gammaprime_bounds, FreeParamMap   [draw-independent]
       -> PsiObjectiveBundleImplicit(... threshold_state = ThresholdAbortState(resolve_threshold_for_delta(δ)) ...)
       -> precompute_pairwise_M / build_extreme_draw_witness (build_screen=true)  [screen construction, INLINE]
  -> draw_design_meta(...)   [draw_design.jl]
```

## 3. Call graph (QMC branches, today)

```
d20_real_setup_design(draw_design=:sobol_randomized | :halton_scrambled)
  -> gen = sobol_U | halton_U                   [qmc_draws.jl]
  -> Uexp = gen(W, D; seed=draw_seed)            [draws + exp_from_uniform01 transform, self-contained -- does not mutate caller-visible global RNG (qmc_draws.jl's _with_saved_global_rng wrapper)]
  -> d20_real_setup_qmc(U_injected=Uexp, ...)         [qmc_context_real_d20.jl]
       -> build_ad_context_real_d20_qmc(U_injected=Uexp, ...)  [qmc_context_real_d20.jl]
            -> master_setup(params)              [SAME function as pseudorandom branch]
            -> master_prestep(...)               [SAME function]
            -> master_prepare_cc_qmc(..., U_injected)    [qmc_context_real_d20.jl -- DUPLICATE of master_prepare_cc]
                 -> U = U_injected  (no draw, no seed)
                 -> createUDerivatives!, buildObjectsForMoments, γHat   [SAME functions, byte-identical call]
       -> build_theta_gammanorm, theoretical_gammaprime_bounds, FreeParamMap   [SAME functions]
       -> PsiObjectiveBundleImplicit(... NO threshold_state kwarg -> defaults to ThresholdAbortState() i.e. threshold=Inf ...)   *** DRIFT ***
       -> (no screen construction here -- d20_real_setup_qmc has no build_screen block)   *** DRIFT ***
  -> [back in draw_design.jl] screen-parity patch block: precompute_pairwise_M / build_extreme_draw_witness called AGAIN, a 3rd copy of the same 2 lines   *** DRIFT COMPENSATION ***
  -> draw_design_meta(...)   [draw_design.jl, SAME function]
```

## 4. All draw-design-specific branches after `U` has been generated (the actual defect surface)

Everything in §2/§3 that is NOT marked "SAME function" is a place code branches on draw design
*after* `U` already exists, which violates the target invariant. Enumerated exhaustively:

1. `master_prepare_cc` vs `master_prepare_cc_qmc` -- two full function bodies, ~280 lines each,
   differing only in how `U` enters (drawn vs injected). `full_aod_diag/d4_exact/qmc_context_real_d20.jl:62-354`
   vs `prepare_cc/master_prepare_cc.jl:1-289`.
2. `build_ad_context_real_d20` vs `build_ad_context_real_d20_qmc` -- routing duplicates.
   `qmc_context_real_d20.jl:357-366` vs `context_real_d20.jl:45-54`.
3. `d20_real_setup` vs `d20_real_setup_qmc` -- ~124 vs ~70 lines, drifted (missing
   `threshold_state`, missing screen block). `qmc_context_real_d20.jl:380-450` vs
   `context_real_d20.jl:66-188`.
4. Screen construction, 3 copies: `context_real_d20.jl:159-168` (inline, real), `draw_design.jl:171-177`
   (compensating patch), and *absent* from `qmc_context_real_d20.jl`'s own setup (the actual bug).
5. Five diagnostic scripts call `d20_real_setup_qmc` **directly**, bypassing `d20_real_setup_design`
   entirely -- these are the only callers of the duplicated pipeline outside the 4 files above:
   - `full_aod_diag/d4_exact/c10_stratmarg_screen_sweep.jl` (lines 160, 225)
   - `full_aod_diag/d4_exact/c10_stratmarg_followup_w20000.jl` (line 85)
   - `full_aod_diag/d4_exact/c10_phase7_short_continuation.jl` (line 169)
   - `full_aod_diag/d4_exact/c10_phase7_qmc_wiring_smoketest.jl` (lines 28, 54, 55)
   - `full_aod_diag/d4_exact/c10_phase7_qmc_precision_comparison.jl` (line 129)

   All five pass a caller-constructed `U_injected` matrix (either a pseudorandom matrix built with
   an explicit seed for a same-seed A/B check, or a QMC matrix) -- i.e. every one of them is
   exactly the `PrecomputedDrawDesign` use case the target architecture (task §10) describes. They
   currently have no way to reach that behavior except by calling the duplicated `_qmc` function
   directly, because `d20_real_setup_design` has no "inject a precomputed matrix" option today.
6. `exp_from_uniform01` (the shared Exp(1) inverse-CDF transform, `-log(1-u)`) is defined once in
   `qmc_context_real_d20.jl:46-49` but is logically shared infrastructure -- `qmc_draws.jl` (the
   legitimate QMC generators) and `c10_stratmarg_draws.jl` (a diagnostic generator) both call it,
   so it cannot simply be deleted with the rest of `qmc_context_real_d20.jl`; it needs a new home
   that isn't itself a "QMC context" file (see reachability doc §3).

## 5. What is already correctly architected (no action needed)

- **Campaign scripts**: `campaign_cm_family_runner.jl`, `campaign_unrestricted_runner.jl`,
  `campaign_cell_io.jl` are fully generic over `draw_design`. No `*_QMC` campaign script exists
  anywhere in the repository (confirmed by `find -iname "*campaign*"` and `-iname "*QMC*"` across
  the full tree) -- there was nothing to consolidate at the campaign-driver level.
- **Checkpoint provenance**: `cm_checkpoint.jl`, `cm_originzc_checkpoint.jl`,
  `cm_checkpoint_fingerprint.jl`, `c10_d20_production_driver*.jl` all carry `draw_design::Symbol`
  as one struct field among many (checkpoint identity / resume-guard metadata) -- this is exactly
  the "provenance metadata, not a numerical branch" pattern the target architecture requires.
  `guard_checkpoint_path` and `reuse_matches` compare `draw_design` for *equality* (does this
  checkpoint match what was asked for), never dispatch different code on its value.
- **Draw generators** (`qmc_draws.jl::pseudorandom_U/halton_U/sobol_U`, `prepare_cc/drawU.jl`,
  `prepare_cc/genRands.jl`, `cc_algo/rhalton.jl`) are correctly design-specific -- this is the one
  place design-specific code SHOULD live (`DRAW_GENERATION_REQUIRED`).

## 6. Historical drift already observed (confirms the audit's "drift risk" claim)

Git-blame-visible history on `qmc_context_real_d20.jl` shows two features were added to
`d20_real_setup`/`context_real_d20.jl` after the QMC fork existed and were not propagated:
Part A (`row_idx`/`destination_sample`, 2026-07-24, *was* eventually ported to the QMC file on
2026-07-26 per its own header comment) and Part C (`threshold_state`, added to the non-QMC file,
never ported). The screen-construction gap was patched at a third location instead of at the
source. This is exactly the failure mode the target architecture (single post-draw pipeline)
eliminates structurally rather than by discipline.
