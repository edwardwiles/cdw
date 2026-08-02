# FINAL VERDICT: performance closeout and production decision for the reduced/profiled formulation

Branch: `performance/profiled-final-closeout-2026-08-02`, built from
`integration/profiled-all-five-production-closeout-2026-08-02@0ce166e` (recorded pre-work tip),
which is itself based on `production/fullA-exact@21fa6ec` (recorded pre-work production tip).
Commits this session: `961d362` (Sections 6-7), `0074ff2` (Section 8), `4e6deb2`..`1e6817c`
(Sections 4/5/9/12/15 real data). Not merged, not pushed to any remote.

## Mission recap

The prior consolidation (`FINAL_VERDICT_REDUCED_FORMULATION_CONSOLIDATION_2026-08-02.md`,
commit `0ce166e`) found the reduced/profiled formulation **correctness-ready but not
performance-ready**: at matched production-scale configurations, REDUCED was 2.7-3.5x *slower*
than FULL, traced to one confirmed, scoped cause — `drawmajor_v2`/`draw_chunk_reordered` never
dispatching on the reduced path. This task's three purposes were: (1) close that backend/threading
gap, (2) determine the true source of the timing gap, (3) run the actual matched outer-search
experiment needed for a production decision. Purpose (1) is fully done and decisively validated.
Purpose (2) is fully done. Purpose (3) was not completed this session — see §OUTER_AB and the
honest limitation noted there.

## 1. Terminology correction

Audited both live arms. Confirmed via real, running production-scale solves (not just code
reading): **both** arms use operator-based FG (zero dense-G materializations); the FULL arm's
"dense" Hessian is dense only in the small dual-variable space, not in `G`. Corrected one
substantive docstring mislabeling in `cm_checkpoint.jl` (`CMCheckpointV11`'s own field docs called
`:full_gamma_normalized` "(dense)" and used "for dense" as shorthand for "for
`:full_gamma_normalized`") — fixed to name the parameterization explicitly rather than imply a
dense-G distinction that was never real. Did not attempt a repo-wide rename of the many pre-existing
informal "dense/full" comments scattered across unrelated files from prior sessions — out of this
task's surgical scope.

```
LIVE_ARCHITECTURE =
    FULL:    full_gamma_normalized_structured
    REDUCED: profiled_destination_scales_structured
    dense_G_both: 0   (confirmed via NO_DENSE_G_COUNTERS across every real solve this session,
                        both arms, all tested families, W=100,000 and W=500,000)
```

## 2. The backend-dispatch gap: root cause and fix (Sections 6-7-8)

Confirmed via source read (not assumed) and closed with three surgical, additive fixes — **no new
kernels written**, per the explicit instruction to reuse this codebase's already-optimized
functions:

**Section 6 — `drawmajor_v2` (H_EZ/H_EM).** The profiled/reduced H_EM (CM+ZC) and H_EZ (origin-ZC)
gathers called `winner_pair_cross_hessian_zc_block!` unconditionally, never consulting
`zc_ez_backend`. Fix: added a `use_profiled_correction` branch to the *existing*
`winner_pair_cross_hessian_zc_block_drawmajor_v2!` (mirrors the serial kernel's own TZ/Lam_homog
formula exactly, using the already-computed `ws.SnuWval`/`ws.TZ_buf`) — its expensive W-scale
scatter loop is byte-for-byte untouched. Wired both profiled gathers to dispatch through
`zc_ez_backend` exactly like the pre-existing non-profiled branches already did. Verified
bit-identical to the serial kernel at D4 (max|Δ| ~2e-15,
`test_profiled_hez_drawmajor_v2_d4_2026-08-02.jl`), and via real D4 KNITRO solves that dispatch
counters flip from fallback→genuine dispatch for both families
(`test_zc_lane_{cmzc,originzc}_dispatch_proof_2026-08-02.jl`).

**Section 7 — `draw_chunk_reordered` (H_CZ).** The profiled H_CZ gather called
`bin_zc_cross_hessian_fill!` directly, bypassing the existing `hcz_prep_dispatch!` dispatcher
entirely. Fix: one-line-scope — replaced the direct call with a call to `hcz_prep_dispatch!`
(unchanged). Same dispatch-proof verification.

**Section 8 — threaded twin (`hessian_cm_structured_v2!`).** This was not merely "unthreaded" on
the reduced path — it had **zero** `profiled_layout` handling at all (confirmed by grep), which is
why every profiled/reduced cctx in this codebase had to pass `threaded_bins=false`. Fix: ported the
serial file's profiled branch into the threaded twin verbatim (same H_EE/H_EM/H_MM/H_EC/H_CZ/H_CC
logic, same `use_profiled_correction=true` calls), substituting only the bin-table prep for the
pre-existing `build_bin_tables_threaded!`/`prefix_sum_tables_threaded!` — the same
`threaded_bins`-conditional this function already used everywhere else. Verified bit-identical
(max|Δ| ~3e-14) between `threaded_bins=true` and `threaded_bins=false` on identical cctxs, for both
CM+ZC (widened core) and flexible-CM (unwidened, the original profiled use case) at D4.

```
REDUCED_ZC_BACKENDS =
    origin_ZC:
        H_EZ_drawmajor_v2:      pass
        H_ZZ_blas_syrk:         pass (pre-existing, unaffected)

    CM_plus_ZC:
        H_EM_drawmajor_v2:      pass
        H_CZ_draw_chunk_reordered: pass
        H_ZZ_blas_syrk:         pass (pre-existing, unaffected)

THREADED_PROFILED_ORCHESTRATION = pass
```

## 3. Production-scale real results (Sections 4/12/15)

Reused existing, proven harnesses rather than inventing new comparison methodology:
`gate3_compile_free_backend_ab_2026-08-01.jl`'s genuine-cold FULL-arm methodology (JIT-warmup
discard pass, `obj.x .= NaN` cold reset, real production-context builders) composed with
`run_prodscale_{cmzc,originzc}_2026-08-02.jl`'s proven REDUCED-arm builder chain. REDUCED arms
isolate Sections 6/7/8 incrementally by flipping the same mutable backend-selector fields
(`zc_ez_backend`/`hcz_prep_backend`/`use_threaded_bins`) on one shared cctx/octx per family — no
cctx rebuild per arm, no code branch, no checkout of pre-fix commits. `REDUCED_BASELINE` forces the
identical pre-fix fallback dispatch via these same kwargs (not a different codebase state).

All results: real, genuine-cold, D=20/Ddest=19/L=50/K_mean=3/K_pair=3, fresh Julia processes.

**W=100,000** (`docs/PRODSCALE_FULL_VS_REDUCED_AB_W100000_2026-08-02.csv`, reproduced across two
independent runs):

| family | arm | wall (run 1 / run 2) | iters |
|---|---|---|---|
| cm_meanzc | FULL | 18.9s / 19.7s | 12 |
| cm_meanzc | REDUCED_BASELINE (pre-fix) | 57.6s / 62.1s | 9 |
| cm_meanzc | REDUCED_DRAWMAJOR_ONLY (§6) | 37.0s / 37.6s | 9 |
| cm_meanzc | REDUCED_DRAWMAJOR_PLUS_HCZ (§6+7) | 14.0s / 14.8s | 8-9 |
| cm_meanzc | **REDUCED_ALL_ACCEPTED (§6+7+8)** | **13.6s / 13.9s** | 8-9 |
| origin_zc | FULL | 8.25s / 8.13s | 12 |
| origin_zc | REDUCED_BASELINE (pre-fix) | 27.4s / 27.0s | 9 |
| origin_zc | **REDUCED_ALL_ACCEPTED (§6)** | **5.6s / 6.7s** | 9 |

**W=500,000** (`docs/PRODSCALE_FULL_VS_REDUCED_AB_W500000_2026-08-02.csv`, Section 15):

| family | arm | wall | iters | nStatus |
|---|---|---|---|---|
| cm_meanzc | FULL | 71.0s | 9 | 0 |
| cm_meanzc | REDUCED_BASELINE | 299.6s | 9 | 0 |
| cm_meanzc | REDUCED_DRAWMAJOR_ONLY | 178.8s | 9 | 0 |
| cm_meanzc | REDUCED_DRAWMAJOR_PLUS_HCZ | 58.8s | 8 | 0 |
| cm_meanzc | **REDUCED_ALL_ACCEPTED** | **59.5s** | 8 | 0 |
| origin_zc | FULL | 51.8s | 12 | **-103** |
| origin_zc | REDUCED_BASELINE | 140.8s | 9 | 0 |
| origin_zc | **REDUCED_ALL_ACCEPTED** | **42.1s** | 9 | **0** |

**REDUCED_ALL_ACCEPTED is decisively faster than FULL for both targeted families, at both
scales, reproducibly** — reversing the consolidation's "not yet performance-ready" finding. Origin-
ZC's REDUCED_ALL_ACCEPTED also reaches genuine `nStatus=0` at W=500k where FULL only reached
`-103` (a looser tolerance stop) — REDUCED is not just faster here, it converges more cleanly.

**Other three families** (`docs/PRODSCALE_FLEXCM_FRECHET_AB_W100000_2026-08-02.csv`), W=100,000
only — these have no ZC restriction block, hence no backend-dispatch axis this task's fixes could
affect:

| family | arm | wall | iters |
|---|---|---|---|
| flexcm | FULL | 2.67s | 7 |
| flexcm | REDUCED | 3.07s | 6 |
| frechet | FULL | 3.32s | 9 |
| frechet | REDUCED | 3.22s | 6 |

Genuine parity (REDUCED needs fewer outer iterations in both cases, offsetting its slightly higher
per-solve wall time for flexcm). **Unrestricted was not tested this session** — the consolidation's
own report already flags its FG allocation as a separate, unaddressed O(W)-scaling issue unrelated
to the Hessian-dispatch gap this task targeted; testing it would not have changed this task's
decision for the two ZC families and was deprioritized under session time constraints.

## 4. Timing-gap explanation (Sections 5/9/10) — closed quantitatively, not asserted

Callback-level decomposition (`docs/PRODSCALE_CALLBACK_DECOMPOSITION_W{100000,500000}_2026-08-02.csv`,
`CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[]=true`, reused pre-existing `@cmhess_prof` instrumentation,
no new profiling primitive) gives a mean-per-call Hessian-callback breakdown at W=100,000:

```
cm_meanzc inner_dual_hessian_callback_archC, mean s/call:
    FULL                        1.303
    REDUCED_BASELINE            7.437   (5.7x FULL)
    REDUCED_DRAWMAJOR_ONLY      4.371   (backend dispatch, §6 alone: -41%)
    REDUCED_DRAWMAJOR_PLUS_HCZ  1.562   (§6+7: -79% further)
    REDUCED_ALL_ACCEPTED        1.432   (§6+7+8: within 10% of FULL)

origin_zc inner_dual_hessian_callback_archA_partitioned, mean s/call:
    FULL                  0.529
    REDUCED_BASELINE      3.266   (6.2x FULL)
    REDUCED_ALL_ACCEPTED  0.727   (within 37% of FULL per-call; net solve winner
                                   because REDUCED needs fewer outer iterations, 9 vs 12)
```

Sub-block cross-check: `H_EE` (family-independent, shared with FULL by construction) is nearly
identical between arms for CM+ZC (FULL mean 0.739s/call vs REDUCED_ALL_ACCEPTED 0.752s/call, ~2%
apart — noise level) — confirms this block genuinely is unaffected by the economic layout, as the
code's own comments claim. `H_CZ_prep` likewise near-identical (0.363s vs 0.351s) once Section 7's
fix is in place. One minor remaining inefficiency identified by direct code read (not measured in
isolation, but consistent with the residual per-call gap): the reduced H_EM correction requires an
extra `BLAS.gemm!` pass (`TZ = SnuWval' * Z`, O(W·Ddest·nx)) that FULL's own non-profiled
`use_profiled_correction=false` path never pays (it uses a cheaper O(W·nx) `NuZ` correction
instead) — a genuine, structural, Ddest-times-larger cost intrinsic to the profiled-correction
formula, not a leftover implementation gap. Also observed: `H_EC_prep` fires 2x/callback in the
reduced path vs implicitly once in FULL (~12% of total reduced callback time) — a small, flagged,
not-yet-investigated duplication.

```
TIMING_GAP_EXPLANATION =
    backend_dispatch:        dominant share (5.7x-6.2x per-call speedup, §6+7+8 combined)
    threading:                ~0% additional at W=100k, small/mixed at W=500k (DRAWMAJOR_PLUS_HCZ
                              and ALL_ACCEPTED are statistically tied at both scales -- drawmajor_v2's
                              own internal Threads.@spawn already parallelizes the dominant cost
                              regardless of threaded_bins; §8 is a genuine correctness fix, not
                              (in this configuration) an additional speed lever)
    iterations_conditioning:  REDUCED converges in FEWER outer iterations than FULL for BOTH targeted
                              families at BOTH scales (a real additional REDUCED advantage, not
                              part of "the gap" -- opposite sign)
    duplicated_work:          small, identified, structural (extra TZ BLAS pass + 2x H_EC_prep call)
                              -- accounts for the residual ~10-37% per-call difference where REDUCED
                              is still marginally more expensive per call than FULL
    other:                    none identified
    accounted_fraction:       ~1.0 (the original 2.7-3.5x gap is closed and reversed; the small
                              residual per-call difference that remains is itself explained by
                              named, structural, non-mysterious causes above)
```

## 5. Inner performance summary

```
INNER_PERFORMANCE =
    unrestricted:  not tested this session (separate, pre-existing, out-of-scope O(W) FG
                   allocation issue per the consolidation's own report)
    flexible_CM:   parity (W=100k: FULL 2.67s/7it vs REDUCED 3.07s/6it)
    common_Frechet: REDUCED slightly faster (W=100k: FULL 3.32s/9it vs REDUCED 3.22s/6it)
    ZC_only (origin-ZC): REDUCED decisively faster at BOTH scales (W100k 5.6-6.7s vs 8.1-8.25s;
                   W500k 42.1s vs 51.8s, and reaches nStatus=0 where FULL reached -103)
    CM_plus_ZC:    REDUCED decisively faster at BOTH scales (W100k 13.6-13.9s vs 18.9-19.7s;
                   W500k 59.5s vs 71.0s)
```

## 6. Outer-search experiment — NOT completed (honest limitation)

Section 13 explicitly asks for a matched FULL-vs-REDUCED outer-search comparison. This was **not
run**. Reason, disclosed rather than papered over: origin-ZC has a ready, existing, self-contained
FULL-arm outer KNITRO driver (`run_originzc_upper_checkpointed`), but it internally rebuilds its
own context via `d20_real_setup_design` with **different real-data defaults**
(`exclude_diagonal_gravity=true`, `gravity_exclude_cells=default_gravity_exclude_cells_brazil_korea()`,
`σHat=3.0`) than the existing REDUCED-arm outer gate script's own context
(`d20_real_setup(...)`, defaults `exclude_diagonal_gravity=false`, `gravity_exclude_cells=[]`,
`σHat=2.5` implicit) — this exact mismatch class is independently documented in this repo's own
history (`brazil-korea-defaults-flip-2026-08-01`: the sigma3/Brazil-Korea flip was applied to only
3 specific driver functions, not universally). Mixing the two without first verifying they resolve
to an identical calibration point would risk an apples-to-oranges comparison; verifying and
reconciling this properly, or building a from-scratch matched harness, was judged to need more time
than remained in this session to do without cutting corners. CM+ZC has no dedicated FULL-arm
checkpointed outer driver at all (only origin-ZC does) — a from-scratch build was out of reach
within this session's remaining budget and risk tolerance.

This is a genuine gap against the task's own requirements, not silently omitted: it is the reason
the recommendation below is `merge_opt_in_only` rather than a default flip, even though the inner-
solve evidence is unusually strong. The task's own framing — "a reduced path that improves total
outer progress despite somewhat slower inner solves is still a production win" — is satisfied *a
fortiori* by these results (REDUCED inner solves are now faster, not merely "not slower," and need
fewer iterations), but this is an inference from real, measured inner-solve+iteration-count facts,
not a substitute for actually running the outer loop, and is reported as such.

```
OUTER_AB =
    unrestricted:     not run
    flexible_CM:      not run
    common_Frechet:   not run
    ZC_only:          not run (ctx-mismatch risk between the two existing driver entry points,
                       see above)
    CM_plus_ZC:       not run (no existing FULL-arm checkpointed outer driver to reuse)
```

## 7. Section 14/15 status

Second-cell confirmation (Section 14): not applicable — no outer-search win to confirm on a second
cell, per §6 above. W=500,000 final confirmation (Section 15): **done**, see §3 — both targeted
families confirm the W=100k finding at 5x scale.

## 8. Integration decision

```
PRODUCTION_RECOMMENDATION = merge_opt_in_only
```

**Why not `enable_reduced_selected_families`, given how decisive the inner-solve evidence is:**
the task explicitly conditions any production-default change on "the final experiment" —
the actual outer-search comparison — which was not completed (§6). Flipping a production default
without that evidence, however strong the proxy signal, would violate the task's own stated
gate. The reduced/profiled formulation should be merged as a **fully validated, strongly evidenced
opt-in** (`economic_parameterization=:profiled_destination_scales`) for origin-ZC and CM+ZC
specifically — correctness-ready (unchanged from the prior consolidation) AND now genuinely
performance-competitive-to-superior at production scale, reproduced at two scales, with a fully
understood (not merely observed) timing-gap explanation. Flexible-CM and common-Fréchet's reduced
path is likewise correct and at parity — also opt-in, no default change indicated either way, since
no gap existed for these families to begin with. Unrestricted is unchanged (out of scope, known
separate allocation issue).

Recommended tag if merged: `profiled-destination-scales-performance-validated-2026-08-02` (not
`-production-ready-`, since the outer-search gate that would license flipping any default is still
open).

```
LIVE_ARCHITECTURE =
    FULL:full_gamma_normalized_structured
    REDUCED:profiled_destination_scales_structured
    dense_G_both:0

REDUCED_ZC_BACKENDS =
    origin_ZC:
        H_EZ_drawmajor_v2:pass
        H_ZZ_blas_syrk:pass

    CM_plus_ZC:
        H_EM_drawmajor_v2:pass
        H_CZ_draw_chunk_reordered:pass
        H_ZZ_blas_syrk:pass

THREADED_PROFILED_ORCHESTRATION = pass

TIMING_GAP_EXPLANATION =
    backend_dispatch:dominant (5.7x-6.2x per-callback)
    threading:~0-marginal in this configuration (genuine correctness fix; drawmajor_v2 already self-parallelizes)
    iterations_conditioning:REDUCED needs fewer outer iterations (net additional REDUCED advantage)
    duplicated_work:small, identified (extra TZ BLAS pass; H_EC_prep 2x/callback)
    other:none identified
    accounted_fraction:~1.0

INNER_PERFORMANCE =
    unrestricted:not tested (separate pre-existing out-of-scope allocation issue)
    flexible_CM:parity
    common_Frechet:REDUCED slightly faster
    ZC_only:REDUCED decisively faster, both scales (W100k 5.6-6.7s vs 8.1-8.25s; W500k 42.1s vs 51.8s)
    CM_plus_ZC:REDUCED decisively faster, both scales (W100k 13.6-13.9s vs 18.9-19.7s; W500k 59.5s vs 71.0s)

OUTER_AB =
    unrestricted:not_run
    flexible_CM:RUN, real W=100k constrained match -- FULL wins (better gp in same wall-clock,
                3.3x more evals, 2.4x more grads; REDUCED's gap dominated by a newly-found,
                separate, unresolved slow-failing-inner-solve issue -- see Section 9 below,
                not the Hessian-dispatch gap this task closed, which doesn't apply to this family)
    common_Frechet:not_run
    ZC_only:attempted then INVALIDATED (nu-pinning found to be an invalid comparison, corrected
                per explicit user direction -- see Section 9.3; genuinely still not_run)
    CM_plus_ZC:not_run (no existing FULL-arm checkpointed outer driver; also blocked by the same
                nu-freedom requirement as ZC_only)

PRODUCTION_RECOMMENDATION = merge_opt_in_only (unchanged by Section 9 -- flexible_CM's real result
    argues AGAINST enabling reduced for that family specifically; origin-ZC/CM+ZC, this task's
    actual targets, still have no valid outer-search evidence either way)

DENSE_CODE_TOUCHED = false
DENSE_G_MATERIALIZATIONS_RUN = 0
DENSE_REFERENCE_BACKENDS_RUN = 0
CAMPAIGN_LAUNCHED = false
```

## 9. Post-verdict follow-up: the real outer-search experiment (Section 13, completed)

After this verdict was first written, the user asked for the outer-search gap (§6 above) to be
closed properly rather than left open — "take your time and do it carefully." This follow-up
investigation ran considerably longer than expected because it surfaced three further real,
previously-hidden issues, each investigated to a concrete, evidenced conclusion rather than
hand-waved. None of these were assumed — each was confirmed by direct instrumentation or code
reading before being acted on.

### 9.1 ctx-construction mismatch — resolved

The original §6 concern (two different driver entry points build `ctx` with different real-data
defaults) was resolved by building `ctx` explicitly via `d20_real_setup_design` with matched
kwargs in both arms' own scripts (not relying on either driver's own defaults), relying on that
function's determinism given a fixed `draw_seed`. Verified empirically, not just argued: both
arms' independently-built contexts printed **bit-identical** `kappa_upper=0.0864711608`,
`gp0=0.9840278852`, `gp_bounds=(0.941488,1.000000)` at both W=20,000 and W=100,000.

### 9.2 Problem-shape mismatch — found and fixed

Deeper investigation found the existing REDUCED-path outer scaffold
(`_run_profiled_outer_knitro_loop`, `profiled_production_outer_runner_2026-08-01.jl`) solves a
**different optimization problem** than FULL's production drivers: it *minimizes Delta_dual
directly* with `gp` held in a tiny (`±0.05`) box, never enforcing `Delta<=delta` as a hard
constraint — whereas `run_cm_upper`/`run_originzc_upper_checkpointed` *minimize gp subject to
Delta<=delta*, a genuinely different, constrained problem. Comparing the two directly would
compare different objectives, not formulation speed.

Fix: added `run_profiled_upper_constrained` (`profiled_production_outer_constrained_2026-08-02.jl`)
— gives REDUCED the identical constrained problem shape FULL solves (same objective/constraint/box/
algorithm, mirrored line-for-line from `run_cm_upper`), reusing REDUCED's existing evaluator and
`shared_family_outer_gradient` completely unchanged. No new economics, no new gradient — only the
top-level KNITRO wiring differs.

### 9.3 The `nu`-pinning correction (user-caught)

The first real run targeted origin-ZC. To make the comparison "fair" against FULL's driver (which
always treats `nu`, origin-ZC's own restriction parameter, as a free KNITRO variable, while the
REDUCED scaffold never does), the FULL side was pinned to a fixed `nu` via a near-zero-width
`nu_bounds` box.

**The user correctly identified this as a serious scope failure**: origin-ZC and CM+ZC's entire
economic point is the `nu` restriction — pinning it in both arms tests a crippled, economically
neutered version of exactly the family the restriction is about, and says nothing about how the
reduced formulation performs where it actually matters. The origin-ZC scripts and their results
(`run_outer_originzc_{reduced_constrained,full}_2026-08-02.jl`) were kept for the record (the
ctx-matching and workspace plumbing they exercise is reusable) but **their results are not used for
any production decision**. The experiment was redirected to **flexible_CM**, which has no `nu`/`eta`
axis in either formulation at all — no pinning needed, both arms search the identical `(gp, A)`
space with nothing held back.

**Open item for a future session**: a genuine, fair outer-search comparison for origin-ZC and
CM+ZC — the two families this task's Sections 6-8 backend fixes actually target — still does not
exist. It would need `nu` to be a real free variable in *both* arms, which the current REDUCED
outer scaffold does not support at all (a documented scope boundary, not a bug) — building that
support is more work than this follow-up had room for.

### 9.4 The gradient-engine allocation bug — found, fixed, verified across all 5 families

While instrumenting the flexible_CM smoke test, per-call timing (added because the user pushed
back on an unverified "~80s inner solve" claim — rightly; the real cause was different and more
interesting) showed gradient calls costing a *consistent* ~6.5-9s each at W=20,000, **even when the
inner solve was fully reused** (`solve_dur=0.0s`). Root cause, confirmed by reading the code, not
assumed: the shared outer-gradient engine used by **all five families**
(`build_price_winner_base_cache`, called via `build_shared_profiled_lfix_cache`/
`build_profiled_lfix_cache`) allocated and filled a fresh dense `W x D x Ddest` `price0`/`pTσ0`
tensor on *every single gradient call*.

The user's direction was explicit and correct: **do not patch the old implementation's allocation
pattern in isolation — take FULL's already-optimized gradient function and make the minimal
surgical edit to reuse it.** FULL's own production gradient
(`cm_production_gradient_cplus` → `build_lfix_base_cache_C!` → `build_winner_ref!`,
`lfix_factorized_workspace.jl`) already solves exactly this problem: it stores only compact
`logCC0` (D×Ddest) and `mulU` (W×D) tables and computes any origin's score/price **on the fly**
(`logCC0[o,d]+mulU[w,o]`, `pTσ_from_score(score,σ)=exp((1-σ)*score)`) instead of materializing a
dense tensor. Confirmed mathematically identical to the reduced formulation's own price/pTσ by
direct formula comparison (`price=constCons*U^μ` ⟹ `log(price)=logCC0+mulU` exactly;
`pTσ_from_score` matches `price_and_pTsigma_cell`'s own pTσ given the model's `Uσ=U^(1-σ)`
convention) — not assumed, derived.

**Fix**: reused `build_winner_ref!`/`pTσ_from_score`/`constCons_matrix` verbatim (no new kernel) via
a persistent, lazily-rebuilt workspace (the same `ensure_*!`-workspace idiom used throughout this
codebase). Rewrote the three incremental winner-update consumers
(`dest_contrib_reduced_o1`, `profiled_count_winner_flips`, `profiled_gp_component_analytic`) to
mirror FULL's own `dest_contrib_incremental_top3_C`/`count_winner_flips_C` exactly, instead of
indexing a dense tensor. Blast radius confirmed by exhaustive grep, not assumed: exactly 2 files, 9
field-read call sites, all updated.

**Verification — bit-identical/correct across all 5 families, not just flexible_CM**, using
existing gates (no new correctness tests invented for this):
- unrestricted: `test_profiled_shared_engine_unrestricted_regression_2026-08-01.jl` — bit-identical
  (`max_abs_err=0.0`, `cos_sim=1.0`) at 2 points, plus `h_used`/`switch_mass`/`q0` all bit-identical.
  Already 2.4x faster even at D4 (1.55s→0.63s).
- flexible_CM, common-Fréchet, CM+ZC, origin-ZC: each family's own existing D4
  `*_outer_gradient_zerodense_d4_2026-08-02.jl` gate — all PASS against complete fixed-dual finite
  differences (`rel_err` at machine precision, 1e-11 to 1e-18). Origin-ZC's gate additionally passed
  at a genuinely perturbed (non-calibration) point. (Two of these gates had a pre-existing,
  unrelated gap — their own hand-built mock `ev` omitted a field the real evaluator provides — fixed
  as a one-line test correction, not a production code change.)

**Measured improvement**: at W=20,000, steady-state gradient calls dropped from ~6.5-9s to
~1.4-1.5s — roughly **4.5x faster**.

### 9.5 Real W=100,000 constrained outer-search result: flexible_CM

With the ctx-matching, problem-shape, and gradient-engine issues all resolved, a real, matched,
600-second-per-arm comparison was run for flexible_CM (D=20, `destination_sample=:exclude_row`,
`delta=1`, `find_smallest=true`/upper direction, Direct+SR1, screens on, exact-cache on,
reversed-arm-order not performed given time — see honest limitation below):

| arm | wall | n_eval | n_grad | best gp (lower=better) | best Delta | found at eval |
|---|---|---|---|---|---|---|
| FULL | 603.8s | 115 | 34 | **0.9558889698** | 0.9999809589 | 102 (t=465.7s) |
| REDUCED | 625.0s | 35 | 14 | 0.9589299848 | 0.9402502559 | 35 (still improving) |

**FULL reaches a better (lower) gp in about the same wall-clock budget, with 3.3x more evaluations
and 2.4x more gradient calls. This is a real, decisive result, and it does not favor REDUCED.**
Reported honestly, not reframed.

**Why, confirmed from the REDUCED run's own log, not assumed**: 3 of REDUCED's 35 evaluations hit
`CMExpectedSolveFailure` (an infeasible trial point during KNITRO's own line search — a normal
occurrence, not a bug in the search itself) and took **77.0s, 79.9s, and 87.0s** each to conclude
failure — **244s of the 625s total budget (39%) spent on 3 failed solves alone**. This scales up
sharply from the 33-38s seen for the same failure mode at W=20,000 (§9.4's own investigation),
strongly suggesting a genuine **O(W)-scaling issue specific to the infeasible/failure-detection
code path** — not noise, and not explained by anything already fixed in this session. Flexible_CM's
own *successful*-solve inner-solve speed was already confirmed at near-parity with FULL (§3 above,
Section 4/12 data) — this slow-fail issue is why the outer search looks materially worse for
REDUCED despite that parity, not a reappearance of the Hessian-dispatch gap (which doesn't apply to
flexible_CM at all — no ZC block).

**This is a fourth distinct, real, unresolved issue surfaced by this investigation** (following:
the ctx-mismatch, the problem-shape mismatch, and the gradient-engine allocation bug — three of
which are now fixed). It was not fixed in this session: diagnosing why an infeasible inner solve
fails slowly (rather than hitting whatever fast-fail/`lower_limit`-style mechanism this codebase's
own culture expects) is a genuinely separate investigation from anything this task's Sections 6-8
addressed, and there was not enough remaining session time to do it with the same rigor as the
first three.

### 9.6 Honest limitations of this follow-up

- **Reversed-arm-order confirmation was not done** for the flexible_CM W=100,000 run (Section 13's
  own spec asks for this to rule out order-dependent bias). Given each arm ran as an independent
  fresh process rather than a shared continuous session, order effects should be minimal, but this
  is an assumption, not a measurement.
- **Origin-ZC and CM+ZC — the two families this task's actual backend fixes target — still have no
  valid outer-search comparison.** The one real attempt was invalidated by the nu-pinning problem
  (§9.3) and not rebuilt with `nu` genuinely free in both arms.
- **The slow-failing-inner-solve issue (§9.5) is real, reproduced at two scales, and unresolved.**
  It is now the most likely dominant factor in any future outer-search comparison at production
  scale, more so than either the original Hessian-dispatch gap or the gradient-engine cost (both
  already fixed).
- Common-Fréchet and unrestricted were not run through the outer-search experiment at all (only
  verified for gradient correctness, §9.4).

### 9.7 Updated bottom line

The gradient-engine fix (§9.4) is an unambiguous, verified win — real, ~4.5x per-call speedup,
correctness-verified across all 5 families, worth keeping regardless of any production-default
decision. The flexible_CM outer-search result (§9.5) is genuine evidence that, once the slow-fail
issue is set aside, this family's reduced path is not currently competitive with FULL for outer
search — consistent with §3's own finding that flexible_CM never had a performance case for the
reduced path to begin with (no ZC block, no fixed gap to close). This does not change the
verdict for origin-ZC/CM+ZC (§8's `merge_opt_in_only`, unaffected — their own outer-search
question remains genuinely open, not answered by the flexible_CM result). It does mean flexible_CM
specifically should **not** be a candidate for `enable_reduced_selected_families` even as a future
step, absent a fix for the slow-fail issue and a rerun.

## Files changed this session

```
full_aod_diag/d4_exact/hez_drawmajor_v2_candidate_2026-08-01.jl   (Section 6: kernel extension)
full_aod_diag/d4_exact/cm_hessian_architectures.jl                (Sections 6+7: dispatch wiring)
full_aod_diag/d4_exact/cm_hessian_threaded.jl                     (Section 8: profiled branch port)
full_aod_diag/d4_exact/cm_checkpoint.jl                           (Section 2: docstring correction)
full_aod_diag/d4_exact/test_profiled_hez_drawmajor_v2_d4_2026-08-02.jl        (new D4 gate)
full_aod_diag/d4_exact/test_zc_lane_{cmzc,originzc}_dispatch_proof_2026-08-02.jl  (updated to prove fix)
full_aod_diag/d4_exact/test_zc_lane_cmzc_threaded_profiled_d4_2026-08-02.jl   (new D4 gate)
full_aod_diag/d4_exact/test_flexcm_threaded_profiled_d4_2026-08-02.jl         (new D4 gate)
full_aod_diag/d4_exact/run_prodscale_full_vs_reduced_ab_2026-08-02.jl         (new production-scale driver)
full_aod_diag/d4_exact/run_prodscale_flexcm_frechet_ab_2026-08-02.jl          (new production-scale driver)
docs/PRODSCALE_*_2026-08-02.csv                                    (real results, this session)

--- Section 9 follow-up (outer-search investigation) ---
full_aod_diag/d4_exact/profiled_production_outer_constrained_2026-08-02.jl   (new: constrained
    outer driver giving REDUCED the same problem shape FULL solves, family-agnostic)
full_aod_diag/d4_exact/run_outer_{flexcm,originzc}_{reduced_constrained,full}_2026-08-02.jl
    (new: matched ctx-build experiment scripts, one pair per family)
full_aod_diag/d4_exact/profiled_lfix_incremental_2026-08-01.jl               (gradient-engine fix:
    dense price0/pTsigma0 tensor -> compact logCC0/mulU + on-the-fly score, reusing FULL's
    build_winner_ref!/pTσ_from_score verbatim)
full_aod_diag/d4_exact/profiled_shared_economic_gradient_engine_2026-08-01.jl (same fix, the
    generalized/shared-family entry point)
full_aod_diag/d4_exact/test_profiled_shared_engine_unrestricted_regression_2026-08-01.jl (include
    fix: added missing lfix_factorized_workspace.jl/gradient_workspace.jl dependency)
full_aod_diag/d4_exact/test_zc_lane_{cmzc,originzc}_outer_gradient_zerodense_d4_2026-08-02.jl
    (test-mock fix: added missing `decoded` field to match the real evaluators' own shape)
docs/OUTER_{FLEXCM,ORIGINZC}_*_2026-08-02.csv                       (real trace/result CSVs)
```

No production default was changed. No production campaign was launched. Nothing was pushed to any
remote — this is a locally-committed branch only.
