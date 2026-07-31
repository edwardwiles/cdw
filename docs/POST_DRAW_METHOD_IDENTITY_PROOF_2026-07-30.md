# Post-draw method identity proof (2026-07-30)

Task requirement (§13): "Prove identical method identity for master_prepare_cc,
build_ad_context_real_d20, d20_real_setup, screen construction, threshold construction, objective
and bundle construction, inner FG, inner Hessian, verification, outer runner... Numerical
similarity alone is not sufficient. Prove literal pipeline reuse."

## Proof by construction, not just by comparison

The strongest available proof of "literal pipeline reuse" is that **the duplicate implementation
no longer exists in the process image at all** -- there is nothing left to diverge to, by
construction, not by discipline. Verified live in this Julia process (this branch, HEAD):

```julia
julia> master_prepare_cc_qmc defined?          false
julia> build_ad_context_real_d20_qmc defined?  false
julia> d20_real_setup_qmc defined?             false
```

`git grep` confirms zero remaining definitions or call sites of any of the three names anywhere
in the repository (see `QMC_PSEUDORANDOM_DUPLICATION_REACHABILITY_2026-07-30.md` §"Total distinct
duplicated/drifted function bodies: 3... now 0").

## `methods()` count: exactly one implementation per unified function

For every function this task requires to be reached identically by every draw design, `methods()`
returns exactly 1 (verified live, this branch, HEAD):

| Function | Method count | Defined at |
|---|---|---|
| `master_prepare_cc` | 1 | `prepare_cc/master_prepare_cc.jl:13` |
| `build_ad_context_real_d20` | 1 | `full_aod_diag/d4_exact/context_real_d20.jl:55` |
| `d20_real_setup` | 1 | `full_aod_diag/d4_exact/context_real_d20.jl:94` |
| `precompute_pairwise_M` (screen construction) | 1 | `full_aod_diag/d4_exact/infeasibility_screen.jl:146` |
| `build_extreme_draw_witness` (screen construction) | 1 | `full_aod_diag/d4_exact/infeasibility_screen.jl:434` |
| `d20_real_setup_design` (the resolver itself) | 1 | `full_aod_diag/d4_exact/draw_design.jl:131` |

Since Julia dispatches on argument types, a function with exactly one method is, by definition,
invoked via the identical code path regardless of which draw design's caller reaches it -- there
is no second specialization it could silently fall through to. This is a stronger guarantee than
comparing two call traces after the fact: it is structurally impossible for :pseudorandom,
:sobol_randomized, :halton_scrambled, and :precomputed to reach different code for any of these
six functions, because only one implementation of each exists in the loaded process.

Threshold construction (`CS.ThresholdAbortState(CS.resolve_threshold_for_delta(δ))`) and objective/
bundle construction (`CS.PsiObjectiveBundleImplicit(...)`) are both single inline call sites
*inside* `d20_real_setup` itself (context_real_d20.jl lines ~199 and ~187 respectively) -- they
are reached by exactly the one `d20_real_setup` method above, so the same one-method argument
applies transitively; they were never separately duplicated in `qmc_context_real_d20.jl` in the
first place (that file duplicated the *function bodies* that call them, not the underlying
`CS.ThresholdAbortState`/`CS.PsiObjectiveBundleImplicit` constructors, which live in `cc_algo/` and
were never touched by this task).

Inner FG, inner Hessian, verification, and the outer runner (KNITRO callback machinery in
`cc_algo/`, `c10_d20_production_driver.jl`) are unconditionally reached through `ctx.obj` --
since `ctx.obj` is built by the single `d20_real_setup` method above from the single, unified `U`
matrix, and every downstream diagnostic/production function (`evaluate_fullA`, `evaluate_fullA_fast`,
`build_pivot_elimination`, `composite_gradient_at_fast_buffered`, the KNITRO callback machinery)
dispatches on `ctx`/`ctx.obj`'s *type*, not on any `draw_design`-tagged branch (grep-confirmed: no
file outside `draw_design.jl`/`draw_design_types.jl` contains a branch on `:pseudorandom`,
`:sobol_randomized`, `:halton_scrambled`, or `:precomputed` -- see the static duplication guard,
`CAMPAIGN_DRIVER_CONSOLIDATION_2026-07-30.md`), these are reached identically for every design as
an immediate consequence of the six-function proof above, not a separately-argued claim.

## The one legitimate, intentional dispatch point

`generate_randoms!` has **4** methods -- and this is correct, not a violation. It is the single,
explicitly-scoped point where draw design is allowed to determine behavior (task §3: "the caller
preallocates U... every design fills the same shape/scalar type/storage layout"). Its 4 methods
dispatch on the 4 `DrawDesign` subtypes (`PseudorandomDesign`, `RandomizedSobolDesign`,
`ScrambledHaltonDesign`, `PrecomputedDrawDesign`) defined in `draw_design_types.jl` -- this is
where design-specific draw generation legitimately lives (`DRAW_GENERATION_REQUIRED` in the
reachability audit), and it is the *only* multi-method function in this whole proof. Everything
downstream of the `Uexp` matrix it fills is single-method, as shown above.

## Empirical confirmation (not just structural)

`test_draw_design.jl` section 5 (task §14) independently confirms, by actually running all 4
designs end-to-end at W=2000/4000 (small-W smoke, see `DRAW_FEATURE_PARITY_GATE_2026-07-30.csv`
for the exact run) and W=80000 (production-scale, see
`DRAW_PIPELINE_OLD_NEW_EQUIVALENCE_D20_2026-07-30.csv`):

- identical `threshold_state.threshold` (10.0 at δ=1<9, Inf at δ=10>=9) for every design;
- identical `ctx.pairwise`/`ctx.witness`/`ctx.screen_setup_wall` presence for every design;
- identical returned context field **set** for every design;
- identical `destination_sample`/`D_dest` routing for every design.

This matches the historical bug the audit found exactly: before this task, `:sobol_randomized`/
`:halton_scrambled` silently ran with `threshold=Inf` (the early-abort optimization disabled) and
required a third, compensating copy of the screen-construction block. Both symptoms are gone by
construction, not by patch.
