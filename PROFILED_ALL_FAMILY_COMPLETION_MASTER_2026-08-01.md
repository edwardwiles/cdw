# Profiled economic block, all-families completion — master status (2026-08-01, continuation session)

## Branch / commits / worktree

- New branch: `architecture/profiled-economic-block-all-families-complete-2026-08-01`
- New worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-economic-block-all-families-complete-2026-08-01`
- Forked from: `architecture/profiled-economic-block-all-families-2026-08-01` @ `79cc0bd` (recorded HEAD in the
  task prompt; confirmed via `git worktree add ... -b <new> architecture/profiled-economic-block-all-families-2026-08-01`,
  0 divergence at fork time)
- Source branch `79cc0bd` itself was **not modified** (worked entirely in the new worktree/branch).
- This session's new commits: see `git log --oneline 79cc0bd..HEAD` in this worktree at hand-off time.
- Not merged, not pushed to `origin`, no production default changed, no campaign launched.

## What this session actually is

The mission is an 18-section, ~2000-line specification asking for a complete, gated, five-family port
(genuine reduced economic layout + D4/D20 correctness gates + inner-solve equivalence + outer-gradient
sharing + performance profiling + W500k smoke tests + a production go/no-go). That is realistically
several more full sessions of work at this codebase's own historical pace — every comparable prior
piece of this port (H_EC, H_EZ, the threaded H_EZ twin, the France row) took one dedicated session each,
with real bugs found only by actually running KNITRO solves and independent brute-force references, not
by design alone (see the source branch's own master doc, "off-by-one row index", "missing kappa0 scale
factor", "latent OOB read"). This session's own honest contribution, in order of confidence:

1. **Resolved a real open design question** the source branch's own docs explicitly flagged as
   unresolved (`PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md` §2's "open question, to resolve during
   implementation, not assumed here" for H_EF, and the more general question of how a restricted
   family's `CMBinHessCtx`/`OriginZCCoreHessCtx` should ever come to hold a genuinely reduced economic
   block at all). **Verified by direct source read**, not assumed:
   - `build_cm_augmented_obj_archB`/`wrap_moments_with_cm_archB` (and, by the same construction pattern,
     `build_originzc_augmented_obj`/`build_cm_meanzc_augmented_obj`) already read the economic-block
     width **generically** off `ctx.obj.d`/`ctx.obj.outer_constr_index` — never a hardcoded
     `D*Ddest`-shaped constant.
   - The dense economic-column materialization (`materialize_dense_factual_structured!`) is **already
     unconditionally skipped** on the current production path (`skip_fill=true`,
     `moment_representation=:operator`, cm_hessian_architectures.jl:314/334/338's `if !skip_fill`
     guards) — the economic block's actual gradient/Hessian flow entirely through `cf` and the shared
     `economic_forward!`/`economic_transpose!`/`fill_core_hessian_upper!`/cross-block primitives, never
     through a dense `G` column read, on the path that matters.
   - **Consequence**: feeding these builders a `ctx.obj`-shaped object whose `.d`/`.outer_constr_index`
     already reflect the reduced width (`1 + layout.total_reduced_economic_moments`, the exact quantity
     `build_profiled_operator_bundle` already computes for the unrestricted family) requires **zero**
     changes to `wrap_moments_with_cm_archB`/`build_cm_augmented_obj_archB` on the production path. This
     collapses what the mission's §5 abstractly describes as "genuine reduced/anchor-omitting
     `CompressedFactual` construction" for the dimension-bookkeeping half of the problem into a single
     small additive helper, not a rewrite of the augmented-obj builders.
   - Separately, confirmed the H_EC/H_EF/H_EZ correction-term functions
     (`winner_pair_cross_hessian_cm_block!`/`_colsum!`/`_esum!`/`_zc_block!`) are **already fully generic
     over `ncolI`/`target_slot`** (they read `wctx.ncolI`/`wctx.target_slot[j]` and index
     `ws.QCScum`/`ws.MCScum`/etc. by the SAME `j`, with no hardcoded assumption about what `j` "means") —
     but their "keep"/winner term (`winner_pair_cross_hessian_fill!`'s `QTab`/`MTab` fill, and
     `_zc_block!`'s own inline scatter loop) hardcode the OLD full-index formula `j = slot + (o-1)*Ddest`
     directly from raw `wctx.winner[w,slot]`, so **do not** support a genuinely-reduced `ncolI` without
     either a new anchor-aware fill (mirroring `ReducedHomogeneousWinnerPairHessCtx`'s own
     `winner_reduced_col==0`-skip pattern) or a different wiring strategy. Resolved: the viable,
     **fully non-invasive** strategy is to keep building `wctx`/`ws` from the FULL (unreduced) `cf` via
     the existing, untouched `build_winner_pair_ctx(cf; bi_slot=...)`, call the existing, untouched
     `winner_pair_cross_hessian_fill!`/`_cm_block!(...; use_profiled_correction=true)`/etc. exactly as
     they are today (producing a FULL-width `Hraw_EC`/`HEZ`), and have the ORCHESTRATOR
     (`hessian_cm_structured!`/`archA_partitioned_hess_cb_builder`) **gather only the retained rows**
     (via `layout.retained_full_factual_j`/`full_factual_to_reduced`) when copying into the now-smaller
     `Hfull`. This touches **zero** lines inside any of the five functions the mission explicitly says
     not to replace (and zero lines inside `winner_pair_cross_hessian_fill!` either, though that one
     wasn't on the "do not replace" list) — the entire reduction happens in the packing/assembly step. It
     is not FLOP-optimal for the cross-block itself (still `O(W*Ddest*D)`, same as before — no cross-block
     compute savings), but it is correct, minimal-diff, and consistent with where this port's actual
     motivation lives (dual-DIMENSION reduction for KNITRO conditioning/speed, not cross-block FLOPs — the
     unrestricted family's own H_EE reduction already captures the compute-savings half of the story).

2. **Implemented and D4-verified** the dimension-bookkeeping half of that plan, for flexible CM
   specifically (the source doc's own "recommended next steps" #1 priority):
   - New file `full_aod_diag/d4_exact/profiled_restricted_family_base_2026-08-01.jl`:
     `build_reduced_base_obj_for_family(ctx, layout, CS)` — shallow-copies `ctx.obj` with
     `.d`/`.outer_constr_index` overridden to `1 + layout.total_reduced_economic_moments`, every other
     field (γ, δ, l, U, N, moments!, inner_loop_opt, ...) unchanged. ~15 lines, additive, zero change to
     any existing file.
   - `cm_hessian_architectures.jl::build_cm_augmented_obj_archB` gained one new optional keyword,
     `base_obj = nothing` (default preserves old behavior byte-for-byte — confirmed by the D4 gate below,
     `aug_full.ncore == ctx.obj.d` for every point tested).
   - **Along the way, found and fixed one genuine pre-existing bug**, unrelated to this task, that blocked
     even the very FIRST direct call to `build_cm_augmented_obj_archB` (baseline, no `base_obj`): two
     methods of `compute_bin_indices` exist (`common_marginals_interval.jl:72`, `z::Vector{Float64}`,
     returns a compact `UInt8`/`UInt16`-typed matrix; `cm_hessian_architectures.jl:104`,
     `z::AbstractVector{Float64}`, returns `Matrix{Int}`) — when both files are included (as every
     relevant test does), Julia's dispatch picks the MORE SPECIFIC first one, producing a
     `Matrix{UInt8}` that then fails `wrap_moments_with_cm_archB`'s declared `Bidx::Matrix{Int}`
     parameter with a hard `MethodError`. Latent because the only pre-existing direct caller,
     `build_cm_production_context`, apparently never triggers this exact dispatch path. Fixed with a
     single explicit `Matrix{Int}(...)` coercion at the `Bidx = ...` call site inside
     `build_cm_augmented_obj_archB` — value-preserving, zero behavior change for any correctly-typed
     input, confirmed by re-running the pre-existing `test_winner_pair_cross_hessian_cm_d4.jl`
     (20/20 PASS, byte-identical `max|Δ|=0.0` results, unchanged from before this fix).
   - New gate: `test_profiled_restricted_family_base_2026-08-01.jl` — **32/32 PASS**, real D4 calibration
     point, both contrast conventions (`:anchored`/`:orthonormal`), both `L∈{10,20}`. Verifies (against
     the SAME calibration point/context every existing D4 cross-Hessian test uses): the layout's own
     retained-moment count formula; `reduced_obj0.d == 1+layout.total_reduced_economic_moments` and `<
     ctx.obj.d` (a genuine reduction, not a no-op); `build_cm_augmented_obj_archB(...; base_obj=nothing)`
     is unchanged from before this session (`aug_full.ncore == ctx.obj.d`); with `base_obj=reduced_obj0`,
     `aug_reduced.ncore`/`obj_cm.d`/`outer_constr_index` all correctly reflect the reduced width, the CM
     restriction block itself (`ncm`) is untouched, and the resulting `cctx_reduced.NCORE`/`Hfull` size
     shrink accordingly.

## What is NOT done — honest accounting against the mission's 18 sections

Everything below is genuinely `not_started`/`not_run`, not "quietly assumed to work":

- **H_EE reduced-kernel wiring** (mission §6, the `_fill_cm_HEE!` swap to
  `reduced_homogeneous_winner_pair_hessian!`/`build_reduced_homogeneous_winner_pair_ctx`): NOT wired.
  Requires threading `θ_full` to the Hessian callback (currently only `cf` is published via
  `core_cf_ref`, a shared `Ref{Any}` box the `moments!` closure fills every call — `θ_full` would need
  an analogous shared box, since `_fill_cm_HEE!`'s signature `(HEE, w, obj, cctx, H, M)` has no `θ`
  argument today), plus a `layout` field on `CMBinHessCtx` (additive, via the SAME
  "outer-constructor-appends-fields" idiom the struct already uses for `hcz_prep_backend`/
  `bin_zc_drawchunk`), plus a branch in `_fill_cm_HEE!` dispatching to the reduced kernel when
  `cctx.profiled_layout !== nothing`.
- **H_EC "gather retained rows" step** (mission §4, the design resolved in item 1 above): NOT
  implemented. Requires the same `layout`/`bi_slot` fields on `cctx`, and a new branch in
  `hessian_cm_structured!` (and its threaded twin `hessian_cm_structured_v2!`) that, when a layout is
  present, builds `Hraw_EC` at full width exactly as today but copies only
  `layout.retained_full_factual_j`-selected rows (plus the France row, plus row 1) into the
  now-smaller `Hfull`.
- **H_EF (`colsum!`/`esum!`) and H_EZ gather**: same pattern as H_EC, not implemented; H_EF additionally
  needs the "open question" from `PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md` §4 resolved by reading
  `CMFrechetExtension`'s actual `Wtab`/`T1` construction (not attempted this session).
- **ZC-only and CM+ZC families**: `build_originzc_augmented_obj`/`build_cm_meanzc_augmented_obj` were
  read (confirmed to follow the identical `ncore_econ = obj0.d` generic-width pattern as flexible CM,
  per the earlier research pass) but **not** given the analogous `base_obj` keyword or tested — this
  should be mechanical once flexible CM's full path (H_EE + H_EC gather) is validated end-to-end, but
  "should be mechanical" is a prediction, not a verified result.
- **Common Fréchet**: reuses flexible CM's `CMBinHessCtx`/`build_cm_augmented_obj_archB` unchanged, so
  the `base_obj` plumbing already technically reaches it, but H_EF's own gather step (above) is not
  done, and it was not tested.
- **Dual bounds / initial dual / moment names / verification slices** (mission §5's explicit checklist):
  not audited. `outer_constr_index`/`d` bookkeeping is confirmed generic and correctly reduces (this
  session's own gate), but KNITRO variable-bound arrays, moment-name diagnostics, and the
  `_verify_inner_solution_operator_cm_core`-style verification path were not checked for hardcoded
  `D*Ddest`-shaped assumptions — likely candidates for a second latent bug, on the pattern of this
  session's own `Bidx` type-dispatch finding.
- **Every D4/D20/inner-equivalence/outer-gradient/performance/W500k gate the mission asks for
  (§8-21)**: not run. No `PROFILED_ALL_FAMILY_D4_FULL_HESSIAN_GATE_2026-08-01.csv`,
  `..._D4_INNER_EQUIVALENCE...`, `..._D20_CROSS_BLOCK_GATE...`, `..._D20_INNER_EQUIVALENCE...`,
  `..._OUTER_GRADIENT_GATE...`, `..._PERFORMANCE_GATE...`, or W500k smoke doc exist from this session
  — writing empty/fabricated versions of these would misrepresent the state of the work, so they are
  omitted rather than stubbed.
- **`FULL_FORMULATION_NO_OVERHEAD_GATE_2026-08-01.csv`** (mission §3): not run. Worth noting the
  profiled-only scratch fields (`MTab`/`MCScum`/`SnuWval`/`TZ_buf`/etc.) were already confirmed, by the
  SOURCE branch's own prior session, to be filled unconditionally but cheaply (one extra
  multiply-add per existing loop iteration, no new O(W) pass) — this session did not re-verify that
  claim or measure allocations/wall-time directly.

## Verdict block (mission's own format)

```text
PRIMITIVE_CROSS_BLOCK_FORMULAS =
    validated   # unchanged from source branch: H_EC/H_EF/H_EZ D4-validated vs independent brute force

OLD_FULL_PATH_OVERHEAD =
    not_remeasured_this_session   # source branch reasoned this is fine (same complexity class); not empirically re-checked here

GENUINE_REDUCED_LAYOUT =
    unrestricted:      pass                          # pre-existing, source branch, unchanged
    flexible_CM:        dimension_bookkeeping_pass_hessian_wiring_not_started
    common_Frechet:      not_started
    ZC_only:             not_started
    CM_plus_ZC:          not_started

HESSIAN_ORCHESTRATOR_WIRING =
    fail_all_four_restricted_families   # H_EE reduced-kernel swap and H_EC/EF/EZ gather step neither implemented

D4_COMPLETE_HESSIAN =
    not_run   # only a dimension/construction gate ran this session, not a numeric Hessian-vs-reference gate

D20_CROSS_BLOCK_REFERENCE =
    not_run

INNER_EQUIVALENCE =
    not_run

SHARED_PROFILED_A_GP_GRADIENT =
    not_started   # unchanged from source branch

RESTRICTION_ONLY_CODE_CHANGED =
    none   # confirmed: only cm_hessian_architectures.jl's build_cm_augmented_obj_archB (additive
           # base_obj kwarg + one pre-existing Bidx-dtype bugfix) and one new additive file touched;
           # H_CC/H_CF/H_FF/H_CZ/H_ZZ/CM bins/Fréchet level moments/Z features/restriction FG/
           # verification/restriction-parameter gradients: byte-for-byte untouched

ZC_OPTIMIZATION_INTEGRATION =
    pending   # this session made no contact with that separate workstream; nothing to integrate yet

W500K_PUBLIC_ENTRY_SMOKE =
    not_run

PRODUCTION_RECOMMENDATION =
    insufficient_evidence   # unchanged from the source branch's own honest verdict; this session did
                             # not add outer-loop A/B evidence, only infrastructure

PRODUCTION_DEFAULT_CHANGED = false
CAMPAIGN_LAUNCHED = false
```

## Recommended next steps (in order, for whoever continues this)

1. Add `θ_full_ref::Base.RefValue{Union{Nothing,Vector{Float64}}}` and
   `profiled_layout::Union{Nothing,ProfiledEconomicMomentLayout}` fields to `CMBinHessCtx` (via the
   struct's own "outer constructor appends new fields with keyword defaults" idiom, matching
   `hcz_prep_backend`/`bin_zc_drawchunk` exactly) and `OriginZCCoreHessCtx`. Publish `θ_full` from
   `wrap_moments_with_cm_archB`'s closure into that ref, alongside its existing `core_cf_ref[] = cf`
   line.
2. In `_fill_cm_HEE!`, when `cctx.profiled_layout !== nothing`, call
   `build_reduced_homogeneous_winner_pair_ctx(cf, cctx.econ_ctx, cctx.θ_full_ref[], cctx.profiled_layout)`
   + `reduced_homogeneous_winner_pair_hessian!` instead of `build_core_exact_hessian_workspace`/
   `fill_core_hessian_upper!`, writing into a correctly-reduced-size `HEE` view.
3. In `hessian_cm_structured!`/`_v2!`, when `cctx.profiled_layout !== nothing`: build `wctx` from the
   FULL `cf` as today (`build_winner_pair_ctx(cf; bi_slot=...)`), call the existing amended H_EC
   functions with `use_profiled_correction=true`, then gather only
   `cctx.profiled_layout.retained_full_factual_j`-selected rows into the reduced `Hfull` (implementing
   item 1's resolved design above).
4. D4 dense-reference gate for flexible CM specifically: compare the fully-wired reduced structured
   Hessian against an independent `G'diag(S)G` reference built directly from a reduced `G` (only
   `layout`-retained columns materialized) — the mission's own §8 gate, restricted to one family first.
5. Only after (4) passes: repeat (1)-(4)'s mechanical parts for ZC-only/CM+ZC/common-Fréchet, resolve
   H_EF's open Wtab/T1 question, then proceed to D20/inner-equivalence/outer-gradient/performance/W500k
   per the mission's own ordering — each is real, separately gate-able work, not a rubber stamp.

Given the source branch's own prior verdict (`PORT_TO_RESTRICTED_FAMILIES = insufficient_evidence`, a
real-but-small 3.85-7.99% unrestricted-only outer-loop edge from 3 points/one seed/upper-direction-only)
was already judged too thin to justify the FULL remaining campaign before this session started, and
this session's own contribution is infrastructure/design resolution rather than new outer-loop evidence,
that judgment call — whether to continue investing in the full five-family port before more unrestricted
evidence exists — still stands open for the user, not resolved here.
