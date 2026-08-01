# Existing optimized economic / cross-block function map (2026-08-01)

All paths relative to `full_aod_diag/d4_exact/` unless stated otherwise. Every name below was
confirmed by directly reading current source in this worktree (branch
`architecture/profiled-economic-block-all-families-2026-08-01`, base commit `f439109`) — not
inferred from docs alone. Several 2026-07-28 docs (`TRUE_OPERATOR_BUNDLE_MASTER_REPORT_2026-07-28.md`,
`FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md`) describe an earlier, more-duplicated state that has
since been further unified; every place current code has moved past what a doc says is flagged
below.

## 1. Economic forward contraction

- `economic_operator.jl:62` — `economic_forward!(out, lambda_E, cf::CompressedFactual, ws::EconomicFGWorkspace)`
- Thin wrapper around `compressed_dual_contraction!`. Sufficient stats: `cf.winner`, `cf.wval`,
  `cf.nrm`, `cf.gdiv`, `cf.Pmat`, `cf.denom`. `ws` (built via `economic_operator_workspace(cf)`)
  owns the `κ`/`C` scratch, reused not reallocated.
- **Single definition, called by all five families.** No threading (serial O(W·Ddest) scatter/gather).
  No BLAS. Not Hessian-packed (writes to caller's `out`, length W).

## 2. Economic transpose contraction

- `economic_operator.jl:81` — `economic_transpose!(grad_E, draw_weights, cf, ws::EconomicFGWorkspace)`
- Wraps `compressed_transpose_contraction!` via `ws.B` scratch; length `cf.oci - 1`.
- **Single definition, called by all five families** (flexible CM `cm_lookup_kernels.jl:454`,
  common Fréchet `cm_frechet_lookup_kernels.jl:267`, shared verification core
  `operator_verification.jl::_verify_inner_solution_operator_cm_core`, unrestricted
  `operator_verification.jl:286`). No threading, no BLAS.

## 3. H_EE assembly

- Core serial kernel: `core_exact_hessian.jl:418` — `winner_pair_hessian!(h, obj, wctx::WinnerPairHessCtx)`.
- Core parallel kernel (production default): `core_exact_hessian.jl:716` —
  `hessian_core_winner_pair!(hess_packed, curvature_weights, obj, workspace::WinnerPairParallelWorkspace; workers=1, storage=:full_stride)`.
  `Threads.@spawn`, 2 spawn rounds (draw-chunked reduction, then group/pair-chunked accumulation),
  fixed order, no atomics.
- **Shared entry point, single definition, called by all five families**:
  `core_exact_hessian.jl:924` — `fill_core_hessian_upper!(Hdense, curvature_weights, obj, workspace::CoreExactHessianWorkspace, global_layout=nothing; backend=:exact_winner_pair_parallel, workers=10, storage=:full_stride)`.
  Dispatches winner-pair-parallel (default) / winner-pair-serial / `:dense_reference` (debug-only
  BLAS `gemm!` fallback via `_dense_reference_core_hessian!`, `:867`).
- Call sites: unrestricted `compressed_live.jl::_callbackEvalH_inner_compressed!` (writes directly
  to `evalResult.hess`, no larger block to view into); flexible CM/common Fréchet/CM+ZC via
  `cm_hessian_architectures.jl::_fill_cm_HEE!` (`:790`, calling `fill_core_hessian_upper!` at
  `:801` into `@view HEE[1:ncore,1:ncore]`); ZC-only via `archA_partitioned_hess_cb_builder`
  (`:1560`, calling it at `:1597` into `@view ∂∂f_∂∂x[1:NCORE,1:NCORE]`).
- Workspace `CoreExactHessianWorkspace` built once per `cf` (guard:
  `cctx.core_ws === nothing || cctx.core_ws_for !== cf`), never reallocated per callback.
- Packed-index convention: caller's own dense view offset IS the index map; no separate table.
  Final KNITRO row-major upper-triangle packing happens downstream in `pack_upper_cm_hessian!`
  (item 7) or the unrestricted callback's direct write.

## 4. H_EC preparation and assembly (economic × flexible-CM)

- Prep (cumulative bin tables, one O(W·Ddest·D) pass): `winner_pair_cross_hessian.jl:118` —
  `winner_pair_cross_hessian_fill!(wctx, ws::WinnerBinCrossScratch, obj, Bidx)`. Builds
  `QCScum`/`NuCScum`/`SOnlyCScum`/`QCfCScum`.
- Per-threshold-block fill (O(NCORE·nO) per call, one call per bin `l`):
  `winner_pair_cross_hessian.jl:198` — `winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, ws, l, origins, refIndex1, M)`.
  **This is the function whose target-correction term section 7 of this port amends.**
- Assembly (inline, into `Hfull`, with optional R-congruence): `cm_hessian_architectures.jl::hessian_cm_structured!`
  (~lines 1146–1184) and `cm_hessian_threaded.jl::hessian_cm_structured_v2!` (~lines 256–291).
  Loops `l in 1:L`, calls the per-block fill, optional `mul!(cctx.block_ec, Hraw_EC, cctx.R)`,
  writes `Hfull[1:NCORE, cols]` **and** its transpose mirror (documented fix: symmetrize-by-averaging
  later needs both triangles pre-populated).
- Gating: `_cm_cross_hessian_wants_winner_bin(cctx, cf)` (`:915`) — winner-bin path is the only
  reachable path in production (dense `CS_`-table fallback is provably unreachable per guard
  comments).
- Callers/families: flexible CM, common Fréchet, CM+ZC (all through the one shared orchestrator).
  ZC-only has no CM-grid block (see item 6 instead).
- Threaded twin exists (`threaded_cross_hessian.jl:110`,
  `winner_pair_cross_hessian_fill_threaded!`), opt-in via `cctx.cross_hessian_threaded` (default
  **false**).
- Scratch: `WinnerBinCrossScratch` (`cctx.cross_scratch`), sized once via
  `_ensure_cm_cross_scratch!`, reused; `Hraw_EC`/`block_ec` persistent `cctx` fields.
- Packed-index convention: `Hfull[1:NCORE, cols]`, `cols = NCORE+(l-1)*nO+1 : NCORE+l*nO`.

## 5. H_EF preparation and assembly (economic × common-Fréchet)

**Confirmed thin extension, not a separate implementation** — this supersedes the 2026-07-28 docs'
"genuinely duplicated" characterization (true then, fixed since):

- `cm_frechet_hessian.jl:239` — `archC_frechet_hess_cb_builder(cctx::CMBinHessCtx, level_targets)`
  resolves a cached `CMFrechetExtension` and calls the **same shared**
  `hessian_cm_structured!`/`hessian_cm_structured_v2!` that flexible CM/CM+ZC call, passing
  `extension=frechet_ext`.
- The Fréchet-only tail, called only when `extension !== nothing`:
  `cm_frechet_hessian.jl:72` — `_fill_frechet_level_blocks!(Hfull, cctx, w, H, M, use_winner_bin, wctx, cross_ws, extension)`.
  Fills H_E,level (= H_EF: `winner_pair_cross_hessian_colsum!` `:443` + `winner_pair_cross_hessian_esum!`
  `:510`), H_CF, H_FF, reusing the already-built `CT`/`CScum` bin tables (no separate O(W) pass).
  **These `colsum`/`esum` functions are the ones section 8 of this port amends for H_EF.**
- `hessian_cm_frechet_structured!` (`:284`) is now a one-line backward-compat wrapper delegating to
  the shared function — kept only because some diagnostic/gate scripts call it by name.
- Callers/families: common Fréchet only.
- Threading/scratch/BLAS: identical to item 4 (same shared orchestrator, same opt-in threaded
  cross-hessian, same persistent `CMFrechetExtension` scratch — `ext.Wtab`/`ext.T1`/`ext.Esum_wb`/
  `ext.colsum`/`ext.Hraw_cmlevel`, sized once, reused).
- Packed-index convention: `level_off = NCORE + ncm_cm`; written at `[j, level_off+l]` (both
  triangles).

## 6. H_EZ preparation and assembly (economic × ZC)

- Prep (O(W) refresh per callback): `winner_pair_cross_hessian.jl:324` —
  `winner_pair_cross_hessian_zc_prep!(ws::WinnerZCCrossScratch, wctx, S)`.
- Fill (serial): `winner_pair_cross_hessian.jl:353` —
  `winner_pair_cross_hessian_zc_block!(HEZ, wctx, ws, S, Z, M)`. Computes `HEZ = (1/M)·E'·diag(S)·Z`
  for an already-centered restriction matrix `Z`. Row 1 (ones/ζ) and the France (cf) row use
  `BLAS.gemv!('T', invM, Z, S, 0.0, row1)`; winner-conditioned bilateral rows via a
  `slot -> x -> w` triple loop. **This is the function whose target-correction term section 9 of
  this port amends** (specifically the row-1/cf-row correction subtraction; the bilateral winner
  rows are the unchanged "keep" term).
- Fill (threaded): `threaded_cross_hessian.jl:214` — `winner_pair_cross_hessian_zc_block_threaded!`,
  `Threads.@spawn`-chunked, per-worker scratch `ws.thread_scratch_ez` (avoids the serial version's
  shared-buffer race).
- Callers/families: CM+ZC (inside `_fill_cm_HEE!`'s widened branch, `cm_hessian_architectures.jl:828-857`,
  filling `HEM`) and ZC-only (inside `archA_partitioned_hess_cb_builder`, `:1600-1630`, filling
  `HER`) — **module header explicitly documents this as one shared primitive used by both call
  sites**, matching this port's "no duplicate EZ algorithm" requirement already.
- Gating: `_cm_zc_cross_hessian_wants_winner_bin` (CM+ZC, `:950`) /
  `_originzc_zc_cross_hessian_wants_winner_bin` (ZC-only, `:1517`).
- Scratch: `WinnerZCCrossScratch` (`cctx.zc_cross_scratch`/`octx.zc_cross_scratch`); centered `Z`
  from `ZCCenteredScratch` (`refresh_zc_centered!`, shared with the H_ZZ block computed right
  after it, so `Z` is centered once, read twice).
- Packed-index convention: `HEZ` sized `(wctx.ncolI+1) × nx`, written into caller's view of
  `HEM`/`HER`.

## 7. Full per-family Hessian assembly / packed-triangle orchestration

- **One shared serial orchestrator** for 3 families: `cm_hessian_architectures.jl:1085` —
  `hessian_cm_structured!(h, obj, cctx::CMBinHessCtx, extension::Any=nothing)`. Body: zero
  `cctx.Hfull` → `_fill_cm_HEE!` (item 3) → build bin tables → H_EC per block (item 4, + H_CZ
  companion when CM+ZC-widened) → `fill_cm_HCC!` (`:1052`, shared between flexible CM and common
  Fréchet since a 2026-07-28 harmonization) → `if extension!==nothing: _fill_frechet_level_blocks!`
  (item 5) → `pack_upper_cm_hessian!` (`:1033`).
- **One shared threaded orchestrator**: `cm_hessian_threaded.jl:175` — `hessian_cm_structured_v2!`,
  same body shape, threaded bin tables when `threaded_bins=true`; tail (H_EC/H_CC/packing) is
  literally shared code with the serial version now (both call the same `pack_upper_cm_hessian!`/
  `fill_cm_HCC!`), not a copy.
- KNITRO-facing callback builders: `archC_hess_cb_builder` (`:1731`, flexible CM/CM+ZC, dispatches
  serial vs `_v2!`), `archC_frechet_hess_cb_builder` (`cm_frechet_hessian.jl:239`, common Fréchet,
  same dispatch, both branches now call the shared function with a resolved extension — this is
  where the 2026-07-28 "duplicated code" bug was actually fixed).
- **Architecturally separate** (correctly, not by drift — different block partition, H_EE/H_ER/H_RR
  not H_EE/H_EC/H_CC): `archA_partitioned_hess_cb_builder` (`:1560`, ZC-only). Persistent
  closure-owned `scratch_full::Matrix{Float64}` (built once per KNITRO solve).
- **No orchestrator at all** for unrestricted: `compressed_live.jl::_callbackEvalH_inner_compressed!`
  writes the whole callback directly since the entire Hessian IS H_EE.
- Summary: one shared function used by 3 families (flexible CM, common Fréchet via extension,
  CM+ZC), one architecturally distinct function for ZC-only, no orchestrator for unrestricted.

## 8. Outer fixed-dual incremental winner cache / update_winner_o1 / top-3

- `lfix_incremental.jl:151` — `update_winner_o1(price_wo, wo, price_ro, ro, o_changed, new_price)`.
  O(1) winner/runner-up update, case analysis, exact for the winner (runner-up documented as an
  inexact placeholder in 2 branches, flagged via `ro_exact` — only the winner matters for `L_fix`).
- `lfix_incremental.jl:85` — `min_secondthirdmin_with_idx(col)` (rank-3 scan), used when 2 origins
  change in the same destination in one FD probe.
- Canonical immutable cache: `lfix_incremental.jl:260` — `struct LFixBaseCache` (`price0`/`pTσ0`
  `W×D×Ddest`; `winner0`/`runnerup0`/`third0` + prices `W×Ddest`; `contrib0`; `λstar`/`ζstar`/`q0`;
  France/gp scalars `wPrime_bi`/`τPrime_bi`/`LPrime_bi`/`Uσ_bi`/`λ_cf`/`cf_contrib0`).
- Allocating builder: `lfix_incremental.jl:386` — `build_lfix_base_cache(x_free0, ctx, base; validate_dense=false)`,
  raises `TiedWinnerError` on exact price ties (detected, not silently mishandled).
- **Production non-allocating variant** (default `:shared` gradient backend): `lfix_base_workspace.jl` —
  `mutable struct LFixBaseWorkspace` (persistent per `(D,Ddest,W)`) +
  `build_lfix_base_cache!(ws, x_free0, ctx, base; validate_dense=false) -> LFixBaseCache` (`:179`),
  fills `ws`'s buffers in place, returned `LFixBaseCache`'s array fields alias them (zero large
  allocation per gradient call). Documented invariant: one `LFixBaseWorkspace` never shared across
  concurrent gradient evaluations.
- Sibling backend variants (bit-identical, different allocation/threading strategy, cross-validated
  per `UNRESTRICTED_OUTER_GRADIENT_CALL_GRAPH_2026-08-01.md` §6): `LFixBaseCacheB`
  (`lfix_pTsigma_only.jl:209`), `LFixBaseCacheKB` (`lfix_kbplus.jl:52`), `LFixBaseCacheC`
  (`lfix_factorized.jl:62`) + per-family wrappers `build_lfix_base_cache_cm`/`_cm_meanzc`/
  `_originzc`/`_cm_frechet` (fold in that family's fixed restriction contribution into `q0`).
- Consumers: `composite_gradient.jl` (`gamma_component_analytic`, `count_winner_flips*`,
  `select_bandwidth`, `a_block_fd_component`, `composite_gradient_at`) and mutating twins in
  `shared_a_gradient.jl` (`economic_A_gradient!` etc.).
- Callers/families: all five, each via its own thin `build_lfix_base_cache_*` wrapper.
- Threading: `economic_A_gradient!`'s per-coordinate loop is `Threads.@threads :static` when
  `threaded=true`, using per-thread pool slots (`GradWorkspacePool`, `TwoOriginScratch`) to avoid
  races; `bandwidth_cache::Dict{Int,Float64}` optionally persisted across outer iterations.

## 9. Outer A/gp gradient — OLD (full) vs NEW (profiled-unrestricted, this branch)

**OLD/full** (all families' production path): `lfix_factorized_workspace.jl:354` —
`composite_gradient_at_Cplus(...)`, and the current production default `shared_a_gradient.jl::economic_A_gradient!`
(bit-identical to Cplus, wired via `resolve_price_cache_backend`, `c10_d20_production_driver.jl:913-916`).
`g[1]` = exact analytic `gamma_component_analytic`; `g[k],k≥2` = central FD via the O(1) incremental
update. Operates over the **full** `D·Ddest`-length (minus gravity pivot) A-block.

**NEW/profiled-unrestricted** (`full_aod_diag/d4_exact/*_2026-08-01.jl`, this branch's 32 commits,
`:unrestricted` only, not wired into any restricted family):
- `profiled_outer_gradient_fd_2026-08-01.jl`: `profiled_lfix_at`, `profiled_composite_gradient_at`
  — first-draft full-rebuild central FD (no incremental cache), superseded by:
- `profiled_lfix_incremental_2026-08-01.jl`: `struct ProfiledLFixCache` (`:37`, profiled analog of
  `LFixBaseCache`, plus reduced-formula fields `κ::Matrix{Float64}` (D×Ddest, **0.0 at every anchor
  cell** — the profiled-destination-scale-specific structural difference), `Cbar_eff`, `contrib0`,
  `const_part`, `cf_raw_κcf`, `κ_cf`, `gpσ`, `bi_slot`, `has_france`); `build_profiled_lfix_cache`
  (`:77`, independently re-verifies `winner0 == cf.winner`); `dest_contrib_reduced_o1` (`:181`, O(1)
  update via `update_winner_o1`/top-3 — same primitives as item 8, no new winner-selection logic);
  `profiled_affected_cells` (`:243`, a coordinate touches its own retained cell **plus** the
  gravity-pivot's retained cell, up to 2 destinations); `profiled_lfix_incremental_at` (`:260`);
  `profiled_gp_component_analytic` (`:308`, exact closed form, went through 2 rounds of bugfixing —
  final form is `-κ_cf·σ·gp^(σ-1)·T_f/M`, since `cf.cf_raw` already embeds the gp-dependent term
  that a first "fix" wrongly re-added as a spurious `LPrime_bi*S_m` term — **do not reintroduce
  that term**, matches CLAUDE.md's standing warning); `profiled_count_winner_flips`/
  `profiled_select_bandwidth` (`:329`/`:393`, same geometric-bisection selector as production);
  `profiled_composite_gradient_at_incremental` (`:441`, **the profiled outer-gradient entry
  point**), over the profiled coordinate vector `w_profiled` (length `1 + n_retained - 1`, vs full
  `1 + D·Ddest - 1`).
- Threading: none in either profiled file yet (serial `for` loops) — a documented, honest gap vs.
  production's `Threads.@threads` item-8 machinery.
- Scratch: `ProfiledLFixCache` allocated fresh per build call — no persistent-workspace analog of
  `LFixBaseWorkspace` exists yet for the profiled path (another honest gap).
- A/B status (git log, corrected per `00_READ_FIRST_CORRECTION.md`): profiled beats full by
  3.85–7.99% at matched iteration count across 3 points, one seed each, upper-bound direction only
  — `insufficient_evidence` for a port recommendation per the archive's own verdict (see
  `PROFILED_ALL_FAMILY_SOURCE_SNAPSHOT_2026-08-01.md`).

## 10. Is H_EE/FG genuinely one shared implementation, or duplicated per family?

**Confirmed: one shared implementation for both FG and H_EE, reused by all five families.**

- `rg -n "^function economic_forward!" *.jl` → exactly one hit (`economic_operator.jl:62`).
- `rg -n "^function economic_transpose!" *.jl` → exactly one hit (`economic_operator.jl:81`).
- `rg -n "^function fill_core_hessian_upper!" *.jl` → exactly one hit (`core_exact_hessian.jl:924`).
- Call-site evidence: reached from `compressed_live.jl:283` (unrestricted),
  `cm_hessian_architectures.jl:796/801` (`_fill_cm_HEE!`, flexible CM/common Fréchet/CM+ZC),
  `cm_hessian_architectures.jl:1592/1597` (`archA_partitioned_hess_cb_builder`, ZC-only).
- `economic_operator.jl`'s own header states this design intent explicitly ("ONE shared
  economic-core FG operator... used by EVERY family"); `FRECHET_CM_SHARED_DISPATCH_PROOF_2026-07-28.md`
  independently confirms via `Base.which` identity checks (flexible CM vs common Fréchet).
- What remains genuinely separate, by architecture (not drift): the cross/self restriction blocks
  (H_CC/H_EC-grid path for CM-family families vs H_ER/H_RR/H_EZ path for ZC-only), because the
  restriction structures themselves genuinely differ.
- **Stale-doc flag**: `TRUE_OPERATOR_BUNDLE_MASTER_REPORT_2026-07-28.md`/`FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md`
  describe `hessian_cm_frechet_structured!`/`_v2!` as separate, duplicated copies as of 2026-07-28.
  Reading `cm_frechet_hessian.jl` today shows this was fixed: `hessian_cm_frechet_structured!` is
  now a one-line wrapper delegating to the shared function; only the genuinely-Fréchet-only
  `_fill_frechet_level_blocks!`/`CMFrechetExtension` piece remains separate, and it is supposed to
  be (it computes blocks — H_EF/H_CF/H_FF — that structurally don't exist for other families).

## Implication for this port's required edits (sections 4-9 of the mission)

Because items 1-3 and 10 above show the FG/H_EE core is *already* one generic, cf-agnostic shared
implementation, **the profiled economic moment layout (a different `CompressedFactual`/`cf`
construction, with fewer retained rows and the M_d-weighted target correction) should flow through
items 1-3 unmodified** — no edit needed to `economic_forward!`, `economic_transpose!`,
`fill_core_hessian_upper!`, or their kernels. The mission's required surgical edits are confined to
exactly the three cross-block target-correction sites identified above:

- **H_EC**: `winner_pair_cross_hessian_cm_block!` (`winner_pair_cross_hessian.jl:198`) — item 4.
- **H_EF**: `winner_pair_cross_hessian_colsum!`/`winner_pair_cross_hessian_esum!`
  (`winner_pair_cross_hessian.jl:443`/`:510`) — item 5 (reached automatically once H_EC's shared
  cumulative-table prep is amended, since H_EF reuses the same tables — needs its own check, not an
  assumption).
- **H_EZ**: `winner_pair_cross_hessian_zc_block!` / `_threaded!` (`winner_pair_cross_hessian.jl:353`,
  `threaded_cross_hessian.jl:214`) — item 6, already the single shared CM+ZC/ZC-only primitive the
  mission requires.

Plus the outer A/gp gradient (item 9): the profiled machinery already exists for unrestricted only;
porting it means (a) making `ProfiledLFixCache`/`build_profiled_lfix_cache`/
`profiled_composite_gradient_at_incremental` generic over which family's restriction-fixed
contribution folds into `q0` (mirroring how `build_lfix_base_cache_cm`/`_originzc`/`_cm_frechet`
already do this for the OLD cache), not writing new per-family copies.
