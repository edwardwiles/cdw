# Full-A to Profiled-Destination-Scales: Call Graph Map

Repo root: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profile-all-destination-scales-2026-07-31`
(branch `architecture/profile-all-destination-scales-2026-07-31`, based on `production/fullA-exact @ cd17235`)

All paths below are relative to `full_aod_diag/` unless stated otherwise. There is no separate
`src/` package directory in this repo — everything is flat `.jl` scripts under `full_aod_diag/`
and `full_aod_diag/d4_exact/` (631 files), loaded via `include(...)` chains, no Julia module/package
boundary. This is PURE RESEARCH — no code was written or edited during this audit.

**Big picture finding up front** (see topic 2): the single most consequential site for this
reparameterization is a duplicated formula, not a shared function — the "outer A-coordinate ->
gauge-normalized level A" transform

```julia
Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1,1] .* τ[1,:]')) .^ (1/μ)) .* (lambda ./ lambda[1,:]')
```

appears independently, copy-pasted, in at least 15 files (production and diagnostic). Critically,
**this formula already singles out origin index 1 as a per-destination gauge anchor** — the
`lambda ./ lambda[1,:]'` term divides every destination-d column of `lambda` (=`ctx.γ.P` reshaped)
by `lambda[1,d]`, i.e. origin 1's factual share to every destination d. This is a *different*
anchor mechanism from the gravity-elimination pivot (`gravity_elimination.jl`'s single scalar
pivot cell, chosen globally by `argmax|c|`, not per destination) and from the France
autarky/gp moment. A "profiled destination scales" design that fixes one A cell per destination as
a gauge needs to decide explicitly whether it reuses this *already-existing* per-destination
origin-1 anchor, or introduces a new one — they are not currently the same object, and code in
>15 places assumes the *current* one.

---

## Topic 1 — Outer coordinate transformation / layout (`outer_coordinate_layout.jl`)

Read in full (245 lines). This is the newest (2026-07-25) shared abstraction unifying fixed/flexible
theta and legacy-z/powered-aspace A-coordinate conventions into one outer-vector convention:
`w = [eta_theta?; gp_coord; A_coord(D*Ddest-1)]`.

- `outer_coordinate_layout.jl:45-64` — `struct OuterCoordinateLayout` / `make_layout(...)`: immutable
  descriptor of 3 independent axes (`trade_elasticity_mode`, `A_coordinate_mode`, `gp_coordinate_mode`).
  **Must be generalized**: a 4th axis (`destination_scale_mode :full | :profiled`, or similar) would
  need to be added here, since this struct is the single source of truth every driver/screen/gradient
  function downstream reads to decide vector shape.
- `outer_coordinate_layout.jl:67` — `outer_dim(layout, D, Ddest)`: `D*Ddest ± 1`. **Must be
  generalized** — under profiled scales the A-block shrinks from `D*Ddest-1` (pivot-eliminated) to
  `D*Ddest - Ddest` (one free coordinate removed per destination, on top of the existing global
  gravity pivot elimination) or similar; this arithmetic is load-bearing everywhere vector lengths
  are asserted (`assert_pivot_layout`, KNITRO variable counts, checkpoint schemas).
- `outer_coordinate_layout.jl:130-165` — `decode_outer_unified(w, ctx, layout, pgc, xy, gs)`: the
  ONE generic outer-vector decoder. Converts `w` -> `(xf, theta, mu, gp, z_nonpivot,
  A_nonpivot_native, eta_theta)`, calling `pivot_expand_cheap` (gravity_elimination.jl) to fill the
  gravity-eliminated pivot cell, and (if `:powered_aspace`) the a<->z map. **Must be generalized** —
  this is the exact function that would grow a per-destination "expand relative coords -> absolute
  cell, filling the anchor cell from the other free cells of the SAME destination" step, analogous to
  what it already does for the single global gravity pivot.
- `outer_coordinate_layout.jl:174-181` — `reduce_to_w_unified(...)`: inverse direction (full A matrix
  -> outer `w`), used to seed KNITRO starts from a calibrated/checkpointed full A matrix. **Must be
  generalized** in lockstep with decode.
- `outer_coordinate_layout.jl:194-202` — `gradient_transform_unified(...)`: rescales the shared
  z-space gradient (`composite_gradient_at_Cplus`, computed once, always in raw gp/z-space) into
  whatever coordinates the layout searches. **Shared formula needs new inputs** — a profiled-scale
  gradient chain rule would need an extra per-destination Jacobian block here (in addition to the
  existing `-theta` rescale for powered a-space), but the underlying z-space gradient computation
  itself would not need to change.
- `outer_coordinate_layout.jl:212-216` — `layout_fingerprint(...)`: cache/checkpoint identity string.
  **Must be generalized** to encode the new axis so old-format and new-format checkpoints/caches
  don't silently collide.
- `outer_coordinate_layout.jl:243-244` — `dual_bank_zfree(d, layout)`: DualBank warm-start key
  construction. **Shared formula needs new inputs** only in the sense that the vector it concatenates
  (`d.z_nonpivot`) would be shorter under profiled scales; the function itself is generic.

Related files found in the same `rg` sweep but NOT part of the core layout (comparison/legacy
scaffolding — **legacy full reference only**): `flexible_theta.jl`, `flexible_theta_aspace_production.jl`
(the z<->a map `APivotXY`/`precompute_aspace_XY`/`z_from_a`/`a_from_z`, algebraically the SAME per-cell
log-linear map as `cm_aspace_coordinate.jl`'s restricted-family counterpart — see Topic 14), and the
several `matched_comparison_*.jl` / `phaseA_*`/`phase6_*` diagnostic scripts.

Call sketch: `run_polish_checkpointed_unified` (c10_d20_production_driver_unified.jl:130) ->
`decode_outer_unified`/`reduce_to_w_unified` -> `pivot_expand_cheap`/`pivot_reduce_cheap`
(gravity_elimination.jl) -> `screened_eval_flexible_A`/`evaluate_fullA_screened_ranged`
(fast_range_screen.jl) -> KNITRO FG callback.

---

## Topic 2 — Full-A reconstruction from outer coordinates (the duplicated gauge transform)

No single function owns this; it is duplicated. Confirmed present (via `rg` + read) in:

| File:line | Context |
|---|---|
| `moments_gammanorm.jl:98` (D=D branch) and `:242` (rectangular D×Ddest branch) | `EK_moments_gammanorm_directgp!` — the ORIGINAL, still-referenced production `moments!` |
| `moments_fast.jl:168` | `EK_moments_gammanorm_directgp_fast!` — byte-mirror with a `MuSigmaPowCache` optimization |
| `autarky_cf.jl` (`EK_moments_gammanorm_directgp_autarkyCF!`, `:98`) and `autarky_cf_v2.jl` (`:134`) | France-autarky specialized variants |
| `gravity_elimination.jl:45` | `gravity_from_logz` — reconstructs level A from a z-matrix for gravity-offset computation |
| `winners.jl:22-31` | `factual_prices(θ_full, ctx)` — winner/price recovery (Topic 4/9 support) |
| `structured_moment_build.jl` (indirectly, via `ctx.γ` fields already gauge-transformed) | |
| `oracle.jl`, `oracle_fast.jl`, `oracle_profiled.jl`, `compressed_live.jl`, `infeasibility_screen.jl`, `fast_range_screen.jl`, `cm_aspace_coordinate.jl:61-68` (as `precompute_cm_aspace_xy`'s `X`/`Y` — the SAME `wHat*τ/(wHat[1,1]τ[1,:]')` and `lambda/lambda[1,:]'` objects, logged), `c10_structured_moment_verify.jl`, `check_movement_and_fixedA.jl` | further copies/near-copies for diagnostics or the flexible-theta a-space port |

**Change category: shared formula needs new inputs** (all sites) — every one of these needs to know,
per destination, which A cell is the "profiled" anchor and how to reconstruct it from the other
free cells for that destination (in addition to / instead of the existing origin-1 gauge divide).
Because the formula is copy-pasted rather than centralized, **this reparameterization has a strong
prerequisite refactor implication**: consolidating this formula into one function (as
`gravity_elimination.jl`/`outer_coordinate_layout.jl` already did for the *pivot* reconstruction)
would sharply cut the number of independent edit sites; leaving it duplicated means the same
per-destination-anchor logic must be replicated correctly >15 times.

`_ctx_ddest(ctx)` (`gravity_elimination.jl:30`) is the one small shared helper (`Ddest==ctx.D` unless
`row_idx` excludes ROW as a destination) already threaded through all of the above — **unchanged**,
reusable as-is for "how many active destinations."

---

## Topic 3 — Gravity elimination / pivot reparameterization (`gravity_elimination.jl`)

Read in full (254 lines) — see file for the standing CLAUDE.md warning about NOT conflating this
file's `zfree=0` reference point with "calibration." Verified directly: `A_od≡1` (`z==0`) is just
the origin of the pivot-reduced coordinate system, not any fitted value.

- `gravity_elimination.jl:25-27` — `gravity_linear_coeffs(ctx; μ)`: `c = μ*q_tilde/N_obs`, exactly
  linear in μ. **Unchanged** — pure data/μ, no per-destination structure to touch.
- `:38-48` — `gravity_from_logz(z, ctx; μ)`: full level-A reconstruction (the Topic-2 formula) +
  `gravity_value` call. **Shared formula needs new inputs** (same reasoning as Topic 2).
- `:51` — `gravity_offset(ctx; μ)`: `g_gravity` at `Aod_theta≡1`. **Unchanged** in form, but callers
  must keep remembering (per CLAUDE.md) that this is a reparam offset, not a fitted point — a
  profiled-scales design doc should say this explicitly too since it introduces a SECOND arbitrary
  gauge (the per-destination anchor) sitting next to this one.
- `:53-78` — `struct PivotGravityElim` + `build_pivot_elimination(ctx; μ)`: chooses ONE global pivot
  cell (`argmax|c|` over the whole D×Ddest block) and returns the affine map
  `z_free (D*Ddest-1) -> full z`. **Must be generalized** conceptually (not necessarily by editing
  this exact function) — the profiled design adds a SECOND, per-destination elimination layered on
  top of this one; the two eliminations must compose correctly (the global gravity pivot cell must
  not coincide with, or must be handled consistently with, whichever cell is each destination's
  profiled anchor).
- `:81-92` — `pivot_expand`/`pivot_reduce`: the O(D·Ddest) full expand/reduce given a
  pre-built `PivotGravityElim`. **Unchanged** as the mechanism (linear scatter/gather); would need a
  second, analogous per-destination expand/reduce layered before/after it, not a modification of this
  one.
- `:94-126` — `NullspaceGravityElim`/`nullspace_expand`/`nullspace_reduce`: an orthonormal-nullspace
  alternative to the pivot parameterization. **Legacy full reference only** — confirmed via `rg` that
  only `test_gravity_elimination.jl` references this; not on any production driver's call path.
- `:153-224` — `PivotGravityElimCache`/`build_pivot_elimination_cheap`/`pivot_expand_cheap`/
  `pivot_reduce_cheap`/`assert_pivot_layout`: the theta-invariant cached version used by the
  flexible-theta production port (`outer_coordinate_layout.jl` depends on this directly, `isdefined`
  guard at line 36). **Must be generalized** in the same way as `PivotGravityElim` above — this is
  the ACTUAL hot-path struct/functions (not the older `PivotGravityElim`), so if the pivot/anchor
  composition is implemented anywhere production-side, it's here.
- `:250-254` — `pivot_elim_from_cache(pgc, μ)`: adapts the cache back into the original struct shape
  for consumers (D=20 production C+/buffered gradient backends) that still expect `PivotGravityElim`.
  **Shared formula needs new inputs** only in that its inputs (`pgc`) would carry the new fields.

Call sketch: `build_pivot_elimination_cheap` built ONCE per outer base point (whenever gp changes) by
`c10_d20_production_driver_unified.jl` / `cm_aspace_coordinate.jl` callers -> `pivot_expand_cheap`
called every FG/gradient evaluation (O(1) incremental, no rebuild) -> feeds `decode_outer_unified`.

---

## Topic 4 — Factual trade-share moment construction (lambda_od / E_F[Q_od])

- `compressed_moments.jl:132-147` — `struct CompressedFactual`: the compressed representation every
  family's FG/Hessian machinery consumes. Field `Pmat::Matrix{Float64}` (`D x D_dest`, "observed
  bilateral shares") IS the factual `lambda_od` moment target; `denom::Vector{Float64}` (length
  `D_dest`) is the per-destination normalization referenced in the task background (`E_F[M_d]=1`).
  **Must be generalized** — this is the concrete data structure that encodes "one normalization
  constraint per destination"; removing a redundant per-destination moment under the profiled design
  means this struct (or its builder) changes shape.
- `structured_moment_build.jl:163-169` — `materialize_dense_factual_structured!(Gview, cf)`: rank-one
  + winner-scatter reconstruction of the dense factual moment matrix from a `CompressedFactual`,
  drop-in replacement for the older `materialize_dense_factual!` (`compressed_moments.jl`). **Shared
  formula needs new inputs** if the moment-column layout changes (one fewer column per destination).
- `structured_moment_build.jl:181-190` — `fill_K_directgp!(Kview, θ_full, ctx)`: fills the objective
  column `K[s] = θ_full[3+D]*SamplingWeights[s]` (the France gp/counterfactual scalar). **Unchanged**
  — this is the gp/rho column, not A_od, no per-destination structure.
- `gravity_tariff.jl:62` — `precompute_q_tilde(τ)`, `:80` — `gravity_value(τ, AodPow, q_tilde, N_obs)`,
  `:101` — `gravity_grad_free!(...)`: the gravity moment itself (a SINGLE scalar constraint across the
  whole A block, not per-destination). **Unchanged** — orthogonal axis to the profiled-destination
  reparam; only consumes `AodPow`, doesn't care how A got reconstructed.
- `economic_operator.jl` (Topic 6, full read) — the shared `economic_forward!`/`economic_transpose!`
  operate on `lambda_E` / `cf::CompressedFactual` generically, already decoupled from how many A
  coordinates are free. **Unchanged.**

---

## Topic 5 — Gamma normalization, France autarky moment, gp

- No functions literally named `rho_from_gp`/`GT_from_gp` exist in this worktree (`rg` returned zero
  hits) — the background brief's assumed naming isn't present; welfare/GT figures are computed in
  ad hoc diagnostic/campaign scripts (e.g. `melitz`-prefixed files under other memory entries, not in
  this repo path), not as a single named pipeline function here.
- `autarky_cf.jl:53-96` — `autarky_cf_scalars`/`fill_autarky_cf_column!`: the France-specific
  counterfactual price-index column construction (`counterType==1` branch only). **Legacy full
  reference / unchanged** — France's moment is keyed off `gp` (an outer scalar) and `AodPow`, not a
  per-destination gauge; no direct interaction with which A cell is a destination's anchor, EXCEPT
  that `AodPow` itself is downstream of Topic 2's transform, so it inherits that dependency
  transitively (**shared formula needs new inputs**, one level removed).
- `autarky_cf_v2.jl:134` — `EK_moments_gammanorm_directgp_autarkyCF_v2!`: same relationship.
- Gamma-normalization proper (`γ`, `γ_prime`, "forced to 1 except focal", `moments_gammanorm.jl:174-176`
  in the read excerpt) is a per-ORIGIN-index-`baseIndex` device unrelated to per-destination anchors —
  **unchanged**.

---

## Topic 6 — Economic operator forward/transpose (`economic_operator.jl`)

Read in full (86 lines).

- `:49` — `economic_operator_workspace(cf)`: thin constructor wrapper around `EconomicFGWorkspace`
  (`compressed_cc_inner.jl`). **Unchanged.**
- `:62-67` — `economic_forward!(out, lambda_E, cf, ws)`: `out[s] = Σ_j lambda_E[j]*E_{s,j}` via
  `compressed_dual_contraction!`, winner-compressed, O(W·Ddest). **Unchanged** — generic over
  `length(lambda_E) == cf.oci-1`; doesn't know or care about A-coordinate count.
- `:81-86` — `economic_transpose!(grad_E, draw_weights, cf, ws)`: the transpose/scatter, via
  `compressed_transpose_contraction!`. **Unchanged**, same reasoning.

This file is explicitly documented (header comment, `:1-36`) as THE single shared kernel every
family (unrestricted `G=E`, flexible-CM `G=[E|C]`, common-Frechet `G=[E|C|F]`, CM+ZC `G=[E|C|Z]`,
ZC-only `G=[E|Z]`) composes with its own restriction operator — a genuinely good sign for this
reparameterization: **the economic core is already decoupled from A-coordinate count/layout**, so
the profiled-scales change should not need to touch this file at all, only its upstream callers
(`prime_operator!`, Topic 13) that decide `θ_econ`/`cf` construction.

`dual_index` (searched per task list) appears only as a local variable/field name inside
Hessian-kernel files (`cm_lookup_kernels.jl`, `cm_hessian_architectures.jl`, etc.) — not a distinct
shared abstraction; folded into Topics 7/8 below.

---

## Topic 7 — Economic self-Hessian (H_EE / winner-pair)

- `core_exact_hessian.jl:373` — `build_winner_pair_ctx(cf::CompressedFactual)`, `:418` —
  `winner_pair_hessian!(h, obj, wctx::WinnerPairHessCtx)` (the H_EE kernel proper), `:601` —
  `build_winner_pair_parallel_workspace`, `:716` — `hessian_core_winner_pair!` (threaded entry).
  **Unchanged** — operates entirely on `cf::CompressedFactual`'s winner/`wval` compressed
  representation (same object Topic 6 consumes), no direct A-coordinate-count dependency. Depends
  transitively on Topic 2/4 only through how `cf` was built upstream.
- `winner_pair_cross_hessian.jl` — despite the filename, also contains the pure economic-diagonal
  support kernels referenced from `winner_pair_hessian!`; same **unchanged** classification for the
  E-only pieces.

---

## Topic 8 — Economic × restriction cross-Hessian (H_EC / H_EZ)

- `winner_pair_cross_hessian.jl:118` — `winner_pair_cross_hessian_fill!` (dispatch),
  `:198` — `winner_pair_cross_hessian_cm_block!` (H_EC, CM-grid restriction),
  `:324`/`:353` — `winner_pair_cross_hessian_zc_prep!`/`winner_pair_cross_hessian_zc_block!` (H_EZ,
  ZC restriction), `:443` — `winner_pair_cross_hessian_colsum!`, `:510` —
  `winner_pair_cross_hessian_esum!`, `:605`/`:641` — `bin_zc_cross_hessian_fill!`/`_block!` (H_CZ,
  CM×ZC cross term). Threaded duplicates in `threaded_cross_hessian.jl` (`:110`, `:214`, `:300`) and
  a sparse-SpMM candidate in `hcz_sparse_spmm_candidate_2026-07-29.jl:60`.
- `cm_hessian_architectures.jl:915/934/950/1517` — `_cm_cross_hessian_wants_winner_bin` and siblings:
  backend-selection predicates (dense vs winner-bin vs direct HCZ), not math.
- `c13_schur_block_elimination.jl:78` — `schur_solve(H_EE, H_EC, H_CC, b, NCORE, ncm)`: consumes the
  assembled blocks for a Schur-complement linear solve.

**Change category for all of the above: unchanged.** Every one of these operates on `cf`
(economic side, via `wctx::WinnerPairHessCtx` built from `CompressedFactual`) and on the
family-specific restriction operator's OWN column structure (CM bins, ZC targets) — neither touches
A-coordinate count/layout directly. They inherit the reparameterization's effect only transitively,
through whatever upstream (`prime_operator!`/Topic 13, `cf_build`) constructs `θ_econ`/`cf` from the
now-shorter outer vector. This is a second good sign: the Hessian machinery's recent "no dense
G/H, operator-only" hardening (2026-07-25 through 2026-07-30 per file headers) already fully
decoupled it from A-coordinate representation.

---

## Topic 9 — Outer C+ gradient / winner-switch machinery

- `composite_gradient.jl:303` — `composite_gradient_at(x_free0, ctx, pe; ...)`: original/reference
  gradient-at-a-point driver (dispatches into the LFix machinery below).
- `lfix_factorized_workspace.jl:354` — `composite_gradient_at_Cplus(x_free0, ctx, pe, pool, ws; ...)`:
  **THE shared numerical kernel** `outer_coordinate_layout.jl`'s own docstring names explicitly
  (line 188 comment: "computed exactly in raw gp/z-space regardless of layout"). **Shared formula
  needs new inputs** — this function's `x_free0`/`z`-space coordinate vector shrinks by `Ddest`
  entries under the profiled design; the finite-difference/analytic-derivative bookkeeping inside
  (`a_block_fd_component_Cplus!`, `lfix_incremental_at_Cplus!`, same file `:298/:326`) needs to know
  which A cells are anchors (no perturbation direction exists for them) vs free.
- Family-specific Cplus wrappers, all thin adapters around the shared kernel above, each stripping
  their own trailing restriction parameters first: `cm_frechet_cplus.jl:162/204`
  (`composite_gradient_at_Cplus_frechet`/`_fast_frechet`), `cm_meanzc_cplus.jl:66`
  (`composite_gradient_at_Cplus_cm_meanzc`), `cm_originzc_cplus.jl:45`
  (`composite_gradient_at_Cplus_originzc`), `lfix_cm_cplus.jl:77/155`
  (`composite_gradient_at_Cplus_from_cache`/`_cm`). **Shared formula needs new inputs**, same
  reasoning, once per family (thin pass-through, so the edit is small per file but touches all 5).
- Older/alternate gradient backends found in the same sweep — `lfix_kbplus.jl:382`,
  `lfix_base_workspace_pooled.jl:22`, `lfix_pTsigma_only.jl:647`, `lfix_kbplus_workspace.jl:318`,
  `lfix_buffer_reuse.jl:136`, `lfix_cm_aware.jl:163`, `composite_gradient_fast.jl:111`,
  `gradient_workspace.jl:214` — **legacy full reference only** for most of these (superseded by
  `composite_gradient_at_Cplus`'s `LFixFactorizedWorkspace` per recent commit history/file headers),
  confirm with a follow-up `rg` of production driver call sites before editing any of them.
- Winner-switch detection itself: `winner_switching.jl`, `winners.jl:22-31`
  (`factual_prices`/`compute_winners`), `winner_certificate.jl` — recompute winner(o,d,draw) from a
  candidate A/theta point using the SAME Topic-2 gauge formula. **Shared formula needs new inputs**
  (same as Topic 2, since they inline the identical reconstruction).

---

## Topic 10 — Free-coordinate layout / pivot indexing (`free_idx`)

Confirmed: `free_idx = vcat(3+D, Aod_offset+1 : Aod_offset+D*Ddest)` (i.e. `[gp; every A_od cell]`
in absolute θ-vector indexing) is constructed **independently in 4 separate context builders**, not
a shared function:

- `context.jl:48` (D=4 fixed-size context)
- `context_scaled.jl:69`
- `context_real_d20.jl:122`
- `qmc_context_real_d20.jl:416`

**Must be generalized, in all 4 places** — under profiled scales `free_idx`'s A-block portion would
need to exclude the per-destination anchor cells (or, more likely, `free_idx` stays as the FULL
θ-vector bookkeeping and a NEW, separate "outer-searched subset of free_idx" concept is layered on
top, analogous to how `gravity_elimination.jl`'s pivot already removes 1 more coordinate from
`free_idx`'s A-block without touching `free_idx` itself). Each of these 4 files also independently
sets `Aod_offset` and constructs `ctx.γ` — worth checking for other latent duplication beyond just
this one line during implementation.

- `outer_coordinate_layout.jl` (Topic 1) and `gravity_elimination.jl` (Topic 3) `other_idx`/
  `pivot_lin`/`pgc.other_idx` are the EXISTING analogous "D*Ddest-1 free positions after removing
  one global pivot" bookkeeping — **must be generalized** the same way, extended to
  "D*Ddest-Ddest free positions after removing one pivot AND one anchor per destination," or
  composed as two sequential eliminations.
- `cm_originzc_target_layout.jl:26/45` — `SharedByPowerLayout`/`OriginByPowerLayout <:
  MeanZCTargetLayout`: a DIFFERENT, unrelated "layout" concept (ZC-restriction target grouping, not
  A-coordinate layout) — **unchanged**, don't conflate with `OuterCoordinateLayout`.

---

## Topic 11 — Feasibility screens

- `infeasibility_screen.jl:276` — `order_destinations`, `:350` — `screen_hard_winners`, `:434` —
  `build_extreme_draw_witness`, `:465` — `query_witness`, `:525` — `infeasible_result`: the original
  (dense) screening machinery, operates per-(o,d) pair on the reconstructed price/`AodPow` matrix.
- `fast_range_screen.jl:235` — `envelope_prewinner_screen`, `:299` — `screen_hard_winners_ranged`,
  `:491` — `range_screen_standalone`, `:572` — `evaluate_fullA_screened_compressed_with_cf`, `:797` —
  `build_ranged_screen_context`, `:814` — `infeasible_result_ranged`, `:868` —
  `evaluate_fullA_screened_ranged`: the current production-path "ranged" screens (compressed,
  windowed price bounds) — this is what `decode_outer_unified`'s output (`xf`) feeds into on the hot
  path.
- `flexible_theta_aspace_production.jl:142/153` — `screened_eval_flexible_A`/`_verify`,
  `flexible_theta.jl:222` — `screened_eval_flexible`, `c10_d20_production_driver.jl:353` —
  `screened_eval` (fixed-theta/legacy-z entry): the per-layout screened-evaluation entry points that
  wrap the above.
- `cm_screen_bridge.jl:68-229` — family-specific screen-precheck/verified-state wrappers
  (`cm_screen_precheck!`, `archC_base_state_screened`, `archC_verified_state_screened`,
  `cm_production_value_verified_screened`, `cm_meanzc_production_value_verified_screened`,
  `cm_originzc_production_value_verified_screened`) and `cm_frechet_cplus.jl:320/334`
  (`archC_frechet_verified_state_screened`, `cm_frechet_production_value_verified_screened`) — the 5
  per-family screened-verification entry points production checkpointed drivers actually call.

**Change category: shared formula needs new inputs**, broadly, for every screen above that reads
`AodPow`/prices per (o,d) — they all inherit Topic 2's dependency. The screens do NOT need to know
"which cell is the anchor" directly (they operate after `Aod_θ -> Aod` reconstruction, on the full
D×Ddest matrix), so this is a comparatively contained change: as long as `decode_outer_unified`
(Topic 1) produces a correct, full, reconstructed `AodPow`, these screens should be largely
**unchanged** in their own internals — flag them "shared formula needs new inputs" mainly for the
upstream dependency, not because their own logic references A-coordinate count.

---

## Topic 12 — Checkpoint / outer-parameter-vector export / resume

- `c10_d20_production_driver.jl:257/263/297` — `save_checkpoint`/`load_checkpoint`/
  `guard_checkpoint_path` (`D20CheckpointV4`): unrestricted-family legacy (fixed-theta,
  pre-`OuterCoordinateLayout`) checkpoint schema. **Legacy full reference only** relative to the
  unified driver, but still loadable/upgradable — check whether `run_polish_checkpointed_unified`
  still calls into this for backward-compat before assuming dead.
- `cm_checkpoint.jl:407-592` — `CMCheckpointV6`/`V8`/`V9` save/load + `upgrade_schema*` chain,
  `run_cm_upper_checkpointed`/`run_cm_lower_checkpointed` (`:592`, `:1264`) — the real production
  entry points for 3 of the 5 families (`:flexible_cm`, `:common_frechet`, `:cm_meanzc`, dispatched
  via `marginal_restriction`/`is_meanzc` flags at `:914-934`).
- `cm_originzc_checkpoint.jl:106-464` — `CMCheckpointV5`/`V7`/`V10` save/load + upgrade chain,
  `run_originzc_upper_checkpointed`/`_lower_checkpointed` (`:464`, `:840`) — real production entry
  point for `:origin_zc`.
- `c10_d20_production_driver_unified.jl:114/130` — `build_unified_ctx(layout, ...)`,
  `run_polish_checkpointed_unified(...)` — real production entry point for `:unrestricted`, the ONE
  driver that already threads `OuterCoordinateLayout` end to end (`:131`).

**Must be generalized** (schema bump) in all of `cm_checkpoint.jl`, `cm_originzc_checkpoint.jl`, and
the unified driver's checkpoint path — every checkpoint schema currently persists a full `w`/A-block
vector sized for the CURRENT (non-profiled) layout; a new schema version + `upgrade_schemaN_to_M`
function (following the exact existing pattern at `cm_checkpoint.jl:454/467/480` and
`cm_originzc_checkpoint.jl:127/329/368`) would be needed, consistent with how every prior coordinate
change in this repo (legacy-z -> powered-aspace, fixed -> flexible theta) was handled — this is a
well-established, low-risk pattern already exercised 3+ times in this codebase, not a new mechanism
to invent.

Note: `outer_parameter_vector` (searched literally per task list) does not exist as a named
identifier anywhere in this worktree — the outer vector is just called `w`/`w0`/`x_free0` throughout;
no single "exporter" function to centralize around.

---

## Topic 13 — `OperatorPsiBundle` (shared no-dense-G/no-legacy-H bundle)

Read `operator_psi_bundle.jl` (159 lines) in full.

- `:63-101` — `@with_kw mutable struct OperatorPsiBundle{T}`: the concrete (not abstract) shared
  bundle type. No `H`/`H_copy`/`K`/`ones`/`moments!` fields by construction — `payoff`, `grav_col`,
  `economic_state::Union{Nothing,Ref{Any}}` (aliases the family's `core_cf_ref`), `restriction_state
  ::Any` (family's own bin/contrast context) instead. **Unchanged** — the struct itself carries no
  A-coordinate-count-dependent field; it's sized by `M`/`outer_constr_index` (inner-solve dimensions),
  not outer A layout.
- `:106-111` — `select_G_from_H`: intentionally throws (fail-fast for the "no dense G" invariant).
  **Unchanged.**
- `:118-124` — `CS.inner_loop_number_variables`/`_lower_bounds`/`_initial_values`/
  `_complementarity_constraints` dispatch methods mirroring `PsiObjectiveBundleImplicit`'s. **Unchanged.**
- `:144-155` — `prime_operator!(obj, θ_econ, ctx, core_cf_ref; restriction_state)`: THE shared
  priming step every restricted family's entry point calls (file header, `:24-58`, cites side-by-side
  confirmation that `wrap_moments_with_cm_archB`/`_cm_meanzc`/`_originzc`/`_cm_frechet_archB` all do
  the identical sequence). Calls `cf_build(θ_econ, ctx; check_ties=false)` (defined elsewhere —
  builds the `CompressedFactual`, itself downstream of Topic-2's gauge transform via `ctx`/`θ_econ`),
  `fill_K_directgp!` (Topic 4, unchanged), `compressed_gravity_raw`/`fill_gravity_column_into!`
  (`compressed_live.jl:105` area — the gravity moment, Topic 4, unchanged). **Shared formula needs
  new inputs** transitively (via `cf_build`), not because `prime_operator!` itself references
  A-coordinate layout.
- 5-family construction sites confirmed by `rg`/read: `compressed_live.jl:447` —
  `build_unrestricted_operator_ctx` (unrestricted); `cm_production_bundle.jl:65` —
  `build_cm_production_context` (flexible-CM); `cm_frechet_level.jl:306` —
  `build_cm_frechet_production_context` (common-Frechet); `cm_meanzc_production.jl:151` —
  `build_cm_meanzc_production_context` (CM+ZC); `cm_originzc_production.jl:48` —
  `build_originzc_production_context` (origin-ZC). Each is a **shared formula needs new inputs** site
  transitively (their own construction logic doesn't reference A-layout, but they call
  `cf_build`/`prime_operator!`/the family's `moments!`-equivalent wrapper which does, via `θ_econ`
  slicing that assumes the CURRENT free-A-block width).
- `production_bundle_api.jl` (own file, read in full, 385 lines) — `prepare_production_run`
  (`:168-181`), `assert_production_operator_bundle!` (`:205-244`), `derive_backend_manifest`
  (`:258-304`), `write_backend_manifest_atomic` (`:374-384`): the family-agnostic factory/validator/
  manifest layer wrapping ALL 5 `build_*` functions above. **Unchanged** — purely structural
  (bundle-type checking, manifest JSON), no A-layout dependency at all.

---

## Topic 14 — Calibration / start-generation entry points (partially covered)

- `cm_aspace_coordinate.jl:98-109` — `cm_w0_from_calibration(ctx, pe, A_coordinate_mode)`: builds a
  fresh `w0` from `ctx.θ0_up[ctx.free_idx]` (the GENUINE calibrated point — confirmed NOT the
  `zfree=0` reference point the CLAUDE.md warning is about) via `pivot_reduce`/`cm_a_from_z`.
  **Must be generalized** — under profiled scales this would need to also strip the per-destination
  anchor cell(s) from `logA_full` before `pivot_reduce`, mirroring how it already strips the global
  gravity pivot cell.
- `cm_aspace_coordinate.jl:61-86` — `precompute_cm_aspace_xy`/`cm_z_from_a`/`cm_a_from_z`: the
  restricted-family-driver counterpart of `flexible_theta_aspace_production.jl`'s `APivotXY`/
  `z_from_a`/`a_from_z` (independently re-derived per that file's own header, algebraically identical
  — a second duplication instance in the same spirit as Topic 2, smaller in scope). **Shared formula
  needs new inputs.**
- Did not reach: a dedicated Pareto/data-only calibration entry point search (referenced in user
  memory as `pareto_calibration.jl` from a DIFFERENT, Melitz-model repo area — grep for that exact
  filename in this worktree returned nothing, so it is likely not part of this repo's gravity/EK
  model at all, not a coverage gap in THIS audit). Flagging as not fully reached per the task's
  budget note.

---

## Topic 15 — Production driver entry points (`RunPurpose`/`ProductionContext`)

Fully covered — see Topics 12/13 for the concrete driver functions. Summary: there are exactly
**3 real, `prepare_production_run`-calling production drivers** (confirmed by `rg -l
"prepare_production_run\("`, excluding `production_bundle_api.jl` itself and test files):

1. `cm_checkpoint.jl` — `run_cm_upper_checkpointed`/`run_cm_lower_checkpointed`, covering 3 families
   (`:flexible_cm`, `:common_frechet`, `:cm_meanzc`) via a `marginal_restriction`/`is_meanzc` dispatch
   (`:727-755`, `:907-934`).
2. `cm_originzc_checkpoint.jl` — `run_originzc_upper_checkpointed`/`_lower_checkpointed`, family
   `:origin_zc`.
3. `c10_d20_production_driver_unified.jl` — `run_polish_checkpointed_unified`, family
   `:unrestricted`, the only driver already fully wired to `OuterCoordinateLayout`.

`production_bundle_api.jl` (Topic 13) is the shared factory ALL THREE call into
(`prepare_production_run(family, runner, build_inner; extra_manifest)`), plus
`RunPurpose`/`ProductionPurpose`/`DenseReferencePurpose` (`:26-49`) as a typed (not stringly-typed)
purpose marker, and `ProductionContext{C,B<:OperatorPsiBundle}`/`DenseReferenceContext{C,B}` (`:88-108`)
as structurally distinct wrapper types. **Unchanged** for the dispatch/validation machinery itself;
each of the 3 driver files is **must be generalized** for the reasons already covered under Topics
1/3/10/12 (layout struct, pivot composition, free_idx, checkpoint schema).

---

## Topic 16 — `gravity_sample_mask` / exclude-diagonal shared utility (not found)

`rg -n "gravity_sample_mask|exclude_diagonal_gravity|offdiag_mask"` across the ENTIRE worktree
(not just `d4_exact/`) returned **zero hits**. This concurrent-task machinery the prompt flagged as
possibly having just landed on the production branch does **not exist in this worktree** as of
`cd17235` (this worktree's base commit) — either it landed on a different branch/worktree after this
one was cut, or the task prompt's premise about it being "just added" doesn't apply here yet. Closest
existing concept found: `gravity_elimination.jl`'s `other_idx`/`pivot_lin` (Topic 3) and
`row_idx`/`_ctx_ddest` (excluding ROW as a destination, Topic 2/10) are the only current
"which cells are eligible/ineligible" masks in this codebase, and they are GLOBAL (one pivot cell
across the whole matrix, one excluded destination), not per-destination-anchor-cell masks. If/when
`gravity_sample_mask` lands, it would be a natural fourth "eligibility mask" alongside these two,
directly reusable for marking each destination's profiled anchor cell as gravity-ineligible.

**Recommendation**: re-run this exact `rg` search against `production/fullA-exact`'s current HEAD (not
just this worktree's `cd17235` base) before finalizing the profiled-scales design, in case it landed
on `production/fullA-exact` after this worktree was branched.
