# Canonical Winner-Engine Integration Manifest — 2026-07-24 (port-prep)

Per the task brief: "Design this branch so that, when the canonical-winner-engine branch lands,
integration requires only swapping or adapting the core winner-state provider... Do not add
another winner-scanning implementation."

## 1. There is currently no dedicated `CoreMomentOperator`/`WinnerState` type

Confirmed by direct grep of `production/fullA-exact @ c55e81e` (no hits for `struct.*Operator`,
`MomentOperator`, `WinnerState`): the canonical carrier of "the core trade/equilibrium moment
block" is the plain `ctx` `NamedTuple` returned by `d20_real_setup`/`d20_real_setup_design`, whose
`.obj::CS.PsiObjectiveBundleImplicit` field holds the moments closure, and whose
`.D`/`.D_dest`/`.row_idx`/`.active_origins`/`.active_destinations` fields describe the active
rectangular layout. This is what every current restricted-model builder
(`build_cm_production_context`, `build_cm_meanzc_production_context`,
`build_cm_originzc_production_context`, and this branch's `build_cm_frechet_production_context`)
consumes and returns wrapped (`ctx_cm = merge(ctx, (obj = obj_cm,))`).

**This port does not invent a "CoreMomentOperator" type either** — it follows the exact same
`merge(ctx, (obj=...))` convention every other restricted-model family already uses, so that if/
when a canonical winner-engine branch introduces a real typed operator, this port's own
`ctx`-consuming code sites (listed in §3 below) are no different in shape from every other
restricted family's, and can be updated by the same mechanical change applied uniformly across
all of them — not a fixed-Fréchet-specific adapter.

## 2. Screens: the current entry point, and how this port uses it unmodified

`cm_screen_precheck!(x_free0, ctx_cm; counters=nothing, use_witness=false)`
(`cm_screen_bridge.jl:110`) runs the fast, draw-free EXACT certificates — pairwise
(`pairwise_certificate`), hard-winner (`screen_hard_winners`), and optionally witness
(`query_witness`) — **before** any KNITRO inner solve, throwing `CMExpectedSolveFailure` the
instant a certificate proves the point infeasible. It is rectangular/`:exclude_row`-safe by
construction: it iterates `ctx_cm.D`/`ctx_cm.U`/the base context's active destination set only.

This port's screened wrappers, `cm_frechet_base_state_screened` /
`cm_frechet_verified_state_screened` (`cm_frechet_production_bundle.jl`), call
`cm_screen_precheck!(x_free0, fpcx.ctx_cm; ...)` — **the identical function flexible-CM's own
`archC_base_state_screened`/`cm_production_value_verified_screened` call** — then delegate to
`cm_frechet_base_state`/`cm_frechet_verified_state`. No fixed-Fréchet-specific screen logic exists
anywhere in this branch.

**Soundness argument (task brief §9, restated precisely for this port)**: a core infeasibility
certificate from `cm_screen_precheck!` is valid for fixed-Fréchet, because the fixed-Fréchet
feasible set is a **subset** of the core-feasible set (adding the CDF/POWER equality restrictions
can only shrink the feasible region — the same nesting property verified numerically at D=4,
`Δ*_flexible ≤ Δ*_frechet,cdf ≤ Δ*_frechet,cdf+power`, gate P3 in
`test_frechet_power_hessian_d4_gates.jl`). Core feasibility does **not** imply fixed-Fréchet
feasibility — only the full inner solve (or, in the future, a fixed-Fréchet-specific certificate
this port does not attempt to build) can certify that. `cm_screen_precheck!` is therefore used
purely as a cheap **pre-filter**, exactly as flexible-CM already uses it, never as a
fixed-Fréchet-sufficient certificate.

## 3. Every file/function in this branch that consumes core winner/moment state

| File | Function | What it consumes from `ctx`/core |
|---|---|---|
| `frechet_reference_targets.jl` | `build_frechet_reference_targets` | `ctx.U` (W×D draws), `ctx.D`, `ctx.μHat`, `ctx.σ` — asserts `size(ctx.U,2)==ctx.D` |
| `cm_frechet_moments.jl` | `precalc_frechet_reference_cdf`, `build_cm_frechet_augmented_obj(_archB)` | `ctx.U`, `ctx.D`, `ctx.γ.refIndex1`, `obj0 = ctx.obj` (for `ncore`/`d`/`outer_constr_index` bookkeeping) |
| `cm_frechet_bases.jl` | `precalc_frechet_reference_{cdf,power_cdf,interval,power_interval}`, `build_cm_frechet_augmented_obj_basis` | same as above, both bases/both families |
| `cm_frechet_hessian.jl` | `build_cm_frechet_bin_ctx` | delegates to production's own `build_cm_bin_ctx(ctx, aug)` — **zero fixed-Fréchet-specific winner logic**, this IS the "swap the provider centrally" seam |
| `cm_frechet_power_hessian_structured.jl` | `build_frechet_power_bin_ctx` | same seam, plus `ctx.U` directly (for `Upow = ctx.U.^pw`) |
| `cm_frechet_production_bundle.jl` | `build_cm_frechet_production_context` | `ctx.obj.threshold_state` (propagated), `ctx.m` (`CS.FreeParamMap`, via `CS.reconstruct_full`), delegates `:common_flexible` to the existing `build_cm_production_context(ctx, CS; ...)` unmodified |
| `cm_frechet_production_bundle.jl` | `cm_frechet_base_state_screened` / `..._verified_state_screened` | `cm_screen_precheck!(x_free0, ctx_cm; ...)` — the shared screen entry point, §2 above |

**No file in this branch re-implements winner scanning, pairwise screening, or hard-winner
detection.** The single seam where a future canonical winner engine would plug in is
`build_cm_bin_ctx(ctx, aug)` (called by both `build_cm_frechet_bin_ctx` and
`build_frechet_power_bin_ctx`) and the `ctx.obj`/`d20_real_setup_design(...)` construction itself —
identical to every other restricted family's seam, not a new one.

## 4. Expected canonical API (best current guess, not binding)

Based on the task brief's own description ("core-moment operator / winner-state interface") and
the absence of any such type today, the most likely shape a canonical winner engine would
introduce is a typed wrapper around what `ctx`/`ctx.obj` currently provide ad hoc — e.g. something
carrying `.D`/`.D_dest`/`.active_destinations`/`.obj`/`.pairwise`/`.witness` under one name. If it
lands as a drop-in replacement for the `ctx` NamedTuple's own field access pattern (i.e. every
current `ctx.D`, `ctx.U`, `ctx.obj` access continues to resolve, whether via NamedTuple field
access or an equivalent property on the new type), **no code in this branch needs to change** —
every access in the table above already goes through exactly that field set. If instead it
replaces `d20_real_setup_design`'s return shape non-additively (removing or renaming fields this
branch reads), the adapter needed is a thin translation at the single point where this branch calls
`d20_real_setup_design`/consumes `ctx` — no fixed-Fréchet-internal logic would need to change.

## 5. Likely merge conflicts

- **File-level**: none expected — every file this branch adds (`cm_frechet_*.jl`,
  `frechet_reference_targets.jl`, `test_frechet_*`) is new; no existing production file is
  modified by this branch.
- **Semantic**: if the canonical winner engine changes `build_cm_bin_ctx`'s signature or
  `CMBinHessCtx`'s field layout (both currently reused unchanged by this branch — see
  `docs/CM_PRODUCTION_HOOK_INTERFACE_SPEC_2026-07-24.md` §1), `cm_frechet_hessian.jl`'s
  `build_cm_frechet_bin_ctx` and `cm_frechet_power_hessian_structured.jl`'s
  `build_frechet_power_bin_ct` would need their calls to `build_cm_bin_ctx` updated to match — this
  is the single, narrow, expected integration point, isolated to two one-line call sites plus
  `FrechetPowerBinHessCtx`'s `cctx::CMBinHessCtx` field type if the struct itself is renamed.
- If the canonical engine changes `cm_screen_precheck!`'s signature, this branch's two screened
  wrappers (`cm_frechet_base_state_screened`/`cm_frechet_verified_state_screened`) need their one
  call site each updated to match — again isolated, not scattered.
