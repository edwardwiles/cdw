# Profiled inner/outer layout contract (2026-08-01)

## Purpose

This document is the human-readable companion to
`full_aod_diag/d4_exact/profiled_outer_gradient_layout_contract_2026-08-01.jl`. It defines the
stable interface between:

- the **inner workstream** (`architecture/profiled-restricted-inner-endtoend-2026-08-01` or its
  descendant) — owns reduced economic layouts, restriction-family inner dual layouts, inner FG/
  Hessian, restriction-preparation freshness, inner KNITRO solve;
- this **outer-gradient workstream** — owns profiled outer-coordinate decoding, the shared economic
  A/gp fixed-dual gradient, family adapters, the gravity-pivot chain rule, outer-gradient caches,
  gradient verification, and the matched outer-search A/B harness.

The outer-gradient workstream consumes **only** the five accessors below (plus two small
convenience accessors, `family_kind`/`layout_checksum`, and one caller-supplied function,
`restriction_contrib0`) — never a hard-coded offset, family dimension, or manual index arithmetic
on top of `ctx.obj`/layout widths.

## The five required accessors

```julia
profiled_economic_layout(fctx) -> ProfiledEconomicMomentLayout
economic_dual_range(fctx)      -> UnitRange{Int}
restriction_dual_ranges(fctx)  -> Vector{RestrictionDualRange}
profiled_anchor_spec(fctx)     -> AnchorSpec
profiled_outer_coordinate_layout(fctx) -> PivotGravityElimOnRetained
```

`fctx` is a small, per-family wrapper object ("family context") — **not** the inner workstream's own
`ctx`/`obj`/`st` objects directly. This indirection is deliberate: it lets each family choose
whatever internal representation is convenient, as long as it can answer these five questions.

### 1. `profiled_economic_layout(fctx) -> ProfiledEconomicMomentLayout`

The shared economic-moment layout. **Identical across all five families by construction** — this is
not an assumption this branch is making ahead of the inner branch; the inner branch's own
`build_reduced_base_obj_for_family` (`profiled_restricted_family_base_2026-08-01.jl`, read-only
reference, not modified here) already reuses `ProfiledEconomicMomentLayout` unchanged for every
restricted family's economic block width bookkeeping (`n = 1 + layout.total_reduced_economic_moments`).

### 2. `economic_dual_range(fctx) -> UnitRange{Int}`

Indices into the **solved dual vector** `β = x[2:end]` (KNITRO primal at the inner optimum — not the
outer `w_profiled` vector) that belong to the economic block, in the same order
`profiled_economic_layout(fctx)`'s `retained_full_factual_j`/`france_ratio_reduced_j` expect. Length
must equal `profiled_economic_layout(fctx).total_reduced_economic_moments` — enforced by
`validate_family_layout_contract`, not assumed.

For every family instantiated so far (the real unrestricted family, and the four mock restricted
families built on the same real economic layout), this range is `1:total_reduced_economic_moments`
— i.e. the economic block occupies the *leading* slice of `β`, with any restriction duals appended
after. This is a natural consequence of `build_reduced_base_obj_for_family`'s own convention, but
`economic_dual_range` is read as an accessor, never assumed to start at 1 by any code in this branch.

### 3. `restriction_dual_ranges(fctx) -> Vector{RestrictionDualRange}`

Empty for the unrestricted family. One `RestrictionDualRange(name::Symbol, range::UnitRange{Int})`
per restriction block for a restricted family (e.g. CM marginals, ZC targets, Frechet shape).
Disjoint from `economic_dual_range(fctx)` and from each other — enforced.

### 4. `profiled_anchor_spec(fctx) -> AnchorSpec`

The same `AnchorSpec` used to build the economic layout. Must satisfy
`(spec.D, spec.Ddest) == (layout.D, layout.Ddest)`.

### 5. `profiled_outer_coordinate_layout(fctx) -> PivotGravityElimOnRetained`

The gravity-pivot-composed-with-anchor-reduction outer coordinate layout (shared across families —
gravity and the anchor reduction are model-level, not family-level, restrictions). Must satisfy
`pe.spec === fctx`'s own `profiled_anchor_spec(fctx)` by object identity, not just structural
equality — enforced.

## Two convenience accessors and one caller-supplied function

```julia
family_kind(fctx)       -> Symbol   # :unrestricted, :flexible_CM, :common_Frechet, :ZC_only, :CM_plus_ZC
layout_checksum(fctx)   -> UInt64   # claimed structural_checksum(...) -- validated, not trusted blindly
restriction_contrib0(fctx, ev) -> Vector{Float64}  # length cf.W, per-draw, A/gp-INDEPENDENT
```

`restriction_contrib0` is the one place a restriction-dual VALUE enters the shared engine at all —
see the economic-gradient theorem section below. It must be `SW`-unweighted (matching
`const_part`/`contrib0`/`cf_raw_κcf`'s own pre-`SW` convention inside `build_shared_profiled_lfix_cache`).

## `validate_family_layout_contract(fctx) -> NamedTuple`

Every call into the shared engine (`build_shared_profiled_lfix_cache`, `shared_family_outer_gradient`,
`diag_profiled_full_rebuild_gradient`) runs this first. It throws (never warns, never silently
coerces) on:

- `economic_dual_range(fctx)` length mismatched against `total_reduced_economic_moments`;
- any restriction range overlapping the economic range or another restriction range;
- `profiled_anchor_spec(fctx)` dimension mismatch against the economic layout;
- `profiled_outer_coordinate_layout(fctx).spec !== profiled_anchor_spec(fctx)`;
- `layout_checksum(fctx)` disagreeing with the independently recomputed `structural_checksum`;
- the economic layout itself failing `assert_no_factual_price_index_moment` (an anchor cell
  smuggled into the retained/economic block).

All eight of these were exercised as explicit negative tests and confirmed to throw
(`PROFILED_RESTRICTED_OUTER_GRADIENT_PREINTEGRATION_GATE_2026-08-01.csv`).

## The economic-gradient theorem this contract exists to exploit

At a fixed family restriction-parameter vector, the restriction moments (CM/Frechet/ZC) do not
depend on relative-A coordinates or `gp`. Their contribution to the fixed-dual functional
`q[w] = -zeta - t[w]` is therefore a per-draw, A/gp-**independent** constant — playing exactly the
same role the unrestricted engine's own `const_part`/`cf_raw_κcf` terms already play (both
draw-dependent, coordinate-independent, folded into `q0` once and never revisited during the
coordinate loop). `restriction_contrib0(fctx, ev)` is that constant.

**Important, checked scope of this theorem** (see the master doc's "on the restriction-invariance
claim" section for the full derivation): this licenses treating `restriction_contrib0` as **fixed
across every ± coordinate probe within one gradient call** — it does NOT mean the resulting gradient
is numerically independent of `restriction_contrib0`'s *magnitude* (`Psi` is nonlinear, so a
different fixed baseline genuinely shifts the FD result). The two claims were kept separate and
tested separately (`PROFILED_RESTRICTION_MOCK_FAMILY_GATE_2026-08-01.csv`, tests A and B) — do not
conflate them.

## Current status (2026-08-01)

Only `UnrestrictedFamilyCtx` (real, wraps the already-validated unrestricted ctx/spec/pe/layout) and
`MockRestrictedFamilyCtx` (synthetic, task §11's "mock restricted layouts") implement this contract
today (`full_aod_diag/d4_exact/profiled_family_adapters_2026-08-01.jl`). The inner branch does not
yet expose live typed accessors for any of the four restricted families. When it does, only the four
`build_mock_restricted_family_ctx`-style mock constructors need to be replaced by real ones over the
inner branch's own restricted-family context types — nothing in the shared engine, the gradient
formula, the full-rebuild reference, or the outer A/B harness needs to change, since all of them
consume the five accessors only.
