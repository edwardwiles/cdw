# Shared economic moment-state builder — 2026-07-27

Branch: `feature/shared-economic-moment-state-builder-2026-07-27`, from `e772e81`
(`release/shared-FG-verification-and-A-gradient-2026-07-27`).

Task: implement one shared, allocation-free production implementation for constructing/refreshing
the common economic moment-state block (winner identities, winner contributions, counterfactual
contribution, draw-independent bookkeeping) consumed by all five production families
(unrestricted, flexible CM, common Fréchet, CM+ZC, ZC-only), one level below the FG/gradient/
Hessian/verification layer a parallel session is unifying on the same release branch.

## 1. What already existed before this session (attribution)

A large fraction of what the addendum asks for was **already built and production-wired** by two
earlier sessions this project ran on 2026-07-25/26 (see MEMORY `allocation-hessian-production-
merge-2026-07-25` and the "Phase E remediation" comments throughout `cm_*.jl`), specifically in
`full_aod_diag/d4_exact/compressed_factual_buffer_reuse.jl`:

- `CompressedFactualWorkspace` — the persistent, campaign-lifetime `winner`/`wval`/`cf_raw` buffer
  struct (2026-07-25 §3.1).
- `build_compressed_factual!(ws, θ_full, ctx; check_ties)` — the validated in-place builder,
  bit-identical to the allocating `build_compressed_factual` (2026-07-25 §3.1, re-verified this
  session — see gate doc).
- `attach_compressed_factual_workspace(ctx, D, Ddest, W)` — attaches a workspace to `ctx` once per
  live production context, no-op-reuse on a matching shape (2026-07-25 §3.1). Already called by
  both `c10_d20_production_driver.jl` and `c10_d20_production_driver_unified.jl`.
- `cf_build(θ_full, ctx; check_ties)` — a dispatch helper used by all **four restricted families**'
  `moments!` closures (`cm_hessian_architectures.jl`, `cm_meanzc_moments.jl`, `cm_frechet_level.jl`,
  `cm_originzc_moments.jl`), added 2026-07-26 ("Phase E remediation"). It already implements
  exactly the addendum's requested dispatch: reuse `ctx.cf_workspace` via `build_compressed_factual!`
  when attached, fall back to the allocating builder otherwise.
- The restriction-append pattern in all 4 restricted families already matched addendum §3's shape
  exactly: `cf = cf_build(θ, ctx)` → `materialize_dense_factual_structured!(view, cf)` → a
  family-specific column-fill function (`fill_cm_columns_from_bins!` for CM, and analogous
  functions for meanZC/Fréchet-level/originZC) — **no separate winner search, no second
  `build_compressed_factual` call, no second compressed object, no separate indexing convention**
  in any of the 4 restricted families. Verified by direct grep of all four files (only one
  `cf_build`/`materialize_dense_factual_structured!` call site each).

**This means 4 of the 5 families already satisfied the addendum's core requirement before this
session started.**

## 2. The real gap found and fixed this session

The **unrestricted family's own real hot path** — `compressed_live.jl::inner_loop_internal_
compressed`, called once per KNITRO inner dual solve under `moment_representation=:compressed`
(the production driver's default mode; see `c10_d20_production_driver.jl`'s
`evaluate_fullA_screened_ranged(...; moment_representation = :compressed, ...)`) — called the
always-allocating `build_compressed_factual(θ_full, ctx; check_ties=true)` **directly**, bypassing
`cf_build`/`ctx.cf_workspace` entirely, even though the production driver always attaches a
`cf_workspace` to `ctx` before any solve. The 2026-07-25/26 remediation fixed the 4 restricted
families' call sites but missed this 5th one — the unrestricted family's own moment build was
still allocating `~24.96 MB` (at D=20/W=80,000, the buffer sizes measured by the pre-existing
`test_compressed_factual_buffer_reuse.jl`) fresh on **every single inner solve**, for the entire
lifetime of every unrestricted-family production run.

A secondary, already-partially-fixed call site (`c10_d20_production_driver.jl`'s dual-bank
warm-start scoring build, §3.1 2026-07-25) had its own hand-inlined `cf_ws === nothing ? ... :
...` ternary duplicating the same dispatch logic — simplified to call the canonical function
directly (behavior unchanged, one less duplicate implementation of the same dispatch rule).

## 3. What this session changed

### 3.1 New canonical name: `build_economic_moment_state!`

`compressed_factual_buffer_reuse.jl` now defines

```julia
build_economic_moment_state!(θ_full::AbstractVector, ctx; check_ties::Bool = true) -> CompressedFactual
```

as THE canonical, single production entry point. It dispatches to the in-place
`build_compressed_factual!` whenever `ctx.cf_workspace::CompressedFactualWorkspace` is attached,
and falls back to the allocating `build_compressed_factual` otherwise (exercised only by
diagnostic/test contexts that never call `attach_compressed_factual_workspace` — no real
production driver hits this branch, verified by the runtime counter gate).

`cf_build` is now a **byte-identical alias** for `build_economic_moment_state!` — kept so the four
already-wired restricted-family call sites need no rename. New call sites (including this
session's fix) spell the canonical name.

**Naming-mismatch disclosure** (per this task's own instruction to report, not paper over, any gap
between the addendum's illustrative names and the actual codebase): the addendum's example names
`reuse_immutable_cm_state`/`update_low_dimensional_zc_targets!` do **not** exist anywhere in this
codebase and were not invented to match — the equivalent, already-production-validated
functionality is `materialize_dense_factual_structured!` (shared across all 4 restricted families)
plus each family's own already-existing column-fill function:

| Family | "append restriction" step (already existed, unchanged) |
|---|---|
| Flexible CM | `fill_cm_columns_from_bins!` (`cm_hessian_architectures.jl`) |
| Common Fréchet | its own level-state fill (`cm_frechet_level.jl`) |
| CM+ZC | its own bin-fill + `update_zc_targets`-equivalent (`cm_meanzc_moments.jl`) |
| ZC-only | its own ZC target fill (`cm_originzc_moments.jl`) |

No family-specific file was renamed or restructured this session — they already called the shared
builder (`cf_build`, now `build_economic_moment_state!`) correctly.

### 3.2 Fixed production hot path

`compressed_live.jl::inner_loop_internal_compressed` (unrestricted family) now calls
`build_economic_moment_state!(θ_full, ctx; check_ties=true)` instead of the raw allocating
builder. Bit-identical output either way (same guarantee `build_compressed_factual!`'s own
docstring establishes) — verified this session at D=4 square, D=4 rectangular, and D=20/W=80,000
real data (see `FIVE_FAMILY_INPLACE_COMPRESSED_FACTUAL_GATE_2026-07-27.md`).

Also updated for consistency (not production hot paths — see the callsite audit doc for the
classification evidence): `compressed_live_v2.jl::inner_loop_internal_compressed_v2` and
`compressed_inner_alt_solvers.jl::inner_loop_internal_compressed_variant` (both explicitly
documented in their own module headers as benchmark-only, never wired into
`evaluate_fullA_fast`'s `moment_representation` dispatch).

### 3.3 Runtime counters (addendum §6)

Added to `compressed_moments.jl` (defined there, not `compressed_factual_buffer_reuse.jl`, so
they exist for every one of the ~50 existing scripts that include only `compressed_moments.jl`):

- `ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS`
- `INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS`
- `ECONOMIC_WORKSPACE_ALLOCATIONS`
- `ECONOMIC_WORKSPACE_REFILLS`
- `ECONOMIC_WORKSPACE_RESIZES`
- `DUPLICATE_ECONOMIC_STATE_BUILDS`

`reset_economic_moment_state_counters!()` and `economic_moment_state_counters()` (snapshot
NamedTuple) are provided for driver-level reporting. `DUPLICATE_ECONOMIC_STATE_BUILDS` is
instrumentation-only (new `last_theta`/`has_last` fields on `CompressedFactualWorkspace`,
compared by value on each `build_compressed_factual!` call) — it does **not** change any
correctness/caching decision; every call still fully recomputes the state, it is purely a runtime
signal for auditing redundant rebuilds. See the gate doc for a caught real instance (the dual-bank
warm-start scoring build and the main inner solve build the SAME `θ_full` twice per warm callback
— a genuine, honestly-reported inefficiency this task's own scope did not include fixing).

### 3.4 Ownership / immutability

- `CompressedFactualWorkspace` (`winner`, `wval`, `cf_raw` buffers): owned by whichever `ctx` it is
  attached to via `attach_compressed_factual_workspace`; lifetime = the live production context's
  lifetime (survives `set_context_delta!` delta-stage changes by reference, per the pre-existing
  2026-07-25 guarantee, unaffected by this session).
- `CompressedFactual` (the struct returned by `build_economic_moment_state!`): a **fresh, cheap**
  wrapper struct on every call (scalars + array references), but its heavy `winner`/`wval`/
  `cf_raw` fields **alias** the workspace's persistent buffers — they are NOT copied. Callers must
  not retain a `CompressedFactual` across a subsequent call to `build_economic_moment_state!` on
  the same workspace (unchanged from `build_compressed_factual!`'s pre-existing contract);
  confirmed safe for every current production consumer (each `cf` is consumed immediately within
  one inner solve / one Hessian callback, never retained across the next moment-state build).
- `ctx.cf_workspace`: read-only from the perspective of every family's `moments!` closure — only
  `attach_compressed_factual_workspace`/`ensure_compressed_factual_workspace!` ever replace it
  (and only on a genuine shape change).

## 4. Downstream consumers (addendum §4)

Confirmed unchanged / already correctly wired to the shared state (no separate winner/value arrays
built per consumer):

- Economic FG forward/transpose: `compressed_cc_value_grad`/`_v2`, unaffected (consume `cf`
  produced by the now-fixed builder).
- Strict operator verification / shared winner-pair (`H_EE`): `core_cf_ref[]` publishing pattern
  in `cm_hessian_architectures.jl` unchanged — still receives the `cf` from `cf_build`
  (`build_economic_moment_state!`).
- C+ outer gradient / structural screens: unaffected (out of this task's scope; not touched).
- Winner-aware `H_ER`: not implemented in this codebase as of this session (consistent with the
  parallel FG/Hessian-unification session's own scope notes) — no consumer to wire.

## 5a. Final verdict block (addendum §10)

```
ECONOMIC_MOMENT_BUILDER =
    unrestricted:build_economic_moment_state!/shared_inplace   (FIXED this session -- was allocating)
    flexible_cm:cf_build(build_economic_moment_state! alias)/shared_inplace   (already shared_inplace pre-session)
    common_frechet:cf_build(build_economic_moment_state! alias)/shared_inplace   (already shared_inplace pre-session)
    cm_plus_zc:cf_build(build_economic_moment_state! alias)/shared_inplace   (already shared_inplace pre-session)
    zc_only:cf_build(build_economic_moment_state! alias)/shared_inplace   (already shared_inplace pre-session)

ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS = 0   (verified across D=4 square, D=4 rectangular, and
    real D=20/W=80,000 unrestricted-family runs with a workspace attached this session -- see gate
    doc; NOT independently re-verified with a live counter check for the 4 restricted families this
    session, though their code path was already unchanged from pre-session and pre-session gates
    re-pass)
ECONOMIC_WORKSPACE_RESIZES_AFTER_WARMUP = 0   (verified this session, unrestricted family, D=4 and
    real D=20/W=80,000)
ECONOMIC_STATE_DUPLICATE_BUILDS = 0 is NOT claimed as the production-default steady state --
    DUPLICATE_ECONOMIC_STATE_BUILDS is a real, nonzero, honestly-measured count in one identified
    spot (c10_d20_production_driver.jl's dual-bank warm-start scoring build re-derives the same
    θ_full the main inner solve then also builds, on essentially every warm production callback --
    see §5 below). This is a genuine pre-existing inefficiency this task's own scope did not
    include fixing, now measurable for the first time via the counter this task added. It was NOT
    introduced by this session's changes (the scoring build already existed and already duplicated
    work before this session; this session only added the instrumentation that reveals it).
```

**Production-ready qualifier**: the unrestricted family (the one this session actually fixed) now
meets the addendum's production-ready bar (`ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS = 0`,
`ECONOMIC_WORKSPACE_RESIZES_AFTER_WARMUP = 0`) for its own moment-state build. Whole-driver
`ECONOMIC_STATE_DUPLICATE_BUILDS = 0` is **not yet true** (see the dual-bank scoring-build overlap
above) — reported honestly, not rounded to a clean pass.

## 5. Honest gaps / out of scope this session

- The dual-bank warm-start scoring build and the main inner-solve build construct the economic
  state **twice** for the same `θ_full` on essentially every warm production callback (now
  measurable via `DUPLICATE_ECONOMIC_STATE_BUILDS`, see gate doc). Fixing this (reusing the
  scoring `cf` for the main solve) was judged out of scope / too risky to validate against
  KNITRO's stateful solve in this session's time budget — reported, not fixed.
- Section 4's "C+ outer gradient / structural screens where compatible" was audited for call-site
  presence only (no `build_compressed_factual` direct call found there), not independently
  re-derived/re-validated end to end this session.
- No changes were made to any file the parallel FG/verification/Hessian session is touching
  (`gravity_elimination.jl` and friends were not opened for editing) — see the "files touched"
  list in the final session report for the exact diff surface.
