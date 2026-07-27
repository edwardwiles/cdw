# Persistent L-fix Base Workspace — 2026-07-27

## Status: NOT ATTEMPTED this pass

Task §4 asks for a persistent workspace + in-place `build_lfix_base_cache!` that preallocates and
refills the base-point arrays (`price0`, `pTσ0`, winner/runner-up/third-place identities and their
score arrays, `contrib0`, `q0`, `cf_contrib0`, `CONST_d`) in place at a new outer point, rather than
allocating them fresh every gradient call. This was explicitly deprioritized per the task's own
guidance ("sections 4-6 only after (c)/(d) have real evidence, since they're riskier structural
changes") — (c) and (d) DO now have real evidence (the reconciliation and the shared entry point
are both gated at D=4 and D=20), but the remaining session time after the two bug investigations
(see `SHARED_OUTER_A_GRADIENT_ARCHITECTURE_2026-07-27.md`) was spent finishing and gating the one
wiring that got done (ZC-only) rather than starting a second risky structural change.

## Why this is the right next target (evidence, not a guess)

Per `A_GRADIENT_D20_ALLOCATION_RECONCILIATION_2026-07-27.md`: after this task's fixes, the
warm-bandwidth-cache steady-state cost of `economic_A_gradient!` at D=4 is 4.80 MB, of which 4.48
MB (93%) is `build_lfix_base_cache`'s own fresh allocation. The coordinate loop itself now
contributes almost nothing to the warm-state total. This means `build_lfix_base_cache` is now THE
dominant remaining allocation site in the shared gradient, by a wide margin, at both D=4 and (by
direct scaling logic, not independently re-measured) D=20.

## Why it is genuinely riskier than what this task did complete

`LFixBaseCache` is an **immutable** Julia `struct` (not `mutable struct`). Its `Vector`/`Matrix`/
`Array{Float64,3}` FIELDS are mutable objects and could in principle be refilled in place without
changing the struct itself — but several of its OTHER fields are plain scalars
(`D::Int`, `Ddest::Int`, `μ::Float64`, `σ::Float64`, `gammafac::Float64`, `λ_cf::Float64`,
`ζstar::Float64`, `wPrime_bi::Float64`, `τPrime_bi::Float64`, `LPrime_bi::Float64`) that CANNOT be
mutated once the struct is constructed. Whether these scalars are actually invariant across
repeated gradient calls at a fixed `(D, Ddest)` depends on which family is calling this: `μ`
(`base.θ_full0[1]`) is fixed for every family covered by this task, but the codebase's own
flexible-theta overlay (mentioned in the task background, not investigated this session) may treat
`μ`/`σ` as free parameters that DO change between successive outer gradient calls — in which case
`gammafac = spgamma(μ*(1-σ)+1)` would also need to be recomputed, not just have its arrays refilled.

A safe in-place `build_lfix_base_cache!` therefore needs to either (a) confirm scalar invariance
is actually guaranteed for every caller before reusing the struct across calls, or (b) make
`LFixBaseCache` a `mutable struct` (or wrap the scalars in `Ref`s) so they can be updated too when
they DO change, adding a small amount of indirection cost to every read of those scalars throughout
the codebase's many `cache.gammafac`/`cache.σ` etc. call sites — a broader structural change than
this task's time budget allows to do carefully, verify, and gate.

## Recommended next-session approach (not attempted, offered as a starting point)

1. First confirm (grep + read, not assume) whether ANY currently-wired caller of
   `composite_gradient_at_fast`/`_buffered`/`_pooled`/`economic_A_gradient!` ever changes
   `μ`/`σ` between two calls that reuse the same cache/workspace. If none do (plausible — `μ`/`σ`
   are calibration primitives typically fixed within an outer-loop A-block optimization), the
   scalar fields are safe to leave immutable and only the array fields need an in-place refill
   path.
2. Add `build_lfix_base_cache!(cache::LFixBaseCache, x_free0, ctx, base)` that asserts
   `cache.D == ctx.D && cache.Ddest == Ddest(ctx)` (dimension-compatibility guard, cheap) and then
   refills every array field via the same formulas `build_lfix_base_cache` already uses, writing
   in place instead of allocating fresh arrays.
3. Gate it exactly like this task's own `select_bandwidth!`/`dest_contrib_incremental_top3!`: bit-
   identical output vs the existing allocating `build_lfix_base_cache`, at D=4 first, then real
   D=20 (ideally under BOTH `:exclude_row` and `:all_legacy` given this task's own discovery that
   the two modes are not always exercised equally by existing tests).
4. Thread the reused cache through `EconomicAGradientWorkspace` (add an `LFixBaseCache`-shaped
   mutable holder, or make `LFixBaseCache` itself mutable) so `economic_A_gradient!` can skip
   `build_lfix_base_cache`'s allocation entirely on repeat calls at a fixed `(D, Ddest, W)`.

No code for this was written this session — this document is a scoped handoff, not a partial
implementation.
