# Persistent L-fix Base Cache Release — 2026-07-27

Task §5's ask for a persistent, in-place `build_lfix_base_cache!(...)` to kill the ~615 MB/gradient
warm allocation. The inherited `shared-outer-a-gradient-2026-07-27` branch explicitly deprioritized
this ("NOT ATTEMPTED this pass") and left a scoping handoff proposing to build it from scratch.

## The real finding: it already existed, square-only

An existing `LFixBaseWorkspace`/`build_lfix_base_cache!`/`composite_gradient_at_Aplus` mechanism
(`lfix_base_workspace.jl`/`lfix_base_workspace_pooled.jl`, from an earlier, unrelated
"finalization task" / "Backend A+") already implements exactly the in-place, persistent-array-
reuse pattern this task's §5 wants — it was missed by the prior handoff's own scoping (which
proposed writing it from scratch). It was, however, hard-guarded square-only
(`Ddest_here == D || error(...)`), a hard error under the real D=20 production default.

## What this session did

Generalized `LFixBaseWorkspace`'s array fields (`price0`/`pTσ0`: `W x D x D` → `W x D x Ddest`;
`winner0`/`contrib0`/etc: `W x D` → `W x Ddest`; `denom`/`CONST_d`: length `D` → length `Ddest`)
and `build_lfix_base_cache!`'s body to mirror `build_lfix_base_cache`'s own already-Ddest-aware
formulas exactly — only the array provenance changes (persistent buffer vs. fresh allocation), no
arithmetic changes. Also fixed `composite_gradient_at_Aplus`'s own square-hardcoded `D2=D^2`
(previously unreachable dead code behind the hard guard, now real) and repointed
`c10_d20_production_driver.jl`'s two `:aplus` backend-selection sites to pass `ctx.D_dest`.

**Gates**: D=4 square regression unchanged (4/4 PASS). Real D=20/W=80,000 rectangular — the first
time this backend has EVER run under `:exclude_row` — 19/19 PASS: cache-field bit-identity against
the allocating reference (with `validate_dense=true` on both sides), full-gradient bit-identity of
`composite_gradient_at_Aplus` vs. the already-D20-gated `composite_gradient_at_fast_pooled`, and a
third fresh point through the same warm workspace to rule out stale-array bugs.

## What this does NOT yet do (the actual §5 allocation target is UNCHANGED)

`economic_A_gradient!` (`shared_a_gradient.jl`) — the shared gradient entry point 3 of 5 families
now call — still calls the ALLOCATING `build_lfix_base_cache` internally, not the now-rectangular-
capable `LFixBaseWorkspace`/`build_lfix_base_cache!`. `composite_gradient_at_Aplus` is a SEPARATE,
parallel gradient entry point (part of the unrestricted family's own backend-selection matrix in
`c10_d20_production_driver.jl`), not the source of `economic_A_gradient!`'s cache.

**The task's own §5 numeric target — an ≥80% additional reduction from the inherited 615 MB
figure — was NOT achieved this session.** `A_GRADIENT_WARM_D20_ALLOCATION` remains 614.83 MB,
unchanged from the inherited state. This session's real contribution is removing the single
largest blocker to closing that gap (the persistent cache existed but couldn't run under the real
D=20 rectangular default at all) — the building block now exists and is gated; wiring it into
`economic_A_gradient!` is the concrete, well-scoped next step, not a from-scratch task anymore.

## Recommended next-session approach

1. Give `EconomicAGradientWorkspace` (the struct `economic_A_gradient!` already threads through
   every family wiring) an `LFixBaseWorkspace` field, built once per `(D, Ddest, W)` alongside the
   pool/scratch it already owns.
2. In `economic_A_gradient!`, replace `cache = cache === nothing ? build_lfix_base_cache(...) :
   cache` with a call to `build_lfix_base_cache!(ws.lfix_ws, ...)` when no explicit `cache=` is
   supplied by the caller (family wrappers that pass their own pre-built restriction-folded cache,
   e.g. `build_lfix_base_cache_cm_meanzc`, are unaffected — this only changes the PLAIN, non-
   restricted cache path, which today is only reachable when `cache=nothing`).
3. Gate exactly like this session's own `test_lfix_base_workspace_d20.jl`: bit-identical vs. the
   allocating path, D=4 then real D=20, at least 2 distinct points through the same warm
   workspace, before trusting any allocation-reduction number.
4. Measure the actual warm-allocation reduction directly (the D=4 reconciliation in the inherited
   `A_GRADIENT_D20_ALLOCATION_RECONCILIATION_2026-07-27.md` already shows `build_lfix_base_cache`'s
   own fresh allocation is ~93% of the D=4 warm total for `economic_A_gradient!` — eliminating it
   is expected, not merely hoped, to clear the task's own ≥80% bar, but this must be MEASURED, not
   assumed, exactly as this project's own standing practice requires).

## Verdict

```text
PERSISTENT_LFIX_BASE_CACHE = rectangular-generalized, gated, NOT wired into economic_A_gradient!
A_GRADIENT_WARM_D20_ALLOCATION = 614.83 MB (unchanged this session)
NEXT_STEP = wire LFixBaseWorkspace into EconomicAGradientWorkspace/economic_A_gradient! (scoped,
    concrete, no longer blocked on a missing building block)
```
