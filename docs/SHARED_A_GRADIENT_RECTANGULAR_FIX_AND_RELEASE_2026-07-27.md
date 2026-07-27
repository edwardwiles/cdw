# Shared A-Gradient Rectangular Fix and Release — 2026-07-27

Task §5's "Fix the unresolved square-layout bug" — the inherited `shared-outer-a-gradient-2026-07-27`
branch disclosed, but did not fix, two real bugs in `composite_gradient_at_fast_buffered`/
`composite_gradient_at_fast_pooled`: a genuine correctness bug (fixed, adopted unchanged from that
branch, `dest_contrib_incremental_o1!`'s `D`-vs-`Ddest` stride error) and a hard crash under the
real D=20 production default (`destination_sample=:exclude_row`) that made the prior audit's own
cited "756.5 MB pooled" allocation figure unreproducible. This release fixes the crash.

## What was found and fixed

`git grep` for `reshape(x_free0[2:end], D, D)` and `D2 = D^2` across every buffered/pooled/shared
gradient file found the same square-hardcoded pattern in **7 files**, not just the 2 the task
description named:

| File | Function | Fix |
|---|---|---|
| `lfix_buffer_reuse.jl` | `composite_gradient_at_fast_buffered` | Full fix: `D2 = D*Ddest`, `reshape(D,Ddest)` |
| `gradient_workspace.jl` | `composite_gradient_at_fast_pooled` | Full fix (this is the function the "756.5 MB" figure referred to) |
| `lfix_base_workspace_pooled.jl` | `composite_gradient_at_Aplus` | Full fix (see `PERSISTENT_LFIX_BASE_CACHE_RELEASE_2026-07-27.md` — required generalizing the underlying persistent cache first) |
| `cm_meanzc_production.jl` | `full_rebuild_gradient_fallback_meanzc` | Full fix (CM+ZC's own gradient validator) |
| `lfix_pTsigma_only.jl` | `build_lfix_base_cache_B` (Backend B) | Defensive `DimensionMismatch`-style guard added, not a full rectangular rewrite (see below) |
| `lfix_kbplus.jl` | `build_lfix_base_cache_KB` (Backend KB) | Same, guard added |
| `lfix_kbplus_workspace.jl` | `build_lfix_base_cache_KB!` (Backend :kbplus) | Same, guard added — this one is reachable from the real production driver (`price_cache_backend=:kbplus`), so the missing guard was a genuine, previously-undisclosed latent crash/corruption risk under `:exclude_row`, not just a diagnostic nit |

**Scoped decision on Backends B/KB/:kbplus**: these three experimental backends' own persistent
base-cache builders are square-hardcoded *internally* (every array sized/looped `D x D`, origin
and destination never distinguished), unlike the allocating reference or Backend A+'s own
workspace. Fully generalizing them mirrors the same restructuring `build_lfix_base_cache` itself
already went through — real, correctness-sensitive numerical work, not wiring. Given they are
non-default, opt-in backends, this release adds a clean early guard (fail loudly with a specific
error) rather than a rushed rewrite, and documents the gap precisely rather than leaving an
undisclosed risk.

## Gates

`test_gradient_workspace.jl` — this is the task's own "permanent D=20 omit-ROW regression test"
requirement, and turned out to need its OWN fix first: its D=20 setup hardcoded the identical
`D^2`/`reshape(D,D)` pattern, meaning this regression test had **never once successfully run**
against the real current D=20 production default before this session (it errored at setup,
`BoundsError` on a `D^2`-sized theta slice, before ever reaching the functions it exists to test).
Fixed; now a genuine, passing, permanent regression test.

Real D=20/W=80,000 (both points x serial+threaded x pooled vs buffered): **9/9 PASS**, bit-identical
(`g_ref == g_new`, exact equality). Allocation now directly measurable for the first time under
`:exclude_row`: pooled=724.3 MB vs buffered=3504.2 MB (4.84x reduction) — this reconciles the
inherited audit's own disclosed gap that its cited "756.5 MB pooled" figure could not be reproduced
under the real production default.

## Verdict

```text
RECTANGULAR_SQUARE_LAYOUT_BUG = fixed (7 sites: 4 fully generalized, 3 defensively guarded)
GRADIENT_WORKSPACE_D20_REGRESSION_GATE = 9/9 PASS, first successful real-D20 run of this test ever
POOLED_VS_BUFFERED_D20_ALLOCATION = pooled 724.3 MB, buffered 3504.2 MB (4.84x)
```
