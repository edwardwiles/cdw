# Five-Family Shared (A)-Gradient Wiring — 2026-07-27

Honest per-family status. "Wired" means the family's public gradient entry point calls
`economic_A_gradient!` (shared_a_gradient.jl) by DEFAULT, with the original unbuffered
`composite_gradient_at_fast` reachable only through an explicit opt-in debug backend.

| Family | Entry point | Status | Evidence |
|---|---|---|---|
| Unrestricted | `c10_d20_production_driver.jl` | **UNCHANGED this task** — still defaults to `composite_gradient_at_fast_buffered` (per a prior session's port), with `composite_gradient_at_fast_pooled` opt-in. NOT repointed to the new shared `economic_A_gradient!` this pass (would require touching a large, actively-used production driver file outside this task's time budget; also `composite_gradient_at_fast_buffered` itself works correctly at D=20 — Bug 1 was specific to the `_pooled`/`gradient_workspace.jl` path `_buffered` doesn't fully share... **correction, see below**) | Not gated this task |
| Flexible CM | `cm_production_bundle.jl:306` (`cm_production_gradient` or equivalent) | **NOT WIRED** — still calls plain `composite_gradient_at_fast` | Not attempted (time) |
| Common Fréchet | `cm_frechet_cplus.jl:190,303` | **NOT WIRED** — still calls plain `composite_gradient_at_fast` | Not attempted (time) |
| CM+ZC | `cm_meanzc_production.jl:304` | **NOT WIRED** — still calls plain `composite_gradient_at_fast` | Not attempted (time) |
| ZC only | `cm_originzc_production.jl:cm_originzc_production_gradient` | **WIRED, DEFAULT** (`gradient_backend=:shared_inplace_pooled`); `:legacy_unbuffered` reachable as an explicit debug backend | D=4: `test_originzc_shared_a_gradient_gate.jl`, 3 `(K_mean,K_pair)` layouts × full gradient (economic block + every `eta_{o,k}` coordinate), 12/12 checks PASS, bit-identical. D=20: the underlying `economic_A_gradient!` itself is gated at real D=20/W=80,000 (`test_shared_a_gradient_d20.jl`, bit-identical to `composite_gradient_at_fast`) but the ZC-only WRAPPER (`cm_originzc_production_gradient`) was not separately re-gated at D=20 this session (time) — the wiring is a thin pass-through of the already-D20-gated function, so risk is low, but this is disclosed as not independently re-verified at the wrapper level at D=20. |

**Correction on the unrestricted family**: `composite_gradient_at_fast_buffered` (the unrestricted
family's actual default) does NOT contain the Bug-1 stride error — that bug was isolated to
`dest_contrib_incremental_o1!` in `gradient_workspace.jl`, used only by `_pooled`'s
`lfix_incremental_at_ws!`. `_buffered`'s own `lfix_incremental_at!` (`lfix_buffer_reuse.jl`) calls
the plain, non-mutating `dest_contrib_incremental_o1` (correct, `cache.Ddest`-based) for its
contribution computation — only its `q`/`psi` buffers are reused, not the price/contribution
computation itself. So the unrestricted family's actual current default (`_buffered`) is believed
correct at D=20/`:exclude_row` (not exhaustively re-verified this session, but the specific bug
found does not apply to it). It DOES, however, share `_pooled`'s OTHER bug (Bug 2, the hardcoded
`reshape(...,D,D)`) if a caller ever switches it to `use_pooled_gradient=true` at the current D=20
default — that path would crash, not silently corrupt.

## Why only one family got wired

This was a large task under a real time budget. Given the two unplanned detours (finding and
fixing Bug 1, discovering and disclosing Bug 2), the remaining time was spent making the ONE
wiring that was completed (ZC-only) solid — real D=4 gate across 3 layouts, full gradient including
every `eta` coordinate, both `:shared_inplace_pooled` and `:legacy_unbuffered` paths compared
bit-for-bit — rather than spreading thin across 4 families with weaker gates on each. This matches
the task's own prioritization guidance ("pick the simplest... honestly reporting the other three as
not-yet-wired if you run out of time").

## What wiring the other three would look like (for the next session)

Flexible CM, common Fréchet, and CM+ZC all follow the exact same shape as ZC-only's
`cm_originzc_production_gradient`:

1. Their gradient entry point already builds a family-specific pre-fixed `LFixBaseCache` (e.g.
   `build_lfix_base_cache_cm_meanzc`, `build_lfix_base_cache_cm`) and passes it to
   `composite_gradient_at_fast` via the existing `cache=` kwarg.
2. Add a `gradient_backend::Symbol = :shared_inplace_pooled` kwarg and an
   `econ_ws::Union{Nothing,EconomicAGradientWorkspace}=nothing` kwarg, mirroring
   `cm_originzc_production.jl`'s pattern exactly (including the `get_or_build_econ_a_grad_ws`
   process-wide cache, or a dedicated one per family if per-family isolation is preferred).
3. Route `:shared_inplace_pooled` through `economic_A_gradient!(g_econ, base, ctx_cm, pe, ws;
   cache=cache, kwargs...)` exactly as `cm_originzc_production_gradient` does.
4. Build a `test_<family>_shared_a_gradient_gate.jl` mirroring
   `test_originzc_shared_a_gradient_gate.jl`'s structure at D=4, then a D=20 companion.

No new machinery is needed — `economic_A_gradient!` already accepts an arbitrary pre-built
`LFixBaseCache`, which is the only family-specific input any of these three would need to supply.
