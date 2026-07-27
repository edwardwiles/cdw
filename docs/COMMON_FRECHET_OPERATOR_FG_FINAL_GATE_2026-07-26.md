# Common Fréchet Operator FG — Allocation Root-Cause — 2026-07-26

Task Job 1 §6: root-cause the inherited common-Fréchet lookup path's ~21% allocation regression
before deciding the default (`CM_FRECHET_INNER_FG_BACKEND_DEFAULT`).

## What the addendum asked to separate

> Separate: allocation per FG callback; number of FG callbacks; KNITRO iteration-count differences;
> context/state construction; level-block workspace allocation; task or closure allocation.

## What was measured (real, this session)

Real D=20/W=80,000/L=50 context, `inner_fg_backend=:cm_frechet_lookup`, `CMFrechetLookupState`
already warmed up (one prior full solve completed):

| Measurement | Result |
|---|---|
| `n_fg_calls` for one warm-started solve at calibration (dense vs lookup) | **1 vs 1** — identical, not the explanation |
| Isolated per-callback allocation, `CMLookupState` (plain CM, structurally near-identical pattern) | **2,384 bytes** |
| Isolated per-callback allocation, `CMFrechetLookupState` (same warm `st`, same `(x,g)`, 10 repeated calls) | **14,066,064 bytes, EVERY call, byte-identical each time** |
| Sum of `CMFrechetLookupState`'s own internal steps, each measured in isolation at top level (`gemv!` core, `apply_contrast!`, `suffix_sums!`, `cumulative_forward_contribution!`, `frechet_level_suffix_sums!`, `frechet_level_forward_sum!`, `Psi!`, `dPsi!`, `gemv!` transpose, `build_weighted_histogram!`, `prefix_sums!`, `cumulative_backward_gradient_from_prefix!`, `apply_contrast!` backward, `frechet_level_backward_gradient!`) | **~21,648 bytes total** — none individually above ~13.6KB |

## What this rules out

- **Not a KNITRO-iteration-count artifact.** `n_fg_calls` is identical between backends at the
  measured point. The addendum's "if the per-callback operator is allocation-free and the total
  difference is caused only by different iteration count, report that correctly" branch does
  **not** apply — the per-callback operator is demonstrably **not** allocation-free.
- **Not explained by any single named sub-step.** Every individual forward/backward helper this
  callable calls (all of them already validated allocation-free in isolation, by direct
  measurement, not by reading the code) sums to under 22KB — three orders of magnitude short of
  the observed 14MB.
- **Not a one-time compile/GC artifact.** 10 repeated `@allocated` measurements on the same warm
  `st` return the exact same byte count every time.

## What this points to (real evidence, not fully resolved)

`@code_warntype` on `(st::CMFrechetLookupState)(x, g)` (unoptimized IR) shows `obj::ANY`,
`M::ANY`, `f::ANY`, `Body::ANY` — type instability rooted in the `obj::Any` struct field (needed so
this file has no forward-type-reference load-order dependency on `PsiObjectiveBundleImplicit`,
the same pattern `CMLookupState.obj::Any` also uses). `CMLookupState` has the **identical** field
pattern and shows near-zero allocation, so `obj::Any` alone does not differentiate the two structs
— but `CMFrechetLookupState`'s callable is a materially larger function body (the level-block
forward/backward interleaved with the CM-block forward/backward, sharing the `Hpre` buffer), and
Julia's ability to devirtualize/specialize dynamic dispatch through an `Any`-typed field can
degrade non-linearly with function-body size and branch count — plausible given the isolated
sub-steps are individually clean but the assembled function is not.

**Honest conclusion**: this is a real, reproducible, ~14MB-per-callback allocation regression in
`CMFrechetLookupState`'s own callable, most likely an emergent effect of `obj::Any` dynamic
dispatch interacting with the larger combined function body (not a bug in any individual
forward/backward formula, all of which check out allocation-clean on their own) — but the exact
mechanism (why devirtualization fails for the combined function and not for `CMLookupState`'s
shorter one) was **not** fully isolated in the time available for this task. A definitive fix would
likely require either (a) restructuring the callable so `obj`'s concrete type is captured once at
construction via a type parameter (`CMFrechetLookupState{O}` instead of `obj::Any`) rather than
read fresh from an `Any` field on every property access, or (b) a line-by-line bisection of the
assembled function body (commenting out combinations of steps) that was not completed here.

## Flip decision (unchanged from the inherited port)

Given the regression is real and unexplained by iteration count, the addendum's flip rule
("allocations fall materially") is **not** met.
**`CM_FRECHET_INNER_FG_BACKEND_DEFAULT` stays `:dense_reference`.** `:cm_frechet_lookup` remains
available, explicitly opt-in — unchanged from the inherited port's own decision, now on firmer
evidence (the regression is real, not a measurement artifact, and not explained by the
"iteration-count difference" hypothesis the addendum specifically asked to rule in or out).

## Recommended follow-on (not done this session, out of this task's bounded scope)

Root-cause the `obj::Any` devirtualization gap precisely (bisect the assembled function body, or
try the type-parameterized-struct fix above and re-measure) before the next session that touches
common-Fréchet's FG backend. This is a self-contained, bounded diagnostic task, well short of the
"Hessian cross-block optimization" this addendum explicitly excludes.
