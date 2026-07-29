# Moment/target construction audit — flexible_cm vs common_frechet (2026-07-29)

## Question

After harmonizing the inner-solve FG, Hessian, verification, and outer-loop gradient, is the
moment/target CONSTRUCTION layer (the trade-share economic moments, the CM moment block, and the
Fréchet level-block moments) also already shared, or does it hide FG-style duplication?

## Verdict: already fully harmonized. No fix needed.

Every function that matters here has exactly one definition in the repo (verified directly, not
just via the audit agent's report — spot-checked with `grep -rn "^function <name>"` for each):

| function | file:line | shared? |
|---|---|---|
| `precalc_common_marginals_cdf` | `common_marginals_moments.jl:50` | **one definition**; `build_cm_frechet_level_augmented_obj` (`cm_frechet_level.jl:149`) calls it with the identical `(ctx.U, refIndex1, L; contrasts, probs)` signature flexible_cm's own `build_cm_augmented_obj` (`common_marginals_moments.jl:205`) uses |
| `wrap_moments_with_cm` | `common_marginals_moments.jl:169` | **one definition** — generic on the supplied moment matrix. Fréchet calls it unchanged (`cm_frechet_level.jl:165`) on `hcat(CM, LEVEL)` |
| `fill_cm_columns_from_bins!` | `cm_hessian_architectures.jl:140` | **one definition**, called unchanged by both production closures |
| `compute_bin_indices` | `cm_hessian_architectures.jl:104` | **one definition**, identical call site in both families |
| `cf_build`/`materialize_dense_factual_structured!`/`compressed_gravity_raw`/`fill_K_directgp!`/`fill_gravity_column_into!` | `compressed_factual_buffer_reuse.jl`/`structured_moment_build.jl`/`compressed_live.jl`/`cm_hessian_architectures.jl` | **one definition each** — `cf_build`'s own docstring names it shared across all four restricted families |

`build_cm_frechet_level_augmented_obj` (`cm_frechet_level.jl:144-183`) and
`wrap_moments_with_cm_frechet_archB` (`cm_frechet_level.jl:213-287`) are the only Fréchet-specific
functions in this layer, and both are thin compositions: call the shared CM/economic machinery
unchanged, then add the genuinely-new level block via `precalc_frechet_level_dense`
(dense-reference path) / `fill_frechet_level_columns_from_bins!` (production path,
`cm_frechet_level.jl:100-121`) — exactly the `[E|C|F]` pattern the FG/Hessian layers already use.
The file's own header comment (`cm_frechet_level.jl:1-18`) states this explicitly: "This is a pure
ADDITIVE extension: it does not modify `common_marginals_moments.jl`, `common_marginals_interval.jl`,
or `cm_hessian_architectures.jl` -- it reuses `wrap_moments_with_cm`... and
`fill_cm_columns_from_bins!`... UNCHANGED."

**Level-block math confirmed θ-independent** (matching "fixed forever"): read
`fill_frechet_level_columns_from_bins!` in full — its only inputs are `Bidx` (bin indices, built
once from `U` and the fixed CDF thresholds, never re-derived per outer point), `D`, `L`, and fixed
`targets` (`probs`-derived, fixed at context-construction time). No `θ`, `σ`, or any outer-loop
parameter appears anywhere in the function.

## One narrow, currently-inert asymmetry found (not a duplicate, not a bug)

`wrap_moments_with_cm_archB` (flexible_cm, `cm_hessian_architectures.jl:314-316,334-336`) gates the
economic dense-fill and its copy into `G` behind `if !skip_fill`. `wrap_moments_with_cm_frechet_archB`
(common_frechet, `cm_frechet_level.jl:255-256,271`) does the same two steps **unconditionally**
(verified directly — no `if !skip_fill` guard on those lines). This is deliberate and documented in
both files: gating this fill behind `skip_fill` for flexible_cm caused an unexplained regression
that was reverted, and it was never re-attempted for Fréchet since Fréchet's own `skip_fill_safe`
is never `true` in production anyway (a separate, out-of-scope item per that same comment). Net
effect: zero difference in any real call today (both closures' production call sites always pass
`skip_fill=false`); would only matter if `skip_fill=true` were ever re-enabled for Fréchet, in which
case Fréchet would harmlessly do slightly more work than flexible_cm's analogous branch. Not a
correctness issue, not touched by this task.

## Conclusion

No refactor needed here. Unlike the pre-harmonization FG-callback layer (which had two
independently-inlined copies of the same `[E|C]` computation), the moment-construction layer never
had that problem: the CM and economic construction functions were single, shared, generic-on-input
functions from the start, and Fréchet's own construction was already written as a thin `[E|C|F]`
composition on top of them.
