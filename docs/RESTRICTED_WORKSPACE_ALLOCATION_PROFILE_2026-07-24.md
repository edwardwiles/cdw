# Restricted-workspace allocation profile + matrix-free follow-up assessment (2026-07-24/25)

## What changed

For `wrap_moments_with_originzc` and `wrap_moments_with_cm_meanzc`, per outer-point moment
construction:

- **Before**: fresh `G_tmp = similar(G, n, ncore_econ)` every call; mean/pair centered blocks
  built via allocating helpers (`mean_columns_direct`/`mean_columns_anchored`/`pair_columns`,
  each returning a full `W x D` or `W x npair` temporary) then copied into the destination `G`
  view (`G[:,cols] .= mean_columns_direct(...)`).
- **After**: `G_tmp` is a closure-captured cache, resized only when `n` changes, reused across
  every call. The centered blocks are written directly into the `G` view via in-place
  `mean_columns_direct!`/`mean_columns_anchored!`/`pair_columns!` -- a single broadcast, zero
  intermediate allocation.

Flexible CM's own CM-grid block (Architecture B/C) already had this treatment before this task
(see the prior 2026-07-24 session's `CURRENT_CM_BASIS_AND_STORAGE_AUDIT_2026-07-24.md`) and is
untouched here.

## Measured allocation reduction (N=20 reps, real D=20/W=80,000/L=50)

| family | point | dense (MB) | cached (MB) | reduction |
|---|---|---|---|---|
| CM+meanZC | P0 | 22,787 | 15,453 | 32.2% |
| CM+meanZC | P1 | 22,982 | 15,558 | 32.3% |
| originZC  | P0 | 22,870 | 15,517 | 32.1% |
| originZC  | P1 | 22,787 | 15,453 | 32.2% |

Consistent ~32% reduction across both families and both points -- mechanically expected (the
eliminated allocation is a fixed fraction of the total moment-build allocation, independent of
which economic point is evaluated) and matching the prior session's ~30% N=5 estimate.

Peak RSS is unchanged between dense and cached at every point measured (see the raw CSV) --
expected, since RSS is dominated by the persistent `U`/`Zraw_all`/`Zpairraw_all`/CM buffers this
change does not touch, not by the transient `G_tmp` and centered-block temporaries this change
eliminates. The win is GC pressure (fewer, smaller collections during a long multi-thousand-
evaluation campaign), not peak memory footprint.

## Is centered-block materialization still a meaningful fraction of runtime after this port?

Partially, and less than before, but not eliminated. From the N=20 benchmark's own instrumented
breakdown (`restricted_workspace_benchmark_raw_2026-07-24.csv`):

- `total_moment_build_s` (20 reps) fell from ~66-72s (dense) to ~50-68s (cached) across the four
  family/point combinations -- a real, consistent per-family reduction, but moment construction
  was never the dominant cost at any point measured: `total_hess_s` and `total_knitro_solve_s`
  are both larger than `total_moment_build_s` at every P1 point (e.g. CM+meanZC/P1: moment_build
  ~62-71s vs hess ~72-75s vs knitro_solve ~88-92s, all over the SAME 20 reps).
- The still-materialized part is the final `@. dest = Z - nu` / `Zpair - nu^2` broadcast itself --
  this task's change removed the allocating *temporary-then-copy* step, not the fill itself,
  which the current dense inner-dual-callback consumer genuinely needs (task's own explicit
  constraint: the callback still requires centered columns in the final moment buffer).

## Recommendation on matrix-free follow-up

**Not justified by this profile.** The remaining centered-block fill is (a) a minority contributor
to total inner-solve wall time at every point measured (behind both the structured Hessian
contraction and the KNITRO dual solve itself), and (b) already reduced to a single in-place
broadcast with no allocation -- there is no further allocation-side win available without the
full matrix-free forward/transpose/Hessian-contraction redesign the task brief explicitly
descoped (section 9). A matrix-free rewrite would target the Hessian/solve cost, which this task's
own numbers show dominates -- but that is a materially larger, architecturally different project
(rewriting the inner dual consumer and structured Hessian interface), not an extension of this
port. Recommend revisiting only if a future profiling pass shows moment construction becoming the
dominant cost at a different (D, W, K) operating point than the one measured here.
