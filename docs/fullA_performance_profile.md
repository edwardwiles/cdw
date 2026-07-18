# Full-A D=4 exact formulation: performance profile

Phase 1 deliverable for the "performance profiling, D=4 completion, and staged scaling"
continuation. Written before any Phase 2+ optimization work, per the task's explicit "profile first,
implement only optimizations supported by the profile" instruction.

## 1. Instrumentation and its validation

`full_aod_diag/d4_exact/instrumentation.jl` provides a minimal `@prof "label" expr` macro (wall time
+ allocated bytes + GC time via `Base.gc_num`/`GC_Diff`, the primitives `@timed`/`@allocated` use
internally — no new dependency), gated by a single `Ref{Bool}` check when disabled.
`full_aod_diag/d4_exact/oracle_profiled.jl` is an additive instrumented mirror of `oracle.jl`'s
`evaluate_fullA` — the original is untouched. `test_oracle_profiled.jl` proves bit-for-bit equivalence
(warm, cold, and cache-hit/miss paths) before any timing conclusion below is trusted.

**Granularity actually achieved** (be precise about what was and wasn't instrumented): stage
boundaries within `evaluate_fullA` (reconstruct, inner solve, moments recompute, gravity, KKT
residual, winner computation) and gradient-method-level wall time/allocation/true-inner-solve-count.
**Not achieved**: per-KNITRO-callback timing inside a single inner or outer solve (would require
patching `cc_algo/`, which this investigation avoided per its additive-only discipline) — where that
granularity matters, existing `knitro.log` files (grepped for eval/iteration counts) remain the
source, and `profile_outer_iteration.csv` is explicitly labeled as *derived* from those logs, not a
fresh live-instrumented run.

## 2. Component-level profiling (Phase 1B)

`profile_components.jl`, N=50 repetitions per condition (warm-up excluded), at the calibration point,
D=4/W=8000. Full data: `results/fullA_d4/fca622e/profile_components/profile_components.csv` and
`profile_allocations.csv`.

### Top hotspots by median wall time (warm start)

| stage | median | % of total (31.1ms) |
|---|---|---|
| `inner_solve` (actual KNITRO CC dual optimization) | 16.6ms | 53% |
| `moments_recompute` | 10.6ms | 34% |
| `winner_compute` | 3.6ms | 12% |
| `kkt_residual_compute` | 0.585ms | 2% |
| `outer_constraint_callback` | 0.299ms | 1% |
| `primal_divergence_compute` | 0.124ms | <1% |
| `gravity_compute` | 0.020ms | <1% |
| `reconstruct_full` | 0.002ms | <1% |
| **TOTAL_evaluate_fullA (warm)** | **31.1ms** | — |
| **TOTAL_evaluate_fullA (cold)** | **38.7ms** | — |

Sum of labeled components (~31.9ms) closely matches the independently-measured total (31.1ms) —
confirms the instrumentation captures essentially all wall time, no large unaccounted gap.

### Top hotspots by allocation

`winner_compute` is the single largest allocator (4.9MB/call, 487MB across 100 calls) despite modest
wall time — `compute_winners` (`winners.jl`) does 32,000 individual `sort(col)` calls per evaluation
(one per `(ω,d)` draw×destination pair at W=8000,D=4) rather than a preallocated partial-selection,
each allocating a small array. `inner_solve` and `moments_recompute` each allocate ~410MB/100 calls;
`kkt_residual_compute`'s list-comprehension pattern (`[abs(sum(m.*G[:,j])) for j in ...]`) allocates
218MB/100 calls from repeated column-slicing.

### Finding #1 (highest priority for Phase 2): a literal redundant second moment computation

`cc_algo/inner_loop_functions.jl`'s `inner_loop_internal(obj::PsiObjectiveBundleImplicit, θ)` already
calls `obj.moments!` once internally (to build the KNITRO-facing `H` matrix) before running
`inner_loop_KNITRO`. `oracle.jl`'s `evaluate_fullA` then calls `obj.moments!` **again**, afterward, to
get `K`/`G` into caller-owned buffers for the post-processing (KKT residual, gravity, moment
residuals). This second call is `moments_recompute`'s entire 34% of wall time — the single highest-
value, lowest-risk optimization target in the whole call graph: caching/reusing the first call's
output (once confirmed the `θ` passed to both calls is identical, which it is by construction) could
remove up to a third of per-evaluation cost with no algorithmic change.

### Finding #2: `winner_compute`'s allocation profile is a clear, contained optimization target

Replacing the per-column `sort(col)` (which computes a full sorted order to extract only the min and
runner-up) with a direct two-pass min/second-min scan would eliminate the dominant allocation source
without changing any computed value — `winners.jl::compute_winners` already computes exactly `wmin,
wo = findmin(col)` and a runner-up gap; only the `sort(col)`-based gap computation needs replacing.

## 3. Gradient-method cost, corrected (Phase 1A)

`profile_gradient_methods.jl` resolves a specific issue the task flagged: the prior continuation's
Phase D table reported `n_inner_solves=34` for `Q_adj_FD`/`L_fix_FD`, which do **not** call the inner
solver at all. Using `CS.INNER_SOLVE_COUNT[]` (the pre-existing production counter) via before/after
diffs instead of FD-probe counting:

| method | median wall (17-dim gradient) | TRUE n_inner_solves | n_moments! calls |
|---|---|---|---|
| `A_pathwise_AD` | 0.08-0.11s | **0** | 2 |
| `Q_adj_FD` | 0.33-0.35s | **0** | 34 |
| `L_fix_FD` | 0.33-0.35s | **0** | 34 |
| `Delta_FD` (optimized-value, ground truth) | 1.10-1.13s | **34** | 68 |

Confirmed directly, not inferred: `Q_adj_FD`/`L_fix_FD` are genuinely inner-solve-free by
construction (frozen duals, only `obj.moments!` is re-evaluated), giving a real ~3.1x wall-clock
speedup over `Delta_FD` — a more precise version of the prior continuation's "2-4x, not orders of
magnitude" finding, now on a corrected metric. `Delta_FD`'s 68 `moments!` calls (not 34) for 34 inner
solves is the same Finding #1 redundancy showing up again at the gradient-method level: every one of
its 34 inner solves triggers the same double-computation `evaluate_fullA` has.

`profile_outer_iteration.csv` (`results/fullA_d4/9f07ff6/`) is a **derived** table (existing Phase B
KNITRO logs, not a fresh instrumented outer run) giving implied evals/outer-iteration and
seconds/outer-iteration per Hessian-mode config — flagged in the CSV itself as derived, not live-
measured, given time constraints this continuation could not also build a fully-instrumented live
outer-loop driver.

## 4. Blockwise gradient accuracy — critical context for interpreting "cheap gradient" cost numbers

Phase 6 (`docs/fullA_d4_final_report.md` update pending, see `phase6_blockwise_gradient_check.jl`)
found that the cost numbers above should NOT be read as "any cheap method is a viable
Delta_FD replacement at ~1/3 the cost." Full-vector cosine (used throughout the prior continuation)
conceals large A-block-specific errors:

- **`A_pathwise_AD`**: A-block cosine goes **negative** (-0.67 to -0.79) at both upper candidates —
  cheapest (0.08-0.11s) but the gravity-tangent A-block gradient points in nearly the wrong direction
  exactly where it matters.
- **`Q_adj_FD`**: A-block direction is moderate (cosine 0.84-0.87) but magnitude is wildly wrong
  (norm ratio 14-17x too large).
- **`L_fix_FD`**: A-block cosine 0.997-0.999, norm ratio 0.98-1.00 at both upper candidates — the one
  method that is genuinely validated blockwise, not just by a gamma-dominated full-vector metric.

**Practical conclusion combining cost and blockwise accuracy**: `L_fix_FD` is the only cheap method
with both a real wall-clock advantage (~3.1x over `Delta_FD`) and blockwise-validated accuracy where
it matters (the gravity-tangent A-block). `Q_adj_FD` is the same cost as `L_fix_FD` but should not be
used without the same blockwise validation it currently fails. `A_pathwise_AD` is far cheaper but not
usable as a search direction near the optimum given its A-block sign problem — consistent with, and
now quantifying precisely, the winner-boundary bias this whole investigation has been tracking since
its first session.

## 5. D/W baseline scaling matrix (Phase 1C)

Complete — see `docs/fullA_scaling_projection.md` for the full table, fitted power-law exponents, and
D=20 projection. Headline: full-gradient cost (`Delta_FD`/`L_fix_FD`/`Q_adj_FD`) scales empirically as
**D^3.5-3.8** at fixed W=8000 (steeper than `n_free`'s exact D² growth alone — a genuine compounding
effect, not just "more coordinates"), while W-scaling at fixed D=4 looks close to linear-or-better
(exact eval ~W^0.94, gradient ~W^0.64, though the latter fit is noisy with only 3 points). Projected
D=20 cost: ~18 minutes per `Delta_FD` gradient vs. ~4.2 minutes per `L_fix_FD` gradient — the
`L_fix`-hybrid cost advantage found in §4 becomes considerably more valuable at larger D, not just a
D=4 curiosity.

## 6. Top-10 hotspots, consolidated

1. `inner_solve` (actual KNITRO CC dual optimization) — 53% of per-eval wall time.
2. `moments_recompute` — 34%, driven almost entirely by **Finding #1**'s redundant double-computation.
3. `winner_compute` — 12% of wall time, but the #1 allocator by far (4.9MB/call) — **Finding #2**.
4. `kkt_residual_compute` — 2% of wall time, #2 allocator (2.2MB/call) from column-slicing patterns.
5. `Delta_FD`'s 34-inner-solve gradient cost — the dominant cost of the current optimized-value-FD
   outer-loop driver; direct consequence of hotspot #1 repeated 34x per gradient.
6. `A_pathwise_AD`'s A-block sign error — not a wall-time hotspot, but the reason the cheapest
   gradient method cannot be used as-is.
7. `Q_adj_FD`'s A-block magnitude error (14-17x) — same cost as `L_fix_FD` with materially worse
   blockwise accuracy; a live solver choosing between them on cost alone would pick the wrong one.
8. Product-finite-difference Hessian mode (`hessopt=4`, the historically-used config): ~15-20x more
   outer-iteration wall time than genuine BFGS/SR1/L-BFGS at matched outer-iteration count (Phase B,
   prior continuation) — not re-measured this session but directly relevant to any wall-clock-matched
   frontier (Phase 7, not attempted this continuation).
9. Context/economy construction (`ctx build time`) is itself non-trivial (33-51s observed for a fresh
   D=4/W=8000 economy in the D/W scaling driver) — a one-time cost per process, irrelevant to steady-
   state per-evaluation cost but relevant to short pilot-run wall-clock budgets.
10. Full-gradient cost's D^3.5-3.8 empirical scaling (§5 / `docs/fullA_scaling_projection.md`) —
    the single biggest reason a naive `Delta_FD`-only, no-block-reuse D=20 run is not attempted this
    continuation: projected ~18 minutes per gradient evaluation before any Phase 2 optimization.
