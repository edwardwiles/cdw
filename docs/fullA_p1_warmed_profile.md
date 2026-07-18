# Continuation 5, Priority 1: warmed profiling (canonical)

Branch `diag/fullA-d4-exact`, worktree `../gravity-fullA-d4`. This document collects Priority 1's
three sub-parts (A: exact-hard evaluation, B: `lfix_composite` gradient, C: smoothed value/gradient).
Written incrementally as each part completes; §C is complete, §A/§B follow.

## Part C: smoothed value/gradient -- the 3.147s contradiction, RESOLVED

Script: `full_aod_diag/d4_exact/profile_p1c_smoothed_warmed.jl`. Raw data:
`results/fullA_d4/<commit>/profile_p1c_smoothed_warmed/{contradiction_resolution.csv,component_breakdown.csv}`.

### Root cause (confirmed by direct measurement, not inferred)

`run_smoothed_homotopy.jl`'s archived "GRADIENT BENCHMARK" section times
`t1 = @elapsed (g1 = ForwardDiff.gradient(ww -> smoothed_fixed_dual_L(...), w_bench))` -- a **single,
unwarmed call of a closure literal that had never been invoked before at that source location**. In
Julia, each `->` literal is its own compiled type; even though `smoothed_fixed_dual_L`/
`ForwardDiff.gradient` were already warm from the homotopy stages' own (differently-located) closure,
this specific closure/dual-number specialization pays a full one-time JIT cost the very first time
it's called -- exactly what an `N=1`, no-pre-warm timing always risks.

Isolated directly (same closure OBJECT reused for both the first call and 30 warmed reps, so there is
no confound from a fresh literal each time):

| rho | point | unwarmed first call | warmed median (N=30) | speedup |
|---|---|---|---|---|
| 0.0054 | upper_lfixcomposite_sr1_60s | **3.8544s** | **0.0445s** | **86.7x** |
| 0.0027 | upper_lfixcomposite_sr1_60s | 0.0539s | 0.0465s | 1.2x |

The rho=0.0054 row reproduces the archived contradiction almost exactly (3.85s vs the archived
3.147s -- same order of magnitude, same root cause, different point/rho so not expected to match to
the digit). The rho=0.0027 row, run immediately after in the SAME process, shows the "first call" is
already fast (0.054s) because the JIT compilation is a genuinely ONE-TIME, whole-process cost (Julia
caches specializations across calls with the same closure/argument types, and ForwardDiff's chunk
size/dual-number machinery is the same at both rho values) -- direct confirmation the effect is
compilation, not a property of rho or the smoothed kernel itself.

**Resolved number: the smoothed gradient's true warmed steady-state cost is ~0.044s (44ms) per
16-dim gradient call at D=4**, not 3.147s. This is consistent with the live homotopy logs (hundreds
of gradient evaluations completing within the 11-18s per-stage wall time budgets) -- do cite this
number, not the archived one, in any future hard-vs-smoothed comparison (Priority 3).

### Warmed component breakdown (N=20-30 reps per component, pre-warmed before timing)

| component | rho=0.0054 | rho=0.0027 |
|---|---|---|
| `smoothed_moments!` (K,G build at theta0) | 5.5ms | 5.1ms |
| `inner_loop_internal` (smoothed CC dual, WARM) | 7.3ms | 6.1ms |
| `inner_loop_internal` (smoothed CC dual, COLD) | 16.1ms | 14.7ms |
| `smoothed_fixed_dual_L` (value callback) | 6.8ms | 5.3ms |
| `ForwardDiff.gradient` of the scalar envelope (warmed) | **43.7ms** | **44.0ms** |
| materialized-Jacobian route (`ForwardDiff.jacobian` + manual contraction, diagnostic only) | 53.4ms | 51.7ms |

Consistent with `fullA_smoothed_consistent_experiment.md`'s original (compilation-contaminated)
finding that the two AD routes are close in wall time (here: scalar-envelope route is a genuine,
warmed **1.19-1.22x faster** than the materialized-tensor route, not the previously reported ~1.05x
from an unwarmed/partially-warmed comparison) -- direction of the finding survives, magnitude is now
measured cleanly. `ForwardDiff.gradient`'s allocation (~77MB/call) remains high (chunk size 8 for
16 free coordinates) -- not addressed this continuation (out of scope for the hard-focused Priority 2
work), flagged as a secondary lever if the smoothed route is pursued further in a future continuation.

## Part A: exact-hard evaluation, fresh warmed breakdown + the F/G callback redundancy audit

Script: `full_aod_diag/d4_exact/profile_p1a_warmed_and_redundancy.jl`. Raw data:
`results/fullA_d4/<commit>/profile_p1a_warmed_and_redundancy/{part1_warmed_breakdown.csv,part2_redundancy_audit.txt}`.

### Fresh warmed breakdown at the current canonical candidate (`upper_lfixcomposite_sr1_60s`)

`evaluate_fullA_fast`, N=50 reps, pre-warmed, median TOTAL = **12.84ms** (vs. the ~17ms/~35ms figures
in `docs/fullA_performance_profile_v2.md` §4/§6, measured at the calibration point on an earlier
commit -- same rank order of hotspots, slightly cheaper here):

| component | median | % of total |
|---|---|---|
| `inner_moment_build` | 7.971ms | 69.2% |
| `winner_compute` | 1.971ms | 14.8% |
| `inner_knitro_dual_solve` (inclusive) | 0.378ms | 2.6% |
| `inner_dual_fg_callback` (nested) | 0.181ms | 1.2% |
| everything else (KKT/moment-resid/primal-weight/divergence/moments_reuse/gravity/reconstruct) | ≤0.17ms each | ≤1.1% each |

`inner_moment_build` remains the dominant, structurally-unremovable cost, confirming
`fullA_performance_profile_v2.md`'s §9 conclusion still holds at the new candidate/commit.

### The F+G callback redundancy audit (task's explicit Priority 1A/2.3 requirement)

Measured directly, not assumed, using the ACTUAL functions `run_d4_optimized_fd.jl`'s `cb_F!`/`cb_G!`
call (`evaluate_fullA` for F, `composite_gradient_at` for G under `D4X_GRADIENT_METHOD=lfix_composite`),
warmed, N=30 reps, at `upper_lfixcomposite_sr1_60s`:

| call pattern | warmed median | inner solves/call |
|---|---|---|
| `eval_F` alone (`evaluate_fullA`) | 21.3ms | 1.00 |
| `composite_gradient_at` alone, **UNSHARED base (current wiring)** | **141.0ms** | 1.00 |
| cb_F!+cb_G! sequence, UNSHARED (as currently wired in `run_d4_optimized_fd.jl`) | 159.5ms | 2.00 |
| cb_F!+cb_G! sequence, **SHARED base (the fix)** | 148.6ms | 1.00 |
| **savings from sharing** | **10.9ms (6.8%)** | 1 fewer inner solve/call |

Equivalence check: `max|g_unshared - g_shared| = 0.0` exactly (sharing the base state changes nothing
mathematically, as it must -- `composite_gradient_at` already accepted an optional `base` kwarg for
exactly this, just never wired up in the live driver).

**Finding, precise and load-bearing for Priority 2's design**: the redundant base-state solve
`eval_grad_dispatch` currently pays IS real (confirmed, not hypothesized) but small -- only **11ms
(6.8%)** of the live composite-gradient callback's cost. The dominant cost is `composite_gradient_at`
ALONE: **141ms**, of which only ~10-12ms is its own (shared-away-able) inner solve -- **the other
~130ms is inside the per-coordinate bandwidth-selection + adaptive-h FD loop itself** (15 A-block
coordinates, `select_bandwidth` + two `a_block_fd_component` calls each), run serially with no
threading and no bandwidth caching across nearby outer points. This number (141ms) reproduces the
continuation prompt's documented 0.14-0.18s live-callback range exactly, and now localizes it: the
base-state-sharing fix (Priority 2 item 3) is worth implementing (it's free, correct, and stacks with
everything else) but is NOT the primary lever -- the A-block loop itself (Priority 2 items 1-2:
threading + bandwidth caching) is where the ~130ms actually goes. See Part B below for a component-
level breakdown of that loop.

### What this changes about prior claims

- `fullA_smoothed_consistent_experiment.md`'s "Gradient benchmark" section (`3.147s`,
  `Method 1 / Method 2 time ratio ~1.05x`) is **not wrong about the underlying computation** -- the
  gradients it reports are correct and were used correctly throughout the homotopy run (the outer
  solve itself never depended on that one timing number) -- but the **speed comparison** drawn from
  it (methods "nearly identical," "avoiding the materialize-tensor path buys little") is now
  superseded by the clean warmed numbers above: the scalar-envelope route IS meaningfully faster
  (~1.2x), just not by the huge margin a naive reading of "44ms vs 3147ms" would have suggested before
  this resolution.
- No smoothed-vs-hard speed comparison should cite the old 3.147s figure going forward; use the
  44ms warmed number in this document instead (Priority 3 does exactly this).
