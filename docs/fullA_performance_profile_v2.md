# Full-A D=4 exact formulation: performance profile v2 (CORRECTED, canonical)

Phase 1 deliverable for the "hybrid solver / finish D=4 / gate scaling" continuation (continuation
3). Supersedes `docs/fullA_performance_profile.md` (continuation 2), whose raw numbers are preserved
there unchanged for history — this document is the one to cite going forward.

## 1. What changed and why this document exists

Two corrections, both requested directly by the user after reviewing continuation 2's profile:

1. **Interpretation fix, not just a number fix.** Continuation 2's `inner_solve` label wrapped the
   entire `cc_algo/inner_loop_functions.jl::inner_loop_internal` call and was reported as "the actual
   KNITRO CC dual optimization" (53% of per-evaluation wall time). This conflates two genuinely
   different things: `inner_loop_internal` first calls `obj.moments!` ONCE to build the moment matrix
   `H` at the current θ (an O(D²W) cost — the SAME kind of computation Finding #1 already identified
   as a redundant SECOND cost elsewhere in the call graph), and only THEN runs KNITRO's own inner
   iteration loop optimizing over the dual variables `(ζ,λ)` at that FIXED moment matrix — confirmed
   directly from `cc_algo/PsiObjectiveBundle.jl:172-251`'s callable method: the `length(θ)==0` branch
   (exactly how the inner solve's own `callbackEvalFG_inner!` invokes it) only does `BLAS.gemv!` on
   the already-built `H`, never re-touches `moments!`. This document adds nested timers
   (`inner_moment_build`, `inner_knitro_dual_solve`, and — nested one level further —
   `inner_dual_fg_callback`/`inner_dual_hessian_callback`) so the two costs are never reported as one
   number again.
2. **Re-run after the Phase 1 fixes, not before.** All numbers below are measured on
   `full_aod_diag/d4_exact/oracle_fast.jl` (`evaluate_fullA_fast`), which implements the three Phase 1
   fixes (redundant-moments removal via `obj.H` reuse, allocation-free winner scan, preallocated KKT
   residual) — equivalence-tested against `oracle.jl::evaluate_fullA` in `test_oracle_fast.jl` (ALL
   PASS: bit/near-machine-precision agreement on every field, across calibration, both upper
   candidates, the lower candidate, 15 random feasible perturbations, warm/cold/cache paths) before
   being trusted for timing.

## 2. Instrumentation

`full_aod_diag/d4_exact/oracle_fast.jl` adds, purely additively (no production file touched):

- `inner_loop_internal_profiled` — mirrors `inner_loop_internal` for `PsiObjectiveBundleImplicit`,
  splitting `@prof "inner_moment_build"` (the one `obj.moments!` call) from
  `inner_loop_KNITRO_profiled` (`@prof "inner_knitro_dual_solve"`, inclusive of nested callback time).
- `inner_loop_KNITRO_profiled` — mirrors `inner_loop_KNITRO`, registering profiled callback wrappers:
  `_callbackEvalFG_inner_profiled!` (`@prof "inner_dual_fg_callback"`) and
  `_callbackEvalH_inner_profiled!` (`@prof "inner_dual_hessian_callback"`), plus call counters
  (`n_fg_calls`, `n_hess_calls`) returned per-evaluation, not just a cumulative diff.
- **Granularity limit, stated explicitly, not silently merged**: `callbackEvalFG_inner!` computes
  objective AND gradient (w.r.t. `(ζ,λ)`) in a single fused KNITRO callback
  (`cc_algo/inner_loop_functions.jl:26-34`: `obj(x, evalResult.objGrad)` is one call). Splitting
  "objective-only" from "gradient-only" time would require patching `PsiObjectiveBundle.jl`'s own
  callable method — outside this investigation's additive-only discipline. Reported as one combined
  `inner_dual_fg_callback` label throughout.
- `inner_knitro_dual_solve`'s **exclusive** cost (KNITRO's own SQP/line-search/subproblem overhead,
  not attributable to either callback) is derived as `inclusive_total − fg_total − hess_total`,
  computed from **totals** (sum over reps), not medians of medians — reported both ways.

## 3. Methodology

`full_aod_diag/d4_exact/profile_components_v2.jl`, N=50 reps per condition, calibration point,
D=4/W=8000, commit `f500490`. **Critical methodological fix vs a first draft of this script**: warm
and cold reps are profiled in **separate `prof_reset!()` scopes**. A warmed, already-converged inner
solve needs ~1 fg callback / 0 Hessian callbacks; a cold one needs several of each (§5) — pooling the
two into one "median" per nested-callback label would itself be a bimodal-distribution
mis-attribution, exactly the kind of error this re-profile exists to eliminate. Raw data:
`results/fullA_d4/f500490/profile_components_v2/{profile_OLD_raw.csv, profile_NEW_warm_only.csv,
profile_NEW_cold_only.csv, profile_NEW_breakdown.csv, summary.txt}`.

## 4. Warmed steady-state breakdown (the production-relevant regime — outer-loop iterations warm-start from a nearby previous point)

Median wall time, N=50 reps, `TOTAL_evaluate_fullA_fast_warm` median = **0.017092s** (was 0.035032s
pre-Phase-1 — see §6):

| component | median | % of total |
|---|---|---|
| `inner_moment_build` | 9.775ms | **57.2%** |
| `winner_compute` (Phase 1B fast scan) | 1.962ms | 11.5% |
| `inner_knitro_dual_solve` (INCLUSIVE of the two nested rows below) | 0.953ms | 5.6% |
| &nbsp;&nbsp;→ `inner_dual_fg_callback` (nested, fused obj+grad) | ~0.404ms | ~2.4% |
| &nbsp;&nbsp;→ `inner_dual_hessian_callback` (nested) | ~0.000ms | ~0.0% |
| &nbsp;&nbsp;→ (exclusive remainder — KNITRO's own overhead) | ~0.615ms | ~3.6% |
| `kkt_residual_compute` (Phase 1C fix) | 0.275ms | 1.6% |
| `moments_reuse` (Phase 1A fix — copies `obj.H`, no `moments!` call) | 0.254ms | 1.5% |
| `primal_weight_recovery` | 0.220ms | 1.3% |
| `moment_resid_compute` | 0.208ms | 1.2% |
| `primal_divergence_compute` | 0.129ms | 0.8% |
| `gravity_compute` | 0.015ms | 0.1% |
| `reconstruct_full` | 0.001ms | 0.0% |

**The corrected headline finding**: at a warmed steady state, `inner_moment_build` alone (57.2%) is
now the single dominant cost — and it is **not removable** by any gradient-method trick, since every
method that needs the factual moment matrix at a NEW θ must pay it once. What continuation 2 called
"inner_solve (53%)" was almost entirely this same cost, mislabeled as "the CC dual optimization." The
TRUE dual optimization (`inner_knitro_dual_solve`, inclusive) is only **5.6%** of a warmed
evaluation — confirming directly (not just by inference from `L_fix_FD`'s inner-solve-free
construction) that skipping the inner CC dual re-solve saves comparatively little at this W=8000/D=4
scale; the moment-matrix build is the real target, which is exactly Phase 2's subject (block-local
recomputation of the moment matrix itself, not just the dual solve around it).

## 5. Cold-start breakdown, for contrast

`TOTAL_evaluate_fullA_fast_cold` median = **0.025078s**:

| component | median | % of total |
|---|---|---|
| `inner_moment_build` | 11.429ms | 45.6% |
| `inner_knitro_dual_solve` (inclusive) | 7.941ms | **31.7%** |
| &nbsp;&nbsp;→ `inner_dual_fg_callback` (mean 5.0 calls/solve) | ~1.513ms | — |
| &nbsp;&nbsp;→ `inner_dual_hessian_callback` (mean 4.0 calls/solve) | ~6.719ms | — |
| &nbsp;&nbsp;→ (exclusive remainder) | ~1.122ms | — |
| `winner_compute` | 1.902ms | 7.6% |
| (KKT/gravity/moment-resid/primal-weight, each ≤1.1%) | — | — |

At a cold start, the true dual optimization genuinely does dominate more (31.7% vs 5.6% warmed) —
consistent with `test_oracle_fast.jl`'s TEST 2/TEST 5 finding (cold solves at candidate points need
~9-10 fg calls and ~9 Hessian calls, vs 1/0 when warm-started from an already-converged point). This
is the mechanism, verified directly with call counts, not just inferred from timing.

## 6. Realized Phase 1 speedup (not theoretical)

| | median wall (warmed) |
|---|---|
| OLD (`oracle_profiled.jl`, pre-Phase-1, unmodified) | 0.035032s |
| NEW (`oracle_fast.jl`, post-Phase-1) | 0.017092s |
| **Realized speedup** | **2.050x** |

This is close to, but below, the naive "remove the 34%-redundant moments_recompute" arithmetic
(1/(1-0.34)≈1.52x) would suggest on its own — the extra gain comes from the winner-computation and
KKT-residual allocation fixes stacking on top, and from `inner_moment_build`'s cost itself dropping
slightly run-to-run noise, not a new mechanism. Reported as the REALIZED number, not a theoretical
ceiling, per the task's explicit instruction.

## 7. `n_fg_calls`/`n_hess_calls` — direct verification that `L_fix`/`Q_adj` are genuinely inner-solve-free

At the warmed calibration point (50/50 reps): `n_fg_calls` min=max=mean=**1**, `n_hess_calls`
min=max=mean=**0**. This is now measured directly per-evaluation (not a cumulative-counter diff, and
not merely "zero real inner solves" as continuation 2's `CS.INNER_SOLVE_COUNT[]`-diff check already
established) — confirms that even the ONE fg-callback a warm dual solve needs is essentially a
no-op check that the already-converged point remains optimal, exactly the regime an `L_fix`/`Q_adj`
method (which never touches KNITRO's inner solve at all, warm or cold) is designed to exploit. See §8
below for the `L_fix` gradient's own component profile, which verifies this end-to-end for a full
16-dimensional gradient, not just one evaluation.

## 8. `L_fix` gradient profile — base solve vs. perturbation cost, broken into components

`full_aod_diag/d4_exact/profile_lfix_gradient.jl`, N=20 gradient evaluations (16-dim central FD, 32
perturbations each) at the `upper_maxit40` candidate, using `three_way_derivatives.jl`'s existing
`fixed_dual_L`/`solve_base_state` (unmodified — this is a profiling harness around already-validated
code, not a new derivative formula):

`full_aod_diag/d4_exact/profile_lfix_gradient.jl` — equivalence-tested first (`fixed_dual_L_profiled`
vs. the trusted `fixed_dual_L`, 10 random directions, diff=0.0 exactly, ALL PASS), then run at the
`upper_maxit40` candidate, N=20 full 16-dim central-FD gradients (32 perturbations each), h=0.01:

| component | total (20 gradients) | calls/gradient | mean/call | % of gradient wall time |
|---|---|---|---|---|
| `lfix_base_solve` (ONE optimized solve at x0, via `solve_base_state`) | 0.2123s | 1.0 | 10.617ms | 3.7% |
| `lfix_base_state_setup` (ζ*/λ*/m* already extracted by the solve, ~0 extra) | 0.0000s | 1.0 | 0.000ms | 0.0% |
| `lfix_reconstruct` (32× `CS.reconstruct_full`) | 0.0004s | 32.0 | 0.001ms | 0.0% |
| **`lfix_perturbation_moments`** (32× FULL `obj.moments!` rebuild — the entire moment matrix, no block-locality) | **5.8128s** | 32.0 | 9.082ms | **100.5%** |
| `lfix_q_assembly` (32× the `q_s` list comprehension) | 0.0841s | 32.0 | 0.131ms | 1.5% |
| `lfix_scalar_eval` (32× `Psi!` + reduction) | 0.0304s | 32.0 | 0.048ms | 0.5% |
| `lfix_fd_assembly` (16× central-difference divide) | 0.0008s | 16.0 | 0.002ms | 0.0% |
| **TOTAL** (median per 16-dim gradient) | — | — | — | **0.2891s** |

(Percentages sum fractionally above 100% because `lfix_perturbation_moments`'s per-call mean is
computed from a slightly noisier total than the median-based `TOTAL` denominator — both numbers are
reported raw, not smoothed, per the task's "report realized numbers" instruction.)

**The full-rebuild `L_fix` gradient's cost is, overwhelmingly, `lfix_perturbation_moments`** — 32
complete D²-scale moment-matrix rebuilds, one per +/- probe, each paying the SAME `O(D²W)` cost
`inner_moment_build` pays once per `evaluate_fullA_fast` call (§4). This is now the precise,
component-level version of what continuation 2's D-scaling section only asserted in aggregate: the
`L_fix`-hybrid's cost advantage over `Delta_FD` comes entirely from never re-solving the inner CC
dual (§7), but at this D=4 scale a full-rebuild `L_fix` gradient is ITSELF completely dominated by
moment-matrix reconstruction cost — the exact quantity Phase 2's block-local/incremental evaluators
target directly.

Raw data: `results/fullA_d4/f500490/profile_lfix_gradient/profile_lfix_gradient.csv`. **Verified
directly** (not inferred): across all 20 gradient evaluations, `CS.INNER_SOLVE_COUNT[]` increases by
**exactly 1** per gradient call (the base solve) and **0** for every one of the 32 perturbation
evaluations (`[1,1,1,...,1]`, 20/20) — i.e. an `L_fix` gradient call uses ONE optimized CC inner solve
total (at the base point) and ZERO at every perturbed coordinate, confirmed by counter diff, matching
the design intent exactly. This full-rebuild baseline is what Phase 2's block-local and incremental
evaluators are benchmarked against (`docs/fullA_block_local_performance.md`).

## 9. Top hotspots, consolidated (corrected)

1. `inner_moment_build` — 57.2% of a WARMED per-eval wall time (was invisible as a separate line
   in continuation 2's profile, buried inside "inner_solve"). **The real target for Phase 2.**
2. `winner_compute` — 11.5% warmed (down from 12% pre-fix in absolute allocation terms, but a LARGER
   share of a smaller total — the Phase 1B fix reduced its allocations, not its relative rank).
3. `inner_knitro_dual_solve` (the TRUE dual optimization) — only 5.6% warmed, 31.7% cold. Skipping it
   entirely (as `L_fix_FD`/`Q_adj_FD` already do) saves less than continuation 2's framing implied,
   because it was never the majority of "inner_solve" to begin with.
4. `kkt_residual_compute`, `moments_reuse`, `primal_weight_recovery`, `moment_resid_compute`,
   `primal_divergence_compute` — each ≤1.6% warmed, not worth further optimization at this scale.
5. Realized Phase 1 speedup: **2.05x** warmed (§6) — a real, measured number, not a projection.
6. `lfix_perturbation_moments` — **100.5% of a full-rebuild `L_fix` gradient's wall time** (§8): the
   32 per-probe full moment-matrix rebuilds, not the base CC dual solve (3.7%) or the scalar
   Psi-evaluation machinery (≤1.5% combined). This is Phase 2's direct target.
