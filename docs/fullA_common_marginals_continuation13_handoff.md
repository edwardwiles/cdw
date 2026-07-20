# Continuation 13: production-grade common-marginals full-A solver, D=20/delta=1 upper bound

Status: D4 gate PASSED with a genuine scientific result (kappa now monotone in L, resolving
Continuation 12's open puzzle). Production bundle (Architecture B moments + Architecture C
Hessian + CM-aware Lfix gradient) built, validated, and wired into a real KNITRO outer loop.
Interval-native Architecture C (user-elevated mid-session) implemented and validated; the
decisive at-scale test recommends AGAINST adopting it for production (conditioning reverses
badly at D20). Real D20/W=80000/delta=1 outer run launched with a 40-min/stage budget; see
Section 8 for the result (filled in once the background run completes -- this document is
being written while that run is still in progress, per the user's "keep going" instruction).

## 1. Branch isolation (non-negotiable per the brief)

- Latest clean, reviewed production commit: `diag/fullA-d4-exact` @ `02583bc` (worktree
  `gravity-fullA-d4`). Confirmed via `git status --branch` (`ahead 144` of origin) that this
  local HEAD, not origin, is authoritative.
- Continuation 12 branch: `diag/fullA-d4-exact-common-marginals` @ `6ee42b6` (worktree
  `gravity-fullA-d4-c12-common-marginals`).
- **Git archaeology finding**: `git merge-base` of these two branches is `02583bc` itself --
  i.e. the C12 branch already sits exactly on top of the "latest production commit" with ZERO
  divergence. The brief's premise ("production branch subsequently acquired additional changes")
  does not hold in the *committed* history; the only real divergence is (a) substantial
  UNCOMMITTED Continuation-11 work (buffered-gradient wiring + ~40 diagnostic scripts) sitting in
  the `gravity-fullA-d4` worktree, never committed, and (b) a separate unmerged branch
  `diag/fullA-d20-fast-infeasibility` with advanced cutting-plane/LP screening infrastructure
  beyond the basic pairwise+witness screens this continuation actually needs. Neither was folded
  in -- documented explicitly rather than silently dropped or silently included.
- New integration branch: **`integration/fullA-d20-common-marginals`**, worktree
  `gravity-fullA-d20-cm-integration`, created via `git worktree add -b ... 02583bc` then a
  (trivial, fast-forward) merge of `diag/fullA-d4-exact-common-marginals`.
- All work this continuation lives as additive commits on top of that merge. Full commit list at
  the end of this document (Section 11).

## 2. Reproduction gate (Section 1 of the brief)

Reproduced before building anything new, all within tolerance of the C12 reports:

| Test | C12 reported | Reproduced | Verdict |
|---|---|---|---|
| D4 dense CM reference (Delta_dual, L=10/20/50) | 0.0033/0.0051/0.0115 | 0.0033/0.0051/0.0115 | MATCH |
| Interval/cumulative equivalence | PASS all L | PASS all L | MATCH |
| Cumulative + Architecture C combined bundle | 1.85x-4.06x speedup, ~1e-16 diff | same | MATCH |
| D20/W=80000 fixed-point inner solve, L=10 | 9.23x speedup | 8.97x | MATCH (noise) |
| D20/W=80000 fixed-point inner solve, L=50 | 17.05x speedup | 15.38x | MATCH (noise) |

No material disagreement -- proceeded per the brief's gate.

## 3. Preserving the correct interpretation of the D4 C12 results (Section 2 of the brief)

Per instruction, the C12 D4 kappa values were NOT treated as final bounds -- the reliable
takeaway carried forward was only "the CM restriction has material, ~9-12% bite," with the
specific L-dependence flagged as unresolved (non-nested grids, single start). Section 6 below
reports what changed this continuation.

## 4. Production inner bundle (Sections 3A/5 of the brief)

`cm_production_bundle.jl` composes, additively:
- **Architecture B moment construction** (`cm_hessian_architectures.jl`, already existed):
  caches the `G_tmp` scratch buffer across calls and fills CM columns directly from bin indices
  (`fill_cm_columns_from_bins!`) instead of copying from a persistent dense `W x ncm` matrix.
  Reconfirmed byte-identical to Architecture A at D20/W=80000 this continuation
  (`c13_diag_archB_slowdown.jl`: `max|GA-GB|=0.0`).
- **Architecture C structured Hessian** (already existed): raw weighted bin-contingency tables,
  2D-prefix-summed for the cumulative basis.
- **`archC_base_state`**: a new Architecture-C-accelerated drop-in for `solve_base_state`, reusing
  `obj.arg1` exactly as every other `BaseDualState` call site in this codebase already does (same
  shared FG callback across architectures -- only the Hessian callback differs) rather than an
  extra recompute.
- **`cm_production_gradient`/`cm_production_value`**: one-call entry points tying `archC_base_state`
  to the Section 5's CM-aware Lfix gradient (below), ready for a KNITRO outer callback.

Validated (`c13_validate_production_bundle.jl`): dense-vs-production `Delta_dual` agreement
1e-16-1e-17 at L=10/20/50; production gradient vs. the dense-inner-solve CM-aware gradient
matches to ~1e-15 (cosine exactly 1.0); nested-grid `probs=` plug in cleanly; a structurally
infeasible point is correctly rejected (`nStatus=-300`).

**One real bug found and fixed while wiring**: `common_marginals_interval.jl` and
`cm_hessian_architectures.jl` both define `compute_bin_indices(U,z)` with overlapping-but-distinct
signatures (`z::Vector{Float64}` vs `z::AbstractVector{Float64}`); Julia's most-specific-method
dispatch silently prefers the former (Unsigned bin dtype) regardless of include order, which is
NOT what `fill_cm_columns_from_bins!` expects (`Matrix{Int}`). Fixed with an explicit `Int.()`
conversion rather than depending on ambient dispatch resolution.

**A second apparent problem that was NOT a bug**: an early D20 microbenchmark run showed
"cold" inner-solve times of 15-20s (vs. C12's reported 2.1-2.4s). Diagnosed
(`c13_diag_archB_slowdown.jl`) as pure one-time JIT compilation: C12's own script ran a DENSE
(Architecture A) solve FIRST, which happened to pre-compile all the shared KNITRO/BLAS glue
before timing Architecture C; my scripts ran Architecture C as the very first solve in a fresh
process, paying that compilation cost inline. The SECOND call in the same process (genuinely
warm w.r.t. compiled code, still a fresh KNITRO instance/context) reproduces the original
2.0-2.4s numbers exactly. No code changes were needed.

## 5. CM-aware Lfix outer gradient (Section 4 of the brief -- the load-bearing piece)

`lfix_cm_aware.jl`. The common-marginals moment block never depends on theta (A_od/gamma'_focal)
-- confirmed directly from `wrap_moments_with_cm`'s implementation, not assumed -- so its
contribution to the augmented base-dual scalar,

```
q_s(theta) = -zeta* - lambda_G*'G_s(theta) - lambda_C*'C_s
```

has a CONSTANT `lambda_C*'C_s` term across every outer coordinate probe at a fixed base dual
solve. `lfix_cm_aware.jl` computes this term exactly ONCE per base point (via the already-validated
O(D)-per-draw cumulative lookup kernel from `cm_lookup_kernels.jl`, never materializing a dense
`W x ncm` matrix), folds it into `LFixBaseCache.q0`, and then reuses `lfix_incremental.jl`/
`composite_gradient.jl` COMPLETELY UNCHANGED for every per-coordinate probe. `with_q0` is the only
new mechanism: a field-generic reconstruction of `LFixBaseCache` with `q0` replaced.

`composite_gradient_fast.jl` gained one additive kwarg, `cache=` (mirrors the existing `base=`
pattern exactly), letting a caller inject a pre-built CM-aware cache; the default path
(`cache=nothing`) is verified bit-identical to the pre-existing behavior.

**Validation** (`c13_validate_cm_aware_lfix.jl`, `c13_gamma_component_baseline_check.jl`), D4/L=10,
against optimized-value FD of the FULLY RE-SOLVED CM-augmented inner problem (not a
self-consistency check against the cache's own internals):

- A-block cosine: 0.996-0.998; norm ratio 0.90-0.99; sign agreement 13-15/15; stable across FD
  bandwidths 0.0025-0.02.
- The gamma'_focal component initially looked ~2x off from FD at h=0.01-0.03. Traced to a
  PRE-EXISTING FD-bandwidth-mismatch artifact of `gamma_component_analytic` (present identically
  in the plain non-CM baseline: ratio 0.29->0.997 as h shrinks 0.02->0.001) -- not a bug. The
  CM-augmented gamma component converges to FD the same way (0.12->0.99 over the same h sweep).
  Per the brief's own instruction, the full-vector cosine (which the gamma component can
  dominate) was NOT used as the primary validation metric -- the A-block-only cosine was.

## 6. Genuinely nested quantile grids (Section 6 of the brief)

`nested_quantile_grids.jl`. The C12 `k/L` grids gave L=10 subset L=20 (both multiples of 0.05)
but L=20 NOT subset L=50 (0.05 is not a multiple of 0.02) -- flagged in the C12 handoff as a
likely contributor to its observed non-monotonic kappa(L). Built via a single largest-gap-bisection
sequence (points only ever added, never reflowed), guaranteeing Q10 subset Q20 subset Q50 by
construction for the ACTUAL requested sizes, not merely by coincidence.

Caught and fixed one real bug in the first draft: slicing the boundary sentinels off an unsorted
post-push array leaked the literal probability `1.0` into a snapshot as a "cutpoint" (degenerate --
trivially true for every draw). Fixed by sorting a fresh copy before every snapshot.

`precalc_common_marginals_cdf`/`build_cm_augmented_obj` got one additive `probs=` kwarg (default
`nothing` = the pre-existing `k/L` behavior, reverified bit-identical) so these grids plug directly
into the existing machinery.

Exact levels (10/20/50 cutpoints):

```
Q10: [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.625, 0.75, 0.875]
Q20: [0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125,
      0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]
Q50: [0.015625, 0.03125, ..., 0.96875]  (49 points, see nested_quantile_grids.jl output)
```

`ncm=(D-1)*L` moment columns per grid, i.e. at D=20: L=10->190, L=20->380, L=50->950 (matches the
existing `ncm=(D-1)L` convention; no off-by-one found).

## 7. D4 outer KNITRO driver + multistart RESULT (Section 5 outer-wiring + Section 7)

`cm_outer_driver.jl`: a constrained-upper-bound KNITRO outer loop (minimize `gamma'_focal`
subject to `Delta_dual(w)<=delta`, over pivot-eliminated reduced coordinates) wired to the
production bundle. Sign convention re-derived and confirmed to match
`c10_d20_production_driver.jl`'s own `run_polish_checkpointed`: minimizing `w[1]` maximizes
kappa (kappa=1-gp^(sigma/(sigma-1)) is decreasing in gp for sigma>1), i.e. an upper bound.

`c13_d4_cm_multistart.jl`: multistart (calibration / existing unrestricted candidate / prior
grid's best) across the nested Q10/Q20/Q50 grids, delta=1.

**RESULT -- the headline D4 scientific finding this continuation produced**:

| L | kappa | Delta_dual | KNITRO status | best start |
|---|---|---|---|---|
| 10 | 0.16166127 | 0.99985 | -102 (genuine KKT) | calibration |
| 20 | 0.16098456 | 1.00000 | -101 (genuine KKT) | prior grid's best |
| 50 | 0.15859104 | 0.99999 | -102 (genuine KKT) | prior grid's best |

**Correctly monotonically DECREASING in L for the first time in this investigation** -- resolves
the open non-monotonicity puzzle C12 left unresolved. Cross-grid incumbent injection confirms
consistency: L=20's and L=50's best points are BOTH feasible under every coarser nested grid,
with the coarser grid's own best kappa >= the denser one's, exactly as nested-restriction theory
requires. A fresh cold dense-reference re-solve of the L=50 final candidate confirms `nStatus=0`,
`Delta_dual=0.999995<=1`, `gravity=-6.1e-18` (~0).

Headline: CM-restricted upper `kappa=0.15859` vs. unrestricted upper `0.17246` -- an **8.05%
reduction**, consistent with (slightly below) C12's 9-12% finding on the old, non-nested,
single-start grids.

Full run log: `docs/fullA_c13_d4_multistart_run.log`. Serialized: `results/fullA_d4/c13_cm_multistart_best.jls`.

## 8. Interval-native Architecture C (user addendum, elevated to pre-production priority)

Mid-session the user asked to raise interval-native Architecture C from optional to a
pre-production decision gate, with an explicit recommendation required before the long D20 run.

`cm_hessian_architecture_interval.jl`: a genuinely interval-native structured Hessian -- raw
weighted bin-contingency tables used DIRECTLY (no 2D prefix-summing at all; a new lean
`CMBinHessCtxInterval` struct carries no `CT`/`CScum` fields, so there is nothing to accidentally
fall back on) for both the common-common block and the economic-common cross block. Pairs with
the already-validated interval lookup FG kernels (`cm_lookup_kernels.jl`, untouched).

**D4 validation** (`c13_validate_interval_native_archC.jl`) against the TRUSTED DENSE
INTERVAL-BASIS reference (not the cumulative Hessian): L in {10,20,50}, both contrasts,
calibration and the unrestricted upper candidate. Worst disagreement across all 16 combinations:
**8.36e-16** (machine precision). Conditioning: interval-native 1.2x-11.3x BETTER than cumulative,
growing with L (matches C12's own reported 11-13x at L=50).

**D4 full comparison** (`c13_cumulative_vs_interval_native_comparison.jl d4`): Delta_dual agrees
to <2e-15 at every feasible point; both architectures consistently and correctly flag a
deliberately difficult 15%-perturbed point as infeasible (`nStatus=-300`, agreeing with each
other to ~1e-9); conditioning 3x-13x better for interval, growing with L; warm-time ratio
**0.983x** (a wash, not a penalty).

**D20/L=50/W=80000 full comparison (the decisive test)**: Delta_dual still agrees to machine
precision (2.4e-17, 2.5e-16); both converge cleanly at both points tested; warm-time ratio
**0.952x** (still a wash). **CONDITIONING REVERSES**: interval-native's Hessian condition number
is **~39x and ~38.5x WORSE** than cumulative's at this scale (7.53e7 vs 1.93e6, and 9.69e7 vs
2.51e6) -- the OPPOSITE of the D4 finding, consistent across both D20 points tested (not a
one-point fluke).

**Explicit recommendation (per the user's own stated adoption criteria: exact-equivalence AND
more-stable AND not >10-20% slower)**: **RETAIN cumulative Architecture C for the D20 production
run.** Interval-native passes exact-equivalence and the timing bar, but FAILS the "more stable"
criterion outright at the scale that matters -- the D4 conditioning win does not generalize, and
in fact inverts. Root cause not chased further this session (candidate hypothesis: the raw
per-bin cell counts become comparatively sparse/uneven at D20's much larger cross-origin table,
nO=19 vs D4's nO=3, at matched L/W; the cumulative `<=l` aggregation may inherently produce a
better-behaved quadratic form at this scale -- not verified). No code changes to
`cm_production_bundle.jl`/`cm_outer_driver.jl` were needed since they already default to
cumulative. Interval-native code is kept, fully validated, and documented as a follow-up
candidate for a future continuation that wants to trace the D20 conditioning regression.

This detour did not block the CM-aware Lfix gradient (Section 5, already complete beforehand) or
the real D20 outer-loop smoke test (Section 9 below), per the user's explicit instruction.

## 9. Real D20/W=80000/delta=1 CM upper bound (Section 9 of the brief)

`c13_d20_cm_upper_continuation.jl`: nested Q10->Q20->Q50 continuation, each stage starting from
the previous stage's best exact-feasible incumbent (with an explicit feasibility check before
handing a warm start to KNITRO -- falls back to calibration if the prior stage's point is not
feasible under the new, more restrictive grid). Checkpoints every stage's best incumbent to disk
via `serialize`.

**Smoke test** (5 min/stage budget, `docs/fullA_c13_d20_outer_smoke_test.log`): no crashes,
checkpoints save correctly, kappa monotonically decreasing across L even this far from convergence
(0.0552->0.0531->0.0502), confirming the wiring is correct before committing to a long run.

**Real run**: launched with a 40-minute/stage budget (2 hours total) in the background.

<!-- RESULT-PLACEHOLDER: filled in once the background run (PID recorded at launch) completes. -->

## 10. Final independent verification

`c13_d20_final_verification.jl`: reads a stage checkpoint, and in the SAME (or ideally a fresh)
process independently reconstructs the full DxD log-A matrix, computes gravity directly, runs a
fresh COLD dense-reference CM solve, verifies primal/dual divergence and the gap, verifies core
AND CM-block KKT residuals, checks parameter bounds, runs gravity-tangent secant directional
checks, and re-evaluates the candidate under every coarser nested grid as a consistency check.
Classifies the result conservatively (verified local candidate / bandwidth-KKT exact-feasible
candidate / best exact-feasible stalled point / unresolved) -- never claims global optimality.

<!-- RESULT-PLACEHOLDER: run against the real D20 result once Section 9 completes. -->

## 11. Commits on `integration/fullA-d20-common-marginals` (base: `02583bc`)

Run `git log --oneline 02583bc..HEAD` on this branch for the authoritative list. As of this
writing: branch setup (ff-merge of C12), CM-aware Lfix gradient + validation, nested grids,
production bundle (ArchB+ArchC combined) + validation, CM outer driver, D4 multistart result,
interval-native Architecture C + D4/D20 validation + comparison + recommendation, D20 smoke test.
None of this has been merged into `diag/fullA-d4-exact` -- awaiting user review per the brief.

## 12. What's exact vs. approximate / what's not done

Everything implemented is EXACT (no smoothing, no hard-max approximation, no change to the CC
divergence normalization) -- same discipline as C12. Explicitly NOT attempted or deferred:

- Interval-native Architecture C is implemented and validated but NOT adopted for production
  (Section 8) -- kept as a documented, tested follow-up candidate.
- Schur-complement block elimination, matrix-free HVPs, smooth-basis approximations,
  tail-weighted CM moments, adaptive quantile activation: all deprioritized per the brief's own
  Section 11 list; not attempted (C12 already found most of these unhelpful or not
  high-value at D4).
- The optional simulation-design diagnostic (balanced marginal draw construction, brief Section
  12): not attempted.
- Checkpoint/resume across PROCESS restarts (not just stage transitions) was not exercised --
  the real run's checkpoints are stage-granular, not iteration-granular like
  `c10_d20_production_driver.jl`'s `D20Checkpoint`. If the real run needs to be resumed after an
  interruption, the next continuation should extend `c13_d20_cm_upper_continuation.jl` toward
  that driver's finer-grained checkpoint cadence.
