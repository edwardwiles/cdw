# Continuation 5, Priority 4: lower direction completion + gamma profile

## 1. Lower direction: COMPLETE, major improvement over the old stalled candidate

`run_d4_optimized_fd.jl` direction=lower, `D4X_GRADIENT_METHOD=lfix_composite_fast`, hessopt=sr1,
`D4X_MAXTIME_REAL=300` (converged well inside budget, at 22.6s):

| | old `lower_stalled` (maxit=15) | new `lower_lfixcomposite_fast_sr1_300s` |
|---|---|---|
| KNITRO status | (ran out of iterations) | -102 (genuine convergence-class stop) |
| outer iters | 15 (iteration-capped) | 51 |
| wall time | -- | 22.6s |
| κ | 0.0106911632 | **0.0054287999** |
| Δ − δ | **−0.101** (far inside, not binding) | **+8.5e-7** (essentially exactly on the boundary) |
| gravity/KKT residual | clean | clean (6.5e-15 / 1.3e-15) |

The old point was honestly labeled `BEST_FEASIBLE_STALLED` — it simply ran out of iteration budget
long before reaching the constraint boundary. The new point genuinely reaches the divergence-budget
boundary (Δ−δ≈8.5e-7, not merely feasible-with-slack) and gives a materially smaller (tighter, more
informative) κ for the lower direction. Canonicalized in `candidate_registry.jl` as
`lower_lfixcomposite_fast_sr1_300s`.

### External revalidation (`phaseA_lower_lfixcomposite_fast_revalidation.jl`)

- **EXACT_FEASIBLE_CANDIDATE: true** (4/4 cold/warm rechecks at 2 tolerances pass).
- **H_BANDWIDTH_KKT_CANDIDATE at h=0.01: false** — one of the 32 h=0.01 directional probes is
  non-finite (a winner-boundary crossing right at this near-degenerate point, where γ' is very close
  to its trivial upper limit of 1), making the `eta` estimate degenerate (`eta=-0.0`,
  `kkt_resid=1.0`). This is a DIFFERENT failure mode from a genuine stationarity problem: at h=0.0025
  and h=0.001 (finer, avoiding the problematic probe), `eta` is small-but-nonzero (0.0022, 0.0040) and
  the residual is 0.6%-2.5% — a much more informative reading than the degenerate h=0.01 result.
- **ROBUST_LOCAL_CANDIDATE: false** — the poll found 3 exact-feasible "improvements," all in the SAME
  random direction (dir_idx=25) at increasing radii, but each is a γ' change of order **1e-7 to 5e-7**
  (e.g. radius=0.02 gives γ': 0.99673917441→0.99673968337) — economically negligible, likely reflecting
  that κ is only weakly sensitive to γ' near γ'≈1 (κ=1−γ'^(σ/(σ-1)) is flat there), not a real
  large-scale improvement opportunity. Classification carried forward honestly rather than silently
  upgraded to ROBUST_LOCAL.
- **Bottom line**: same qualitative tier as the upper candidate (`EXACT_FEASIBLE_CANDIDATE` +
  bandwidth-dependent KKT check, not fully ROBUST_LOCAL), but now a MUCH more informative, boundary-
  binding point than the prior stalled candidate — this alone is a substantial completion of the task's
  "finish the lower direction" requirement.

## 2. Gamma profile: partial success, with a genuine new bug discovered and documented

`gamma_profile.jl` implements `profile_delta_at_gamma(g, ...)`: an unconstrained KNITRO NLP over the
15-dim A-block only (γ'=g held fixed, not a decision variable), objective = `Delta_dual` itself,
gradient = `composite_gradient_at_fast`'s own A-block sub-vector with shared base-state + threading
(same Priority 2 levers). Walked outward from the incumbent (`g=0.8926359584642946`) in both
directions, continuation-warm-started.

### A genuine, reproducible bug found: `build_lfix_base_cache`'s self-validation fails away from previously-tested (g, A) neighborhoods

At `g=0.9105...` (one step up from the incumbent), using the INCUMBENT's own (previously-validated)
`z_free`, `build_lfix_base_cache`'s internal algebraic self-check
(`max|q0_true - q0_cache| < 1e-8`) failed with **max error 3.03** — not a numerical-tolerance issue, a
real, large discrepancy. Reproduced deterministically (not a race condition or fluke) in a standalone
script. The `try/catch` fallback added to `gamma_profile.jl`'s gradient callback (needed as a genuine
"a failed inner solve/cache build must not crash the whole outer loop" safeguard, matching this
investigation's established non-fatal-fallback discipline) caught it and returned a zero gradient
rather than crashing KNITRO — but this means **most of the "up" direction and several "down" direction
grid points did NOT get genuinely optimized** (KNITRO accepted the unchanged starting point trivially,
`n_eval=1`, since a zero gradient gives it nothing to improve on).

**Diagnostic work done this session** (not a full root-cause, an honest partial isolation):
1. The winner computation (`price_and_pTsigma_cell` + `argmin`) matches the TRUE moment matrix's
   implied winner **exactly**, 0/32000 mismatches — ruled out.
2. The counterfactual piece (`cf_contrib0`, the ONE piece that structurally depends on γ') matches
   **exactly** (0.0 diff) — ruled out, and confirms γ' itself is not directly implicated (the factual
   side's formulas, `price_and_pTsigma_cell`/`aod_pow_cell`, never read `θ_full[3+D]`=γ' at all).
3. The FACTUAL piece (`contrib0`, built from `price0`/`pTσ0`/`CONST_d`/`winner0`) is where the 3.03
   discrepancy lives, despite the winner matching exactly at every draw. Root cause NOT isolated within
   this session's remaining budget — a plausible next step is checking `price_and_pTsigma_cell`'s
   `pTσ0` value (not just its use in winner-finding) against `hFunction!`'s own internal
   `pricesTempσ[winner]` directly (this session confirmed the WINNER matches but did not fully verify
   the WINNER'S OWN VALUE matches at the specific draws where the discrepancy concentrates).
4. This was NOT triggered by any of Continuation 4/5's own validated live KNITRO runs (Priority
   0/2/3's results never hit this — all their trajectories stayed within a validated neighborhood of
   jointly-(γ,A)-optimized points). This is specifically a hazard for code (like this gamma-profile
   driver) that evaluates the composite gradient at γ/A COMBINATIONS decoupled from any prior joint
   optimization trajectory — **flagged as a priority item for whoever continues**, since Priority 4's
   fuller gamma-profile mandate and Priority 6's "upper polish across a wider search" both need this
   fixed to be trustworthy over a WIDE parameter range.

### What the RELIABLE points show

Excluding the `n_eval=1` (trivial, unoptimized) and the one partially-corrupted point (`g=0.9105`,
`n_eval=217` but SOME fraction of its gradient calls hit the zero-gradient fallback mid-run, per the
log — not trusted), three points ARE genuinely KNITRO-optimized (`n_eval` in the hundreds, no
self-validation failure during their own run):

| g | knitro_status | n_eval | profile_Delta(g) = min_A Delta(g,A) |
|---|---|---|---|
| 0.8793462377640321 | -101 | 170 | 1.9364 (infeasible: Δ>δ) |
| 0.8859910981141633 | -102 | 670 | 1.3817 (infeasible: Δ>δ) |
| 0.8926359584642946 (incumbent) | -101 | 67 | 0.99999 (essentially exactly Δ=δ) |

**This is a genuine, useful confirmation of the incumbent's validity**: `profile_Delta(g)` is strictly
DEcreasing as g increases toward the incumbent's own value (1.94 → 1.38 → 1.00), confirming the
incumbent sits very close to the TRUE `profile_Delta(g)=delta` crossing for the upper direction — moving
g even slightly below 0.8926 makes the constraint genuinely infeasible EVEN AFTER re-optimizing the
entire A-block, not just along the specific direction KNITRO's own outer-loop trajectory happened to
explore. This is independent, structural evidence supporting `upper_lfixcomposite_sr1_60s` as a
credible near-boundary point, beyond the poll/multi-h checks Priority 0 already ran.

**Not established**: the WIDER profile (g up to 1.0, g down to the theoretical lower bound 0.8528) due
to the bug above. A genuinely reliable coarse-to-fine gamma profile across the FULL theoretical
interval remains open work, gated on root-causing/fixing the `build_lfix_base_cache` issue found here.

## 3. What was not attempted

- Multistart from the smoothed-homotopy candidate, historical poll improvements, or fixed-A/calibration
  starts (Priority 4 item 2) — not reached this session given the gamma-profile bug investigation's
  time cost.
- Refining every `profile_Delta(g)=delta` crossing (only the one near the existing incumbent was
  characterized, and only qualitatively — no root-finding/bisection on g was run).
- A deeper root-cause fix for the `build_lfix_base_cache` bug — diagnosed but not resolved.
