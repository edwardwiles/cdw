# Full-A D=4: W-stability

Phase 3 deliverable. **Status: partial** — cost-scaling data is complete (Phase 1C); genuine
candidate-stability re-optimization at W=20,000/80,000 is not.

## What is done

1. **Cost scaling at D=4 across W=8000/20000/80000** (`docs/fullA_scaling_projection.md` §"W scaling
   at fixed D=4", data in `results/fullA_d4/cd8adc2/profile_D_W_scaling/profile_D_W_scaling.csv`):
   exact-evaluation cost grows close to linearly in W (empirical exponent ≈0.94); full-gradient cost
   grows sub-linearly in the fitted exponent (≈0.64, noisy, only 3 points) — reassuring in that W does
   not appear to compound the D-driven cost blowup, but not yet confirmed with enough repetitions or
   D>4 cross-points to trust the exponent tightly.
2. **maxit=40 upper candidate feasibility check at W=20,000** (`full_aod_diag/d4_exact/
   phase3_w20000_recheck_maxit40.jl`, `results/fullA_d4/64a1761/phase3_w20000_recheck/`): the SAME
   theta (not re-optimized) remains exactly feasible at W=20,000 under an **independent** draw set —
   gravity stays at machine-zero (deterministic, draw-independent), and `Delta` actually decreases
   (0.999→0.876, i.e. more slack under the divergence budget, not less). kappa is unchanged (a direct
   function of `gamma_focal_prime` alone, draw-independent).

## What is explicitly NOT done (do not read the above as "W-stability confirmed")

1. **Nested/common draws.** The task specifies W=8000 should be a reproducible prefix of W=20,000 and
   W=80,000 so that "how far a W=8000 point moves at larger W" is measured against the SAME underlying
   randomness, not a fresh independent sample. This continuation's `phase3_w20000_recheck_maxit40.jl`
   uses `d_exact_setup_scaled`'s own independent draws at each W (different `seedU`/draw realization
   per W internally, not verified to nest) — a faster approximation, clearly flagged in that script,
   not the rigorous design. Implementing genuine common draws requires either confirming the existing
   RNG usage already nests sequentially (untested) or explicitly constructing a shared oversized draw
   pool and slicing it — neither attempted this continuation.
2. **Re-optimization/continuation at larger W.** The task's explicit fallback — "if a W=8000 point is
   not feasible at larger W, do not call that a failure of the method; continue/re-optimize from it at
   the larger W" — was not exercised because the one point tested stayed feasible; whether OTHER
   candidates (maxit=15, the three poll-improved points, the lower stalled point) also stay feasible
   is untested.
3. **W=80,000 candidate recheck.** Only the cost-scaling benchmark ran at W=80,000 (calibration point,
   Phase 1C); no actual candidate (upper or lower) was re-evaluated there.
4. **h-grid re-selection at each W.** The task specifies choosing h per-W by target switching mass, not
   holding it mechanically fixed. Not attempted — the one W=20,000 check above did not need an FD
   gradient at all (just an exact value/feasibility recheck).
5. **The lower candidate, and the three poll-improved upper points** (`results/fullA_d4/1b2a3a0/
   phaseA_upper_revalidation/step6_poll.csv`), were not re-checked at any W beyond 8,000.
6. **The sequential solution** located this continuation (Phase 5) was not cross-checked at other W.

## Recommended next step

Given the one data point available is reassuring (feasibility improves, not degrades, at W=20,000),
the highest-value next step is breadth (check the lower candidate and the poll-improved points at
W=20,000, still cheap per Phase 1C's cost data — an exact recheck at W=20,000 costs ~84ms, trivial),
before depth (W=80,000, nested draws, or re-optimization) — consistent with the task's own "start with
the cheaper exact rechecks... only launch W=80000 optimizations after those results... are known"
sequencing.
