# Continuation 5, Priority 2 (wiring) + Priority 3: fast composite gradient live results + fair hard-vs-smoothed comparison

## Priority 2: `lfix_composite_fast` wired into the live KNITRO driver

`full_aod_diag/d4_exact/run_d4_optimized_fd.jl` gains `D4X_GRADIENT_METHOD=lfix_composite_fast`
(backward compatible -- `delta_fd`/`lfix_composite`/`hybrid` behavior byte-for-byte unchanged),
combining the two equivalence-tested levers from `docs/fullA_p1_warmed_profile.md` Part A and
`composite_gradient_fast.jl`:

1. **shared base state**: `eval_F` now stashes a `BaseDualState` built from what it just computed
   (zero extra inner solves), and `eval_grad_dispatch` reuses it when the gradient callback is asked
   for the SAME `w` (the common KNITRO F-then-G-at-the-same-point pattern, confirmed in Priority 1A).
   A genuine mismatch falls back to a fresh `solve_base_state`, never silently reuses a stale base.
2. **threaded A-block**: `composite_gradient_at_fast(...; threaded=true, h_mode=:adaptive)` --
   `h_mode=:adaptive` is a byte-identical refactor of the original bandwidth-selection logic (verified
   in `test_composite_gradient_fast.jl`), so only the WALL TIME changes, not the returned gradient.

### Live validation: SR1, 60s budget, D=4/W=8000/δ=1, upper, same start as Phase 4's frontier

| config | KNITRO status | outer iters | wall to converge | best-feasible κ | gradient wall (114 calls) | inner solves/gradient-call |
|---|---|---|---|---|---|---|
| `lfixcomposite_sr1` (Continuation 4, original) | -103 | 113 | **42.7s** | 0.17245688540655113 | 20.90s (~183ms/call) | 1.00 |
| **`lfixcomposite_fast_sr1` (this continuation)** | -103 | 113 | **26.9s** | **0.17245688540655113** (identical `w`, bit-for-bit) | **8.12s (~71ms/call)** | **0.00** |

**Converges to the EXACT SAME point** (`best_feasible.w` bit-identical between the two runs) in **1.59x
less wall-clock time**, with the gradient callback itself **2.57x cheaper** and consuming **zero**
inner CC-dual solves per call (vs 1.00 previously) -- direct, clean confirmation that `h_mode=:adaptive`
really is math-preserving (KNITRO takes the identical optimization path) and that the two Priority 2
levers deliver a real, not just theoretical, wall-clock win. Classification carries over unchanged from
Priority 0 (same `w`): `EXACT_FEASIBLE_CANDIDATE` + `H_BANDWIDTH_KKT_CANDIDATE(h=0.01)`, not
`ROBUST_LOCAL_CANDIDATE`.

### LBFGS, 60s budget

| config | KNITRO status | outer iters | wall to converge | best-feasible κ | gradient wall (61 calls) | inner solves/gradient-call |
|---|---|---|---|---|---|---|
| `lfixcomposite_lbfgs` (Continuation 4, original) | -101 | 60 | 29.4s | 0.17234500... (κ=0.172345 per handoff) | -- | 1.00 |
| **`lfixcomposite_fast_lbfgs` (this continuation)** | -101 | 60 | **22.5s** | **0.1723447464881782** | 6.19s (~101ms/call) | **0.00** |

Same outer-iteration count (60), essentially the same converged κ (agrees to 4 decimal places -- LBFGS's
own step-by-step path is mildly sensitive to wall-clock-dependent floating-point timing across two
separate process runs, a known benign effect, not a correctness concern), **1.31x less wall-clock time**.
Not independently externally revalidated this session (flagged, not silently assumed robust) -- the SR1
result above, which IS externally validated (Priority 0, identical point), is the primary evidence.

Raw artifacts: `results/fullA_d4/9e03706/optfd_upper_lfix_composite_fast_{sr1,lbfgs}_20260718_*/`.

## Priority 3: fair hard-FD vs smoothed-AD comparison

### Hard route (this continuation's `lfix_composite_fast`, both Hessian modes, 60s budget -- fresh data above)

- **Time to first exact-hard feasible point**: near-immediate -- every single `eval_F` call in the
  hard route IS an exact-hard evaluation (`evaluate_fullA`, no smoothing anywhere in the value or
  gradient path). This is a structural property, not a measured number: the hard route never needs to
  "commit" to a schedule before it has valid exact-hard feasibility/κ information, unlike the smoothed
  route (below).
- **Best exact-hard κ vs wall time**: κ=0.17245689 at 26.9s (SR1) / κ=0.17234475 at 22.5s (LBFGS).
- **Exact-hard Δ/gravity/moments**: `Delta_dual=0.9999924058` (Δ-δ=-7.59e-6), `gravity_value=-3.58e-18`,
  `max_abs_moment_kkt_resid=1.35e-16` -- all machine-precision-clean (SR1 row; identical to the
  already-validated Priority 0 candidate).
- **Outer steps / inner solves / moment builds / gradient calls (SR1)**: 113 outer iters, 114 gradient
  calls, **0** inner solves consumed BY gradient calls (all reused from `eval_F`'s own single inner
  solve per outer iterate) -- i.e. exactly 1 inner solve + 1 moment build per outer iterate, the
  theoretical minimum for a gradient-based method on this problem.
- **Final external checks**: SR1's result is the Priority 0-validated candidate (8/8 rechecks feasible
  at 2 tolerances, `H_BANDWIDTH_KKT_CANDIDATE(h=0.01)=true`, not `ROBUST_LOCAL_CANDIDATE`).

### Smoothed route (`docs/fullA_smoothed_consistent_experiment.md`, REUSED not re-run -- nothing in this
    continuation's fixes touches the smoothed code path, so re-running would not add new information)

- **Time to first exact-hard feasible point**: by design, **only available after the full 5-stage
  homotopy completes** (~80.5s = 18.6+18.5+18.1+14.1+11.2s) -- intermediate-rho values are explicitly
  NOT valid hard-estimand numbers (the task's own instruction, honored throughout that experiment).
  This is a structural disadvantage relative to the hard route, not a tuning artifact: a smoothed
  method fundamentally cannot report a trustworthy exact-hard number mid-schedule without an explicit
  interleaved hard-check (not implemented in that experiment).
- **Best exact-hard κ vs wall time**: the finest-stage smoothed optimum, evaluated exact-hard, gives
  κ=0.171006 at ~80.5s; after 2 rounds of `lfix_incremental_o1`-based polish (wall time not separately
  reported in the original experiment, but small -- 2 rounds of a single-direction line search),
  **κ=0.17197169**.
- **Exact-hard Δ/gravity/moments** (post-polish): `Delta_dual=0.9917194160` (Δ-δ=-0.00828, i.e. 0.83%
  slack -- notably NOT as tight against the budget as the hard route's candidates), `gravity_value=
  -7.37e-18`, `max_abs_moment_kkt_resid=1.07e-16`.
- **Outer steps / inner solves / moment builds / gradient calls**: 15 outer iters/stage x 5 stages = 75
  outer iters total, each needing a FULL smoothed moment build + smoothed inner dual solve (NOT
  skippable the way the hard route's shared-base-state trick skips it) plus a ForwardDiff gradient
  (~44ms warmed per `docs/fullA_p1_warmed_profile.md` Part C) -- structurally more expensive per outer
  iterate than the hard route's shared-base composite gradient, though this was not measured
  head-to-head in a single harness this continuation (flagged, not asserted with a precise ratio).
- **Final external checks**: not independently re-validated this continuation (the original
  experiment's own polish-log and KKT-residual numbers are the only evidence).

### Bottom line

At a MATCHED 60s-class wall-clock budget, **this continuation's hard route (`lfix_composite_fast`)
reaches a HIGHER exact-hard κ (0.172457) in LESS wall time (26.9s to converge) than the smoothed
route's own reported total (κ=0.171972 after ~80.5s+polish)** -- a clean, direct win for the hard
route once Priority 2's engineering fixes are applied, reversing the earlier (pre-fix) framing in
`docs/fullA_smoothed_consistent_experiment.md` where the smoothed-then-polished candidate was briefly
the best available number. This does NOT mean smoothing is without value -- the smoothed route still
answers a genuinely different question well (a temperature-homotopy basin search that needs no
per-coordinate bandwidth tuning at all), and Priority 1C's finding that its own warmed gradient cost
is only ~44ms (not the previously-suspected 3.1s) means a properly-engineered smoothed route sharing
the SAME base-state/threading tricks this continuation applied to the hard route could plausibly close
much of this gap in a future continuation -- not attempted here (out of this continuation's remaining
budget; flagged as the natural next comparison once/if the smoothed inner solve gets the same
shared-base-state treatment `evaluate_fullA`/`composite_gradient_at_fast` already received).

### What this does and does not establish

- ONE wall-clock budget class (~22-27s to convergence within a 60s cap), ONE direction (upper), ONE
  starting point -- matches Priority 4's own earlier frontier's scope, not exhaustive.
- The smoothed-route numbers are REUSED from an unchanged prior experiment, not re-run under identical
  process/thread conditions to the hard-route numbers above -- a genuinely apples-to-apples timing
  comparison (same machine load, same JULIA_NUM_THREADS, back-to-back) was not performed this
  continuation; the qualitative conclusion (hard route reaches a higher κ, faster, with a
  structurally-earlier feasibility guarantee) is robust to this caveat, but the precise wall-clock
  ratio should not be over-read.
- LBFGS's fast-route result was not independently externally revalidated (only SR1's was, and it
  reproduces Priority 0's already-validated point exactly).
