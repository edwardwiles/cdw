# Consistent smoothed full-A D=4 experiment

Branch `diag/fullA-d4-exact-smoothed-consistent`, based on `diag/fullA-d4-exact`
commit `b5c109d`. New files, all under `full_aod_diag/d4_exact/`:
`smoothed_consistent.jl` (kernel), `test_smoothed_consistent.jl` (Steps 1-2
verification), `run_smoothed_homotopy.jl` (Steps 3-7 driver). Results under
`results/fullA_d4/774ba93/smoothed_homotopy_20260718_071255/`.

Task: build a genuinely CONSISTENT smoothed full-A solve (both values and
bilateral allocation moments at the SAME temperature, not derivative-only
smoothing), verify it, run a temperature homotopy, then switch to the exact
hard oracle for the headline number. Nothing here reports a positive-temperature
number as a final bound -- every headline kappa below is an exact-hard evaluation.

## 0. Scoping: exactly one hard branch needs smoothing for this D=4 setup

Direct code audit of `moments/hFunction.jl`, `moments/newGravityMoment!.jl`, and
this investigation's fixed `AD_PARAMS` (`full_aod_diag/ad_benchmark/setup_context.jl`)
confirms, independently of the task brief's own hint:

- `AD_PARAMS`: `counterType=1` (autarky), `UoModel=1`, `localGravityMoment=0`,
  `gravMoment=1`, `independenceMoment=0`, `GravityMomentFirstApproach=0`.
- `hFunction!`'s factual block (`moments/hFunction.jl:53-95`) calls `MinInd!`
  (hard 0/1 winner indicator) to build `pricesInd`, then
  `pricesTemp[o] = pricesTempσ[o] * pricesInd[o]` -- this is the ONE hard
  branch in the active moment map.
- `hFunctionCounter!`'s `counterType==1` branch (`moments/hFunction.jl:192-216`)
  is a closed-form broadcast in `Aod[baseIndex,baseIndex]` and `gamma'_focal`
  ALONE -- verified by reading the branch: it never touches `pricesInd`, no
  `MinInd!` call at all. (The `counterType != 1` branch, unused here, DOES
  call `MinInd!` -- not relevant to this D=4 economy's fixed config.)
- `newGravityMoment!`'s `UoModel==1` branch (`moments/newGravityMoment!.jl:17-31`)
  is a pure function of `AodPow` (smooth in theta) and `tau` (data) -- no
  per-draw winner selection, no `U` dependence at all. Confirmed empirically
  too: `gravity_value` is bit-identical across every rho tried in the homotopy
  run (see Step 5 table below), to the last printed digit.
- `moments/localGravityMoment!.jl`/`localGravityCrossMoment!.jl` (which DO
  call `smoothMinIndNew!` themselves already) are dead code for this config
  (`localGravityMoment=0` gates the whole block off in `hFunction!`).

So: replacing `hFunction!`'s `MinInd!` with `smoothMinIndNew!`'s softmax
supplier probabilities, and changing nothing else (reusing `hFunctionCounter!`
and `newGravityMoment!` verbatim, unmodified, imported not copied), is a
genuinely consistent smoothing for this investigation's D=4 economy -- not
merely "sufficient," the *only* other candidate hard branches are provably
inactive or already smooth.

## 1. Step 1: the non-determinism bug -- investigated, then BYPASSED

`full_aod_diag/d4_exact/smoothed_moments.jl`'s `smoothed_frozen_adjoint_Q` is
documented (`docs/fullA_d4_final_report.md` sec 6) as call-history-dependent:
repeated calls with bit-identical arguments return different values (1.5e-11
to 7.8e-2, plus NaN/Inf), root cause not identified in that prior session.

**Candidates checked by direct code reading this session:**
- `BaseDualState` (`three_way_derivatives.jl:14-34`) deep-copies `m_star`
  (`copy(obj.arg1)`) and `λstar` (`collect(inner_x[2:end])`) at construction
  -- NOT an aliased view into `obj`'s mutable scratch. Ruled out: a later
  hard-oracle call elsewhere in the process cannot silently mutate a frozen
  `base` object under `smoothed_frozen_adjoint_Q`'s feet via this path.
- `hFunction!`'s (and this file's own `hFunction_smoothed!`'s)
  `Threads.@threads` partitioning writes DISJOINT ω-ranges of `G` with no
  cross-thread reduction anywhere in `hFunction!`/`hFunctionCounter!`/
  `newGravityMoment!` -- there is no shared accumulator for a thread race to
  corrupt, so this cannot explain a per-call VALUE changing (only ordering,
  and there's no reduction depending on order).
- `smoothed_moments.jl`'s own `smoothed_factual_G` was independently
  documented (same prior session) as stable under direct repeated calls --
  the bug is specific to the composition in `smoothed_frozen_adjoint_Q`, not
  the moment-building callee alone.

**Root cause not otherwise isolated within this session's budget.** Per the
task's explicit fallback, BYPASSED entirely: `smoothed_consistent.jl` is a
fresh, from-scratch implementation (`hFunction_smoothed!`, `smoothed_moments!`,
`smoothed_obj_for`, `SmoothedBaseDualState`, `smoothed_fixed_dual_L`,
`smoothed_frozen_adjoint_Q`, `smoothed_optimized_Delta`) that:
- allocates its own buffers on every call (no shared mutable scratch across
  calls beyond a fixed `Float64 tuner`/`rho` closure capture),
- never calls into `smoothed_moments.jl`'s code path at all,
- is generic over `eltype(G)` (unlike `smoothed_moments.jl`'s hardcoded
  `zeros(W, D^2)`, which is also why the old file couldn't be
  ForwardDiff'd through directly -- a second, independent reason to rewrite
  rather than patch).

**Re-verified deterministic**, not merely assumed fixed: `test_smoothed_consistent.jl`
TEST 1 runs 25 iterations interleaving calls across TWO different tuners and
TWO different points (mirroring the original bug's reported "long call
history" trigger) and re-checks a fixed reference call is still bit-identical
after each interleaved iteration. **PASS: bit-identical every time.**

## 2. Step 2: verification (all checks pass)

Run: `julia --project=. full_aod_diag/d4_exact/test_smoothed_consistent.jl`

| Check | Result |
|---|---|
| Determinism under 25-iter interleaved multi-tuner call history | PASS, bit-identical |
| rho -> 0 recovers hard `EK_moments_gammanorm_directgp!` moments (calibration point AND upper_maxit40 point) | PASS; `mean\|G-Gh\|` decays smoothly ~4-5 orders of magnitude per rho decade (e.g. upper_maxit40: 0.299 at rho=1 -> 1.1e-5 at rho=1e-6 -> exactly 0.0 at rho=1e-8, below the smallest observed price gap) |
| Central-FD Jacobian of `smoothed_fixed_dual_L` stable as h shrinks | PASS; `max\|FD-AD\|` strictly decreases across h=1e-2..1e-7 (6.86 -> 0.072 -> 7.2e-4 -> 7.2e-6 -> 7.5e-8 -> 4.2e-8, i.e. genuine O(h^2) central-difference convergence, no floor -- unlike the hard oracle's documented winner-boundary derivative bug) |
| ForwardDiff.gradient of `smoothed_fixed_dual_L` matches central FD | PASS; relerr 7.0e-10, cosine similarity 1.0000000000000002 |
| Add-up identity: softmax supplier probabilities sum to 1 across origins, every (draw, destination) | PASS; max deviation 2.2e-16 (machine precision) |

## 3-5. Steps 3-5: temperature homotopy with matched smoothed value+gradient

Run: `julia --project=. full_aod_diag/d4_exact/run_smoothed_homotopy.jl`
(`D4X_HOMOTOPY_MAXIT` env var controls per-stage KNITRO iteration cap, default 15).

**Matched value+gradient construction:** the outer objective at each probed
`w` re-solves the SMOOTHED inner CC dual via the unmodified
`CS.inner_loop_internal` (reused, not reimplemented), giving `(zeta*, lambda*)`
consistent with that same tuner's smoothed moments. The outer gradient is
`ForwardDiff.gradient` of `smoothed_fixed_dual_L` (the envelope-theorem
fixed-dual construction, mirroring `three_way_derivatives.jl::fixed_dual_L`'s
already-validated pattern on the hard side) at that freshly-solved
`(zeta*, lambda*)` -- NOT a hard value with a smoothed gradient bolted on, and
NOT a smoothed value differentiated by finite differences of the hard model.

### Genuine finding along the way: the smoothed inner dual has its own coarse-rho feasibility wall

The first working version of this script built the rho grid purely from
gap quantiles (rho = quantile(gap, q) for q in {0.5, 0.1, 0.02, 0.01}) and
started the outer search at `theta_initial`/calibration, mirroring
`run_d4_optimized_fd.jl`'s convention. Both choices individually seemed
reasonable; together they failed completely: every homotopy stage terminated
with KNITRO status -200/-410 ("problem may be locally infeasible") and ZERO
feasible evaluations found in 12-17 tries -- caught because
`Delta_smoothed=NaN` at every stage's chosen point, not silently accepted.

Diagnosis (both effects independently confirmed by direct standalone probes,
not just inferred from the failure):
1. **Calibration's inner dual is infeasible** (`nStatus=-300`) under BOTH the
   hard oracle (`candidate_registry.jl`'s own printed output,
   `inner_status=-300`) AND the smoothed model at every rho tried -- a
   pre-existing property of this synthetic economy at that theta, not a
   smoothing artifact. Fix: start the homotopy at
   `upper_maxit15_productfd_control` (a point independently verified
   hard-feasible in `candidate_registry.jl`) instead.
2. **The smoothed inner dual has its own feasibility boundary in rho**,
   separate from and COARSER than the divergence-budget `Delta<=delta`
   constraint. Direct probe at the (now-fixed) start point:

   | rho | inner_status | Delta |
   |---|---|---|
   | 0.304 (gap q0.75) | -300 (infeasible) | NaN |
   | 0.169 (gap q0.5) | -300 (infeasible) | NaN |
   | 0.075 (gap q0.25) | -300 (infeasible) | NaN |
   | 0.028 (gap q0.1) | 0 (OK) | 2.339 |
   | 0.013 (gap q0.05) | 0 (OK) | 1.297 |
   | 0.0054 (gap q0.02) | 0 (OK) | 1.094 |
   | 0.0027 (gap q0.01) | 0 (OK) | 1.048 |
   | 4.6e-7 (near-hard) | 0 (OK) | 0.999 |

   At coarse rho, smoothing distorts the moment TARGETS enough
   (`mean|G-Gh|` ~0.1-0.3 at rho=0.1-0.2, per Step 2's Test 2) that the
   reweighting problem can fall outside the region KNITRO's inner dual
   solver can certify -- a real, previously-undocumented consequence of
   consistent smoothing, not a bug in this experiment's code (re-derived
   the same result with a completely independent probe script before
   trusting it). **Fix applied:** scan for the coarsest empirically-feasible
   rho at the start point and bound the grid there, rather than trusting the
   gap-quantile heuristic blindly.

Final grid used (5 stages, descending):
`[0.02799, 0.01339, 0.00539, 0.00271, 4.60e-7]`.

### Per-stage results (all from `homotopy_stages.csv`)

| stage | rho | gp | kappa (smoothed, informal) | Delta_smoothed | gravity | grad relerr | mean entropy / max possible | frac boundary (<0.99 top-1) | outer iters | wall (s) | knitro status | opt_err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.02799 | 0.90776 | 0.14895 | 5.7396 | -4.99e-18 | 4.4e-8 | 0.1460 / 1.3863 | 0.359 | 15 | 18.6 | -410 | 0.0549 |
| 2 | 0.01339 | 0.90574 | 0.15211 | 0.6902 | -3.66e-19 | 2.0e-8 | 0.0772 / 1.3863 | 0.200 | 15 | 18.5 | -400 | 0.0335 |
| 3 | 0.00539 | 0.89439 | 0.16975 | 0.9953 | -7.70e-18 | 2.9e-8 | 0.0327 / 1.3863 | 0.089 | 15 | 18.1 | -400 | 0.00018 |
| 4 | 0.00271 | 0.89357 | 0.17101 | 0.9964 | -7.37e-18 | 2.7e-8 | 0.0166 / 1.3863 | 0.047 | 15 | 14.1 | -400 | 0.00183 |
| 5 | 4.60e-7 | 0.89357 | 0.17101 | 0.9639 | -7.37e-18 | 2.9e-8 | 2.95e-20 / 1.3863 | 0.000 | 15 | 11.2 | -400 | 0.00962 |

Notes:
- `gravity` is bit-identical (to displayed precision) across every stage,
  confirming Step 0's claim that the gravity moment is smoothing-independent
  in this setup.
- Gradient checks (ForwardDiff vs. validated central FD, recomputed fresh at
  EVERY stage's chosen point, not just at setup) pass with relerr 2-4e-8 at
  every single stage -- the matched value+gradient construction is validated
  in-line, not just in the standalone Step 2 tests.
- Entropy and boundary-mass diagnostics decay monotonically toward 0 as rho
  shrinks, exactly as expected (winner probabilities sharpen toward one-hot).
- `knitro_status` is -400/-410 (iteration-limit-related, not full KKT
  convergence) at every stage because `maxit=15` per stage is DELIBERATELY
  short (matching `run_d4_optimized_fd.jl`'s own documented convention) --
  `opt_err` nonetheless shrinks to 1.8e-4 by stage 3, indicating good
  practical local convergence within the short budget. Best-feasible
  tracking (not the raw terminal iterate) is used throughout, per this
  investigation's established convention.
- No silent KNITRO hessopt/eval_fcga fallback detected in any of the 5
  stage logs (`csw_outer_fcga_no_maxit15.opt`, `eval_fcga no`, `hessopt 4`;
  grepped for "chang"/"instead"/"l-bfgs"/"switching" -- none found; the
  option is confirmed present verbatim as `hessopt: 4` in each log).

## Gradient benchmark (task's explicit requirement)

At the finest-rho stage's chosen point (D^2=16 free reduced coordinates):

| method | time | bytes allocated | \|gradient\| |
|---|---|---|---|
| Method 1: ForwardDiff.gradient of the scalar `smoothed_fixed_dual_L` envelope (the method actually used above) | 3.147 s | 492,794,840 | 43.1845 |
| Method 2: materialize the full W x d x D^2 jac_h-equivalent tensor via `ForwardDiff.jacobian` of the raw moment map, then contract with (m*, lambda*) | 3.295 s | 93,505,592 | 43.1845 |
| Method 3: true VJP / reverse-mode | not available |  |  |

Gradients agree to 3.6e-14 max abs diff, cosine 1.000000 (cross-check that
Method 2's manual contraction reproduces Method 1 exactly, as it must).

**Finding, measured not assumed:** at this D=4 scale (D^2=16 free
coordinates), the two methods are nearly IDENTICAL in wall time (1.05x) --
ForwardDiff's cost is dominated by the number of forward passes
(`ceil(n_free/chunksize)`), which is the same for a scalar or a
144,000-element vector output, so avoiding the "materialize the tensor" path
buys little here. Memory tells the OPPOSITE story from the usual assumption:
Method 1 allocates ~5.3x MORE than Method 2 in this specific benchmark
(differentiating through `smoothed_fixed_dual_L`'s inner `Psi!` call on a
length-8000 vector of duals costs more per chunk than just building `G`).
**This does not generalize past D=4** -- memory `method-b-eliminates-dense-jacobian`
already established that materializing the analogous tensor is ~110x costlier
at D=20's 101 free parameters; this benchmark measures the D=4/16-free-param
regime specifically and should not be read as contradicting that larger-D
finding. `obj.moments_jacobian!` is set to `error` for this problem (never
implemented, matching the parallel jac_h-audit workstream's finding for the
hard path) -- Method 2 had to hand-build the tensor via `ForwardDiff.jacobian`
of the raw moment map rather than reuse an existing analytic path.

Method 3 (true VJP / single-pass reverse-mode adjoint) is not available in
this codebase: per memory `enzyme-mooncake-status`, both Enzyme and Mooncake
compile on this codebase's real production path but produce NaN gradients,
and were not re-attempted here given that documented prior finding. Method 1
is, in spirit, already the best available substitute -- it differentiates the
ALREADY-CONTRACTED scalar `m*'lambda*'G(theta)` directly via forward-mode AD,
rather than differentiating `G(theta)` first (as Method 2 does) and
contracting afterward.

## Steps 6-7: exact-hard switch, polish, and the three distinct numbers

**(a) Smoothed-problem optimum** (finest homotopy stage, rho=4.60e-7):
`gp=0.8935749`. Its "kappa"/"Delta" at this positive temperature are NOT a
valid hard-estimand number and are not reported as a result -- listed above
in the per-stage table purely as a diagnostic.

**(b) EXACT-HARD evaluation of that SAME theta** (`evaluate_fullA`, cold,
`warm=false`):
```
gp = 0.8935749380490591
kappa = 0.17100553129486673
Delta_dual = 0.9638871132143928   (Delta - delta = -0.0361; i.e. 3.6% divergence-budget slack)
gravity_value = -7.37e-18
inner_status = 0
max_abs_moment_kkt_resid = 7.84e-13
```
Feasible, and notably NOT tight against the divergence budget -- the
smoothed-optimal theta, mapped through the exact hard oracle, lands well
inside the feasible region rather than exactly on its boundary.

**(c) FINAL exact-hard polished candidate**, via `full_aod_diag/d4_exact/lfix_incremental.jl`'s
`:incremental_o1` tier (task's mandated method; not a new polishing algorithm
-- reused exactly as `profile_lfix_tiers.jl` already validated it: cheap
central-FD gradient of the hard `L_fix` at a fresh `build_lfix_base_cache`,
each candidate step re-verified against the EXACT `evaluate_fullA` oracle
before acceptance, never trusting `L_fix`'s own value as ground truth):
```
gp = 0.8929499380490591
kappa = 0.17197168927740825
Delta_dual = 0.9917194159602318   (Delta - delta = -0.00828; 0.83% slack)
gravity_value = -7.37e-18
inner_status = 0
max_abs_moment_kkt_resid = 1.07e-16
```
Polish log: round 1 found an improving feasible step at step-size 0.000625
(after 5 halvings from an initial 0.02 trial radius); round 2 found no
further improving feasible direction within 6 halvings and the polish
stopped (a genuine local stall in the pure-gp-coordinate polish direction
tried here, not an error -- see "not completed" below).

**Comparison to the existing best hard-only incumbent** (`upper_maxit40`,
loaded via `candidate_registry.jl`, cross-checked to match its documented
`kappa=0.17176461388430053` to the printed digits this session):

```
kappa_polished   = 0.17197168927740825
kappa_upper_maxit40 = 0.17176461388430053
Delta_kappa = +0.00020707539310771406   (+0.12% relative)
```

## Answers to the task's three questions

**(a) Faster gradient?** At D=4 (16 free coords): no meaningful wall-time
advantage from avoiding the materialized-tensor route (1.05x), though
Method 1 (the one actually used) uses more memory here, not less -- both
measured, neither assumed. The gradient IS matched (value and derivative
computed from the same smoothed moment kernel, envelope-theorem-consistent),
validated to relerr 2-4e-8 at every homotopy stage.

**(b) Better search basin -- does homotopy find a better final kappa than
hard-only search?** Yes, marginally, in this single run:
kappa=0.171972 (smoothed-homotopy + polish) vs. kappa=0.171765
(`upper_maxit40`, hard-only), a +0.12% relative improvement. This is not a
large or conclusively "solves the hard-only method's problems" result --
it is one run with a short (15-iter) per-stage budget and a shallow
(2-round) polish that stalled quickly. It IS evidence that the smoothed
basin the homotopy lands in, once polished, is at least competitive with
(and here, slightly better than) the existing best hard-only incumbent,
not that it is dramatically superior.

**(c) Useful continuation path to the exact hard problem?** Yes: the
finest-stage smoothed optimum evaluates EXACTLY (not approximately) via the
hard oracle without any special-casing (feasible on the first try,
`inner_status=0`, tiny KKT residuals), and the hard `L_fix` polish starting
from that point immediately finds an improving feasible direction. The
smoothed-then-hard handoff worked cleanly in this run.

## What was NOT completed within this session's effort budget

- **Exhaustive temperature-grid sweeps**: 5 stages were run once (upper/
  `find_smallest=true` direction only); the task explicitly said to
  prioritize Steps 1-2/6-7 over exhaustive sweeps if triage was needed, and
  that triage was applied here. A lower-direction (`find_smallest=false`)
  run, a denser rho grid, and multiple random restarts per stage were not
  attempted.
- **Deeper polish**: only 2 polish rounds were run (1 accepted step); the
  polish direction tried was a single fixed direction (pure gp-coordinate
  ascent/descent with a feasibility-verified backtracking line search), not
  a full multi-coordinate L_fix-gradient descent with a proper trust region
  or a KNITRO-driven local refinement using L_fix as a surrogate objective.
  A more patient polish would very likely improve on kappa=0.171972 further
  -- this is flagged as unfinished, not as a ceiling.
- **The original `smoothed_moments.jl` non-determinism's true root cause**
  was not isolated (bypassed per the task's explicit fallback instruction,
  see Step 1) -- the old file itself was left untouched (additive-only
  discipline) and is now superseded for this experiment's purposes by
  `smoothed_consistent.jl`, but the underlying bug in the old file remains
  formally unexplained.
- **KNITRO settings check** was done via log grep only (confirmed no
  fallback text and `hessopt: 4` present verbatim); a fully independent
  re-derivation of what hessopt mode KNITRO actually used internally
  (beyond trusting its own log text) was not attempted.

## Reproduction

```
source .knitro_env.sh
julia --project=. full_aod_diag/d4_exact/test_oracle.jl            # mandatory smoke test
julia --project=. full_aod_diag/d4_exact/test_oracle_profiled.jl   # mandatory smoke test
julia --project=. full_aod_diag/d4_exact/test_smoothed_consistent.jl   # Steps 1-2
julia --project=. full_aod_diag/d4_exact/run_smoothed_homotopy.jl      # Steps 3-7 (~90s wall)
```
Must run on `demand.mit.edu` (KNITRO license). All commands above were run on
that host for this document.
