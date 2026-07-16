# Correcting the winner-boundary derivative fix for the full (D+2)-moment sequential problem

Status: Stage A (D=4, BOTH bounds) fully validated and passing cleanly. A real-data D=20,
W=80000, upper-bound, delta=1 run (`GRADIENT_METHOD=fixed_dual_fd_full`) was launched
overnight per explicit user request, ahead of the originally-planned Stage B/C ordering, and
finished quickly -- but post-hoc verification found a critical, UNRESOLVED issue (a likely
severe gradient-scale mismatch that appears to have prevented A from moving at all, for either
gradient method) that must be resolved before that result (or the pre-existing pointwise_ad one)
can be trusted -- see section 7. Boundary-derivative correction (Part 6) not yet started.

## 1. Architecture note (Part 1)

Repo: `trade_robustness_modular_perf`, branch `feature/sequential-inversion-perf`. Entry point:
`sequential_gravity/run_profiled_production.jl`.

1. **D+2 moment vector**: built by `make_stateful_moments`'s closure `m!`
   (`run_profiled_production.jl:392-427`). `G[:,1:D+1]` = D focal trade shares + 1 price-index
   moment, via `EK_moments_focal_norm_directgp!` (`focal_moments_directgp.jl`) -- exact hard
   argmin winners recomputed fresh every call. `G[:,D+2]` = the gravity-linearized moment: for
   `Float64` theta that differs from the cached `lastθ[]`, a FULL re-solve (`seq_gravcol`,
   including re-inverting the D-1 omitted destinations) is triggered and `G[:,D+2]` is set to the
   fresh per-draw column `gcol[]`; for `ForwardDiff.Dual` theta (i.e. inside any AD call), the
   linearization is FROZEN: `G[ω,D+2] = (gcol[ω]-lastRcol) + (lastRcol + dRdθ'(θ-lastθ))`, an
   affine surrogate around the last successfully-evaluated Float64 point.
2. **Ordering**: `H = [K | 1 | G]`, `G` columns `1..D` = focal trade shares, `D+1` = price-index
   (gamma') moment, `D+2` = gravity-linearized moment. `d=D+2`, `outer_constr_index=d+1=D+3`,
   confirmed directly from `outer_solve_nested_cached` (`d = D + 2; oci = d + 1`).
3. **Full inner dual criterion**: `PsiObjectiveBundleDelta`'s callable
   (`cc_algo/PsiObjectiveBundle.jl:298`, `f = sum(Psi(arg0))/M + zeta`,
   `arg0 = H[:,2:1+oci]*(-x)`) -- generic in `d`, works unchanged for `d=D+2`.
4. **Full inner problem solver**: `inner_loop`/`inner_loop_internal(obj::PsiObjectiveBundleDelta, θ)`
   (`cc_algo/inner_loop_functions.jl:222`) -- a real KNITRO solve over `(zeta,lambda)`, generic in
   `d`.
5. **Outer parameterization**: `theta = [mu, sigma, gamma'_focal, A[.,focal] (length D)]`, RAW
   LEVEL (not log). `free_idx = vcat(3, 4:3+D)` (gamma'_focal + all of A[.,focal]); mu, sigma
   pinned via `FreeParamMap`. Confirmed directly, not assumed.
6. **Outer gradient consumer**: `make_seq_div_grad_fn!`/`_methodB_envelope_scalar`
   (`PsiObjectiveBundleImplicitMethodB.jl`), a ForwardDiff scalar gradient of the divergence-budget
   outer constraint, through the hard argmin winner -- this is what's wrong (misses the
   winner-boundary term).
7. **Inversion <-> gravity-moment relationship**: `seq_gravcol` iterates: solve blind D+1-moment
   CC problem -> LFD `p` -> invert D-1 omitted destinations given `p` -> gravity residual `R` ->
   linearize `R` in theta (`grad_R_theta`, exact ForwardDiff through `focal_u`/`dest_share`, NOT
   through the hard winner since gravity's own gradient is analytic/smooth) -> augment moments
   with the linearized column -> re-solve -> repeat to `|R|<=tol`.
8. **Is A[.,focal] direct?** Yes (Case A) -- `θ[4:3+D]` is copied straight into `x_free`, no
   inversion in between. Confirmed via `focal_moments_directgp.jl:19` and the `FreeParamMap`
   construction.

## 2. The bug in the previous session's wiring (confirmed by direct inspection, not by trusting the report)

`derivative_diagnostics/fixed_dual_criterion.jl`/`fixed_dual_fd.jl`/`boundary_derivative.jl` were
validated and wired using `D1 = D+1` and `EK_moments_focal_norm_directgp!` alone -- **no gravity
moment, no lambda_R** anywhere in that machinery. `gradient_method_wiring.jl`'s
`make_seq_div_grad_fn_corrected!` then computed a correction on that reduced problem
(`λ_baseline = λfull[1:D1]`, dropping `λfull[D+2]` entirely) and added it to the full-(D+2) AD
gradient. This is exactly the invalid pattern the task describes: `lambda_R*G_R` is a common
offset inside the nonlinear `Psi(arg0)`, so dropping it and patching the boundary term in
afterward is not generally valid. Confirmed directly in the source, not inferred from the report's
prose.

A **second, independent bug** in that same file: `build_fixed_dual_bundle` never passed
`find_smallest`, so `bundle_tmp` always used the struct default (`true`). Whether this happened to
be correct depended on which "true" meant for that specific temp bundle -- see the finding in
section 4 below (this turned out to be the CORRECT convention by accident, not a bug -- see
"correction to my own earlier claim").

## 3. What was built (Parts 2-3-4-7)

- `full_fixed_dual_criterion.jl`: `make_frozen_gravity_moments` (builds the FULL D+2 frozen-moments
  function, replicating `make_stateful_moments`'s own Dual-eltype formula but usable at Float64
  theta too, since finite differences perturb Float64), `freeze_gravity_linearization` (runs the
  REAL `seq_gravcol`+`grad_R_theta` at a real theta to get a frozen linearization),
  `test_full_fixed_dual_identity`.
- `fixed_dual_criterion.jl`/`fixed_dual_fd.jl` (already generic in `d`/`moments_fn` from the prior
  session) were REUSED UNCHANGED for the full-D+2 problem -- no rewrite needed, just called with
  `d=D+2` and the new frozen moments function.
- **Identity test** `Q_full(theta_k,x_k*) == delta*(theta_k)`: **passes exactly** (rel_diff=0.0,
  not merely small) at the Frechet benchmark AND an off-benchmark target, D=4, W=8000/32000.
- **Full-(D+2) fixed-dual FD gradient vs fully re-solved profile FD**, D=4, W=8000/32000, 8
  directions (coordinate/random/multi-origin/negative-gradient x 2 targets): FD tracks the
  expensive ground truth to **0.05%-2% relative error** (worse, ~10-28%, only in the near-degenerate
  Frechet-point "random" direction case, a known zero-crossing artifact per the earlier session's
  own Part 4 finding). Current pointwise AD is off by **60%-230%** at the SAME points. This is the
  central claim of the task, now confirmed at the ACTUAL full-D+2 production problem, not the
  reduced D+1 diagnostic.
- **Genuinely sample-exact D+2 negative control** (`full_sample_exact_control.jl`): recenters every
  one of the D+2 moment columns by its OWN frozen sample mean at `(A*, gamma'_frechet)`, computed on
  the exact draw matrix. `delta*(A*,gamma'_frechet)=0` to solver precision (not merely small).
  50-point perturbation scan (10 directions x 5 step sizes): zero negative divergences, quadratic
  scaling with step size as expected, giving a numerical floor of ~1e-10 (h=1e-4) to ~1e-4 (h=0.1)
  at W=8000.

## 4. Bugs found and fixed while integrating into the real outer loop (Parts 8-10)

1. **`find_smallest` sign convention for `PsiObjectiveBundleDelta`** -- the most consequential
   finding. `PsiObjectiveBundleDelta` computes the genuine CC divergence delta*(theta), a quantity
   with NO dependence on which OUTER bound direction (lower/upper) is being searched --
   production's own `recover_lfd` NEVER varies it, always using the struct default `true`. An
   earlier version of this session's own code (mirroring what looked like a bug in the prior
   session's `gradient_method_wiring.jl`) threaded the REAL outer bound's `find_smallest` through
   into `build_fixed_dual_bundle` calls, believing this was fixing a missed-sign bug. It was not:
   caught by cross-checking `dual_criterion_fixed_x` against `inner_loop` directly at the SAME
   theta under both conventions -- `find_smallest=true` gives `delta*=+2.78e-4` (matching every
   Part 1-7 validation), `find_smallest=false` gives `delta*=-2.78e-4` (same magnitude, wrong
   sign, mathematically impossible for a genuine divergence). Fixed: `find_smallest=true` is now
   hardcoded everywhere a `PsiObjectiveBundleDelta` is built for computing delta* in this session's
   new files, independent of the real outer bound direction. (Net effect on the actual outer-loop
   *gradient* fed to KNITRO turned out to be nil for the lower bound specifically, because a second,
   compensating `-1` had been folded into the unit-conversion factor -- the two errors canceled.
   The `Part 10` exact-audit function, which has no such compensating factor, WAS materially wrong
   before this fix and is now correct.)
2. **`best_κ` is a misnomer** -- `make_stateful_moments` tracks the best point by comparing raw
   `K[1]=gamma'_focal`, not the actual `kappa=1-gamma'^(sigma/(sigma-1))`. Since kappa is a
   strictly DECREASING function of gamma', treating the stored value as kappa directly (as this
   session's driver initially did) silently flips every "does full-A beat fixed-A" comparison.
   Always convert via `gp2kappa`.
3. **The raw KNITRO outer endpoint is not always gravity-feasible.** `constr[1]` (the divergence
   budget) is evaluated using `make_stateful_moments`'s placeholder `INFCOL` gravity column
   whenever `seq_gravcol` fails to converge at a trial theta -- so the outer search can wander into,
   and even terminate at, a gravity-infeasible point while KNITRO reports it as converged. This is
   PRE-EXISTING production behavior (not introduced this session) -- exactly why
   `make_stateful_moments` already tracks `best_θ`/`best_κ` (the best point verified
   gravity-feasible via a real `seq_gravcol` call at the time). Both `outer_solve_fixedA` and
   `outer_solve_nested_cached` results must be read off `best_θ`, never the raw KNITRO endpoint --
   this session's test driver was fixed to do so.
4. `freeze_gravity_linearization` unconditionally `error()`-ed on an infeasible theta, making it
   impossible for an audit function to check `.ok` gracefully at an arbitrary (possibly
   KNITRO-endpoint) theta. Changed to return `(ok=false, ...)` gracefully; validation-drivers that
   need a hard failure assert `.ok` themselves with task-specific context.
5. **Gravity-feasibility conflated with divergence-budget-feasibility** (caught by the user
   challenging a nonsensical "A\* is provably gravity-infeasible" claim in an early draft of this
   report -- see section 5.1). `seq_gravcol` returns a COMBINED `ok = gravity_ok && δ_ok`; the
   audit code read that combined flag and reported delta\*=Inf ("infeasible") whenever it was
   false, without distinguishing "gravity genuinely violated" from "gravity fine, but this one
   internal budget check (irrelevant to an exact-delta\* audit) didn't pass." Fixed:
   `freeze_gravity_linearization` now defaults to `δ=Inf` internally, decoupling its own
   `.ok`/`.gravity_ok` from any budget, and separately exposes `div_p`.
6. **`seq_gravcol`'s default `maxit=20` is sometimes an iteration-limit artifact, not genuine
   non-convergence.** Found finalizing the upper-bound audit: a fixed-A comparator point showed
   `gravity_ok=false` at maxit=20 (R_mean=-1.21e-3, tol=5e-4) but converged cleanly by maxit=50
   (R_mean=-2.76e-4) and stayed stable through maxit=200 -- i.e. it just needed more sequential
   iterations, not less divergence. Fixed: `freeze_gravity_linearization`'s default `maxit` raised
   to 100.

## 5. D=4 Stage-A outer-loop result (BOTH bounds, delta=1, W=8000, maxit=1000) -- FINAL

| bound | kappa_fixed | kappa_full | Part-9 sanity | rel‖ΔA‖ | δ\*\_movedA | δ\*\_fixedA (same γ') | Δδ |
|---|---|---|---|---|---|---|---|
| lower | 0.005297 | **0.004062** | PASS | 30.9% | 0.9915 | 1.2411 | **+0.2496** |
| upper | 0.159080 | **0.172109** | PASS | 45.7% | 1.0180 | 1.4102 | **+0.3922** |

Both bounds: full-A weakly (in fact strictly) dominates fixed-A\*, and moving A buys a genuine,
finite divergence-budget saving (25-39%) at both ends of the kappa interval. This is the clean,
final Stage-A result -- both directions now pass every required check (Part 9 sanity, Part 10
exact audit) with no open anomalies. For comparison, the original (pre-this-session, D+1-flawed)
report's own lower-bound "extra_delta" for its `fixed_dual_fd` method was +0.26 at this same
delta=1, D=4 setting -- close to this session's corrected +0.25, a useful cross-check even though
that earlier number came from an invalid derivative.

**Two bugs surfaced finalizing this table, both now fixed** (full detail in section 4, items 5-6):
1. An early draft of this report claimed "A\* is provably gravity-infeasible" at the lower-bound
   comparator point -- **wrong, and correctly challenged by the user**. The gravity residual R
   depends only on the imputed u-matrix (A, tau, and -- for non-focal destinations -- the
   reweighting p used to invert their observed shares), never on gamma' or F, and A\* is
   gravity-consistent at the Frechet benchmark by construction. The real issue:
   `freeze_gravity_linearization` returned `seq_gravcol`'s COMBINED `gravity_ok && δ_ok` flag, so
   "gravity fine but this budget check didn't pass" got reported as "infeasible." A verbose trace
   confirmed `gravity_ok=true`, `δ_ok=false` (div(p)=1.24 > budget 1) -- an ordinary "needs more
   budget" finding, not an impossibility. Fixed by decoupling gravity-feasibility from any budget
   (default `δ=Inf` inside the freeze).
2. The upper-bound fixed-A comparator then showed the SAME symptom (`gravity_ok=false`) even with
   the budget decoupled -- this time a genuine second bug: `seq_gravcol`'s default `maxit=20`
   sequential-loop cap wasn't enough iterations at that point (converges cleanly by maxit=50,
   stable through 200). Fixed by raising the default to 100.

## 6. What remains (not yet done)

- Boundary-derivative correction to full D+2 (Part 6) -- deliberately deferred per the task's own
  priority order ("only after all [FD] tests pass").
- W-scaling of the OUTER-LOOP result itself (Part 5 asked for W=8k/32k/128k/800k derivative
  stability, done for the derivative estimate in isolation; the full outer-loop *search* has only
  been re-run end-to-end at W=8000 so far).
- D=10 scaling (Stage B) -- skipped directly to D=20 real data per explicit user request (see
  section 7) ahead of the originally-planned Stage B/C ordering.
- Final markdown report answering all 8 required questions in full, with Monte Carlo/step-size
  plots and runtime tables.

## 7. D=20 real-data run (launched ahead of the original Stage B/C ordering, per explicit user request)

The user explicitly asked to skip directly to a real-data D=20 production run overnight (W=80000,
upper bound, delta=1, `gradient_method=fixed_dual_fd_full`, `PARALLEL_INVERSION=true`,
`julia -t 19`), accepting that Stage A/B's own incremental validation had not yet reached D=20 --
an explicit, informed instruction to proceed, not something this session decided on its own.

Launched via `run_d20_upper_delta1_W80000_fixeddualfdfull.sh` (fully detached: `nohup setsid ...
& disown`, survives the session ending). A `GRADIENT_METHOD` env var was added to
`run_profiled_production.jl`'s batch loop (previously hardcoded to `:pointwise_ad`) so this could
run without hand-editing code. Uses a SEPARATE `OUT_DIR`
(`batch_out_realD20_W80000_fixeddualfdfull`) from the pre-existing `pointwise_ad` D=20/W=80000
results, since checkpoint filenames don't encode gradient_method and the resume logic would
otherwise silently skip this run entirely.

- Log: `sequential_gravity/d20_upper_delta1_W80000_fixeddualfdfull_run.log`
- Result on completion: `sequential_gravity/batch_out_realD20_W80000_fixeddualfdfull/seq_upper_delta1.0.jld2`
- Sentinel: `sequential_gravity/d20_upper_delta1_W80000_fixeddualfdfull_ALLDONE`
- KNITRO's own `maxtime_real=10000s` (~2.8h, from `csw_outer_1000.opt`) caps a runaway outer solve.

Confirmed progressing normally at launch (21 free variables = D+1 as expected, objective
descending over the first several iterations, no errors). It finished fast: 5 minutes, 11 inner
solves, KNITRO status 0 ("locally optimal"), gravity-feasible, gamma'=0.953252, kappa=0.076693.

**Post-hoc check found a real, important issue -- this result should NOT be trusted as-is.**
Comparing against the pre-existing `pointwise_ad` result at the identical setting
(`batch_out_realD20_W80000/seq_upper_delta1.0.jld2`), the two runs' `gamma_p`/`kappa` match to
10 significant figures, and **`theta_star`'s Acol block is EXACTLY (bit-identical, rel diff
0.0) equal to A\* -- the search never moved A at all**, under either gradient method.

This is NOT the "moving A doesn't help here" finding it might look like. Direct inspection (see
`debug_d20_gradient_check.jl`) confirms the FD path genuinely IS being exercised and genuinely
DOES differ from AD at theta_r0 -- by up to ~5x in individual coordinates (e.g. one Acol
coordinate: AD=-2.08, FD=-10.21) -- so this is not a silent AD fallback bug. The likely
explanation is a **severe cross-variable gradient-scale mismatch** in the outer KNITRO
parameterization: at this real D=20 point, `g_free[1]` (gamma'_focal's own component) is
`~-4.1e8`, while the entire Acol block is only `~10 to 11500` in magnitude -- a ratio of order
**10^4-10^5**. At D=4 the SAME ratio was only ~160x (`g_free[1]~1.6e9` vs Acol block `~1e6-1e7`,
already converted to native units) -- two orders of magnitude milder. It is plausible KNITRO's
unscaled SQP/interior-point solver, faced with a gradient this lopsided, effectively cannot
"see" the Acol directions as worth moving within the handful of iterations it takes before its
own (relative) convergence criteria are satisfied in the dominant gamma' direction -- for EITHER
gradient method, since both share the same γ'-component and the same catastrophic scale gap.

**Action for next session, before trusting any D=10/D=20 result (old OR new)**: check whether this
scale mismatch is present at D=10 too (scaling with D, or specific to this D=20 real economy);
consider whether the outer free-parameter parameterization needs rescaling (e.g. differencing in
log(Acol) rather than raw level, or an explicit KNITRO variable-scaling option) so the Acol block
gets genuinely explored. Until that's understood, neither this session's `fixed_dual_fd_full`
D=20 result NOR the pre-existing `pointwise_ad` D=20 result should be read as "A* is already
optimal at D=20" -- both may simply never have looked.

### 7.1 A first attempt at KNITRO variable scaling (user-requested follow-up) -- mixed results, NOT a clean fix

Added `KN_set_var_scalings_all` support (`var_scales` kwarg on `cc_algo/outer_loop_cached.jl`,
`use_var_scaling`/`scaling_power` kwargs and a `USE_VAR_SCALING`/`SCALING_POWER` env var on
`outer_solve_nested_cached`) -- additive, off by default, verified byte-identical to the
unscaled baseline when off. Precedent: `full_aod_diag/solve_scaled.jl` (a DIFFERENT method, the
full-A_od outer loop, using the same KNITRO API for a different ill-conditioning reason).

Design: a probe inner solve + one gradient evaluation at theta_init gives a representative
constraint-gradient magnitude per free coordinate. **First attempt (scale EVERY coordinate,
including gamma'_focal, by `1/|g_probe[i]|`) was wrong and caught immediately**: KNITRO variable
scaling is shared between the objective and constraint (it is a property of the variable, not of
which function differentiates it), and the outer OBJECTIVE's own gradient is a hard-coded +-1 at
the gamma'_focal coordinate only. Scaling that coordinate by `1/|g_probe[1]|` (~1e-9 at D=4)
shrunk the SCALED objective gradient to ~0 everywhere, so KNITRO's own KKT check was satisfied
trivially at multiplier~0 -- the D=4 test converged in ZERO iterations, at theta_init, having
moved nothing -- WORSE than no scaling at all. **Fixed** by leaving gamma'_focal's own scale at
1.0 (preserving the objective's natural units) and rescaling ONLY the Acol block relative to
gamma's own constraint-gradient magnitude, with a caller-tunable `scaling_power` interpolating
between no correction (Acol scale=1, i.e. `scaling_power=0`) and full magnitude-matching
(`scaling_power=1`).

**D=4 test results, delta=1, lower bound, W=8000** (same setting Stage A already validated
cleanly without scaling):

| scaling_power | kappa | rel‖ΔA‖ | wall | KNITRO status |
|---|---|---|---|---|
| none (baseline) | **0.004062** | 30.9% | ~115-130s | -103 (feasible) |
| 0.5 (sqrt) | 0.005710 | 103.3% | 176.6s | -103 (feasible) |
| 1.0 (full match) | 0.026406 | 383.3% | 774.6s | -102 (worse, not converged) |

**More scaling monotonically makes D=4 WORSE, not better** -- kappa gets larger (worse for the
lower bound), wall time balloons, and the highest setting fails to converge cleanly. The scaled
search DOES move A much further, but overshoots into a worse region rather than finding a better
one -- consistent with KNITRO taking too-aggressive Newton/interior-point steps once the Acol
directions are artificially inflated, at a D where the UNSCALED search was already exploring A
just fine (31% movement, clean convergence) without any help. This is the opposite of what
naively "the search wasn't looking at A" would predict, and is worth taking seriously: **this
specific scaling design is not a clean, unconditional win.**

### 7.2 D=20 real-data result WITH scaling (labeled scaling_power=0.5 -- CORRECTED to 1.0, see below) -- genuine improvement, but expensive and imperfect

**CORRECTION found later the same day**: `run_one_bound` (the batch-loop entry point that
actually produced this result and every later delta-grid point) never threaded the
`scaling_power` keyword through to `outer_solve_nested_cached` -- the `SCALING_POWER=0.5`
env var set at launch was silently ignored, and the run used the function default,
`scaling_power=1.0`, not 0.5. Fixed in `run_profiled_production.jl` for future runs. The
numeric RESULTS below are unaffected and independently verified (see section 7.3 and
`verify_d20_deltagrid.jl`) -- only the "power=0.5" label throughout this section and 7.3 is
wrong; read it as 1.0. This also means the D=4 finding just below (power=1.0 "overcorrects
badly") did NOT reproduce at this D=20 setting, which used power=1.0 throughout and
converged to a clean, genuinely-improving, verified result -- an open discrepancy, not yet
understood, worth investigating (does the D=4-vs-D=20 scaling_power sensitivity actually
differ, or was the D=4 test simply a different/harder case for other reasons?).

Same setting as section 7 (W=80000, upper bound, delta=1), `USE_VAR_SCALING=true` (nominal
launch used `SCALING_POWER=0.5`, ACTUAL run used the silently-defaulted 1.0 -- see
correction above), separate `OUT_DIR` (`batch_out_realD20_W80000_fixeddualfdfull_scaled05`):

| | unscaled (section 7) | scaled (power=0.5) |
|---|---|---|
| kappa (best-feasible) | 0.076693 | **0.081518** |
| gamma'\_upper (best-feasible) | 0.953252 | 0.950260 |
| rel‖ΔA‖ (best-feasible) | **0.0% (bit-identical to A\*)** | **522.5%** |
| raw KNITRO endpoint status | 0 (locally optimal) | -101 (iteration/tolerance limited) |
| raw endpoint gravity-feasible | true | **false** (had to fall back to best-feasible) |
| wall time | 210s (~3.5 min) | **2742s (~46 min, ~13x slower)** |
| inner solves | 11 | 130 |

**This time the direction matters: this is the UPPER bound (find_smallest=true, maximizes kappa),
so a LARGER kappa is BETTER here** -- unlike section 7.1's D=4 LOWER-bound test, where a larger
kappa was worse. Reading both correctly: scaling **genuinely helped at D=20** (kappa rose from
0.0767 to 0.0815, a real ~6.3% gain, achieved through a huge, clearly-not-just-noise A movement of
523%) while it **genuinely hurt at D=4** (section 7.1, lower bound, kappa rose from 0.0041 to
0.0057-0.0264, worse for that direction). The two results are NOT contradictory once the bound
direction is accounted for -- but they still don't add up to "scaling is the fix": at D=20 the
scaled search is far more expensive (13x wall time) and does NOT converge cleanly (status -101,
raw endpoint gravity-infeasible, only the best-feasible fallback is usable) -- a materially worse
NUMERICAL outcome than D=4's clean unscaled convergence, even though the ECONOMIC number
(kappa) improved.

**Overall conclusion on rescaling**: it is a genuine, confirmed lever -- at D=20 it converts "A
literally never moves, kappa=point-estimate-adjacent" into "A moves substantially and kappa
improves by ~6%", which is real signal that moving A helps even at production scale. But it is
NOT a clean, unconditional fix to adopt as a new default:
- At D=4 (where the unscaled search already explores fine) the SAME scaling makes results
  strictly worse and slower.
- At D=20 it fixes the "doesn't move at all" pathology but introduces a new one (expensive,
  unclean convergence) -- likely because a single constant `scaling_power` chosen from one probe
  gradient at theta_init is too blunt an instrument; the true sensitivity ratio probably varies
  substantially across the search trajectory, especially over a 5x movement in A.
- `scaling_power` was only spot-checked at two values (0.5, 1.0) at D=4 and one (0.5) at D=20 --
  not remotely a systematic sweep.

**Recommendation for next session**: do NOT adopt `USE_VAR_SCALING=true` as a new default. Worth
pursuing further (in rough priority order): (a) a genuinely adaptive/multi-point scaling (e.g.
re-probe periodically during the search, not just once at theta_init); (b) initializing the
scaled D=20 search from the ALREADY-KNOWN unscaled result (gamma'=0.953, A=A\*) instead of cold
from theta_r0, which might reach a comparably good point much faster since it wouldn't need to
rediscover that A\* is a decent starting basin; (c) the fixed-A-incumbent warm-start design from
Part 9 (`fixed_A_incumbent.jl`), never actually wired into the production batch loop, which might
sidestep the scaling problem entirely by starting the full-A search from a good point rather than
needing to explore blindly from theta_r0; (d) a genuine reparameterization (log-Acol) instead of
KNITRO-level scaling. The kappa=0.0815 number from this session should be read as "moving A likely
helps at D=20 real data, by a similar order of magnitude to D=4's finding" -- suggestive and
consistent with the D=4 result, not yet a publication-quality bound.

### 7.3 D=20 delta grid (0.1, 1.0, 2.0, 5.0) -- a real bug found in the warm-start chain, fixed

Extended the D=20/W=80000/upper-bound/scaled(power=0.5) run to a delta grid, matching what
had already been run with `pointwise_ad` (`batch_out_realD20_W80000`). Production's batch
loop (`run_profiled_production.jl`, the `for delta in DELTA_GRID` block) warm-starts each
delta from the PREVIOUS delta's result, ascending.

**Bug found**: both the fresh-solve and resume code paths warm-started the next delta from
`r.θstar` / `existing["theta_star"]` -- the RAW KNITRO endpoint -- never `best_θ` (the
verified gravity-feasible point production already tracks separately, see bug 3 in section
4). At delta=1.0, the raw endpoint happened to be gravity-INFEASIBLE (only `best_θ` was
usable, kappa=0.0815 -- see section 7.2). Chaining delta=2.0 and delta=5.0 from that broken
raw endpoint meant BOTH found literally ZERO feasible points in ~50-55 minutes of search
each (`best_feasible_kappa=NaN`) -- not merely worse results, complete failures, because
they started from a bad point and the search never recovered.

**Fix**: both warm-start sites now use `best_θ`/`existing["best_feasible_theta"]`, falling
back to the raw endpoint (with a printed warning) only if no feasible point was ever found
at all. Re-ran delta=2.0 and delta=5.0 (delta=0.1/1.0 resumed from disk, confirmed
unaffected). Verified the fix directly: delta=2.0's KNITRO trace now starts at iteration 0
objective = 0.950260 -- exactly delta=1.0's `best_θ` value, not the old 0.937687 raw
endpoint.

**Final delta-grid result** (D=20 real data, W=80000, upper bound,
`gradient_method=:fixed_dual_fd_full`, `use_var_scaling=true scaling_power=0.5`):

| delta | best-feasible kappa | gravity-feasible | wall |
|---|---|---|---|
| 0.1 | 0.04471 | yes | 2.2 hr |
| 1.0 | 0.08152 | yes | 46 min |
| 2.0 | **0.09149** | yes | 56 min |
| 5.0 | **0.09836** | yes | 1.6 hr |

Monotonically increasing in delta, as required (a larger divergence budget can only weakly
increase the achievable kappa for the upper bound) -- a basic sanity check the broken
warm-start chain could not have passed (it produced no result at all for two of the four
points). Total wall time ~4.4 hours for the corrected run across all 4 points at W=80,000;
scaling to W=800,000 (mentioned as a possible next step) should be expected to cost
substantially more and was not attempted this session.

These numbers have NOT been individually post-hoc-validated the way delta=1.0's endpoint
was (see the parallel profiled-reformulation session's finding, memory entry
`profiled-reformulation-fix-gp-min-deltastar`: delta=1.0's scaled endpoint verified as a
genuine local optimum via an independent unconstrained-minimization check). Recommend the
same check be run at delta=0.1/2.0/5.0 before treating this table as final.

### 7.4 Independent re-verification of all four delta-grid points -- PASSES

All four points (`verify_d20_deltagrid.jl`) independently re-checked from scratch: fresh
cold re-solve, gravity residual, ALL D trade-share moments (focal at the correct hard-argmin
rho=0, omitted destinations at the correct smoothed rho=0.002 -- two false-alarm bugs in the
verification script itself, not the production solution, were found and fixed getting this
right; full account in `D20_METHOD_WRITEUP.md` section 5), and an independent exact-delta*
audit at each point. All four pass cleanly (trade-share errors ~1e-6/1e-7, delta* within
0.3% of the nominal budget). The fixed-A* comparison shows genuine, and at delta=2.0/5.0
ABSOLUTE (not merely costlier), superiority of the moved-A result -- A* cannot reach those
gamma' targets at all. Full table and the exact method used: `D20_METHOD_WRITEUP.md`.

## Files (this session)

- `full_fixed_dual_criterion.jl`, `full_profile_resolve.jl` (generalized from D1-only),
  `full_sample_exact_control.jl`, `fixed_A_incumbent.jl`, `full_gradient_method_wiring.jl` --
  the corrected full-(D+2) machinery.
- `run_full_d2_validation.jl`, `run_full_d2_gradient_validation.jl`, `run_full_d2_sample_exact.jl`,
  `run_full_d2_outer_loop_test.jl` -- Stage-A drivers.
- `fixed_dual_criterion.jl`/`fixed_dual_fd.jl` -- unmodified in substance (already generic),
  `find_smallest` kwarg added for correctness/explicitness.
- Modified (additive): `../run_profiled_production.jl` -- `make_stateful_moments` now also returns
  `lastθ/lastRcol/dRdθ/lastok` Refs; new `gradient_method` values `:fixed_dual_fd_full`/
  `:boundary_full` wired via `full_gradient_method_wiring.jl`.
