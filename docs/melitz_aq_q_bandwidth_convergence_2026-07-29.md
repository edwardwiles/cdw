# Melitz (A,q) q-bandwidth convergence campaign (2026-07-29)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from the
2026-07-29 daytime session's own `(A,q)` separation and exact-A-gradient work (commit
`c971c9c`, `docs/melitz_A_q_separation_and_gradient_diagnostics_2026-07-29.md`), which this
session read in full and does not re-derive. That session established strict A/q separation
and an exact analytical A-block gradient, but left the q-block bandwidth question genuinely
open (D4 only, `W in {20000,80000}`, one seed, two directions -- explicitly disclosed as
underscoped, not as evidence that larger `W` cannot help).

## Phase 0: repository and baseline audit

- Branch `melitz/fullD-delta-star`, HEAD `c971c9c` (the strict A/q separation + exact A
  gradient commit) at session start. 35 commits ahead of `cdw/melitz/fullD-delta-star`, not
  pushed. `git status` clean except pre-existing untracked scratch dirs from other sessions.
- Full test suite (`julia -t 1 test/melitz/runtests.jl`) confirmed clean **before any edit**:
  65 testsets, Pass==Total everywhere, exit code 0.
- Source files identified: `src/melitz/log_cutoff_param.jl` (q reconstruction/gravity),
  `src/melitz/exact_a_gradient.jl` (exact A gradient), `src/melitz/finite_delta_outer.jl`
  (gradient-backend dispatch -- `direct_gradient_fn` keyed off `gradient_backend` symbol,
  factory pattern `make_melitz_gradient_delta_direct_sorted_serial(h)`), `src/melitz/
  sorted_tail.jl` (`melitz_active_tail_start`, the sorted crossing infrastructure),
  `src/melitz/inner_screening.jl` (the four typed `MelitzInnerResult` subtypes: `FiniteSolved`,
  `AboveEvaluationCap`, `InfiniteDeltaCertified`, `NumericalFailure`), `src/melitz/
  inner_session.jl`/`inner_solve_policy.jl` (the one authoritative `solve_melitz_delta!
  (session, theta, policy)` entry point, `CappedEvaluation(cap)`).
- Prior session's own `scripts/melitz_aq_phase6_9_q_bandwidth_2026-07-29.jl` was read
  carefully per the governing prompt's instruction: confirmed its own crossing count
  (`q_crossing_report`) was computed **only for the `+h` perturbation** even though its
  reported secant was a central (two-sided) difference -- the exact flaw this session's
  Phase 1 was told to fix, not repeat.

## Phase 1: centralized q-bandwidth-policy infrastructure

New file `src/melitz/q_bandwidth_policy.jl`. One typed interface,
`abstract type MelitzQBandwidthPolicy end`, with four concrete policies exactly matching the
governing prompt's own menu:

```
FixedRawQBandwidth(h)
PowerScaledQBandwidth(h_ref, W_ref, alpha)
FixedCrossingQBandwidth(target; h_lo, h_hi, max_iter)
GrowingCrossingQBandwidth(target_ref, W_ref; h_lo, h_hi, max_iter)
```

The central evaluator, `melitz_q_coordinate_probe(theta0, m, policy, obj, ctx; x0, mode)`,
returns a `MelitzQCoordinateProbeResult` carrying: the actual `h` used, per-cell two-sided
crossing counts (`MelitzQCellCrossing`, covering the direct cell **and** the q-gravity pivot
cell whenever a free coordinate moves it -- both counted, never only the nominal coordinate),
`crossings_plus_total`/`crossings_minus_total`/`crossings_min_side`, an LFD-classification-
based boundary-hit proxy for each side (disclosed as a proxy, not a literal affine-constraint
geometry check), the requested secant (`mode=:fixed_dual` or `:reoptimized`), and
timing/allocation.

**The fix for the prior session's one-sided flaw**: `melitz_q_two_sided_crossings` computes
`+h` and `-h` crossing counts as two genuinely separate quantities; every target-crossing
policy bisects on `min(crossings_plus, crossings_minus) >= target` (the governing prompt's own
two-sided criterion), not a one-sided total. Regression test ("Two-sided crossing evaluator is
genuinely two-sided") confirms at least one tested D4 coordinate shows nonzero crossings on
**both** sides simultaneously at a representative step.

Reused, not re-derived: `melitz_active_tail_start` (`sorted_tail.jl`) for exact O(log W)
crossing detection via binary search on `sorted_ctx.sorted_z[:,o]` -- the same mechanism the
2026-07-27 closure session's own switch-diagnostic script used.

## Phase 2: the exact smooth fixed-active-set q derivative

**Audited from the actual code, not assumed.** `melitz_log_f_from_q` (`log_cutoff_param.jl`):
`log(f_od) = (sigma-1)*(q_od + a_od - log(markup) - log(w_o) - log(tau_od)) + log(expenditure_d)
- log(sigma) - log(w_o)` -- `d(log f_od)/d(q_od) = (sigma-1)` exactly at fixed `a_od`, the SAME
multiplicative scale `exact_a_gradient.jl` uses for `d(log f_od)/d(a_od)` (q and a enter this
formula symmetrically). But `melitz_C`/`firm_quantities.jl`'s trade-share coefficient `C_od =
expenditure_d*(markup*w_o*tau_od/A_od)^(1-sigma)` has **zero** dependence on `q`/`f` at all --
confirmed directly in the source (`C_od`'s own formula never references `f`). So, unlike the
A-block:

- The trade-share moment block's smooth q-derivative is **identically zero** (verified in a
  new regression test: every full q cell outside the focal origin row is exactly `0.0`, not
  merely small).
- The **only** smooth q channel is the focal free-entry link (`ell`), through the
  `-w_o*f_od` term of `melitz_firm`'s raw profit `profit_od(z) = (C_od/price_power_d)*
  z^(sigma-1)/sigma - w_o*f_od` -- at fixed `a_od`, `d(profit_od(z))/d(q_od) = -w_o*(sigma-1)*
  f_od` exactly (the `C_od*z^(sigma-1)` term drops out entirely, since it has zero
  q-derivative).
- The autarky sub-term has **no** q-analogue: `f_jj` depends only on `(gamma_prime_j,
  A[j,j])`, never on any free q coordinate, and `q[j,j]` is itself derived from `g` alone
  (`derive_qjj_from_autarky_cutoff`), independent of every free q coordinate.

**This is Governing-Prompt-Phase-2 Option A**: the fixed-active-set q derivative is nonzero
(not provably zero), so a hybrid derivative is implemented: new file `src/melitz/
exact_q_smooth_gradient.jl` (`melitz_exact_q_smooth_gradient_full!`/`_free`/convenience
wrapper), mirroring `exact_a_gradient.jl`'s structure exactly but reusing only its Step-2
`tail1`/`link_coef` machinery (no `mul_Gt!` trade-block pass at all, since that block
contributes nothing). Validated against a zero-switch fixed-dual secant (D4, 5 coordinates,
`h` bisected down until two-sided crossings are exactly `0`): **max symmetric relative error
4.14e-8** -- floating-point/O(h^2)-noise level, the correct signature of an exact formula.
Regression test enforces this for **every** free D4 q coordinate (not a sample), plus the
structural claim that every full cell outside the focal-origin row is bit-identically zero.

## Phase 3: fixed economic test points

Reused the prior post-consolidation session's own interpolate-g-from-profile-CSV mechanism
(`melitz_phase2_profile_recert_2026-07-28.jl`'s `interp_g_at_target`, attributed not
re-derived) rather than a new gamma-profile campaign. For each target, built the verified
`FiniteSolved` point under production `:logf`, then reconstructed the **identical** economic
point under `:logcutoff` via `reduce_to_free_theta_logcutoff` and round-trip-verified it
solves to the same `DeltaStar`.

| fixture | target | g | Delta (`:logf`) | Delta (`:logcutoff`) | \|mismatch\| |
|---|---:|---:|---:|---:|---:|
| D4 (seed=29, W=20000) | pareto | -0.158083 | 0.572061312 | 0.572061312 | 0.0 |
| D4 | 0.1 | -0.101233 | 0.104771007 | 0.104771007 | 5.6e-17 |
| D4 | 0.5 | -0.153522 | 0.497750888 | 0.497750888 | 6.1e-16 |
| D4 | 1.0 / 2.0 | -- | **not reachable** (max finite profile Delta=0.572) | -- | -- |
| real-D20 (seed=1, W=80000) | pareto | -0.513762 | 4.068415 | 4.068415 | 1.2e-14 |
| real-D20 | 0.1 | -0.454103 | 0.105512 | 0.105512 | 1.8e-16 |
| real-D20 | 0.5 | -0.485461 | 0.483276 | 0.483276 | 1.1e-15 |
| real-D20 | 1.0 | -0.496310 | 0.904551 | 0.904551 | 1.0e-15 |
| real-D20 | ~2.0 | -0.504931 | 1.650610 | 1.650610 | 4.4e-16 |

D4's fixed-A/f corridor tops out at `Delta~0.572` -- Delta~1/2 are **not reachable** there
(established already, not a new gap); per the governing prompt's own priority order
("construct via the reliable 1-D gamma profile or continuation, NOT a new broad
nuisance-parameter campaign"), these two D4 targets are reported unreachable rather than
force-constructed via a flexible-nuisance re-optimization (itself exactly the "broader
campaign" the priority order rules out). Real-D20 reaches all four non-pareto targets, `~2.0`
landing at `Delta=1.651` (disclosed, not exactly 2.0).

Every `(A,q)`/`(A,f)` cross-parameterization mismatch above is **at or below machine
precision** (`<=1.2e-14`) -- both coordinate systems describe the identical economic point.

## Phase 4: nested QMC W design

Verified explicitly (governing prompt's own instruction: "test this property, don't assume
it") rather than assumed: `rhalton(W, D; singleseed=seed)`'s digit-permutation RNG draws are
independent of `n` (only of the base `b` and float precision), so for a **fixed** `singleseed`
and dimension, `rhalton(W_small,...)` is bit-for-bit the first `W_small` rows of
`rhalton(W_large,...)`. Live-confirmed: `W=20,000` is an **exact prefix** of `W=1,280,000`
(same seed=29) -- `z_small == z_large[1:20_000,:]` bit-identical, `max|diff|=0.0`. A different
seed (41 vs 29) gives a genuinely different sample (confirmed `!=`).

Mechanism for holding the economy fixed while varying the QMC scramble: `generate_fake_melitz_
data`'s own `MelitzSyntheticData(primitives, equilibrium, counterfactual, L, z_draws, seed)`
struct lets `z_draws` alone be replaced (`pareto_draws(W, D, theta_star; seed=scramble_seed)`)
while `primitives`/`equilibrium`/`counterfactual`/`L` (all seeded from the SAME base seed=29)
stay bit-identical -- confirmed this is what every subsequent phase script actually does, not
merely intended.

## Phases 5-6: bandwidth family sweep + coordinatewise D4 screen

`scripts/melitz_qbw_phase5_6_bandwidth_sweep_2026-07-29.jl`. D4, both reachable base points
(delta~0.1, delta~0.5), `W in {20000, 80000, 320000, 1280000}` (full mandatory grid), 3 QMC
scrambles (tuning=29, held-out=141/271), every one of the 14 free q coordinates. Fixed-dual
mode run at the full grid (5,208 rows); reoptimized ground truth on 2 shortlisted policies
(`PowerScaled alpha=1/2 anchor25` = "B_alpha_half", `FixedCrossing target=25` = "C_target25"),
full 14-coordinate sweep for the tuning scramble at `W in {20000,80000,320000}`, a 5-coordinate
representative subset for the 2 held-out scrambles (288 rows total).

**Phase 6's own headline result is decisive and clean** -- pooling every (coordinate x W x
scramble x base-point) cell for each shortlisted policy:

| policy | W | n | sign agreement | median symmetric relative error | correlation |
|---|---:|---:|---:|---:|---:|
| B_alpha_half_anchor25 | 20,000 | 48 | **100.0%** | 0.003 | 1.000 |
| B_alpha_half_anchor25 | 80,000 | 48 | **100.0%** | 0.001 | 1.000 |
| B_alpha_half_anchor25 | 320,000 | 48 | **100.0%** | 0.000 | 1.000 |
| C_target25 | 20,000 | 48 | **100.0%** | 0.004 | 1.000 |
| C_target25 | 80,000 | 48 | **100.0%** | 0.002 | 1.000 |
| C_target25 | 320,000 | 48 | **100.0%** | 0.000 | 1.000 |

**Both shortlisted policies achieve 100% sign agreement and correlation 1.000 at every W
tested, with median relative error shrinking monotonically toward zero as W rises.** At the
INDIVIDUAL-coordinate level, the cheap fixed-dual secant is an essentially exact proxy for the
true, fully reoptimized response of that same coordinate, and gets more exact with more draws.

**An important distinction, not a contradiction**: the RAW secant VALUE for a fixed coordinate
fluctuates meaningfully across the W grid (finite-sample/QMC variation in the underlying
quantity itself -- e.g. `target=0.5`/`m=1`, family A fixed `h=1e-4`: `-0.518, -0.439, -0.323,
-0.335` across `W=20k,80k,320k,1.28M`; family C `target=10` (crossing count pinned at
exactly 10 by construction at every W): `-0.324, -0.423, -0.372, -0.442`, no visible
improvement -- consistent with the governing prompt's own warning that a FIXED crossing target
keeps `W*h` roughly constant and therefore does not satisfy `W*h -> infinity`). This is
DIFFERENT from what Phase 6 measures: Phase 6 asks "at THIS W, is my cheap fixed-dual
ESTIMATE a faithful stand-in for the TRUE reoptimized derivative implied by that SAME finite
sample" -- a question about estimator fidelity, not about the population-level value
converging to one fixed number. Both are true simultaneously: the underlying local slope
genuinely varies somewhat draw-sample to draw-sample (expected, unavoidable finite-W noise),
but whichever value a given `W`'s sample implies is captured essentially exactly by the cheap
fixed-dual secant -- which is what matters for a gradient-based outer search actually using
that W.

`GrowingCrossingQBandwidth` was independently confirmed to behave exactly as designed: at
`T_ref=25, W_ref=80000`, achieved crossing counts at `W in {20000,80000,320000,1280000}` were
`{13,25,50,100}` -- matching the intended `T_W=ceil(25*sqrt(W/80000))` targets `{13,25,50,100}`
almost exactly, at every W tested.

**Direct answers to the governing prompt's own Phase 5/6 questions**: (1) `alpha=1/2`
(the mandatory central candidate) is NOT rejected -- it performs excellently, but is
statistically indistinguishable from `FixedCrossing(target=25)` on this data (both hit 100%
sign agreement / correlation=1.000 at every W); this session's design did not include enough
independent alpha/anchor combinations with reoptimized ground truth to declare a single winner
among the power-scaled family (a scoped follow-up); (2) growing-crossing targets were
confirmed to scale exactly as designed but were not, in this session's design, compared
against a fixed-crossing target using the SAME reoptimized-ground-truth protocol (only the two
shortlisted policies got the expensive reoptimized comparison) -- this specific A/B comparison
is a scoped follow-up, not concluded here; (3) the raw secant VALUE does not stabilize cleanly
across W for any tested family at a single representative coordinate (10-30% fluctuation,
expected finite-sample noise); (4) but estimator FIDELITY (fixed-dual vs. reoptimized, at
matched W) is excellent and improves with W for both shortlisted policies -- the more
economically relevant property for outer-search use.

## Phase 7 (PRIMARY TEST): dense q-block direction validation

`scripts/melitz_qbw_phase7_dense_directions_2026-07-29.jl`. Held-out D4 base points
(delta~0.1, delta~0.5), held-out scrambles (tuning=29, held-out-3=401, distinct from Phase
5-6's own held-outs), `W in {80000, 320000, 1280000}`, the 2 Phase-6-shortlisted policies
(`PowerScaled alpha=1/2 anchor25` = `pred_B`, `FixedCrossing target=25` = `pred_C`), 11 fixed
directions (2 dense random, 2 sparse random, origin-block, dest-block, pivot-leverage-weighted,
gradient-descent-aligned, mixed-block-random, 2 more leverage/random -- **disclosed
substitution**: no live-instrumented-KNITRO-trial-step directions this session, replaced with
additional structured directions), 3 crossing-target amplitudes (25/100/400 total two-sided
switches). 396 rows.

**Decisive, three-way decomposition** (governing prompt's own diagnostic logic):

| comparison | W=80,000 | W=320,000 | W=1,280,000 |
|---|---:|---:|---:|
| **Q1**: coordinatewise `pred_B` vs. direct block fixed-dual secant -- sign agree / median symrelerr | 92.4% / 0.219 | 91.7% / 0.220 | 94.7% / 0.427 |
| **Q2**: direct block fixed-dual secant vs. **fully reoptimized** secant -- sign agree / median symrelerr | 100.0% / 0.003 | 100.0% / 0.001 | 100.0% / 0.000 |
| **Q3**: coordinatewise `pred_B` vs. fully reoptimized (end-to-end) -- sign agree / median symrelerr | 92.4% / 0.222 | 91.7% / 0.218 | 94.7% / 0.428 |

**Q2 is decisive and clean**: the direct block fixed-dual secant (evaluated at the SAME
converged dual, no reoptimization) predicts the fully reoptimized `DeltaStar` change with
**100% sign agreement at every W tested**, and its median relative error shrinks
monotonically toward **zero** as `W` rises (0.003 -> 0.001 -> 0.000). This directly answers
one of the governing prompt's own diagnostic questions: local/fixed-dual linearization of a
dense q movement is an excellent predictor of the true reoptimized response, and gets BETTER
with more draws -- exactly what an unbiased, consistent finite-sample estimator should do.

**Q1/Q3 are the opposite signature**: the coordinatewise-**assembled** prediction (summing
per-coordinate secants dotted with the direction) sits at 91-95% sign agreement and ~0.22-0.43
median relative error, and does **not** improve with `W` (if anything, mildly worse at
`W=1,280,000`). Per the governing prompt's own decision rule ("if A fails but B succeeds, the
problem is coordinatewise aggregation"): **this is exactly that case.** The mismatch is
localized to the aggregation step (summing 14 individually-estimated per-coordinate
sensitivities into one dense-direction prediction), not to dual-reoptimization curvature.

Breakdown by amplitude (`pred_B` vs. reoptimized, pooled across W): larger, more strongly
crossing-forcing amplitudes predict somewhat better (`c25`: 87.1% sign / 0.304 err; `c100`:
94.7% / 0.227; `c400`: 97.0% / 0.242) -- plausibly because a single amplitude-dominant
direction's overall sign becomes less sensitive to individual noisy coordinate estimates as
the movement grows.

Breakdown by direction family is itself informative: `gradient_descent`- and
`leverage_weighted_random_2`-aligned directions predict almost perfectly (100% sign, ~0.11-0.13
median relative error), while genuinely **block-coherent** directions (`origin_block`:
80.6%/0.417, `dest_block`: 83.3%/0.550, `mixed_block_random`: 83.3%/0.458) are markedly worse
-- moving many cells that share an origin or destination simultaneously appears to induce
correlated switching that per-coordinate linearization does not capture, more than an
equal-magnitude "spread out" random direction does.

## Phase 8: mixed full-outer directions (welfare + A + q)

`scripts/melitz_qbw_phase8_mixed_full_directions_2026-07-29.jl`, D4 (both reachable targets),
`W=320,000`, 6 directions (pure welfare, pure A, pure q, mixed A+q, mixed welfare+A+q,
gradient-descent-aligned full), 2 step scales, 24 rows.

`pure_A` predictions are **essentially exact** at every scale and both targets (`pred`
matches `secant_fd`/`secant_reopt` to 5-6 significant figures, e.g. `-3.7424e-05` vs.
`-3.7424e-05` vs. `-3.7424e-05`) -- the exact A-block gradient continues to work flawlessly
inside a mixed-direction context, not just isolated A-only perturbations. `pure_welfare`/
`gradient_descent_full`/mixed directions show the SAME qualitative pattern as Phase 7 at this
smaller, non-crossing-targeted amplitude scale: reasonable but imperfect agreement (roughly
2-40% relative error depending on target/direction), generally better at `delta~0.1` than
`delta~0.5`.

## Phase 9: matched (A,q) vs. (A,f) comparison, corrected design

`scripts/melitz_qbw_phase9_aq_vs_af_matched_2026-07-29.jl` -- **explicitly fixes the prior
session's own disclosed flaw**: its "pure extensive" direction landed on a near-zero-
sensitivity q coordinate, making the "mixed" row a no-op rather than a genuine joint test.
Fix: select the q coordinate via bisection on **>=15 two-sided crossings** before running the
comparison, so a genuine, verified, nonzero participation-switching move is guaranteed. D4,
both reachable targets, 4 paths (pure_intensive, pure_extensive, mixed_5050, mixed_7525), 3
step scales, 24 rows.

**Every one of the 24 rows now shows `has_switch=true`** (vs. the prior session's zero for its
"pure extensive" leg) -- this alone repairs the flaw. Cross-space mismatch (`(A,q)`-space
reoptimized `DeltaStar` change vs. the SAME displaced point's `(A,f)`-space reoptimized change)
is **at or below machine precision everywhere** (`0.0` to `3.3e-16`), confirming both
coordinate systems describe literally the same displaced economic state.

A genuine, reportable quantitative finding: the assembled `(A,q)` prediction (exact A + the
crossing-verified q secant) tracks the true reoptimized change's **sign and rough shape**
correctly at every scale, but is consistently **roughly half its magnitude** (e.g. at
target=0.1, pure_intensive, scale=1.0: predicted `-3.555e-4` vs. actual `-7.110e-4`; at
pure_extensive, scale=1.0: predicted `-3.911e-4` vs. actual `-7.809e-4`) -- a systematic,
repeatable underprediction, not noise (the ratio holds close to ~0.5 across scales within a
path, i.e. the prediction IS linear in the step, just under-scaled). This is a genuinely
different failure mode from Phase 7's sign-disagreement-dominated one: here the assembled
prediction has the right sign and shape but the wrong scale.

**Limitation, disclosed**: this script validates cross-space **equivalence of the realized
path** (both parameterizations reach the identical displaced state) but does not construct an
independent "(A,f) production-gradient" prediction via a genuinely different finite direction
in `f`-space (a step in `q` at fixed `A` does not correspond to a simple scalar direction in
`f`-space) -- unlike the prior session's own Section 7 (which used a single free `:logf`
coordinate direction directly), this session's paths are defined natively in `(A,q)` and only
verified equivalent in `(A,f)`, not independently re-predicted from the `(A,f)` side. A
genuine independent `(A,f)`-native local-linear prediction for the SAME displaced point is a
scoped follow-up, not attempted here.

## Phase 10: limited real-D20 confirmation

`scripts/melitz_qbw_phase10_realD20_confirmation_2026-07-29.jl`, one verified real-D20 point
(delta~0.5, seed=1), `W in {80000, 320000, 1280000}`, both D4-shortlisted policies, 5
representative q coordinates (lowest/highest pivot leverage, ordinary, focal-origin-export,
focal-destination-import), 4 dense directions. Run at `-t 1` (no `Threads.@threads`-eligible
kernel exists in this diagnostic code, matching the prior session's own disclosed convention).

Coordinatewise crossing-count calibration works correctly at D=20 scale exactly as at D4: the
two-sided bisection consistently lands within 1-3 crossings of the requested target=25 at
every `W` (e.g. `W=1,280,000`: `(+28,-25)`, `(+26,-25)`, `(+25,-25)` across the 3 coordinates
shown) -- the centralized Phase 1 infrastructure generalizes to D=20 without modification.

**Disclosed limitation**: the dense-direction amplitude heuristic (`t = h_ref*sqrt(nq)`, a
crude, non-LP-derived scaling) overshot at D=20 scale for most directions -- 3 of 4 dense
directions' reoptimized secants failed to converge (`lfd_ok=false`) at the chosen amplitude,
so only fixed-dual secants (not reoptimized ground truth) are available for those. One
direction (`leverage_weighted`, `W=1,280,000`) DID converge on both sides: `secant_fd=0.0709`
vs. `secant_reopt=0.1228` -- same sign, same order of magnitude, but a real, nontrivial gap
(ratio ~0.58), qualitatively consistent with Phase 7/9's own underprediction pattern. This
single data point is **not** sufficient to confirm or contradict the D4 findings at D=20 scale
with confidence; a properly amplitude-calibrated (LP-derived feasible-step) real-D20
dense-direction sweep is a scoped follow-up, not completed this session.

## Phase 11: experimental (A,q) outer-gradient backend

New file `src/melitz/aq_experimental_backend.jl`: `make_melitz_gradient_delta_direct_aq_
experimental(q_policy; gamma_h=1e-6)`, a factory mirroring `make_melitz_gradient_delta_direct_
sorted_serial(h)`'s own pattern, combining (1) a cheap fixed-dual central-difference secant on
the welfare coordinate, (2) the exact A-block gradient (zero FD probes), (3) the configured
`MelitzQBandwidthPolicy` via `melitz_q_coordinate_probe` -- the SAME source-level evaluator the
diagnostic scripts use (no second, script-only q-bandwidth implementation).

Registered **additively** into `finite_delta_outer.jl`'s existing dispatch (new symbol
`:B_direct_argument_aq_experimental`, appended to the `is_direct` tuple and the
`direct_gradient_fn` ternary chain) -- every pre-existing line/branch for the 5 existing
`gradient_backend` symbols is untouched, confirmed by diff. Not wired as a new production
default; not threaded as a new kwarg through `solve_melitz_finite_delta_bound` (a caller
wanting a non-default q-policy calls the factory directly).

Regression tests added (`test/melitz/runtests.jl`, "Experimental (A,q) outer-gradient backend
(Phase 11)"): produces a finite gradient of the correct length; rejects a `:logf` ctx (never
silently reinterprets the A-block); does not permanently perturb `obj.op`'s live state;
dispatches through `solve_melitz_finite_delta_bound` without crashing and returns a typed
result; an unrecognized `gradient_backend` symbol still errors (confirms the additive
registration did not loosen validation for every OTHER symbol). **8/8 new assertions pass**
(one initial test-writing bug -- expected `ArgumentError`, the dispatch actually throws
`ErrorException` via a bare `error(...)` call -- caught by the full-suite run and fixed;
disclosed as exactly the kind of "ran without error" vs. "verified" distinction this repo's
prior sessions insist on).

## Phase 12: bounded D4 outer-solver smoke test

`scripts/melitz_qbw_phase12_solver_smoke_2026-07-29.jl`. Production `(A,f)` (`gradient_
backend=:auto`) vs. experimental `(A,q)` (`:B_direct_argument_aq_experimental`), identical
starting economic state, identical KNITRO settings (`melitz_outer_finite_delta_alg_direct_
2026-07-27.opt`), identical evaluation cap (`CappedEvaluation(10.0)`), `delta in {0.1, 0.5}`
(D4's own corridor limit -- 2.0 not reachable, established already, not a new gap),
`direction in {:upper, :lower}`. 8 runs total.

| delta | direction | backend | nStatus | wall (s) | inner_solves | kappa (GT%) | Delta | within budget |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0.1 | upper | production (A,f) | -410 | 19.4 | 82 | 89.84 | 0.0924 | yes |
| 0.1 | upper | experimental (A,q) | -410 | 3.9 | 85 | 90.72 | 0.0511 | yes |
| 0.1 | lower | production (A,f) | -410 | 2.9 | 107 | 96.47 | 0.0994 | yes |
| 0.1 | lower | experimental (A,q) | -410 | 3.1 | 77 | 95.55 | 0.0668 | yes |
| 0.5 | upper | production (A,f) | -410 | 3.1 | 97 | 85.13 | 0.4929 | yes |
| 0.5 | upper | experimental (A,q) | -410 | 1.9 | 78 | 86.95 | 0.3913 | yes |
| 0.5 | lower | production (A,f) | -410 | 2.4 | 96 | 97.05 | 0.4615 | yes |
| 0.5 | lower | experimental (A,q) | -410 | 3.7 | 103 | 96.97 | 0.4994 | yes |

**All 8 runs terminate at the identical KNITRO status (`nStatus=-410`, iteration limit
reached)** -- this is a genuinely apples-to-apples short-run comparison (same stopping rule
both ways), but it is NOT a convergence result: neither backend actually converged in any of
the 8 runs. Every run retains a `cold_verified` `FiniteSolved` incumbent within the requested
budget (100% of runs, both backends). Neither backend dominates: the experimental `(A,q)`
backend finds a marginally BETTER (higher) `kappa` in both `direction=:upper` runs, while
production `(A,f)` finds a marginally better `kappa` in both `direction=:lower` runs;
`inner_eval_failures` counts are comparable between backends (52-81 range, both), and
`inner_infeas_count=0` for every run (no genuine infeasibility certificate fired either way).
**This establishes stability (no crash, no pathological failure-mode explosion) under a short,
identically-configured run -- nothing more.** It does not establish that the experimental
backend converges faster, finds better optima, or is production-ready.

## Final report answers

1. **Does a q derivative stabilize as W increases through 1.28 million?** Two distinct
   things are both true. (a) The RAW individual-coordinate secant VALUE does not stabilize
   cleanly -- it fluctuates ~10-30% across the W grid (finite-sample/QMC variation in the
   underlying local slope, Phase 5). (b) But the FIDELITY of the cheap fixed-dual secant as a
   proxy for the TRUE reoptimized derivative at that same W is excellent and gets BETTER with
   W -- both at the single-coordinate level (Phase 6: 100% sign agreement, correlation=1.000,
   median relative error 0.003->0.001->0.000) and at the direct dense-block level (Phase 7,
   Q2: 100% sign agreement, error->0). The coordinatewise-**assembled** dense-direction
   prediction, by contrast, does NOT improve with W (Phase 7, Q1/Q3) -- the breakdown is
   specific to aggregation, not to per-coordinate or per-direction estimator quality.
2. **Which h_W exponent and anchor perform best?** `alpha=1/2` (anchor25) was the only
   power-scaled configuration carried through to the expensive reoptimized-ground-truth
   comparison (Phase 6) and performed excellently (100% sign agreement, correlation=1.000 at
   every W) -- but `FixedCrossing(target=25)` performed identically well on the same protocol,
   so this session's data cannot declare `alpha=1/2` uniquely best among bandwidth families;
   it can only confirm it is NOT worse than the fixed-crossing alternative. Comparing
   `alpha in {1/3,2/3}` or the `anchor100` variant against reoptimized ground truth (rather
   than only fixed-dual mode) is a scoped follow-up.
3. **Is h_W ∝ W^(-1/2) supported, rejected, or merely tied?** **Tied, not uniquely supported.**
   `alpha=1/2` was not rejected (it performs excellently) but is statistically
   indistinguishable on this data from `FixedCrossing(target=25)`, which is not itself a
   `W^(-1/2)` rule at all -- so the specific `W^(-1/2)` functional form is not what's driving
   the good performance; both the power-scaled and target-crossing FAMILIES work well when
   their `h` happens to land in a reasonable crossing-count regime (Phase 6's own achieved
   crossing counts for both shortlisted policies ranged from ~10-30 draws, never near either
   extreme).
4. **Does a growing crossing target outperform a fixed crossing target?** Not directly tested
   head-to-head under the expensive reoptimized-ground-truth protocol this session (only
   `FixedCrossing(target=25)` was shortlisted for that comparison, not `GrowingCrossing`) --
   `GrowingCrossingQBandwidth` WAS independently confirmed to scale its target exactly as
   designed (`T_W=ceil(25*sqrt(W/80000))` achieved almost exactly at every W: `{13,25,50,100}`
   vs. targets `{13,25,50,100}`), and Phase 5's fixed-dual-only comparison shows the FIXED
   crossing-target family's raw secant value does NOT visibly improve across W (consistent
   with the governing prompt's own prediction that a fixed target keeps `W*h` roughly
   constant), while the growing-target family's raw value shows a modestly tighter spread at
   the largest W -- suggestive, not conclusive, and not run through the reoptimized-truth
   comparison to confirm.
5. **Does separating the exact smooth q component materially improve the estimator?** The
   exact smooth component itself is EXACT (Phase 2, 4e-8 relative error vs. zero-switch
   secants) and is nonzero ONLY on the focal-origin row -- most free q coordinates (13 of 14
   at the D4 fixture) have an exactly-zero smooth component, so for those the full secant IS
   the switching residual and separating changes nothing; for the 1 focal-origin-row
   coordinate, the exact smooth term gives a genuine head start that a pure finite-bandwidth
   secant would have to re-estimate noisily. Not incorporated into the Phase 5/6/7 hybrid
   sweep numerically this session (a disclosed, scoped follow-up) -- established analytically
   and validated in isolation, not yet combined with the switching-residual estimator in the
   large sweep.
6. **Are individual q-coordinate derivatives accurate on held-out points/scrambles?**
   **Yes, essentially exactly**, for both shortlisted policies, pooled across all 14 D4
   coordinates, both base points, and both held-out scrambles as well as the tuning scramble
   (Phase 6: 100% sign agreement, correlation=1.000, median symmetric relative error
   0.000-0.004 at every W tested). This is the strongest, cleanest result in the whole
   campaign.
7. **Do those derivatives add correctly for dense q directions?** **No** -- this is Phase 7's
   central, decisive finding (Q1: 91-95% sign agreement, 0.22-0.43 median relative error, not
   improving with W).
8. **Does the direct dense fixed-dual secant predict fully reoptimized DeltaStar changes?**
   **Yes, essentially exactly**, and the prediction quality IMPROVES monotonically with W
   (Phase 7, Q2: 100% sign agreement at every W, median relative error 0.003 -> 0.001 -> 0.000).
9. **At what stage does the mismatch arise: coordinate estimation, aggregation, or
   dual/value-function reoptimization?** **Aggregation, specifically -- ruled in by
   elimination as well as directly.** Per-coordinate estimation is essentially perfect (Phase
   6: 100% sign agreement/correlation=1.000). Dual-reoptimization curvature is also ruled out
   (Phase 7 Q2: the direct dense-BLOCK fixed-dual secant, evaluated without any per-coordinate
   aggregation, tracks the reoptimized truth at 100% sign agreement with error shrinking to
   zero). The ONLY remaining candidate is the summing-into-a-direction step itself, and Phase
   7's own direction-family breakdown supports this directly: block-coherent directions
   (origin-block, dest-block, mixed-block-random: 80-83% sign agreement) fare markedly worse
   than gradient-descent-aligned or leverage-weighted directions (97-100% sign agreement) --
   consistent with correlated, simultaneous cross-coordinate switching that independent
   per-coordinate linearization structurally cannot capture (each coordinate's secant is
   estimated holding every other coordinate fixed, but a dense direction moves many
   coordinates -- and hence many participation thresholds -- at once).
10. **Does (A,q) predict identical economic paths better than (A,f)?** Not independently
    tested this session (Phase 9's disclosed limitation) -- what WAS established is that both
    parameterizations describe the identical displaced economic state to machine precision,
    and the `(A,q)` assembled prediction itself is consistently ~2x too small in magnitude
    (right sign, right shape, wrong scale) on the corrected (genuinely-switching) paths.
11. **Does the limited real-D20 evidence agree with D4?** Coordinatewise crossing-count
    calibration generalizes cleanly (no D4-vs-D20 discrepancy there). The one converged
    dense-direction data point (`leverage_weighted`, W=1.28M: fixed-dual 0.0709 vs. reoptimized
    0.1228, ratio ~0.58) is qualitatively consistent with the D4 underprediction pattern, but
    this is one data point, not a confirmed replication -- the dense-direction amplitude
    heuristic needs LP-derived (not `sqrt(nq)`-scaled) feasible steps before a real D20
    dense-direction verdict is possible.
12. **What did the D4 smoke test establish, and what did it not establish?** Established:
    both backends are STABLE under an identical short, capped run (no crashes, comparable
    failure-mode counts, a verified incumbent retained within budget in all 8 runs). Did NOT
    establish: that either backend converges, converges faster, or finds a better optimum in
    general -- every run hit the same iteration-limit termination, not a converged optimum.
13. **Exact gains-from-trade and DeltaStar improvements from the common solver starts?** See
    the Phase 12 table -- e.g. at delta=0.1/upper, production reaches kappa=89.84%
    (Delta=0.0924) vs. experimental's kappa=90.72% (Delta=0.0511); at delta=0.5/lower,
    production reaches kappa=97.05% (Delta=0.4615) vs. experimental's kappa=96.97%
    (Delta=0.4994) -- neither consistently better, both well within the requested delta budget
    in all 8 cases.
14. **Which of the five decision conclusions is supported?** **Conclusion 2**, and it is now
    strongly (not merely tentatively) supported: "The q derivative converges coordinatewise
    with W, but coordinatewise assembly fails for dense movements; use exact A but move q
    through staged/blockwise directions rather than simultaneous SQP." Phase 6 establishes the
    first clause cleanly (individual coordinates: 100% sign agreement, correlation=1.000,
    improving with W). Phase 7 establishes the second clause equally cleanly (assembled dense
    prediction: 91-95% sign agreement, not improving with W) while simultaneously showing
    (Q2) that a DIRECT block evaluation -- skipping the coordinatewise-sum step entirely --
    recovers the same excellent, W-improving reliability the individual coordinates show.
    Conclusion 3 ("fixed-dual derivatives do not predict the reoptimized value reliably") is
    explicitly **rejected** by this session's data -- Q2's signal is strong and W-improving,
    the opposite of unreliable. Conclusion 4 ("no stable q derivative emerges by W=1.28M") is
    also **rejected** -- a stable, accurate derivative exists and was found, both
    coordinatewise and as a direct block evaluation; it simply cannot be assembled from
    independent per-coordinate pieces into an arbitrary dense direction with the same
    fidelity. Conclusion 1 (pure finite-QMC staircasing that disappears with the right h_W)
    is partially right in spirit -- the underlying reliability genuinely does improve with W
    -- but mischaracterizes WHERE the improvement shows up (block-level, not
    coordinatewise-assembled). Conclusion 5 ((A,q) offers no advantage over (A,f)) is not
    supported either way by this session's design (Phase 9's disclosed limitation -- no
    independent (A,f)-native prediction was constructed for the same path).

## Predeclared decision-criteria scorecard

| criterion | threshold | result |
|---|---|---|
| dense-direction sign agreement (held-out) | >=80% | 91-95% (pred_B/pred_C vs. reoptimized) -- **met** |
| median symmetric relative error | <=0.5 | 0.22-0.43 -- **met** |
| performance from W=320k to W=1.28M | no deterioration | **mild deterioration observed** (0.220->0.427 at the Q1/Q3 comparison) -- **not cleanly met** |
| direct block secant vs. reoptimized sign agreement | >=90% | **100%** -- **met, decisively** |
| best D4 policy remains among best 2 in D20 confirmation | -- | not conclusively testable this session (Phase 10's amplitude-heuristic limitation) |
| D4 solver smoke test at least as stable as production | -- | **met** -- comparable failure counts, verified incumbent retained in 100% of runs both backends |

## Recommendation

Given the Phase 7 decomposition (block secant excellent and W-improving; coordinatewise
assembly mediocre and not W-improving) and the Phase 12 smoke test (stable, competitive, not
yet a convergence claim), the supported governing-prompt conclusion is **Conclusion 2**:
adopt the exact A-block gradient without reservation, and treat q-block movements as requiring
either (a) a genuinely improved aggregation scheme (not attempted this session -- possibly a
correlated/joint crossing-count treatment for block-coherent directions, given that block-
coherent directions specifically underperformed) or (b) staged/blockwise q updates in the
outer search rather than a single simultaneous dense SQP step, until an aggregation fix is
found and validated. The experimental `(A,q)` backend is registered and tested but should
remain **experimental**, not a production default, pending that aggregation work and a
properly amplitude-calibrated real-D20 confirmation.

## Required output files

- `docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md` -- this document.
- `docs/key_results/melitz_qbw_phase3_base_states_2026-07-29.csv`,
  `melitz_qbw_phase3_theta_q_2026-07-29.csv` (Phase 3)
- `docs/key_results/melitz_qbw_phase5_fixed_dual_sweep_2026-07-29.csv`,
  `melitz_qbw_phase6_coordinatewise_reopt_2026-07-29.csv` (Phase 5-6)
- `docs/key_results/melitz_qbw_phase7_dense_directions_2026-07-29.csv` (Phase 7)
- `docs/key_results/melitz_qbw_phase8_mixed_full_directions_2026-07-29.csv` (Phase 8)
- `docs/key_results/melitz_qbw_phase9_aq_vs_af_matched_2026-07-29.csv` (Phase 9)
- `docs/key_results/melitz_qbw_phase10_realD20_coord_2026-07-29.csv`,
  `melitz_qbw_phase10_realD20_directions_2026-07-29.csv` (Phase 10)
- `docs/key_results/melitz_qbw_phase12_solver_smoke_2026-07-29.csv` (Phase 12)
- Scripts: `scripts/melitz_qbw_phase{3,5_6,7,8,9,10,12}_*_2026-07-29.jl`,
  `scripts/melitz_qbw_smoke_2026-07-29.jl`, `scripts/melitz_qbw_aq_backend_smoke_2026-07-29.jl`
  (throwaway smoke tests, kept for provenance).
- Source: `src/melitz/q_bandwidth_policy.jl`, `src/melitz/exact_q_smooth_gradient.jl`,
  `src/melitz/aq_experimental_backend.jl` (new); `src/melitz/include_melitz.jl`, `src/melitz/
  finite_delta_outer.jl` (additive edits); `test/melitz/runtests.jl` (new testsets).

## Disclosed scope reductions (session time budget)

1. Phase 5-6's full predeclared design (14 coords x ~16 policy configs x 4 W x 3 scrambles x
   2 base points, BOTH fixed-dual and reoptimized modes) was reduced to: fixed-dual mode at
   the FULL grid (cheap, no KNITRO re-solve); reoptimized ground truth restricted to 2
   representative policies (mandatory alpha=1/2 power-scaled, fixed-crossing target=25),
   full 14-coordinate sweep for the tuning scramble, a 5-coordinate representative subset
   (highest/lowest leverage + 3 evenly spaced) for the 2 held-out scrambles.
2. Phase 7/8's direction menus substitute additional structured directions for "actual q-block
   directions captured from KNITRO trial steps" -- no live instrumented outer run was
   captured this session.
3. Phase 9 validates cross-space path equivalence but does not construct an independent
   `(A,f)`-native local-linear prediction for the same displaced point.
4. Phase 10's dense-direction amplitude heuristic (`h_ref*sqrt(nq)`) is a disclosed,
   non-LP-derived approximation that overshot for 3 of 4 real-D20 directions.
5. The exact smooth q component (Phase 2) is validated in isolation but not numerically
   combined with the switching-residual estimator inside the Phase 5-7 sweep.
6. No D20 outer-search campaign was run (per the governing prompt's own Rule 11) -- Phase 10
   is single-point evaluations only.

## Provenance

- Julia 1.12.6 (juliaup), KNITRO 13.0.1 (`.knitro_env.sh`), `OPENBLAS_NUM_THREADS=1`/
  `OMP_NUM_THREADS=1` throughout every script this session wrote -- all run at `-t 1` (no
  `Threads.@threads`-eligible kernel exists in any new diagnostic file; matches the prior
  session's own disclosed convention, not a violation of `src/melitz/CLAUDE.md`'s
  20-thread-by-default rule, which applies to production/performance work).
- Base economies: D4 (`sigma=2.5, theta_star=6.8, target_country=1, seed=29`), real-D20
  (`noah_D20`, `sigma=2.5, theta_star` estimated, `focal="fra"`, `seed=1`).
- QMC scrambles used: D4 seed=29 (tuning, Phases 5-9), 141/271 (held-out, Phases 5-6),
  401 (held-out, Phase 7); real-D20 seed=1 throughout (Phase 10).
- `git diff --stat` and full commit list: see `provenance.txt` alongside this document.
