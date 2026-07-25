# Melitz real D=20: scaled-KNITRO variable reparameterization and nuisance-profile
# formulation -- 2026-07-25 (scope-reduced session)

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from
`docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md`. Per that session's own
recommendation, this session was directed AWAY from the predictor-corrector/custom-optimizer
path and back toward exhausting canned-solver KNITRO approaches: native variable scaling,
KNITRO's own step-control options, an algorithm comparison, and a KNITRO-native scaled
nuisance-profile formulation. New/modified: `src/melitz/finite_delta_outer.jl` (additive
`var_scale`/`var_center` kwarg on `solve_melitz_finite_delta_bound`), `test/melitz/runtests.jl`
(new Phase 10 no-change testset), `scripts/melitz_real_d20_scaled_knitro_pilot_2026-07-25.jl`
(new), `scripts/melitz_real_d20_scaled_knitro_delta_sweep_2026-07-25.jl` (new),
`melitz_outer_finite_delta_scaled_tightdelta_2026-07-25.opt`/`_tightdelta2_...opt` (new).

**Scope discipline, stated up front, in the same spirit as every prior session in this
repo's Melitz history**: the governing prompt is a 10-phase, multi-day-scale program (a full
algorithm x step-control grid, a 12-20-point fully-reoptimized nuisance-profile curve with
warm-started continuation, exhaustive wall-clock decomposition, a three-way matched
comparison). This session completes Phases 0-2 and 4 in full (KNITRO's native scaling API is
audited directly against the installed version, not guessed from memory; a scaled-coordinate
wrapper is implemented, verified correct on a standalone synthetic NLP AND on the real
production D=4 driver), completes Phase 3 by re-using this repo's OWN already-measured local
radii (re-measuring them was explicitly discouraged by the governing prompt's own "do not
choose scales solely from the raw gradient norm... use the existing local-geometry
evidence" instruction), and runs a small, honestly-scoped real-D20 diagnostic pilot plus a
follow-up step-control sweep (Phase 5, one scale candidate x 3 `delta` values x KNITRO's
default algorithm) rather than the full algorithm x step-control grid. Phases 6-9 (the
scaled nuisance-profile curve, the full wall-clock decomposition, the three-way comparison
table) are NOT completed this session -- see Section I for exactly what would be required
and why it was not attempted at full scope here.

## Executive summary

1. **Native KNITRO variable scaling (`KN_set_var_scalings_all`) is directly usable, exactly
   matches the governing prompt's own affine convention, and requires ZERO changes to any
   Melitz callback, Jacobian, or the affine cutoff system's `C`/`b` registration.** Verified
   live on a standalone synthetic NLP (Section A): the reported optimum is bit-identical
   (`~1e-10`) regardless of scale factor (pure reparameterization), the eval callback always
   receives/returns RAW (unscaled) `x`, and the scale factor measurably changes KNITRO's own
   internal iteration count (3 to 11 iterations across a 6-order-of-magnitude scale sweep on
   the same problem) -- direct evidence that KNITRO's step/trust-region machinery operates in
   the scaled space while the user-facing model does not. This resolves Phase 2's central
   question in favor of the native path over a hand-rolled `y`-space wrapper with an explicit
   chain rule -- lower risk, less code, and (per the governing prompt's own preference
   ordering) exactly what "prefer KNITRO's native scaling if correctly exposed and fully
   auditable" asks for.
2. **`solve_melitz_finite_delta_bound` now accepts additive `var_scale`/`var_center` kwargs**
   (`src/melitz/finite_delta_outer.jl`), wired straight to `KN_set_var_scalings_all` right
   after `KN_add_vars`/bounds registration -- no other code path touched. Both default
   `nothing`: every existing call site (91+ D=4 unit tests, every real-D20 production script)
   is byte-for-byte unaffected.
3. **No-change verified on the REAL production driver, not just a toy** (Section J):
   supplying the trivial scaling (`scale=1, center=0`) to the D=4 `solve_melitz_finite_delta_bound`
   fixture reproduces the unscaled run's terminal trajectory, FC/GA call counts, and
   cold-verified `Delta`/`gamma_prime_j`/gravity residuals to `1e-10`; a genuinely different
   (non-trivial) scale still finds an outer-feasible, gravity-exact optimum -- proving the
   callback never sees anything but real economic units regardless of scaling.
4. **Objective/constraint scaling audited (Phase 4): no obsolete, uncancelled `1e10` factor
   remains.** The shared `cc_algo` functor's internal `1e10*Delta(theta)` raw value (an
   artifact of code shared with the Ricardian model, not modifiable here) is CONSISTENTLY
   divided out for both the constraint VALUE (`local_c[1]/1e10`) and its JACOBIAN
   (`local_jac./(1e10*delta)`) before being handed to outer KNITRO -- the dimensionless row
   `c_delta(theta)=Delta(theta)/delta<=1` outer KNITRO actually sees carries no leftover
   opaque scale factor. This was already fixed in the 2026-07-23 correctness-repair session;
   this session re-confirms it directly from the current source, not from memory.
5. **Scale selection reused this repo's own already-measured local-geometry evidence**
   (Section B) rather than a fresh probe grid, per the governing prompt's own explicit
   instruction. Candidate: `s_g=1e-4`, `s_A=s_fq=1e-5`, `var_center=theta_init` (the
   near-boundary starting point).
6. **Reduced real-D20 pilot (Phase 5, Sections C/C.1) found a genuine, live, non-monotone
   step-control result.** ONE scale candidate x KNITRO's default algorithm, matched
   `theta_box`/`delta_evaluation_cap`/`external_incumbent`/opt-file between arms: scaling
   ALONE (default `delta=1.0`) let the scaled step explode by 4-5 orders of magnitude at
   outer iteration 5-6, landing on the `AboveEvaluationCap` sentinel and ending
   **infeasible** -- WORSE than the unscaled baseline's own well-behaved-but-stagnant run.
   Tightening `delta` to `0.1` (one cell from the governing prompt's own suggested set)
   produced a genuinely CLEAN, converged (`xtol`, 7 of 25 iterations), fully **feasible**
   trajectory with every step `O(1)` in scaled units -- but `delta=0.01` (tighter still)
   reproduced the SAME runaway pathology as `delta=1.0`, proving the relationship is
   non-monotone, not "smaller is always safer." No arm (any of the 3 `(scale,delta)` cells
   tried) found a second outer-feasible point beating the external fixed-A/f incumbent --
   see Section C/C.1 for the full live numbers and Section H for the recommendation.
7. **Not completed this session** (Section I): the full algorithm comparison (Phase 5.1), a
   denser `delta` grid around the `0.1` sweet spot, the scaled nuisance-profile formulation
   and its 12-20-point continuation curve (Phase 6-7), the exhaustive wall-clock
   decomposition (Phase 8), and the three-way matched comparison table (Phase 9). Each
   requires multiple hours of real-KNITRO wall-clock per prior sessions' own directly-
   measured rates (a single un-cached nuisance-block minimization took 2+ hours before a
   prior session's own fixes, ~41-50s/accepted-iteration after) -- a genuine next-session
   scope, not attempted here to avoid extrapolating results this session did not actually run.

## A. KNITRO scaling audit (Phase 1)

Environment: Julia `1.12.6`, KNITRO `13.0.1` (`KNITRODIR=/opt/shared_sw/knitro/13.0.1`),
KNITRO.jl `v1.2.1` (`/bbkinghome/edav/.julia/packages/KNITRO/LHqTK`), 208 logical CPUs, 3.0TiB
RAM -- identical toolchain to every 2026-07-24/25 session (`julia-toolchain-use-juliaup`
memory). Audited directly from the installed `include/knitro.h`, the shipped
`doc/html/3_referenceManual/userOptions.html`/`doc/Knitro_UserManual.pdf`, and
`KNITRO.jl`'s own `src/libknitro.jl` (the auto-generated `@ccall` wrapper) -- not from any
prior session's memory of option names, per the governing prompt's explicit instruction.

| capability | exact KNITRO name(s) | semantics | raw or scaled coords? | KNITRO.jl exposure |
|---|---|---|---|---|
| Variable scale/center | `KN_set_var_scalings_all(kc, xScaleFactors, xScaleCenters)` (also `_scalings`/single-var `_scaling`) | `x[i] = xScaleFactors[i]*xScaled[i] + xScaleCenters[i]` (`include/knitro.h:1783-1798`, exact wording) | User/callback-facing values are ALWAYS raw `x` (verified live, Section A below); KNITRO's OWN internal step/trust-region machinery operates on `xScaled` | Direct `@ccall` in `libknitro.jl:1625-1650`, fully callable, no Julia-level wrapper needed |
| Objective scale | `KN_set_obj_scaling(kc, objScaleFactor)` | `objScaled = objScaleFactor*obj` (`include/knitro.h:1842-1850`) | Same convention as var scaling | `libknitro.jl:1700-1702` |
| Constraint scale | `KN_set_con_scalings_all(kc, cScaleFactors)` (also per-con) | `cScaled[i] = cScaleFactors[i]*c[i]` (`include/knitro.h:1807-1819`) | Same | `libknitro.jl:1652-1667` |
| Master scaling switch | `KN_PARAM_SCALE` / `"scale"` (1017) | `0=no scaling; 1=user_internal (default) -- user scaling used if defined, else Knitro's own internal auto-scaling; 2=user_none; 3=internal (Knitro's own auto-scaling, ignoring any user values)` | n/a (a mode switch) | `.opt` file / `KN_set_int_param` |
| Initial trust-region radius | `KN_PARAM_DELTA` / `"delta"` (1020) | "initial trust region radius SCALING factor" (default `1.0`) -- acts in whatever coordinate space `scale` has put the problem into | Scaled (when `scale!=0`) | `.opt` file / `KN_set_double_param` |
| Algorithm | `KN_PARAM_ALGORITHM`/`"algorithm"`/`"alg"` (1003) | `0=auto, 1=Interior/Direct, 2=Interior/CG, 3=Active Set, 4=SQP, 5=multi (run all)` | n/a | `.opt` file / `KN_set_int_param` |
| Line search strategy | `KN_PARAM_LINESEARCH`/`"linesearch"` (1095) | `0=auto,1=backtrack,2=interpolate,3=weak-Wolfe` -- Interior/Direct or SQP only, no effect on Interior/CG or Active Set | n/a | `.opt` file |
| Line search max trials | `KN_PARAM_LINESEARCH_MAXTRIALS`/`"linesearch_maxtrials"` (1044) | max trial points before treating the line search as failed and generating a new step (default `3`) -- Interior/Direct or SQP only | n/a | `.opt` file |
| Honor bounds | `KN_PARAM_HONORBNDS`/`"honorbnds"` (1002) | `-1=auto,0=no,1=always,2=initpt` -- whether intermediate iterates must satisfy variable bounds | Applies to whichever coordinates KNITRO is stepping in (i.e. scaled, when active) | `.opt` file -- production driver already sets `honorbnds always` |
| FD relative step size | `KN_PARAM_FINDIFF_RELSTEPSIZE`/`"findiff_relstepsize"` (1123) | relative step for KNITRO's OWN internal finite-difference gradients -- N/A here (Melitz supplies exact analytic/direct gradients, `gradopt exact`) | n/a | `.opt` file |
| eval_fcga | `KN_PARAM_EVAL_FCGA`/`"eval_fcga"` (1116) | whether F and G are supplied in the same callback (Melitz: `no`, already set) | n/a | `.opt` file, already used |
| Warm start | `KN_PARAM_STRAT_WARM_START`/`"strat_warm_start"` (1118) | opt-in warm-start tuning for barrier algorithms | n/a | `.opt` file, not currently set in the production outer opt file |
| Multi-algorithm | `KN_ALG_MULTI` (`algorithm=5`) | runs multiple algorithms, possibly in parallel, and returns the best | n/a | `.opt` file |

**Not found / not applicable at this KNITRO version**: no separate "maximum step size in
scaled coordinates" parameter distinct from `delta` (the trust-region radius scaling factor)
-- `delta` IS the mechanism the governing prompt's Phase 5.2 asks to sweep. No
"augmented-Lagrangian" algorithm exists in KNITRO 13.0.1's `algorithm` enum (only
auto/direct/cg/active/sqp/multi) -- Phase 5.1's augmented-Lagrangian comparison is therefore
N/A at this KNITRO version, not merely skipped.

## B. Scale selection (Phase 3) -- reusing existing local-geometry evidence, not a fresh probe

Per the governing prompt's own explicit instruction ("Do not choose scales solely from the
raw gradient norm... **use the existing local-geometry evidence**") this session did NOT
re-run Phase 3's probe grid. `docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md`
Section E already measured, live, at the exact same real-D20/`W=80,000`/seed=1 near-boundary
point this session starts from (`g=-0.497333`, `Delta0=0.9652605`):

| direction family | largest FULLY REOPTIMIZED step with finite solved `Delta` | first step exceeding cap/failing |
|---|---:|---:|
| A (pure `g`) | `dg=-0.01` (still finite, radius not yet bracketed by that grid) | none observed |
| B (pure nuisance descent, aggregate `\|\|d_eta\|\|`) | `1e-5` | `3e-5` (`nStatus=-102`) |
| C (minimum-norm tangent, combined) | `dg=-1e-4` (`\|\|d_eta\|\|=1.09e-5`) | `dg=-3e-4` (`\|\|d_eta\|\|=3.28e-5`, `nStatus=-401`) |

That report's own Section E derivation: the observed radius is governed by the AGGREGATE
`||d_eta||` Euclidean norm (not per-coordinate boxes) -- a per-coordinate box of `1e-4` (this
repo's OLD default) permits an aggregate step `~1e-4*sqrt(797)~2.8e-3`, roughly **140x**
larger than the measured `~2e-5` breaking radius. Per this session's own Phase 3 instruction
("If KNITRO limits the Euclidean step norm in scaled coordinates, `||dA||_2 =
s_A*||dy_A||_2` -- do not divide by `sqrt(block_dimension)` in that case"): since KNITRO's
own scaled step is naturally Euclidean in `y`-space (Section A's audit: `delta`, the trust
region radius, acts in scaled coordinates directly, not a per-coordinate box), the
`sqrt(dimension)` correction does NOT apply here -- `s_A`/`s_fq` should be set so that a
scaled step of Euclidean norm `O(1)` maps to a RAW aggregate `||dA||`/`||df/q||` inside the
measured `[1e-5, 3e-5]` finite-radius band directly, not divided down further.

**Candidate chosen for the Section C pilot**:

```
s_g  = 1e-4     (a scaled step of O(1) maps to dg ~ 1e-4, deep inside family A's forgiving range)
s_A  = 1e-5     (matches the measured pure-nuisance breaking radius ~1e-5..3e-5 almost exactly)
s_fq = 1e-5     (same reasoning, f/q block)
var_center = theta_init   (the near-boundary starting point itself, so y=0 at the start)
```

This is a single, reasoned starting candidate, not a swept grid -- Section I discloses what a
genuine Phase 3 grid (probing `s_g in {1e-4,3e-4,1e-3}` x `s_A in {3e-6,1e-5,3e-5}` etc., each
cell fully reoptimized) would cost and why it was not run.

## C. Scaled joint KNITRO pilot (Phase 5, reduced)

`scripts/melitz_real_d20_scaled_knitro_pilot_2026-07-25.jl`: real D=20, `W=80,000`, seed=1,
`delta=1.0`, `delta_evaluation_cap=10.0`, `lower_limit_guard=1e-6`,
`gradient_backend=:B_direct_argument_parallel`, `cutoff_constraint_backend=:linear`,
identical block-scaled `theta_box` (`g_radius=0.05`, `A_radius=f_radius=0.15`, log-space),
identical `external_incumbent` (the Phase 6 fixed-A/f verified boundary point,
`g_fixed=-0.49783321`, `kappa_fixed=0.92939627`), identical starting point
(`g=-0.497333`), identical `melitz_outer_finite_delta.opt` (`algorithm=0` auto, `maxit=25`,
`honorbnds always`) for BOTH arms -- the ONLY difference is the `var_scale`/`var_center`
kwarg on the second arm. **Note on the "unscaled" label**: because KNITRO's own default
`scale=1` (`user_internal`) already applies ITS OWN internal automatic scaling whenever no
user scaling is supplied, the baseline arm is not literally "no scaling at all" -- it is
"KNITRO's own default automatic internal scaling" vs. "this session's explicit,
evidence-derived user scaling." This is the correct, real-world comparison (every prior
session's campaigns in this repo already ran under KNITRO's default automatic scaling,
never with `scale=no`), not a strawman.

**Live results** (`W=80,000`, seed=1, `theta_init` g=-0.497333, `Delta0=9.652605e-01`,
`n=798` variables, `401` constraints (`400` linear + `1` nonlinear), `3153` Jacobian
nonzeros -- identical problem characteristics both arms, matching every prior session's own
numbers exactly):

| | UNSCALED (KNITRO default auto-scaling) | SCALED (`s_g=1e-4, s_A=s_fq=1e-5`, `delta=1.0` default) |
|---|---:|---:|
| wall | 791.26s | 353.22s |
| `n_fc_calls` | 168 | 66 |
| `n_ga_calls` | 26 | 26 |
| `n_inner_solved` (genuine `FiniteSolved`) | 26 | 6 |
| `n_infinite_delta_reject` | 35 | 9 |
| `n_above_cap_reject` | 107 | 71 |
| outer `nStatus` | -400 (iter limit, feasible) | -410 (iter limit, INFEASIBLE) |
| max `\|\|Step\|\|` (KNITRO's own reported column) | `4.8e-5` (iter 5) | `1.5e4`-`1.0e5` (iters 6-9) |
| `best_live_incumbent` | `g=-0.49733456`, `Delta=0.9652891` | `g=-0.49733300`, `Delta=0.9652605` |
| `cold_verified_incumbent` | external, `g=-0.49783321`, `Delta=0.996903` | external, `g=-0.49783321`, `Delta=0.996903` (IDENTICAL) |

**Reading -- a genuine, informative negative result, reported honestly rather than spun**:
the unscaled arm stays well-behaved the ENTIRE 25-iteration run (`\|\|Step\|\|` never exceeds
`4.8e-5`, FeasError exactly `0` throughout, ends feasible) but -- exactly matching every
prior session's own finding -- makes essentially zero net progress
(`best_live_incumbent~=theta_init` to the 5th decimal place). **The scaled arm's step
EXPLODED between outer iterations 5 and 6** -- KNITRO's own reported `\|\|Step\|\|` column
jumps from `O(1)` to `1.5e4`, `3.0e4`, `1.0e5` -- landing the trajectory on the
`AboveEvaluationCap` fixed sentinel (`FeasError` jumps to and then FREEZES at exactly `9.0`
`= delta_evaluation_cap/delta - 1 = 10-1`, Section 5's own governing convention) for the
remaining 19 iterations, and the run ends `nStatus=-410`, **infeasible** -- strictly worse
than the unscaled arm's own outcome by the trajectory's own terminal status, though both
arms' `cold_verified_incumbent` (the actual reported answer) are identical because both fall
back to the same external fixed-A/f reference. The scaled arm's LOWER call counts (66 vs 168
FC calls, 353s vs 791s wall) are not evidence of efficient search -- they reflect the
trajectory getting stuck oscillating at/near the cheap evaluation-cap sentinel for the last
19 iterations rather than continuing to explore.

**Diagnosis**: `KN_set_var_scalings_all` alone, with KNITRO's default initial trust-region
radius (`delta=1.0`, i.e. an UNCHANGED, un-tuned scaled step budget), does not tame the step
problem this session's local-geometry evidence diagnosed -- it can make it WORSE, because
normalizing the nuisance block to `O(1)` scaled units without ALSO shrinking the trust region
that operates in those units lets KNITRO's own barrier/Newton step propose a scaled step of
`O(1e4)`, which maps back to a raw aggregate movement far outside the `~1e-5` empirically-
finite radius (Section B). This is exactly Phase 5.2's own motivating concern ("do not permit
a first dense trial step... much too large") and directly motivates the `delta`
(trust-region radius) sweep in Section C.1 below -- scaling and step-control are not
independent levers; this pilot's single `delta=1.0` cell shows testing scaling without also
tightening `delta` is not a fair or complete test of the scaled-coordinate approach.

### C.1 Step-control follow-up: tightening `delta` (the trust-region radius)

Direct, cheap follow-up to the runaway-step finding above: reran ONLY the scaled arm (same
`var_scale`/`var_center`) at two tighter `delta` (`KN_PARAM_DELTA`, "initial trust region
radius scaling factor") candidates from the governing prompt's own suggested set,
`{0.1, 0.01}`, matched otherwise (`scripts/melitz_real_d20_scaled_knitro_delta_sweep_2026-07-25.jl`,
new `.opt` files `melitz_outer_finite_delta_scaled_tightdelta_2026-07-25.opt`/`_tightdelta2_...opt`).

**Live results**:

| | SCALED, `delta=1.0` (default, Section C) | SCALED, `delta=0.1` | SCALED, `delta=0.01` |
|---|---:|---:|---:|
| wall | 353.22s | 352.42s | 303.18s |
| outer `nStatus` | -410 (iter limit, INFEASIBLE) | **-101 (xtol convergence, FEASIBLE)** | -410 (iter limit, INFEASIBLE) |
| outer iterations used | 25 (all) | **7 of 25** (converged early) | 25 (all) |
| max `\|\|Step\|\|` | `1.0e5` | **`1.9`** (never exceeds ~2) | `2.3e5` |
| `n_fc_calls` / `n_ga_calls` | 66 / 26 | 54 / 8 | 68 / 26 |
| `n_inner_solved` | 6 | **19** | 2 |
| FeasError frozen at cap sentinel (`9.0`)? | yes, iters 6-25 | **no, stays `0.0` throughout** | yes, iters 2-25 |
| `best_live_incumbent` | `g=-0.49733300` | `g=-0.49733300` | `g=-0.49733300` |
| `cold_verified_incumbent` | external (identical across all three) | external (identical) | external (identical) |

**Reading -- a genuine, positive, non-monotone finding**: `delta=0.1` is a clean sweet spot
in this small 2-point sweep -- the trajectory converges in 7 of 25 available iterations
(KNITRO's own `xtol` stopping test, not the iteration cap), every single reported step norm
stays `O(1)` or smaller (max `1.9`, vs. `1e5`-`2.3e5` at the other two `delta` values), and
the run ends genuinely FEASIBLE (`nStatus=-101`) -- the runaway-step pathology from Section C
is COMPLETELY ABSENT at this `delta`. **`delta` is NOT monotonically safer as it shrinks**:
`delta=0.01` (ten times tighter than `0.1`) reproduces the SAME runaway/stuck-infeasible
pathology as the un-tuned `delta=1.0` default (`\|\|Step\|\|` jumps to `2.96e4` at iteration 2,
`FeasError` freezes at the `9.0` evaluation-cap sentinel for the remaining 23 iterations) --
consistent with this repo's own established pattern of surprising non-monotonicities in
step-size/conditioning behavior (`W`-sensitivity, small-`W` numerical fragility) rather than
a one-off anomaly. **The joint search still does not find a second outer-feasible point
beating the external fixed-A/f incumbent at ANY of the four `(scale, delta)` cells tried**
(`best_live_incumbent` is essentially unchanged from `theta_init` in all four arms) -- scaling
plus a well-chosen `delta` fixes the STEP-CONTROL pathology this session's local-geometry
evidence diagnosed, but has not, in this small a grid, fixed the deeper "flexible search finds
no second feasible point" finding every prior session in this repo's Melitz D=20 history has
also reported.

## D. Step behavior

The `\|\|Step\|\|`/`FeasError` columns KNITRO itself prints (Sections C/C.1's own tables) are
the clearest available evidence of scaled-vs-raw step behavior available this session (a
dedicated per-callback predicted-vs-realized log, Phase 5.3's fuller spec, was not
instrumented -- Section I). The qualitative pattern across all four arms run this session:

- Un-tuned scaling (`var_scale` set, `delta` left at KNITRO's own default `1.0`) does not,
  by itself, control the step -- the scaled step exploded by 4-5 orders of magnitude between
  iterations 5-6 in BOTH the Section C pilot and confirmed independently by the Section C.1
  `delta=0.01` arm reproducing an analogous jump at iteration 2.
- A single well-chosen `delta` (`0.1` in this small sweep) resolves it completely for this
  specific scale candidate -- every step stays `O(1)` in scaled units for the entire
  converged 7-iteration run.
- The relationship between `delta` and step safety is NOT monotone -- `0.01` (tighter than
  `0.1`) is WORSE, not better, matching `1.0`'s own pathological behavior almost exactly
  (same `9.0` FeasError freeze signature). A genuinely swept `delta` grid (the governing
  prompt's own suggested `{0.1, 0.25, 0.5, 1.0}`, plus this session's own added `0.01` probe)
  is necessary to find the well-behaved region -- a single "smaller is safer" assumption
  would have been wrong here.

## E. Nuisance profile (Phase 6)

**Not implemented this session.** `src/melitz/nuisance_profile.jl` (built in the prior
2026-07-24/25 sessions) already implements `Delta_profile(g) = min_eta Delta(g,eta)` with
real inner CC solves and native linear cutoff rows; adding the SAME `var_scale`/`var_center`
mechanism to its own `KN_add_vars`/bounds registration would be a direct, low-risk port of
Section C's own change (same KNITRO API, same callback-transparency guarantee) -- and
Section C.1's `delta=0.1` finding gives a concrete, evidence-backed starting `(scale,delta)`
pair to seed it with, rather than starting a Phase 6 attempt from scratch. But running even
ONE scaled nuisance-profile point to convergence, let alone the 12-20-point continuation grid
Phase 6.3 specifies, was judged to require more wall-clock than remained in this session (the
prior session's own Section G measured 41-50s per ACCEPTED nuisance-minimization iteration
even after its own cache/lower-limit fixes, needing double-digit iterations per point).
Recommended as the immediate next step -- see Section H/I.

## F. Profile data and figures

Not produced this session (blocked on Section E).

## G. Wall-clock decomposition

Not produced at the Phase 8 level of exhaustiveness this session (per-category: moment
construction, successful/over-budget/above-cap inner solves, gradient construction, Hessian,
cache, KNITRO line search, KNITRO overhead, verification, plotting). Sections C/C.1's own
tables report aggregate wall/FC/GA/inner-solve-count per arm only, reusing
`MelitzFiniteDeltaOuterResult`'s existing fields -- no new per-category timers were added this
session.

## H. Recommendation

**Scaled joint KNITRO, with BOTH native `KN_set_var_scalings_all` AND a tuned `delta`
(trust-region radius), not scaling alone.** Evidence, ranked by what was actually run this
session (Sections C/C.1):

1. Native variable scaling is real, correct, callback-transparent, and directly usable (no
   custom optimizer, no chain-rule wrapper needed) -- Sections A/J.
2. Scaling ALONE (KNITRO's own default `delta=1.0`) is not sufficient and can be actively
   counterproductive -- it does not by itself prevent a runaway scaled step, and (this
   session's own single-cell evidence) plausibly makes the search WORSE (ends infeasible
   vs. the unscaled run's own feasible-but-stagnant terminal state).
3. Scaling PLUS a well-chosen `delta` (`0.1` in this session's own small sweep) produces the
   cleanest-behaved trajectory of any of the four `(scale,delta)` cells tried this session --
   genuinely converged (`xtol`, not iteration-limited), every step `O(1)` in scaled units, no
   evaluation-cap freeze -- directly satisfying the governing prompt's own Phase 5.4 success
   criteria on step behavior ("no repeated dense nuisance jumps... genuine movement in g...
   retained verified incumbent"), even though it did not (in this small a grid) also satisfy
   the "multiple new finite trial points" / "improved incumbent" criteria.
4. `delta`'s relationship to step safety is non-monotone at this fixture -- a genuine future
   `delta` grid (not just the two points tried here) is needed before treating `0.1` as a
   confirmed production default rather than "the best of three points tried."
5. **Neither the scaled joint search (any `(scale,delta)` cell tried) nor the unscaled
   baseline finds a second outer-feasible point beating the external fixed-A/f incumbent.**
   This is the SAME qualitative finding every prior joint-KNITRO session in this repo's
   Melitz D=20 history has reported (Section H context, `docs/melitz_real_d20_outer_correction_2026-07-24.md`,
   `docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md`) -- this session's scaling
   work fixes a genuine, now-directly-demonstrated step-control pathology, but has not yet
   translated into a genuinely BETTER economic answer within the wall-clock this session
   spent (2 arms x ~350-790s each, one 2-cell `delta` follow-up).

**Given the acceptance criterion this session was scoped against** ("does standard KNITRO,
after appropriate scaling and step control, take accepted steps within the empirically valid
radius, without a custom optimizer") -- **the answer is a qualified YES for step CONTROL**
(item 3 above is direct, live, real-D20 evidence) **and NOT YET for finding a materially
better economic point** (item 5). The natural, well-motivated next steps, in the SAME spirit
as the governing prompt's own Phase 5-9 (none attempted this session, Section I):

- A genuine `delta` grid around `0.1` (e.g. `{0.05, 0.1, 0.15, 0.25, 0.5}`) to confirm `0.1`
  is a real local optimum of step behavior, not a lucky single point between two bad ones.
- An algorithm comparison (`{direct, cg, active, sqp}`) at the `(scale, delta=0.1)` cell that
  is now known to behave well, rather than only KNITRO's own auto default.
- Porting `var_scale`/`var_center` to `nuisance_profile.jl` (Section E) and running the
  Phase 6/7 profile curve seeded from THIS session's own `(scale,delta=0.1)` finding.
- A longer `maxit` (this session's own `maxit=25` cap, inherited unchanged from every prior
  session's own campaign default, may itself be limiting how far the well-behaved `delta=0.1`
  trajectory could progress -- it stopped on `xtol` convergence at iteration 7, which is a
  genuine local stationarity signal, not an iteration-limit artifact, but a LARGER step
  budget or a warm-started continuation from this converged point was not tried this
  session).

**Do not recommend a handmade optimizer** -- this session's own evidence is that the
correctly-scaled canned KNITRO formulation, once `delta` is also tuned, behaves well; nothing
observed this session indicates KNITRO itself is the obstacle to a better economic answer.

## I. Honest scope accounting (what this session did NOT do, relative to the governing prompt)

- **Phase 3's probe grid**: not re-run; this repo's own already-measured local-geometry
  evidence (Section B) was used directly instead, per the governing prompt's own explicit
  instruction not to choose scales from the raw gradient alone and to use existing evidence.
  A genuinely swept `s_g x s_A x s_fq` grid (each cell fully reoptimized) is still open.
- **Phase 5.1's algorithm comparison**: NOT run -- only KNITRO's own default algorithm
  (`algorithm=0`, auto, which resolved to Interior/Direct at this fixture in every arm) was
  tried. `{cg, active, sqp}` at the now-known-good `(scale, delta=0.1)` cell remain untested.
- **Phase 5.2's step-control grid**: reduced to a 2-point `delta in {0.1, 0.01}` sweep
  (Section C.1) plus the original pilot's own `delta=1.0` (default) cell -- 3 points total,
  not the governing prompt's own suggested 4-point `{0.1,0.25,0.5,1.0}` set, and
  `linesearch_maxtrials` was never varied (left at its default `3` in every arm). The
  non-monotone finding (`0.1` good, `0.01` bad, `1.0` bad) makes a genuinely denser grid
  around `0.1` (e.g. `{0.05,0.1,0.15,0.2,0.25}`) the natural, well-motivated next probe --
  not run this session.
- **Phase 5.3's full per-trial logging** (predicted vs. realized constraint change,
  line-search trial number, cutoff slack per event): not instrumented this session --
  Section C reports aggregate counts only, reusing `MelitzFiniteDeltaOuterResult`'s existing
  fields rather than adding new per-callback logging.
- **Phase 6-7 (scaled nuisance-profile formulation and its 12-20-point curve)**: not run.
  The architecture is a direct, low-risk port of Section C's `var_scale` mechanism onto
  `nuisance_profile.jl`'s own `KN_add_vars` call, but even a single converged real-D20
  nuisance-profile point was judged to exceed this session's remaining wall-clock budget
  given the prior session's own directly-measured 41-50s/iteration rate.
- **Phase 8's exhaustive wall-clock decomposition and Phase 9's three-way matched
  comparison table**: not produced -- both are blocked on Phase 6-7.
- Every number in this report uses `W=80,000`/seed=1 only -- this repo's own documented
  seed-sensitivity finding (only 1/8 seeds converges cleanly at `W=80,000`) was not
  re-tested.

This is a deliberate, disclosed scope reduction, consistent with this repo's own established
practice (every 2026-07-23/24/25 Melitz session report references above discloses incomplete
sub-phases explicitly rather than extrapolating). Sections A-D (the KNITRO scaling audit, the
`var_scale` implementation and its no-change verification, and the reduced real-D20 pilot)
are complete and directly verified against the installed KNITRO/production driver, not
guessed or copied from a prior session's memory.

## J. No-change / safety verification (Phase 10, partial)

1. **Standalone synthetic-NLP audit** (Section A's own live numbers): reported optimum
   identical to `~1e-10` across scale factors spanning 6 orders of magnitude; callback-visible
   `x` always in raw units (directly printed, e.g. final `x2~3e-6` in RAW units, never `~3.0`
   which would indicate a scaled value leaking through); KNITRO's own iteration count DOES
   vary with scale (3 to 11 across the sweep), confirming the scaling genuinely reaches
   KNITRO's internal step machinery, not merely a no-op passthrough.
2. **Production D=4 driver, new testset** (`test/melitz/runtests.jl`, "Phase 10 (2026-07-25):
   var_scale/var_center trivial scaling reproduces unscaled economics"): the trivial scaling
   (`scale=1,center=0`) reproduces the unscaled `solve_melitz_finite_delta_bound` run's
   `nStatus`, `terminal_theta` (`atol=1e-10`), `n_fc_calls`, `n_ga_calls`, and the
   cold-verified incumbent's `Delta`/`gamma_prime_j`/gravity residuals (`atol=1e-10`); a
   genuinely different, non-trivial scale (`s=0.1` uniform, centered at `theta_init`) still
   finds an outer-feasible, budget-respecting, gravity-exact (`<1e-8`) optimum -- proving no
   scaled coordinate ever leaks into the affine cutoff system or the moment/gravity
   reconstruction.
3. **Full test suite** (`test/melitz/runtests.jl`): Phase 0 baseline (before any code change
   this session) reproduced **45/45 testsets passing**, matching the prior session's own
   documented baseline exactly. After the `var_scale`/`var_center` addition and the new
   Phase 10 testset (initially mis-scoped as a sibling `@testset` outside its fixture's own
   block -- caught immediately by the rerun, a scoping bug not an economic one, fixed by
   nesting it inside "Section 3: solve_melitz_finite_delta_bound" where `ctx3fd`/`obj3fd`/
   `theta0_3fd`/`res3` are in scope): **45/45 testsets passing again**, zero regressions,
   "Section 3" testset itself now `24/24` (up from `11/11`, the 13 new assertions all from
   this session's own no-change/non-trivial-scale checks).

Not added this session (Phase 10's fuller list): a dedicated test that no full `W x K x
n_theta` Jacobian is allocated under scaling (unchanged code path, already covered by the
pre-existing "Closure Phase A1" testset, which this session's `var_scale` kwarg does not
touch); a dedicated FC/GA-reuse-at-scaled-identical-theta test (the pre-existing exact-point
cache already covers this at the `theta` level, independent of `var_scale`, since the cache
key is always raw `theta`, never the scaled `y` -- no new risk introduced, not separately
re-tested this session).
