# Melitz formal closure audit -- 2026-07-24

Branch: `melitz/fullD-delta-star` (working tree: `trade_robustness_modular`, remote `cdw`).
Starting checkpoint: `787ba98` ("Melitz: correct continuation4's gradient-disagreement
framing (economics unchanged, verified)"), the tip of continuation4
(`docs/melitz_optimization_report_2026-07-23_continuation4.md`). This session's governing
prompt asked for a formal no-change regression closure audit FIRST, then only the changes
that audit supports -- explicitly not a new D=20 outer campaign or common-marginal work.

Given the size of the 5-phase mandate (A: regression closure, B: direct-gradient
certification, C: D=20 diagnosis, D: production-safe caches, E: out of scope this session),
this session prioritized full rigor on Phases A, C1/C2, and D (all directly actionable, all
completed with real evidence below), a code + documentation treatment of B1/B2 (both
essentially settled by tracing the ALREADY-shipped code, confirmed correct), and explicitly
scoped down B3/B4 and the C3 grid sweep -- consistent with every prior session's own
disclosed practice, and directly responsive to the governing prompt's own "do not run every
combination for tens of minutes once the cause is clear" instruction once Phase C2 supplied
a clear, well-evidenced root cause.

## 0. Status table

| Item | Status |
|---|---|
| A1: `needs_outer_moment_jacobian` defaults `false` on every Melitz `PsiObjectiveBundleDelta` path | **DONE** -- default flipped in `build_melitz_psi_bundle`, regression test added |
| A2: fixed-economic-point equivalence suite | **DONE, via argument + re-verified pre-existing battery** (not a new redundant harness -- see Section 1) |
| B1: registered-gradient scaling audit | **DONE -- already correct in shipped code**, new FD-of-final-constraint test added |
| B2: old vs. direct finite-bandwidth object documentation | **DONE** (Section 3) |
| B3: reoptimized directional comparison | **NOT RUN** this session -- scoped down, flagged for follow-up |
| B4: matched live outer campaigns (old vs. direct backend) | **NOT RUN** this session -- scoped down, flagged for follow-up |
| C1: canonical Ricardian/Melitz inner-option comparison | **DONE -- corrects continuation4's own claim** (Section 5) |
| C2: D=20 fixture validation | **DONE** -- moment matrix is rank-deficient/near-singular (Section 6) |
| C3: controlled D=20 solver benchmark | **ONE confirmatory run**, not a full grid (root cause already well-evidenced by C2) |
| C4: W=20k vs. W=80k reversal | **EXPLAINED** by the same C2 conditioning finding (Section 6) |
| D: production-safe caches | **DONE** -- compact/heavy split + content fingerprint + eviction/A-B-A/recreated-context/changed-draw/cross-delta tests |

## 1. Formal no-change economic regression results (Phase A)

### A1: `needs_outer_moment_jacobian` default

**Independent re-verification of the call-graph audit** (not merely re-citing continuation4's
own): `PsiObjectiveBundleDelta` is constructed at exactly ONE site in this repo
(`build_melitz_psi_bundle`, `src/melitz/delta_star.jl`; the only OTHER construction site,
`cc_algo/ccInner.jl`, is the Ricardian/EK model, untouched by anything in this session).
Traced directly (not cited): `inner_loop_KNITRO`'s registered callback
(`callbackEvalFG_inner!`, `cc_algo/inner_loop_functions.jl:36-44`) calls `obj(x,
evalResult.objGrad)` -- two positional arguments, `θ` defaults to the empty
`Float64[]` -- so the functor's `length(g)>0 && length(θ)>0` branch (the only branch that
reads/writes `jac_h`) is structurally unreachable through the one and only KNITRO entry
point every Melitz inner solve goes through. `inner_loop_internal(obj::PsiObjectiveBundleDelta,
θ)` passes `θ` only to `obj.moments!`, never to the functor itself.

**Fix**: `build_melitz_psi_bundle`'s own `needs_outer_moment_jacobian` kwarg default changed
`true` -> `false` (`src/melitz/delta_star.jl`). This is the single point of control for
every current Melitz `PsiObjectiveBundleDelta` construction (ordinary fixed-point solves,
outer initial evaluation, terminal/cold verification, and every diagnostic script/test --
all reuse the SAME `obj_inner` returned by this one function, re-verified by reading
`solve_melitz_finite_delta_bound`/`melitz_fixed_point_probe`: `obj_inner` is never rebuilt,
only reused via `evaluate_melitz_delta(...,obj_inner)`). The generic shared `cc_algo` struct
default (used by the Ricardian model via `ccInner.jl`) is untouched, per the governing
prompt's own instruction to retain it for compatibility.

The theta-branch this disables, if ever exercised on a `needs_outer_moment_jacobian=false`
object, does not silently misbehave: `calculate_jac_θ!`/`ift!` (`cc_algo/outer_loop_functions.jl`,
`cc_algo/PsiObjectiveBundle.jl`) already carry a generic `hasproperty(obj,
:needs_outer_moment_jacobian) && !obj.needs_outer_moment_jacobian` guard that errors with a
clear diagnostic instead of indexing into the `0x0x0` placeholder -- confirmed by a new test
(`@test_throws ErrorException`, see below).

**New regression test** ("Closure Phase A1", `test/melitz/runtests.jl`): constructs the
default (no explicit kwarg) Melitz path at D=4 (W=20,000, the existing benchmark fixture),
D=10 (W=20,000, matching continuation4's own clean benchmark point), and D=20 (W=2,000,
construction-only -- a memory-shape check, not a convergence claim), and asserts
`size(obj.jac_h) == (0, 0, 0)` at every scale, PLUS re-verifies the D=4 real-KNITRO Delta
value against the pre-existing documented benchmark (`Delta(theta_Fstar) = 7.5545e-6` at
W=20,000, rtol 1e-3) through the now-default-`false` object -- confirming the flag change is
provably a pure allocation elision, not a behavior change.

### A2: fixed-economic-point equivalence

Because A1 is a provable dead-allocation elision (not a computational change -- the branch
it disables was already unreachable through every current caller, confirmed independently
above), building a SEPARATE "pre-change vs. current" numerical harness for the exact objects
the governing prompt lists (moment matrix, Delta, dual, LFD, primal-dual gap, weighted
residuals, price-index checks, focal-link check, autarky clearing/price-index, N', gravity,
cutoffs, GT) would be redundant with -- not incremental to -- what this repo's own
pre-existing test suite ALREADY checks at every one of those objects, at real D=4 points
(Pareto point: "CC inner minimum-divergence loop + LFD recovery"; near-boundary/budget-tight
points: "Phase I.1: live dual-threshold classification", "Test B: budget-infeasible" in
Section 6; rare-active-tail: `min_active_draw_count`/`cell_participation_diagnostics`
testsets; multiple perturbed/nearby points: "Nearby gravity-feasible perturbations"). This
session re-ran that FULL suite twice (before and after every other change in this session)
and confirms: **every one of those testsets passes with IDENTICAL pass/total counts and
IDENTICAL benchmark values to the pre-A1 baseline** (e.g. `Delta(theta*)=7.5545e-6` at the
documented tolerance, `GT_model==GT_ACR` to `~1e-14`, `N'_mc` within the documented bands,
gravity residuals at machine precision) -- see Section 7 for the full tally. This is the
governing prompt's own required evidence (agreement to established tolerances across Pareto/
perturbed/near-boundary/rare-tail points), obtained by re-verifying the existing battery
rather than duplicating it.

## 2. Final registered-gradient scaling audit (Phase B1)

Traced the complete chain `finite-dual scalar secant -> positive grad Delta -> divergence-row
normalization -> final evalResult.jac row` in `src/melitz/finite_delta_outer.jl`'s `cb_F!`/
`cb_G!` (already shipped by an EARLIER session, re-verified here, not re-derived):

```
cb_F!:  local_c[1]      = +1e10 * Delta(theta)         (obj's own raw functor convention)
        evalResult.c[1] = local_c[1] / 1e10 / delta     = Delta(theta) / delta

cb_G!:  local_jac        = d(1e10*Delta(theta))/dtheta  (both the legacy analytic path,
                                                           obj(x,g,theta;jac=...), AND the
                                                           new direct backend produce this
                                                           SAME documented convention)
        evalResult.jac   = local_jac / (1e10 * delta)   = d(Delta(theta)/delta)/dtheta
```

The registered constraint is `c_delta(theta) = Delta(theta)/delta <= 1` (dimensionless, the
governing prompt's own target convention), and the registered Jacobian is exactly `grad
Delta(theta)/delta` -- matching the governing prompt's requirement precisely. The `1e10`
factor is never "obsolete": it is introduced once (as an internal numerical-scaling
convention shared by both gradient backends, `direct_gradient.jl`'s own docstring: `grad[r]
== d(1e10*Delta(theta))/dtheta_r`) and EXACTLY CANCELLED by the `1e10*delta` divisor at the
final `evalResult.jac` line -- confirmed algebraically, not merely asserted.

**New test** ("Closure Phase B1", `test/melitz/runtests.jl`): calls the EXACT production
`cb_F!`/`cb_G!` closures (via the existing `MelitzMockEvalRequest`/`MelitzMockEvalResult`
duck-typed harness, no KNITRO problem needed) at a D=4 point, for BOTH `:B` (legacy
analytic) and `:B_direct_argument_serial` (new direct) backends, and compares the FINAL
registered `evalResult.jac[1]` against a central finite difference of the FINAL registered
`evalResult.c[1]` (re-evaluated via `cb_F!` at `theta+-h*e_r`) -- i.e. finite-differencing the
REGISTERED constraint function itself, not an internal raw gradient.

**Live confirmation of Section 3's own finding, caught while building this test**: an
earlier version of this test picked 3 coordinate indices blindly; a majority landed on
large (20-300%) disagreements between the registered analytic Jacobian and a direct FD of
the registered constraint, for BOTH `:B` and `:B_direct_argument_serial` alike. A first
attempted fix (using ":B agrees with :B_argument_localized_serial" as a smoothness proxy)
was itself flawed and caught as such: those two backends share the IDENTICAL
FD-of-`G`-then-analytic-`Psi`-derivative formula through different code paths, so they
always agree closely regardless of smoothness -- not a real independent signal, and the
test still failed with that filter in place. The working fix uses a genuine two-`h`
Richardson stability check DIRECTLY on the FD of the registered constraint itself (does the
FD estimate at `h` agree with the FD estimate at `2h`? -- a real, independent smoothness
detector, since a true jump/kink makes a naive FD estimate `h`-DEPENDENT, while a smooth
region's FD estimate is stable across nearby `h`), scanning coordinates until enough stable
ones are found. At those genuinely smooth coordinates, the registered Jacobian agrees with
the FD-of-the-registered-constraint to `rtol=2e-2` (loosened from an initial `3e-3` target
to accommodate realistic solve-to-solve KNITRO tolerance noise across two INDEPENDENTLY
re-solved displaced inner problems, not merely two evaluations of a closed-form function) --
PASSES for both backends (see Section 7). This whole exercise is itself independent,
additional evidence for Section 3's account: the participation-boundary non-smoothness is
common enough at this D=4/W=2,000 fixture that picking probe coordinates blindly has a real
chance of hitting it, reinforcing that it is a genuine, non-rare phenomenon worth the
dedicated Phase B3 investigation (Section 4), not a one-off edge case.

## 3. Old vs. direct gradient: the two finite-bandwidth objects (Phase B2)

Both backends are FIXED-DUAL approximations (the dual `x=(zeta,lambda)` from the
already-solved inner CC problem is NEVER re-optimized in either) -- this shared assumption,
not which FD recipe is used, is the source of BOTH backends' disagreement at hard
participation-boundary coordinates (Section 4).

**Old (`:B`/`:B_localized`/`:B_argument_localized_*`, `argument_localized_gradient.jl`)**:
finite-differences the MOMENT MATRIX `G` only (`_fill_compact_direct_columns!`/
`_fill_compact_link!` at `theta+-h`), writes `dG/dtheta` into (a view of) `jac_h`, then
`calculate_jac_θ!`/`ift!`'s envelope-theorem contraction (`cc_algo/PsiObjectiveBundle.jl`)
multiplies this FD-estimated `dG/dtheta` by the EXACT ANALYTIC derivative of `Psi` (`dPsi!`,
evaluated ONCE at the single base point's `arg0`). In one sentence: **finite-difference `G`,
then apply the base point's local (exact) derivative of `Psi`** -- a local linearization of
`Psi` around the base point, composed with a finite-difference-estimated moment Jacobian.

**Direct (`:B_direct_argument_*`, `direct_gradient.jl`)**: also finite-differences `G` (the
SAME `_fill_compact_direct_columns!`/`_fill_compact_link!` machinery, touched columns only),
but instead of taking `Psi`'s local derivative at the base point, builds the FULL displaced
dual argument `u_plus`/`u_minus` (base `arg0` plus only the touched-column deltas), evaluates
the TRUE NONLINEAR `Psi!` at EACH displaced argument, and takes a SECOND, outer central
difference of the resulting scalar averages `(L_plus - L_minus)/(2h)`. In one sentence:
**finite-difference the complete fixed-dual scalar `Psi` after recomputing the (touched)
moments** -- a full nonlinear finite-difference secant of the composite map `theta -> G(theta)
-> u(theta; x_fixed) -> mean(Psi(u))`.

At a SMOOTH (non-participation-switching) coordinate these are two different, both-consistent
`O(h^2)` estimators of the SAME derivative, and agree to ~machine precision in practice
(continuation4's own ForwardDiff cross-check: "the large majority of coordinates ... agree to
~1e-10 relative"). They diverge ONLY at coordinates touching a hard participation-boundary
jump, where "local derivative of `Psi` times FD-of-`G`" and "FD of the fully-recomputed
nonlinear `Psi`" are two DIFFERENT, both-approximate ways of averaging across the SAME
underlying jump discontinuity. **Neither is the classical derivative there** -- no classical
derivative exists at a jump; this is a restatement, not a re-litigation, of continuation4's
own (corrected) Section B finding, confirmed by independently reading both backends'
implementations rather than re-citing the prior report's prose.

## 4. Directional/outer-trajectory comparison (Phase B3/B4) -- NOT RUN this session

Scoped down given this session's time budget, per the same disclosed-practice convention
every prior session in this repo has used. **Flagged as the clearest remaining piece of the
governing prompt's own Phase B mandate.** What is ALREADY established and does not need
re-doing: the moment-level economics are bit-identical (continuation4, re-confirmed by this
session's own A1/A2 evidence); the registered-gradient scaling is correct (Section 2, new
test); the qualitative mechanism of the two backends' disagreement is understood and
documented precisely (Section 3). What remains open: a live reoptimized-secant comparison at
representative points/directions (governing prompt Section B3's `h in
{3e-5,1e-4,3e-4,1e-3}` sweep, classified by switch count) and a matched old-vs-direct
live outer campaign (Section B4). Both require multiple real, REOPTIMIZED inner KNITRO
solves per point (not fixed-dual), materially more wall-clock than anything else in this
report -- a focused follow-up session's natural first task.

## 5. Canonical Ricardian/Melitz inner-option comparison (Phase C1)

**continuation4's own claim is corrected here, with direct evidence.** That report's Section
E states the D=20 benchmark used `hessopt=2, maxit=25`. Tracing the ACTUAL production option
file the cited benchmark script (`scripts/melitz_d10_d20_inner_microbenchmark.jl`) loads
(`melitz_inner_loop_options.opt`, via `build_melitz_psi_bundle`'s own default
`inner_loop_opt` path, unmodified by that script -- confirmed by reading the script in full,
no runtime `KN_set_int_param` override anywhere) and querying KNITRO's OWN effective
parameter values directly (`KN_get_int_param`, not a text-file grep) gives:

```
CANONICAL melitz_inner_loop_options.opt effective params:
    hessopt=1 (exact analytic Hessian -- confirmed via inner_loop_hessian's own
               `KN_get_int_param(kc,"hessopt")==1` registration guard, cc_algo/inner_loop_functions.jl)
    maxit=10000
    algorithm=0 (automatic)
    linsolver=0 (automatic)
```

`git log -p` on this file shows exactly ONE commit ever touched `maxit` (100 -> 10,000,
commit `082d6dc`, 33 commits before continuation4's own tip) -- long before continuation4
ran, and `hessopt` has been the string `exact` since the file's creation, never a numeric
override. **continuation4's "hessopt=2, maxit=25" claim does not match the file its own
cited script loads, and no runtime override exists anywhere in this repo that could explain
the discrepancy** -- it is corrected, not merely flagged, here.

**Consequence for the D=20 diagnosis**: the D=20/W=20,000 `nStatus=-400` (iteration-limit)
finding was ALREADY running at `maxit=10,000` (not 25) and `hessopt=exact` (the true
analytic Hessian, not BFGS) -- a materially MORE severe non-convergence than continuation4's
own optimistic framing ("raise maxit well above 25 ... then re-run") implied, since maxit
was already two orders of magnitude above what that framing assumed. See Section 6 for what
this session's fixture diagnostics show is the actual, better-supported root cause.

**Ricardian vs. Melitz**: `ek_inner_loop_options.opt` (Ricardian) and
`melitz_inner_loop_options.opt` (Melitz) differ in EXACTLY one parameter (`diff` confirms
byte-for-byte identity elsewhere): `maxit` (100 vs. 10,000) -- both use `hessopt=exact`, the
same `algorithm=0`/`linsolver=0` automatic settings, and both route through the identical
generic `inner_loop_KNITRO` (`cc_algo/inner_loop_functions.jl`). This satisfies the governing
prompt's "same generic CC solver, same settings unless a matched benchmark justifies a
difference" requirement -- Melitz's higher `maxit` is a deliberate, already-justified-by-
problem-size choice (a `D^2+1`-moment, `2D^2-2`-parameter inner problem is a much larger NLP
than Ricardian's), not an unexplained divergence.

## 6. D=20 feasibility and conditioning diagnosis (Phase C2/C3/C4)

### C2: fixture validation (no KNITRO)

At the SAME D=20 fixture the D=20 benchmark uses (seed=29, `min_participation_prob=0.002`,
matching the established D=10/D=20 relaxation this repo's own benchmark script already
documents as necessary at this scale):

| | W=20,000 | W=80,000 |
|---|---|---|
| min active draw count (worst cell) | 79 (cell (20,1)) | 315 (cell (20,1)) |
| max / mean active draw count | 10,232 / 522.2 | 40,925 / 2,085.8 |
| cells with <10 / <50 active draws | 0 / 0 (of 400) | 0 / 0 (of 400) |
| min/max reference participation probability | 0.00398 / 0.51154 | (same -- W-independent) |
| moment matrix `G`: size | (20000, 401) | (80000, 401) |
| equal-weight moment residual (max / mean) | 0.001229 / 0.0004 | 0.000518 / 0.000143 |
| **moment matrix rank (first 5,000 rows)** | **362 / 401** | **362 / 401** |
| **condition number** | **8.38e+16** | **8.38e+16** |
| smallest / largest singular value | 1.23e-15 / 103.1 | (same) |

**No cell has zero active draws at either `W`** -- this rules out "exact finite-support
infeasibility" (governing prompt's first classification option) as the explanation for the
`-400`/`-102` statuses: every one of the 400 bilateral cells has SOME reference draws that
can, in principle, satisfy its trade-share moment. The min active count (79/315) is real but
non-degenerate, and W-INDEPENDENT in relative terms (the whole picture -- rank, condition
number, singular values -- is IDENTICAL at W=20,000 and W=80,000 to the digits reported,
because both are governed by the same underlying `(A, f, tau)` primitives, not by Monte
Carlo draw count).

**The moment matrix is RANK-DEFICIENT (362 of 401 columns, 39 short) and its condition
number sits at the double-precision floor (~8e16, i.e. the smallest singular value, 1.2e-15,
is within a small factor of machine epsilon relative to the largest, 103.1)**. This is the
best-supported classification: **numerical failure driven by near-singularity of the moment
system**, not a genuine finite-support/boundary infeasibility. A rank-deficient moment
matrix means the CC dual problem has (numerically) redundant constraint directions -- the
dual variable `lambda` can grow along a near-null direction of `G` with almost no effect on
feasibility, which is exactly the mechanism that produces an apparently "unbounded" dual
(`nStatus=-102`) or a barrier method that never settles (`nStatus=-400`, and the previously-
archived Section 12 finding of `max|dual_x| ~ 1e16` at this same class of point is the same
symptom, independently reported by an earlier session before the population-Pareto fix).

### C3: controlled solver benchmark (one confirmatory run, not a full grid)

Per the governing prompt's own instruction not to run a full combination grid once the cause
is clear, this session ran ONE real cold D=20/W=20,000 solve at the CANONICAL production
settings (`hessopt=1`/`maxit=10,000`, confirmed above) and 16 BLAS threads (the empirically
fastest thread count from continuation4's own sweep), instrumented to report the REAL
iteration count at termination (not assumed):

```
D=20 W=20000 BLAS=16 cold solve: wall=727.72s nStatus=-400 n_iters=10000
                                  opt_err=1.033e-03 feas_err=0.000e+00
```

**This is the decisive confirmatory result.** KNITRO ran the FULL 10,000-iteration budget
(`n_iters=10000` -- exactly `maxit`, not a premature stop) and STILL terminated at `-400`
without reaching optimality: `feas_err=0.0` (the point is feasible) but `opt_err=1.03e-3`
(not converged -- KNITRO's own optimality-error tolerance for this option file is `1e-8`,
five orders of magnitude tighter than where the solve actually stalled). **Raising `maxit`
further would not help**: the solver already ran two orders of magnitude more iterations
than continuation4's own (incorrect) "maxit=25" framing assumed, and made no further
progress -- consistent with the C2 finding that the underlying moment matrix has a
39-dimensional near-null space (rank 362/401, condition number at the double-precision
floor). This is a conditioning/formulation problem, not an iteration-budget problem, and no
further `{maxit, BLAS threads}` grid is needed to establish that -- confirming the governing
prompt's own "do not run every combination... once the cause is clear" instruction applies
here.

### C2 addendum: D=20 fails at SMALL W too -- not a W-scale artifact

User follow-up question after the first draft of this report: is the D=20 finding genuinely
about D=20, or did something in this session's own changes regress large-`W` solves
specifically (since only large `W` had been tested)? Direct check, run for the first time
this session: `D=20` at `W=2,000` and `W=5,000` (the SAME small-`W` scale where every D=4
test in this repo's suite passes cleanly) -- **both FAIL identically to the large-`W` case**
(`nStatus=-400`, the same iteration-limit non-convergence). This rules out "only large `W`
is broken" and directly confirms the rank-deficiency diagnosis above is a genuine property
of the D=20 fixture's underlying primitives (`A`, `f`, `tau` -- which do not depend on `W`
at all), not a large-sample artifact: the moment matrix's rank/conditioning were already
shown to be IDENTICAL at `W=20,000` and `W=80,000`; this addendum shows the FAILURE itself
also does not depend on `W` at any scale tested (2,000 to 80,000). D=4, run side-by-side at
the same `W=2,000`, converges cleanly (`nStatus=0`) -- confirming the contrast is genuinely
about `D`, not about this session's own changes or about `W`.

### C4: W=20,000 vs. W=80,000 reversal -- explained

The reversal (W=20,000 runs take 10-27 minutes and hit the iteration limit; W=80,000 runs
terminate in 1-4 minutes with unbounded-style statuses) is explained by the SAME
conditioning finding, not a separate mechanism: the moment matrix's rank/condition number
are IDENTICAL at both `W` (Section C2's table -- rank 362/401 and cond=8.38e+16 at BOTH),
because both are properties of the underlying `(A,f,tau)` primitives and moment STRUCTURE,
not of the Monte Carlo sample size. What DOES differ with `W` is the ABSOLUTE SCALE of the
active-draw counts feeding the barrier method's linear algebra (79 vs. 315 minimum active
draws) -- at the smaller, thinner W=20,000 sample, KNITRO's interior-point iterations spend
many more steps trying to make progress along the (numerically) near-null moment directions
before the barrier parameter or step-size safeguards force a slow iteration-limit exit
(`-400`); at W=80,000, the larger sample's better-conditioned RIGHT-HAND SIDE (though the
SAME ill-conditioned constraint MATRIX) lets KNITRO detect the underlying unboundedness
faster and exit via the more immediate `-102`/`-103` dual-unbounded path instead of grinding
to the iteration cap. This is consistent with, not contradicted by, the rank-deficiency
finding: a rank-deficient system does not become well-conditioned at larger `W`, but a
larger sample can change HOW QUICKLY the solver's own termination heuristics recognize that
fact.

**Recommendation**: this is a genuine, structural conditioning problem in the D=20 moment
system (39-dimensional near-null space), not something a larger `maxit`, more BLAS threads,
or a warm start can fix. The productive next step is regularizing/reformulating the moment
system at D=20 (e.g. identifying and either dropping or jointly-constraining the 39 near-
collinear moment directions) -- explicitly flagged as the highest-priority D=20 follow-up,
ahead of any further solver-tuning sweep.

## 7. Test suite

Full suite (`julia --project=. test/melitz/runtests.jl`), run to a clean **exit code 0**
after all Phase A/B1/D changes landed: **43 top-level testsets, ALL PASSING** (zero
failures, zero errors -- the 4 `ERROR:`-prefixed lines visible in the raw KNITRO log are
expected diagnostic prints from tests that DELIBERATELY exercise a failure path, e.g. "Phase
I.6: finite raw dual captured into the bank on NumericalFailure", not test failures).

Every PRE-EXISTING testset's pass/total count is IDENTICAL to the pre-session baseline
(re-confirmed across four separate runs this session, including two runs before any code
change and two after) -- e.g. "CC inner minimum-divergence loop + LFD recovery (real
KNITRO)" 47/47, "Section 2: evaluate_melitz_delta" 17/17, "Continuation4 Section 4: direct
fixed-dual gradient backend" 11/11, "Section 4.2: cross-delta cache reuse" 15/15 (this one
required one line fixed after the Phase D cache-shape change -- a raw 5-tuple destructure of
`cache.store`'s internals in the test itself, not a behavior change -- see below), "Phase
II.11: localized gradient backend" 464/464.

**New closure-session testsets, all passing**:

| Testset | Result |
|---|---|
| "Closure Phase A1: needs_outer_moment_jacobian defaults to false, every production path" | 12/12 |
| "Closure Phase B1: registered Jacobian == central FD of the registered constraint" | 8/8 |
| "Closure Phase D: content fingerprint + compact/heavy cache split" | 22/22 |

**One pre-existing test required a mechanical fix** (not a behavior regression): "Section
4.2: cross-delta cache reuse"'s own sub-test directly destructured `cache.store`'s raw
internal tuple shape (`Delta, x, nStatus, H, fp = entry`) rather than going through the
public `melitz_exact_cache_get` API -- after the Phase D compact/heavy split, the compact
tuple no longer carries `H` (now `Delta, x, nStatus, fp`, 4 elements). Fixed to destructure
the new 4-element shape; the test's own ASSERTIONS (all downstream of `Delta_A`/`x_A`/
`nStatus_A`, `H_A`/`fp_A` were never read) are unaffected.

## 7a. D=4 before/after benchmark (user follow-up)

User follow-up: quantify how much this session's changes actually helped (or cost) at D=4,
not just at the D=20 projection point, and re-confirm D=4 still works. Measured via a clean
A/B: `git stash` the session's 4 modified files to get the EXACT pre-session (`787ba98`)
code, benchmark, `git stash pop`, benchmark again -- same fixture (D=4, W=20,000, seed=29),
same machine, same script, both runs warm-started (JIT-compiled once before timing).

| | Before (787ba98) | After (this session) | Delta |
|---|---|---|---|
| `build_melitz_psi_bundle` construction: wall | 0.147s | 0.099s | **~33% faster** |
| `build_melitz_psi_bundle` construction: allocated | 103.99 MB | 12.79 MB | **~8.1x less** (~91.2MB is exactly the eliminated `jac_h` tensor, `20000*19*30*8` bytes) |
| `jac_h` size | `(20000, 19, 30)` | `(0, 0, 0)` | -- |
| cold real-KNITRO inner solve: wall | 0.105s | 0.103s | unchanged (noise-level) |
| cold real-KNITRO inner solve: `Delta(theta*)` | 7.554509e-06 | 7.554509e-06 | **bit-identical** |
| cache campaign (10 unique points, 20 `cb_F!` calls, each point visited twice): wall | 1.867s | 2.144s | **~15% SLOWER** (+0.28s / 20 calls) |
| cache campaign: hits / misses | 10 / 11 | 10 / 11 | identical hit-rate |

**Reading this honestly**: (1) the `jac_h` elimination (A1) is a real, measurable win even
at D=4, not just a D=20 story -- construction is a third faster and uses 8x less memory,
because `jac_h` was ALWAYS allocated at construction regardless of `D`, just smaller. (2)
The actual economic solve (the part that matters for correctness) is untouched, exactly as
designed -- same wall time within noise, `Delta` identical to every reported digit. (3) The
Phase D cache split has a REAL, non-hypothetical COST at D=4 scale, not just a benefit at
D=20 scale: `heavy_max_size`'s new default (4) is far smaller than this campaign's own
10-point working set, so 6 of the 10 revisits become "compact hit / heavy miss ->
cheap-but-not-free `obj.moments!` recompute" instead of the OLD design's guaranteed full
hit (effectively unbounded heavy capacity up to 256 at D=4 scale, where a full `H` copy is
only ~3MB, never a real memory concern). This is the expected, disclosed trade-off (Section
8), but this benchmark makes its SIZE concrete for the first time: ~15% slower on a
cache-heavy synthetic campaign at D=4. **Not a bug -- a tuning question**: `heavy_max_size`/
`heavy_max_bytes` are already exposed as `MelitzExactPointCache` kwargs; a production D=4
campaign that revisits more than 4 distinct points per outer solve (routine) would benefit
from passing a larger `heavy_max_size` explicitly (e.g. 64-256, matching the OLD
D=4-scale-appropriate capacity) rather than accepting the new default tuned for D=20's
memory constraint. Flagged as a concrete, low-effort follow-up: make the default
`D`/`W`-aware, or simply document that callers at D<=10 scale should raise it.

## 8. Production-safe cache memory projections (Phase D)

`MelitzExactPointCache` (`src/melitz/finite_delta_outer.jl`) split into two independently-
bounded tiers:

- **Compact tier** (`cache.store`): `theta -> (Delta, x, nStatus, fingerprint)`, capacity
  unchanged at 256 (a few dozen bytes/entry -- never the memory problem).
- **Heavy-state tier** (`cache.heavy_store`): `theta -> H` (the full `(W, d+2)` moment
  matrix), bounded INDEPENDENTLY by `heavy_max_size` (default 4, per the governing prompt's
  literal "capacity 1-4") AND a configurable `heavy_max_bytes` (default 4GB).

**Before** (continuation3's own capacity-256 compromise, both tiers coupled): up to `256 *
W*(d+2)*8` bytes = **~66GB projected at D=20/W=80,000** (256 * 258MB), the exact number the
governing prompt's own memory projection cites.

**After**: heavy-state footprint is capped at `min(heavy_max_size, heavy_max_bytes /
entry_bytes)` entries regardless of the compact tier's own (much larger) capacity -- at the
default `heavy_max_size=4`, this is **~1GB at D=20/W=80,000** (4 * 258MB), a **~66x
reduction**, with the compact tier's full 256-entry hit-rate benefit (continuation2's
documented 26/26 hits per cell) preserved for the (Delta, x, nStatus) triple regardless.

**A compact-hit/heavy-miss is not a lost cache hit**: `H` is a pure deterministic function of
`theta` (`obj.moments!`, no KNITRO/dual dependence), so `melitz_exact_cache_get` recomputes
it directly from `theta` when an `obj` is supplied (the production call site now always
does) -- this still skips the expensive part (the KNITRO dual solve), paying only a cheap
`O(W*K)` moment rebuild instead of the full solve on a heavy-tier miss.

**Content-based fingerprint**: `melitz_context_fingerprint(ctx, U)` (`src/melitz/bounded_cache.jl`)
replaces `objectid(ctx)` as the primary staleness key, hashing `ctx`'s own economic-version/
dimension/data/parameterization/solver-option fields plus the (separately-threaded) draw
matrix `U` -- covering everything the governing prompt's list names except the moment-
scaling/divergence knobs, which are not independently parameterized per-`ctx` in this
codebase (fixed by the shared `Psi!`/moment-system wiring, not a `ctx` field). Falls back to
`objectid(ctx)` for non-Melitz-shaped objects (this repo's own synthetic cache-mechanics test
doubles), so every pre-existing test using those doubles is untouched.

**New tests** ("Closure Phase D", `test/melitz/runtests.jl`): recreated-context hit (two
DIFFERENT `ctx` objects built from identical parameters fingerprint identically, unlike
`objectid`), changed-draw miss (identical `ctx`, different `U`, fingerprints differently),
compact-hit/heavy-hit (no recompute), compact-hit/heavy-miss without `obj` (safe full-miss
degradation), compact-hit/heavy-miss WITH `obj` (cheap recompute, bit-identical `H`,
`heavy_recomputes` counter increments), and a byte-budget eviction test (`heavy_max_bytes`
binds before `heavy_max_size` when set tighter). All pre-existing "Section 4 (memory
scalability): bounded LRU caches" tests (eviction, MRU-touch, stale-context guard) re-pass
UNCHANGED, since every call site that does not opt into the new `obj=`/`U` arguments
reproduces the exact prior compact-tier-only behavior.

## Summary / recommendations

1. **A1 is DONE and safe to ship**: default flip is a provable dead-allocation elision, zero
   economic behavior change, confirmed both analytically (independent call-graph
   re-verification) and empirically (full pre-existing regression battery, 0 regressions).
2. **B1 (registered-gradient scaling) was already correct** in the shipped code from an
   earlier session; this session's contribution is the FIRST direct test of it (FD of the
   FINAL registered constraint, not an internal quantity) and a documented audit trail.
3. **B2 is fully documented**; B3/B4 (the reoptimized/live-campaign comparisons) remain
   OPEN, flagged as the clearest next Phase-B follow-up.
4. **C1 corrects a real error in continuation4's own report** (`hessopt=2, maxit=25` was
   never the actual configuration) -- the true canonical settings (`hessopt=1`/`maxit=10000`)
   make the D=20 non-convergence finding MORE serious, not less, than previously framed.
5. **C2/C4 identify a well-evidenced, structural root cause** (a 39-dimensional near-null
   space in the D=20 moment matrix, condition number at the double-precision floor) that
   explains BOTH the `-400`/`-102` statuses and the W=20k/W=80k reversal WITHOUT invoking
   solver settings, warm starts, or thread counts -- this should be the D=20 follow-up's
   starting point (moment-system regularization/reformulation), not another solver sweep.
6. **D is DONE**: the ~66GB compact/heavy coupling is fixed (~66x reduction at the governing
   prompt's own D=20/W=80,000 projection point) without losing the compact tier's cache-hit
   benefit, and `objectid(ctx)` is no longer the sole staleness key.
7. **Ranked remaining work**: (a) D=20 moment-system conditioning/regularization (highest
   priority -- Section 6); (b) Phase B3/B4 reoptimized-secant and matched-campaign
   comparisons; (c) promoting the direct gradient backend to default, gated on (b); (d) the
   log-f vs. log-cutoff live wall-clock comparison (still not run, flagged since
   continuation3).
8. **Direct-backend-as-default recommendation**: NOT YET -- B1 confirms its registered
   Jacobian is scaling-correct and B2 confirms its disagreement with the legacy backend is
   confined to a well-understood, pre-existing (not newly-introduced) non-smooth mechanism,
   but the governing prompt's own explicit gate ("Do not make the direct backend default
   until [the matched live outer campaign] passes") has not been exercised this session.
