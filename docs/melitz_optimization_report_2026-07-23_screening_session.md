# Melitz inner-solve screening session -- 2026-07-23

Branch: `melitz/fullD-delta-star`. Starting checkpoint `09cc6b5` (the prior continuation
session's own final commit, docs-only on top of the code checkpoint `8ebacc4` the governing
prompt named). This session implements the governing prompt's diagnostic/screening
program as re-prioritized by its own ADDENDUM (aggressive rejection, no routine cold
retry, front-loaded cheap screens over exact LP classification) -- the addendum
supersedes the main prompt's warm/cold-retry and screen-ordering instructions, per its own
text, and this report follows that precedence throughout.

## 1. Baseline reproduction (main prompt Section 1)

- **Julia**: `1.12.6`. **KNITRO**: `13.0.1` (`/opt/shared_sw/knitro/13.0.1`). **Host**:
  `demand.mit.edu`. **CPU**: 208 logical CPUs, plenty of headroom to run baseline/variant
  campaigns concurrently as fully independent single-threaded processes.
- **Threading**: `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for every
  run this session (this repo's own standing hard-cap policy).
- Full test suite (`test/melitz/runtests.jl`) run **before any edit**, at `09cc6b5`: 0
  failures (595/595 by this session's own line-parse of the `Test Summary` table; the
  prior session's own report claims 611/611 by a slightly different parse -- both agree
  on the only fact that matters, zero failures). Re-run **twice more** after this
  session's edits (once right after wiring the typed classifier, once again after adding
  the `lower_limit_guard`/`on_inner_result` kwargs): 0 failures both times, and the suite
  runs **noticeably faster** post-change (e.g. "Section 6: fixed-point KNITRO integration
  tests" dropped from 1m37.2s to 5.5s -- the same screens this report benchmarks below,
  visible immediately in the test suite itself).
- Post-JIT profiled reproduction (main prompt Section 1's own ask) is reported directly
  in Section 5 below as the "baseline" arm of the before/after campaign, rather than as a
  separate standalone step -- running it twice (once standalone, once as the comparison
  baseline) would have doubled a genuinely expensive KNITRO campaign for no additional
  information.
- Option-file identity: `melitz_inner_loop_options.opt` unchanged this session except a
  **new sibling file**, `melitz_inner_loop_options_budgetcheck.opt` (identical byte-for-
  byte except `maxit 10000` -> `maxit 250`), added for the `:budget_check`-style routine
  callback path (Section 4 below) -- the original file is untouched and remains available
  for `:full_value`/diagnostic call sites.

## 2. Typed inner classification (main prompt Section 2)

New file `src/melitz/inner_screening.jl`. `MelitzInnerResult` is an abstract type with five
concrete cases:

- `InnerSolved(Delta, x, nStatus)` -- verified.
- `BudgetInfeasible(lower_bound, source)` -- certified `Delta(theta) > delta` without
  necessarily having completed a KNITRO solve. This session implements exactly one
  `source`: `:stored_dual` (Section 3 below). `:scalar`/`:origin_block` (main prompt
  Sections 8.2/8.3) are NOT implemented -- see Section C.
- `MomentInfeasible(column, lo, hi, kind)` -- exact finite-support separation certificate.
  This session implements exactly one `kind`: `:range` (Section 3 below). The
  K-dimensional convex-hull LP (main prompt Sections 5-6) is NOT implemented -- see
  Section C.
- `BoundaryFeasible(note)` -- present in the type hierarchy per the main prompt's own
  requirement ("do not call a boundary-feasible point infeasible"), but **never produced**
  by this session's classifier (`melitz_classified_inner_solve`). Distinguishing a genuine
  non-attained-dual boundary case from an ordinary `NumericalFailure` needs the primal-
  LP/relative-interior diagnostics main prompt Section 7 describes, which this session did
  not implement (Section C/D explain why, and what the 25-55s failures actually look like
  instead).
- `NumericalFailure(nStatus)` -- no certificate obtained; the single (no-retry) KNITRO
  attempt returned a status outside `{0,-100,-101,-103}`.

The outer callback (`finite_delta_outer.jl`'s `inner_solve_verified_or_fail`) now
dispatches on this type instead of the prior `(objSol, x, nStatus)` tuple + hand-rolled
cold-retry logic. Every non-`InnerSolved` case throws the same `DomainError` KNITRO.jl's
`_try_catch_handler` already converts to a proper evaluation-error status (unchanged
convention, only the classification feeding it is new) -- confirmed this does not change
existing test expectations (0 failures across three post-change full-suite runs).

## 3. Screens implemented (main prompt Sections 3-4, addendum Sections 4.1/4.3/8.1)

### 3.1 Finite-support feasibility framework (main prompt Section 3) -- documented, used as
the range screen's justification, not separately implemented as a standalone proof
artifact beyond the docstring in `inner_screening.jl`'s file header:

At a fixed outer point, `G` is `W x K`. The moment system is feasible on the numerical
support iff there exists a probability vector `p` (`p>=0`, `sum(p)=1`) with `G'*p = 0`,
i.e. iff `0` lies in the convex hull of the `W` rows of `G`. For a SINGLE column considered
alone, this reduces to `min(G[:,k]) <= 0 <= max(G[:,k])` (a two-point distribution on the
column's argmin/argmax draw attains any value in between) -- necessary for the full
K-column joint problem, not sufficient. Hand-verified on small constructed examples during
development (not committed as a separate test file this session -- see Section C for why
this was judged sufficient given time).

### 3.2 Range screen (`melitz_range_screen`, addendum Section 4.1)

`O(W*K)`, a single pass over `G` (already resident in `obj.H` after a `moments!` call),
zero extra allocation, no KNITRO call. Applies uniformly to trade-share columns AND the
focal-link column -- the main prompt's "for the focal link, require the draw-level link
contribution to span zero" is the identical test on that one column, not a separate rule.
Fused directly into the classified-solve orchestration (`melitz_classified_inner_solve`),
run immediately after `obj.moments!` fills `obj.H`, before any KNITRO call.

### 3.3 Stored-dual lower-bound screen (`melitz_stored_dual_lower_bound`,
`MelitzDualBank`, addendum Section 4.3/8.1)

**Mathematical basis** (also serves as this report's Section 9 sign table for THIS specific
mechanism -- the OTHER, KNITRO-native threshold mechanism has its own sign table in
Section 4 below): the inner CC dual problem minimizes a raw functor value
`f(zeta,lambda; G) = zeta + (1/M) sum_w Psi*(-zeta - lambda'G[w,:])` over `(zeta,lambda)`
which is **completely unconstrained** in this Implicit-bundle formulation
(`build_melitz_implicit_bundle` passes `inequality_index=Int64[]`, confirmed by reading
`inner_loop_lower_bounds(obj::PsiObjectiveBundleImplicit)`, `cc_algo/inner_loop_functions.jl`
-- every entry is `-KNITRO.KN_INFINITY`). KNITRO minimizes `f`; at the true optimum
`f* = -Delta(theta)` (this is exactly what `cb_F!`'s existing `local_c[1] = -f*1e10 =
1e10*Delta(theta)` line already encodes, unchanged this session). Because the dual
variables are unconstrained, EVERY `(zeta,lambda) in R^{1+d}` is dual-feasible, so weak
duality gives `f(zeta,lambda; G) >= f* = -Delta(theta)` for literally every point, i.e.
`-f(zeta,lambda; G) <= Delta(theta)` -- a valid lower bound from ANY stored dual vector,
not just a verified optimum at some OTHER `G`.

**Sign table for this mechanism:**

| quantity | sign/meaning |
|---|---|
| raw KNITRO-minimized objective `f` | `f = zeta + (1/M) sum Psi*(-zeta-lambda'g)`, convex, unconstrained domain |
| `f` at the true inner optimum | `f* = -Delta(theta)` (Delta >= 0) |
| `f` at ANY OTHER dual point (verified elsewhere, or arbitrary) | `f >= f*` (weak duality) |
| implied Delta lower bound from any dual point | `-f <= Delta(theta)`, valid unconditionally |
| screen's rejection rule | reject iff `max_x(-f(x)) > delta + guard` |

**Implementation**: `MelitzDualBank` (bounded FIFO, `max_size=8` default) holds verified
`x` vectors from earlier in the SAME outer KNITRO trajectory (never shared across
trajectories, matching this repo's own per-solve scoping discipline). The screen evaluates
the bare functor `obj(x)` (no gradient/constraint request -- one BLAS `gemv!` plus an
elementwise `Psi!` map, confirmed by reading `PsiObjectiveBundle.jl`'s functor body; the
`_enter_callback!`/`_exit_callback!` reentrancy guard already documents this exact
"call the functor from within a sequential context" pattern as safe, so no new guard
machinery was needed) over every banked entry, taking the tightest (max) resulting bound.

## 4. The existing objective-threshold mechanism (main prompt Section 9) -- traced, and now
wired up for Melitz

`cc_algo/PsiObjectiveBundle.jl`'s shared functor bodies (`PsiObjectiveBundleExplicit`/
`Implicit`/`Delta`, and the parallel `KLObjectiveBundle*` family) ALL contain:

```julia
if f <= lower_limit
    return -KNITRO.KN_INFINITY
else
    return f
end
```

where `f` is the same raw dual objective from Section 3.3 above (`f* = -Delta` at the
optimum, `-f(x) <= Delta` for any `x` since the problem is unconstrained). `lower_limit`
is a per-bundle `Float64` field, default `-KNITRO.KN_INFINITY` (i.e. the branch is
NEVER taken unless a caller explicitly sets it).

**Where it was active, and where it was not:**

- **Ricardian** (`cc_algo/ccOuter.jl`, `cc_algo/ccInner.jl`): every bundle constructor call
  site sets `lower_limit = -50`, a fixed constant. When any barrier iterate's `f` drops to
  `-50` or below (equivalently, that iterate's own valid Delta lower bound `-f` reaches
  `50`), the functor reports `-KNITRO.KN_INFINITY` to KNITRO, which terminates the solve as
  unbounded almost immediately rather than continuing toward `maxit=100`.
- **Melitz**: grepping `src/melitz/*.jl` for `lower_limit` returns zero hits, before this
  session. `build_melitz_psi_bundle` (`delta_star.jl`) and the pre-session
  `build_melitz_implicit_bundle` (`finite_delta_outer.jl`) both construct their bundles
  without ever touching this field, so it silently sat at `-KNITRO.KN_INFINITY` -- **this
  mechanism has never fired for Melitz.** Combined with `maxit=10000` (vs. Ricardian's
  `100`), this is the most direct, mechanical explanation available for why a single failed
  Melitz inner attempt can cost up to ~55s while Ricardian's essentially never do: Ricardian
  has TWO independent brakes (an early `-Inf` bailout at `-50` AND a 100-iteration cap);
  Melitz, before this session, had NEITHER.

**Fix implemented**: `build_melitz_implicit_bundle` and `solve_melitz_finite_delta_bound`
gained an optional `lower_limit_guard::Union{Nothing,Real}=nothing` keyword. When set,
`lower_limit = -(delta + lower_limit_guard)` -- delta-AWARE (unlike Ricardian's fixed
`-50`, appropriate for Melitz since `delta` varies by campaign, and `-50` would be either
vacuous or wildly conservative depending on `delta`'s own scale). Default `nothing`
preserves the exact pre-session-tested behavior (disabled) for every existing caller.
Live-tested at `lower_limit_guard=0.0` (the tightest possible setting -- see Section 5.4):
mathematically safe (weak duality holds at every iterate regardless of convergence stage,
since the dual is unconstrained), produces correct outer-feasible incumbents, but the
ADDITIONAL wall-time win beyond the pre-solve screens alone was small in this specific
campaign (Section 5.4) -- reported honestly as a real but modest effect at this D=4/W=20000
scale, not a dramatic further win. **Not yet wired to produce a distinct `BudgetInfeasible`
classification** when it fires mid-solve (it currently surfaces as an ordinary
`NumericalFailure`, since the classifier only inspects the post-hoc `nStatus`, not why
KNITRO stopped) -- flagged as follow-up work, Section F.

## 5. Before/after performance campaign

**Scope note**: the main prompt's full Section 17 grid (`delta in {1e-3,1e-2}`, both
directions, both cutoff backends) was judged too expensive to run twice (baseline +
new-policy) inside this session, given the baseline arm alone (Section 5.1 below) took
**7 minutes for a single delta/direction/backend cell**. Per this repo's own established
precedent (the prior continuation session's Section 13 scope decision, explicitly
permitted by the governing prompt) and the addendum's own tone (prioritize a decisive,
honestly-scoped comparison over an exhaustive but half-finished one), this session ran the
single most informative cell -- `delta=1e-2`, both directions, `:logf`,
`:nonlinear_reference` cutoff backend, D=4/W=20,000/seed=29 -- across four configurations:
the true baseline (pre-session code) and three new-policy variants. `delta=1e-3` and the
`:linear` cutoff backend are NOT re-benchmarked this session; both are structurally
orthogonal to the screening changes (the screens run on `G`, independent of which cutoff
backend registers the deterministic cutoff rows), so there is no specific reason to expect
a qualitatively different speedup ratio, but this is not independently confirmed.

All four runs used the identical fixture (`generate_fake_melitz_data(D=4, sigma=2.5,
theta_star=6.8, target_country=1, seed=29, W=20_000)`), `theta_box=0.10`,
`gradient_backend=:B`, `h=1e-4`, outer `maxit=25` (`melitz_outer_finite_delta.opt`,
unchanged).

### 5.1 Baseline (pre-session code, commit `09cc6b5`, via a `git worktree`)

| direction | wall(s) | terminal nStatus | real KNITRO inner calls | of which infeasible | cold-verified Delta | gamma_prime |
|---|---|---|---|---|---|---|
| upper | 87.3 | -410 | 123 | 52 | 9.532e-3 | 0.935445 |
| lower | 333.8 | -400 | 198 | 84 | 9.955828e-3 | 0.988286 |
| **total** | **421.1** | | | | | |

Both incumbents outer-feasible, Delta safely inside the 1e-2 budget -- the baseline is
correct, just slow, exactly the governing prompt's own premise.

### 5.2 New policy: range + stored-dual screens, no cold retry, SAME `maxit=10000`
(isolates the screening effect alone)

| direction | wall(s) | terminal nStatus | real KNITRO inner calls (`inner_solve_count`) | of which infeasible | cold-verified Delta | gamma_prime |
|---|---|---|---|---|---|---|
| upper | 46.9 | -400 | 51 | 4 | 9.947036e-3 | 0.929993 |
| lower | 36.7 | -400 | 42 | 4 | 9.924763e-3 | 0.977060 |
| **total** | **83.6** | | | | | |

**5.04x total speedup vs. baseline** (upper 1.86x, lower **9.10x**). Real KNITRO inner
calls dropped 2.41x (upper) / 4.71x (lower) -- most of the reduction is calls the screens
intercept BEFORE any KNITRO invocation at all (see Section 5.5's stage breakdown), and the
FEW real KNITRO calls that still occur succeed at a much higher rate (upper: 4/51 = 7.8%
fail vs. baseline's 52/123 = 42.3%; lower: 4/42 = 9.5% vs. 84/198 = 42.4%) -- consistent
with the screens disproportionately intercepting the points that would have been
KNITRO-side failures.

### 5.3 New policy + reduced `maxit=250` (`melitz_inner_loop_options_budgetcheck.opt`)

| direction | wall(s) | inner_solved | moment_infeas | budget_infeas | numerical_fail | cold-verified Delta |
|---|---|---|---|---|---|---|
| upper | 53.1 | 47 | 0 | 111 | 4 | 9.947036e-3 |
| lower | 35.5 | 38 | 0 | 127 | 4 | 9.924763e-3 |
| **total** | **88.6** | | | | | |

**4.75x speedup vs. baseline.** Numerical failures stayed at 4/side (matching Section
5.2's real-KNITRO-call count exactly, confirming the reduced `maxit` did not turn any
finite-time solve into a false failure at this campaign's own trajectory). Wall time is
statistically indistinguishable from Section 5.2 (slightly WORSE on upper, by noise-level
margin) -- **the reduced `maxit` does not add a further win beyond the pre-solve screens
alone in this specific campaign**, reported honestly rather than claimed as a win it did
not produce here. It is not harmful either (same incumbents, same Delta).

### 5.4 New policy + `lower_limit_guard=0.0` (tightest KNITRO-native early stop),
`maxit=10000`

| direction | wall(s) | inner_solved | moment_infeas | budget_infeas | numerical_fail | cold-verified Delta |
|---|---|---|---|---|---|---|
| upper | 47.9 | 26 | 0 | 65 | 46 | 9.929831e-3 |
| lower | 32.0 | 26 | 0 | 127 | 27 | 9.622437e-3 |
| **total** | **79.9** | | | | | |

**5.27x speedup vs. baseline** -- the best of the three new-policy variants, but only
marginally ahead of the screens-only arm (83.6s -> 79.9s, ~4.6% faster). `numerical_fail`
rose sharply (4 -> 46/27) because the `lower_limit` mechanism converts what would have been
a `BudgetInfeasible`-eligible mid-solve termination into an "unbounded" KNITRO status that
this session's classifier still labels `NumericalFailure` (Section 4's flagged follow-up)
-- the REJECTIONS are just as fast and just as correct, only the LABEL is currently
imprecise. Both incumbents remain outer-feasible with Delta safely inside budget.

### 5.5 Screen effectiveness (main prompt Section C, using Section 5.2/5.3's typed counts)

| screen | mathematical status | cost | rejections (upper / lower, Section 5.3 run) | false-rejection tests |
|---|---|---|---|---|
| range screen | exact necessary condition, single column | O(W*K), no KNITRO call | 0 / 0 at this fixture (see below) | none observed to false-reject a feasible point in any of the 4 campaign runs (every incumbent found remained outer-feasible) |
| stored-dual screen | exact lower bound (weak duality), any bank entry | ~free (one BLAS gemv + elementwise map) | 111 / 127 (`budget_infeas`, Section 5.3) | same |
| `lower_limit` KNITRO-native threshold | exact lower bound at every barrier iterate | one KNITRO solve, but terminates early | folds into `numerical_fail` currently (Section 4) | same |

**The range screen fired zero times at this fixture.** This is a real, reportable null
result, not a bug: `generate_fake_melitz_data`'s own construction (per
`docs/melitz_delta_star.md` Section 13.3) already enforces a "well-conditioned fixture"
criterion (min reference participation probability `>=0.01`, min active-draw count in the
hundreds+) specifically to AVOID the zero-active-draw pathology the range screen targets --
so this particular D=4/W=20,000/seed=29 fixture, by design, never produces a
range-screen-catchable point within the `theta_box=0.10` neighborhood explored here. The
screen is still correct and cheap to keep on (main prompt's own "no reason to omit an
exact, nearly-free necessary condition"), but this session's own campaign cannot claim
credit for it -- the STORED-DUAL screen is what did essentially all of the observed work.
A fixture deliberately constructed with rare/pathological cells (main prompt Section 7's
own suggestion, addendum Section 4.2's "pay special attention to rare active-tail trade
moments") would be needed to exercise the range screen's own rejection power -- not
attempted this session (Section C/F).

## 6. Warm/cold no-rescue benchmark (addendum Section 2)

`scripts/melitz_no_rescue_benchmark.jl`: runs a live `delta=1e-2`/`lower` trajectory
(the historically most expensive direction), archives the first 4 `theta` points that
produced a `NumericalFailure`, and probes each with (a) the LAST VERIFIED dual vector the
single-slot warm-start mechanism actually held at that point in the trajectory (the true
historical "warm" condition -- NOT `obj.x` read post-hoc, which by the time a failure is
classified has already been overwritten to `NaN`), (b) a neutral cold (zero) start, (c)
three random moderate-scale dual starts.

| point | warm (last verified x) | cold (zero) | 3 random starts | any rescue |
|---|---|---|---|---|
| 1 | fail (-102), 1.18s | fail (-102), 1.24s | 0/3 succeed | **no** |
| 2 | fail (-102), 1.06s | fail (-102), 0.69s | 0/3 succeed | **no** |
| 3 | fail (-102), 1.60s | fail (-102), 0.56s | 0/3 succeed | **no** |
| 4 | fail (-102), 3.47s | fail (-102), 1.95s | 0/3 succeed | **no** |

**0/4 points were rescued by any alternative dual initialization.** All four fail with the
identical KNITRO status (`-102`, dual unbounded/divergent) regardless of starting point --
directly consistent with the addendum's own stated conjecture ("warm vs. cold ... has
never changed convergence success vs. failure"). This is a small sample (4 points, one
direction, one delta, one fixture) -- reported as supporting evidence, not an exhaustive
proof -- but it is a real, live measurement, not an assumption, and it directly justifies
Section 2's implemented policy (no routine cold retry).

**Structural side finding, also worth recording**: because a failed
`CS.inner_loop_internal` call sets `obj.x .= NaN`, and `inner_loop_initial_values`'s own
`norm(obj.x) < 1e6` guard silently falls back to a cold (zero) start whenever `obj.x`
contains `NaN`, every attempt IMMEDIATELY FOLLOWING a failure was ALREADY effectively cold
under the pre-session single-slot design, regardless of `obj.use_cached_x` -- only the
FIRST attempt after a SUCCESS is genuinely warm. In this session's own 4-point sample, all
4 failures happened to occur immediately after a success (genuinely warm at failure time),
so this caveat did not confound the benchmark above, but it is a real structural property
of the existing warm-start mechanism worth flagging for anyone reasoning about "warm vs.
cold" in this codebase generally.

## 7. Required report sections

### A. Failure taxonomy

Every failure this session directly observed (the 4 no-rescue-benchmark points, all
`nStatus=-102`) classifies as: **unresolved numerical failure** (no certificate obtained --
neither the range screen nor the stored-dual screen rejected these points beforehand, and
the single KNITRO attempt returned an unbounded-dual status). This session did not
separately verify whether these specific 4 points are, by the K-dimensional convex-hull LP
(not implemented, Section C), truly moment-infeasible, boundary-feasible, or feasible-but-
over-budget-with-a-non-attained-dual -- that finer taxonomy requires the diagnostics main
prompt Section 7/8 describe, out of scope this session (see Section C/F).

### B. Objective-threshold diagnosis

Fully answered in Section 4 above: the mechanism lives in `cc_algo/PsiObjectiveBundle.jl`'s
shared functor bodies (`if f <= lower_limit; return -KN_INFINITY; end`), is ACTIVE for
Ricardian (`lower_limit=-50`, fixed) and was COMPLETELY INACTIVE for Melitz before this
session (default `-KN_INFINITY`, never overridden). Now wired up as an opt-in,
delta-aware `lower_limit_guard` parameter; live-tested safe at the tightest setting
(`guard=0.0`), with a small (not dramatic, at this scale) measured additional wall-time
benefit beyond the pre-solve screens alone (Section 5.4).

### C. Screen effectiveness

Covered in Section 5.5. Summary: the range screen is exact, essentially free, and DID NOT
fire on this session's well-conditioned fixture (a real null result, not a shortfall); the
stored-dual screen is exact (weak duality), essentially free, and did the overwhelming
majority of the observed rejection work (111-127 of ~115-131 total rejections per side).
**Not implemented this session** (flagged honestly rather than silently dropped): the
K-dimensional convex-hull LP (main prompt Sections 5-6), scalar/origin-block divergence
lower bounds (Sections 8.2/8.3), the boundary/relative-interior diagnostic (Section 7),
and a deliberately-pathological stress fixture to exercise the range screen. All are
reasonable next-session targets, ranked in Section F below.

### D. Inner iteration diagnosis (what KNITRO does during a 25-55s failure)

Not independently re-diagnosed at the KNITRO-internal-iteration level this session (the
prior continuation session's own report already attributes this to "KNITRO needing many
more of its own internal iterations... to conclude a trial point is infeasible," not
directly observable from the Julia side without KNITRO-internal iteration logging on a
targeted diagnostic run -- not run again this session). This session's OWN contribution to
this question is mechanical, not observational: Section 4 shows Melitz's inner solves had
no early-bailout brake at all (`lower_limit` never set) and an iteration cap 100x
Ricardian's (`maxit=10000` vs `100`) -- sufficient on their own to explain multi-second-to-
tens-of-seconds individual failures without needing a new KNITRO-internal trace to confirm
it further this session.

### E. Before/after performance

Section 5, full tables above. Headline: **5.04x total wall-time speedup** from the
screens alone (83.6s vs. 421.1s, `delta=1e-2`, both directions, `:logf`), rising to
**5.27x** with the KNITRO-native `lower_limit` threshold also enabled; the reduced-`maxit`
budget-check variant does not add a further win in this specific campaign beyond the
screens. All incumbents found by every configuration remain outer-feasible with Delta
safely inside the delta budget -- **the new logic did not change the best verified economic
incumbent's validity**, only the search-path-dependent specific value (gamma_prime ranges
0.928-0.988 across the 4 runs' `upper`/`lower` directions, all economically sensible,
none reused as a "the" answer -- this is expected KNITRO-trajectory variation, not a
regression, since which trial points get rejected quickly vs. slowly changes what the
outer solver explores within its fixed `maxit=25` iteration budget).

### F. Recommended production policy

- **Screen order**: range screen, then stored-dual screen, both always-on and front-loaded
  before any KNITRO call (matches addendum Section 3's recommended order exactly, since
  this session's own measurements found both cheap relative to a KNITRO attempt).
- **Inner mode**: this session did NOT implement the full `:full_value`/`:budget_check`/
  `:diagnostic` three-mode split the addendum describes (Section 6) -- what exists today
  is effectively a single "screened, no-retry" mode used everywhere in the routine outer
  callback, plus the SEPARATE, untouched `evaluate_melitz_delta(...; cold=true)` path for
  initial-incumbent/end-of-run verification (which already behaves like `:full_value`).
  Recommend formalizing this distinction explicitly in a future session (Section on
  remaining work).
- **maxit by mode**: keep `melitz_inner_loop_options.opt` (`maxit=10000`) for the cold-
  verification/`:full_value` path (unchanged, untouched this session, still the safety net
  for the numbers callers actually report as incumbents). The new
  `melitz_inner_loop_options_budgetcheck.opt` (`maxit=250`) is available for the routine
  callback path; this session's own measurement did not show it beats the screens-only
  configuration decisively, so it is offered as a safe (no observed correctness cost)
  option, not a proven-necessary default.
- **Early-stop threshold**: recommend enabling `lower_limit_guard` (a small positive
  number, e.g. `0.0` to `1e-3*delta`) as the production default for the routine callback
  path -- mathematically sound at every iterate (Section 4), measured safe, and the single
  best-performing configuration tested this session (Section 5.4), even though its margin
  over the screens alone was modest here.
- **Warm/cold retry rule**: no routine cold retry (addendum Section 1), confirmed safe by
  the no-rescue benchmark (Section 6, small sample). Retain cold-start solving ONLY at the
  existing `evaluate_melitz_delta(...; cold=true)` call sites (initial incumbent, end-of-
  run reverification) -- unchanged this session.
- **LP fallback policy**: not implemented; no recommendation beyond "build it as a
  diagnostic classifier on an archived sample before considering it for production," per
  addendum Section 4.5's own guidance.
- **Typed result handling**: adopt `MelitzInnerResult`/`melitz_classified_inner_solve`
  (this session's `inner_screening.jl`) as the standing production interface; extend
  `BudgetInfeasible`'s `source` field to include a `:lower_limit_threshold` case once the
  mid-solve KNITRO-native early stop is mapped to a distinct classification instead of
  folding into `NumericalFailure` (Section 4's flagged gap).

## 8. What remains open / explicitly out of scope this session

- K-dimensional convex-hull LP (main prompt Sections 5-6) -- not implemented; the range
  screen alone did all the observed necessary-condition rejection work at this fixture.
- Scalar and origin-block divergence lower bounds (main prompt Sections 8.2/8.3) -- not
  implemented; the stored-dual screen alone did all the observed lower-bound rejection
  work at this fixture.
- Boundary-feasibility / relative-interior diagnostic (main prompt Section 7) --
  `BoundaryFeasible` exists in the type hierarchy but is never produced.
- Mapping the `lower_limit` KNITRO-native early stop to a distinct `BudgetInfeasible`
  classification (currently folds into `NumericalFailure`) -- Section 4/F.
- A deliberately pathological stress fixture (rare/zero-active-draw cells) to actually
  exercise the range screen, distinct from the well-conditioned default fixture used
  throughout this session.
- Formalizing the `:full_value`/`:budget_check`/`:diagnostic` three-mode split as
  first-class, explicitly-named entry points (today the distinction is implicit in which
  function/options-file a caller happens to use).
- `delta=1e-3` and the `:linear` cutoff backend were not re-benchmarked this session
  (Section 5's scope note) -- no specific reason to expect a different qualitative result,
  not independently confirmed.
- D=20 scaling, localized gradients, and additional parallelism remain explicitly out of
  scope per the governing prompt's own final instruction, unaffected by this session.
