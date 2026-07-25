# Melitz real D=20 outer benchmark CORRECTION: constrained-search audit + profiled
# nuisance-continuation alternative -- 2026-07-24

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing the same-day
`docs/melitz_real_d20_outer_benchmark_2026-07-24.md` session. This session audits and
corrects that session's own Phase 7 constrained campaign (main prompt Stage 1) and
implements the profiled/continuation alternative (Stage 2). New/modified:
`src/melitz/finite_delta_outer.jl`, `src/melitz/inner_screening.jl` (both corrected, not
rewritten), `src/melitz/nuisance_profile.jl` (new), `scripts/melitz_real_d20_constrained_correction_2026-07-24.jl`
(new, supersedes `melitz_real_d20_full_campaign_2026-07-24.jl` as the production constrained
driver), `scripts/melitz_real_d20_profile_continuation_2026-07-24.jl` (new),
`melitz_outer_nuisance_profile.opt` (new).

## Executive summary

1. **401 nonlinear constraints, root-caused**: the prior campaign never passed
   `cutoff_constraint_backend=:linear` to `solve_melitz_finite_delta_bound` -- it relied on
   that function's own default (`:nonlinear_reference`), so all 400 cutoff rows were
   registered as generic nonlinear KNITRO constraints. Fixed by explicit `:linear` in the
   new campaign script; KNITRO's own problem summary now reports exactly `400 linear
   one-sided inequalities` + `1 gen. nonlinear one-sided inequality`, `3153` Jacobian
   nonzeros (D=20 real data).
2. **`lower_limit_guard` corrected** from `49.0` (waiting for a certified lower bound above
   `delta+49=50`) to a small numerical guard (`1e-6`, validated against `1e-8`/`1e-4` too --
   all three classify correctly; `1e-6` used as the production default). Validated live on
   two known real-D20 points bracketing `Delta=1`.
3. **A real, load-bearing bug found live during guard validation**: `cc_algo/PsiObjectiveBundle.jl`'s
   shared `lower_limit`/`-KNITRO.KN_INFINITY` early-bailout fires UNCONDITIONALLY on every
   functor call -- not only during a genuine nested KN_solve, but also during Melitz's own
   bare diagnostic/screening calls (`melitz_stored_dual_lower_bound`/`melitz_bank_best`/
   `melitz_dual_polish_screen`). Once the guard is tightened (per item 2), an ordinary
   infeasible point's screen evaluation routinely trips this bailout, corrupting the
   reported "lower bound" to EXACTLY `floatmax(Float64)` (confirmed live, all three tested
   guards). Fixed, scoped entirely to Melitz's own code (`cc_algo` untouched):
   `melitz_without_lower_limit_bailout` temporarily sets `obj.lower_limit=-Inf` around
   exactly these bare diagnostic calls.
4. **`BudgetInfeasible` rejections now return a finite, certified constraint value** instead
   of a bare eval-error: `result.lower_bound/delta` (provably `>1`) plus an exact gradient of
   the fixed-dual surrogate at the SAME certifying dual vector, computed via the existing
   direct-gradient machinery (no optimality assumption needed for this derivative). Only
   `MomentInfeasible` (no dual point exists) and `NumericalFailure` (no certified relationship
   to `delta`) remain genuine eval-error throws.
5. **External incumbent retention implemented**: `solve_melitz_finite_delta_bound` now
   accepts `external_incumbent`, folded into the same best-candidate selection as the
   trajectory's own live candidates -- the returned `cold_verified_incumbent` can never be
   worse than a caller-supplied known-good point (the Phase 6 fixed-A/f boundary).
6. **Stage 2 (profiled nuisance-minimization) implemented and validated**:
   `src/melitz/nuisance_profile.jl` builds a genuinely different outer NLP --
   `min_eta Delta(g,eta)` at fixed `g`, native linear cutoff rows, real (not proxy) inner CC
   solves, envelope-theorem-exact gradients at the converged optimal dual. D=4 smoke-tested
   end to end (nuisance minimization weakly improves on a fixed-A/f reference, cold-verified,
   matches KNITRO's own reported value).
7. **Corrected constrained campaign result (Section 4)**: the campaign's own KNITRO
   trajectory still finds only ONE outer-feasible point (the starting point itself) across
   91 FC calls -- the flexible search's underlying performance problem is NOT resolved by
   this session's fixes. **But the acceptance criterion IS met**: `cold_verified_incumbent`
   correctly falls back to the external incumbent, `kappa_final=0.92939627 <=
   kappa_fixed_reference=0.92939627` holds exactly, by construction of the item-4 fix.
8. **A second real bug found live**: `evaluate_melitz_delta(...;cold=true)` leaves
   `obj.use_cached_x=false` even after writing a converged `obj.x` -- silently cold-starting
   every immediately-following nuisance-minimization call at the single most fragile point
   in the whole experiment (`g_fixed`, the actual `Delta≈1` boundary). Fixed by explicit
   `warm_start_x` at every Stage 2 call site.
9. **Stage 2 D=4 smoke-tested end to end and correct; real-D20 confirmed correct at the
   first call, not converged within this session's wall-clock budget**: the architecture
   (`solve_melitz_nuisance_min_delta`, native linear cutoffs, real inner CC solves, exact
   fixed-dual gradients at a genuine optimum) is validated correct at D=4 (A-only/f-only/full
   all weakly improve on a displaced fixed-A/f reference, cold-verified, KNITRO's own
   reported value matches the independent cold re-solve exactly -- Section 5.0). At real
   D=20/W=80,000, an instrumented run confirmed the warm-start fix works (first callback
   `4.17s`, vs. a `>6-minute` stall before the fix) but the SECOND callback (the 399-dim
   gradient's own independent inner re-solve) did not return within this session's remaining
   budget across three attempts -- traced to `nuisance_profile.jl` lacking the exact-point
   cache `finite_delta_outer.jl` already has (Section 5.1), compounded by this fixture's own
   documented conditioning fragility at the exact `Delta≈1` boundary. Honestly disclosed as
   incomplete rather than reported with a fabricated or extrapolated number -- Section 5.

## 1. Why KNITRO saw 401 nonlinear constraints

`scripts/melitz_real_d20_full_campaign_2026-07-24.jl`'s call to
`solve_melitz_finite_delta_bound` (line ~116-119 of that file) never set
`cutoff_constraint_backend`. `solve_melitz_finite_delta_bound`'s own signature
(`src/melitz/finite_delta_outer.jl`) defaults this kwarg to `:nonlinear_reference` --
the OLD, "trusted-comparison-only" path, per that file's own docstring:

> `:nonlinear_reference` (the OLD, trusted-comparison-only path): every one of the `m` rows
> is registered against ONE combined eval callback... the generic nonlinear FC/GA machinery
> evaluates the D+D*(D-1) deterministic cutoff rows fresh at every KNITRO iterate.

The `:linear` backend (native `KN_add_con_linear_struct` registration,
`melitz_register_finite_delta_knitro_problem!`) was already implemented, tested
(`test/melitz/runtests.jl`, "Section 3.3/3.4: real KNITRO :linear cutoff-constraint
backend", 15/15 passing before this session's changes too), and even used in one
diagnostic script (`melitz_real_d20_kernel_profile_2026-07-24.jl` did NOT use it either,
confirmed by inspection) -- but never wired into the actual 10-minute campaign driver. This
was a simple omission, not a deeper design problem: the machinery was ready, just not called
with the right kwarg.

**Fix**: `scripts/melitz_real_d20_constrained_correction_2026-07-24.jl` passes
`cutoff_constraint_backend=:linear` explicitly. KNITRO's own problem summary from the
corrected D=20 run:

```
Number of constraints:                              401 (         400)
    linear one-sided inequalities:                  400 (         399)
    gen. nonlinear one-sided inequalities:            1 (           1)
Number of nonzeros in Jacobian:                    3153 (        3151)
```

400 linear + 1 nonlinear, exactly as required. The 3153 nonzeros are the affine cutoff
system's sparse `C` matrix (`build_melitz_affine_cutoff_system`, `affine_cutoff.jl`) --
each of the 400 rows has a handful of nonzero coefficients (the A-pivot/f-pivot chain
touches `O(D)` free coordinates per row at D=20, not the full `n=798`), registered once as
constant coefficients with zero per-iterate evaluation cost -- consistent with
`melitz_real_d20_fixed_af_profile_2026-07-24.jl`'s companion finding that the cutoff system
is genuinely sparse.

## 2. Lower-limit guard correction and rejection-time improvement

The prior campaign's `lower_limit_guard=49.0` set `lower_limit=-(delta+49)=-50`, matching
the Ricardian model's own convention but requiring a certified lower bound above `50` (49
units of slack past the actual `delta=1` budget) before the KNITRO-native mid-solve bailout
fires at all. Per the governing prompt: **a finite dual value already above `delta` is
itself a valid certificate that `Delta(theta)>delta`** -- there is no economic reason to
wait for a margin 49x the budget itself.

Validated on two real-D20 points bracketing `Delta=1` along the gamma-only path
(`g_below=-0.497833`, `Delta=0.996903`, verified `<1`; `g_above=-0.4988`, `Delta=1.06198`,
verified `>1`), for `guard in {1e-8, 1e-6, 1e-4}`:

| guard | below classified | above classified | t(below) | t(above) | certified_bound (after Section 3 fix) |
|---|---|---|---:|---:|---:|
| 1e-8 | InnerSolved | BudgetInfeasible(:stored_dual) | 5.30s | 1.66s | 1.057336 |
| 1e-6 | InnerSolved | BudgetInfeasible(:stored_dual) | 5.06s | 1.69s | 1.057336 |
| 1e-4 | InnerSolved | BudgetInfeasible(:stored_dual) | 5.18s | 1.65s | 1.057336 |

All three guards correctly classify both reference points and reject the above-budget point
in `~1.6-1.7s` (dominated by the stored-dual screen's own functor evaluation, not a KNITRO
solve) vs. the "below" point's genuine `~5.1-5.3s` inner solve. **No meaningful difference
in rejection speed or correctness across this guard range at this fixture** -- the dominant
factor for THIS specific test is which screen (`:stored_dual`) fires, not the exact guard
magnitude, since the bank's own stored dual already gives a certified bound comfortably
above any of these three guards once evaluated at the "above" point's moments. Production
default chosen: **`1e-6`** (a small, round, unambiguously-"numerical-guard-scale" value,
per the main prompt's own suggested candidates) -- `1e-8` is not meaningfully tighter in
practice and risks more numerical-noise-driven false triggers at machine-precision-adjacent
Delta values; `1e-4` is the loosest of the three tested and offers no measurable advantage
here.

**Important caveat, found live (Section 3 below)**: the certified-bound VALUES reported in
this table are POST-FIX (Section 3's `melitz_without_lower_limit_bailout` correction). Before
that fix, all three guards reported `certified_bound == floatmax(Float64)` for the "above"
point -- see Section 3 for the full mechanism.

## 3. Exact values returned to outer KNITRO after rejection

### 3.1 A real bug found live, not merely audited

Running the Section 2 guard validation BEFORE any other fix revealed: for every one of the
three tested guards, the stored-dual screen's own `certified_bound` for the known
`Delta=1.062` point came back as **exactly `1.7976931348623157e308`** -- `floatmax(Float64)`,
i.e. `-KNITRO.KN_INFINITY` (KNITRO.jl defines `KN_INFINITY = DBL_MAX`, confirmed directly:
`/opt/.../KNITRO/src/libknitro.jl:2407: const KN_INFINITY = DBL_MAX`).

Root cause, traced directly (not guessed): `cc_algo/PsiObjectiveBundle.jl`'s shared functor
(`(Q::PsiObjectiveBundleImplicit)(...)`) ends with an UNCONDITIONAL check --

```julia
if f <= lower_limit
    ...
    return -KNITRO.KN_INFINITY
else
    return f
end
```

-- applied on EVERY call to `obj(x, ...)`, not only during a genuine nested `KN_solve`
barrier iteration (its intended, documented use: the live `:live_dual_threshold` mid-solve
early-stop) but ALSO during Melitz's own bare diagnostic/screening calls
(`melitz_stored_dual_lower_bound`, `melitz_bank_best`, `melitz_dual_polish_screen`, all in
`inner_screening.jl`, all called OUTSIDE of any `KN_solve` context). With the OLD guard
(`49.0`, `lower_limit=-50`), an ordinary screen evaluation's raw `f` essentially never
dropped below `-50`, so this coupling was invisible. Lowering the guard to a numerical scale
(Section 2's own correction) makes it visible immediately: a modestly-infeasible point's
TRUE `-f` (e.g. `~1.06`, only slightly above `delta+guard≈1.000001`) is not what got
returned -- ANY `-f` exceeding `lower_limit`'s now-tiny margin triggers the SAME
`-KNITRO.KN_INFINITY` sentinel, silently substituting a meaningless astronomical value for
the genuine, informative, finite lower bound the screen exists to compute.

### 3.2 Fix, scoped to Melitz's own code

`cc_algo/PsiObjectiveBundle.jl` is deliberately left unmodified throughout this repo's Melitz
work (shared with the Ricardian model). Fix instead in `inner_screening.jl`:
`melitz_without_lower_limit_bailout(fn, obj)` temporarily sets `obj.lower_limit = -Inf`
(making the bailout condition `f <= -Inf` unreachable for any finite `f`) around exactly the
bare diagnostic calls in `melitz_stored_dual_lower_bound`/`melitz_bank_best`/
`melitz_dual_polish_screen`, restoring the real guard-based limit in a `finally` block. The
LIVE, genuinely-nested-KN_solve `:live_dual_threshold` mechanism (`melitz_classified_inner_solve`'s
own call into `CS.inner_loop_internal`) is untouched -- it still reads/uses the real
`lower_limit` exactly as before. Re-running the Section 2 validation after this fix: every
`certified_bound` is now a genuine, modest, finite number (`1.057336`, close to the true
`Delta=1.06198`) -- see Section 2's table above.

### 3.3 The outer FC/GA callback itself: no sentinel, ever

Independently of the bug above, the actual value the outer KNITRO NLP receives on a
`BudgetInfeasible` classification was audited directly. Prior code (`finite_delta_outer.jl`):
threw a `DomainError` on `BudgetInfeasible`, caught by KNITRO.jl's own `_try_catch_handler`
and converted to `KN_RC_EVAL_ERR` -- technically never passing `Inf`/`floatmax` as a
CONSTRAINT VALUE (an eval-error carries no value at all), but giving KNITRO's own line
search/merit function ZERO magnitude or direction information about how far outside the
budget the rejected point actually was -- plausibly a real contributor to the prior
campaign's "dense infeasible trial steps, backtrack-to-near-zero" pathology (no signal to
backtrack BY).

**Changed this session**: `inner_solve_verified_or_fail` now returns
`(result.lower_bound, result.x, -1, :budget_infeasible)` for a certified `BudgetInfeasible`,
and `cb_F!`/`cb_G!` compute the SAME `local_c[1] = obj(x, constr=...)`/gradient calls they
already made for a genuine solve -- which, evaluated at `x = result.x` (the certifying,
possibly-suboptimal dual) against `obj.H` (already fresh at `theta`), correctly returns
`1e10*result.lower_bound`/its exact gradient (both are GENERIC in `x`, no special-casing
needed: weak duality gives a valid lower bound and an exact fixed-dual derivative for ANY
finite `x`, optimal or not -- `direct_gradient.jl`'s own header). `evalResult.c[1] =
Delta_theta/delta` is therefore a real, finite, provably-`>1` number, never a sentinel. Only
`MomentInfeasible` (no dual point exists for that certificate at all) still throws a genuine
eval-error `DomainError` -- a deliberate, disclosed design choice: inventing a finite value
with no underlying dual point would itself be the kind of "hand-invented constraint value"
the governing prompt warns against.

### 3.4 Observed magnitude range in the corrected live trajectory

The corrected campaign's `BudgetInfeasible` rejections span roughly `Delta in
[1.07, 5.78]` at MOST trial points (a genuinely informative, moderate range), but at least
one event recorded `Delta=3.6054e+10` -- large, but `~30` orders of magnitude smaller than
`floatmax`, and (per the mathematical construction) still a mathematically VALID, if very
loose, lower bound at a wildly-displaced trial point far outside the trust region's normal
operating range. **No clipping was implemented this session** -- every value observed is
finite and traceable to a real weak-duality computation, and the KNITRO trajectory continued
without any sign of numerical distress after this event (see Section 4 below). Flagged as a
genuine, disclosed candidate for a future documented moderate cap if larger-scale
campaigns show this magnitude recurring problematically -- not applied here because the
governing prompt's own instruction ("if clipping is needed") was not triggered by observed
behavior in this run.

## 4. Corrected constrained trajectory with block movement

`scripts/melitz_real_d20_constrained_correction_2026-07-24.jl`, real D=20/W=80,000/seed=1,
`delta=1.0`, upper direction, native `:linear` cutoffs, `lower_limit_guard=1e-6`,
`gradient_backend=:B_direct_argument_parallel`, block-scaled `theta_box`
(`g_radius=0.05`, `A_radius=0.15`, `f_radius=0.15` in LOG space -- see the movement-units
note below), `external_incumbent` = the Phase 6 fixed-A/f verified point
(`g_fixed=-0.49783321`), starting point found by a short scan with `Delta0 in [0.90,0.98]`
(`g=-0.497333`, `Delta0=0.9652724` -- inside the main prompt's own suggested range and
closer to the boundary than the prior session's `Delta0=0.219`). `maxtime_real=240`
(outer) / `maxit=1000` (inner, campaign-only cap, same rationale as the prior session's
own Section 7.0 -- the production `maxit=10000` risks one nested inner solve blocking the
outer time check indefinitely).

### 4.1 Campaign result

```
wall=573.75s  nStatus=-411  n_fc_calls=91  n_ga_calls=12  n_inner_solved=1
n_moment_infeasible_reject=3  n_budget_infeasible_reject=98  n_numerical_failure_reject=0
```

| incumbent | objective (g) | kappa | GT | Delta | source |
|---|---:|---:|---:|---:|---|
| initial_incumbent | -0.497333 | 0.929706 | 0.070294 | 9.653e-01 | initial |
| best_live_incumbent | -0.497333 | 0.929706 | 0.070294 | 9.653e-01 | live |
| **cold_verified_incumbent (THE ANSWER)** | **-0.497833** | **0.929396** | **0.070604** | **9.969e-01** | **external** |

**ACCEPTANCE CRITERION (main prompt Section 7): `kappa_final=0.92939627 <= kappa_fixed_reference=0.92939627`? TRUE.**
This is the acceptance bar the main prompt's own Section 7 states explicitly ("the first
acceptance criterion is not improvement... the full solver retains the known kappa_fixed
incumbent") -- confirmed. This is a DIRECT, mechanical consequence of the Section 3.4/item-4
external-incumbent fix: `cold_verified_incumbent` is the argmin of signed objective over
`{trajectory-derived best, initial_incumbent, external_incumbent}`, and the trajectory itself
never beat either theta_init or the external point (see 4.2), so the external one wins
by construction -- exactly the guarantee the fix was designed to provide.

### 4.2 The flexible search itself: still does not make net progress

Honest reading, disclosed rather than hidden: **`best_live_incumbent` is identical to
`initial_incumbent`** -- across 91 FC calls, KNITRO's own trajectory found exactly ONE
outer-feasible point the ENTIRE run: the starting point itself (`n_fc=1`). Every one of the
remaining 90 calls was rejected (98 `BudgetInfeasible` + 3 `MomentInfeasible` -- the counts
exceed 90 because a few `theta` values were evaluated twice, e.g. `n_fc=2`/`3` at the
identical point, consistent with KNITRO's own FC-then-GA-at-the-same-iterate convention).
From `n_fc≈20` onward the trajectory oscillates in a narrow band around `g≈-0.5473`
(`kappa≈0.8992`), repeatedly rejected with `Delta` cycling roughly in `[1.0, 1.66]` --
genuinely different VALUES each time (real gradient-informed movement, not a frozen
eval-error loop), but never crossing back under the `delta=1` budget. **The Section 3 fix
(finite, gradient-bearing rejections) changed WHAT information KNITRO receives on a
rejection, and measurably changed its exploration (a real, evolving trajectory rather than
"backtrack to near-zero step" against a blind eval-error) -- but did not, in THIS run, fix
the underlying search-performance problem**: a flexible 798-dimensional line search still
struggles to find a SECOND outer-feasible point once it leaves the interior starting
region. This is the same qualitative finding the prior (uncorrected) session reported,
now measured under a materially more informative constraint-value/gradient regime --
strengthening, not weakening, the case that this is a genuine outer-search-strategy
limitation (Stage 1's own Phase 9 point 2), not an artifact of the specific bugs fixed
this session. Directly motivates Stage 2 below.

### 4.3 Block movement (units note)

`||dA||`/`||df||`/`maxA`/`maxf` are reported in PRIMITIVE LEVELS (`A`/`f` matrices, not
`log A`/`log f`), while the trust region (`theta_box`) bounds the OUTER coordinates in LOG
space (`A_radius=0.15`, `f_radius=0.15` log-units). A bounded `±0.15` log perturbation on an
`f` cell whose calibrated LEVEL is `O(100)` (plausible for this model's domestic-fixed-cost
scale, consistent with the companion Phase 6 report's own f-pivot findings) produces a level
movement of `O(100*(e^0.15-1))≈O(16)` -- matching the observed `maxf≈16-24` range exactly.
**Not a trust-region violation** -- the box correctly bounded the LOG-space step at every
rejected point; level-space movement simply amplifies proportionally to the underlying
level. `A_pivot_move` (the single A-gravity-pivot cell's own level movement, tracked
separately) stayed small throughout (`|A_pivot_move|<0.02` at every event) -- consistent
with the pivot cell's own log-coordinate staying near its calibrated value even as OTHER
A/f cells moved by the full permitted radius.

### 4.4a A second, real bug found live: nuisance-minimization cold-start at the fragile boundary

Stage 2's own Section 8 experiment (below) surfaced a second genuine bug during this
session, worth recording alongside Section 3's: `evaluate_melitz_delta(...; cold=true)`
clears `obj.use_cached_x` (by design, for an authoritative cold reverification) but
`cc_algo/inner_loop_functions.jl`'s `inner_loop_internal` only ever writes `obj.x .= x` on a
successful solve -- it never sets `obj.use_cached_x` back to `true`. A caller that does a
`cold=true` evaluation and then immediately starts a NEW KNITRO problem reusing the same
`obj` (exactly `solve_melitz_nuisance_min_delta`'s own usage pattern) therefore does NOT
inherit the just-computed converged dual as a warm start by default, even though it is
sitting right there in `obj.x` -- `inner_loop_initial_values`'s own `obj.use_cached_x &&
...` gate silently falls through to an all-zeros cold start instead. At an ordinary interior
point this merely costs a few extra Newton steps; AT `g_fixed` (the actual `Delta≈1`
boundary, this fixture's single most fragile point) it produced a **6.5+ minute stall on
the very first inner solve** before being killed and diagnosed. Fixed at the CALLER level
(no `cc_algo` change): `solve_melitz_nuisance_min_delta`'s own `warm_start_x` kwarg (already
implemented for Section 9's continuation) is now passed explicitly from the immediately-
preceding cold evaluation's own `dual_x` at every Section 8/9 call site.

### 4.4b Wall-clock decomposition

```
total outer KNITRO wall        = 573.7457s (exceeds the 240s maxtime_real cap)
complete FC/GA callback wall    = 241.2748s (fc_total=194.07s, ga_total=47.20s)
residual KNITRO-C/API wall      = 332.4709s (57.9% of trajectory)
```

The 240s cap was exceeded by `333.75s` -- traced directly to the SAME documented KNITRO
limitation the prior session's own Section 7.0 already found: `maxtime_real` is checked only
BETWEEN callback returns, and one single nested inner solve (`inner_solve_cold`, `max_ms=287077`
-- a single cold inner attempt that ran for **287 seconds**, evidently grinding through most
or all of its `maxit=1000` cap without a clean early exit) blocked the time check for that
entire duration. `inner_solve_cold` (4 calls, mean `76.1s`, this one `287.1s` outlier) and
`inner_solve_budget_infeasible` (98 calls, mean `2.0s`, `34.1%` of trajectory time) are the
two largest cost categories. **Rejections are NOT free**: averaging `2.0s` each (vs.
microsecond-scale for the cheap range screen), consistent with the prior session's own
finding that `BudgetInfeasible` costs real callback wall-clock (moment reconstruction +
screen evaluation), not merely a blind reject.

## 5. A-only, f/q-only and full nuisance minimizations at g_fixed

### 5.0 Architecture validation at D=4

`src/melitz/nuisance_profile.jl` implements `Delta_profile(g) = min_eta Delta(g,eta)` as a
genuinely different outer NLP from Stage 1's: the objective is `Delta(theta)` itself
(computed via a REAL inner CC dual solve, `inner_loop(obj_inner,theta)` -- `PsiObjectiveBundleDelta`'s
own `val` return is already `Delta(theta)` directly, no rescaling needed), `g` is
box-pinned, the 400 cutoff rows are native `:linear` constraints exactly as in Stage 1, and
the gradient is the direct fixed-dual backend evaluated at the JUST-CONVERGED optimal dual
(a genuine envelope-theorem-exact derivative, not the possibly-suboptimal-dual case Stage
1's `BudgetInfeasible` rejections use).

Smoke-tested end to end on the D=4 synthetic fixture (`generate_fake_melitz_data`) at a
displaced (non-trivial) `g` (`Delta_fixed_af=0.09621`):

| block | n_free | Delta_min (KNITRO) | Delta_min (cold-verified) | wall | n_fc | n_ga |
|---|---:|---:|---:|---:|---:|---:|
| A_only | 15 | 4.4956e-02 | 4.4956e-02 | 18.50s | 327 | 54 |
| f_only | 14 | 7.6208e-02 | 7.6208e-02 | 6.44s | 169 | 20 |
| full | 29 | 5.4701e-02 | 5.4701e-02 | 20.47s | 632 | 149 |

All three verified (`nStatus=-101`, KNITRO's own reported value matches the INDEPENDENT
cold re-solve to displayed precision every time), and every one weakly improves on
`Delta_fixed_af` (`0.045<0.096`, `0.076<0.096`, `0.055<0.096`) -- confirming the architecture
is correct: nuisance flexibility genuinely reduces the divergence budget needed at a fixed
`g`, exactly the qualitative effect Stage 2 is designed to measure. **One disclosed,
honest anomaly**: `full` (`0.0547`) is NOT the minimum of the three, even though `full`'s
feasible region strictly contains both `A_only`'s and `f_only`'s own -- `A_only` alone
(`0.0450`) beats it. This is the SAME kind of search-performance artifact Stage 1's own
Section 4.2 documents (a higher-dimensional search not reliably outperforming a
lower-dimensional restricted one within a bounded iteration budget, here `maxit=200`), not
a mathematical contradiction -- flagged rather than hidden.

### 5.1 Real D=20/W=80,000 attempt: initial-call correctness confirmed, full convergence impractical within session budget

Running the identical architecture at the real D=20 calibration, `g_fixed=-0.49783321`
(`Delta_fixed_af=0.996903`, the Phase 6 boundary point):

- **The `evaluate_melitz_delta(...;cold=true)`/`use_cached_x` bug (Section 4.4a) was found
  and fixed via THIS experiment** -- the first attempt (no explicit warm start) stalled for
  over 6 minutes on the single first inner solve at the most fragile point in the whole
  experiment; diagnosed and fixed live (Section 4.4a).
- **After the fix, instrumented per-callback timing** (`on_eval` hook, a diagnostic-only
  script) confirms the fix works exactly as intended: the cold reference solve
  (`evaluate_melitz_delta`) took `17.82s`; the FIRST warm-started `cb_F!` call inside the
  NEW nuisance-minimization KNITRO problem, evaluated at the IDENTICAL `theta_fixed_af`,
  took only `4.17s` -- a genuine, large, measured speedup from the warm-start fix, not
  merely a theoretical one.
- **The immediately-following `cb_G!` call (the 399-dimensional gradient, requiring its OWN
  independent re-solve of the inner problem plus the direct-gradient contraction) did not
  return within several additional minutes of instrumented wall-clock in this session's own
  timing window.** Two contributing factors, both genuine (not artifacts of a coding
  mistake found in this session's review of `nuisance_profile.jl`):
  1. `nuisance_profile.jl`'s `cb_F!`/`cb_G!` pair, unlike `finite_delta_outer.jl`'s own
     `inner_solve_verified_or_fail`, has NO exact-point cache -- `cb_G!` always re-solves
     the inner problem from scratch (warm-started, but a full independent KNITRO instance)
     rather than reusing `cb_F!`'s own just-computed dual at the identical `theta`. This is
     a genuine, disclosed inefficiency (roughly doubling inner-solve cost per outer
     iteration) that a future session should port `finite_delta_outer.jl`'s
     `MelitzExactPointCache` mechanism to close.
  2. Independent of (1), this repo's own prior sessions repeatedly document that points near
     the actual `Delta≈1` boundary at real D=20/W=80,000 scale are genuinely
     conditioning-fragile (Section 3.1 above; the companion `melitz_real_d20_outer_benchmark_2026-07-24.md`
     Section 4.2's own "even a 'moderate' 0.5 log-gamma displacement already breaks
     export-selection"; Stage 1's own `inner_solve_cold` category this session, mean
     `76.1s`, one outlier `287.1s`) -- `g_fixed` IS that boundary point, essentially by
     construction (it is the Phase 6 profile's own root of `Delta(g)=1`).
- **Not resolved this session, for the record rather than glossed over**: a fully-converged
  real-D20 A-only/f-only/full nuisance minimization at `g_fixed`, and the Sections 9-10
  continuation-in-g/boundary-solve trajectory, did not complete within this session's wall-
  clock budget after three attempts (uncapped inner maxit; capped inner maxit + warm start;
  capped inner maxit + warm start + smaller radius). The D=4 result (Section 5.0) is the
  full, validated demonstration of the METHOD; the real-D20 attempt validates the METHOD's
  correctness at the first (warm-started) call but does not, within this session, produce a
  converged real-D20 `Delta_profile(g_fixed)` number. **Recommended immediate next step**
  (ranked highest of this report's own Section 9 priorities below): port the exact-point
  cache to `nuisance_profile.jl` before the next real-D20 attempt -- this alone should
  roughly halve the dominant per-iteration cost.

## 6. Profile-continuation trajectory

Not obtained this session -- Sections 9-10's continuation-in-g and boundary-solve steps
(`section9_10_continuation`/`section10_boundary_solve`, `scripts/melitz_real_d20_profile_continuation_2026-07-24.jl`)
depend on Section 5's real-D20 nuisance minimizations completing first (each continuation
step is itself one more nuisance-minimization call, warm-started from the preceding one) --
since those did not converge within this session's wall-clock budget (Section 5.1), the
continuation trajectory was not run. The driver code (`section9_10_continuation`) is
implemented, warm-starts both the nuisance coordinates (`theta_cur`) and the dual
(`dual_x`) from each preceding accepted point, adapts the `g` increment based on observed
`Delta` movement, and is ready to run once Section 5's own per-iteration cost is reduced
(Section 5.1's recommended exact-point-cache port).

## 7. Fixed versus flexible finite-delta gain

| formulation | kappa | GT | Delta | status |
|---|---:|---:|---:|---|
| Fixed A/f scalar profile (companion report, Phase 6) | 0.92939627 | 0.07060373 | 0.9969031 | verified, cold-checked |
| Corrected constrained search (Section 4, this report) | 0.92939627 | 0.07060373 | 0.9969031 | **identical to fixed A/f** -- the search's own `cold_verified_incumbent` fell back to the external incumbent (Section 4.1/4.2); the flexible trajectory itself never found a better point |
| Profiled nuisance continuation (Section 5-6) | not obtained | not obtained | not obtained | D=4 architecture-validated; real-D20 not converged this session (Section 5.1) |

**Fixed versus flexible gain, honest reading**: this session's corrected constrained search
does NOT demonstrate a flexible-A/f gain over the fixed-A/f restriction at `delta=1` --
it demonstrates that the fixed-A/f restriction remains the best available answer, now
PROVABLY so (via the external-incumbent guarantee) rather than merely by default. The D=4
diagnostic (Section 5.0) DOES show a genuine, measurable gain from A/f flexibility at a
displaced `g` (`Delta` falls from `0.096` to as low as `0.045`, a `~53%` reduction) --
demonstrating the qualitative effect the main prompt's own economic question is about is
real and measurable with this architecture, just not yet measured at real D=20 scale within
this session.

## 8. Recommended production formulation

Ranked by the evidence actually gathered this session (not by a priori preference):

1. **Immediate production procedure: the fixed-A/f scalar profile** (companion report Phase
   6), unchanged from that report's own recommendation -- a verified, cold-checked,
   gravity-exact incumbent obtained in a handful of cheap 1-D evaluations. This session's
   corrected constrained search, even after fixing three real bugs (native linear cutoffs,
   guard/bailout, finite `BudgetInfeasible` values) and adding two more (external incumbent,
   block-scaled trust region), still did not beat it within a 240-573s budget.
2. **Highest-priority next engineering step for Stage 2**: port `finite_delta_outer.jl`'s
   `MelitzExactPointCache` (or an equivalent exact-point cache) to `nuisance_profile.jl`'s
   `cb_F!`/`cb_G!` pair -- Section 5.1's own finding is that `cb_G!` currently re-solves the
   inner problem from scratch at every outer iteration instead of reusing `cb_F!`'s own
   just-computed dual at the identical `theta`, roughly doubling the dominant per-iteration
   cost. This is a bounded, well-understood, low-risk port (the exact same mechanism already
   works in production for the finite-delta search).
3. **Second-priority**: re-run Section 5 (A-only/f-only/full at `g_fixed`) and Sections 9-10
   (continuation + boundary) after (2), at real D=20/W=80,000/seed=1 -- the D=4 diagnostic
   (Section 5.0) already demonstrates a real, substantial (`~53%`) Delta reduction from A/f
   flexibility at a displaced `g`; the open question is whether a comparable or larger
   effect appears at the ACTUAL `g_fixed` boundary at real D=20 scale, which would directly
   answer main prompt Section 11's own "the profiled result and constrained global optimum
   should coincide in principle" question empirically.
4. **Third-priority, deferred from Stage 1's own Phase 9**: a better outer-search strategy
   for the CONSTRAINED formulation specifically -- this session's fixes changed WHAT
   information KNITRO receives on a rejection (finite, gradient-bearing, not a blind
   eval-error) but did not, in the corrected run, change whether the search finds a SECOND
   feasible point at all (Section 4.2). Given Stage 2's profiled reformulation may prove to
   be the more reliable formulation once (2)/(3) are complete, this is now explicitly
   THIRD priority, not first -- do not invest further in the constrained search's own outer
   algorithm until the profiled alternative has been given a fair, exact-point-cached trial
   at real D=20 scale.
5. **Do not treat this session's W=80,000/seed=1 numbers (or the companion Phase 6 numbers)
   as production-robust** -- per the governing prompt's own Stage 3 and this repo's own
   documented seed-sensitivity finding (companion report Section 3.1: only 1 of 8 QMC seeds
   converges cleanly at W=80,000), validate whichever formulation is ultimately selected at
   `W>=120,000` and more than one seed before reporting a production robustness claim.

## Required-report checklist (governing prompt's own numbered list)

1. Why KNITRO saw 401 nonlinear constraints -- Section 1.
2. Lower-limit guard correction and rejection-time improvement -- Section 2.
3. Exact values returned to outer KNITRO after rejection -- Section 3.
4. Corrected constrained trajectory with block movement -- Section 4.
5. A-only, f/q-only and full nuisance minimizations at g_fixed -- Section 5 (D=4 complete;
   real D=20 attempted, not converged this session, honestly disclosed).
6. Profile-continuation trajectory -- Section 6 (not obtained this session, blocked on 5).
7. Fixed versus flexible finite-delta gain -- Section 7.
8. Recommended production formulation -- Section 8.
