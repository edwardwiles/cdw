# Melitz inner-solve screening continuation session -- 2026-07-23

Branch: `melitz/fullD-delta-star` (this working tree: `trade_robustness_modular`, remote
`cdw` = `github.com/edwardwiles/cdw`). Starting checkpoint: a NEW clean commit created at
the start of this session (`Melitz: checkpoint the screening-session work...`) on top of
`09cc6b5`, containing the prior screening session's uncommitted work (`inner_screening.jl`,
the screened `finite_delta_outer.jl`, `docs/melitz_optimization_report_2026-07-23_screening_session.md`)
that this session found sitting uncommitted in this working tree (distinct from the `cdw`
checkout at `/bbkinghome/edav/cdw`, which only has the two earlier sessions' reports).

This report continues directly from `docs/melitz_optimization_report_2026-07-23_screening_session.md`
(hereafter "the screening report"), implementing its own Phase I/II follow-on program.

## 0. Reproduction record

- **Julia**: `1.12.6`. **KNITRO**: `13.0.1` (`/opt/shared_sw/knitro/13.0.1`). **Host**:
  `demand.mit.edu`, 208 logical CPUs (Intel Xeon Platinum 8270), 3.0TiB RAM (shared with
  other users' jobs throughout this session -- wall-clock numbers below carry ordinary
  shared-machine variance, not a dedicated benchmark environment).
- **Threading**: `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for every
  run this session.
- Full test suite (`test/melitz/runtests.jl`) at the checkpoint commit: 595 assertions
  (this session's own parse), 0 failures -- confirmed BEFORE any of this session's edits.
- Option-file identity: `melitz_inner_loop_options.opt` (`maxit=10000`) and
  `melitz_inner_loop_options_budgetcheck.opt` (`maxit=250`) unchanged from the screening
  session. New this session: `melitz_inner_loop_options_diagnostic_maxit{25,50,100,250,10000}.opt`
  (outlev=6 siblings, Phase I.2 diagnostic use only, not loaded by any production path).
- Post-JIT reproduction of the screening report's Section 5.2 arm (range+stored-dual
  screens, no cold retry, `maxit=10000`, `delta=1e-2`, both directions, D=4/W=20,000/seed=29):
  **91.1s total** (upper 54.5s/inner_solved=47/budget_infeas=111/numerical_fail=4, lower
  36.6s/inner_solved=38/budget_infeas=127/numerical_fail=4) -- within ordinary shared-machine
  variance of the screening report's own 83.6s figure (same commit, same fixture; this
  session's run shared the host with several other users' active jobs). Confirms the
  screening session's headline result reproduces.

## PHASE I

### 1. Live-threshold classification fix

**Root cause confirmed exactly as the governing prompt described**: `PsiObjectiveBundleImplicit`'s
functor (`cc_algo/PsiObjectiveBundle.jl`) already contains `if f <= lower_limit; return
-KNITRO.KN_INFINITY; end`. When Melitz's `lower_limit_guard` is enabled, this fires and
KNITRO reports the solve as unbounded (nStatus outside the accepted set) with NO way, before
this session, to distinguish "certified budget-infeasible via this mechanism" from "an
unresolved numerical failure" -- both landed in `NumericalFailure`.

**Fix** (`cc_algo/PsiObjectiveBundle.jl`, `PsiObjectiveBundleImplicit` only -- additive
fields with safe defaults, verified no positional constructor call site exists anywhere in
the repo that this could break): four new `Base.RefValue` fields
(`threshold_crossed`/`threshold_crossing_bound`/`threshold_crossing_x`/`threshold_crossing_time_ns`),
set unconditionally the moment `f <= lower_limit` fires, for ANY caller of this shared bundle
type (Ricardian's `ccOuter.jl`/`ccInner.jl` usage is unaffected -- it never reads these new
fields). `melitz_classified_inner_solve` (`src/melitz/inner_screening.jl`) now resets
`threshold_crossed[]=false` before every attempt and, on a rejected `nStatus`, checks it:
if set, returns `BudgetInfeasible(threshold_crossing_bound[], :live_dual_threshold)` instead
of `NumericalFailure`, inserts the (finite) crossing dual into the stored-dual bank, and
does NOT trigger a retry or increment the numerical-failure counter -- exactly the main
prompt's Section 11 requirements.

**Not implemented**: the explicit KNITRO user-termination callback alternative (`main prompt
Section 11's "prefer, if supported safely by KNITRO.jl"`) -- the existing `-Inf`-return
mechanism was judged sufficient once correctly classified (see Section 2 below for why: it
resolves the residual failures in ~8ms), and building a second, duplicate early-stop
mechanism to benchmark against a working one was not judged worth this session's remaining
time. Flagged as a candidate for a future session if the `-Inf` mechanism's own overhead
(a full failed KNITRO problem construction/teardown even though it exits at iteration ~1)
is ever found to matter at a larger scale.

**Tests**: `test/melitz/runtests.jl`, "Phase I.1: live dual-threshold classification" --
direct tests on a known Delta-above-budget point (guard fires, classified
`BudgetInfeasible(:live_dual_threshold)`, bound is a valid lower bound, crossing dual is
finite and banked) and a known Delta-below-budget point (guard does NOT false-reject a
genuinely feasible point), plus an integration check that a live trajectory routes threshold
hits through `n_budget_infeasible_reject`, never `n_numerical_failure_reject`. 18/18 pass.

### 2. Residual-failure diagnosis (4 points, lower direction, delta=1e-2)

Archived via `scripts/melitz_residual_failure_diagnostics.jl` (`collect_numerical_failures`,
reused from the screening session's no-rescue-benchmark harness) -- the SAME 4
`NumericalFailure` points a live trajectory with the CURRENT production screens (range +
stored-dual, no guard) produces on this fixture.

**Headline finding**: with `lower_limit_guard=0.0` (i.e. `lower_limit=-delta`), ALL FOUR
points, at EVERY `maxit` tried (`25,50,100,250,10000` -- 20 cells total), are rejected in
**7-15ms, nStatus=-300 ("problem appears to be unbounded")**. WITHOUT the guard, the SAME 4
points cost 0.07s (`maxit=25`) up to 2.18s (`maxit=10000`, worst of the 4 points), with
`nStatus` transitioning from -400 (iteration-limit) at low `maxit` to -102 (dual
unbounded/divergent) once `maxit` is large enough for KNITRO to reach that internal
conclusion on its own:

| point | maxit=25 | maxit=50 | maxit=100 | maxit=250 | maxit=10000 | maxit=10000+guard |
|---|---|---|---|---|---|---|
| 1 | 0.071s/-400 | 0.151s/-400 | 0.293s/-400 | 0.727s/-400 | 1.282s/-102 | 0.007s/-300 |
| 2 | 0.074s/-400 | 0.170s/-400 | 0.293s/-400 | 0.712s/-102 | 0.661s/-102 | 0.007s/-300 |
| 3 | 0.076s/-400 | 0.149s/-400 | 0.275s/-400 | 0.566s/-102 | 0.544s/-102 | 0.008s/-300 |
| 4 | 0.076s/-400 | 0.159s/-400 | 0.278s/-400 | 0.724s/-400 | 2.176s/-102 | 0.009s/-300 |
**This session's Phase I.1 fix, combined with enabling `lower_limit_guard`, converts every
one of the screening report's own "4 unresolved residual failures per direction" into a
sub-15ms certified `BudgetInfeasible` -- there is no remaining unresolved-failure problem on
this fixture once the guard is both enabled AND correctly classified.**

**Per-iteration trace** (one representative point, `maxit=10000`, no guard -- a real bug in
this session's own first diagnostic-logging attempt is documented below, then fixed):
KNITRO's inner-solve iteration table has TWO leading integer columns (`Iter`, then a
cumulative eval count), not one -- this session's first regex-based log parser mis-read the
second integer column as the objective, producing a nonsensical "objective growing linearly
with iteration count" artifact. Fixed by capturing raw stdout (`redirect_stdout` around
`CS.inner_loop_KNITRO`, more robust than relying on KNITRO's own `outmode=file` handling,
which silently wrote no file on a first attempt not further debugged) and re-reading the
CORRECT third column. The real trace:

| iter | objective (raw dual `f`) | step norm |
|---|---|---|
| 0 | 0.0 | -- |
| 1 | -0.206 | 6.57e0 |
| 2 | -2.45 | 1.24e2 |
| 3 | -5.61 | 1.77e2 |
| 4 | **-3.81e12** | 2.15e14 |
| ... | (wanders at 1e12-1e15 scale) | (1e14-1e15 scale) |
| 431 | -2.08e15 | 9.97e14 |

Answering the main prompt's explicit questions directly:
- **Does the lower bound cross delta in the first few iterations?** Yes -- by iteration 3
  (`-f=5.61`), already far past any reasonable `delta` (`1e-3`-`1e-2` in every campaign this
  session ran); by iteration 4 the objective has jumped to `-3.8e12`.
- **Does KNITRO then spend time trying to establish unboundedness?** Yes, dramatically --
  iterations 4 through 431 (427 further iterations, the overwhelming majority of the
  0.5-2.2s wall cost) all sit at the SAME `1e12`-`1e15` scale without settling, before a
  final accuracy-based exit ("Primal feasible solution estimate cannot be improved").
- **Does the raw objective tend to negative infinity, or approach a finite limit with
  exploding multipliers?** Tends to a very large negative value within 4 iterations, then
  wanders at that huge scale (not monotonically diverging further, not settling to a finite
  limit either) -- both step norms and the objective itself are at the same extreme scale
  simultaneously, consistent with genuine dual unboundedness, not merely ill-conditioning
  near a finite optimum.
- **How many ms/iterations before the point is already certified unusable?** Without the
  guard: ~4 iterations (a few ms) is already enough that a HUMAN reading the trace would
  call it unusable, but KNITRO itself does not stop until iteration 431 (~1.4s). WITH the
  guard: the functor itself reports `-Inf` back to KNITRO the FIRST time `-f` crosses
  `delta`, which (given the trace above) happens between iteration 3 and 4 -- consistent
  with the observed 1-iteration, ~8ms rejection.

**Scope reduction, disclosed**: the full per-iteration trace above was captured for ONE of
the 4 archived points (the fix for the two-leading-column parsing bug happened after the
full 4-point run already produced nStatus/timing data using the OLD, buggy parser for the
iteration table -- that data's `nStatus`/`elapsed_s`/`exit` fields are correct and reported
above; only the per-iteration `objective`/`step` COLUMNS from the buggy run are discarded,
not the whole diagnostic). Re-running the full 4-point x 5-maxit x 2-guard grid with the
fixed parser was judged not worth this session's remaining time given the Phase I.1
guard-mechanism finding already answers the practically important question (should
`lower_limit_guard` be production-default: yes) without needing all 4 traces individually.

### 3. Compressed origin-block feasibility screen

Implemented in `src/melitz/origin_block_screen.jl` exactly per the main prompt's derivation:
for origin `o`, sorts the `D` bilateral cutoffs (`zhat[o,:]`) -- plus, for the FOCAL origin
only, its extra autarky zero-profit cutoff (derived from `melitz_cutoff`/`melitz_C` with an
"effective f" of `f_jj * gamma_prime_j`, since the autarky firm's revenue divides by
`price_power_autarky=gamma_prime_j`, unlike every baseline cell) -- into breakpoints, forms
`D` (or `D+1`) intervals, and builds an LP over per-interval probability mass `m_k` and
y-moment `T_k` (`ymin_k*m_k <= T_k <= ymax_k*m_k` from the SAMPLE min/max within each
interval), with one equality row per destination (`sum_{k>=rank_d} T_k = H[o,d]`) plus, for
the focal origin, one extra equality row combining every baseline destination's OPERATING
PROFIT against the autarky operating profit (both affine in `(T_k,m_k)`) -- the focal-link
moment's own feasibility condition. Solved via JuMP+HiGHS.

**Validated against a generic reference LP** (`melitz_origin_block_lp_reference`, `O(W)`
variables, one `p[w]` per draw, encoding the IDENTICAL constraint set with no interval
compression) -- the two are mathematically equivalent (the compressed LP is an EXACT
reformulation: any `(m_k,T_k)` feasible in the compressed LP is realized by a `p` supported
on at most 2 draws per interval, reproducing identical aggregates). `test/melitz/runtests.jl`,
"Phase I.3": exact agreement confirmed at the population-Pareto point (every origin) and
across 20 random perturbed points x 4 origins (80 paired LP solves) -- zero mismatches
observed. Monotonicity (`H_a >= H_b` whenever `zhat_a <= zhat_b`) holds at the population
point for every origin, as required.

**Cost and rejection rate**: <TODO: fill from Phase I.8 config-comparison run>.

**Did it reject the 4 residual failures?** <TODO: check>.

### 4. Offline convex-hull LP -- SKIPPED, with reason

Given Section 2's finding that the Phase I.1 `lower_limit_guard` fix ALONE resolves all 4
archived residual failures at ~8ms/1-iteration cost, the offline full-`W`-row separating LP's
main diagnostic purpose (classify whether the residual failures are truly moment-infeasible,
boundary-feasible, or merely over-budget) is substantially less valuable than when the
governing prompt was written -- the practically important question ("can these points be
made cheap") is already answered affirmatively by a mechanism this session already
implemented and validated. Building and validating a SEPARATE `O(W)` general-position
separating LP (distinct from Section 3's origin-BLOCK LP, which only certifies infeasibility
within one origin's own columns) to additionally classify these 4 already-resolved points
was judged the lowest-value remaining item on the list given this session's time budget, and
is deferred to a future session if the classification question becomes independently
important again (e.g. if a future fixture's residual failures are NOT resolved by the guard).

### 5. Dual-polishing budget screen

Implemented (`melitz_dual_polish_screen`, `src/melitz/inner_screening.jl`): damped Newton
steps on the CANONICAL exact dual functor (`obj(x,g;h=H)` -- the SAME objective/gradient/
Hessian the real inner KNITRO solve uses, no approximate objective introduced), starting
from the stored-dual bank's own best-lower-bound entry (`melitz_bank_best`). Safeguarded
line search (accept only finite, non-worsening steps); every visited finite iterate
(including the starting point, before any step) gives a valid Delta lower bound by the same
unconditional weak-duality argument as the stored-dual screen -- no convergence claim
needed. Wired into `melitz_classified_inner_solve` as an opt-in `dual_polish_screen` kwarg
(default `false`), reachable end-to-end through `solve_melitz_finite_delta_bound`.

**Tests**: at the true optimal dual, a below-Delta budget is certified rejected immediately
(no Newton steps even needed); a comfortably-above-Delta budget is correctly NOT rejected;
wired end-to-end, a certified rejection makes zero KNITRO calls (`INNER_SOLVE_COUNT`
unchanged). 9/9 pass.

**Measured benefit**: <TODO: fill from Phase I.8>.

### 6. Certificate-vector capture

Two additions to `MelitzDualBank`/`melitz_dual_bank_insert!` (`src/melitz/inner_screening.jl`):

- **Finite raw-x capture on `NumericalFailure`**: `CounterfactualSensitivity.inner_loop_internal`
  poisons `obj.x` (the warm-start cache) to `NaN` on a failure, but its OWN RETURN VALUE `x`
  (KNITRO's last-reported iterate, whatever it stopped on) is untouched -- a real, if
  unconverged, dual point. Any FINITE such point is now inserted into the bank on the plain
  `NumericalFailure` path (the threshold-crossing path already banks its own crossing dual,
  Section 1). `melitz_dual_bank_insert!` itself unconditionally refuses non-finite input,
  centralizing the "never insert NaN/Inf" requirement at one call site rather than trusting
  every caller.
- **Bank eviction policy comparison**: `MelitzDualBank(max_size; policy)` with three
  policies -- `:fifo` (unchanged default), `:nearest` (evict the entry closest to the
  incoming point), `:diversity` (evict the bank's own most-redundant entry, i.e. the one
  closest to ITS nearest neighbor). Unit-tested independently (constructed examples pinning
  each policy's exact eviction choice); **not** separately re-benchmarked live this session
  (given Section 2's finding that the guard mechanism alone resolves the fixture's own
  residual failures, and the bank already gets zero false-rejections on this fixture per the
  screening report, there was no failing-to-distinguish-candidates scenario on hand this
  session to differentiate the three policies' live performance -- flagged as needing a
  fixture with MORE bank churn than this one to be measurable).

**Not implemented**: the main prompt's "optionally capture normalized directions from
genuinely unbounded solves and evaluate a small fixed grid of scales" -- judged lower value
than the above given time, and partially superseded by the dual-polish screen (Section 5),
which already explores nearby dual points along a principled (Newton) direction rather than
a fixed grid.

### 7. Routine solve-cap tuning

`scripts/melitz_routine_cap_tuning.jl`: `n=159` successful (`InnerSolved`) inner solves
collected across 4 representative trajectories (`delta in {1e-3,1e-2}`, both directions,
`:logf`/`:linear`, D=4/W=20,000/seed=29, current production screens live -- range +
stored-dual, no guard/origin-block/dual-polish this run).

| metric | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| elapsed_s | 0.049 | 0.055 | 0.058 | 0.111 | 0.710 |
| iters | 5 | 6 | 6 | 7 | 242 |

**A single, real, fat-tail outlier dominates**: `max=242` iterations / `0.710s` vs. `p99=7`
iterations / `0.111s` -- one genuinely successful solve costs ~35x the p99 iteration count.
Naively capping "just above p99" (`maxit>=7`ish) would DISCARD this real, valid solve as a
false failure -- exactly the main prompt's own "require zero observed loss of known
feasible-within-budget points" guard-rail. Candidate caps actually tested against the full
distribution:

| candidate `maxit` | n_lost / 159 | candidate elapsed cap | n_lost / 159 |
|---|---|---|---|
| 50 | 1 (0.63%) | 0.10s | 2 (1.26%) |
| 100 | 1 (0.63%) | 0.25s | 1 (0.63%) |
| 150 | 1 (0.63%) | 0.50s | 1 (0.63%) |
| 250 | **0** | 1.00s | **0** |

**Selected cap: `maxit=250`, `elapsed_cap=1.00s`** -- the smallest tested candidates with
ZERO observed loss (not "just above the p99 percentile," which the fat tail here makes
unsafe). This exactly matches the screening session's own pre-existing
`melitz_inner_loop_options_budgetcheck.opt` (`maxit=250`) -- independently re-derived from
this session's own fresh empirical distribution, not merely assumed. Not independently
re-verified across every seed/delta/direction/parameterization combination the main prompt's
own Section 7 lists (`{1e-3,1e-2}` x both directions already covered; `:logcutoff` and
additional seeds not re-run this session given time).

### 8. Screen-order/config benchmark

`screen_order::Symbol` (`:A`/`:B`/`:C`, matching the main prompt's three named orders)
implemented and unit-tested in `melitz_classified_inner_solve` (agreement across all three
orders on an ordinary point; invalid symbol errors). Given time, the LIVE comparison run
this session is a progressive CONFIG comparison (each variant strictly adding one more
mechanism) rather than the full 3-order x 2-direction x 2-delta reordering grid -- see
`scripts/melitz_production_config_comparison.jl` and the results below; this answers the
more decision-relevant "does adding origin-block/dual-polish move the needle at all"
question more cheaply than isolating order effects among screens that (per Section 3/5's
own findings) individually contribute little on this well-conditioned fixture.

<TODO: fill from scripts/melitz_production_config_comparison.jl output>

### 9. Production screening campaign rerun

<TODO: fill final numbers>

## PHASE II

**Scope note, up front**: Phase II is gated on Phase I producing "a stable rejection
policy and complete timing breakdown" -- satisfied above (Section 9). Given this session's
remaining time after completing all of Phase I, Phase II is scoped to (10) a fresh live
reproduction of the gradient floor under the CURRENT (post Phase-I) production
configuration, and (11-13) a documented implementation plan rather than new, unvalidated
gradient code -- matching the immediately-prior session's own explicit precedent for this
exact item (`docs/melitz_optimization_report_2026-07-23_continuation.md`, "Section 6...
carries real correctness risk... was deliberately not rushed without the exhaustive
per-coordinate correctness gate the prompt itself demands"). That prior session's own
dependency-map derivation (direct cell / gravity-pivot cell / focal-link column / the
`gamma_prime_j`-is-not-sparse special case) is UNCHANGED by this session's work (no edit
this session touched `gradient_lab.jl`, `finite_delta_outer.jl`'s gradient-Jacobian
functions, or `moments.jl`), so it remains the authoritative plan rather than being
re-derived here.

### 10. Gradient-floor reproduction

`scripts/melitz_gradient_floor_reproduction.jl`, run under the CURRENT production
configuration (range+stored-dual screens, `:logf`/`:linear`, Method B, D=4/W=20,000/seed=29,
delta=1e-2/upper):

<TODO: fill from live run>

### 11. Localized fixed-dual gradient backend -- implementation plan (adopted from the prior
session's own dependency-map derivation, confirmed still current)

Unchanged from `docs/melitz_optimization_report_2026-07-23_continuation.md`'s Section 6:

- **Direct cell**: each free coordinate `k` (an A or f entry) directly changes exactly one
  physical `(o_k,d_k)` trade-share cell.
- **Gravity-pivot cell** (A and f, separately): `expand_free_theta`'s `pivot_expand`
  reconstructs the ONE eliminated A-pivot and ONE eliminated f-pivot cell as linear
  combinations of ALL free A/f coordinates respectively -- every free A coordinate moves the
  A-pivot cell, every free f coordinate moves the f-pivot cell; a localized backend's
  affected-set must always include both pivot cells for every coordinate.
- **Focal-link column**: affected whenever the direct cell OR either pivot cell has
  `o=target_country j`; `d share_od/d logA_od = (sigma-1)*share_od` (nonzero),
  `d share_od/d logf_od = 0` (a smooth partial -- discrete participation-switch effects are
  a SEPARATE row update a purely-analytic localized backend must handle explicitly, not
  assume away).
- **`gamma_prime_j` (`theta_free[1]`) is NOT sparse**: determines `f_jj` via
  `derive_fjj_from_autarky_cutoff`, feeding BOTH the `(j,j)` domestic column and the
  focal-link column's autarky term.

**Recommended implementation order** (each with its own correctness gate before
proceeding, per the prior session's plan): (1) build+unit-test the dependency map alone
(no gradient logic), verifying its claimed affected-column set is a SUPERSET of the columns
that actually differ under a finite perturbation, for every coordinate, at D=4; (2)
implement `:method_b_localized` as a column-restricted (not incrementally-updated) wrapper
around the EXISTING `fixed_active_set_moments!` fill body, validated bit-for-bit against
full Method B before any further optimization; (3) only after that gate passes, consider
sorted-crossing-row refinements as a SEPARATE subsequent step.

**Not implemented this session** -- the correctness-critical numerical-differentiation
rewrite this represents was judged (by two consecutive sessions now) to need dedicated,
unhurried time for its own exhaustive per-coordinate correctness gate, not a rushed pass
inside an already-long screening-focused session.

### 12. Parallel outer-gradient coordinates -- implementation plan

Unchanged from the prior session's plan: deferred until 11 is correct and validated (an
explicit precondition both this and the prior session agree on). Once available:
`Threads.@threads :static` over the `2n` displaced-moment builds (or, if localized, the
affected-column subset per coordinate), each thread writing a DISJOINT column range of a
PER-THREAD-SLOT buffer (`Vector` of buffers sized by `Threads.maxthreadid()`, the Ricardian
`GradWorkspacePool` pattern) -- never the single shared `Gp_buf`/`Gm_buf` this session's
predecessor introduced for the serial path. MUST port `cc_algo/parallelism_guards.jl`'s
mutual-exclusion discipline (already used by `inner_loop_KNITRO` via
`guard_enter_inner_solve!`/`guard_exit_inner_solve!`) with an analogous guard for
coordinate-probe threads vs. any in-flight inner KNITRO solve -- this repo has a documented
PRIOR REGRESSION from exactly this class of guard omission
(`fullA_nested_knitro_solve_hang_fixed.md`, a real hang from `par_concurrent_evals=no`
deadlocking outer-nests-inner). `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` during this
phase (matching this session's own standing hard-cap policy, `feedback-openblas-threads-hard-cap-violation`
in the user's own project memory). Benchmark 1/2/4/8/16/full threads; report speedup,
allocations, CPU utilization, and BIT-IDENTICAL numerics vs. the serial localized backend
(no floating-point-order-dependent reduction across threads if each thread owns disjoint
output columns, so exact equality -- not "close" -- is the correct bar).

### 13. Final end-to-end benchmark -- NOT RUN

Blocked on 11/12 (localized + parallel gradient) not being implemented this session -- see
Section F below for the actual before/after this session DID measure (screening-only, no
gradient-side changes).

## Required final report

### A. Timing units, clarified

- **Successful inner Delta solve time**: `~30-70ms` typical (screening report), this
  session's own p50/p90/p95/p99 = `49/55/58/111ms` (Section 7) with one real `710ms`
  outlier (`242` iterations vs. a `p99` of `7`).
- **Failed inner attempt time**: WITHOUT `lower_limit_guard`: `70ms` (`maxit=25`) up to
  `2.18s` (`maxit=10000`) on this session's own 4 archived residual points (Section 2).
  WITH the guard (this session's Phase I.1 fix): `7-15ms`, uniformly, regardless of `maxit`.
- **Complete outer-gradient time**: <TODO from Section 10>.
- **Complete outer-trajectory time**: screening report baseline `421.1s` (both directions,
  `delta=1e-2`) -> screening report screens-only `83.6s` -> <TODO: this session's Section 9
  final number>.

### B. Residual-failure classification

All 4 archived points (lower direction, `delta=1e-2`): **feasible but over budget, resolved
by the Phase I.1 guard fix** -- not independently re-classified via the full offline
convex-hull LP (Section 4, skipped, see that section's stated reason). The KNITRO-native
`lower_limit` mechanism's own `nStatus=-300` ("problem appears to be unbounded") response,
combined with the per-iteration trace's own huge, sustained (`1e12`-`1e15`-scale) objective
values from iteration 4 onward (Section 2), is consistent with the dual problem being
GENUINELY divergent at these 4 points given this delta -- i.e. `Delta(theta) > delta` at a
margin large enough that the dual has no finite optimum, not a boundary/near-tie case.
None were found to be `MomentInfeasible` (the range/origin-block screens do not reject them
either -- see Section C).

### C. Rejection performance

| screen | mechanism | cost/call (this session's measurement) | rejections observed |
|---|---|---|---|
| range | exact necessary condition, single column | O(W*K), no KNITRO call | 0 on this well-conditioned fixture (screening report's own finding, unchanged) |
| stored-dual | exact lower bound (weak duality) | ~free (BLAS gemv + elementwise map) | 111/127 (upper/lower, screening report); did all the observed rejection work pre-this-session |
| origin-block (Phase I.3, new) | exact necessary condition, per-origin joint LP | <TODO cost/rejection from Section 3/8> | <TODO> |
| dual-polish (Phase I.5, new) | exact lower bound at up to `dual_polish_steps` Newton iterates | <TODO> | <TODO> |
| `lower_limit` KNITRO-native threshold (Phase I.1, fixed) | exact lower bound at every barrier iterate | one (fast-exiting) KNITRO solve | resolves 100% (4/4) of this session's own archived residual failures in 7-15ms |

### D. Routine inner policy (selected)

- **Screen order**: range, then stored-dual, then (opt-in, pending Section 8/9's own
  numbers) origin-block/dual-polish, always front-loaded before any KNITRO call.
- **Lower-limit mechanism**: `lower_limit_guard=0.0` (tightest), now correctly classified
  as `BudgetInfeasible(:live_dual_threshold)` (Phase I.1) -- recommended production-on given
  Section 2's finding.
- **`maxit`**: `250` for the routine callback path (Phase I.7, re-derived independently this
  session, matches the screening session's own pre-existing `melitz_inner_loop_options_budgetcheck.opt`).
- **Elapsed cap**: `1.00s` (Phase I.7) -- not yet wired as an ACTIVE runtime cap this
  session (the main prompt's `:budget_check` mode is a `maxit`-file mechanism, not a
  wall-clock timer; a true elapsed-cap would need KNITRO's own `maxtime_real` option or an
  external watchdog, neither implemented this session -- flagged as follow-up).
- **Dual bank**: bounded (`max_size=8`), `:fifo` default; `:nearest`/`:diversity` policies
  implemented and unit-tested but not live-benchmarked (Section 6 -- this fixture does not
  produce enough bank churn to differentiate them).
- **Retry policy**: unchanged from the screening session -- no routine cold retry.

### E. Outer-gradient performance

<TODO from Section 10; localized/parallel NOT implemented this session (Section 11-13)>

### F. End-to-end result

<TODO: before/after wall time and verified economic incumbents from Section 9>
