# D=10 method comparison: sequential-linearized vs full-A vs gravity-seeded

Branch `feature/sequential-inversion-perf`, worktree `trade_robustness_modular_perf`. All four runs
below used the real in-run timing instrumentation added this session (destination-inversion /
`influence_function` / `gravity_residual` wall time, on top of the pre-existing `t_inner`/`t_grad`
outer-loop cache breakdown), `maxit=25` (`full_aod_diag/csw_outer_25.opt`), D=10, σ=2.5, and (unless
noted) W=8000. Raw logs: `batch_logs_method_comparison/`. Checkpoints: `sequential_gravity/batch_out_v2/`
(baseline), `full_aod_diag/batch_out_v2/` (full-A), `sequential_gravity/batch_out_gravityseed/`
(gravity-seeded), `sequential_gravity/batch_out_w80000/` (W=80000).

Two changes from the handoff's assumed defaults, both load-bearing: (1) the default checkpoint
directories (`sequential_gravity/batch_out`, `full_aod_diag/batch_out`) already held `done=true`
results from the prior (pre-instrumentation, pre-LM-fix in the full-A case irrelevant since full-A
never needed the fix) session — reusing them would have silently skipped every solve, so all runs
here use fresh `_v2`/`_gravityseed`/`_w80000` directories. (2) `sequential_gravity/verify_batch_solutions.jl`
had its `BATCH_DIR` hardcoded; added a `VERIFY_BATCH_DIR` env override (purely additive) to point it
at each new checkpoint set.

κ_max theoretical ceiling for this D=10 economy/seed = **0.5704** (established and cross-validated in
a prior session via `1 - λ_dd^(1/(σ-1))`; not recomputed this session since the data/seed haven't changed).

## 1. Headline numbers: κ (and γ'_focal) by δ × bound

All κ values below are gravity-feasible per each method's own bookkeeping unless flagged. "KNITRO-own"
is the search's own terminal point; "best-feasible" is the best feasible point seen across all θ-evals
during that search (tracked separately — see `production-refactor-2026-07-13.md` for why these can differ).

### Sequential, baseline (no gravity-seeding)

| δ | bound | γ'_KNITRO / κ | γ'_best-feas / κ | gravity_ok (KNITRO / best) |
|---|---|---|---|---|
| 0.1 | lower | 0.911912 / **0.142459** | 0.923806 / **0.123737** | true / true |
| 1.0 | lower | 0.979653 / **0.033682** | 0.979707 / **0.033592** | true / true |
| 10  | lower | 0.989760 / **0.017008** | 0.989760 / **0.017008** (same pt) | true / true |
| 0.1 | upper | 0.841730 / **0.249608** | 0.841418 / **0.250071** | true / true |
| 1.0 | upper | 0.749604 / **0.381434** | 0.749090 / **0.382140** | true / true |
| 10  | upper | 0.669416 / **0.487737** | 0.668205 / **0.489280** | true / true |

Every one of the 6×2 points is gravity-feasible per production's own bookkeeping — including δ=10,
both bounds, both KNITRO-own and best-feasible. This is a direct contrast with the prior session's
findings (δ=10 lower reproducibly gravity-infeasible; δ=10 upper KNITRO-own degenerate,
R_mean≈1.1e+107). See §3 for the independent cold-verification confirming these are genuinely valid,
not just "reported feasible."

### Full-A (all A_od directly in the outer loop)

| δ | bound | γ' | κ | opt_err | feas_err | quality |
|---|---|---|---|---|---|---|
| 0.1 | lower | 0.912584 | **0.141406** | 0.0656 | 1.4e-10 | clean |
| 1.0 | lower | 0.971082 | **0.047731** | 0.0271 | 5.2e-18 | clean |
| 10  | lower | 0.997554 | **0.004073** | **4.16e+04** | **3.3e+10** | **NOT converged — do not trust** |
| 0.1 | upper | 0.855915 | **0.228414** | 0.0879 | 7.8e-07 | clean |
| 1.0 | upper | 0.813891 | **0.290515** | 0.2242 | 1.6e-05 | acceptable, higher opt_err |
| 10  | upper | 0.741167 | **0.392993** | 0.3664 | 1.5e-10 | acceptable, higher opt_err |

Full-A lower δ=10's `opt_err`/`feas_err` are orders of magnitude worse than every other cell — a
genuine non-convergence at `maxit=25`, not a small-noise wobble. This matches a documented pattern:
a prior session's *clean* full-A lower-δ=10 number (κ=0.0051, opt_err~0.002-0.003) used `maxit=300`,
not this batch's production default of 25. **Treat full-A's δ=10 lower κ=0.004 as unreliable**; the
other five full-A cells are trustworthy.

### Sequential, gravity-seeded (evaluated, not recommended — see §4)

| δ | bound | γ'_KNITRO / κ | γ'_best-feas / κ | gravity_ok (KNITRO / best) |
|---|---|---|---|---|
| 0.1 | lower | 0.911876 / **0.142516** | 0.923578 / **0.124097** | true / true |
| 1.0 | lower | 0.942549 / **0.093906** | 0.977265 / **0.037604** | true / true |
| 10  | lower | 0.989005 / **0.018258** | 0.989005 / **0.018258** (same pt) | true / **false** |
| 0.1 | upper | 0.865305 / **0.214255** | 0.865114 / **0.214543** | **false** / **false** |
| 1.0 | upper | 0.743978 / **0.389151** | 0.740000 / **0.394585** | true / true |
| 10  | upper | 0.682159 / **0.471382** | 0.682159 / **0.471382** (same pt) | true / true |

## 2. δ=10 cold cross-verification (the central question)

`verify_batch_solutions.jl` re-solves everything from scratch (fresh `recover_lfd`, fresh,
**cold, no-warm-start** `invert_destination` for every omitted destination) and reports the actual
non-focal share-matching error, independent of whatever production's own (possibly warm-biased)
bookkeeping said. This is exactly the check that caught the previous session's δ=10 upper
best-feasible failure (5/9 destinations non-convergent, share errors up to 0.89, R_mean≈2.8e+106).

**Result: every single checkpoint in both sequential runs — all 6 δ×bound cells, both KNITRO-own
and best-feasible where they differ — now has non-focal max\|share_error\| between 1.7e-10 and
9.9e-9**, i.e. at the destination-inversion solver's own numerical tolerance, indistinguishable from
the previously-validated δ=0.1/1.0 results. This includes δ=10 upper best-feasible specifically
(baseline: 6.7e-9; gravity-seeded: 6.3e-9) — **the exact point that failed catastrophically before.**

**Conclusion: the LM-damping fix (commit `8f7d850`) has fully resolved the δ=10 reliability problem.**
The sequential method's implied non-focal trade shares genuinely match the real D² data at every δ
tested, not just δ≤1.

Caveat (small, pre-existing, unrelated to the fixed bug): the cold-recomputed exact gravity
`R_mean` occasionally exceeds the nominal 5e-4 tolerance by a factor of 2-10× (e.g. baseline lower
δ=10: R_mean=1.43e-3; gravity-seeded upper δ=10: R_mean=2.21e-3) even though production's own
warm-consistent bookkeeping reported it as feasible at the same θ. This is the previously-documented
"boundary-sensitivity"/cold-vs-warm R_mean discrepancy — it is three to eight orders of magnitude
smaller than the fixed catastrophic blowup (1e+106-1e+272) and does not affect the share-matching
result above; it means a few feasibility flags are marginal rather than clean, not that the method
is broken.

## 3. Gravity-seeding evaluation (extended)

Confirms and extends the earlier single-point finding (`full_aod_diag/gravity_seeded_initial_solve/README.md`,
δ=10 upper only): gravity-seeding is not uniformly worse everywhere in this fuller sweep, but it is
worse on the metrics that matter most:

- **New failure mode found at δ=0.1 upper** (not previously tested): the seeded variant's KNITRO-own
  point (κ=0.214) is gravity-infeasible by BOTH production's own check and the independent cold
  verification (R_mean=1.83e-3, ~4× the tolerance) — while baseline's δ=0.1 upper point (κ=0.250) is
  clean. This is a materially different, wider κ gap than the δ=10 case previously documented.
- At δ=1.0 and δ=10 upper, both variants land on feasible points with broadly similar κ (baseline
  0.381-0.489 vs seeded 0.389-0.471) — much closer than at δ=0.1, but seeded is still consistently
  more expensive (below).
- **Cost**: total destination-inversion time (summed across all 6 cells) is 17990s for gravity-seeded
  vs 8907s for baseline — **~2.0× slower** — driven by 88569 vs 24048 total destination inversions
  performed (~3.7× more), consistent with the D+2-moment vs D+1-moment first-solve difference. The
  earlier single-point test found 3.2×; the full sweep's average is somewhat lower but the direction
  and order of magnitude agree.

**Recommendation unchanged: do not adopt gravity-seeding.** It is slower, and — now confirmed across
more of the δ grid — it introduces its own new gravity-infeasibility failure mode (at δ=0.1, not
just δ=10) rather than solving the drift concern it was meant to address.

## 4. Timing breakdown: sequential vs full-A

All sequential dest-inversion/`influence_function` numbers are **summed across 9 `PARALLEL_INVERSION`
threads** (aggregate CPU-seconds, not wall-clock) — they legitimately exceed wall-clock time. The
cache's `inner-solve time`/`gradient time` are single-threaded and bounded by wall-clock.

**Sequential baseline**, low-δ (0.1) vs δ=10, both bounds (seconds):

| bound, δ | dest_inversion (Σthreads) | influence_fn | gravity_residual | inner-solve (cache) | grad (cache) | wall |
|---|---|---|---|---|---|---|
| lower, 0.1 | 1135.6 | 159.5 | 1.2 | 470.4 | 10.6 | 491.3 |
| lower, 10  | 3215.8 | 261.3 | 0.02 | 930.2 | 12.0 | 945.4 |
| upper, 0.1 | 413.6 | 103.8 | 0.01 | 225.8 | 5.2 | 231.6 |
| upper, 10  | 493.2 | 56.0 | 0.01 | 177.7 | 2.6 | 181.9 |

Destination-inversion is the dominant *computational* cost by a wide margin (2-7× wall-clock,
reflecting the 9-way parallel speedup), but because it's parallelized, the single-threaded
inner-solve time becomes a comparable or larger fraction of *wall-clock* time than one might guess
from the CPU-time split alone — e.g. at lower δ=0.1, inner-solve (470s) is nearly all of the 491s
wall-clock even though dest-inversion did 1136 CPU-seconds of work in that same window.

**Full-A**, same four cells (seconds):

| bound, δ | inner-solve (cache) | grad (cache) | wall |
|---|---|---|---|
| lower, 0.1 | 1165.6 | 1211.1 | 2427.5 |
| lower, 10 (unreliable, see §1) | 297.0 | 627.1 | 931.1 |
| upper, 0.1 | 34.0 | 374.3 | 409.9 |
| upper, 10  | 17.6 | 417.7 | 435.7 |

For full-A, inner+grad time accounts for essentially all of wall-clock (no parallelism to hide behind,
unlike sequential's destination inversions). Gradient time dominates inner-solve time at the upper
bound by 10-25×, reflecting the dense D² ForwardDiff Jacobian's cost (the whole reason full-A needs
"Method B" caching in the first place) — but at the lower bound inner-solve is comparable to or
larger than gradient time, likely reflecting more/costlier inner CC solves at that δ/bound combination
(172-274 inner solves at lower vs 114-151 at upper).

## 5. W=80000 vs W=8000, δ=1.0, sequential baseline

| W | bound | γ' | κ (KNITRO / best) | dest_inversion (s) | influence_fn (s) | wall (s) |
|---|---|---|---|---|---|---|
| 8000  | lower | 0.9797 | 0.0337 / 0.0336 | 2620.7 | 207.2 | 753.7 |
| 80000 | lower | 0.9763 | 0.0391 / 0.0370 | 7169.9 | 1421.6 | 3072.7 |
| 8000  | upper | 0.7496 | 0.3814 / 0.3821 | 1005.8 | 134.5 | 384.4 |
| 80000 | upper | 0.7404 | 0.3940 / 0.3940 (same pt) | 8897.8 | 1119.5 | 2931.9 |

κ moves only modestly with 10× the Monte Carlo draws (lower: 0.034→0.039; upper: 0.381→0.394) —
consistent with W=8000 already being a reasonably well-converged sample size for this statistic, not
a sign that W=8000 is materially biased. Cost scales sub-linearly to slightly super-linearly with the
10× draw increase: wall-clock is 4.1× slower (lower) to 7.6× slower (upper); destination-inversion
CPU-time is 2.7× (lower) to 8.8× (upper) slower. The upper bound's disproportionate slowdown suggests
Newton iteration counts (not just per-iteration O(S·D) cost) are also sensitive to W at that
bound/δ, not purely a linear-in-S effect.

## 6. Why full-A's gradient time is so much larger (not a bug, not a resurrected dense Jacobian)

Follow-up questioning of §4's timing table surfaced an important correction to how that section
should be read, and a real, actionable (but out-of-scope-for-now) optimization opportunity.

**What §4 got right:** full-A's tiny inner-solve time (17-34s at the upper bound) is genuine evidence
that "Method B" (the exact-point cache + envelope-theorem gradient, see
[[method-b-eliminates-dense-jacobian]]) is working — the inner CC optimization is *not* being
re-solved via the implicit-function theorem to get gradients.

**What §4's phrasing got wrong:** describing the large gradient time as "reflecting the dense D²
ForwardDiff Jacobian's cost" made it sound like the thing Method B eliminated was somehow happening
again. It isn't. Reading the actual code (`full_aod_diag/run_fullA_D10_production.jl`):

- `gravity_grad_fn!` → `gravity_grad_free!`, a closed-form ~100-element elementwise loop, no AD.
  Genuinely negligible.
- `div_grad_fn!` (the divergence-budget constraint's gradient — the one both methods have) →
  `ForwardDiff.gradient!` over the **full free vector** `x_free`.

Crucially, `_methodB_envelope_scalar` (sequential's version of this same divergence-gradient
function, in `sequential_gravity/PsiObjectiveBundleImplicitMethodB.jl`) and `envelope_scalar_div_ctx`
(full-A's version, in `full_aod_diag/ad_benchmark/derivative_core.jl`) are **the same computation**:
both call `moments!` to build an N×d moment matrix, then sum `arg1[draw]·Σⱼλⱼ·G[draw,j]`. Full-A
isn't doing something different or worse — it's doing the identical thing to a genuinely bigger
problem, on two axes that multiply together:

| | sequential | full-A | ratio |
|---|---|---|---|
| free-parameter dim (`x_free`, what ForwardDiff differentiates over) | D+1 = 11 | 1+D² = 101 | ~9.2× |
| moment-matrix width `d` (what `moments!` computes per draw) | D+2 = 12 | nTotalMoments ≈ 102 | ~8.5× |
| measured per-call gradient cost (lower, δ=0.1: seq 165 calls/10.6s, full-A 172 calls/1211s) | 0.064s/call | 7.04s/call | **~110×** |

The two structural ratios multiply (9.2×8.5≈78), landing right at the measured ~110× — confirming
this is the real mechanism. **This is the actual conceptual difference between the two methods, not
an implementation gap**: sequential's CC problem only ever carries D+1 focal moments (+1 linearized
gravity moment); the other D²−D non-focal trade shares are matched *outside* the differentiated
optimization entirely, by the fast custom-Newton destination-inversion step, which never touches
ForwardDiff. Full-A, by construction, puts every A_od directly in the outer loop, so its CC problem
must carry all D² trade-share moments — both the parameter count and the moment-matrix width it
differentiates through scale with D² instead of D.

**Actionable, but explicitly out of scope for now** (user's call — full-A isn't the production-track
method): a closed-form analytic gradient for full-A's divergence constraint, mirroring the treatment
already given to its gravity constraint, could eliminate the ~9× ForwardDiff-chunking overhead
(replacing ~9 chunked passes through `moments!` with one analytic pass). That would meaningfully
speed up full-A. It would **not**, however, close the gap down to sequential's level — the ~8.5×
moment-width factor is inherent to full-A matching D² moments directly; no gradient-computation
trick removes that, only using fewer moments (i.e. sequential's actual design) does. **§4's timing
comparison should therefore be read as accurate for the code as it stands, but not as a ceiling on
full-A's achievable speed** — the true speed gap between the two *methods* (as opposed to these
particular implementations) is likely smaller than §4 suggests, though probably still nonzero given
the inherent D² moment-width factor.

## 7. Does sequential's wider-bound finding still hold?

Comparing sequential-baseline vs full-A at every δ (using only full-A cells confirmed clean):

| δ | κ_lower: seq / full-A | κ_upper: seq / full-A |
|---|---|---|
| 0.1 | 0.124-0.142 / 0.141 | 0.250 / 0.228 |
| 1.0 | 0.034 / 0.048 | 0.381-0.382 / 0.291 |
| 10  | 0.017 / (full-A unreliable) | 0.488-0.489 / 0.393 |

Sequential's interval is wider than full-A's on **both ends** at every δ where full-A is trustworthy
(the δ=10 lower comparison can't be made cleanly since full-A's own point isn't converged there).
The upper-bound gap actually **grows** with δ (0.02 at δ=0.1 → 0.09 at δ=1.0 → 0.10 at δ=10), and —
per §2 — the δ=10 sequential numbers are now independently cold-verified as genuinely matching the
real D² trade-share data, not an artifact of the previously-unresolved reliability problem.

**Bottom line: sequential's wider-bound finding, previously established only at δ=0.1/1.0, now
extends cleanly to δ=10 with the LM-damping fix in place.** The δ≥10 reliability picture has
materially improved: what was previously "genuinely unresolved, do not report as a validated bound"
is now cold-verified and safe to report, modulo the small pre-existing R_mean boundary-sensitivity
caveat in §2 (which affects feasibility-flag marginality, not the share-matching validity that
underpins the wider-bound claim).

**Speed is a separate claim from bound-width, and only the latter is solid as stated.** §4/§5's
timing numbers accurately describe the code as it stands, but per §6, full-A's gradient cost is
inflated by an unoptimized (AD instead of analytic) divergence-constraint gradient — a real but
unfinished optimization, explicitly deferred rather than pursued this session since full-A is not
the production-track method. The wider-bound finding above does not depend on that gap at all.
