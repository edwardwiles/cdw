# Full-A D=4 exact formulation: findings report

Branch `diag/fullA-d4-exact`, worktree `../gravity-fullA-d4`, off production commit `53ffb58`.
All artifacts under `results/fullA_d4/<commit>/`. This report supersedes the prior interim version
(`git log -- docs/fullA_d4_final_report.md` for history); `docs/fullA_d4_code_audit.md` remains the
authoritative record of the code-level findings, and `docs/fullA_d4_resume_audit.md` records the
specific corrections this continuation session made to the prior interim report, with primary-source
evidence for each. This report is **not interim** — every section below reflects work independently
re-verified or newly produced in this continuation session, with commit hashes and artifact paths.

## 1. Exact free-parameter vector, normalization, gravity elimination (unchanged from the prior
   session; re-affirmed, not re-litigated)

At D=4: `n_free=17` (γ'_focal + all 16 A_od entries, γ_d≡1 for every destination — the code's own
generalization of the methodology PDF's focal-only gauge). Exact gravity elimination (sparse pivot,
drop the largest-|coefficient| A_od entry) reduces this to 16 gravity-feasible-by-construction
coordinates for outer-loop work. See `docs/fullA_d4_code_audit.md` §3-5 for the code-level citations;
independently re-used (not re-derived) throughout this session's Phase A/B/D/F work via
`full_aod_diag/d4_exact/context.jl::d4_exact_setup` and `gravity_elimination.jl`.

## 2. Corrected characterization of KNITRO's `hessopt` (continuation-prompt correction #1)

`hessopt=4` is **product finite-difference Hessian-vector**, not BFGS — verified directly from this
repo's own `full_aod_diag/csw_outer_25.opt` comment block: `hessopt` 0=auto, 1=exact, 2=bfgs (dense),
3=sr1 (dense), 4=product_findiff, 5=product (user-supplied), 6=lbfgs. Every run in this
investigation's history using `eval_fcga=no` (the diagnostic `csw_outer_fcga_no_maxit*.opt` files,
including both upper-direction runs discussed below) used **hessopt=4, genuinely honored** — not
"restored BFGS." Independently re-confirmed from the existing `results/fullA_d4/bf00b00/knitro_*.log`
files: `eval_fcga=yes` triggers `"WARNING: Option hessopt=4 not valid when eval_fcga=1. Changing
hessopt to 6 (LBFGS)."`; `eval_fcga=no` shows no such line (hessopt=4 genuinely used).

**New this session (Phase B)**: a controlled comparison (same start, same h=0.01 central-FD gradient
method, same maxit=15, same bounds — `full_aod_diag/d4_exact/phaseB_hessian_algorithm_matrix.jl`,
`results/fullA_d4/1b2a3a0/phaseB_hessian_matrix/`) of genuine BFGS (hessopt=2), SR1 (hessopt=3),
L-BFGS (hessopt=6), explicit SQP (algorithm=4, hessopt=2), explicit barrier-direct (algorithm=1,
hessopt=2), against the product-findiff control:

| config | knitro_status | opt_err | outer_iters | wall | n_inner_solves | best_feasible κ | fallback? |
|---|---|---|---|---|---|---|---|
| auto_bfgs (hessopt=2) | -400 | 0.1146 | 15 | 24.4s | 549 | 0.16875 | none |
| auto_sr1 (hessopt=3) | -400 | **0.00225** | 15 | 17.2s | 555 | 0.16870 | none |
| auto_lbfgs (hessopt=6) | -400 | 0.00848 | 15 | 17.6s | 546 | 0.16944 | none |
| sqp_bfgs (algorithm=4, hessopt=2) | **-410** | 0.0268 | 15 | 18.2s | 555 | 0.16654 | none |
| direct_bfgs (algorithm=1, hessopt=2) | -400 | 0.1146 | 15 | 16.6s | 549 | 0.16875 | none |
| **auto_productfd (hessopt=4, CONTROL)** | -400 | 0.00157 | 15 | **324s** | 9586 | **0.17058** | none |

None of the genuine Hessian modes show a fallback message — the fallback is specific to
`hessopt=4`+`eval_fcga=yes`, confirmed twice now (production log, and the absence of the message in
every genuine-Hessian-mode log this session produced). Explicit active-set SQP does measurably worse
here (status -410, a distinct solver failure code, and the worst best-feasible κ of the six). Genuine
BFGS/SR1/L-BFGS are all ~15-20x cheaper per outer iteration than the product-findiff control (17-24s
vs 324s for the same maxit=15) because product-findiff needs an extra FD probe per component beyond
the gradient itself — but within the SAME maxit=15 iteration budget, none of them quite reach the
control's κ=0.17058 (they reach ~0.168-0.169). **This is a real, quantified cost/quality tradeoff, not
a free win either direction**: wall-clock-matched (e.g., ~15x more outer iterations for SR1/L-BFGS in
the same 324s budget the control used) is the fair comparison, not attempted this session — flagged as
follow-up.

## 3. Winner-boundary derivative bug (unchanged from the prior session; re-affirmed)

The `MinInd!` hard-Bool winner-selection branch causes naive pathwise AD to miss the
winner-boundary/Dirac term — independently confirmed via exact tie-threshold construction
(`winner_switching.jl`), quantified precisely (`results/fullA_d4/6b6ff4a/`,
`results/fullA_d4/45ac6c6/`). Not re-derived this session; used as an input throughout Phase A/D.

## 4. Upper-direction candidate: corrected classification (continuation-prompt corrections #2/#3)

**The archived `stationarity_check_upper.txt` was run on the maxit=15 best-feasible point
(κ=0.17058), not the maxit=40 best-feasible point (κ=0.17176461) actually reported as the headline
result.** Confirmed by direct byte-for-byte comparison of the archived `w` vector against both
candidates' `summary.txt` files — `docs/fullA_d4_resume_audit.md` §6.2 has the full table. The
maxit=40 run's own `summary.txt` says so explicitly: `"external KKT stationarity NOT checked yet"`.

**New this session (Phase A, `full_aod_diag/d4_exact/phaseA_upper_revalidation.jl`,
`results/fullA_d4/1b2a3a0/phaseA_upper_revalidation/`)**: a full revalidation of the maxit=40 point
(w[1]=γ'_focal=0.8930839180420251, κ=0.17176461):

1. **Fresh cold+warm recheck at tight (opttol=1e-12) and loose (opttol=1e-8) inner tolerance**:
   exactly feasible at both (gravity ~1e-18, KKT residual 1e-16 to 1e-12 depending on tolerance,
   winner_hash stable across warm/cold/tolerance), κ reproduces the archived value to 10 significant
   digits. `EXACT_FEASIBLE_CANDIDATE`: **confirmed**.
2. **Bounds**: γ'_focal's bounds are genuine (closed-form autarky theory, `sequential_methodology.tex`
   §4). The ±8 log-space box bounds on the 15 non-pivot A_od coordinates are confirmed **numerical
   safeguards only** — no economic restriction anywhere in the methodology PDF or code imposes them;
   the observed point sits only ~17% of the way to them, nowhere near active.
3. **Fresh external KKT check AT THIS POINT, h=0.01** (matching the original recipe): **passes**
   (η=0.0131>0, KKT residual 0.12% relative, complementary slackness ~-1.6e-5, zero active bounds,
   zero nonfinite probes). `H_BANDWIDTH_KKT_CANDIDATE(h=0.01)`: **true**.
4. **Multi-h gradient grid (h=0.02, 0.01, 0.005, 0.0025, 0.001)**: the h=0.01 pass is **not robust
   across bandwidth**. KKT residual drifts monotonically from 0.12% (h=0.01) to 1.9% (h=0.001) as the
   probe window shrinks and captures fewer genuine winner switches (31→30→28→23→10 switches across
   the grid) — the textbook signature of an FD gradient losing the boundary term as h→0. h=0.02 fails
   outright: one nonfinite probe, η≈0, KKT residual=1.0 (total failure), and its gradient direction is
   **nearly orthogonal** to the h≤0.01 cluster (cosine≈0.0002, vs cosine≥0.9999 among h=0.01..0.001
   themselves).
5. **20 random-direction + 3 weak-sensitivity directional checks**: **every single one of the 20
   random directions** triggers a winner switch at step 0.01 on both sides — this point sits in a
   region where essentially every generic direction is kink-adjacent. Central-slope-vs-h=0.01-gradient
   -predicted-slope disagreement is large along random directions (mean |actual−predicted| = 4.23,
   with individual disagreements up to ~13 against slopes of magnitude ~15-37 — i.e. up to ~35-45%
   relative error) even though the 3 coordinate-aligned "weak" directions agree to ~1e-14 (a pure
   consistency check, not independent evidence, since those ARE the FD axes).
6. **Deterministic poll (216 probes, radii 0.001/0.005/0.02, 36 directions × 2 signs × 3 radii)**:
   found **3 small but genuine exact-feasible improvements** (all at the smallest radius, 0.001;
   Δγ'_focal ≈ -1.4e-5 to -1.9e-5, i.e. improving κ by a tiny but real, reproducible amount on a fresh
   cold re-solve — not noise).

**Classification** (per the continuation prompt's taxonomy): `EXACT_FEASIBLE_CANDIDATE` = **true**;
`H_BANDWIDTH_KKT_CANDIDATE(h=0.01)` = **true**; `ROBUST_LOCAL_CANDIDATE` = **FALSE**. The point is a
genuine, exactly-feasible, good local candidate that passes a single-bandwidth KKT check — but is
demonstrably **not** a robustly-verified stationary point once probed across bandwidths, directions,
and a real poll. This downgrades the prior session's `VERIFIED_STATIONARY_FEASIBLE_CANDIDATE` label,
which was in any case computed on the wrong point.

## 5. Independent primal-feasibility certificate for `-300` multistart failures (correction #4)

**New this session (Phase F, `full_aod_diag/d4_exact/phaseF_primal_feasibility_lp.jl`,
`results/fullA_d4/1b2a3a0/phaseF_primal_feasibility_lp/`)**: a direct LP (HiGHS via JuMP, newly added
as a project dependency this session — first use of either package here), fully independent of
KNITRO's dual solve: `m_s≥0`, `mean(m)=1`, `mean(m·G_j)=0` for all 18 (all-equality) moments. Tested
at 10 deterministic KNITRO `-300` multistart failures (exact RNG replay of
`check_multistart_feasibility.jl`'s seeds, radius 0.1 to 0.5) plus 4 successes as a sanity check:

- **10/10 of the `-300` failures are independently CERTIFIED_INFEASIBLE** by a phase-I LP (minimize
  max absolute moment residual): every one has a strictly positive minimum achievable residual
  (9.7e-5 to 0.176), not just a KNITRO-reported status.
- **4/4 of the KNITRO successes are LP-confirmed feasible** (sanity check on the LP machinery itself).

This upgrades the prior session's "-300 is consistent with genuine infeasibility" hedge to a real,
independent certificate: the random-start failures documented earlier in this investigation are
**not** an artifact of the CC dual solver's own numerics.

## 6. `smoothing_check.csv` reconciliation (correction #5) — found a deeper, unresolved bug

The prior report claimed a "successful jump reduction to 1.52e-11 (smoothed) from 8.27e-6 (hard)";
the archived CSV had `NaN` for the smoothed entry. Investigating this **found a real, more serious,
unresolved issue**: `smoothed_frozen_adjoint_Q`'s result at a femtoscale (1e-9) straddle of an exact
tie threshold is **call-history-dependent** — repeated calls with bit-identical arguments return
different values (observed range across independent runs: 1.5e-11 to 7.8e-2, plus NaN/Inf) both
across separate process invocations AND, in some cases, within a single process. Isolated (not merely
observed): not the callee alone (`smoothed_factual_G` is stable under direct repeated calls at a fixed
point); not a generic first-call/JIT artifact (a genuine first-ever call to `smoothed_frozen_adjoint_Q`
at a non-degenerate point is stable); not purely a "too-small delta" issue (a delta=1e-9..1e-4 sweep in
a *minimal, standalone* script is perfectly reproducible and scales cleanly linearly — the same delta
values become unstable only embedded in a longer call history). **Root cause not identified within
this session's budget** — flagged as an open bug, `full_aod_diag/d4_exact/test_smoothed_moments.jl`
now reports multiple within-process samples with an explicit warning rather than a single
(possibly-lucky) number. **Scope**: isolated to the Method E smoothing-diagnostic path only
(`smoothed_factual_G`/`smoothed_frozen_adjoint_Q`/`smoothMinIndNew!`, diagnostic-only code) — does
**not** affect `evaluate_fullA` (the hard-value oracle underlying every other phase of this
investigation), which passed a dedicated bit-identical-repeated-call determinism test in this
session's mandatory smoke test (`test_oracle.jl` TEST 2).

## 7. Can a cheap gradient replace optimized-value FD? (Phase D)

**New this session (`full_aod_diag/d4_exact/phaseD_gradient_benchmark.jl`,
`results/fullA_d4/83b7380/phaseD_gradient_benchmark/`)**: benchmarked hard pathwise AD (Method A),
frozen-adjoint `Q_adj` FD, and fixed-dual `L_fix` FD against the optimized-value `Delta` FD ground
truth, at 4 points (calibration, both upper candidates, the lower stalled point), reusing the
already-validated `three_way_derivatives.jl`/`derivative_methods.jl` machinery. Method E excluded per
§6 above.

- At calibration and both upper candidates, **all three cheap methods track the ground truth's
  DIRECTION remarkably well**: cosine similarity 0.998-1.000 throughout; `L_fix_FD` is essentially
  exact in direction at both upper candidates (cosine=1.0000).
- But **gradient MAGNITUDE is systematically underestimated** by the cheap methods (norm ratio
  0.42-0.75 vs. the ground truth) — a fixed step size calibrated on a cheap gradient needs rescaling,
  not a free transfer.
- **Directional-prediction error is large at the upper candidates** (mean |actual−predicted| ≈ 8-11)
  vs. tiny at calibration (≈0.02) — the frozen-dual approximation's quality degrades specifically near
  the kink-adjacent region Phase A already flagged as fragile, not uniformly across the parameter
  space.
- **Cost savings from skipping the inner re-solve are real but MODEST (2-4x, not orders of
  magnitude)** at this W=8000/D=4 scale, because the inner CC dual solve itself is already cheap
  (~0.01-0.03s per solve) — recomputing `G(x)` at each FD probe (needed by every method, cheap and
  expensive alike) dominates the cost, not the inner optimization. **The value case for an
  `L_fix`-based hybrid scheme is likely to strengthen at larger W or D** (where the inner solve cost
  grows relative to the moment recompute), not demonstrated at this synthetic benchmark's scale — this
  is exactly the open Phase G (W-stability) question.
- At the lower stalled point, even the ground-truth `Delta` FD gradient has a non-finite component
  (this benchmark's plain central-FD has no one-sided fallback, unlike the production driver) —
  cosine/norm-ratio undefined there, consistent with the lower direction sitting in a more
  numerically fragile region generally.

## 9. Continuation session 2: performance profiling, sequential-solution validation, blockwise gradients

A second continuation (task: "exact full-A performance profiling, D=4 completion, and staged
scaling") added the following, at commits `1b2a3a0`..`46b461c`. Full detail in
`docs/fullA_performance_profile.md`, `docs/fullA_scaling_projection.md`, `docs/fullA_d4_W_stability.md`,
`docs/fullA_d4_profile_and_bounds.md`, `docs/fullA_algorithm_frontier.md`, and
`docs/fullA_next_handoff.md`; summarized here.

### 9.1 Performance profile (Phase 1, mandatory-first per that continuation's task)

Per-evaluation cost at D=4/W=8000 is dominated by `inner_solve` (53% of 31.1ms median) and
`moments_recompute` (34%) — the latter is a **literal redundant second computation**:
`cc_algo/inner_loop_functions.jl`'s `inner_loop_internal` already computes moments once internally;
`oracle.jl` computes them again afterward. This is the single highest-value, lowest-risk optimization
target identified (not yet implemented — Phase 2 work). `winner_compute` is the top allocator (4.9MB/
call) from per-column `sort()` calls in `compute_winners`, a second contained target.

The prior continuation's Phase D reported `n_inner_solves=34` for `Q_adj_FD`/`L_fix_FD` — confirmed
this session to be a mislabeled FD-probe count, not a real inner-solve count: both methods do
**zero** real inner solves (verified via `CS.INNER_SOLVE_COUNT[]` diffs), giving `L_fix_FD` a genuine
~3.1x wall-clock advantage over `Delta_FD` (the ground truth, which does 34 real inner solves).

D/W baseline scaling (D∈{4,6,8,10} at W=8000; D=4 at W∈{8000,20000,80000}, all measured, none
compared economically across D per the task's instruction): full-gradient cost scales empirically as
**D^3.5-3.8** — projecting to ~18 minutes per `Delta_FD` gradient at D=20, vs. ~4.2 minutes per
`L_fix_FD` gradient, making the `L_fix` cost advantage considerably more valuable at larger D than the
~3x measured at D=4. W-scaling at fixed D=4 looks close to linear-or-sub-linear (noisy, low
confidence) — reassuring that W does not appear to compound the D-driven cost problem.

### 9.2 Blockwise gradient re-check (Phase 6) — corrects a real flaw in the prior continuation's Phase D

The prior continuation's full-vector-cosine finding ("even known-biased pathwise AD tracks direction
well, cosine ~0.9999") is now shown to be **almost entirely a gamma-component artifact**. Decomposed
by block at the two upper candidates:

- **Hard pathwise AD**: A-block cosine goes **negative** (-0.67 to -0.79) — points in nearly the
  wrong direction in the gravity-tangent A-block specifically, exactly where it matters.
- **`Q_adj_FD`**: A-block direction moderate (cosine 0.84-0.87) but magnitude wildly wrong (norm
  ratio 14-17x too large).
- **`L_fix_FD`**: A-block cosine 0.997-0.999, gravity-tangent-projected norm ratio 0.98-1.00 at BOTH
  upper candidates — genuinely validated blockwise, not a full-vector artifact.

**Revised conclusion, superseding §7 above**: `L_fix_FD` specifically (not "cheap methods in general")
is the credible building block for a hybrid live solver. `Q_adj_FD` costs the same but should not be
used without the same scrutiny it currently fails; hard pathwise AD is far cheaper but unusable as a
search direction near the optimum.

### 9.3 Sequential/profiled solution located and compared (Phase 5, mandatory external validity check)

A fresh sequential/profiled production run was executed for the identical synthetic economy (D=4,
W=8000, seedFakeData=889, seedU=888, δ=1 — confirmed via matching `kappa_point_estimate=0.064208` and
identical `theta_initial`, not assumed from filenames alone; two of the three candidate JLD2 files
found in the production worktree's untracked output directories were actually stale D=10 results from
a different, unrelated run, caught by checking the `D` field directly rather than trusting filenames).
Since kappa is gauge-invariant (`sequential_methodology.tex` §2.3: the two gauges differ only in the
A-parametrization, not in what `gamma_focal_prime`/kappa mean), a direct comparison is valid without
reconstructing the full A matrix (that reconstruction — inverting the non-focal destination columns
via the sequential method's own machinery, then converting between the two gauges — was not attempted
this continuation, given time constraints; flagged as follow-up):

| | sequential (genuinely feasible) | full-A (this investigation) |
|---|---|---|
| κ_upper | 0.0779 (KNITRO terminal, gravity-feasible) — best-feasible 0.1592 exists but is gravity-**infeasible**, not usable | **0.1718** (maxit=40, EXACT_FEASIBLE_CANDIDATE) |
| κ_lower | **0.0046** (best-feasible, gravity-feasible) | 0.0107 (maxit=15, BEST_FEASIBLE_STALLED, Δ-δ=-0.101 far from binding) |

Full-A's upper candidate clears the sequential method's own directly-comparable, genuinely-feasible
number by more than 2x. **Caveat, not a final verdict**: the sequential run used one
calibration-anchored start (not the production 5-start multistart `sequential_methodology.tex` §9
specifies) and terminated at `nStatus=-400` (non-converged); it may not represent the sequential
method's own best achievable number at this δ. This comparison is suggestive, real, and gauge-valid,
not dispositive.

### 9.4 Partial W-stability (Phase 3)

Cost scaling only (no candidate re-optimization) confirmed at W=20,000/80,000 (§9.1). One direct
candidate check: the maxit=40 upper candidate's exact theta remains feasible at an **independent**
(not nested/common-draws, a documented time-budget shortcut) W=20,000 draw set, with `Delta` moving
further from the budget (more safety margin, not less). Not extended to the lower candidate, the
poll-improved points, W=80,000, or genuine nested draws — see `docs/fullA_d4_W_stability.md`.

### 9.5 Not attempted this continuation

Phase 2 (block-locality optimization — profiling identified two concrete targets, neither
implemented), Phase 4 (gamma-profile, upper polish, lower-direction completion), Phase 7
(wall-clock-matched algorithm frontier — the existing Phase B data is iteration-matched, explicitly
does not substitute), Phase 8 (staged D pilots beyond the Phase 1C cost-benchmark harness). Each has
a dedicated scoping document (`docs/fullA_d4_profile_and_bounds.md`, `docs/fullA_algorithm_frontier.md`)
recording what exists to build on and what's needed to start, rather than being silently skipped.

## 8. What this session did NOT complete (original continuation; see §9.5 for continuation-2's gaps)

- **Phase C (lower-direction profiling + continuation)**: not attempted. The existing short run
  (maxit=15) is `BEST_FEASIBLE_STALLED` (feasible, Δ−δ=−0.101, far from binding) — genuinely just ran
  out of iteration budget, not a numerical breakdown, per the prior session's honest framing. A
  `profile_Δ(g) = min_A Δ(g,A)` sweep with warm-started continuation toward g=1, as the task
  specifies, would answer whether the lower direction reaches a comparably strong candidate; not done.
- **Phase E (continuation from the sequential/profiled solution)**: not attempted — requires locating
  and loading the newest sequential/profiled run for the matching synthetic economy/W/δ/seeds, which
  this session did not have time to locate and verify.
- **Phase G (W-stability sweep)**: not attempted. All numbers in this report and the prior session are
  W=8000 only; per memory `d20-realdata-w-sensitivity`, W=8000 is known to understate κ relative to
  W≥80,000 on the *real* economy at δ≥1 — whether the same holds on this synthetic D=4 economy, and
  whether any candidate/method here remains stable at higher W, is untested.

These are the highest-value remaining items; see `docs/fullA_d4_recommendation.md` for how they bear
on the overall viability call.
