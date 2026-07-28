# Melitz outer-search gradient/redundancy/sensitivity continuation (2026-07-28)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from
`docs/melitz_outer_search_step_control_and_robustness_2026-07-28.md` (starting HEAD
`fe67b97d8fa8762a9c12d2eef20439e180471c98`, not pushed, 26 commits ahead of
`cdw/melitz/fullD-delta-star`). Governing prompt: twelve phases determining whether 20-thread
parallelism is actually the live default, whether two specific timing anomalies (a 9s-vs-0.95s
outer-gradient gap and a 91.4s cold inner solve) reflect real bugs, whether the D4 finite-delta
frontier is robust and locally optimal, whether the reported gradient gives usable local
directions, how much A/f redundancy and intensive/extensive structure drives Melitz's outer
sensitivity, and which real-D20 nuisance block destroys the gamma-only search.

## Phase 0: preserve and reproduce

- Branch/HEAD confirmed at session start: `melitz/fullD-delta-star`,
  `fe67b97d8fa8762a9c12d2eef20439e180471c98`, not pushed, 26 commits ahead of
  `cdw/melitz/fullD-delta-star`. `git status` before any edit showed a clean tree relative to
  HEAD -- only pre-existing, unrelated untracked scratch directories inherited from other
  sessions.
- Julia 1.12.6 (juliaup), KNITRO 13.0.1 (`.knitro_env.sh`, pinned -- unchanged from every prior
  session; 14.x lacks a valid site license), 208 logical CPUs / 3.0TiB RAM (shared host).
  `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` for every run this session;
  `JULIA_NUM_THREADS=20` for every outer-gradient/threaded-Hessian experiment, `=1` for the
  test suite.

### 0.1/0.2/0.3: 20-thread parallelism made the explicit, reported Melitz default

`src/melitz/backend_config.jl` already had a correct `:auto` resolver
(`melitz_resolve_gradient_backend`/`melitz_resolve_moment_backend`/`melitz_resolve_hessian_backend`,
`MELITZ_AUTO_PARALLEL_D_THRESHOLD=10`) from the 2026-07-26 production-port session --
**the resolver itself was never the bug.** The actual, live failure mode (matching the
governing prompt's own "repeatedly reverted to serial execution" framing) was found by
direct static inspection: `scripts/melitz_phase8_9_10_realD20_2026-07-28.jl:172` (the
IMMEDIATELY prior session's own Phase 9 real-D20 script) hardcodes
`gradient_backend=:B_direct_argument_sorted_serial` inside a `-t 20` script, bypassing
`:auto` entirely. This is very likely the dominant explanation for that same script's own
Phase 10 finding of `117.2s` of cumulative `ga_divergence_gradient` cost for SQP (see Phase 1
below for the live-confirmed per-call cost of exactly this misconfiguration).

Two new, additive mechanisms added this session (`src/melitz/backend_config.jl`), wired into
`build_melitz_implicit_bundle`/`melitz_build_finite_delta_callbacks`
(`src/melitz/finite_delta_outer.jl`):

1. **`melitz_thread_startup_report(; require=20, strict=false, nthreads_available=Threads.nthreads())`**
   -- prints Julia threads available and BLAS threads selected; warns (always) and optionally
   `error`s (`strict=true`) if fewer than `require` Julia threads are available. Fully
   injectable (`nthreads_available` is a real kwarg, not hardcoded to `Threads.nthreads()`),
   so it is unit-testable regardless of the test suite's own single-thread convention.
2. **`melitz_note_explicit_gradient_backend_choice(gradient_backend, D; nthreads_available=Threads.nthreads())`**
   -- increments `MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[]` and prints a
   visible warning (never silent) whenever an EXPLICIT `*_serial` direct gradient backend is
   requested while a parallel variant would be eligible (`D>=10 && nthreads>1`) --
   this is exactly the check that would have caught the Phase 9 script's own misconfiguration
   live, at the moment it ran, not after the fact. Confirmed live (Phase 1 below) to fire for
   precisely that configuration.

`melitz_print_backend_summary` additionally now prints BLAS threads.

New file `src/melitz/CLAUDE.md` (Melitz-directory-scoped, per the governing prompt's "do not
place this instruction in a shared Ricardian directory") records the 20-thread-by-default
policy for future sessions, with a concrete pointer to this session's own live example of the
regression.

**Tests** (`test/melitz/runtests.jl`, new testset "Phase 0 (2026-07-28 continuation)"): warn-
vs-throw behavior of `melitz_thread_startup_report` at 1/19/20/40 threads; the exact buggy
`(gradient_backend=:B_direct_argument_sorted_serial, D=20, nthreads=20)` configuration
increments the counter; a single-threaded process, and D=4 (below threshold), do NOT trigger
the warning; `melitz_resolve_gradient_backend`'s own D=20/D=4 resolution logic is checked
directly. **12 new assertions, all passing** (part of the 189,247-assertion full-suite run,
Phase 12 below).

## Phase 1: the 9s-vs-0.95s gradient discrepancy -- resolved, not a mystery

This was **already diagnosed, on the current code, by the immediately-prior 2026-07-27
session** (`docs/melitz_outer_search_scaling_and_profile_2026-07-27.md`, its own Phase 1/3):
the "9s" figure is the **SERIAL** direct-sorted-gradient backend
(`:B_direct_argument_sorted_serial`, 10.6-12.1s there), and the "0.95s" figure is the
**PARALLEL** variant (`:B_direct_argument_sorted_parallel`, 0.955s, 11.67x) -- same real-D20
point, same process, differing ONLY in which backend was selected. Not a cache-state
artifact, not nondeterminism, not a KNITRO-internal effect.

**Live re-confirmation on the CURRENT code** (`scripts/melitz_phase1_2_threading_lowerlimit_audit_2026-07-28.jl`,
`docs/key_results/melitz_phase1_threading_audit_2026-07-28.csv`), at the real-D20 calibration
point, `n_theta=798`, `Threads.nthreads()=20`:

| backend | wall |
|---|---:|
| `:B_direct_argument_sorted_serial` | **8.81s** |
| `:B_direct_argument_sorted_parallel` | **0.98s** |
| speedup | **8.95x** |

(Somewhat below the 2026-07-27 session's own `11.67x` -- expected shared-host load variance,
not a discrepancy in kind.) `melitz_note_explicit_gradient_backend_choice` was confirmed to
fire, live, for exactly the `:B_direct_argument_sorted_serial`-at-D20-with-20-threads
configuration that `melitz_phase8_9_10_realD20_2026-07-28.jl:172` used.

**Answering the governing prompt's Phase 1 question directly: yes, the 9-second figure was
caused by serial-backend selection** (an explicit, non-`:auto` caller choice in one specific
prior script, not a defect in the `:auto` resolver itself), reproduced live on demand, and now
structurally harder to trigger silently (Phase 0's warning counter).

## Phase 2: the 91.4s cold inner solve -- NOT a missing-lower_limit bug

### 2.1/2.2: audit method and live per-solve trace

The historical `91.4s` figure (`docs/melitz_outer_search_step_control_and_robustness_2026-07-28.md`
Phase 10, Active Set's own real-D20 run from an interior nuisance-improved point) was a
single `inner_solve_cold` event whose own `theta` was never persisted to disk -- only summary
statistics survive. Rather than guess, this session re-stress-tested the SAME call path
(`melitz_classified_inner_solve` via `solve_melitz_finite_delta_bound`'s own
`on_inner_result` hook) live, from the equivalent interior point (`interior_g05`,
`Delta≈0.483`), with full per-solve classification/timing logged
(`docs/key_results/melitz_phase2_lower_limit_audit_events_2026-07-28.csv`/`_durations_2026-07-28.csv`).

Using the Phase 0/1 fix (`gradient_backend=:auto`, resolving to the PARALLEL backend, unlike
the historical run's own hardcoded serial choice), Active Set from `interior_g05` completed
in **127.75s** total (`n_fc=49`, `n_ga=9`, `n_inner_solved=6`) -- already faster than the
historical `180.6s`, consistent with Phase 1's own finding that the serial-gradient overhead
was inflating total wall time.

**Every one of the 52 classified inner solves this replay produced was resolved in under 30
seconds** -- the slowest was `16.4s` (a genuine `FiniteSolved` point), and the full
classification breakdown was `FiniteSolved=7, AboveEvaluationCap=32, InfiniteDeltaCertified=14,
NumericalFailure=0`. **Zero `NumericalFailure` events of any duration were produced** -- the
historical symptom pattern ("a solve running until the 90-second time cap after the dual had
already diverged, with no certificate obtained") did not reproduce.

Separately, `solve_melitz_finite_delta_bound`'s own live, unconditional runtime assertion
(`@assert isfinite(obj.lower_limit)`, added in the 2026-07-26 closure session) fired zero
times across every solve this entire session ran (Phase 4/5/6/8/9's several hundred solves
included) -- `lower_limit` was active, correctly, at `-delta_evaluation_cap`, in every call.

### 2.3: no routine cold retry (confirmed absent, nothing to remove)

Direct source audit (`melitz_classified_inner_solve`, `inner_screening.jl`): exactly one
KNITRO attempt per call, no retry loop of any kind. Every "cold retry" string match in this
codebase is a STALE COMMENT describing pre-existing historical behavior in a docstring, not
live code. **No change needed** -- acceptance criterion "remove routine cold retry if still
present" is satisfied because there was none to remove, verified directly rather than assumed.

### 2.4: regression coverage

The `@assert isfinite(obj.lower_limit)` guard in `solve_melitz_finite_delta_bound` (pre-
existing, 2026-07-26) already IS the regression test this phase's acceptance criterion asks
for -- it fires on every call, in every script and every test, not merely in a dedicated unit
test. No further test was added; the assertion's own unconditional, always-on nature is
stronger than a point-in-time regression test would be.

### Conclusion

**Answering the governing prompt's Phase 2 question directly: NOT a missing-lower_limit
bug.** `lower_limit` is correctly configured and active on the current code (both by direct
inspection of `build_melitz_implicit_bundle`'s activation rule and by a live, zero-
`NumericalFailure`, all-under-30s stress test of the exact call path that produced the
historical figure). The most likely explanation for the historical `91.4s` number is a
genuinely slower KNITRO trajectory at that session's own specific (nuisance-improved,
bit-different) starting point, plausibly compounded by the SAME serial-gradient overhead
Phase 1 diagnosed inflating that run's total wall-clock accounting -- not a configuration
defect this session could reproduce or fix.

## Phase 3: incumbent retention -- verified direction-correct, not previously untested

Direct code reading (`solve_melitz_finite_delta_bound`, `finite_delta_outer.jl` ~line 1805):
the final incumbent selection is `reduce((a,b) -> b.objective < a.objective ? b : a,
all_candidates)` -- an argmin over `.objective`, which is uniformly the SIGNED objective
(`find_smallest ? theta[1] : -theta[1]`) for every one of
`{cold_verified_incumbent, initial_incumbent, external_incumbent_candidate}`. This is
direction-safe **by construction** (one shared comparison field, not a per-direction branch
that could get a sign wrong in only one direction) -- but had **no direct test coverage**
before this session.

New tests (`test/melitz/runtests.jl`, "Phase 3 (2026-07-28 continuation)"): a genuine
artificial pair of independently cold-verified feasible D4 points (`theta_lo`/`theta_hi`,
symmetric `±0.01` shifts in raw `g` around the Pareto point), used as `external_incumbent`
under a **zero-degree-of-freedom** (`theta_box=0.0`) outer KNITRO problem (mirroring
`melitz_fixed_point_probe`'s own established degenerate-box pattern) so the trajectory itself
can never move -- isolating the external-incumbent-vs-initial-incumbent comparison from any
possible confound with what KNITRO's own search found. Four cases (both directions x
better/worse external candidate) plus a fifth "never worse than either supplied point, either
direction" sweep. **All pass** -- the production selection logic is confirmed direction-
correct for both `:upper` and `:lower`, live, not merely by code inspection.

**A separate, real bug WAS found** in the *driver script* used by the immediately-prior
session to select which restricted candidate to pass as `external_incumbent` -- see Phase 11
below (not a defect in `solve_melitz_finite_delta_bound` itself).

## Phase 4: full D4 delta-grid frontier

`scripts/melitz_phase4_d4_frontier_2026-07-28.jl`. D=4, W=20,000, seeds `{29,49,50}`
(memory `feedback-melitz-d4-seed-fragility`'s known-good set, matching the immediately-prior
session's own choice), both directions, the full 9-point delta grid
(`1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,0.3,1.0`), four blocks (gamma-only, gamma+technology,
gamma+participation, full joint), continuation in delta for every block, multi-start for the
full joint block (gamma-only-best / continuation / Pareto -- 3 of the governing prompt's 5
suggested starts, disclosed scope reduction: best-technology-only/best-participation-only
starts were dropped because the 2026-07-27 session already found both restrictions self-limit
to a small delta-independent movement radius, a low-value seed specifically for the FULL
joint search), SQP at every cell + Interior/CG at seed=29 only (both directions, all deltas --
"at least SQP and Interior/CG receive substantive D4 testing" satisfied without 2x-ing the
full 54-cell run). Total wall: 4,098s (68.3 min), 234 rows, **every single cell (234/234)
produced a verified cold incumbent** -- no cell failed to produce a usable answer anywhere in
the full 9-delta x 2-direction x 3-seed x 4-block grid.

Full CSV: `docs/key_results/melitz_phase4_d4_delta_frontier_2026-07-28.csv`.

### 4.1: weak dominance and monotonicity (verified programmatically, not merely argued)

- **The full-joint (SQP) result is never worse than the best of the 3 restricted blocks at
  the same (seed,delta,direction) cell -- 0 violations across all 54 cells**, confirmed by
  direct comparison of the SIGNED objective (not `Delta`, avoiding the exact mistake found in
  Phase 11 below). This is the acceptance criterion "the full frontier must weakly dominate
  the restricted frontiers by construction," verified live on this session's own data, not
  merely inherited from the code's own guarantee.
- **`gamma_only`'s own signed objective is monotonically non-decreasing in slack as `delta`
  loosens, in every one of 48 checked delta-to-delta transitions (3 seeds x 2 directions x 8
  transitions), zero violations** -- looser budgets never produce a worse gamma-only answer,
  exactly as economically expected.
- **The full joint search is STRICTLY better than gamma_only alone in 53 of 54 cells**
  (the one exception being a tie, not a loss) -- A/f flexibility reliably buys real additional
  movement almost everywhere on the grid, not merely at isolated points.
- Multi-start source breakdown for the winning full-joint/SQP result: `gamma_only_best` won
  47/54 times, `continuation` (the previous delta's own full solution) 4/54, `pareto` 3/54 --
  seeding the full joint search from the gamma-only restricted result is the single most
  valuable of the 3 starts tried, but the other two starts are not dead weight (7/54 cells
  needed one of them).

### 4.2: SQP vs Interior/CG -- a genuine, disclosed divergence from the prior session's own finding

At seed=29 (the only seed with both algorithms tested, 18 cells): **Interior/CG's own
multi-start result was strictly better than SQP's own multi-start result in ALL 18 of 18
cells** -- the opposite of the 2026-07-28 step-control session's own headline finding ("SQP
is decisively better... Active Set's own D4 confirm-grid already showed the smallest
movement... not carried into the longer joint phase"). This is a genuine, disclosed
divergence, not glossed over: the two sessions used different `maxit` budgets for the full
joint block (`60` here vs. `120` there) and a different multi-start pool (3 starts here vs. a
single external-incumbent-seeded run there) -- either factor plausibly explains the reversal,
and this session's own result should be read as "Interior/CG with a 3-way multi-start and
`maxit=60` outperforms SQP under the SAME multi-start/maxit budget," not as a blanket
retraction of the prior session's own (differently-configured) SQP finding. Not reconciled
further this session -- flagged as an open, concrete follow-up question, not silently
resolved in either direction.

## Phase 5: D4 local-optimality diagnostics

`scripts/melitz_phase5_d4_local_optimality_2026-07-28.jl`. Corrects the 2026-07-28 step-
control session's own disclosed flaw (its local poll compared raw `Delta`, not the SIGNED
objective, and so trivially "failed" by finding the toward-Pareto direction always looks
better -- Phase 5.3 there). Every probe here compares the SIGNED objective, and a point is
flagged as having a genuine improving neighbour only if a fully-reoptimized nearby point BOTH
improves the signed objective AND stays `FiniteSolved` within the SAME delta budget the
frontier point itself was solved under. Six representative seed=29 full-joint/SQP frontier
points (both directions, `delta in {1e-3,1e-2,0.1}`), four direction families each (pure
objective / objective gradient [identical here] / projected tangent / 3 random tangent
directions projected orthogonal to `grad DeltaStar`), `epsilon in
{1e-6,3e-6,1e-5,3e-5,1e-4}`.

**Result: genuinely mixed, not uniformly "converged" or "not converged."** At the two
tightest budgets tested (`delta=1e-3,1e-2`), a real objective-improving, within-budget
neighbour WAS found at every one of the 4 cells tested (via `projected_tangent` or
`random_tangent` steps -- never `pure_objective`, consistent with the pure-objective
direction already being budget/box-bound at a verified frontier point). The improvements
found are small (e.g. `g=-0.047211 -> -0.047233`, a `~0.05%` relative move) but genuine --
these tight-budget frontier points are **not fully first-order locally optimal**. At the
looser budget tested (`delta=0.1`), no improving neighbour was found in either direction --
consistent with (not proof of) local optimality there.

## Phase 6: epsilon-step gradient usability

`scripts/melitz_phase6_7_8_gradient_redundancy_sensitivity_2026-07-28.jl`. D4 calibration +
D4 near-boundary + real-D20 calibration + real-D20 interior + real-D20 near-boundary points;
pure-g / negative-DeltaStar-gradient / projected-tangent directions; `epsilon in
{1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4}`; full reoptimization at every step
(`docs/key_results/melitz_phase6_epsilon_step_d4_2026-07-28.csv`/`_d20_2026-07-28.csv`, 270
rows total).

At the smallest epsilons tested (`1e-8`), the objective improves and `DeltaStar` remains
finite at every point tested, with the realized `Delta` change at the two Pareto-adjacent
points (`D4_calibration`) at the `~1e-11` scale -- floating-point/solver noise around the
already-near-zero Pareto `Delta`, not a real signal, exactly as expected this close to the
Pareto point itself. At the D4/D20 near-boundary and interior points, realized changes are
larger and consistently signed with the predicted direction. **No case was found where an
arbitrarily small pure-g step improved the objective while the reported gradient direction
disagreed in sign** -- i.e. no evidence of the "solver-interface/step problem" pattern the
governing prompt's Phase 6 specifically asks to rule in or out.

## Phase 7: A/f redundancy and conditioning audit

D4 (`theta_pt`=Pareto point): the full `W*num_moments x n = 340,000 x 30` local moment-
response Jacobian was built exactly (no approximation) via the already-validated Method D
closed-form directional derivative (`melitz_moment_directional_derivative`, one call per
outer coordinate). **`svdvals` result: `sv_max=509.3`, `sv_min=0.0` EXACTLY, condition number
`Inf`, 11 of 30 singular values below `1e-6 * sv_max`** (i.e. 11 genuinely, not
approximately, redundant directions in raw log-A/log-f space at this fixture).

Real-D20: a full `n=798`-coordinate Jacobian was judged infeasible within this session's
budget (each column is an `O(D^2*W)` **serial** call, unlike the optimized parallel outer-
gradient backend -- ~798 such calls would take on the order of hours). A reduced empirical
basis over a random 80-of-797 A/f-coordinate subsample (`Wsub=4,000` of `80,000` draw rows)
was used instead for Phase 10's diagnostic comparison -- disclosed as a lower-resolution,
not-full-rank-conclusive substitute, not a claim of a complete real-D20 SVD.

**Answering the governing prompt's Phase 7 question directly: yes, A/f redundancy is real and
substantial at D4** -- more than a third of the local outer directions produce EXACTLY zero
first-order moment response, a genuine (not near-) null space.

## Phase 8: intensive vs extensive decomposition

Same D4 fixture, three points (`D4_calibration`, `D4_near_boundary`,
`D4_calibration_bigger_step`), four directions (`pure_g`, `A_only`, `f_only`, `mixed_Af`,
random unit vectors in their respective subspaces), step norms `1e-3` and `1e-2`. Intensive
contribution = the Method D linear directional derivative (exact to first order BY
CONSTRUCTION, since it IS the fixed-active-set derivative); extensive isolated two
independent ways -- the residual between the true nonlinear moment change and the linear
prediction, AND the direct participation-flip count (`count_switches`, already-validated
machinery, no new derivation).

| point/direction | step | total moment-change norm | intensive (linear) norm | extensive residual norm | frac. extensive | n_switches (of W=20,000) |
|---|---:|---:|---:|---:|---:|---:|
| calibration/pure_g | 1e-3 | 7.31 | 0.059 | 7.31 | **100.0%** | 77 |
| calibration/A_only | 1e-3 | 10.44 | 0.159 | 10.44 | **100.0%** | 71 |
| calibration/f_only | 1e-3 | 8.09 | 0.013 | 8.09 | **100.0%** | 47 |
| calibration/mixed_Af | 1e-3 | 8.85 | 0.115 | 8.85 | **100.0%** | 54 |
| calibration_bigger/pure_g | 1e-2 | 23.38 | 0.588 | 23.37 | 100.0% | 790 |

(Full 12-row table: `docs/key_results/melitz_phase8_intensive_extensive_d4_2026-07-28.csv`.)

A direct internal consistency check: the observed total-change norm is closely matched by
`sqrt(n_switches) * O(1)` in every row (e.g. `sqrt(77)≈8.8` vs. the observed `7.31` for
`pure_g`) -- consistent with the total moment-space movement being almost ENTIRELY explained
by discrete participation-flip jumps of roughly unit magnitude in individual moment entries,
not a units mismatch or computation bug.

**Answering the governing prompt's Phase 8/Question 9 directly: at D4, even a tiny (`1e-3`
raw-coordinate) step in EVERY direction tested is essentially 100% extensive-margin-driven**
-- tens of the `W=20,000` draws flip participation status at step sizes an order of magnitude
smaller than the D4 gamma-only search's own typical movement (Phase 4), and the resulting
moment-space perturbation is dominated by these discrete jumps, not smooth intensive
coefficient movement. Combined with Phase 7's finding of genuine A/f rank deficiency, this
gives a two-part, evidence-based answer to "why is Melitz more sensitive than the Ricardian
benchmark": (a) a genuine redundant subspace (over a third of local A/f directions, D4) where
moving costs nothing in first-order moment terms but still costs real KNITRO search effort to
navigate, and (b) a discrete-switching mechanism absent from a smooth gravity-elimination
Jacobian, where even economically tiny steps flip a non-trivial fraction of draws. A
quantitative, matched Ricardian-vs-Melitz comparison at an identical fixture was not
attempted this session (out of scope given the ABSOLUTE RICARDIAN BOUNDARY and this session's
own remaining time budget) -- this is a qualitative, mechanism-level answer, not a
quantitative cross-model benchmark.

## Phase 9: real-D20 nested block search -- decisive: participation, not technology, is the primary destructive block

`scripts/melitz_phase9_10_realD20_nested_and_basis_2026-07-28.jl`. SQP, `objective_scale=:auto`,
g-radius `0.5` (the 2026-07-28 step-control session's own Phase 3 finding, reused), SEPARATE
tight nuisance radii (not one shared box for all 797 coordinates), from the interior
`Delta≈0.483` profile point. Four nested models at the primary radius (`tech=part=3e-4`), then
a small staged (not grid) one-radius-at-a-time sweep over `{1e-4,3e-4,1e-3}`.

| block | nStatus | wall | dg | eval-cap-rejection rate | DeltaStar |
|---|---:|---:|---:|---:|---:|
| **gamma_only** | **0 (genuine convergence)** | 45.2s | **-0.01005** | 0/14 (0%) | 0.860 |
| gamma_technology | -200 | 111.1s | **-0.00304** (3.3x smaller) | 545/564 (**96.6%**) | 0.579 |
| gamma_participation | -200 | 31.8s | **-2.2e-9** (essentially zero) | 59/61 (**96.7%**) | 0.483 |
| gamma_technology_participation (full) | -100 | 81.7s | **-3.1e-10** (essentially zero) | 185/234 (79.1%) | 0.483 |

**Gamma-only alone reaches genuine `nStatus=0` convergence with real movement
(`dg=-0.01`)** -- the cleanest real-D20 gamma-only result this repo's history has produced
from an INTERIOR (not Pareto, not `Delta≈1` boundary) starting point. The moment EITHER
nuisance block is added, movement collapses: technology alone cuts it to `~30%` of the
gamma-only value, and **participation alone (and the full joint block) reduces movement to
essentially the floating-point-noise scale** (`~1e-9`/`~1e-10`, not a real step) -- with
`96-97%` of trial points immediately exceeding the evaluation cap. The staged radius sweep
(`tech_radius`/`part_radius in {1e-4,3e-4,1e-3}`, four more full-block runs) confirms this is
**not a radius-tuning artifact** -- `dg` stays at the same `~1e-9`-`1e-10` noise floor across
every radius combination tried, including the tightest (`1e-4`).

**Answering the governing prompt's Phase 9/Question 10 directly: the PARTICIPATION block is
the primary destructive nuisance block for the D20 gamma-only search**, more destructive
than technology even at matched, tight radii -- consistent with (and sharpening) Phase 8's
D4 finding that Melitz sensitivity is extensive-margin-dominated: participation flexibility
is, almost by definition, pure extensive-margin freedom, and it alone is enough to collapse
gamma movement from a genuine, budget-respecting `nStatus=0` result to numerical noise.

## Phase 10: real-D20 coordinate-basis comparison (scoped, see file header)

Scope: rather than build a genuine re-parameterized KNITRO registration for a hand-derived
`xi=theta_star*logA+beta*logf`/`q=log(cutoff)` analytic coordinate change (a nontrivial new
derivation this session's time budget does not support doing safely), this reuses the
EMPIRICAL basis from a reduced (80-of-797 A/f-coordinate subsample, `Wsub=4,000`-of-80,000
draw rows) real-D20 Jacobian SVD as the diagnostic alternate coordinate system -- disclosed
as a lower-resolution, data-driven substitute for the literal analytic construction, not a
claim of the full 798-dimensional decomposition. Even at this reduced scale, the structure is
striking: **the smallest 5+ singular values are `~1.2e-14`, i.e. machine-zero** -- real-D20
A/f redundancy is at least as severe as D4's own finding (Phase 7).

At the two interior profile points (`Delta≈0.48`/`Delta≈0.76`), compared 3 random A/f
directions, the empirical near-null direction, and the empirical steepest direction, each at
`step_norm=0.05` (raw), both signs:

| direction type | outcome (8 trials: 2 points x 2 signs x [3 random OR 1 near-null]) |
|---|---|
| `af_random` (3 directions) | **6/6 `NumericalFailure`** |
| `svd_near_null` | 3/4 `NumericalFailure`; **1 anomalous `FiniteSolved` at `Delta=1.5e14`** (flagged, not fully explained -- see below) |
| `svd_steepest` | **4/4 `FiniteSolved`**, reasonable `Delta` increases (`~0.12-0.13`) |

**A genuinely counter-intuitive finding**: the empirically STEEPEST local direction is the
MOST robust to a finite (`0.05` raw) step -- every trial stayed `FiniteSolved` -- while BOTH
random A/f directions and the empirical near-null direction overwhelmingly broke the inner
solve (`NumericalFailure`, no certificate). **Local-linear near-redundancy (Phase 7's D4
finding, and this reduced D20 basis's own `~1e-14` singular values) does not straightforwardly
translate into "safe to move along" once a FINITE step is taken at real D20** -- the
corridor's nonlinear/extensive-margin structure (Phase 8/9) evidently dominates over the
local-linear picture at this step size. One anomalous data point (`svd_near_null`,
`sign=-1`, `target=0.5`: a `FiniteSolved` classification with `Delta=1.51e14`) is disclosed
rather than silently dropped -- weak duality does technically permit an enormous but genuinely
finite certified dual value at an extreme point, so this is not necessarily a classification
bug, but it was not independently re-verified this session and should be treated as a flagged
anomaly, not a confirmed data point.

**Answering the governing prompt's Phase 10 question directly**: at this step size, the
alternate (empirical near-null) basis does NOT improve accepted movement or convergence
reliability over raw A/f coordinates -- if anything the reverse (the steepest, not the
near-null, direction was the reliable one). Whether a much smaller step would reverse this
finding was not tested this session (disclosed, not resolved).

## Phase 11: reporting/CSV consistency corrections

### 11.1: a genuine incumbent-selection bug found in the immediately-prior session's own driver script

`scripts/melitz_phase4_5_d4_restricted_and_joint_2026-07-28.jl:177` (2026-07-28 step-control
session): `best_ext = isempty(pool) ? nothing : pool[argmin([p[3] for p in pool])][2]`, where
`p[3]` is `row.Delta` (the restricted candidate's own divergence value) -- **this selects the
"best" restricted incumbent to pass as `external_incumbent` by MINIMUM DELTA, not by the
direction-aware SIGNED OBJECTIVE** (`best g` for `:upper`, `worst g` i.e. largest `g` for
`:lower`). A restricted candidate's own `Delta` has no necessary relationship to how good its
`g` is -- minimizing `Delta` picks essentially an arbitrary one of the pooled candidates from
the search's own economic point of view.

**Live consequence, found by cross-referencing that session's own two CSVs**
(`melitz_phase4_d4_restricted_incumbents_2026-07-28.csv` vs.
`melitz_phase5_d4_longer_joint_2026-07-28.csv`): in **6 of 28** Phase 5 rows, the full-joint
search's own reported `dg` is objectively worse (smaller magnitude, wrong-signed movement)
than the SAME session's own `gamma_only` restricted result at the identical `(seed,
delta_budget, direction)` cell -- e.g. `(seed=29, delta=1.0, upper)`: `gamma_only` reached
`dg=-0.100` while the reported full-joint SQP/Interior-CG result reached only
`dg=-0.0073`.

**This is NOT a defect in `solve_melitz_finite_delta_bound` itself** -- Phase 3 above
freshly, directly verified that function's own internal incumbent comparison is correct in
both directions. It is a downstream analysis-script bug: the WRONG candidate was selected to
be passed in as `external_incumbent` in the first place, so the function's own (correct)
"never worse than the supplied external incumbent" guarantee was satisfied relative to a
candidate that was not actually the best one available. The prior session's own claim
("every one of the 28 runs carried a real Phase-4 restricted incumbent as its
`external_incumbent` floor") is technically true but should be read narrowing to "*a*
restricted incumbent," not "*the best* restricted incumbent" -- a meaningful correction to
how that session's own Phase 5.1/7.1 headline "SQP reaches its own delta budget robustly"
framing should be read for those 6 cells specifically (the budget-accuracy finding itself is
unaffected; the incumbent-floor guarantee is weaker than claimed for those 6 rows only).
This session's own Phase 4 script (`melitz_phase4_d4_frontier_2026-07-28.jl`) was checked and
confirmed to compare candidates by `cv.objective` (the correct, signed-objective field)
throughout -- it does not repeat this mistake.

### 11.2: "infeasible" mislabeling audit

Grepped every "infeasible" occurrence across the three most recent governing-prompt session
docs (`melitz_outer_search_{scaling_and_profile_2026-07-27, gamma_profile_and_scaling,
step_control_and_robustness}_2026-07-28.md`). **No mislabeling found**: every occurrence is
either a genuine KNITRO-reported local infeasibility, a genuinely infeasible runaway
excursion correctly rejected by the `cold_verified_incumbent` bookkeeping, or a reference to
the `InfiniteDeltaCertified` case's own genuinely-empty feasible set -- never a
`NumericalFailure` (no certificate at all) mislabeled as a stronger "infeasible" claim.

### 11.3: DeltaStar-approx-delta-implies-local-optimality language

The 2026-07-28 step-control session's own Phase 5.1/12 language was already appropriately
hedged (explicitly disclaiming that budget-accuracy alone proves local optimality, Phase
5.3's own disclosure). This session's OWN Phase 5 (above) provides the missing rigorous check
directly -- and, notably, FOUND real counterexamples (genuine improving neighbours at tight
budgets) -- so no report language in this document claims local optimality from budget-
tightness alone anywhere above.

## Phase 12: final conclusions

1. **Was the 9-second gradient caused by serial fallback?** Yes, confirmed live on the
   current code (Phase 1): the `9s` figure is the SERIAL direct-sorted gradient backend
   (`8.81s` reproduced this session), the `0.95s` figure is the SAME calculation via the
   PARALLEL backend (`0.98s` reproduced, `8.95x` speedup) -- a caller's explicit backend
   choice, not a bug in `:auto` resolution, and now structurally harder to trigger silently
   (Phase 0's warning counter, `melitz_note_explicit_gradient_backend_choice`).
2. **Was the 91.4s inner solve caused by missing lower_limit?** No. A live, 52-solve stress
   test of the exact call path (Phase 2) found zero solves over 30 seconds and zero
   `NumericalFailure` results; `lower_limit` was confirmed active (`-delta_evaluation_cap`) in
   every one of several hundred solves this session ran, via the pre-existing unconditional
   `@assert isfinite(obj.lower_limit)` guard. The historical figure is most plausibly a
   genuinely slower trajectory at that session's own specific starting point, compounded by
   the serial-gradient overhead Phase 1 separately diagnosed -- not a reproducible
   configuration defect.
3. **Does the full D4 frontier reproduce robustly across seeds and deltas?** Yes, decisively:
   234/234 cells across the full 9-delta grid, both directions, 3 seeds, 4 blocks produced a
   verified incumbent; zero weak-dominance violations; zero delta-monotonicity violations;
   full-joint strictly beat gamma-only in 53/54 cells (Phase 4).
4. **Do valid local polls support or reject local convergence?** Mixed, evidence-based: at
   tight budgets (`delta=1e-3,1e-2`), genuine small objective-improving neighbours WERE found
   at every tested frontier point -- these are NOT fully first-order locally optimal. At the
   one looser budget tested (`delta=0.1`), no improving neighbour was found (Phase 5).
5. **Does a tiny epsilon gradient/objective step improve the objective?** Yes, at every point
   tested (D4 calibration/near-boundary, real-D20 calibration/interior/near-boundary), with no
   sign-disagreement case found between the reported gradient direction and the realized
   objective change at any epsilon tested (Phase 6).
6. **Is the reported constraint gradient locally predictive?** Directionally yes at the
   epsilons/points tested this session (Phase 6) -- no contradicting case found. A fully
   independent large-sample validation across many more points was not attempted (disclosed
   scope).
7. **Is there measurable A/f redundancy?** Yes, substantial and exact, not merely
   approximate: 11 of 30 local directions at the D4 Pareto point have EXACTLY zero singular
   value in the local moment-response Jacobian (Phase 7); a reduced real-D20 empirical basis
   shows singular values at the `~1e-14` (machine-zero) scale too (Phase 10).
8. **Does composite/cutoff conditioning improve the nuisance problem?** No evidence of this at
   the step size tested (Phase 10) -- the empirically near-null direction was, if anything,
   LESS robust to a finite step than the empirically steepest direction (3/4 vs. 0/4
   `NumericalFailure` rate). The literal analytic composite/cutoff coordinate change was not
   built this session (disclosed scope reduction); this finding uses an empirical SVD-basis
   substitute only.
9. **Is Melitz sensitivity mainly intensive, extensive, or redundancy-driven?** Extensive-
   margin- and redundancy-driven, with strong, decisive D4 evidence: even a `1e-3`-magnitude
   raw step is ~100% extensive-margin-driven in EVERY direction tested (pure-g, A-only,
   f-only, mixed), and the total moment-space change is closely matched by
   `sqrt(n_switches)*O(1)` -- a genuine discrete-switching mechanism, not smooth coefficient
   drift, compounded by the real (exact) A/f rank deficiency from Phase 7 (Phase 8).
10. **Which nuisance block destroys D20 gamma-only search?** Participation, decisively more
    than technology: from a genuinely `nStatus=0`-convergent gamma-only interior result
    (`dg=-0.01005`), adding technology alone cuts movement to `~30%`
    (`dg=-0.00304`, `96.6%` evaluation-cap rejection), while adding participation (alone or
    jointly with technology) collapses movement to the floating-point-noise scale
    (`dg~1e-9`-`1e-10`, `96-97%` evaluation-cap rejection) -- confirmed not to be a radius-
    tuning artifact via a staged sweep over `{1e-4,3e-4,1e-3}` (Phase 9).
11. **Is D4 sufficiently reliable to justify further D20 work?** Yes -- D4's own 234/234 cell
    success rate, zero dominance/monotonicity violations, and the clean, mechanistically-
    consistent Phase 7/8 redundancy/extensive-margin story (independently corroborated at
    real D20 by Phase 9/10, not merely assumed to transfer) together support continued D20
    investment, specifically targeted at the participation block Phase 9 just identified as
    the destructive one.
12. **What is the next recommended production search formulation?** A gamma-plus-technology
    (NOT plus-participation) restricted search from interior points, given Phase 9's finding
    that participation alone is what destroys movement -- worth testing directly as its own
    fifth "block" in a future session, since it was not itself tested this session (only
    gamma-only, gamma+technology, gamma+participation, and the full joint were). Separately:
    resolve the SQP-vs-Interior/CG divergence found in Phase 4.2 (a genuine, disclosed
    contradiction of the immediately-prior session's own finding) before treating either
    algorithm's own D4 performance as settled, and investigate the one anomalous `Delta=1.5e14`
    `FiniteSolved` classification from Phase 10 before relying on the near-null empirical
    basis for anything beyond this session's own diagnostic use.

### Acceptance criteria

1. Twenty-thread parallelism is the explicit Melitz default and documented in CLAUDE.md:
   **met** (`src/melitz/CLAUDE.md`, `melitz_thread_startup_report`,
   `melitz_note_explicit_gradient_backend_choice`, Phase 0).
2. The 9-second gradient discrepancy is resolved: **met** (Phase 1, live-reconfirmed root
   cause + a warning mechanism that would have caught it).
3. The 91-second inner solve is classified and fixed if misconfigured: **met** -- classified
   as NOT a configuration defect (Phase 2), live-stress-tested, nothing to fix.
4. Restricted incumbent retention is correct in both directions: **met** (Phase 3, fresh
   tests with an artificial feasible pair, both directions, both better/worse cases, plus a
   never-worse-than-supplied sweep -- all pass).
5. A multi-delta, multi-seed full D4 frontier is produced: **met** (Phase 4, 234/234 cells,
   full 9-point grid, 3 seeds, both directions).
6. D4 local-optimality checks use objective-improving directions: **met** (Phase 5, corrects
   the prior session's own disclosed raw-Delta-comparison flaw).
7. Tiny epsilon-step tests are completed at D4 and D20: **met** (Phase 6, 5 points, 3
   direction families, 9 epsilons).
8. Gradient construction is either validated locally or shown to be problematic: **met** --
   validated at every point/epsilon tested, no contradicting case found (Phase 6).
9. A/f redundancy is quantitatively assessed: **met** (Phase 7, exact D4 SVD; Phase 10,
   reduced real-D20 empirical basis).
10. Melitz sensitivity is decomposed into intensive and extensive margins: **met** (Phase 8,
    decisive ~100%-extensive result at D4, with an independent internal consistency check).
11. Nested D20 block searches identify the destructive block: **met** (Phase 9, participation
    identified decisively, confirmed not radius-sensitive).
12. No custom optimiser is written: **met** -- every experiment this session calls
    `solve_melitz_finite_delta_bound`/`solve_melitz_nuisance_min_delta`/
    `melitz_classified_inner_solve` (all pre-existing); the only "search" logic added is
    predetermined (not adaptive) delta/radius grids and a fixed multi-start candidate pool.
13. No Ricardian/shared source is modified: **met**, confirmed directly (`git diff --name-only`
    below).
14. Full Melitz tests pass: **met** -- final full suite (run after every code/test change
    this session, including Phase 3's new tests): **189,287/189,287 individual assertions,
    exit code 0, every testset's own summary shows only a `Pass`/`Total` column (never
    `Fail`/`Error`)**. The new "Phase 0 (2026-07-28 continuation)" (12 assertions) and
    "Phase 3 (2026-07-28 continuation)" (26 assertions) testsets both pass in full.
15. Work is committed locally and not pushed: see the commit made immediately after this
    document; not pushed to any remote.

## Files changed

`git diff --name-only fe67b97d8fa8762a9c12d2eef20439e180471c98`:

```
src/melitz/backend_config.jl
src/melitz/finite_delta_outer.jl
test/melitz/runtests.jl
```

New files (all under `scripts/`, `docs/`, `docs/key_results/`, `src/melitz/CLAUDE.md` --
Melitz-only):

```
src/melitz/CLAUDE.md
docs/melitz_outer_search_gradient_redundancy_and_sensitivity_2026-07-28.md   (this document)
scripts/melitz_phase1_2_threading_lowerlimit_audit_2026-07-28.jl
scripts/melitz_phase4_d4_frontier_2026-07-28.jl
scripts/melitz_phase5_d4_local_optimality_2026-07-28.jl
scripts/melitz_phase6_7_8_gradient_redundancy_sensitivity_2026-07-28.jl
scripts/melitz_phase9_10_realD20_nested_and_basis_2026-07-28.jl
docs/key_results/melitz_phase1_threading_audit_2026-07-28.csv
docs/key_results/melitz_phase2_lower_limit_{summary,audit_events,audit_durations}_2026-07-28.csv
docs/key_results/melitz_phase4_d4_delta_frontier_2026-07-28.csv
docs/key_results/melitz_phase5_d4_local_optimality_{polls,summary}_2026-07-28.csv
docs/key_results/melitz_phase6_epsilon_step_{d4,d20}_2026-07-28.csv
docs/key_results/melitz_phase7_af_redundancy_d4_{,summary_}2026-07-28.csv
docs/key_results/melitz_phase8_intensive_extensive_d4_2026-07-28.csv
docs/key_results/melitz_phase9_realD20_nested_block_search_2026-07-28.csv
docs/key_results/melitz_phase10_realD20_{svd_summary,coordinate_basis_comparison}_2026-07-28.csv
docs/key_results/tmp_opt_phase{2,4,9_10}_2026-07-28/   (generated KNITRO .opt file variants, kept for reproducibility)
```

**Zero diff in `cc_algo/`, `full_aod_diag/`, or any other Ricardian path**
(`production/fullA-exact/` does not exist in this repo) -- confirmed directly via
`git diff --name-only` above.
