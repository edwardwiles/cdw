# Full-D Melitz Δ\* benchmark: design document (ACTIVE closure)

Branch: `melitz/fullD-delta-star` (from `production/fullA-exact` @ `670eac4`). This
document describes the **active minimal-moment / LFD-recovery closure**, which supersedes
an earlier f_entry-primitive / N-derived / N'=N closure implemented on this same branch.
The full historical derivation of the superseded closure (firm algebra, Pareto
tail-moment formulas, wage-solve fixed point, gravity-restriction construction — all of
which are REUSED unchanged below) is preserved in
`docs/melitz_delta_star_v1_superseded_closure.md`; this document does not re-derive that
material, only what changed and why.

**2026-07-22 update (critical bug fix + population-Pareto reconstruction, see Section 13):**
Sections 11-12 below (the exact-sample-correction benchmark and its `nStatus=-102`
failure) are **RETRACTED**. Two real bugs made that entire benchmark point internally
inconsistent — not a genuine CC-feasibility failure as previously reported. Both are fixed;
Section 13 documents the fix and the now-passing real-KNITRO result
(`nStatus=0`, `Delta(theta*)` small/positive/declining in `W`).

**2026-07-22/23 update (Gate A validation, see Section 14): a further real bug was found
and fixed** — the gains-from-trade formula reported in Section 13.4 below omitted the
baseline/autarky wage ratio and is **WRONG SIGN** at this benchmark (`-0.0075` reported,
`+0.052` correct, confirmed via the ACR/Chaney identity to `~1e-14`). A SECOND real bug was
also found and fixed: the gravity restriction (`gravity_residuals`/
`gravity_coefficient_vector`/`GravityPivot`, used throughout `equilibrium.jl`/
`delta_star.jl`/`fake_data.jl`) used `doubleDiff` (an anchored, asymmetric contrast),
which does NOT reproduce `production/fullA-exact`'s own canonical OLS-two-way-FE gravity
coefficient (`withinTransform`, universal in every production run config) — verified live,
off by up to 2.3 in the coefficient and occasionally the wrong SIGN. Both are fixed; the
default fixture seed also changed (2 → 29) as a consequence of the gravity-transform fix.
See Section 14 for the full Gate A report.

## 0. Scope of this milestone

Per the governing prompt: implement exactly the reduced `D^2+1`-moment system, solve the
fixed-parameter Christensen–Connault minimum-divergence problem, recover the least-
favorable distribution (LFD), reconstruct every equilibrium object profiled out of the
active moments, and verify every omitted equation ex post under that LFD — for the exact
Pareto benchmark and a handful of nearby gravity-feasible perturbations. Finite-δ
upper/lower bound programs remain **out of scope**.

## 1. What changed vs. the superseded closure

| claim (superseded doc) | status now |
|---|---|
| `f_entry[o]` is an observed primitive | **superseded** — not a field of `MelitzPrimitives` at all; recovered from the LFD post-solve (`recover_entry_costs_from_lfd`) |
| `N_o` (baseline entrant mass) derived from `f_entry_o`, generally ≠1 | **superseded** — `N_o≡1` for every origin, a universal normalization (`normalize_baseline_entrant_mass` converts an arbitrary economy to this) |
| `N'[target] = N[target]` (reused unchanged) | **superseded** — `N'[target]` is recovered post-LFD from TWO independent formulas (market clearing, price index) and compared, never imposed |
| `gamma_prime[target]` (`price_power_prime`) precomputed in closed form before the inner solve | **superseded** — `gamma_prime_target` is a SEARCHED field of `MelitzPrimitives`, part of the packed outer vector |
| `f[target,target]` derived from a fixed-mass ACR-style baseline-cutoff-pinning trick | **superseded** — derived directly from `(gamma_prime_target, A[target,target], L[target])` via the autarky cutoff-at-one condition (`derive_fjj_from_autarky_cutoff`), independent of any baseline data |
| moment system: `D^2` trade-share + `D` free-entry moments (`D^2+D` total) | **superseded** — `D^2` trade-share + exactly 1 focal baseline-vs-autarky free-entry LINK moment (`D^2+1` total) |
| `run_melitz_inner_delta` returns `(val, x, nStatus)` only | **superseded** — returns a `MelitzLFDResult` including the recovered, normalized LFD weights and every moment residual evaluated under them |
| reference draws: `MersenneTwister` pseudorandom | **superseded as default** — scrambled Halton (`cc_algo/rhalton.jl`) is now the default (`pareto_draws(...; mode=:halton)`); pseudorandom kept as an explicit `mode=:pseudorandom` robustness check |
| gravity restrictions evaluated via `doubleDiff(A)`/`doubleDiff(f)` on levels | **unchanged, but re-verified**: `misc/doubleDiff.jl`'s `doubleDiff` already logs internally, so this always was `DD(log A)`/`DD(log f)` — confirmed correct, not re-derived differently (an interim "fix" wrapping inputs in an extra `log.()` was tried, found to double-log and blow up at `tau=1`, and reverted) |

Also **new** (not present, or present differently, in the superseded closure): the two
gravity restrictions are now eliminated via an explicit `GravityPivot` construction
(`equilibrium.jl`), giving a genuine `2D^2-2`-dimensional FREE coordinate system (the
outer search space) on top of the `2D^2`-dimensional ECONOMIC outer vector — the
superseded closure carried the two restrictions as separate outer-loop equality
constraints rather than eliminating them structurally.

## 2. Active parameter/moment counts (D=4)

| quantity | count | formula |
|---|---|---|
| stochastic moments | 17 | `D^2 + 1` |
| economic outer parameters | 32 | `1 (gamma_prime_j) + D^2 (A) + (D^2-1) (f, excl. f[j,j])` `= 2D^2` |
| free numerical coordinates (post gravity-pivot) | 30 | `2D^2 - 2` |

## 3. Object status table (main prompt Section 14)

| object | status | dimension | where determined |
|---|---|---|---|
| `A[o,d]` | searched | `D^2` | outer vector (`delta_star.jl`) |
| `f[o,d]`, except `f[j,j]` | searched | `D^2-1` | outer vector |
| `gamma_prime[j]` | searched | 1 | outer vector |
| `f[j,j]` | derived | 1 | `derive_fjj_from_autarky_cutoff` (`equilibrium.jl`) from the autarky cutoff-at-one condition |
| baseline `gamma[d]` | normalized to 1 | `D` | universal, not stored |
| baseline `N[o]` | normalized to 1 | `D` | universal, not stored (`normalize_baseline_entrant_mass` converts an arbitrary economy) |
| `f_entry[o]` | recovered from LFD | `D` | `recover_entry_costs_from_lfd` |
| `N'[j]` | recovered from LFD + autarky market clearing | 1 | `recover_N_prime_market_clearing` (cross-checked against `recover_N_prime_price_index`) |
| baseline wages `w[o]` | solved from data | `D` | `melitz_solve_wages` (unchanged, reused verbatim) |
| `w'[j]` | numeraire | 1 | fixed at 1 |
| gravity restrictions | deterministic outer equalities | 2 | eliminated via `GravityPivot` (A over the full `D^2` domain, `f` over the `D^2-1` free cells) |

## 4. Moment system (main prompt Section 4)

`melitz_moments!` (`moments.jl`) fills exactly `D^2+1` columns:

- **`D^2` trade-share moments**: `g_trade[o,d](z) = price_od(z_o)^(1-sigma)*active_od(z_o)
  - lambda_data[o,d]`. With baseline `N_o≡1`, `gamma_d≡1` this is literally
  `realized_revenue_od(z)/expenditure_d - lambda_data[o,d]` — no entrant-mass factor (the
  superseded closure's `entrant_mass[o]*realized_revenue/expenditure_d` collapses since
  `entrant_mass[o]≡1`).
- **1 focal free-entry LINK moment**: `g_free_entry_link(z) = Pi_baseline_j(z)/w[j] -
  Pi_autarky_j(z)/w'[j]`, `Pi_baseline_j(z) = sum_d realized_operating_profit[j,d](z)`,
  `Pi_autarky_j(z) = realized_operating_profit'_jj(z)`. Replaces the superseded closure's
  `D` per-origin free-entry moments AND the `f_entry[j]` parameter simultaneously.

The two gravity restrictions are never `G` columns — F-independent, enforced exactly
(machine precision) by the `GravityPivot` construction, not duplicated as moments.

`K` (`obj.H[:,1]`) is set to `gamma_prime_target - 1`, a documented, UNCONSUMED
placeholder this milestone — confirmed by grepping `cc_algo/inner_loop_functions.jl` that
`K`/`H[:,1]` is never read by a fixed-theta `inner_loop` call (only the full outer
δ-search reads it, out of scope here), and the LFD-recovery reconstruction
(`melitz_recover_lfd`) itself never touches `K` either.

## 5. Outer parameterization and gravity pivots (main prompt Section 2/5)

Two representations (`delta_star.jl`):

- **Economic vector** (`melitz_outer_layout`, length `2D^2`): `(log(gamma_prime_j),
  vec(log A), log_f_free)` — NOT assumed gravity-feasible.
- **Free vector** (the ACTUAL `theta` packed for `PsiObjectiveBundleDelta`, length
  `2D^2-2`): `(log(gamma_prime_j), A_free [pivoted, D^2-1], f_free_free [pivoted,
  D^2-2])` — gravity-feasible BY CONSTRUCTION.

`GravityPivot` (`equilibrium.jl`) solves one cell out of an affine constraint
`dot(c,z)+g0=0` to satisfy `Cov(DD(log A),DD(log tau))=0` (A: `g0=0` always, every A cell
free) or `Cov(DD(log f),DD(log tau))=0` (`f`: `g0` depends on the current `f[j,j]`,
rebuilt every evaluation). `c` is computed via `gravity_coefficient_vector`, NOT the naive
`vec(doubleDiff(tau))` guess — `doubleDiff`'s within-transform differences against a
FIXED reference row/column (not a symmetric demeaning projection), so the true
coefficient vector is `M^T @ vec(T)` for `doubleDiff`'s own linear map `M`; `c=vec(T)` was
tried, verified wrong live (residual −2.8, not ~0), and replaced with a robust
unit-log-perturbation computation. The two pivots' cells are kept DISTINCT
(`f_pivot_avoid_index`) — letting both land on the same physical cell was found to
over-concentrate both corrections there, driving that cell's Pareto participation
probability toward 0.

`fake_data.jl`'s fixture construction uses a DIFFERENT (minimum-L2,
`project_to_gravity_manifold`) projection for the SAME two restrictions — dumping an
entire correction onto one pivot cell was found live to occasionally produce an
economically extreme (cutoff below 1, infeasible) synthetic economy; spreading the
correction evenly across every free cell avoids this. The two projection methods serve
different purposes (fixture construction wants a "nice" matrix; the outer coordinate
system needs a well-defined free/pivot split for the eventual optimizer) and are not
required to agree.

## 6. LFD recovery (main prompt Section 9)

`melitz_recover_lfd` (`delta_star.jl`) reuses the EXACT conjugate-derivative recipe
already used (copy-pasted 11×) in `sequential_gravity/run_profiled_production.jl`'s
`recover_lfd`: after `inner_loop(obj,theta)` returns the KNITRO-optimal dual vector
`x=(zeta,lambda...)`, per-draw `arg0[w] = -x[1] - dot(G[w,1:d], x[2:end])`, `dPsi!` maps
`arg0` to unnormalized LFD weights, normalized by their sum. One shared copy was written
for Melitz rather than adding a 12th copy-paste of this pattern.

`MelitzLFDResult` bundles: `Delta`, the dual vector, KNITRO status, normalized weights,
an `lfd_ok` flag, every moment residual evaluated under the LFD, the pre-normalization
sum, and min/max weight + max density deviation (`|W*weight-1|`).

## 7. Ex-post equilibrium checks (main prompt Section 9)

`check_profiled_melitz_equilibrium` (`equilibrium.jl`) computes, ALL under the recovered
LFD (never reference/equal weights): baseline price-index identities (9.1), baseline and
autarky free-entry residuals (9.2/9.3) alongside the entry costs they're built from
(`recover_entry_costs_from_lfd`, `recover_focal_autarky_entry_cost`), autarky market
clearing (9.4), the autarky price-index identity — "the key omitted moment" — and its
independent `N'[j]` cross-check (9.5), the autarky cutoff identity (9.6), cutoff
inequalities (9.9), and gravity (9.8). Every profiled/omitted equation the active moment
system does NOT enforce directly is checked here.

## 8. Reused unchanged from the superseded closure

`firm_quantities.jl` in full (shared per-draw firm routine, markup/price/revenue/profit,
no hybrid/ρ/Bertrand branches); `melitz_solve_wages`/`melitz_K1`/`cell_from_cutoff`/
`build_equilibrium` (still used for Pareto-benchmark construction);
`gravity_residuals`'s formula (log-additive via `doubleDiff`, re-verified not re-derived);
`pareto_tail_prob`/`pareto_tail_power_mean`(`_numeric`); `cc_algo`'s
`PsiObjectiveBundleDelta`/`inner_loop`/`Psi.jl`/the `parallelism_guards.jl` fix, all
unmodified.

## 9. Refactored

`types.jl` (new struct fields/layout), `equilibrium.jl` (demoted `pareto_*` diagnostics +
new gravity-pivot/LFD-recovery/ex-post-check functions), `moments.jl` (`D^2+1` system),
`pareto.jl` (Halton draws), `delta_star.jl` (outer/free layouts, LFD recovery,
general — not fixed-theta\*-only — moment adapter), `fake_data.jl` (`N_o=1` construction,
ACR-seeded `gamma_prime_j`), `fstar_solver.jl` (exact-sample correction solve, see §10).

## 10. Archived / demoted to Pareto-only diagnostics

`equilibrium.jl`'s `pareto_entrant_mass_from_labor`, `pareto_autarky_fixed_cost_fixed_mass`,
`pareto_target_cutoff_fixed_mass`, `pareto_entry_cost_from_free_entry`,
`pareto_solve_autarky_counterfactual_fixed_mass` — the exact SUPERSEDED closed forms,
kept only for the cross-check that the new and old closures agree at the exact Pareto
benchmark (addendum Section 15), never called from the active moment/delta_star path.

## 11. The exact-sample correction solve (`fstar_solver.jl`)

`fake_data.jl` builds a gravity-feasible fixture whose `gamma_prime_target` (ACR-seeded)
and gravity-projected `A`/`f` are NOT yet exact-sample-consistent with the fixed
`X_data=eq.trade_flow` — the ≤2 gravity-pivot-adjusted cells generically mismatch. Main
prompt Section 11/addendum Section 13.1 ask for a correction solve finding a nearby point
where the equal-weight sample-mean of every one of the `D^2+1` moments is ~0.

**Method: per-coordinate bisection (`Roots.jl`), Gauss-Seidel swept, not a gradient-based
minimizer.** `Optim.LBFGS` (both `autodiff=:forward` and `:finite`) was tried first and
found live to stall far from zero or crash: `melitz_firm`'s `active=profit>0`
participation gate makes the mean residual have small `O(1/W)` JUMPS (one draw crossing
the cutoff), the same class of issue as this repo's own documented "winner-boundary
derivative" bug in the related full-A codebase — `ForwardDiff`/finite-difference
gradients are locally blind to (or, for finite differences with a large enough step,
mis-estimate) that jump, so the line search satisfies its stopping criteria on a biased
gradient well before the true residual is small. Bisection needs no derivative; each
trade-share residual is (to bisection resolution) monotonic in its cell's own `A_od` or
`f_od`. An early version let the search bracket grow unbounded (found to occasionally
reach numerical overflow at the bracket endpoints, corrupting the sign-change test and
converging to a spurious root far from anywhere sensible — one coordinate's "correction"
made its own residual 60× worse); capped at `±6` in log-space, generous but safe.

Coordinate assignment: each of the `D^2-1` non-A-pivot free `A` coordinates zeros its own
cell's trade residual; `gamma_prime_j` is zeroed against the focal link residual. The
A-pivot cell and the f-pivot cell (generically DISTINCT cells) are mutually coupled --
`A[A-pivot]` depends on every OTHER `A_free` coordinate (including the one dedicated to
the f-pivot cell) via the affine gravity constraint, and symmetrically for `f[f-pivot]` --
so naive alternating 1D bisection between them was found LIVE to converge geometrically
for a few sweeps then STALL at a nonzero fixed point (confirmed reproducible and NOT a
bisection-specific artifact: an independent full-30-dimension `Optim.LBFGS` polish landed
at the EXACT same stuck point). Fixed by solving this one pair JOINTLY
(`solve_pivot_pair!`, a damped 2D Newton step with a deliberately wide finite-difference
step to average over the `O(1/W)` jump noise rather than differentiate through it) instead
of alternating.

**This joint fix substantially improves but does NOT fully resolve the stall** — see the
live numbers in §12: the corrected fixture still plateaus around max residual ~0.46
(down from ~3-11 at the raw fixture, roughly an order of magnitude, but not the
1e-8 target). This remains flagged as an OPEN finding, not silently accepted.

**RETRACTED 2026-07-22 (see Section 13):** the entire exact-sample-correction approach
(this section and the numbers in §12 below) is superseded by the addendum's population-
Pareto construction. `solve_fstar`/`fstar_solver.jl` is archived as an optional debugging
utility, no longer part of the active fixture-construction path.

## 12. Numerical results (RETRACTED — see Section 13)

**RETRACTED 2026-07-22.** The `nStatus=-102`/`Delta=1e10` failure reported below was
caused by the benchmark point being INTERNALLY INCONSISTENT (two real bugs, Section 13.1),
not a genuine finite-support CC-infeasibility. Kept verbatim for the historical record;
do not cite these numbers.

Fixed benchmark point throughout: D=4, sigma=2.5, theta_star=6.8, target_country=1,
seed=2 (NOT seed=1234 -- the superseded closure's own default seed is infeasible under
the active closure's N_o=1 + minimum-L2 gravity projection; seed=2 is the first verified
feasible seed under the new construction, see fake_data.jl).

**Fixture construction + gravity/feasibility (exact, W-independent):**

| quantity | value |
|---|---|
| `gamma_prime_target` (ACR-seeded initial value) | 0.7561231654704139 |
| `f[j,j]` (derived) | 6.954120717040161 |
| `A` gravity residual | -8.3e-17 (machine precision) |
| `f` gravity residual | -2.1e-15 (machine precision) |
| min baseline cutoff | 1.2049 (feasible, `>=1`) |
| min export-minus-domestic cutoff | > 0 (feasible) |

**Exact-sample correction (`solve_fstar`, addendum Section 13.1) — OPEN FINDING, target not fully met:**

| quantity | W=20,000 | W=80,000 |
|---|---|---|
| correction wall time | 1m54s | 9m39s |
| max trade-share residual (equal weights) | 0.460 | 0.460 |
| max focal link residual (equal weights) | 0.050 | 0.051 |
| gravity residuals (post-correction) | machine precision | machine precision |
| min cutoff (post-correction) | feasible | feasible |

The residual plateaus at essentially the SAME value regardless of `W` (0.46/0.050 at both
20k and 80k) -- consistent with a genuine coupled-coordinate stationary point (§11), not
Monte Carlo noise. Target per main prompt Section 11 is `<=1e-8`; **not reached**. This is
reported honestly, not worked around by loosening the target without cause (the
`test/melitz/runtests.jl` assertion for this step is `<1.0`, documented as a
"substantially reduced, not machine-precision" check, not a silent pass).

**Real KNITRO CC inner minimum-divergence loop, on the corrected fixture:**

| quantity | W=20,000 | W=80,000 |
|---|---|---|
| min active-draw count (worst cell) | 3 | 13 |
| KNITRO status (`nStatus`) | -102 (dual unbounded) | -102 (dual unbounded) |
| `Delta(theta)` reported | 1e10 (sentinel, not a real value) | 1e10 (sentinel) |
| max `\|dual_x\|` | 1.66e16 | 1.58e16 |
| LFD reconstruction | numerically degenerate (near-point-mass; `lfd_ok=true` by the finite/nonneg check, but NOT a trustworthy LFD given `nStatus != 0`) | same |

**KNITRO does not reach a bounded dual (nStatus=0) at either `W` on this fixture,
raw OR corrected.** This mirrors — but is quantitatively worse than — the superseded
closure's own documented small-`W` finding (docs Section 14 of the archived report):
there, `W=80,000` was sufficient because the target residual magnitude was ~0.02-0.04;
here, even after correction, the residual (~0.46/0.05) is roughly an order of magnitude
larger, driven by the SAME unresolved pivot-pair coupling. Consequently the LFD-recovery
and ex-post-equilibrium-check machinery (§6-7, `melitz_recover_lfd`,
`check_profiled_melitz_equilibrium`) is implemented, unit-tested for its own internal
logic (normalization, finiteness, structural correctness), and wired end-to-end into a
real KNITRO call that runs without crashing — but has NOT yet been exercised against a
genuinely converged (`nStatus=0`) real dual solution. This is the primary open item for
follow-up work (see "Remaining discrepancy" below).

**Non-KNITRO test suite: 44/44 passing** (Pareto draws, firm-level calcs, gravity
coefficient/pivot machinery, `N_o=1` normalization invariance, fixture
construction/feasibility, `D^2+1` moment structure/sensitivity,
`min_active_draw_count`/`cell_participation_diagnostics`, nearby gravity-feasible
perturbations, legacy Pareto-only diagnostics). The `solve_fstar` and real-KNITRO
testsets pass their STRUCTURAL assertions (gravity exactness, cutoff feasibility, no
crash) and honestly report (via `@test_broken`, not silently) the two numerical items
above that do not yet meet their target tolerance.

**Remaining discrepancy / next steps:** the coupled A-pivot/f-pivot cell pair is the
single identified root cause blocking both the exact-sample correction and, downstream,
a bounded real KNITRO dual. Candidate follow-ups (not attempted this session, flagged for
the next one): (a) choose `target_country`/seed combinations where the two pivots'
cells are further apart in the gravity-coefficient ranking, empirically reducing
coupling strength; (b) extend `solve_pivot_pair!`-style joint solving to a small cluster
of cells if 2 turns out not to be the true coupling order; (c) revisit whether the
single-cell `GravityPivot` (vs. the minimum-L2 `project_to_gravity_manifold` used only in
`fake_data.jl`) is the right choice for the ACTIVE outer coordinate system, given it
appears to concentrate correction difficulty onto specific cells by construction.

See the session's live-run report (pushed to Dropbox) for the fully populated table,
nearby-perturbation results, and the reused/refactored/archived file lists.

## 13. 2026-07-22: critical bug fix + population-Pareto reconstruction

Two independent real bugs made the §11-12 benchmark point internally inconsistent. Both
are fixed. The exact-sample-correction approach itself is also replaced (per the governing
addendum) with a genuine population-Pareto construction. The result: the SAME class of
real-KNITRO CC inner problem that previously failed (`nStatus=-102`) now converges cleanly
(`nStatus=0`) with a small, positive, `W`-declining `Delta(theta*)`.

### 13.1 Bug 1: autarky revenue evaluated at the wrong price power

`f[j,j]` was correctly derived from the autarky cutoff-at-one condition using
`price_power_d = gamma_prime_target`. But every ACTIVE autarky firm evaluation (7 call
sites: `melitz_moments!`, `focal_link_residual`, `recover_focal_autarky_entry_cost`,
`recover_N_prime_price_index`, both autarky terms inside
`check_profiled_melitz_equilibrium`, and the productivity-one cutoff diagnostic) called
`melitz_firm` with `price_power_d = 1.0` (the BASELINE normalization) instead. Exact
algebraic signature, reproduced live: `residual_autarky_cutoff = -1.6959...  ==
(gamma_prime_target - 1) * f[j,j]`. Fixed at all 7 sites; `residual_autarky_cutoff` is now
`-8.9e-16` (machine precision) at the same point. New unit test
(`test/melitz/runtests.jl`, "Autarky price-power fix"): for several arbitrary
`(gamma_prime, A_jj, expenditure_prime)` triples, verifies both that the CORRECT price
power gives exact zero profit at `z=1`, and that the BUGGY price power gives EXACTLY
`(gamma_prime-1)*w'*f_jj` — the same signature, not just "some nonzero number".

### 13.2 Bug 2: A[j,j]/f[j,j] self-consistency under the GE price-index normalization

Found while rebuilding the fixture as a genuine general equilibrium (§13.3): the baseline
price-index normalization `gamma_d==1` is enforced by rescaling EACH DESTINATION COLUMN of
`A` by a closed-form factor `s_d = (E_d/colsum_d(X))^(1/theta_star)` (this is gravity-
neutral — `doubleDiff` is invariant to per-column log-additive shifts, verified directly
from its definition). Since country `j` is itself a destination, this ALSO rescales
`A[j,j]`. But `f[j,j] = derive_fjj_from_autarky_cutoff(gamma_prime_target, ..., A[j,j],
...)` was derived using the PRE-rescale `A[j,j]`, while every later use of `f[j,j]`
(moments, ex-post checks) pairs it with the POST-rescale `A_final[j,j]`. Live signature:
`melitz_firm` at `z=1` gave `operating_profit=0.28...`, not `0`, even though
`derive_fjj_from_autarky_cutoff` is algebraically exact given the RIGHT `A_jj` (verified by
hand: `firm.unconstrained_revenue` at `z=1` implied an `A_jj` inconsistent with the one fed
to `derive_fjj_from_autarky_cutoff`). Fixed with a small inner fixed point in
`fake_data.jl`'s `build_at`: guess `A[j,j]` → derive `f[j,j]` → solve GE → get new
`A_final[j,j]` → re-derive `f[j,j]` → repeat to convergence (weak coupling, converges in a
handful of iterations). After the fix, `firm.operating_profit` at `z=1` is `3.9e-15` for
every `gamma_prime_target` tried.

This bug ALSO explains why a naive Monte-Carlo evaluation of the focal link residual can
be badly wrong even when the closed-form population residual is exactly zero: at
`W=2,000,000`, before the fix the MC focal-link residual sat at `-0.36` (NOT shrinking with
`W` — the signature of a real bias, not sampling noise); after the fix it converges to
`~5.6e-7` at the same `W`.

### 13.3 Population-Pareto reconstruction (addendum, supersedes Sections 2/11/12 above)

`fake_data.jl` was rebuilt end-to-end per the addendum's "population Pareto fake data"
instructions. Construction order (CRITICAL sequencing rule: never compute a trade
share/flow before `(A,f,w)` are ALL final):

1. `tau`, `L` chosen exogenously.
2. Raw heterogeneous `A`, gravity-projected EXACTLY via minimum-L2
   `project_to_gravity_manifold` (NOT the single-cell `GravityPivot` used by
   `delta_star.jl`'s own outer coordinate system — the single-cell pivot was found LIVE to
   occasionally dump an entire gravity correction onto one cell, producing backwards
   export-selection or extreme cutoffs; L2 spreads it evenly, matching the ALREADY-
   established fix for exactly this failure mode).
3. Raw (pre-gravity) log-levels of the `D^2-1` free `f`-cells chosen with domestic cells
   systematically cheaper than export cells (avoids the same backwards-export-selection
   failure mode when domestic/export levels are drawn symmetrically).
4. `gamma_prime_target` is SOLVED (not chosen/seeded), by 1-D bisection (`Roots.jl`,
   canned) on the POPULATION-level focal free-entry link residual
   (`population_focal_link_residual`, closed form, no Monte Carlo) — a genuine equilibrium
   condition, not a free parameter. An arbitrarily chosen `gamma_prime_target` was found
   LIVE to leave this residual stuck around `-0.3` regardless of `W`, the signature of an
   uncalibrated parameter, not sampling noise.
5. Given `(A,f)` final, baseline wages `w` are a genuine GENERAL-EQUILIBRIUM output
   (`melitz_solve_wages_ge`), NOT a normalization: unlike the superseded closure's
   data-driven `melitz_solve_wages` (which solves against exogenous share DATA and has a
   genuine nominal wage-scale indeterminacy, resolved by a numeraire pick), this GE ties
   absolute wage levels to the REAL primitives (`f_od`/`A_od` are labor-value/productivity
   quantities) — there is NO free wage-scale normalization here. Verified live: forcing
   `w[j]=1` post-hoc broke the fixed point (factor-market residual jumped from `~1e-12` to
   `~1`). `melitz_solve_wages_ge` is the SAME damped-Jacobi income-redistribution iteration
   as `melitz_solve_wages`/the Ricardian model (not NLsolve/autodiff — tried first, found
   unnecessarily complex for what is, at heart, the same trivial fixed-point iteration),
   just recomputing `lambda`/`X` from the Melitz-Pareto closed form
   (`population_X`/`pareto_tail_power_mean`) at each iterate instead of taking it as fixed
   data. Requires `damping~0.1` (not the data-driven solver's `0.6` default) — the
   Melitz-Pareto share elasticity in wages is effectively `theta_star` (steeper than a
   plain CES gravity elasticity `sigma-1`), so the naive damped-Jacobi map is not a
   contraction at large damping (`damping=0.5` diverges/oscillates live).
6. Feasibility (`q>=1`, export-selection) and the well-conditioned-fixture criterion (min
   reference participation probability `>=0.01`) verified on the FINAL cutoff matrix;
   gravity, factor-market clearing, and the focal-link identity all verified to `<1e-8`.

`generate_fake_melitz_data`'s tuned defaults (`tau_offdiag_logrange`, `L_range`,
`logA_noise_sd`, `logf_domestic_mean`/`logf_export_mean`/`logf_noise_sd`) were found by a
grid search minimizing cutoff spread while keeping export-selection satisfied — a UNIFORM
level rescaling of `f` alone cannot fix a too-wide cutoff spread (verified: the max/min
cutoff ratio is invariant to uniform rescaling), so the noise/heterogeneity SCALE itself
had to be tightened, not just the level.

`solve_fstar`/`fstar_solver.jl` (the exact-sample-correction machinery) is ARCHIVED per the
addendum: kept in the repo as an optional debugging utility (one structural smoke test,
`test/melitz/runtests.jl`), no longer part of the active fixture-construction or
validation path.

### 13.4 Results at the corrected D=4/seed=2 benchmark

`gamma_prime_target` (SOLVED, not chosen) `= 1.0113`; `GT_j = 1-gamma_prime^(1/(sigma-1))
= -0.0075` (this particular random draw happens to have very small/near-zero implied
gains from trade for the target country — an economically valid but unremarkable outcome,
not tuned for a "large gains" narrative). Cutoff range `[1.055, 1.814]`; min reference
participation probability `0.0174` (`>=0.01` target met); min active-draw count at
`W=80,000` is `1,393` (worst cell), comfortably above the `>=500` well-conditioned-fixture
threshold.

**W-convergence (equal-weight raw residuals, addendum Section 3 — expect small, declining,
NOT exact-zero):**

| `W` | max trade-share residual | focal-link residual | min active count |
|---|---|---|---|
| 5,000 | 0.00173 | 0.00023 | 87 |
| 20,000 | 0.00065 | 0.000034 | 349 |
| 80,000 | 0.00022 | -0.000014 | 1,393 |
| 200,000 | 0.00011 | 0.0000087 | 3,483 |

Both decline with `W` as expected (roughly the `O(1/sqrt(W))` Monte-Carlo rate); contrast
the PRE-fix focal-link residual, which sat at `-0.31` to `-0.36` regardless of `W` (Section
13.2's bug signature, not sampling noise).

**Real KNITRO CC inner minimum-divergence loop (`run_melitz_inner_delta`, on the RAW
population fixture, no correction):**

| quantity | W=20,000 | W=80,000 |
|---|---|---|
| wall time | 7.4s | 0.4s |
| KNITRO status (`nStatus`) | **0 (optimal)** | **0 (optimal)** |
| `Delta(theta*)` | 1.39e-6 | 1.56e-7 |
| max `\|moment residual\|` under LFD | 3.0e-17 | 1.8e-13 |
| `lfd_ok` (new, stricter, Section 13.5) | true | true |
| min/max LFD weight | [4.97e-5, 5.11e-5] | [1.233e-5, 1.265e-5] |
| `residual_autarky_cutoff` (ex-post) | 3.9e-15 | 3.9e-15 |
| gravity residuals (ex-post) | machine precision | machine precision |
| `N'[j]` two-formula rel. diff | 2.2e-15 | 3.1e-13 |

**This directly replaces the previous `nStatus=-102`/`Delta=1e10` failure.** `Delta(theta*)`
is small, positive, and declining in `W`, exactly the addendum's expected pattern — not a
sentinel. The previous "coupled A-pivot/f-pivot cell" diagnosis (§11) is now understood to
have been chasing a symptom of Bugs 1-2 (an internally-inconsistent benchmark point), not
a genuine finite-support CC infeasibility.

### 13.5 LFD validity rule strengthened (addendum Section 4)

`melitz_recover_lfd`'s `lfd_ok` previously required only: dual `x` finite; raw LFD weights
finite and nonnegative with positive sum. This let a numerically degenerate reconstruction
through as `lfd_ok=true` even with `nStatus=-102` (the exact failure mode documented in the
retracted §12). Now requires ALL of: `nStatus==0` (checked BEFORE any normalization is
attempted — an unbounded/failed dual is never normalized and presented as an LFD); dual `x`
finite; raw LFD weights finite/nonnegative; pre-normalization sum close to `W`
(`|s/W-1|<1e-6`); every imposed moment holding under the recovered weights
(`max|residual|<1e-6`); and the PRIMAL divergence at the recovered weights
(`melitz_primal_divergence`, the exact convex-conjugate counterpart of `Psi!`/`dPsi!`,
ported from `sequential_gravity/run_profiled_production.jl`'s `divergence_of`) agreeing
with the dual objective `val` (`<1e-4`).

### 13.6 Iceberg-cost factor-market closure (addendum Section 6)

Already satisfied by construction, no change needed: `population_X`'s market-clearing
condition is the plain `w_o*L_o = sum_d X_od` (income=sales), with NO `1/tau_od`
tariff-revenue term anywhere in `melitz_solve_wages_ge`/`population_X`. `tau_od` enters only
through marginal cost (`w_o*tau_od/(A_od*z)`), consistent with an iceberg-cost
interpretation throughout.

### 13.7 What remains open / out of scope this pass

Not attempted this session (flagged for follow-up, none block the above):
- `W=200,000`/`800,000` KNITRO runs (only the closed-form equal-weight residuals were
  checked at these `W`, not the real inner solve — cheap to add, just more KNITRO wall
  time).
- Cross-repo gravity-equivalence tests against `production/fullA-exact`'s own gravity
  residual (main prompt Section 6) — not run this pass.
- A dedicated D=4 stress-test fixture with deliberately rare cells (kept as a SEPARATE
  fixture from the well-conditioned default, per main prompt Section 7's own guidance —
  not yet built).
- An outer `Delta*` search over structural parameters (explicitly out of scope per the
  addendum: the fixed-`theta*` inner problem is the milestone).

## 14. Gate A validation (2026-07-22/23): two real bugs found and fixed, then the full
## fixed-point system re-verified

Governing prompt: finish Gate A (fixed-point validation) before touching Gate B (the
outer `Delta*` search) or Gate C (gradient methods). This section reports Gate A only.
Gate B/C are **not started** this pass — they are explicitly gated behind Gate A in the
governing prompt, and this session's time went entirely into Gate A, including two
newly-found blocking bugs that were not on anyone's radar going in.

### 14.1 Bug 3: gains-from-trade formula omitted the wage ratio (main prompt Section A1)

Section 13.4 reported `GT_j = 1 - gamma_prime^(1/(sigma-1)) = -0.0075` and called this
"an economically valid but unremarkable... near-zero" result. It is neither: the formula
silently assumes `w_prime_j/w_j == 1`, which does NOT hold here — `w[target_country]` is a
genuine general-equilibrium output of `melitz_solve_wages_ge` (never renormalized; see
that function's own docstring, which documents LIVE that forcing `w[j]=1` post-hoc breaks
its fixed point), while `w_prime_j == 1` always (the autarky numeraire).

**Correct formula** (derived from `price_power_d ≡ P_d^{-(sigma-1)}` and baseline
`gamma_d==1 ⟹ P_d==1` in these units):

```
GT_j = 1 - (w_prime_j/w_j) * gamma_prime_j^(1/(sigma-1))
```

implemented as `melitz_gains_from_trade(p, cf)` (`equilibrium.jl`). Chose the "keep
unequal wages, use the wage-ratio formula everywhere" route the main prompt explicitly
offers as an alternative to a common-numeraire reconstruction of the whole fixture: a
full renormalization would have to re-derive invariance through
`melitz_solve_wages_ge`'s hard-won, previously-buggy GE fixed point
([[melitz-population-pareto-bugfix-2026-07-22]]), a materially higher-risk change than
fixing the welfare formula itself, and the wage-ratio formula is valid for ANY `w_j` so
nothing is lost.

**Verified live at the (old, seed=2) fixture**: `w[target]=1.0628`, `gamma_prime_target
=1.0113` →

| formula | value |
|---|---|
| naive (Section 13.4, WRONG) | `-0.0075` |
| `melitz_gains_from_trade` (correct) | `+0.0520` |
| `acr_gains_from_trade` (ACR/Chaney, independent) | `+0.0520` |
| `\|GT_model - GT_ACR\|` | `3.6e-14` |

The naive formula does not merely rescale the answer — it flips the **sign** of the
reported gains from trade at this benchmark.

### 14.2 ACR/Chaney cross-check restored (main prompt Section A2)

`acr_gains_from_trade(p, eq)` (`equilibrium.jl`) computes `lambda_jj =
X[j,j]/expenditure[j]`, `GT_ACR = 1 - lambda_jj^(1/theta_star)` — a REAL
(nominal-normalization-invariant) quantity that needs no wage-ratio correction. Verified
to agree with `melitz_gains_from_trade` to `~1e-14` (population-level, both are closed
forms) across every `(D, seed, W)` combination tried in Section 14.6's campaign below —
never merely "close," always within 2 orders of magnitude of machine epsilon.

Also verified (main prompt Section A2's second requirement): the recovered autarky
entrant mass satisfies `N_prime[j]/N[j] ≈ 1` (`N[j]≡1` by the universal baseline
normalization) **without being imposed anywhere in the construction** — this is a genuine,
nontrivial equilibrium property of the closure (not a bookkeeping artifact), confirmed to
converge toward 1 as `W` grows: `|N'_mc - 1|` is `2.6e-5` at `W=20,000` and `6.7e-8` at
`W=80,000` at the original diagnostic point, and `N'_mc ∈ [0.999998, 1.000003]` across
every `(W, seed)` combination in the Section 14.6 campaign.

### 14.3 Ex-post equilibrium residuals: tight tolerances, classified (main prompt Section A3)

Added an explicit, tightly-toleranced testset (`test/melitz/runtests.jl`, "Gate A3")
replacing the previous loose `N_prime_diff_rel < 1e-2` / `residual_autarky_cutoff < 1e-4`
gate. Every field of `MelitzEquilibriumCheck` is now asserted individually, classified by
what it actually tests:

| residual | classification | tolerance | typical value (W=80,000) |
|---|---|---|---|
| `residual_gamma_baseline` | IMPLIED (== `sum_o` of the D imposed trade-share moment residuals, verified algebraically) | `1e-8` | `4.9e-13` |
| `residual_free_entry_baseline`, `residual_free_entry_autarky` | DEFINITIONAL (`f_entry := E[Pi]/w`, so the residual is identically 0) | `1e-10` | `0.0` exactly |
| `\|f_entry_recovered[j] - f_entry_autarky_recovered\|` | IMPLIED by the focal free-entry LINK moment (this difference IS that moment) | `1e-8` | `1.4e-13` |
| `residual_market_clearing_autarky` | INDEPENDENT (ties labor-cost-side `N'` to revenue-side accounting) | `1e-8` | `3.7e-13` |
| `residual_gamma_autarky`, `N_prime_diff_rel` | INDEPENDENT (the KEY omitted moment: ties labor-cost-side `N'` to price-index/demand-side `N'`) | `1e-8` | `3.1e-13` / `3.1e-13` |
| `residual_autarky_cutoff` | DEFINITIONAL (`derive_fjj_from_autarky_cutoff`'s own construction) | `1e-10` | `3.9e-15` |
| `min_cutoff_minus_one`, `min_export_minus_domestic` | feasibility inequalities, pre-LFD | `>= -1e-6` | `>0` |
| `gravity_residual_A`, `gravity_residual_f` | exact by pivot construction | `1e-8` | `~1e-17` |

Every residual clears its tolerance by 3-5 orders of magnitude at every `(W, seed)` tried
— none is "bounded away from zero," and the two genuinely INDEPENDENT cross-checks
(market clearing vs. price index for `N'[j]`) are the most informative, per the main
prompt's own guidance.

### 14.4 Primal-dual diagnostics stored explicitly (main prompt Section A4)

`MelitzLFDResult` (`types.jl`) gained explicit fields: `primal_divergence`,
`dual_divergence` (`== Delta`), `primal_dual_gap`, `maximum_weighted_moment_residual`,
`probability_normalization_residual`, `kkt_opt_error`, `kkt_feas_error`. The last two are
KNITRO's own `KN_get_abs_opt_error`/`KN_get_abs_feas_error`, captured via a new
non-invasive global-Ref instrumentation pair (`INNER_LAST_OPT_ERR`/`INNER_LAST_FEAS_ERR`,
`cc_algo/inner_loop_functions.jl`) rather than changing `inner_loop`'s shared return
signature (used by every method in this repo, including `production/fullA-exact`).

`melitz_recover_lfd`'s `lfd_ok` gate replaced the flat `divergence_tol=1e-4` (documented
as larger than `Delta(theta_Fstar)` itself, hence unable to discriminate) with the
requested scale-aware test: `gap <= max(1e-10, 1e-6*max(1,|primal|,|dual|))`.

Verified live (W=80,000, seed=29): `primal_divergence = dual_divergence = 3.5146e-07`
(agree to `3.0e-19`), `kkt_opt_error = 4.6e-17`, `kkt_feas_error = 0.0` — both orders of
magnitude inside the new gate, at every `(W, seed)` tried.

### 14.5 Gravity equivalence vs. `production/fullA-exact` (main prompt Section A5) — BLOCKING BUG FOUND AND FIXED

This was the most consequential Gate A finding. `gravity_residuals`/
`gravity_coefficient_vector`/`GravityPivot`/`project_to_gravity_manifold` were all built
on `misc/doubleDiff.jl`'s `doubleDiff` — an "anchored" contrast that differences against a
FIXED reference row (1) and column (2), NOT a symmetric projection.
`production/fullA-exact`'s own canonical gravity moment
(`moments/newGravityMoment!.jl`'s `UoModel==1` branch — confirmed via `grep` to be the
UNIVERSAL setting in every production run config in this repo, `GravityMomentFirstApproach=0`
everywhere too) is instead built on `withinTransform` — the symmetric two-way
origin+destination fixed-effects "within" residual, which the repo's OWN code comment in
that file states reproduces the OLS-two-way-FE gravity coefficient EXACTLY (Frisch-Waugh-
Lovell), while explicitly noting **"the old cell double-difference did not."**

Verified live via three independent lines of evidence (main prompt Section A5's own
warning — do not assume equivalence just because each returns zero on its own data —
taken seriously):

1. **Coefficient-vector non-proportionality**: `sum(doubleDiff(tau).*doubleDiff(X))`'s
   per-cell unit-perturbation coefficients are NOT a constant multiple of
   `sum(withinTransform(tau).*withinTransform(X))`'s (ratio ranges from `-33.9` to
   `92.9` across cells on a random D=4 draw — not remotely proportional).
2. **Repo's own pre-existing `gravity_check.jl`** (re-run, not re-derived): over 2000
   random `(tau, A)` trials, `withinTransform` matches an explicit OLS-with-two-way-FE-
   dummies regression coefficient to `1.8e-15`; `doubleDiff` is off by up to `2.31`, and
   the two transforms' bilinear-form VALUES can have opposite SIGNS on the same data.
3. **`full_aod_diag/test_free_param_and_gravity.jl`** independently cross-validates
   `gravity_value` (`full_aod_diag/d4_exact/gravity_elimination.jl`, also
   `withinTransform`-based) against `newGravityMoment!` — the two agree, both being
   `withinTransform`-based, neither ever built on `doubleDiff`.

Both transforms DO share the same null space (both vanish exactly on any log-additive-FE
`A`/`f` — verified), which is why using `doubleDiff` never tripped a visible failure in
this module's own self-constructed tests. As a general restriction on genuine (non-
additive) bilateral structure, though, it is a different, non-canonical choice from what
`production/fullA-exact` actually enforces.

**Fixed**: `gravity_residuals`/`gravity_coefficient_vector` (`equilibrium.jl`) now use
`withinTransform`. A useful simplification falls out for free: `withinTransform` is
self-adjoint (an orthogonal two-way-FE projection), so `gravity_coefficient_vector`'s
`c` is now provably `== vec(withinTransform(tau))` exactly (added as an explicit test) —
`doubleDiff`'s `c` needed the more roundabout unit-perturbation derivation because it is
NOT self-adjoint.

**Consequence for the fixture (`fake_data.jl`)**: the coefficient vector's structure
changed qualitatively — under `withinTransform`, `|c|` is now systematically LARGEST on
DIAGONAL (domestic, `o==d`) cells (verified: every one of D=4 diagonal entries dominated
all 12 off-diagonal entries in a live check), whereas `doubleDiff`'s `c` had no such
preference. The min-L2 `project_to_gravity_manifold` (which weights the correction by
`c` itself) therefore started dumping most of the A/f-gravity correction onto domestic
costs, pushing them above their own export costs and breaking export-selection — 100% of
the first 60 seeds tried failed immediately after the transform switch, and retuning the
domestic/export noise means/gaps alone (even at extreme settings) did NOT fix it,
confirming the failure tracked the projection's implicit weighting, not fixture noise.
**Fix**: a new `project_to_gravity_manifold_weighted` (equilibrium.jl) down-weights
domestic cells (weight 1000 vs. 1 for export cells) in both the A and f projections,
routing the correction onto export costs, which have far more feasibility slack. With
this, roughly 9% of seeds are feasible (vs. ~0% unweighted); **seed=29 replaces seed=2**
as the new default (`generate_fake_melitz_data`'s `seed` keyword default and every
explicit `seed=2` call site in `test/melitz/runtests.jl` updated). This is a re-tuning of
an arbitrary fixture draw, not a new economic finding.

A regression test (`test/melitz/runtests.jl`, "Gate A5") pins both the coefficient-vector
self-adjointness and the `withinTransform`-matches/`doubleDiff`-doesn't OLS-FE comparison
so this cannot silently regress.

### 14.6 Additional fixed-point checks: W-convergence + 3-seed campaign (main prompt Section A6)

Full real-KNITRO inner solve + LFD recovery + ex-post check, at D=4/target=1, for the
corrected (`withinTransform`, seed=29) fixture:

**W-convergence (seed=29):**

| `W` | `Delta(theta*)` | max trade-share resid (equal wt) | primal-dual gap | KKT opt error | `N'_mc` | min active |
|---|---|---|---|---|---|---|
| 20,000 | `7.55e-6` | `1.27e-3` | `1.4e-18` | `1.2e-16` | `0.999998` | 395 |
| 80,000 | `3.51e-7` | `2.69e-4` | `3.0e-19` | `4.6e-17` | `1.000001` | 1558 |
| 200,000 | `1.40e-7` | `1.54e-4` | `1.8e-19` | `1.8e-17` | `0.999999` | 3893 |

`Delta(theta*)` and the equal-weight trade-share residual both decline monotonically with
`W` as expected (roughly the `O(1/sqrt(W))` Monte-Carlo rate); `nStatus=0` and
`lfd_ok=true` at every `W`; gravity residuals stay at machine precision (`~2e-17`)
throughout, as they must (gravity is `W`-independent by pivot construction).

**3-seed robustness check (W=20,000):**

| seed | `Delta(theta*)` | `nStatus` | `GT_model==GT_ACR` | `N'_mc` | min active | min ref. prob |
|---|---|---|---|---|---|---|
| 29 | `7.55e-6` | 0 | `0.0652` (`\|diff\|=1.9e-14`) | `0.999998` | 395 | 0.0194 |
| 49 | `1.54e-6` | 0 | `0.0895` (`\|diff\|=4.3e-14`) | `1.000003` | 792 | 0.0396 |
| 50 | `3.12e-6` | 0 | `0.1279` (`\|diff\|=6.6e-15`) | `1.000001` | 633 | 0.0317 |

All three independently-generated fixtures converge cleanly (`nStatus=0`, `lfd_ok=true`);
`GT_model`/`GT_ACR` agree to `~1e-14` at every seed (each seed implies a genuinely
different gains-from-trade level, as expected — no false convergence to a common value);
`N'[j]` sits within `3e-6` of 1 at every seed; no seed shows a residual "stuck" away from
zero. Per main prompt Section A6, monotonicity across seeds is NOT required and none is
claimed — only the absence of a structurally-bounded-away-from-zero residual, which holds.

`W=200,000`/`800,000` real-KNITRO runs (not just closed-form residuals) are now done (see
table above); `W=800,000` was not attempted (no evidence anything is currently wrong at
`W=200,000` that a further-`W` run would diagnose).

### 14.7 Known non-blocking regression: `solve_fstar`'s archived smoke test got much slower

`solve_fstar` (`fstar_solver.jl`, archived per Section 13.3, exercised only by one
structural smoke test) went from `~1m10s` to `~20m28s` wall time after the seed/transform
change. It still passes (produces a `MelitzFStarResult`, no crash) — its bisection-based
per-coordinate solve is presumably converging much more slowly against the new gravity
projection's coefficient structure. Since this utility is explicitly NOT part of the
active validation path, its performance was not investigated further this pass; flagged
for whoever next touches `fstar_solver.jl` or wants faster test-suite iteration.

### 14.8 Test suite status

All new Gate A assertions (Sections 14.1-14.4's GT/ACR/N', ex-post-residual, and primal-
dual-diagnostic checks) were added to `test/melitz/runtests.jl` and pass at the new
default fixture (D=4, seed=29). See the session's live run output for the exact updated
pass count (`test/melitz/runtests.jl`'s own summary lines are authoritative — this
document does not restate a specific number that could drift out of sync with the code).

### 14.9 What remains open before Gate B

Nothing found in Gate A blocks Gate B on economic grounds — every checked identity holds
at 3-5 orders of magnitude inside its tolerance, at every `W` and seed tried. Still open,
flagged for whoever picks up Gate B next:
- A dedicated rare-cell stress-test fixture (Section 13.7, still not built — optional,
  not blocking).
- `solve_fstar`'s slowdown (Section 14.7, archived utility only).
- Gate B (the actual `Delta_star = min_theta Delta(theta)` outer search) and Gate C
  (gradient-method comparison) are **NOT STARTED**. Per the governing prompt's own gate
  ordering, they should not begin until Gate A's report is reviewed/accepted.

## 15. 2026-07-22 session: outer-parameterization safety, direct F* solve, nested
## Delta-star outer search, gradient laboratory (Gate B/C, partial)

Governing session prompt (separate from, and following, the main/addendum prompts above):
freeze Gate A, make the outer parameterization safe, solve the finite-draw Delta-star
problem at D=4, build a gradient laboratory, and (time permitting) run one short
finite-delta smoke test. Checkpoint commit `287185b` freezes Gate A exactly as reported in
Section 14 before any of this section's changes. Given the session's realistic time
budget against this prompt's very large stated scope, work was explicitly PRIORITIZED and
SCOPE-CUT: Sections 0-4 below are complete with tests; the gradient laboratory (Section 5
of the governing prompt) is implemented and validated for 3 of its 5 requested methods
(A/B/C, not D/E) at a REDUCED test scale; the finite-delta smoke test (governing prompt
Section 8) was not reached. Each cut is flagged explicitly below, not silently dropped.

### 15.1 Off-diagonal, mutually-distinct outer gravity pivots (governing prompt Section 1.1)

Confirmed the diagnosed bug live: the UNRESTRICTED max-`|c|` A-pivot at the D=4/seed=29
benchmark lands on cell `(2,2)` — DIAGONAL, entangling gravity-feasibility with that
country's own cutoffs. Fixed: `build_gravity_pivots` (`delta_star.jl`) now restricts the
A-pivot to off-diagonal cells; a new `f_gravity_pivot_avoid_indices` (`equilibrium.jl`)
combines the existing domestic-avoid set with the A-pivot's own cell so the f-pivot is
simultaneously off-diagonal AND distinct from the A-pivot. At the benchmark: A-pivot
`(4,2)`, f-pivot `(3,1)` — both off-diagonal, distinct, neither `(1,1)`. A new
`pivot_conditioning_diagnostics` reports the (diagnostic-only) leverage comparison:
unrestricted pivot `0.915`, restricted (production) pivot `1.398`, orthonormal/null-space
reference `1.0` — the off-diagonal restriction costs a modest conditioning penalty (as
expected, since it is no longer free to choose literally the largest-`|c|` cell), not a
severe one. 15 new regression tests (`test/melitz/runtests.jl`, "Section 1.1") cover
`o!=d`, `A pivot != f pivot`, `reduce(expand(theta))==theta`, `expand(reduce(p))==p`, and
machine-precision gravity residuals at 4 random seeds plus the benchmark fixture.

### 15.2 Cutoff-safe displaced-point evaluator (governing prompt Section 1.2)

`ctx.cutoff` (read at every displaced outer point by the OLD `melitz_moments_adapter!`,
even though `melitz_moments!` itself never consumes `eq.cutoff`) renamed to
`ctx.benchmark_cutoff` and documented as reporting-only. New `melitz_outer_state(theta_free,
ctx; obj=nothing, evaluate_inner=false, warm_start=nothing, cold=false)`
(`delta_star.jl`) is now the single authoritative displaced-point state builder: expands
`theta_free`, recomputes the FULL baseline cutoff matrix fresh via the new
`melitz_baseline_cutoff` (`equilibrium.jl`), evaluates the Section 1.3 deterministic
cutoff constraints, and (optionally) runs the real inner CC solve on a caller-supplied
`obj`. `melitz_moments_adapter!` itself was also fixed to build its internal
`MelitzEquilibrium` with a freshly computed cutoff rather than `ctx.benchmark_cutoff`. 6
regression tests confirm the fresh cutoff matches the fixture's own benchmark cutoff AT
the benchmark point (`~1e-14`) and genuinely DIFFERS from the stale benchmark cutoff at a
displaced point.

### 15.3 Deterministic cutoff constraints + exact Jacobian (governing prompt Section 1.3)

`melitz_deterministic_cutoff_constraints` (`equilibrium.jl`) implements the minimal system
— `log zhat[o,o]>=0` (`D` domestic constraints) plus `log zhat[o,d]-log zhat[o,o]>=0`
(`D*(D-1)` export-selection constraints), `D^2` total, no redundant second system.
`melitz_cutoff_constraint_jacobian` (`delta_star.jl`) differentiates straight through
`expand_free_theta -> melitz_baseline_cutoff -> melitz_deterministic_cutoff_constraints`
with `ForwardDiff` — legitimate here (unlike `Delta(theta)`) because this call chain is
fully smooth (no participation/Boolean gate anywhere in it), automatically capturing the
A-pivot chain rule, the f-pivot chain rule, `f[j,j]`'s derivation, and `gamma_prime`'s
effect without hand-deriving each piece. Validated against central finite differences at
the benchmark point and a random perturbation: `max|J_ForwardDiff - J_FD| ~ 1.4e-10`
(domestic) / `~2.6e-10` (export), consistent with `h=1e-6` FD truncation error, i.e. exact
to essentially machine precision.

### 15.4 Authoritative evaluator + cache (governing prompt Section 2)

`MelitzDeltaEvalResult` (`types.jl`) bundles everything a caller needs from one fixed-
outer-point evaluation: `theta_free`, full `A`/`f`/`gamma_prime_j`/`f_jj`, the fresh
cutoff, the Section 1.3 constraint values and `min_slack`/`feasible`, the moment matrix
`G` (optional, `store_G`), the dual solution, LFD weights, `Delta`, primal/dual
divergence and gap, moment residuals, KKT diagnostics, solver status, the full ex-post
`MelitzEquilibriumCheck` (only when `verified`), and timings split into `state_time`
(cheap, no KNITRO) and `inner_time`. `evaluate_melitz_delta` (`delta_star.jl`) builds this
and `MelitzDeltaEvalCache` stores it keyed by exact `theta_free` value — but ONLY when
`result.verified = feasible && lfd_ok && nStatus==0`; a grossly infeasible/failed probe is
returned to the caller (so nothing hides a failure) but never cached, verified live: a
`theta0 .+ 3.0.*randn(...)` probe returned `nStatus=-400`/`verified=false` and left the
cache empty. At the benchmark point, `evaluate_melitz_delta` reproduces `Delta=7.5545e-6`
at `W=20,000` EXACTLY matching Section 14.6's own table — confirming the cutoff-safety
refactor changes nothing at the point where the old and new cutoff happen to coincide (the
benchmark itself), only at genuinely displaced points. 17 new tests pass.

### 15.5 Direct F* feasibility solve (governing prompt Section 3)

`solve_fstar_direct` (`fstar_direct.jl`, NEW file, replaces the archived `solve_fstar` for
the active minimal system) minimizes `0.5*||S*m_Fstar(theta)||^2 + rho/2*||theta -
theta_population||^2` subject to the Section 1.3 cutoff inequalities (exterior quadratic
penalty), via `Optim.LBFGS` with a DELIBERATELY WIDE-bandwidth (`h=1e-3`) central-
difference gradient — matching the archived `solve_fstar`'s own documented finding that a
naive small-`h`/autodiff gradient is locally blind to (or badly noise-dominated by) the
`O(1/W)` participation-jump in the raw moment mean. Real-KNITRO run at `D=4`,
`seed=29`, `W=20,000`, cold-verified inner solve at each result:

| `rho` | max\|m_initial\| | max\|m_final\| | \|\|theta_final-theta_pop\|\| | cold `Delta` | `nStatus` |
|---|---|---|---|---|---|
| `1e-6` | `1.265e-3` | `3.05e-4` | `3.69e-3` | `1.038e-6` | `0` |
| `1e-4` | `1.265e-3` | `2.65e-4` | `4.09e-3` | `2.546e-6` | `0` |
| `1e-2` | `1.265e-3` | `4.01e-4` | `2.62e-3` | `2.908e-6` | `0` |

Every `rho` produces the SAME qualitative result — max\|m_final\| within a factor of ~1.5
of each other across 4 orders of magnitude of `rho`, all feasible (`min_slack~0.0166` at
every `rho`, essentially unchanged from the starting `~0.0165`), all `nStatus=0`/verified
— confirming the moments are, as expected for this deliberately underidentified system
(17 moments, 30 free coordinates), insensitive to the regularizer's exact weight. `Optim`
did NOT fully converge within its 60-iteration/90s-per-run budget (`Optim.converged=false`
at every `rho`, `~11` iterations completed) — the max\|m_final\| improvement (`~4x` from
`1.265e-3` to `~3e-4`) and the cold `Delta` improvement (`7.55e-6` at `theta_population`
down to `~1-3e-6`) are genuine but PARTIAL, not the governing prompt's stated `<=1e-9`
acceptance target. Given every result is nonetheless cold-verified `nStatus=0`/`verified`
with `Delta` already an order of magnitude below `theta_population`'s own `Delta`, and
divergence is nonnegative, this is read as strong (not yet fully converged) evidence
toward `Delta_star approx 0`, cross-confirmed independently by Section 15.6's outer search.
No claim of unique `A`/`f` recovery is made — `theta_distance_from_population` (`~0.003-
0.004` in the 30-dimensional free-coordinate norm) is reported, not treated as informative
about a "true" point, consistent with the underidentification.

### 15.6 Nested Delta-star outer search (governing prompt Section 4)

`solve_melitz_delta_star_outer` (`outer_solve.jl`, NEW file): `minimize_theta Delta(theta)`
s.t. the Section 1.3 cutoff inequalities, using Method B (Section 15.7) for the gradient
(re-solved fresh at every `g!` call to refresh the fixed base dual) and the same exterior-
penalty technique as Section 15.5, `Optim.LBFGS`, bounded `time_limit`. Small-`W` (5,000)
smoke test: `Delta` `2.667e-5 -> 2.238e-5` in 4 LBFGS iterations (437 total inner solves,
80s, not converged), COLD-verified `nStatus=0`, `Delta_star <= Delta(theta_population)`
confirmed.

**`W=20,000` run (the reported result):** 9 LBFGS iterations, 477 total inner solves,
315.3s wall. `Delta_init` (population-Pareto) `= 7.5545e-6 -> Delta_final_warm =
6.6127e-6`. `Optim.converged = false` (hit the iteration/time budget, not a stationarity
gate) -- again a GENUINE, cold-verified, PARTIAL improvement, not full convergence.
COLD-verified incumbent: `nStatus=0`, `verified=true`, `feasible=true`, `Delta=6.6127e-6`
-- independently re-verified a SECOND time (a fresh `evaluate_melitz_delta(...;
cold=true)` call on the returned `theta_final`) with an IDENTICAL `Delta`, confirming no
warm-start-dependent artifact. `Delta_star <= Delta(theta_population)`: `6.6127e-6 <=
7.5545e-6` -- **CONFIRMED**, satisfying the governing prompt's own required inequality.

Full Gate A ex-post equilibrium check re-run at the incumbent (`check_profiled_melitz_
equilibrium` under the incumbent's own recovered LFD, not reference weights):

| residual | value |
|---|---|
| `max\|residual_gamma_baseline\|` | `3.81e-14` |
| `residual_market_clearing_autarky` | `-2.22e-15` |
| `N_prime_diff_rel` | `1.11e-15` |
| `residual_autarky_cutoff` | `1.11e-16` |
| `min_cutoff_minus_one` (feasibility) | `0.0698` (comfortably `>0`) |
| `gravity_residual_A` / `gravity_residual_f` | `-3.25e-17` / `1.91e-17` |

Every residual is at or near machine precision, matching Gate A's own tolerances exactly
-- the outer search did not degrade any of the fixed-point identities Gate A validated.

### 15.7 Gradient laboratory (governing prompt Section 5/6) — Methods A/B/C only, reduced scale

Implemented (`gradient_lab.jl`, NEW file): Method A (fully reoptimized central finite
difference, cold-restarted at each displaced point — the expensive reference), Method B
(fixed-dual finite-bandwidth secant `[L_fix(theta+hv;x_base)-L_fix(theta-hv;x_base)]/2h`,
`melitz_fixed_dual_criterion` reusing `PsiObjectiveBundleDelta`'s own functor at a FIXED
dual), and Method C (`ForwardDiff` through a dedicated type-generic fixed-active-set
scalar evaluator, `melitz_fixed_active_set_scalar` — built FRESH rather than reusing
`obj`'s preallocated `Float64` `H` buffer, exactly the Dual-incompatibility this session's
prompt warned about). Methods D (hand-derived analytic branch derivative) and E (smooth
surrogate) were NOT implemented this session — flagged as the clearest follow-up item, not
silently dropped (Method D's formulas are given explicitly in the governing prompt and are
mechanical to implement given Method C's fixed-active-set machinery already exists to
cross-validate against).

Zero-switch validation (`W=5,000`, benchmark point, random tangent direction, `h=1e-6`
confirmed 0 switches both directions): Method B `0.0032989` vs Method C (exact)
`0.0032760` — agree to `~0.7%`, confirming the fixed-dual-criterion construction is
internally consistent (Section 6 core question 1: yes, methods agree when the active set
is unchanged). Method A at `h=1e-4` (same direction) gave a visibly different value
(`0.000859`) — expected, since Method A re-optimizes the dual at each displaced point
(capturing genuine curvature/re-optimization effects Method B/C's fixed-`x_base`
construction does not), and `h=1e-4` is two orders of magnitude coarser than the zero-
switch confirmation bandwidth.

**Reduced battery (`W=20,000`, real KNITRO, 60 probes total):** 2 points (population-
Pareto, `Delta=7.55e-6`; a `scale=0.01` random perturbation with `Delta=1.16e-3`,
matching the governing prompt's requested "`Delta` around `1e-3`" point) x 6 directions
(`gamma`, `ordinary_A`, `ordinary_f`, `random_tangent`, `f_high_switch`,
`A_pivot_sensitive`) x 5 bandwidths (`1e-6` to `1e-2`), Method A run on a `{1e-4,1e-3}`
subset only (2 cold KNITRO solves/probe) — a deliberate scale-down from the governing
prompt's full `30 coordinates x 11 bandwidths x >=5 points` grid (infeasible in this
session's time budget). Full CSV in the session's pushed output.

**Known limitation in this run**: `f_high_switch` was implemented as the SAME coordinate
as `ordinary_f` (`theta_free` index `2+D^2-1`, the first free f-cell) rather than a
distinct high-switch-inducing f direction — confirmed by their IDENTICAL results at every
bandwidth in the CSV. Not corrected mid-run (would have cost another full pass); flagged
honestly rather than silently presented as two independent directions. A genuinely
separate high-switch f direction (e.g. a large-magnitude perturbation concentrated on a
near-marginal cutoff cell) is a cheap fix for the next session.

**Answers to the Section 6 core questions, from this reduced battery:**

1. *Do analytic/ForwardDiff (Method C) and the fixed-dual secant (Method B) agree when
   the activity set is unchanged?* YES at confirmed-zero-switch bandwidths: e.g.
   `A_pivot_sensitive` at `h=1e-6` (0 switches both sides), `B=-0.000400272` vs
   `C=-0.000400264` — agree to `0.002%`. `random_tangent`/`ordinary_A` at `h=1e-6` agree
   to `0.7%`/`4.5%`. The `gamma` direction is the interesting counterexample: even at
   `h=1e-6` it already has 1 switch on the minus side (`gamma` moves the autarky price
   power, which shifts participation broadly), and `B`/`C` diverge sharply there
   (`0.00586` vs `0.000987`) — exactly the switch-driven disagreement the construction
   predicts, not a bug.

2. *Do branch derivatives agree with very-small-`h` reoptimized (Method A) differences
   for `A` coordinates?* **NOT ANSERED this session** — `bandwidths_for_A` was
   `{1e-4,1e-3}` only (cost-driven), and even `h=1e-4` already shows 7-8 switches for
   `ordinary_A`, so no genuinely zero-switch Method A data point was collected. At
   `h=1e-4` Method A and Method C even disagree in SIGN for `ordinary_A`
   (`A=+0.00620` vs `C=-0.000617`) — consistent with Method C simply missing the
   (already-present) switches' contribution, but not a clean answer to the question as
   posed. Flagged for the next session: run Method A at `h<=1e-6` for a handful of `A`
   coordinates specifically.

3. *Do branch derivatives fail in `f` directions as expected?* YES, cleanly. At
   confirmed-zero-switch bandwidths (`h<=1e-5`), Method C reports EXACTLY `0.0` for
   `ordinary_f` — matching the given closed-form prediction `d(trade share)/d log f_od =
   0` conditional on fixed activity precisely, not merely approximately. Method B also
   reports exactly `0.0` there (consistent: at zero switches its finite difference of the
   same fixed-active-set criterion has nothing to differentiate). Once switches appear
   (`h=1e-3`, 6-7 switches), Method B jumps to a genuinely nonzero `-0.00122` — confirming
   "for `f` coordinates, essentially the entire trade-share response comes from firms
   crossing the cutoff."

4. *Which fixed-dual bandwidth best predicts independently reoptimized (Method A)
   changes?* The clearest pattern in this battery is not about bandwidth but about
   POINT: at the population-Pareto point (`Delta approx 0`), Method B tracks Method A
   poorly (`ordinary_A` `h=1e-4`: `A=0.00620` vs `B=0.00519`, 16% off; `A_pivot_sensitive`
   `h=1e-3`: `A=-0.0112` vs `B=-0.00186`, off by `6x`). At the `perturbed` point
   (`Delta=1.16e-3`, away from near-degeneracy), Method B tracks Method A remarkably
   closely at BOTH tested bandwidths: `ordinary_A` `h=1e-4`: `A=-0.3488` vs `B=-0.3488`
   (`<0.01%`); `gamma` `h=1e-4`: `A=0.2608` vs `B=0.2613` (`0.17%`); `random_tangent`
   `h=1e-4`: `A=-0.2140` vs `B=-0.2107` (`1.5%`). Plausible mechanism (not independently
   confirmed this session): near `Delta approx 0` the optimal dual is itself close to a
   degenerate/boundary configuration, making the "hold `x` fixed" envelope approximation
   less stable than away from degeneracy. If this holds up, it is a genuinely useful,
   non-obvious operational finding for a future finite-`delta` campaign (where `delta>0`
   keeps the search away from the `Delta=0` boundary).

5. *Is there a stable bandwidth across points, or should `h` be coordinate-scaled?*
   Switch counts at matched `h` are broadly similar across `gamma`/`ordinary_A`/
   `random_tangent`/`A_pivot_sensitive` (`~0` at `h<=1e-5`, `~6-10` at `h=1e-4`, `~65-90`
   at `h=1e-3`, `~600-900` at `h=1e-2`) but MUCH milder for the `f` direction tested
   (`ordinary_f`: `0`/`0`/`7`/`88` at the same four bandwidths) — consistent with `f`
   affecting only its own cell's participation margin directly (the given formula) while
   `A`/`gamma` affect prices, and hence participation, more broadly. A mildly LARGER safe
   bandwidth for `f`-only coordinates than for `A`/`gamma` coordinates is weakly supported,
   not dramatically.

6. *Does a hybrid gradient (branch derivative for `A`, bandwidth secant for `f`) work
   better?* Qualitatively well-motivated by this battery, not exhaustively tested:
   Method C gives a well-defined, informative nonzero baseline for `A` directions
   (further refined by switches Method B captures); for `f` directions Method C gives
   EXACTLY zero (correctly reflecting no smooth component) and essentially ALL the signal
   comes from switches, i.e. from Method B. A production hybrid (Method-D-once-built for
   `A`, Method B for `f`/cutoff-sensitive coordinates) is a reasonable next step, not
   validated end-to-end this session.

### 15.8 Not reached: finite-delta smoke test (governing prompt Section 8)

Not attempted this session — Sections 15.1-15.7 consumed the available time budget. No
infrastructure for the upper/lower gains-from-trade programs (`minimize`/`maximize log
gamma_prime` s.t. `Delta(theta)<=delta`) was built. Flagged for the next session, gated
(per the governing prompt's own ordering) behind a more complete gradient laboratory
(Method D at minimum) so the finite-delta programs have a trustworthy gradient to use.

### 15.9 What remains open

- Methods D/E of the gradient laboratory (Section 15.7).
- The full-scale gradient battery (30 coordinates, 11 bandwidths) — only a reduced subset
  run this session.
- Full convergence of both the direct F* solve (governing prompt's `<=1e-9` target) and
  the nested outer search (`Optim.converged=false` in both W=5,000 and W=20,000 runs so
  far) — both show genuine, cold-verified, partial improvement, not full convergence,
  within this session's bounded iteration/time budgets.
- The finite-delta smoke test (Section 15.8).
- A KNITRO-native outer solve (matching `production/fullA-exact`'s own nested-KNITRO
  pattern via `ccOuter.jl`/`outer_loop_functions.jl`) was NOT attempted — this session's
  outer solve uses `Optim.LBFGS` with an exterior penalty instead, adequate for an
  infrastructure smoke test but not wired into this repo's own production outer-loop
  machinery.

## 16. 2026-07-23: governing conceptual correction — Section 15.6 was never an economic
## result

Section 15.6 above (`solve_melitz_delta_star_outer`, `Delta_init -> Delta_final_warm`,
"`Delta_star <= Delta(theta_population)`") is kept verbatim as a historical record of what
actually ran, but its framing was wrong and must not be cited as an economic finding.
`minimize_theta Delta(theta)` over every outer coordinate (`g` AND every free `A`/`f`
nuisance parameter simultaneously) is not a Christensen–Connault estimand: the governing
correction's outer problem extremizes the counterfactual `g = log gamma_prime[j]` subject
to a divergence BUDGET `Delta(g,eta) <= delta`, i.e. it optimizes only the ONE coordinate
that has direct economic content, treating the rest (`eta`) as nuisance parameters whose
role is to make the budget constraint as easy as possible to satisfy — not to jointly
shrink `Delta` for its own sake. An outer point is not "better" merely for having smaller
`Delta` while `g` sits near its benchmark value.

Consequently:

- `solve_melitz_delta_star_outer` is renamed `run_minimum_divergence_outer_smoke_test`
  (`outer_solve.jl`); its result struct is renamed `MinimumDivergenceSmokeTestResult`. Both
  are documented as an INFRASTRUCTURE REGRESSION TEST ONLY — confirming the nested
  outer-search machinery (Method B gradient, exterior-penalty cutoff constraints, cold
  verification) runs end to end without crashing. Do not spend further effort tightening
  its convergence; it is not a paper estimand.
- `solve_fstar_direct` (`fstar_direct.jl`, Section 3 above) is similarly a numerical
  feasibility diagnostic (does a nearby gravity-feasible point exist at which equal
  reference weights are ~feasible), not an optimization the paper cares about the minimizer
  of. Its docstrings' former `Delta_star approx 0` phrasing is corrected to `Delta(theta)
  approx 0` (a statement about ONE theta, not a minimum).
- Going forward in this document and the codebase: `Delta(theta)` (or
  `minimum_divergence_at_theta`) denotes the fixed-point CC inner minimum divergence at a
  COMPLETE outer point; `delta_profile(g) = min_eta Delta(g,eta)` denotes the nuisance-only
  profile at a FIXED counterfactual `g` (Section 5 of the finite-delta campaign, below);
  "upper/lower finite-delta bound" denotes the actual target programs (`min`/`max g` s.t.
  `Delta(g,eta)<=delta`). `Delta_star` is retired as a name in this codebase.

See the finite-delta campaign sections below (this document's continuation, 2026-07-23
session) for the actual upper/lower gains-from-trade programs.

## 17. 2026-07-23 session: gradient reconciliation (Section 1) + the direct finite-delta
## KNITRO outer NLP (Section 3) — built, bug-fixed, and stress-tested; Section 4's
## campaign did not yield a verified incumbent this session

Governing session prompt: fix the terminology (Section 16 above), resolve the Method B/C
zero-switch gradient discrepancy flagged as a blocking task, then build and run the actual
finite-delta upper/lower gains-from-trade programs via a real KNITRO NLP with explicit
nonlinear constraints, reusing the Ricardian model's own `PsiObjectiveBundleImplicit`/
`outer_loop` machinery rather than inventing a new bundle type. Given this session's time
budget against the full scope (Sections 1 through 8), work was explicitly prioritized:
Sections 1-3 are complete, tested, and committed; Section 4 was attempted at real scale
(W=20,000) and produced a genuine, informative negative result rather than a converged
incumbent; Sections 5-7 were not reached. Each cut is flagged explicitly below.

### 17.A Conceptual correction (Section 8.A)

No further global minimization of `Delta` over every outer coordinate was pursued this
session (Section 16 already retired that framing). The target implemented in Section 3/4
below is exactly the governing correction's own program: `minimize`/`maximize g =
log gamma_prime[target]` subject to `Delta(theta) <= delta` and the deterministic cutoff
inequalities — a single-coordinate objective with a divergence BUDGET constraint, not an
unconstrained minimum-divergence search. `run_minimum_divergence_outer_smoke_test`
(Section 16) remains a software regression test only and was not touched further this
session beyond the rename already applied.

### 17.B Gradient reconciliation (Section 1, main prompt Section B)

**Root cause of the reported zero-switch B/C discrepancy, found and fixed
(`src/melitz/gradient_lab.jl`, commit `6702e05`):** the prior Gate B session's Method C
(`melitz_fixed_active_set_scalar`) held participation fixed via a precomputed
`active_mask` array (`base_active_mask`) whose `(j,j)` AUTARKY slot incorrectly REUSED the
BASELINE `(j,j)` domestic participation decision (computed with `price_power=1`) as a
proxy for the autarky decision (which needs `price_power=gamma_prime_j`) — two
economically different gates (different implied cutoff, since price_power/wage/
expenditure/tau all differ between the baseline-domestic and autarky evaluations of cell
`(j,j)`). Method B was already correct (it called the real, type-generic `moments!`, which
has always used the right autarky formula). The bug meant `count_switches`'s own
diagnostic never independently checked the autarky decision, so a probe could be reported
as "zero switches" while the (never-checked) autarky participation had, in fact, flipped —
exactly the `gamma`-direction counterexample the prior session's own CSV had documented
(16% B/C disagreement near `Delta≈0`, at the time misattributed to "near-degeneracy",
docs Section 15.7) but never diagnosed.

**Fix**: `fixed_dual_scalar` (later split into `fixed_active_set_moments` +
`dual_scalar_at_fixed_G`, commit `9aab8c2`) is now the ONE authoritative, type-generic
implementation both Method B (finite difference) and Method C (`ForwardDiff.derivative`)
call — recomputing participation fresh, per cell, with the CORRECT autarky formula,
whether `theta_free` is `Float64`- or `Dual`-typed. No separate duplicate implementation
of the fixed-dual criterion is maintained.

**Verification (real KNITRO, D=4/seed=29, W=20,000 unless noted):**

- Zero-switch bandwidth sequence (`h` from `1e-8` to `1e-5`, 4 directions incl. `gamma`,
  the prior session's own counterexample direction): Methods B and C now agree to
  `~1e-8`–`1e-14` relative error at every confirmed-zero-switch probe (both baseline AND
  autarky confirmed unchanged) — fully resolving the reported discrepancy, including at
  the exact `gamma` direction that previously diverged.
- Per-moment-column directional-derivative audit (`fixed_active_set_moments`,
  ForwardDiff vs. central FD): at confirmed zero-switch probes, max absolute error across
  both trade-share and the focal-link columns is `~1e-8`–`1e-9` (FD-truncation-level, not
  a defect); at a probe with one genuine switch (`gamma`, `h=1e-6`), the naive per-column
  FD explodes (`~3.9e5` on one row) exactly as expected for a real participation flip —
  confirming the diagnosis, not a new bug.
- Reduced derivative battery (population point plus two random perturbations at
  `Delta≈3.5e-3` and `Delta≈1.85e-2`; 6 directions incl. the now-fixed `f_high_switch`;
  bandwidths `1e-6` to `1e-3`): confirmed-zero-switch agreement is uniformly tight
  (`~1e-10`–`1e-14` relative error) across ALL three points and all directions, resolving
  the prior session's own tentative "B tracks A poorly near Delta≈0" finding — that WAS
  the bug, not a genuine near-degeneracy artifact. Method A (fully reoptimized) tracks
  Method B closely once switches are present and the point is away from `Delta≈0` (e.g.
  `ordinary_A` at `h=1e-4`: `B=-0.605` vs `A=-0.608`, ~1% apart), consistent with the
  prior session's own hypothesis, now confirmed cleanly on the corrected machinery.

**Method D** (`method_d_hand_derived`, an independently-coded hand-derived closed-form
cross-check, chaining the given analytic per-cell partials through
`ForwardDiff.derivative` on `expand_theta_econ_vector` ALONE — not the full
fixed-dual-scalar chain Method C differentiates) agrees with Method C at every
confirmed-zero-switch probe tried, to machine precision — an independent validation of
both the gravity-pivot chain rule (already separately validated, Section 1.3) and the
firm-level revenue/profit algebra.

**`f_high_switch` fixed** (`f_high_switch_direction`): the prior session's version was an
accidental duplicate of `ordinary_f`. The new version empirically selects, from a base
active-set snapshot, the free-`f` coordinate producing the most participation switches
(baseline + autarky) at a probe bandwidth — confirmed to produce 5-8x more switches than
`ordinary_f` at matched bandwidths on real data.

**Not reached** (flagged, not silently dropped): the full `30 coordinates x 11 bandwidths
x >=5 points` grid (Section 2's own stated scale) — a REDUCED, representative battery was
run instead (3 points, 6 directions, 4 bandwidths), matching the prior session's own
scope-cut precedent; the reduced result was clean and consistent enough that the full grid
is not expected to change the conclusion, but was not run to confirm this.

### 17.C The direct finite-delta constrained outer problem (Section 3, main prompt
### Section 3)

**Built** (`src/melitz/finite_delta_outer.jl`, commit `53e6ca2`): reuses
`PsiObjectiveBundleImplicit` and the `cc_algo` outer-loop calling convention
(`KN_new`/`KN_add_vars`/`KN_add_eval_callback`/`KN_set_cb_grad`/`KN_solve`/
`KN_get_solution`) unmodified, per explicit instruction — Melitz supplies its own
`moments!` (`melitz_moments_adapter_outer!`, `K := theta_free[1]` constant across draws,
which is what makes this bundle "Implicit" in this codebase's own vocabulary, exactly
analogous to the Ricardian GT counterfactual) and `moments_jacobian!` (Methods B or D from
Section 1, NOT naive ForwardDiff through the hard participation gate — the whole reason
Section 1's gradient laboratory exists) into the SAME generic struct/functor the Ricardian
model's `ccOuter.jl` (`counterType==1` branch) uses. The deterministic cutoff inequalities
(Section 1.3) have no Ricardian analogue (that model has no gravity-pivot/cutoff-
feasibility system) and are genuinely new: added as extra rows in ONE combined KNITRO
callback (see the crash finding below for why one, not two, callback contexts).

**Two real bugs found via direct empirical testing (not just reading the code), both
documented in the file itself:**

1. **A vacuous constraint bound.** The Ricardian model's own `outer_loop_constraints!`
   (`cc_algo/outer_loop_functions.jl`, and the independent `PsiObjectiveBundleImplicitMethodB`
   variant in `sequential_gravity/`) sets `KN_set_con_upbnd(cIndices[1], 1e10*obj.δ)`
   against the functor's `constr[1] = -1e10*f`. Algebraically this reduces to
   `f >= -delta`, which is ALWAYS true since `f = Delta(theta) >= 0` by convex duality —
   vacuous for a `Delta(theta) <= delta` budget. Confirmed empirically, not just
   algebraically: a smoke run using that exact bound let `theta` drift for 60 real outer
   iterations with the reported feasibility error pinned at `0.000e+00` throughout
   regardless of where `theta` went; the terminal point's `Delta` only happened to satisfy
   the intended budget by chance (the search never moved far in that particular run).
   **Fixed**: the correct bound is a LOWER bound, `constr[1] >= -1e10*delta` (i.e.
   `Delta(theta) <= delta`) — set directly in this file's own driver (NOT a change to the
   shared `cc_algo/outer_loop_functions.jl`, which other models still rely on as-is).
   Re-verified empirically: with the corrected bound, a tight `delta=1e-3` test pushed the
   search hard enough to reach a numerically pathological inner-dual region
   (`nStatus=-102`), which the verified-success gate correctly refused to report as a
   result — i.e. the corrected constraint is genuinely binding, unlike the original.

2. **A reproducible KNITRO callback crash from two separate callback contexts.**
   Registering the divergence-budget block (reusing the existing cc_algo callbacks
   directly) and the new cutoff-constraint block as TWO separate
   `KN_add_eval_callback` contexts on the same outer problem reproducibly crashed KNITRO
   (`-500`, "could not evaluate objective or constraints") at the very first evaluation —
   even though EACH block, registered ALONE on its own KNITRO problem, ran flawlessly (the
   delta-only block completed 60 real outer iterations; the cutoff-only block converged in
   6). Direct calls to both new callback functions (bypassing KNITRO entirely) also worked
   without error, and `par_concurrent_evals` (the cause of an unrelated, previously-
   documented full-A nested-solve hang) was ruled out as the cause here — removing it did
   not fix this crash. An initial hypothesis (a second, independent nested inner KNITRO
   solve running inside the cutoff callback, on top of the delta callback's own) was
   tested directly and ruled out: removing that second nested solve left an identical
   crash. The exact root cause was not pinned down further given the time this would take
   to isolate inside KNITRO's own C library; **fixed** by merging into ONE combined
   callback context (`melitz_combined_callback_F!`/`_G!`) covering all `1+D^2`
   constraints, which sidesteps the issue entirely and is also simply less new code
   (verified via a real KNITRO run: 60 iterations, no crash, `nStatus=-400`/"iteration
   limit, current point feasible" — a normal non-convergence outcome, not an error).

**Regression test added** (`test/melitz/runtests.jl`, "Section 3: solve_melitz_finite_delta_bound"):
a short, loose run (small `W`, `theta_box=0.5`) confirming the solver runs end to end
without the `-500` crash and that any cold-verified incumbent found respects the delta
budget. Not a converged economic result — that is Section 4's job. 307/307 Melitz tests
pass with this section included.

### 17.D Primary finite-delta campaign (Section 4) — attempted, no verified incumbent
### found this session

D=4, W=20,000, seed=29, population-Pareto start (`Delta_pop = 7.55e-6`, `gamma_prime =
0.9585`, `GT = 0.0652`, matching `GT_ACR` to `~1e-14`), Backend B (`h=1e-4`), `delta=1e-3`
(the first continuation checkpoint).

**Attempt 1** (single KNITRO solve per direction, `maxit=25`, `theta_box=10` then `2.0`):
both directions ran to completion without crashing (`~470-535s` wall each) but drifted far
enough that the terminal point was numerically pathological (`nStatus=-102`/`-410`,
`Delta` on the order of `1e16`–garbage) — correctly identified as UNVERIFIED by the
verified-success gate (`cold_verified_incumbent = nothing` reported honestly, not a false
positive).

**Attempt 2** (genuine short-burst continuation: `theta_box=0.15`, `maxit=5` per burst,
re-launching from the previous terminal point only if it cold-verified as feasible,
aborting a direction's continuation otherwise): revealed a more informative and more
concerning finding. The UPPER direction's first burst reported `nStatus=0` ("locally
optimal solution found"), feasibility error `0.000e+00` at every one of its 6 iterations,
and looked like a clean, converged, feasible result (`gamma=0.825`, `GT=0.154`) — but an
INDEPENDENT cold reverification at that exact terminal `theta` (a fresh inner solve, no
warm start) gives `Delta = 2.12`, roughly 2000x the `delta=1e-3` budget. The LOWER
direction's first burst reached `nStatus=-102` (`Delta` again garbage on cold
reverification). Both were correctly rejected; the continuation logic stopped rather than
building further bursts on an unverified point (per its own design, matching Section 3.4's
"never replace a verified incumbent with an unverified terminal point").

**This is read as a genuine, informative negative result, not an infrastructure failure**:
the SAME discipline this session's own design insists on (never trust a live/warm KNITRO
trajectory's own reported status; always independently cold-verify) is exactly what caught
this. The finding itself — that KNITRO's own internal constraint tracking during a
trajectory can report `nStatus=0`/feasibility error `0.000` at a point that, independently
reverified from a cold start, is nowhere near feasible — is consistent with, and a fresh
concrete instance of, this same repository's own previously-documented risk class around
inner-solve status reliability (`[[recover-lfd-nstatus-bug-fix]]`: "silently accepted
UNBOUNDED inner KNITRO dual solves (`isfinite(x)` but wrong `nStatus`)", fixed repo-wide in
a prior session for a different call path). The exact mechanism here (whether it is a
genuine non-uniqueness/numerical-conditioning issue in the inner CC dual solve at this
theta, an issue specific to the `Float64`, all-zero-initialized cold start `inner_loop_internal`
uses by default here, or something else) was not diagnosed further this session given the
wall-clock cost of each real-KNITRO attempt (`200`–`1000`+ seconds each).

**No verified incumbent was found for either direction at `delta=1e-3` this session.**
`delta=1e-2`/`1e-1` were not attempted (the governing prompt's own continuation ordering
starts at the tightest budget and only widens if that succeeds comfortably — it did not).

### 17.E Backend comparison (Section 4.1/8.D)

Only Backend B was exercised in real KNITRO campaigns this session (Backend R does not
fit the `moments_jacobian!` hook at all — it requires full inner reoptimization at
displaced points, a structurally different algorithm, not a plug-in gradient function;
Backend D's `moments_jacobian!` analogue, `make_melitz_moments_jacobian_d`, was
implemented in `finite_delta_outer.jl` but not run in a real campaign this session; Backend
H was not implemented for the KNITRO-native path). No cold-verified outer progress exists
yet to compare backends against.

### 17.F Sections 5-7: not reached

The profiled inverse (`delta_profile(g)`, Section 5), local verification/polling of
finite-delta incumbents (Section 6), and the W=80,000 confirmation pass (Section 7) all
depend on having at least one cold-verified finite-delta incumbent to work from (Section
4). None was produced this session, so none of Sections 5-7 were attempted.

### 17.G Remaining risks (Section 8.F)

- **The central open risk**: a real, reproducible discrepancy between KNITRO's own live
  inner-solve status/feasibility tracking during an outer trajectory and an independent
  cold reverification at the identical terminal point (Section 17.D). This must be
  understood and resolved (or a robust workaround found — e.g. always cold-restarting the
  inner solve inside the outer callback rather than relying on `inner_loop_internal`'s own
  default zero-start, tightening the outer `feastol`, or adding a periodic mid-trajectory
  cold-verification check that aborts/rejects a KNITRO run early once it drifts) before
  Section 4's campaign can be trusted to produce a real result, however long it runs.
- Backend B's reliability from a from-scratch (non-continuation) start at `delta=1e-3` is
  poor at this `W`/starting configuration — genuine continuation (very small trust-region
  bursts) is necessary but was not, in the time available, sufficient to produce a single
  verified incumbent.
- No evidence yet on whether Backend D or a hybrid backend would behave better or worse —
  not tested in a real campaign.
- The exact root cause of the two-separate-callback-contexts KNITRO crash (Section 17.C,
  finding 2) was not pinned down — worked around, not fully explained. If a future session
  needs to split constraint blocks again (e.g. for a Jacobian-sparsity or performance
  reason), this is worth revisiting rather than assuming the workaround generalizes.
- `theta_box`/`maxit`/`h` defaults in `finite_delta_outer.jl` and
  `melitz_outer_finite_delta.opt` were set empirically during this session's debugging
  (tightened after the attempt-1 blowup) and are not independently tuned or validated
  against a wider grid.
- Sections 5-7 (profile, local polling, W=80,000 confirmation) are entirely unstarted.

## 18. 2026-07-23 follow-up session: the "central open risk" (Section 17.G) was a
## reversed divergence-constraint sign, not a KNITRO live-tracking bug — found and fixed

Governing session prompt: do not trust Section 17.G's "KNITRO live feasibility tracking
disagrees with cold verification" conclusion as established; trace the Ricardian
sign/scalar convention line by line through the actual active call path before doing any
further optimization or gradient tuning.

**Diagnosis.** Traced every layer between the inner KNITRO solve and the outer
divergence-budget constraint, without inferring any scalar's meaning from its name:

| layer | function | scalar | equals |
|---|---|---|---|
| raw inner KNITRO solve | `inner_loop_KNITRO` (`cc_algo/inner_loop_functions.jl`) | `objSol` | the functor's raw `f` at its minimizer -- **not** `Delta(theta)` directly |
| `PsiObjectiveBundleImplicit` functor | `(Q::PsiObjectiveBundleImplicit)(x; constr=...)` (`cc_algo/PsiObjectiveBundle.jl:275-361`) | `f = sum(Psi(arg0))/M + zeta` | raw dual value at trial `x`; sign convention NOT documented in the functor itself |
| find_smallest correction | `inner_loop` (`cc_algo/inner_loop_functions.jl:186-196`, the code block quoted in this session's own governing prompt) | `val = objSol * (-1)^find_smallest` (conceptually; literal code negates `val` iff `find_smallest==true`) | `Delta(theta)` (positive, reported) -- `PsiObjectiveBundleDelta` (the ground-truth `D^2+1` inner solve, `build_melitz_psi_bundle`) has `find_smallest=true` ALWAYS, so its raw `objSol` is `-Delta(theta)` |
| `melitz_combined_callback_F!` (`finite_delta_outer.jl`) | `obj(x, constr=local_c)` -- calls the SAME functor directly, bypassing `inner_loop`'s correction entirely | `local_c[1] = -f*1e10` | since this functor call is UNCORRECTED (no `inner_loop` wrapper), `f` here is the same raw, sign-unflipped value as `PsiObjectiveBundleDelta`'s raw `objSol` — i.e. `f = -Delta(theta)`, so `local_c[1] = +1e10*Delta(theta)` |

**Empirically confirmed** (no full outer KNITRO solve needed — a single inner solve plus
one direct functor call, compared against the independently-verified ground truth from
`evaluate_melitz_delta`/`PsiObjectiveBundleDelta`): at a converged benchmark point
(`Delta_truth=2.653045e-4`), `obj(x, constr=c)` gives `c[1]=+2.653045353181363e6`, EXACTLY
`+1e10*Delta_truth` (not `-1e10*Delta_truth`) to full floating-point precision. At a
deliberately blown-up point (`Delta_truth` sentinel `1e10`), `c[1]=+1.3088e23`, again
exactly `+1e10*Delta_truth`. Confirmed identical regardless of the Implicit bundle's own
`find_smallest` (true/false) -- the raw functor's `f` does not read `find_smallest` at all
(only the separate outer-gradient branch does), so this sign is a fixed property of the
shared functor, not something direction-dependent.

**Consequence**: `finite_delta_outer.jl`'s prior constraint bound, `KN_set_con_lobnd(kc,
cIndices[1], -1e10*delta)` (i.e. `constr[1] >= -1e10*delta`), was reasoned out under the
assumption `f=+Delta(theta)` — the OPPOSITE of what a direct, uncorrected functor call
actually returns. Since `constr[1] = +1e10*Delta(theta) >= 0` always, and `delta >= 0`
always, `constr[1] >= -1e10*delta` is trivially satisfied for EVERY `theta` — vacuous,
confirmed empirically at both the tiny-`Delta` and blown-up-`Delta` test points across a
`delta` grid spanning both sides of `Delta_truth` (Section 18's own regression test,
`test/melitz/runtests.jl`). This is the SAME failure mode Section 17.C's own fix was
written to eliminate (the Ricardian model's unmodified UPPER-bound convention, reasoned
under the same `f=+Delta(theta)` assumption, is *also* vacuous under that assumption) —
the fix flipped the bound direction but kept the same wrong sign assumption, so it
reintroduced an equivalent vacuous constraint via the opposite algebraic path, rather than
fixing it.

**This fully retracts Section 17.G's "central open risk."** There never was a
discrepancy between KNITRO's live feasibility tracking and cold reverification — KNITRO
correctly reported feasibility error `0.000` throughout Section 17.D's trajectories
because the divergence-budget constraint, as coded, was never capable of being violated,
regardless of where `theta` went. The `nStatus=0`/"clean, feasible, converged" terminal
point that cold-reverified to `Delta=2.12` (~2000x the stated `delta=1e-3` budget,
Section 17.D) is the expected, unremarkable consequence of an effectively unconstrained
search in that one dimension — not evidence of any KNITRO internal-state bug.

**Fix** (`finite_delta_outer.jl`): `KN_set_con_lobnd(kc, cIndices[1], -1e10*delta)`
replaced with `KN_set_con_upbnd(kc, cIndices[1], 1e10*delta)` — i.e. `constr[1] <=
1e10*delta`, which given the now-confirmed `constr[1]=+1e10*Delta(theta)` correctly
encodes `Delta(theta)<=delta`. This is EXACTLY the Ricardian model's own unmodified
`outer_loop_constraints!` convention (`cc_algo/outer_loop_functions.jl:216-228`,
`KN_set_con_upbnd(cIndices[1], 1e10*obj.δ)`) — the correct fix is therefore to trust the
shared cc_algo convention as originally written and NOT special-case Melitz's own driver
at all, the opposite of Section 17.C's own change.

**New regression test** (`test/melitz/runtests.jl`, "Section 18: divergence-budget
constraint sign"): builds the Implicit bundle at the population-Pareto benchmark, solves
the inner problem once, and asserts `constr[1]` equals `+1e10*Delta(theta)` (not
`-1e10*Delta(theta)`) against the independently-verified ground truth from
`evaluate_melitz_delta`, then confirms the ACTIVE (upper) bound convention correctly
discriminates feasible (`delta` above `Delta(theta)`) from infeasible (`delta` below), and
explicitly pins that the PRIOR lower-bound convention is vacuous at both tested `delta`
values (would have passed neither check pre-fix). Full suite: all testsets pass after the
fix (`test/melitz/runtests.jl`), no regressions.

**What remains open**: Section 4's actual finite-delta campaign (the real economic
upper/lower gains-from-trade programs) has NOT been re-run under the corrected
constraint this session — the diagnosis-and-repair work above consumed the session's
scope, per the governing prompt's own instruction to fix the constraint integration
BEFORE any further optimization/gradient tuning. Re-running Section 4's campaign with the
now-genuinely-binding budget constraint (starting from the same `delta=1e-3` continuation
strategy) is the immediate next step, and should be expected to behave differently now
that the constraint can actually reject a drifting `theta` — Backend B's own reliability
under a REAL binding constraint has not yet been observed and may differ from the
(vacuously-constrained) Section 17.D attempts.

## 19. 2026-07-23 same-session follow-up: re-ran Section 4 under the corrected constraint
## — genuinely binding, but no verified incumbent found at `delta=1e-3` in six real
## KNITRO attempts; an honest negative result, not an infrastructure failure

Immediately following Section 18's fix, re-ran the real finite-delta campaign at the
SAME setup as Section 17.D (`D=4`, `W=20,000`, `seed=29`, population-Pareto start,
Backend B `h=1e-4`, `delta=1e-3`), across six real KNITRO configurations (three per
direction), ~77 minutes of total real KNITRO wall time. **No cold-verified incumbent was
found for either direction at this `delta`** — but the RESULTS ARE QUALITATIVELY
DIFFERENT from Section 17.D's, and confirm the constraint is now genuinely binding.

| direction | config (`theta_box`, `maxit`) | wall | terminal `nStatus` | terminal `Delta` | vs. `delta=1e-3` |
|---|---|---|---|---|---|
| upper | `2.0`, `25` (direct, matching 17.D's Attempt 1) | 341s | `-410` (iter limit, infeasible) | `0.0010043` | **0.4% over** |
| upper | `2.0`, `100` | 1227s | `-410` | `0.0012557` | 26% over |
| upper | `0.15`, `100` | 555s | `-410` | `0.0027734` | 177% over |
| lower | `2.0`, `25` (direct) | 882s | `-410`, cold `Delta` sentinel `1e10` (genuine inner-solve failure, not iter-limit-near-miss) | `1.0e10` | garbage |
| lower | `0.15`, `25` | 246s | `-410` | `0.0024926` | 149% over |
| lower | `0.15`, `100` | 1355s | `-410` | `0.0019026` | 90% over |

**This is a categorically different failure mode than Section 17.D's** (retracted by
Section 18): there, the vacuous constraint let `theta` drift to `Delta≈2.12` (~2000x the
budget) or outright numerical garbage, with KNITRO's OWN tracking falsely reporting
feasibility `0.000` throughout. Here, every terminal `Delta` is within `0.4x`-`2x` of the
budget — the search is now visibly TRYING to satisfy `Delta(theta)<=delta` and getting
close, exactly what a genuinely binding constraint should produce — and KNITRO correctly
reports `-410`/nonzero feasibility error at every terminal point (no false "clean" reports
this time; the diagnosis that Section 17.D's live/cold disagreement was purely the
vacuous-constraint artifact, not a tracking bug, is further corroborated: this round shows
NO live/cold disagreement anywhere — KNITRO's own live feasibility error and the cold
`Delta` recheck agree in direction every time).

**Two distinct sub-failure-modes, not one**:

1. **Large `theta_box` (`2.0`) risks genuine inner-CC-dual pathology**, not just constraint
   violation: lower's direct attempt (`theta_box=2.0`) hit the `1e10` cold-`Delta` sentinel
   (a real inner-solve failure, `nStatus` outside the accepted set) partway through the
   trajectory — the SAME class of risk Section 17.G already flagged ("Backend B's
   reliability from a from-scratch start ... is poor"), now confirmed under a constraint
   that is actually trying to prevent exactly this kind of excursion but doesn't always
   succeed within KNITRO's own step-acceptance logic.
2. **Small `theta_box` (`0.15`) avoids the inner-solve pathology but converges too slowly**:
   every `0.15`-box run shows a clean, monotonically DECLINING feasibility error (upper's
   `0.15`/`100` run stayed EXACTLY feasible for its first 56 iterations before a late
   destabilization at iteration 57 — the cleanest trajectory of the six) but does not reach
   the tight `delta=1e-3` tolerance within the iteration budgets tried (`25` or `100`).
   Critically, MORE iterations did not monotonically help: upper's `0.15`/`100` run
   (`Delta=0.00277`) finished WORSE than its own `2.0`/`25` run (`Delta=0.0010043`) despite
   3x the wall time — confirming this is genuine optimizer/gradient-noise difficulty at
   this tight a budget, not simply an iteration-count shortfall that a bigger `maxit` alone
   resolves.

**Interpretation**: `delta=1e-3` is tight relative to Backend B's own gradient accuracy
(a finite-bandwidth secant, `h=1e-4`, chosen for the switch-noise reasons documented in
Section 1/17.B, not for maximal precision) — the search can approach the budget boundary
but the approximate gradient does not reliably keep it there once close, consistent with
`docs` Section 17.G's own (still-valid) flagged risk about Backend B's reliability near a
tight budget. This is now understood as a genuine numerical-difficulty finding, not
conflated with the (retracted) constraint-sign bug.

**Recommended next steps** (not attempted this pass, this session's real-KNITRO budget
having already gone to diagnosing + re-testing the Section 18 fix):
- Reverse the continuation ORDER Section 4 originally specified (tightest `delta` first,
  widen only on failure): given `delta=1e-3` fails from a cold population start at every
  config tried here, START from a looser `delta` (e.g. `1e-2` or `5e-3`, where the
  population point's own `Delta_pop=7.55e-6` is comfortably inside budget with more slack
  for gradient noise), get a genuinely verified incumbent there, and continue INWARD
  (tightening `delta`) from that verified point rather than from the raw population start.
- Try Backend D (`make_melitz_moments_jacobian_d`, the hand-derived exact branch
  derivative, implemented in `finite_delta_outer.jl` but never run in a real campaign —
  Section 17.E) in place of Backend B, since it removes finite-difference bandwidth noise
  from the constraint Jacobian entirely.
- A proper trust-region-aware continuation (shrinking `theta_box` adaptively based on
  whether the previous burst's feasibility error declined vs. grew) rather than a single
  fixed `theta_box` per attempt.

Full logs (all six KNITRO runs, driver scripts) pushed to Dropbox,
`melitz_finite_delta_constraint_sign_fix_2026-07-23/section4_followup/`.

## 20. 2026-07-23 same-day follow-up: three integration/bookkeeping bugs fixed (objective
## contamination, opaque 1e10 constraint scaling, garbage-value inner-failure handling) +
## incumbent bookkeeping rebuilt — first-ever cold-verified incumbents at every
## `(delta, direction)` combination tried

Governing session prompt: do not simply re-run longer KNITRO trajectories on top of
Section 18/19's sign fix — the latest results (Section 19) exposed additional integration
defects that had to be repaired first: (1) the reported "no cold-verified incumbent" was a
bookkeeping bug, not a real infeasibility; (2) the outer objective was contaminated by the
inner solve's own failure sentinel; (3) the divergence constraint's `1e10` scaling made
KNITRO's feasibility-error reporting economically meaningless; (4) a failed inner solve was
written into the KNITRO problem as a garbage constraint value instead of signaled as an
evaluation failure. All four are fixed below, verified with new regression tests passing
through the exact production KNITRO callback (not helper-function bypasses), and the
Section 4 campaign was re-run — for the first time, every one of 4 `(delta, direction)`
combinations tried produced a genuine, cold-verified, budget-respecting incumbent.

### 20.A Incumbent bug

**Diagnosis.** The pre-existing `solve_melitz_finite_delta_bound` only ever inspected the
KNITRO trajectory's OWN terminal point (`terminal_eval`) for incumbent purposes. Section
19's six real campaigns repeatedly found feasible, low-`Delta` points mid-trajectory
(iterations 0-10, feasibility error `0.000`, objective genuinely improving) that were never
captured, because nothing recorded them — only the (often drifted-infeasible) terminal
point was ever checked. Separately, the cold-verified starting point itself
(`Delta_start=7.5545e-6 < delta` at every `delta` tried) was never installed as a fallback
incumbent before `KN_solve` ran at all, so a pathological trajectory (or, as tested, a
trajectory artificially capped at 1 iteration) could report "no incumbent" even though a
perfectly good one — `theta_init` itself — was available for free.

**Fix** (`src/melitz/finite_delta_outer.jl`): `evaluate_melitz_delta(theta_init, ctx,
obj_inner; cold=true)` is now run BEFORE `KN_solve`, classified via the new
`melitz_classify_outer_feasibility` (Section 20.C below), and installed as
`initial_incumbent` whenever outer-feasible. During the trajectory, every successful
callback evaluation is classified and (if outer-feasible) pushed into a bounded,
objective-ranked `live_candidates` list (`register_live_candidate!`, up to 5 tracked) — at
NO extra KNITRO-solve cost (see Section 20.D). After `KN_solve` returns, live candidates
are cold-reverified best-first until one survives; the result falls back to
`initial_incumbent` if none does. `MelitzFiniteDeltaOuterResult` now exposes
`initial_incumbent`/`best_live_incumbent`/`cold_verified_incumbent` as three DISTINCT
fields (previously conflated into one `cold_verified_incumbent` that only ever looked at
the terminal point).

**Regression test** (`test/melitz/runtests.jl`, "Section 2.1/12: initial incumbent
survives a KNITRO run limited to one iteration"): builds a temporary options file with
`maxit 1`, runs `solve_melitz_finite_delta_bound` with it, and asserts
`initial_incumbent !== nothing`, `initial_incumbent.classification.outer_feasible`, and
`cold_verified_incumbent.eval.Delta <= delta` — this test would have failed under the
prior bookkeeping (a 1-iteration KNITRO run has essentially no opportunity to converge, so
the terminal-point-only logic would very likely report `nothing`). 4/4 assertions pass.

### 20.B Objective contamination

**Diagnosis.** The prior combined callback set `evalResult.obj[1] = -objSol`, where
`objSol` came from `inner_loop_internal(obj::PsiObjectiveBundleImplicit, theta)`. On a
SUCCESSFUL inner solve this equals `obj.H_save = theta[1]*(-1)^find_smallest` (the correct,
finite `K`-based objective) — but on a FAILED inner solve, `inner_loop_internal` returns
the fixed sentinel `-1e10` regardless of `theta` (`cc_algo/inner_loop_functions.jl:217-241`,
shared, unmodified code). So `evalResult.obj[1]` silently became `+1e10` (or `-1e10`)
whenever the inner solve failed at a trial point — exactly the documented "lower run's
outer objective becomes 1.0e10" symptom. The outer objective (`theta[1]`, always finite and
well-defined regardless of whether the inner divergence problem happens to solve at that
point) was needlessly coupled to the inner solve's own success/failure.

**Fix**: the objective is now computed DIRECTLY from `theta`, with no dependence on the
inner solve's return value at all: `evalResult.obj[1] = find_smallest ? theta[1] :
-theta[1]` (Upper-GT problem: `obj(theta)=theta[g_index]`; Lower-GT: `obj(theta) =
-theta[g_index]`; gradient exactly `+e_g`/`-e_g`, set directly, not through
`calculate_grad_k!`'s legacy K-gradient path).

**Verification**: a dedicated fixed-point test (`melitz_fixed_point_probe`, Test D) fixes
`theta` at a point independently confirmed to make the inner CC dual solve fail even after
a cold retry (`evaluate_melitz_delta` at that `theta` gives `nStatus ∉
{0,-100,-101,-103}`), then confirms via the REAL, registered KNITRO callback that the
evaluation is rejected as a callback failure (`eval_failed=true`, KNITRO status `-502`) —
`obj_value` is `NaN` (never touched, never `1e10`, never a garbage number), so the
objective-contamination question is moot at this point (a rejected evaluation has no
objective at all, not a corrupted one). At a FEASIBLE point (Test A), `obj_value` is
verified to equal `theta[1]` exactly (`atol=1e-10`). Combined, these two probes bracket the
claim precisely: the objective is correct when the inner solve succeeds, and is never
computed as (or contaminated by) the inner solve's own sentinel when it fails, because the
inner solve is never READ for the objective at all.

### 20.C Constraint scaling

**Old**: `constr[1] = -f*1e10` (the shared `PsiObjectiveBundleImplicit` functor,
unmodified) `<= 1e10*delta`. Correct in SIGN (Section 18) but badly scaled: at
`delta=1e-3`, a raw divergence excess of `4.26e-6` produced a KNITRO feasibility error
around `42,600` — seven orders of magnitude larger than the cutoff rows' own `O(1)` scale,
making KNITRO's own reported feasibility error economically meaningless and effectively
un-comparable across the constraint block.

**New** (Section 4.1's preferred dimensionless representation): `c_delta(theta) =
Delta(theta)/delta <= 1`. Implementation reuses the SAME underlying `1e10*Delta(theta)`
raw value the shared functor already computes (no `cc_algo` edits) — both the constraint
VALUE and its JACOBIAN are divided by the identical factor `1e10*delta` before being
handed to KNITRO, so value/Jacobian/finite-difference-checks/tolerances all share one
scaling consistently (Section 4.2).

| quantity | old (1e10 scaling) | new (Delta/delta scaling) |
|---|---|---|
| constraint value at a converged point (`Delta≈7.5e-6`, `delta=1e-3`) | `7.5e1` | `7.5e-3` |
| constraint value at the SAME `delta=1e-2` | `7.5e1` (same — scale is `delta`-independent under the old bug) | `7.5e-4` |
| feasibility error for a `4.26e-6` divergence excess at `delta=1e-3` | `~42,600` | `~4.3e-3` |
| KNITRO feasibility error observed live at a `-410` terminal point (Section 20.F, `delta=1e-2`, upper) | (old convention would show `~2.1e10`) | `2.13e-1` (Delta itself is genuinely `0.21`, `21x` over budget — an economically legible number, not a scale artifact) |

**Regression tests**: Test A (feasible Pareto point) confirms `c[1] == Delta(theta)/delta`
to `rtol=1e-6` and `c[1] < 1`; Test B (budget-infeasible, inner-valid point, constructed at
`scale=0.005` perturbation, `nStatus=0`) confirms `c[1] == Delta(theta)/delta > 1` and that
KNITRO reports genuine infeasibility (`nStatus ∉ {0}`, specifically `-201` observed live);
a boundary case is implicit in the Section 8.1 root-finding cross-check (Section 20.E)
where the KNITRO-found endpoint's cold `Delta` sits within `0.002%`-`2%` of `delta` in
every one of 4 cases.

### 20.D Inner failure handling

**Diagnosis**: the prior callback wrote `local_c[1] = 1e9` (hand-invented) whenever
`abs(objSol)==1e10`, telling KNITRO the point WAS successfully evaluated with a huge but
finite constraint value. This is exactly the failure mode `full_aod_diag/d4_exact/
c9_phase8_d20_pilot.jl` (a prior session, Ricardian model) diagnosed and fixed: a fabricated
"successful" evaluation with a huge objective/constraint (and, worse, a PAIRED zero
gradient at the corresponding `evalResult.jac`) trivially satisfies first-order optimality
and can make KNITRO falsely declare convergence.

**Fix**: `inner_solve_verified_or_fail` (shared by `cb_F!`/`cb_G!` via
`melitz_build_finite_delta_callbacks`) retries once from a neutral cold start
(`obj.use_cached_x=false`) on a bad `nStatus`; if that ALSO fails, it `throw`s a
`DomainError`. KNITRO.jl's own `_try_catch_handler` (`C_wrapper.jl`) catches this and
converts it to a proper `KN_RC_EVAL_ERR`/`KN_RC_CALLBACK_ERR` status, telling KNITRO to
reject/backtrack from the trial point — matching `c9_phase8_d20_pilot.jl`'s own documented
convention exactly (chosen deliberately over re-deriving a failure convention from
scratch).

**Truth table** (`melitz_fixed_point_probe`, exercising the real registered KNITRO
callback, D=4/W=2,000/seed=29 unless noted):

| case | inner solve | `eval_failed` | KNITRO `nStatus` | `obj_value` | notes |
|---|---|---|---|---|---|
| Test A: feasible Pareto point | succeeds (`nStatus=0`) | `false` | `0` | `theta[1]` exactly | `c[1]<1`, all cutoff rows `>=0` |
| Test B: budget-infeasible, inner-valid | succeeds (`nStatus=0`) | `false` | `-201` (genuine infeasible) | `theta[1]` exactly | `c[1]>1`, a REAL violated constraint, not a numerical failure |
| Test C: cutoff-infeasible, inner-valid | **not constructed** — see below | — | — | — | at this D=4/W∈{500,2000} fixture, every accessible single-coordinate perturbation large enough to violate a deterministic cutoff constraint ALSO destabilized the inner CC dual (a systematic search over 25+ coordinates x both directions x magnitudes from `0.001` to `2.0` found zero exceptions) — reported honestly as a fixture-scale limitation, not forced |
| Test D: inner numerical failure | fails even after cold retry (`nStatus=-400`, confirmed independently via `evaluate_melitz_delta`) | `true` | `-502` (`KN_RC_EVAL_ERR`) | `NaN` (never `1e10`, never fabricated) | `live_candidates` empty — no incumbent contamination |

Test C's non-constructibility is itself informative: it suggests that, at this fixture's
`W` scale, "mild economic irregularity" (a barely-violated cutoff) and "numerical
pathology in the CC dual" are not cleanly separable failure modes reachable by small
perturbations of a single free coordinate — consistent with (though not conclusive proof
of) the gravity-pivot construction's coupling of nominally-local coordinate changes across
cells. Not investigated further this session (out of scope: the callback's OWN behavior in
this combined scenario — reject cleanly via the Section 5 throw path — is already covered
by Test D, which is the behaviorally relevant case regardless of which condition triggered
the inner failure).

**Section 7 (combined-callback audit) A/B/A test**: builds ONE shared `obj`/callback pair
(`melitz_build_finite_delta_callbacks`), evaluates feasible point A (via duck-typed mock
`EvalRequest`/`EvalResult` structs calling `cb_F!`/`cb_G!` directly — the exact production
closures, not reimplementations), evaluates a known-failing point B (confirms it throws a
`DomainError`), then re-evaluates point A again. Requires COMPLETE restoration of
objective/constraint/gradient/Jacobian (`==`, not `≈`) and confirms the failed point B never
enters `live_candidates` (2 entries recorded, both A, none B). 7/7 assertions pass.

### 20.E Restricted outer tests (Section 8)

**8.1 (1-D gamma-only test, D=4/W=20,000/seed=29, real KNITRO throughout).** Holding all 29
other free coordinates fixed at the population-Pareto benchmark, a 21-point grid of
`g=theta[1]` around the benchmark (step `0.03`, span `±0.30`) confirms `Delta(g)` has a
sharp, narrow trough at `g_0=-0.042387` (`Delta(g_0)=7.5545e-6`, exactly the known
population value) — outside a band of roughly `±0.02`, either the inner CC dual solve
fails (`nStatus∈{-102,-400,-101}`) or `Delta` explodes (`>1e15`, a numerical, not
economic, blow-up). The ACTUAL 1-D-free KNITRO NLP (registering the exact production
callback, only `theta[1]` free, `theta_box` up to `0.5`) was then solved for both
directions at `delta∈{1e-3,1e-2}`:

| `delta` | direction | KNITRO `nStatus` | KNITRO endpoint `g` | independent bisection root `g*` | agreement | cold `Delta` at endpoint |
|---|---|---|---|---|---|---|
| `1e-3` | upper | `-400` (iter limit, feasible) | `-0.047184` | `-0.047189` | `5e-6` | `9.990e-4` (`<=1e-3` ✓) |
| `1e-3` | lower | `-410` (iter limit, marginal) | `-0.038133` | `-0.038132` | `1e-6` | `1.000004e-3` (`4e-9` relative over) |
| `1e-2` | upper | `-410` (iter limit, marginal) | `-0.058765` | `-0.058755` | `1e-5` | `1.0057e-2` (`0.6%` over) |
| `1e-2` | lower | `0` (optimal) | `-0.030087` | `-0.030055` | `3e-5` | `9.877e-3` (`<=1e-2` ✓) |

The bisection root is an INDEPENDENT ground truth (`evaluate_melitz_delta`'s own real
KNITRO inner solve, no relation to the outer NLP's constraint/gradient machinery) — KNITRO's
own endpoint agreeing with it to 4-6 significant figures in every one of 4 cases is strong,
clean evidence that the objective sign, the (rescaled) divergence constraint's VALUE, and
its JACOBIAN are all correct: a wrong-signed or badly-scaled Jacobian would not let a
gradient-based Newton-type method converge to the true root this precisely. This is exactly
the "transparent end-to-end test of objective sign, divergence constraint, constraint
Jacobian, upper/lower direction, incumbent tracking" the main prompt's Section 8.1 asked
for, and it passes cleanly.

**8.2 (small-coordinate test, gamma + 1 ordinary `A` + 1 ordinary `f` free, same
fixture).** A coarse `3^3` grid (step `±0.5` per coordinate — deliberately the SAME box
used for the KNITRO solve below) found only the benchmark point itself feasible at
`delta=1e-2`; every other grid point either failed the inner solve or was wildly over
budget — confirming the feasible neighborhood is narrow relative to a `0.5`-scale grid
(consistent with 8.1's own `±0.02` gamma-only band). The 3-free-coordinate KNITRO NLP
(`theta_box=0.5`, `delta=1e-2`, real production callback) for both directions:

| direction | KNITRO `nStatus` | endpoint `(g,A,f)` | cold `Delta` | vs. `delta=1e-2` |
|---|---|---|---|---|
| upper | `-410` (iter limit, marginal) | `(-0.0667, 0.3122, -0.4886)` | `2.135e-2` | `2.1x` over |
| lower | `-410` (iter limit, marginal) | `(-0.0307, 0.3146, -0.5585)` | `1.019e-2` | `1.9%` over |

No crashes, no objective/constraint contamination, correct classification at every
evaluation — the machinery scales cleanly beyond the trivial 1-D case, but convergence
(within the shared `maxit=25`, Backend B finite-difference gradient) visibly gets harder as
free dimensionality grows from 1 to 3, exactly the kind of staged difficulty escalation
Section 8's own ordering is designed to surface before committing to the full 30-free-
coordinate problem.

### 20.F Full outer results (Section 9/10 campaign, D=4/W=20,000/seed=29, full 30-free-
### coordinate problem, `theta_box=0.10`, Backend B `h=1e-4`, `maxit=25` — the shared
### default `melitz_outer_finite_delta.opt`)

Population-Pareto initial incumbent at every run: `gamma_prime=0.958498`,
`Delta=7.5545e-6`, `nStatus=0`, `min_slack=0.0166`, gravity residuals `~1e-17`. **For the
first time across every session that has attempted this exact campaign (Sections 17.D,
19), all 4 `(delta,direction)` combinations produced a genuine, cold-verified,
budget-respecting incumbent** — none required falling back to the initial incumbent (all 4
found a strictly better live candidate that also survived independent cold
reverification):

| `delta` | direction | terminal `nStatus` | terminal outer-feasible | cold-verified `gamma_prime` | cold `Delta` | budget slack (`delta - Delta`) | min cutoff slack | gravity resid. (A/f) | wall time | inner solves (infeas / eval-failures) |
|---|---|---|---|---|---|---|---|---|---|---|
| `1e-2` | upper | `-410` | false (`Delta=0.213`, `21x` over) | `0.935445` | `9.532e-3` | `4.68e-4` | `0.0205` | `-2.3e-17` / `2.3e-17` | `96.2s` | `149` (`52` / `26`) |
| `1e-2` | lower | `-400` | **true** (terminal itself verified) | `0.988286` | `9.956e-3` | `4.42e-5` | `0.0164` | `-1.2e-17` / `1.9e-17` | `373.6s` | `224` (`84` / `42`) |
| `1e-3` | upper | `-410` | false (`Delta=9.86e-3`, `9.9x` over) | `0.952339` | `8.808e-4` | `1.19e-4` | `0.0174` | `-3.7e-17` / `2.1e-17` | `147.8s` | `226` (`70` / `35`) |
| `1e-3` | lower | `-410` | false (`Delta=2.03e-3`, `2.0x` over) | `0.961709` | `9.792e-4` | `2.08e-5` | `0.0190` | `-1.0e-17` / `1.9e-17` | `342.6s` | `224` (`90` / `45`) |

Gradient backend: B throughout (`h=1e-4`); Backend D was not exercised in a real campaign
this session (still available, unrun, per Section 17.E's own carryover). Every
`cold_verified_incumbent` moved `gamma_prime` in the economically correct direction (down
for upper/minimize, up for lower/maximize) relative to the initial `0.958498`, by an amount
that scales sensibly with the budget (`1e-2` allows a larger move than `1e-3`, as expected).
In 3 of 4 cases the terminal KNITRO trajectory point itself drifted outside the budget
(by `2x`-`21x`) even though a genuinely better, cold-verified point existed earlier in the
SAME trajectory — this is exactly the scenario the Section 20.A incumbent-bookkeeping fix
was built to rescue, and it did so in every case tried. The one case where the terminal
point itself was outer-feasible (`delta=1e-2`, lower) had the terminal and
`cold_verified_incumbent` coincide.

Not run this session (flagged, not silently dropped): the reversed-continuation-order
strategy (start looser, tighten via short bursts) and Backend D, both recommended by
Section 19's own carryover — not needed this pass since the direct single-shot solve
already succeeded at both `delta` values tried; `theta_box∈{0.05,0.15}` (only `0.10`
tried); `W=80,000` confirmation (Section 7 of the finite-delta campaign spec).

### 20.G Remaining numerical risks

- **Stable incumbent tracking**: yes, verified both by construction (Section 20.A) and by
  the campaign's own live results (Section 20.F) — every run returned a genuinely better,
  cold-verified incumbent than the trivial fallback.
- **Stable inner retries**: the one-cold-retry-then-reject convention (Section 20.D) is
  now uniform across `cb_F!`/`cb_G!`; `inner_eval_failures` (26-45 per run out of 149-226
  total inner solves, i.e. roughly 15-20%) confirms the retry-then-throw path is exercised
  routinely at this `theta_box`/`delta` combination, not a rare edge case — KNITRO
  correctly backtracked from every one of these without crashing.
- **Meaningful constraint scaling**: yes (Section 20.C) — feasibility errors and
  constraint values are now O(1)-to-O(tens) at worst, not O(1e4)-O(1e7), and directly
  interpretable as "how many multiples of the budget `Delta` currently sits at."
- **Reliable gradient behavior**: qualified yes. Backend B (finite-bandwidth secant,
  `h=1e-4`) is what every result in Sections 20.E-F used; it visibly does NOT fully
  converge within `maxit=25` at this `theta_box`/`W` combination (every terminal `nStatus`
  but one was a `-4xx` iteration-limit code, not `0`) — but the INCUMBENT-TRACKING
  machinery makes this non-fatal: a genuinely better, verified point is still recovered
  from mid-trajectory in every case. Backend D (exact hand-derived branch derivative,
  implemented but never run in a real full-scale campaign) remains the clearest lever for
  actually reaching KNITRO-native convergence (`nStatus=0`) rather than relying on
  incumbent rescue.
- **Sensitivity to `h`/`theta_box`**: only `h=1e-4`/`theta_box=0.10` tried this session for
  the full campaign (Section 8.1/8.2 tried `theta_box` up to `0.5` at reduced
  dimensionality only). Not swept.
- **Test C's non-constructibility** (Section 20.D) is flagged as an open, not fully
  explained, structural question for whoever next touches the gravity-pivot/cutoff
  machinery — worth revisiting if a future session needs a clean cutoff-only-infeasible
  fixture point for some other purpose.

### 20.H Reproduction record (main prompt Section 1)

- git commit at session start: `1a9b5aa` (`melitz/fullD-delta-star`, 19 commits ahead of
  `origin/production/fullA-exact`); working tree had only unrelated untracked `output/`,
  `stata/` directories, no uncommitted tracked-file changes. This session's changes
  (`src/melitz/delta_star.jl`, `src/melitz/finite_delta_outer.jl`,
  `test/melitz/runtests.jl`) are UNCOMMITTED as of this report — left for the user to
  review/commit.
- Julia `1.12.6`; KNITRO `13.0.1` (`/opt/shared_sw/knitro/13.0.1`).
- Option-file SHA-256 (unchanged from session start): `melitz_inner_loop_options.opt`
  `9bc9c73b...`, `melitz_outer_finite_delta.opt` `a80b0409...`, `ek_inner_loop_options.opt`
  `f303308a...`, `ek_outer_loop_options.opt` `8b90810b...`.
- Pre-change baseline test run: 100% pass (all existing testsets green) — confirms the
  bugs fixed this session were genuine integration defects, not something the existing
  test suite already caught.
- Population-Pareto starting point independently re-confirmed (D=4, W=20,000, seed=29):
  `Delta_start=7.5545087570375445e-6` (matches the governing prompt's own cited
  `~7.5545e-6`), `gamma_prime=0.958498472749465` (matches `~0.958498`), `nStatus=0`,
  `lfd_ok=true`, cutoff-feasible (`min_slack=0.01655`), gravity exact
  (`~1.5e-17`/`4.2e-17`), `verified=true` — a cold-verified outer-feasible incumbent, per
  the prompt's own description.
- Full post-change test suite: **337/337 passing**, zero regressions, including 43 new
  assertions across 4 new testsets (Section 6: 21, Section 7: 7, Section 2.1/12: 4, plus
  11 in the strengthened Section 3 test, up from 5).

Session outputs (this write-up, campaign logs, Section 8 validation logs) pushed to
Dropbox, `Gravity robustness/Analysis/Server Output/melitz_finite_delta_bookkeeping_fix_2026-07-23/`.

## 21. 2026-07-23 continuation session: affine cutoff constraints as true KNITRO linear
## rows + experimental log-cutoff parameterization (governing prompt Sections 1-5 done;
## Sections 6-11 not reached)

Governing session prompt: freeze/validate Section 20's state, correct gains-from-trade
reporting everywhere, derive and register the deterministic cutoff restrictions as true
affine KNITRO constraints (replacing the generic nonlinear FC/GA evaluation), implement an
experimental log-cutoff outer parameterization, profile the outer computation, diagnose
whether A/f/cutoff coordinates are meaningfully searched, and compare the two
parameterizations. Given this session's real-KNITRO time budget against the full 12-section
scope, work was explicitly prioritized: Sections 1-5 are complete, tested, and committed;
Section 4's benchmark ran one matched delta (not the full grid); Sections 6-11 were not
reached. Each cut is flagged explicitly below, not silently dropped.

### 21.A Current-state checkpoint (Section 1)

Commit at session start: `32336b6` (19 commits ahead of `origin/production/fullA-exact`,
per Section 20.H); `git status` was clean (only untracked `output/`/`stata/`, never
committed in this repo's history) -- no uncommitted tracked-file changes to freeze. Julia
`1.12.6`; KNITRO `13.0.1`; option-file SHA-256 hashes UNCHANGED from Section 20.H's own
recorded values. Pre-change test suite: 337/337, matching the prior session's own count
exactly -- confirms no drift.

Reproduced all four finite-delta incumbents ONCE, exactly (`scripts/
melitz_finite_delta_campaign.jl`, NEW; D=4, W=20,000, seed=29, Backend B `h=1e-4`,
`theta_box=0.10`):

| delta | direction | nStatus | Delta | gamma_prime | GT (correct, wage-ratio) | GT (ACR, fixed reference) |
|---|---|---|---|---|---|---|
| 1e-2 | upper | -410 | 9.532018e-03 | 0.935445 | 0.080287 | 0.065237 |
| 1e-2 | lower | -400 | 9.955828e-03 | 0.988286 | 0.045970 | 0.065237 |
| 1e-3 | upper | -410 | 8.807682e-04 | 0.952339 | 0.069246 | 0.065237 |
| 1e-3 | lower | -410 | 9.791978e-04 | 0.961709 | 0.063151 | 0.065237 |

`gamma_prime` matches Section 20.F's own reported values to displayed precision at every
one of the 4 combinations -- full determinism confirmed, environment unchanged.

### 21.B Corrected gains-from-trade reporting (Section 2)

The correct formula (`melitz_gains_from_trade`, `GT_j = 1 -
(w_prime_j/w_j)*gamma_prime_j^(1/(sigma-1))`) was already implemented and tested in Gate A
(Section 14.1) -- no code change was needed in `equilibrium.jl` itself. What was missing
was a live campaign driver that actually prints it: the only prior "campaign logs"
reporting `GT(naive)` were ad hoc/uncommitted scratch scripts from a previous session, not
part of this repo. The new `scripts/melitz_finite_delta_campaign.jl` reports BOTH
`melitz_gains_from_trade` (correct) and `acr_gains_from_trade` (independent cross-check)
for the population start and every incumbent -- table above. This makes the ECONOMIC
STAKES of the Gate-A fix concrete for the first time against real finite-delta incumbents:
the naive formula would give a different number at every incumbent, since
`w[target]/w'[target] != 1` at this fixture (a genuine GE output, per
`melitz_solve_wages_ge`'s own docstring). `GT(ACR)` is a fixed reference constant across
rows because `acr_gains_from_trade` uses only the empirical baseline domestic trade share
(fixed data), not the searched outer point.

### 21.C Affine cutoff constraints (Section 3, NEW file `src/melitz/affine_cutoff.jl`)

**Derivation**: `expand_free_theta`'s entire chain (A-gravity pivot, `f[j,j]`'s
autarky-cutoff derivation, f-gravity pivot) is affine in `theta_free`, so `q(theta_free) =
log(zhat(theta_free)) = q0 + Q*theta_free` exactly (not a local linearization).

**Two independent constructions of `(Q,q0)`**, required to agree at machine precision:
`affine_cutoff_map_basis` (authoritative -- unit-basis probes around a fixed origin,
treating `expand_free_theta -> melitz_baseline_cutoff -> log` as a black box, verified
base-independent) and `affine_cutoff_map_analytical` (independent cross-check -- hand-
derived by chaining the `GravityPivot` structs' own linear algebra directly, never calling
`expand_free_theta`). Agreement: `max|Q_basis-Q_analytical| = 4.4e-16`,
`max|q0_basis-q0_analytical| = 4.4e-16`.

`build_melitz_affine_cutoff_system` assembles the `D + D*(D-1) = 16` (D=4) row system
`C*theta_free+b>=0` (`C=S*Q`, `b=S*q0-epsilon`), row-scaled (`scale[row]=max(1,
||C_raw[row,:]||_2)`). 140 equivalence assertions: 1000 random free vectors match
`melitz_cutoff_constraints_at` to `2.2e-15`; `C_raw` matches
`melitz_cutoff_constraint_jacobian`'s ForwardDiff-exact Jacobian to `~1e-15`; row scaling
preserves feasibility sign; a cutoff-feasible point plus a domestic-support-infeasible and
an export-selection-infeasible point were constructed DETERMINISTICALLY (moving from the
population-Pareto point along a chosen row's negative normal direction -- the target
row's slack becomes exactly `-margin` by the affine structure) -- the prior session's own
Section 20.D flagged this construction as NOT achievable via single-coordinate search;
Section C's construction sidesteps that entirely.

**KNITRO registration** (`melitz_register_finite_delta_knitro_problem!`, new function in
`finite_delta_outer.jl`; new `cutoff_constraint_backend::Symbol` option --
`:linear`/`:nonlinear_reference` -- threaded through `melitz_build_finite_delta_callbacks`,
`solve_melitz_finite_delta_bound`, `melitz_fixed_point_probe`): under `:linear`, the 16
cutoff rows are registered via `KN_add_con_linear_struct` (constant coefficients, no
per-iterate callback cost); the eval callback covers ONLY the genuinely nonlinear
divergence-budget row. Still ONE eval-callback context throughout, deliberately avoiding
Section 17.C's documented "two separate callback contexts" crash.

**A genuine, reproducible KNITRO limitation was found and fixed in the process**: once
this problem's own callback has triggered >=1 nested `KN_new`/`KN_solve`/`KN_free` cycle
(exactly what happens every evaluation here, for the real CC inner solve), a POST-SOLVE
`KN_get_con_values_all` query for the natively-registered linear rows comes back
corrupted/stale -- reproduced in a minimal example completely outside Melitz (a bare
2-row linear-constraint toy problem; without the nested KN instance the reported values
are exact, with it they are wrong). Solve-time enforcement itself is UNAFFECTED (cross-
checked: KNITRO's own presolve-deduced-infeasibility message for a constructed infeasible
point reported the exact expected violation, `-0.02`, matching the constructed margin to
machine precision, even though a post-hoc query for the same row would have been
corrupted). Fixed by recomputing the cutoff-row values directly in Julia
(`cutoff_sys.C*theta+cutoff_sys.b`) in `melitz_fixed_point_probe`'s return value; the
production incumbent-tracking path (`solve_melitz_finite_delta_bound`) was already
unaffected (it never calls `KN_get_con_values_all` for cutoff rows -- every feasibility
decision goes through the independent Julia-side `evaluate_melitz_delta`/
`evaluate_melitz_delta_from_solution`). 17 real-KNITRO integration assertions
("Section 3.3/3.4") confirm `:linear` matches `:nonlinear_reference` to `1e-8`, a
constructed domestic-infeasible point is correctly rejected (`nStatus=-204`), and
`solve_melitz_finite_delta_bound` runs end to end under `:linear`.

Full suite after Section 3 alone: 473/473 (up from 337).

### 21.D Section 4: matched linear-vs-nonlinear benchmark

One matched delta (`1e-2`, both directions -- `1e-3` not run this session), same
`theta_init`/`gradient_backend=:B`/`h=1e-4`/`theta_box=0.10`/`maxit=25`, real KNITRO
throughout (`scripts/melitz_cutoff_backend_benchmark.jl`, NEW), complete-trajectory wall
time (per the governing prompt's own instruction not to infer a speedup from isolated
cutoff-evaluation timing):

| delta | dir | backend | wall(s) | nStatus | FC calls | GA calls | inner solves (infeas) | Delta | gamma_prime |
|---|---|---|---|---|---|---|---|---|---|
| 1e-2 | upper | nonlinear_reference | 88.1 | -410 | 97 | 26 | 149 (52) | 9.532e-3 | 0.935445 |
| 1e-2 | upper | linear | 76.6 | -410 | 107 | 26 | 162 (58) | 8.859e-3 | 0.940030 |
| 1e-2 | lower | nonlinear_reference | 334.3 | -400 | 156 | 26 | 224 (84) | 9.956e-3 | 0.988286 |
| 1e-2 | lower | linear | 264.0 | -410 | 141 | 26 | 197 (60) | 9.871e-3 | 0.977397 |

**Speedup: 1.15x (upper), 1.27x (lower)** -- genuine but modest, not dramatic: total wall
time is dominated by the nested CC inner KNITRO solves (149-224 per run), not by the outer
cutoff-constraint Jacobian evaluation `:linear` eliminates -- exactly the outcome the
governing prompt's own warning anticipated. `GA_calls` identical between backends at
matched settings (26 in all 4 runs, driven by KNITRO's own outer-iteration count under the
shared `maxit=25`); `FC_calls` differ moderately. Both backends terminate at the same
iteration-limit-family status (not `nStatus=0`) at this budget -- consistent with the
prior session's own finding that `maxit=25`/`theta_box=0.10` does not reach full KNITRO-
native convergence on the 30-free-coordinate problem. The two backends' cold-verified
incumbents differ by `0.5-1.1%` in `gamma_prime` (both genuinely cold-verified,
budget-respecting, feasible -- KNITRO's own internal step behavior differs slightly once
the Jacobian sparsity/evaluation split changes, even though constraint VALUES agree to
`1e-8` at any fixed point; not evidence either backend is wrong, and the feasible set
itself was already verified identical at machine precision in Section 21.C).

**Recommendation**: switch the production default to `cutoff_constraint_backend=:linear`
for future finite-delta campaigns -- mathematically exact, faster in every one of the 4
real runs tried, and simpler (the callback shrinks to the one genuinely nonlinear row).
Kept at `:nonlinear_reference` as the DEFAULT in `solve_melitz_finite_delta_bound`'s own
signature this session (zero behavioral change to existing call sites); flipping the
default is a one-line follow-up once this recommendation is reviewed.

### 21.E Log-cutoff parameterization: economic core (Section 5, NEW file
### `src/melitz/log_cutoff_param.jl`)

A parallel `:logcutoff` outer coordinate system searching directly over baseline
log-cutoffs `q_od = log(zhat_od)` instead of `log f_od`, same `2D^2-2` free dimension,
built alongside (not replacing) `:logf`.

`melitz_log_f_from_q` inverts the baseline cutoff formula for `log(f_od)` given
`(q_od,a_od)`. The focal domestic cell is NEVER a free q coordinate:
`derive_qjj_from_autarky_cutoff` derives `q[j,j]` from `(g,fixed primitives)` alone,
algebraically INDEPENDENT of `A[j,j]` -- derived symbolically (substituting `log(f_jj)`'s
own affine dependence on `(log A_jj, g)` into the general `q_od` formula, the `-a_jj`/
`+a_jj` terms cancel exactly) and confirmed numerically (`f[j,j]` recovered this way
matches `derive_fjj_from_autarky_cutoff` to `1e-10` at 10 random points, D=4/seed=29).

The q-gravity pivot's affine offset (`build_q_gravity_offset`) substitutes
`melitz_log_f_from_q` into the f-gravity restriction `dot(c_full,vec(log f))=0`: **a real
bug was found and fixed live via the Section 5.5 round-trip test** -- an earlier version
omitted the `c_full[jj_lin]*q_jj` contribution to the full-`D^2`-cell restriction entirely
(since `q[j,j]` is excluded from the q-pivot's own free domain but its value still enters
the FULL-vector gravity sum), giving `gravity_residual_f ~ 0.0065` instead of machine
precision; fixed by including that term.

**Full cross-parameterization equivalence** (Section 5.5, 63 new assertions): starting
from the EXISTING `:logf` fixture, encoding the equivalent `:logcutoff` free vector,
expanding back, and comparing against the ORIGINAL `:logf` expansion and the TRUE
`gravity_residuals` (not a second copy of the same derived formula) -- `A`, `f`,
`gamma_prime`, `f[j,j]`, the full baseline cutoff matrix, BOTH gravity residuals, and the
`D^2+1`-column moment matrix `G` itself all agree to `1e-8`-`4.4e-16` (`G` itself: `max|G_f
- G_q| = 4.4e-16`, machine precision). This satisfies the governing prompt's own
precondition ("no comparison of outer solver behavior is meaningful until these
fixed-point equivalence tests pass tightly") for any future live comparison.

Full suite including Section 5: **536/536 passing** (up from 337 baseline), zero
regressions.

### 21.F What was NOT reached this session

- **Section 4's full grid**: only `delta=1e-2` (both directions) was matched-compared;
  `delta=1e-3` was not (Section 21.C's own machine-precision equivalence tests make a
  qualitatively different result there unlikely, but it was not run).
- **Live KNITRO wiring for `:logcutoff`** (Sections 6/9): an `outer_parameterization`
  switch analogous to `cutoff_constraint_backend`, needing its own `moments!` adapter
  (swap `expand_free_theta` for `expand_free_theta_logcutoff`) and its own q-space affine
  cutoff system (structurally easy given `affine_cutoff.jl`'s existing basis-probe
  machinery) -- not implemented. No claim is made about which parameterization performs
  better in a live search; that question requires this wiring.
- **Section 6** (gradient backends under `:logcutoff`, incl. the switch-calibrated
  q-bandwidth idea): blocked on the above, not started.
- **Section 7** (fine-grained per-category profiling instrumentation and allocation
  profiling at 4 representative points): not built. The only new instrumentation is
  coarse: `MelitzFiniteDeltaOuterResult` now carries `n_fc_calls`/`n_ga_calls` (total
  callback invocation counts), which supported Section 21.D's comparison but not a
  category-level breakdown.
- **Section 8** (restricted-search G/GA/GF/GAF comparisons, parameter-movement/
  activity-switch diagnostics): not run. The only available evidence is indirect, from
  Section 21.A's reproduction: incumbent `gamma_prime` differs from the gamma-only-profile
  roots quoted in the governing prompt (e.g. `delta=1e-2` upper: full-search `0.935445` vs
  gamma-only `~0.942938`), confirming (as the prior session already established) that
  nuisance coordinates move the bound non-trivially, especially at the lower direction --
  but WHICH coordinates do that work was not investigated further.
- **Section 10** (burst-based continuation strategy): not built. This session's own
  single-shot `theta_box=0.10`/`maxit=25` reproduction (Section 21.A) still terminates at
  `-410`/`-400` (iteration limit) in 3 of 4 cases, not `nStatus=0` -- unchanged from
  Section 20.F, since no continuation logic was added this session.
- **Section 11** (performance experiments guided by profiling): blocked on Section 7, not
  started.
- D=20 scaling and common-marginal restrictions: explicitly out of scope per the governing
  prompt, not attempted.

### 21.G Next priorities (ranked, based on what this session measured, not conjecture)

1. Wire `:logcutoff` into `solve_melitz_finite_delta_bound` (mechanically small given the
   validated Section 21.E core) -- unblocks Sections 6/9 directly.
2. Run Section 4's benchmark at the full `2 delta x 2 direction` grid.
3. Build the fine-grained Section 7 profiling instrumentation before attempting Section
   8's restricted-search campaigns -- the upper/lower wall-time asymmetry Section 17.G/
   20.G already flagged remains unexplained; Section 21.D's FC/GA/inner-solve counters are
   too coarse to diagnose it further.
4. Section 8's restricted-search comparisons (gamma-only vs GA vs GF/GQ vs full), seeded
   from the already-known gamma-only roots quoted in the governing prompt.
5. Section 10's burst-based continuation strategy, starting from the already-cold-verified
   incumbents this session reproduced.

Session commits: `2bfecaf` (Section 3/4/5 implementation + tests), `7f354ac` (docstring
accuracy fix). New files: `src/melitz/affine_cutoff.jl`, `src/melitz/log_cutoff_param.jl`,
`scripts/melitz_finite_delta_campaign.jl`, `scripts/melitz_cutoff_backend_benchmark.jl`.
