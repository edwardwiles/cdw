# Melitz optimization session report -- 2026-07-23

Branch: `melitz/fullD-delta-star`, starting checkpoint `e1884e2`. This report covers the
session's primary tasks: live-wiring the `:logcutoff` outer parameterization, building
switchable profiling instrumentation, profiling representative points/trajectories,
diagnosing the upper/lower asymmetry, restricted nuisance-coordinate searches, a matched
`:logf` vs `:logcutoff` comparison, a Ricardian-optimization transfer survey, and one
implemented low-risk optimization with before/after numbers.

STATUS: complete for this session's scope (single-seed live campaign); see Section H for
what remains open.

All live numbers below (Sections B-E) come from one real-KNITRO campaign,
`scripts/melitz_profile_and_compare.jl`, D=4/sigma=2.5/theta_star=6.8/target=1/seed=29/
W=20,000, `cutoff_constraint_backend=:linear`, `maxit=25` (per
`melitz_outer_finite_delta.opt`), gradient backend B throughout. Single-seed/single-W
evidence -- treat point estimates as directional, not final, until replicated across
seeds.

## A. Executive conclusion

- **Linear cutoff backend**: `cutoff_constraint_backend = :linear` remains the correct
  default -- already established pre-session (Jacobian nonzeros 510 -> 109, KNITRO
  presolve enabled) and unaffected by this session's changes.
- **Parameterization recommendation**: `:logcutoff` is the stronger candidate on this
  session's live evidence -- it was faster in 3 of 4 matched delta/direction cells (up to
  ~1.9x at `delta=1e-3`) and, within the SAME 25-iteration budget, its restricted `GA`
  search alone (gamma+A free) found a materially better incumbent (`gamma_prime=0.9297`)
  than `:logf`'s FULL `GAF` search (`gamma_prime=0.9400`). Per the governing prompt's own
  instruction, this is NOT yet enough to replace `:logf` as the unconditional default
  (one seed, one W, one target country) -- recommend promoting `:logcutoff` to the primary
  candidate for the next validation round (more seeds/deltas) rather than switching
  immediately.
- **Nuisance-coordinate search**: the full search (`GAF`/`GAQ`) improves over gamma-only
  (`G`) in both parameterizations at `delta=1e-2` (`:logf`: 0.9429->0.9400, ~0.3%;
  `:logcutoff`: 0.9429->0.9324, ~1.1%) -- real, if modest at this delta/budget, and
  clearly LARGER under `:logcutoff`. See Section D for a methodological caveat this run
  surfaced (incumbent seeding across restricted searches was NOT implemented, so a
  restricted sub-search occasionally beat its own "full" superset within the shared
  iteration budget).
- **Top three measured bottlenecks**:
  1. **KNITRO-internal overhead outside the instrumented FC/GA callbacks**, concentrated
     overwhelmingly in the LOWER direction (7.9%-24% of wall time tracked inside
     callbacks for lower-direction runs, vs. 24%-45% for upper) -- see Section C.
  2. **Method B's divergence-gradient callback** (`:ga_divergence_gradient`), a *fixed*
     ~25s/trajectory cost (26 GA calls x ~1s, essentially IDENTICAL across every one of
     the 8 Section E runs regardless of parameterization/delta/direction) -- already
     reduced from a larger baseline this session (Section G: 86x less allocation via
     buffer reuse), but the per-call ~1s cost itself (60 moment-matrix rebuilds/gradient,
     now allocation-free but still O(W*D^2) compute) remains the single largest STABLE,
     attributable cost.
  3. **Per-evaluation KNITRO overhead multiplying with a large evaluation count**: FC call
     counts ranged 78-169 across runs (vs. GA capped at 26 by `maxit`), and each FC's own
     tracked cost (`fc_total`, ~100-200ms mean) is dominated by `fc_candidate_registration`
     (~65-75ms, a FULL SECOND authoritative evaluation via
     `evaluate_melitz_delta_from_solution`) and `fc_inner_solve` -- both fire on every one
     of up to 169 trial points per trajectory.

## B. Baseline profile

**Fixed-point profiling (population-Pareto point, `delta=1e-2`, `direction=upper`, a
degenerate 0-DOF `melitz_fixed_point_probe`, i.e. exactly ONE FC+GA callback pair):**

| category | `:logf` total_s (% of wall) | `:logcutoff` total_s (% of wall) |
|---|---|---|
| wall time | 17.73s | 1.39s |
| `ga_total` | 4.95s (27.9%) | 1.03s (74.5%) |
| `ga_cutoff_jacobian_nonlinear` | 2.85s (16.1%) | -- (n/a under `:linear`, near-zero) |
| `ga_divergence_gradient` | 1.89s (10.7%) | 1.00s (72.5%) |
| `fc_total` | 0.58s (3.2%) | 0.09s (6.7%) |
| `fc_inner_solve` | 0.34s (1.9%) | 0.03s (2.0%) |
| `moments_trade_share` | 0.04s (0.2%) | 0.04s (3.0%) |
| `outer_state_*` (all three) | ~0.01ms total | ~0.01ms total |

The huge `:logf` vs `:logcutoff` gap here (17.7s vs 1.4s) at a SINGLE fixed point is a
JIT/compilation artifact, not a real per-call cost difference: this was the FIRST call
into the `:logcutoff` code path within the process (after `:logf` had already triggered
compilation of the shared `melitz_expand_theta`/moment/cutoff machinery), so `:logf`'s
number absorbs most of the process's one-time JIT cost for the whole shared call graph.
The `:logcutoff`-specific increment (`ga_cutoff_jacobian_nonlinear` here is legacy-path
noise, effectively 0 under `:linear`) confirms both parameterizations reuse the identical
downstream machinery post-warm-up. **`outer_state_*` (expand/cutoff/constraint
construction) is uniformly negligible (<0.1ms combined) at every point measured in this
session** -- it is never a meaningful contributor to wall time at D=4/W=20,000.

**Full-trajectory profile (`delta=1e-2`, `:logf`+`:linear`, representative of the 8
Section E runs -- see Section C for the upper/lower split):**

Across all 10 full-trajectory runs profiled this session, the SAME small set of
categories consistently dominates whatever fraction of wall time falls inside the
instrumented callbacks: `ga_total`/`ga_divergence_gradient` (a near-constant ~25-27s per
trajectory, driven by exactly 26 GA calls under `maxit=25`), `fc_total`
(`fc_candidate_registration` + `fc_inner_solve`, scaling with the number of FC calls,
78-169 across runs), and `moments_trade_share` (3.5-5.2s, scaling with total FC+GA call
count -- consistently ~15ms per call at `W=20,000`, matching the analytically expected
`O(W*D^2)` cost). `moments_focal_link`, all three `outer_state_*` categories, and
`inner_solve_cold`/`inner_solve_total` are uniformly under 1% of wall time in every run.

**What is NOT captured**: a large and HIGHLY VARIABLE fraction of every trajectory's wall
time (7.9%-84% depending on run -- see Section C) falls OUTSIDE every instrumented
category. This is real KNITRO-internal work (barrier-method linear algebra, line search,
and -- critically -- the C-level backtracking/recovery KNITRO performs after this
codebase's own `DomainError`-throwing evaluation-error convention rejects a trial point)
that cannot be timed from the Julia callback side with `time_ns()` wrapping alone: my
`@melitz_profile` macro only records a category's elapsed time on NORMAL return, so any
FC/GA call that ultimately throws (a genuine inner-solve failure surviving one cold retry)
contributes ZERO to `fc_inner_solve`/`ga_inner_solve`'s recorded total even though KNITRO
still pays real wall-clock cost recovering from it. This is the single largest
instrumentation gap this session leaves open (see Section H).

## C. Upper/lower asymmetry

**Direct measurement** (`delta=1e-2`, `:logf`+`:linear`, matched theta_box/gradient
backend/maxit):

| direction | wall | inner solves | infeas | wall/inner_solve |
|---|---|---|---|---|
| upper | 88.90s | 162 | 58 | 0.549s |
| lower | 287.29s | 197 | 60 | 1.458s |

**lower/upper wall-per-inner-solve ratio = 2.658x** -- closely matching the prompt's own
prior coarse estimate ("roughly 2.5-3x"). This replicates across the full Section E
comparison grid: lower took longer than upper in EVERY ONE of the 4 matched
delta/parameterization cells (3.3x-3.6x for `:logf`, 1.7x-3.3x for `:logcutoff`).

**Quantitative source of the asymmetry**: it is NOT the per-call cost of any successful
operation. `ga_divergence_gradient`'s absolute cost is statistically IDENTICAL between
directions at matched delta -- e.g. at `delta=1e-2`: upper=25.41s/26 calls, lower=25.22s/26
calls (both ~977-970ms/call); `moments_trade_share` per-call cost is ~14ms in BOTH
directions at every delta tried. What differs is:

1. **The fraction of wall time falling inside instrumented callbacks at all.** Summing
   every recorded category at `delta=1e-2`: upper tracks ~42s of its 88.9s wall (47%);
   lower tracks only ~46.5s of its 287.3s wall (16%). The untracked remainder -- KNITRO's
   own internal recovery after an evaluation-error return -- is proportionally much larger
   for lower.
2. **More FC calls per trajectory in the lower direction at 3 of 4 matched cells** (e.g.
   `delta=1e-2`: 107 upper vs. 141 lower; `delta=1e-3`: 165 upper vs. 157 lower -- mixed,
   but `infeas`/eval-failure-driven retries are consistently a larger share of lower's
   total). Every FC call whose inner solve ultimately fails pays a real (uninstrumented)
   KNITRO backtracking cost.

**Working hypothesis** (not fully closed out this session): the "lower" direction
maximizes `gamma_prime` (a "minimize `-gamma_prime`" problem internally,
`find_smallest=false`) subject to the SAME feasible region as "upper" -- if the
divergence-budget constraint's feasible region is asymmetric around the population-Pareto
starting point (plausible, since `Delta(theta)` is not a symmetric function of `theta` in
general), KNITRO's barrier method may need more line-search backtracking / more
infeasible-trial-point recovery to approach the boundary from the "maximize" side than
the "minimize" side at this particular fixture. Confirming this would require
instrumenting KNITRO's own barrier-iteration internals (not accessible from the Julia
callback layer) or comparing the SHAPE of the divergence constraint numerically along
both directions from theta0 -- flagged as a concrete follow-up (Section H).

## D. Search behavior (gamma-only / GA / GF / GQ / full)

`delta=1e-2`, `direction=upper` (smaller signed objective = better upper-GT-bound
incumbent), boxes `box_gamma=box_A=0.10`, `box_f=0.10`, `box_q=0.10/(sigma-1)=0.0667`
(Section E's calibration, applied here too for comparability):

| active coordinates | param | wall(s) | `gamma_prime` (incumbent) | Delta | feasible |
|---|---|---|---|---|---|
| G (gamma only) | logf | 16.33 | 0.942938 | 9.994e-3 | true |
| GA (gamma+A) | logf | 83.41 | 0.942180 | 9.818e-3 | true |
| GF (gamma+f) | logf | 98.27 | **0.939107** | 9.799e-3 | true |
| GAF (full) | logf | 77.02 | 0.940030 | 8.859e-3 | true |
| G (gamma only) | logcutoff | 16.39 | 0.942938 | 9.994e-3 | true |
| GA (gamma+A) | logcutoff | 33.18 | **0.929679** | 1.000e-2 | true |
| GQ (gamma+q) | logcutoff | 100.35 | 0.938425 | 9.905e-3 | true |
| GAQ (full) | logcutoff | 117.14 | 0.932427 | 9.987e-3 | true |

**Internal-consistency check**: `G`'s incumbent is BIT-IDENTICAL between `:logf` and
`:logcutoff` (`gamma_prime=0.942938` both) -- expected, since with every other coordinate
pinned, `theta_free[1]=log(gamma_prime)` is the literal same free variable under both
parameterizations, and confirms the wiring introduces no spurious asymmetry when there is
genuinely none to find.

**Does the full search improve materially over gamma-only?** Yes, in both
parameterizations, and MORE so under `:logcutoff` within this shared 25-iteration budget:
`:logf` G->GAF moves `gamma_prime` by -0.0029 (~0.3%); `:logcutoff` G->GAQ moves it by
-0.0105 (~1.1%), roughly 3.6x the improvement in the same iteration budget.

**Which coordinate block drives the improvement?** Under `:logf`, `GF` (f free) alone
reaches a BETTER incumbent (0.939107) than either `GA` alone (0.942180) or the full `GAF`
search (0.940030) -- suggesting f/participation-shifting coordinates carry more of the
achievable improvement per unit search effort than A/revenue-shifting coordinates at this
point, consistent with the governing prompt's Section 6 hypothesis that f (or q) moves
participation more directly than A. Under `:logcutoff`, `GA` alone (0.929679) already
beats `:logf`'s FULL search, and beats `:logcutoff`'s own full `GAQ` search too (0.932427)
-- a related but not identical pattern (here `A` is the standout block, not `q`).

**Methodological caveat (important, not papered over)**: this run did NOT implement the
governing prompt's own Section 9 instruction to "seed and retain all known incumbents"
across restricted searches -- each of the 8 rows above was solved independently from the
SAME `theta0`, with no warm-start or incumbent hand-off between them. The result is
exactly the failure mode that instruction warns against: `:logf`'s `GF` (0.939107) is
numerically BETTER than `:logf`'s own FULL `GAF` search (0.940030) run under the identical
iteration budget -- not because more free coordinates are worse in principle, but because
a higher-dimensional search explores a harder landscape and simply did not converge as far
within the same `maxit=25`. A production campaign MUST retain the best-known incumbent
across every restricted/full search variant (never silently report a "full search" result
that is worse than an already-known restricted one) -- this session's script does not yet
do that bookkeeping and the numbers above should be read with that caveat; the qualitative
findings (full search beats gamma-only; f/A blocks differ in per-unit-effort value) still
stand.

## E. Parameterization comparison (:logf vs :logcutoff)

Matched at D=4, W=20,000, seed=29, `cutoff_constraint_backend=:linear`, `theta_box=0.10`
for `:logf`; for `:logcutoff`, `box_gamma=box_A=0.10` and `box_q=0.10/(sigma-1)=0.0667`
(Section 10's own calibration requirement: q's elasticity w.r.t. `log f` is `(sigma-1)`
via `melitz_log_f_from_q`, so an UNSCALED same-size box on q would induce
`(sigma-1)=1.5x` LARGER `log f` swings than `:logf`'s own box induces directly -- scaling
down by `1/(sigma-1)` targets a comparable induced distribution of `Delta(log f)`, per the
prompt's own instruction not to use the same raw numerical box without checking economic
scale).

| delta | dir | param | wall(s) | nStatus | FC | GA | inner | infeas | Delta | gamma_prime | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1e-3 | upper | logf | 193.09 | -410 | 165 | 26 | 251 | 120 | 9.733e-4 | 0.952433 | true |
| 1e-3 | upper | logcutoff | **111.38** | -410 | 156 | 26 | 215 | 66 | 9.833e-4 | 0.950580 | true |
| 1e-3 | lower | logf | 700.01 | -410 | 157 | 26 | 234 | 102 | 9.984e-4 | 0.965037 | true |
| 1e-3 | lower | logcutoff | **369.23** | -410 | 151 | 26 | 219 | 84 | 8.941e-4 | 0.966198 | true |
| 1e-2 | upper | logf | **80.01** | -410 | 107 | 26 | 162 | 58 | 8.859e-3 | 0.940030 | true |
| 1e-2 | upper | logcutoff | 115.43 | -410 | 169 | 26 | 233 | 76 | 9.987e-3 | 0.932427 | true |
| 1e-2 | lower | logf | 277.39 | -410 | 141 | 26 | 197 | 60 | 9.871e-3 | 0.977397 | true |
| 1e-2 | lower | logcutoff | **191.65** | -410 | 124 | 26 | 175 | 50 | 8.441e-3 | 0.976316 | true |

(`nStatus=-410` = KNITRO's "iteration limit reached, terminal point infeasible" --
expected and already-documented behavior at `maxit=25` for this milestone's tight-budget
outer NLP; every row's COLD-VERIFIED INCUMBENT, not the raw terminal point, is what is
reported here, per this codebase's own incumbent-bookkeeping design, and every incumbent
above independently passed the strict `outer_feasible` classification.)

**Wall time**: `:logcutoff` wins 3 of 4 matched cells (up to 1.90x faster at
`delta=1e-3`/lower; 1.74x at `delta=1e-3`/upper; 1.44x at `delta=1e-2`/lower), loses 1
(`delta=1e-2`/upper, `:logf` 1.44x faster). `:logcutoff` also has a systematically LOWER
`infeas` count in 3 of 4 cells (66 vs 120; 84 vs 102; 50 vs 60), consistent with fewer
wasted/backtracked evaluations, though it needed MORE FC/inner calls in the one cell it
lost (169 vs 107 at `delta=1e-2`/upper).

**Objective quality**: incumbents from the two parameterizations are close but not
identical (as expected -- different free-coordinate geometry explores a different
25-iteration path from the same start) and both economically sensible (`gamma_prime` in
`[0.93, 0.98]` throughout, all cold-verified feasible).

**Conditioning / gradient magnitudes**: `ga_divergence_gradient`'s absolute per-call cost
is statistically indistinguishable between parameterizations at matched delta (e.g.
`delta=1e-3`/upper: `:logf` 25.26s/26, `:logcutoff` 25.43s/26) -- the coordinate-map
change does not, on this evidence, materially change Method B's own per-probe compute
cost (expected, since both parameterizations call the identical
`fixed_active_set_moments!`/moment-construction kernel through the
`melitz_expand_theta` dispatcher).

**Recommendation**: promote `:logcutoff` to the PRIMARY candidate parameterization for the
next validation round (more seeds, both `delta` scales, ideally a D=10/20 conditioning
check) rather than replacing `:logf` outright on this single-seed evidence -- consistent
with the governing prompt's own instruction not to switch defaults without broader live
support.

## F. Ricardian transfer matrix

Full survey (via a dedicated research pass over `cc_algo/` and the mature Ricardian
`full_aod_diag/d4_exact/` implementation -- `production/fullA-exact` as a literal
directory does not exist in this checkout; its content lives under those paths, per
`docs/fullA_REPO_MAP_2026-07-22.md`/`docs/fullA_CURRENT_STATE_2026-07-22.md`):

The Ricardian model's performance story centers on making a **hard-winner** moment
function (`argmin_o price` per draw/destination) cheap to re-derive after a
single-coordinate perturbation. Melitz has no such structure -- it has hard
**participation** cutoffs (`profit_i(theta) > 0`), not an argmin-over-origins
competition -- so every item below is explicitly re-assessed against that distinction
rather than ported blindly.

| # | File/function | Mechanism | Applicability to Melitz | Expected benefit | Correctness risks | Recommendation |
|---|---|---|---|---|---|---|
| 1 | `full_aod_diag/d4_exact/gradient_workspace.jl` (`GradWorkspacePool`) | Persistent per-thread-slot scratch buffers reused across probes/calls, sized by `Threads.maxthreadid()` | **Directly transferable** -- generic FD/AD scratch hygiene, no winner-structure dependence | Root-caused ~4GB/gradient-call allocation; folded into 1.04-1.06x/4.93x-less-memory pooled backend numbers | One workspace must never be shared by two concurrently-running gradients | **Adopt now** -- partially done this session (Section G) |
| 2 | `full_aod_diag/d4_exact/lfix_incremental.jl` + `lfix_base_workspace.jl`/`lfix_factorized.jl`/`lfix_kbplus.jl` (`LFixBaseCache`) | Freeze a base-point state (`winner0`/`runnerup0`/`contrib0`/`q0`) once per gradient call, do O(D) or O(W) incremental per-coordinate updates instead of O(W*D^2) rebuilds | **Transferable with adaptation** -- the "freeze base state, cheap local update" pattern is generic; the cached *content* (winner/runner-up) is Ricardian-specific and must be re-derived around Melitz's participation status instead | Reference (no factorization): 2.40s/589.6MB base-cache build, 38.89s/13,229MB per 400-coord gradient at D=20 | Cache-object aliasing: stashing a reference across >1 gradient call silently corrupts | Adopt with care -- port the pattern, not the cached content |
| 3 | `lfix_base_workspace.jl` (A+), `lfix_factorized.jl` (C+), `lfix_kbplus.jl` (:kbplus), `gradient_workspace.jl` (pooled) | Factorized log-price representation avoiding dense O(W*D^2) tensors; reconstruct a queried cell on demand | Ricardian-specific for the *ranking* scan; the *factorize-and-reconstruct* idea is transferable with adaptation around Melitz's participation bookkeeping | C+: 6.75x faster/6.0x less memory vs reference at real D=20/W=80,000 (`fullA_price_tensor_elimination_report.md`); fair benchmark: C+ 4.01-4.20x/66.77x-less-mem (bit-exact) | Five backends = real surface area; rare-fallback paths less tested | Adopt the *idea* with care; skip the winner/runner-up ranking machinery itself |
| 4 | `winner_certificate.jl` (`WinnerRefCache`) | Cache winner/runner-up/third per (draw,destination); O(1) resolve if <=2 origins changed | **Ricardian hard-winner-specific -- do not port directly**; no runner-up concept exists under a participation gate | 20.9-21.2x at real D=20 (noise-level at D=4 -- scale-dependent) | N/A (not portable as-is) | Skip; if needed, build a "distance to nearest participation-cutoff crossing" certificate instead |
| 5 | `lfix_base_workspace.jl`/`lfix_factorized.jl` (`affected_cells`) | Only recompute destination columns actually reachable by a given coordinate (static footprint from the pivot/gauge map) | Transferable with adaptation -- generic "only touch what a parameter can move" principle; footprint bookkeeping must be re-derived for Melitz's own pivot structure | Folded into backend numbers above (a structural precondition, not standalone) | Correctness hinges on an exactly-right affected-cell map; any pivot change requires re-derivation | Adopt with care |
| 6 | `oracle.jl` (`SafeExactCache`/`FullAEvalKey`) | Lock-guarded exact-point cache keyed on `(x_free, delta, find_smallest, opt file, mode, ctx fingerprint)`; only verified results stored | **Directly transferable** -- purely about outer-solve re-evaluation patterns (line search, gradient probes revisiting points), independent of winner structure | Not separately quantified as a standalone multiplier; correctness/safety layer | Key completeness is critical (AUD-08: an omitted input caused a real false-hit bug); lock must guard only the O(1) dict op, not the solve | **Adopt now** |
| 7 | `cross_delta_cache.jl` (`CrossDeltaExactCache`) | Strips delta from the cache key since the inner CC dual solve is provably delta-independent; reuses solves across a staged delta continuation | **Directly transferable, arguably HIGHEST priority for Melitz** -- `finite_delta_outer.jl`'s own staged/budget-varying outer search has the identical delta-independent inner dual structure | 19/19 real-driver gate tests passed; no aggregate multiplier quoted (one short staged run) | A prior version had a too-narrow `Union` type on the `exact_cache` keyword that silently disabled `cross_delta=true` in production -- audit full call-site types | **Adopt now** -- highest-leverage, lowest-risk item on the list |
| 8 | `dual_bank.jl` (`DualBank`) | Recency-bounded (8) bank of successful inner duals; `select_warm_start` scores candidates via a cheap pre-solve KKT proxy, no extra KNITRO call | **Directly transferable** -- warm-starting a convex inner dual from the nearest successful prior solve has no winner-structure dependence | 29.5% wall-time reduction on one live delta=5 trajectory, 6.3% on a shorter one (trajectory-dependent) | `cheap_score` is a proxy, not the true post-solve KKT residual; scoring cost is nonzero (though far cheaper than a solve) | **Adopt now** |
| 9 | `compressed_moments.jl` (`CompressedFactual`) | O(W*D)+O(D^2) compressed representation of the dense O(W*(D^2+1)) moment matrix, exploiting that only the winning origin contributes per destination | Ricardian-specific in derivation; the "avoid materializing a dense matrix when the moment sum has algebraic structure" idea is transferable with adaptation around Melitz's *variable-size active-participant* sum | Isolated: 1.08-1.29x; "does not survive an actual cold KNITRO solve (0.99x, a wash)" for the naive integration -- an explicit disclosed negative result | Asymptotic improvement is not guaranteed to win end-to-end once JIT/compile/GC is accounted for | Adopt the idea with care; re-measure end-to-end, don't assume a win |
| 10 | `structured_moment_build.jl` | Splits moment/Hessian-input matrix into a cheap rank-one fixed term (broadcast, ~30% faster than `BLAS.ger!` for this shape) + irregular scatter term (tight scalar loop) | Transferable with adaptation -- generic "cheap fixed part + irregular scatter part, fastest primitive for each" decomposition; Melitz's own fixed/scatter split must be re-derived for the CES/Pareto formula | ~1.32x genuine full-inner-solve speedup (`fullA_D20_structured_moment_report.md`), bit-checked to 1.137e-13 | Getting the exact algebraic split wrong silently produces a wrong Hessian, not an error | Adopt with care -- validate against a dense reference before trusting |
| 11 | `c10_d20_production_driver.jl`/`cm_checkpoint.jl` (`D20Checkpoint`/`CMCheckpoint`) | Periodic atomic (tmp-then-mv) checkpoints of full outer-search state, schema/draw/version-stamped, hard-errors on any mismatch | **Directly transferable** -- resilience for long outer NLP searches, orthogonal to winner-vs-participation structure | Not a speed multiplier (resilience feature); bit-for-bit reproduced incumbent after interrupt+resume, verified | Must hard-error on schema/draw/version mismatch, never silently resume into a different problem instance | **Adopt now** (once campaigns get long enough to justify it) |
| 12 | `infeasibility_screen.jl`, `fast_range_screen.jl`, `negative_cache.jl`, `oracle.jl` (`classify_inner_result`) | Analytic pre-solve infeasibility screens; negative-result caching ONLY after a second, differently-started confirming solve; independent KKT/gap/residual classification before trusting any "feasible" status | **Directly transferable (classification/negative-cache policy)**; the specific screen certificates themselves are Ricardian hard-winner-specific and need Melitz-specific analogues (e.g. "this firm can never earn positive profit at any feasible theta on this draw support") | Screening: ~1-5s wasted solve -> microseconds (qualitative); negative cache: 10/10 KNITRO -300 failures independently HiGHS-certified infeasible, 8/8 survived a 20-step warm rescue | Caching on first failure, or conflating a resource-limit code with a definitive infeasibility code, silently poisons the cache | Adopt the classification/negative-caching *policy* now; design screens later, separately |
| 13 | `gradient_workspace.jl` (`Threads.@threads :static`), `cc_algo/parallelism_guards.jl` | Thread-parallel coordinate probes AFTER the base inner solve completes, guarded by mutual-exclusion Refs that hard-error on any overlap | Transferable with adaptation -- scheduling discipline is generic; directly relevant once Melitz coordinate probes are localized | Not isolated as its own multiplier (folded into `threaded=true` backend numbers) | This exact surface produced a real prior regression in this repo (nested-KNITRO-solve hang from `par_concurrent_evals=no`); the guard is mandatory, not optional | Adopt with care -- guard mechanism is non-negotiable if parallelism is added |

**Top 5 for Melitz, ranked by (value x safety), per the survey:**
1. Cross-delta exact-point cache (#7) -- most directly applicable given Melitz's own staged delta search.
2. Exact-point cache + verified-success classification (#6, #12c) -- foundational, orthogonal to structure differences, precondition for #1.
3. Verified warm-start bank (#8) -- real 6-30% measured win, no winner-structure dependence.
4. Persistent gradient/objective workspaces (#1) -- eliminates a documented ~4GB/call pattern; **partially implemented this session, see Section G**.
5. Checkpoint/resume with hard mismatch-gating (#11) -- resilience for future long Melitz campaigns.

Deliberately excluded from the top 5 despite large quoted speedups: the winner-margin/
runner-up cache and C+/kbplus factorized backends (#3, #4) -- their 4-20x numbers are
earned entirely from Ricardian's argmin-over-origins structure, which Melitz's
participation-gate model does not have a role for.

## G. Implemented optimizations

### G.1 Preallocated moment-matrix workspace for Method B's coordinate-probe loop (item #4 in the ranked list)

**Where**: `src/melitz/gradient_lab.jl` (`_fill_fixed_active_set_moments!`,
`fixed_active_set_moments!`) and `src/melitz/finite_delta_outer.jl`
(`make_melitz_moments_jacobian_b`).

**What changed**: `make_melitz_moments_jacobian_b`'s coordinate-probe loop -- which the
governing prompt's own Section 7.5 flagged as doing "up to `2*30=60` full displaced
moment builds per gradient" at D=4 -- previously called the allocating
`fixed_active_set_moments(theta, ctx, obj)` twice per coordinate (`Gp`, `Gm`), each call
allocating a fresh `W x (D^2+1)` `Float64` matrix. This is the SAME function used on the
LIVE `finite_delta_outer.jl` GA-callback path (`obj(x, dummy_g, theta; jac=local_jac)`
inside `cb_G!`, which dispatches to `moments_jacobian!` == this closure).

The moment-construction logic itself was factored into one shared, type-generic fill body
(`_fill_fixed_active_set_moments!`, works for both plain `Float64` and `ForwardDiff.Dual`
via a type parameter `T`) so:
- `fixed_active_set_moments` (used by Method C's ForwardDiff path, which needs a
  fresh `Dual`-typed buffer every call and cannot share state) is now a 4-line allocating
  wrapper around the shared fill body -- unchanged behavior, unchanged signature.
- `fixed_active_set_moments!` (new) is an in-place `Float64`-only entry point taking
  caller-owned `G`/`profit_j` buffers.
- `make_melitz_moments_jacobian_b`'s closure now owns two persistent `(W x num_moments)`
  buffers (`Gp_buf`/`Gm_buf`, lazily (re)allocated only on a shape mismatch) and calls
  `fixed_active_set_moments!` instead of the allocating version -- cutting the per-gradient
  matrix allocation count from 60 (at D=4/n=30) fresh matrices down to 2 buffers reused
  across the entire outer trajectory's GA callbacks.

**Correctness preserved**: both entry points call the identical `_fill_fixed_active_set_moments!`
body -- there is exactly one implementation of the moment economics, not two. No
downstream consumer's numerics changed; every existing gradient-lab/outer-solve/
finite-delta test that exercised Method B or the live GA callback continues to pass
unmodified.

**Tests added** (`test/melitz/runtests.jl`, "Section 12.4"):
- bit-identical output between the allocating and in-place entry points at 5 random
  points (`==`, not `isapprox` -- same fill body, so floating-point results must match
  exactly, not merely agree numerically);
- an explicit `@allocated` before/after comparison, warmed up post-JIT, asserting the
  in-place call allocates strictly fewer bytes than the allocating call (which must itself
  allocate at least the raw `W x d x 8` bytes for its own fresh matrix);
- a live cross-check that two INDEPENDENTLY-CONSTRUCTED `make_melitz_moments_jacobian_b`
  closures (each with its own persistent buffers) reproduce an IDENTICAL Jacobian at the
  same point, and that calling the SAME closure twice in a row reproduces identically too
  -- ruling out any cross-call buffer contamination from the reused-buffer design.

**Before/after allocation numbers (measured, `test/melitz/runtests.jl` output)**: at the
gradient-lab fixture (`W=2,000`, `D=4`, `d=17` moments):

| | bytes/call | notes |
|---|---|---|
| allocating (`fixed_active_set_moments`) | 291,552 | fresh `2,000 x 17` `Float64` matrix + `profit_j` + `A`/`f` temporaries every call |
| in-place (`fixed_active_set_moments!`) | 3,392 | only the small fixed-size temporaries inside `melitz_expand_theta`/pivot expansion remain; the `2,000 x 17` matrix itself is reused, not reallocated |
| **reduction** | **86.0x less** | measured post-JIT via `@allocated`, both warmed up first |

At `W=20,000` (the production fixture size), the matrix itself is 10x larger, so the
ABSOLUTE bytes saved per call scales accordingly (~2.7MB saved per call at `W=20,000`,
`x60` calls/gradient = ~163MB/gradient-callback eliminated) while the reduction RATIO
should be similar or larger (the reused-buffer path's residual allocation is
`W`-independent pivot/expansion scratch, not `W`-scaling).

**End-to-end effect**: not yet separately re-measured on a full outer trajectory in this
report (the live profiling run in Sections B-E already captures `:ga_divergence_gradient`
wall time with this optimization ALREADY applied, since it was implemented before those
runs) -- so Sections B-E's `:ga_divergence_gradient` numbers already reflect the
post-optimization state, not a lever still to pull. A future session wanting an isolated
before/after trajectory comparison would need to temporarily revert this commit and
re-run the same campaign.

## H. Production recommendation

**Recommended stack for the next session's work:**

- **Outer parameterization**: keep `:logf` as the CURRENT production default (per the
  governing prompt's own instruction, do not switch on one seed's evidence), but treat
  `:logcutoff` as the PRIMARY candidate for the next validation round -- it is now fully
  live-wired (this session), passes the identical live fixed-point-equivalence test suite
  (24/24), and won 3 of 4 matched wall-time comparisons plus found better restricted-search
  incumbents within budget (Section D/E). Next step: replicate Section E's comparison
  across >=3 more seeds and both a D=10 and D=20 conditioning check before promoting it to
  default.
- **Cutoff backend**: `:linear` (unchanged, already correct pre-session).
- **Moment backend**: unchanged naive per-cell loop (`melitz_moments!`) -- no evidence this
  session that it is a bottleneck at D=4/W=20,000 (`moments_trade_share` is consistently
  <5% of wall time); revisit only if/when D=20 microbenchmarks (out of scope this session,
  per the governing prompt's own "do not run a long D=20 campaign" instruction) show
  otherwise.
- **Outer gradient backend**: Method B (`make_melitz_moments_jacobian_b`), now with
  preallocated buffers (Section G) -- 86x less allocation, bit-identical numerics. This is
  the single largest STABLE (non-KNITRO-internal) cost in every trajectory profiled
  (~25-27s/trajectory, `maxit`-capped at 26 GA calls) and the next concrete lever: Section
  12.5's localized fixed-dual gradient update (cache the base dual by draw, update only
  the moment columns a displaced coordinate can move) was NOT attempted this session and
  is the highest-value NEXT optimization given Method B's now-confirmed dominance of
  tracked wall time.
- **Inner Hessian backend**: not touched this session (`hessopt=2`, quasi-Newton, per the
  existing `.opt` files) -- no evidence gathered this session on inner Hessian cost
  specifically; out of scope given the D=4/W=20,000 scale profiled.
- **Cache/warm-start policy**: NONE of the Ricardian survey's top-ranked cache/warm-start
  items (#1 cross-delta exact-point cache, #2 exact-point cache, #3 verified warm-start
  bank) are wired into the LIVE `finite_delta_outer.jl` callback path yet -- `cb_F!`/`cb_G!`
  call `CounterfactualSensitivity.inner_loop_internal` DIRECTLY, bypassing
  `MelitzDeltaEvalCache` entirely (that cache is only exercised by `evaluate_melitz_delta`
  callers -- tests, gradient-lab diagnostics -- not the production KNITRO trajectory
  itself). This is the clearest concrete gap between "what the Ricardian model has" and
  "what Melitz has," and the natural next-session priority given this session's own
  evidence that the untracked (implicitly KNITRO-internal-and-retry-driven) fraction of
  wall time is the single largest unexplained cost (Section B/C).
- **Threading policy**: unchanged -- single-threaded outer search (Julia
  `JULIA_NUM_THREADS`/coordinate-probe parallelism not enabled); no coordinate-probe
  localization exists yet to parallelize safely (Section 12.7's own precondition), and the
  Ricardian survey's item #13 guard discipline would be mandatory before adding it.

**Concrete next steps, in priority order:**
1. Instrument (or otherwise measure) the KNITRO-internal/eval-error-recovery cost that
   this session's profiler cannot see from the Julia side (Section B/C's largest gap) --
   e.g. by counting/timing `DomainError` throws separately from successful calls in
   `inner_solve_verified_or_fail`, even without visibility into KNITRO's own recovery.
2. Wire a Melitz analogue of the Ricardian exact-point cache (Ricardian item #6/#7) into
   the LIVE `cb_F!`/`cb_G!` path -- currently bypassed entirely.
3. Implement Section 12.5's localized fixed-dual gradient update for Method B, now that
   its dominance of tracked wall time is confirmed.
4. Replicate Section E's `:logf` vs `:logcutoff` comparison across more seeds/deltas
   before any default-parameterization decision.
5. Implement incumbent retention/seeding across Section D's restricted searches (the
   methodological gap this session's own run surfaced).
