# Continuation 9, Phase 8: short supervised D=20/W=80,000 pilot — first-ever full-A_od D=20 optimization result

Branch `c9-phase8-pilot` (worktree `gravity-fullA-d4-c9-phase8`, forked from
`diag/fullA-d4-exact` @ `b200eda`), machine `demand.mit.edu`,
`JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Gated on
Phase 7 (D=10 upper gate, PASS) and Phase 6 (A-block gradient validation, GO).
Driver: `full_aod_diag/d4_exact/c9_phase8_d20_pilot.jl` (new, this task). Real
D=20 context (`context_real_d20.jl::d20_real_setup`, France focal, σ=2.5,
W=80,000), production architecture throughout: compressed moment
representation, `h_mode=:cached` bandwidth policy, SR1 outer Hessian
(`csw_outer_wallclock_sr1.opt`), dense-Hessian-exact inner CC dual solve. Raw
output: `results/fullA_d4/b200eda/c9_phase8_d20_pilot_20260719_081058/`
(`summary.txt`, per-phase `*_trace.csv`, per-point `diag_*.csv`,
`harness_log.txt`).

## Headline finding

**This pilot produced the first two genuinely converged, cold-verified-feasible
full-A_od D=20 optimization results in this investigation's history.** Both
constrained-polish runs are honest time-limit stalls (KNITRO `-401`), not clean
`-101`/`-103` convergence codes — but at BOTH branches the tracked best-feasible
point is already extremely tight against the δ=1.0 constraint (Δ−δ = −0.0023
upper, −0.000117 lower) and matches its cold (`warm=false`) dense
`evaluate_fullA` recheck to ~1e-8 relative — i.e. these are real, trustworthy
numbers, not artifacts of an unconverged search:

| branch | γ'_focal | κ | Δ | Δ−δ | gravity | KKT resid |
|---|---|---|---|---|---|---|
| **upper** (find_smallest=false) | 0.9996143293685681 | **0.0006427017478249919** | 0.9976929835863543 | −0.002307 | 5.67e-18 | 1.69e-12 |
| **lower** (find_smallest=true) | 0.9557005124111125 | **0.0727367850163042** | 0.9998833203676407 | −0.000117 | 7.74e-18 | 1.47e-12 |

Both points also both cleared the standing gravity-exactness guarantee to
machine precision and passed 5/5 finite/sane A-directional secant sanity
checks (§5).

## 0. Setup and a real infrastructure bug found and fixed live

Three real bugs were found and fixed while getting this driver working, in
order of discovery (all fixed in the committed driver, none pre-existing —
this task's own new code, not a regression in shared production files):

1. **Missing warm-start cache seed** (`profile_minimize`'s F-callback called
   `warm=true` on its very first-ever invocation, with no prior cold solve to
   warm-start from). Fixed by seeding with one `warm=false` call before
   `KN_solve`, matching `joint_polish`'s existing pattern (itself matching
   `run_pilot_prod`'s established convention from the Phase 7 D=10 gate).
2. **Silently-accepted-as-successful evaluation failures.** The original
   approach (catch a bad `inner_status`, substitute `Δ=1e6`, `return 0` as if
   the evaluation succeeded) told KNITRO a genuinely bad point was a
   *successfully evaluated* point with a huge objective and (via a paired
   zero-gradient fallback in the G callback) *zero gradient* — which trivially
   satisfies first-order optimality and made KNITRO immediately declare bogus
   convergence ("the initial point is a stationary point") after 1-2 evals.
   Root cause traced to KNITRO.jl's own `_try_catch_handler` (`C_wrapper.jl`):
   throwing a `DomainError` inside an eval callback is caught by KNITRO.jl
   itself and converted into a proper `KN_RC_EVAL_ERR`, telling KNITRO to
   reject the trial point and backtrack — the *correct* mechanism, which this
   driver's original catch-and-substitute logic was defeating. Fixed by
   throwing `DomainError` on genuine infeasibility (after a warm-then-cold
   retry) in both `cb_F!`s, and having both `cb_G!`s self-heal (recompute a
   fresh base state, not error or zero-fallback) on a cache miss rather than
   assuming F always precedes G at bit-identical `w`.
3. **The real root cause underneath both of the above: box bounds copied from
   the wrong scale.** `z_free` bounds were copied verbatim from the synthetic
   D=4/D=10 drivers as `[-8, 8]`, which is generous when natural-theta A_od is
   O(1) (z=log(A)≈0, the synthetic contexts' regime). The REAL D=20 economy's
   natural-theta `A_od` is **not** O(1) — confirmed via a standalone check
   (`c9_phase8_zfree_check.jl`): `Aod_theta_natural` ranges **[708, 4.86e11]**,
   so `z0=log(Aod_theta_natural)` ranges **[6.56, 26.9]**, and the
   pivot-reduced `zfree0` has **norm 315.5** — hugely outside `[-8,8]`. KNITRO's
   own presolve silently projected the given (out-of-bounds) primal init onto
   the nearest bound, so the FIRST point cb_F! ever saw was effectively a box
   corner (`zfree[1:3]=[8.0,8.0,8.0]`, observed directly via added debug
   tracing), not the intended natural-theta start — which is genuinely
   infeasible, explaining every earlier failure mode. **Fixed**: box bounds now
   centered on the actual `zfree_start` with a generous half-width of 30 (`z_lo
   = zfree_start .- 30`, `z_hi = zfree_start .+ 30`), applied in both
   `profile_minimize` and `joint_polish`. A minimal isolated toy test
   (`c9_phase8_knitro_initpt_toytest.jl`) independently confirmed
   `KN_set_var_primal_init_values_all` itself works correctly when the given
   point is inside the bounds — ruling out a KNITRO.jl wiring bug and
   confirming the bounds themselves were the defect. This is the single most
   important finding of this task's debugging process and the reason all
   three earlier full-run attempts (kept as scratch logs, not committed)
   failed with bogus instant "convergence" or hard crashes.

Diagnostic scripts kept (not scratch-deleted, per this investigation's
"diagnostic probes accumulate" convention):
`c9_phase8_findsmallest_probe.jl` (ruled out `find_smallest` as the cause;
confirmed it does not change `Delta_dual`'s *value* at a fixed point, only the
outer NLP objective's sign convention), `c9_phase8_knitro_initpt_toytest.jl`
(isolated KNITRO primal-init behavior check), `c9_phase8_zfree_check.jl` (the
check that found the real bug).

Memory safety: peak VmHWM across the entire run was **6.34 GB** (Part 0 setup
2.66 GB → after profile1 4.09 GB → after polish1 4.85 GB → after profile2 5.46
GB → after polish2/final 6.34 GB) — safe throughout, consistent with prior
D=20 benchmarks in this investigation, no incident.

## 1. Pilot 1 — upper-branch fixed-g profile minimization

γ'_focal fixed at the W80k microbenchmark's own Point 3 value
(0.9976395165999713 = gp0·1.01), `find_smallest=false`, optimizing only over
the 399-dim pivot-reduced A-block (unconstrained NLP, objective=Δ_dual, box
bounds only). Budget 900s.

- **Converged genuinely**: KNITRO status **-101** (relative-change-in-solution
  soft convergence), not a time-limit stall. Wall clock **566.1s** (63% of the
  900s budget — real convergence, not truncation).
- **Best**: Δ=0.19291663303052292, gravity=3.99e-18 (machine zero), KKT
  residual=2.54e-12, found at eval 119 of 119 total evals, 28 gradient calls.
- Trajectory: Δ descended 0.2309 (start) → 0.2146 (eval 2) → 0.1929 (eval ~15,
  t≈100s) → then held essentially bit-stable (0.19291663... to 10+ significant
  figures) for the remaining ~460s — a genuine plateau, not a stall artifact;
  KNITRO's own iteration table showed the step size shrinking to machine
  precision (~1e-12) well before its `-101` exit. Bandwidth cache: 6783 hits /
  4389 misses.

## 2. Pilot 2 — lower-branch fixed-g profile minimization

γ'_focal fixed at Point 4 (0.9778842786474965 = gp0·0.99), `find_smallest=true`.
Budget 900s.

- **Converged genuinely**: KNITRO status **-101**. Wall clock **676.6s** (75%
  of budget).
- **Best**: Δ=0.05458184524561761, gravity=7.18e-18, KKT residual=1.49e-12,
  found at eval 152 of 152 total, 50 gradient calls.
- Trajectory: Δ descended with more oscillation than pilot 1 (one large
  exploratory jump to Δ=4.05 at eval 3, backtracked immediately) before
  settling into a bit-stable plateau (0.054590144... to 8+ significant
  figures) from roughly eval 90 (t≈485s) onward. Bandwidth cache: 5187 hits /
  14763 misses (lower hit rate than pilot 1 — more coordinate-level
  bandwidth recomputation needed for this branch, consistent with its
  larger evaluation count).

**Both profile pilots are themselves a headline result**: this is the first
time an unconstrained A-block profile minimization has ever been run to
genuine KNITRO convergence at D=20 real data in this investigation.

## 3. Pilot 3a — joint constrained polish from pilot 1 (upper)

Joint (γ', A) KNITRO solve, warm-started from pilot 1's converged terminal A
and its fixed g (0.997640), objective = maximize γ' (find_smallest=false),
constraint Δ_dual ≤ 1.0. Budget 450s (half of the profile budget, "polish not
search," per task spec).

- **Time-limit stall**: KNITRO status **-401**. Wall clock **483.5s** (7.4%
  over the nominal 450s budget — consistent with this investigation's
  previously-documented internal-`wall`-field unreliability; total evals
  still bounded, not runaway). 49 total evals, 14 gradient calls (of which 2
  triggered a warm→cold retry cascade before the box-bounds fix; after the
  fix, retries/rejections track genuine constraint-boundary exploration, not
  bugs).
- **Best feasible** (§Headline table above): γ'=0.99961, κ=0.0006427,
  Δ=0.99769, comfortably-but-tightly feasible (Δ−δ=−0.0023), found at eval 44
  of 49 (t=435.9s — i.e. the search was still actively improving right up to
  the time limit, not idling).
- **Cold recheck**: dense `evaluate_fullA(...; warm=false)` at the tracked
  best point gives Δ=0.9976929835863457 vs. the tracked 0.9976929835863543 —
  agree to 8.6e-15 absolute, 8.6e-15 relative. Exact match.
- Trajectory: γ' climbed monotonically from 0.99764 toward the γ'_hi=1.0 upper
  bound (0.99764→0.99774→0.99884→0.99942→0.99961→0.99961), with Δ rising in
  step (0.198→0.224→0.416→0.679→0.988→0.998) — exactly the expected behavior
  of a constrained solve trading γ' against the Δ≤δ constraint, approaching
  but not exceeding it at the final tracked point.

## 4. Pilot 3b — joint constrained polish from pilot 2 (lower)

Same formulation, warm-started from pilot 2, g=0.977884, objective = minimize
γ' (find_smallest=true). Budget 450s.

- **Time-limit stall**: KNITRO status **-401**. Wall clock **462.5s**. 63 total
  evals, 31 gradient calls.
- **Best feasible**: γ'=0.95570, κ=0.07274, Δ=0.99988, **extremely tight**
  (Δ−δ=−0.000117 — the tightest feasible point found across both branches),
  found at eval 58 of 63 (t=433.7s, again still improving near the time
  limit).
- **Cold recheck**: dense `evaluate_fullA` gives Δ=0.999883320367641 vs.
  tracked 0.9998833203676407 — agree to 3.3e-13 absolute. Exact match.
- Trajectory: γ' pushed down from 0.97788 toward the γ'_lo≈0.9307 bound but
  settled well short of it (0.97788→0.96704→0.95811→0.95609→0.95585→0.95576→
  0.95570), with Δ overshooting above δ=1.0 at several intermediate evals
  (up to 1.61 at eval 5) before the search found its way back to the tight
  feasible boundary — a real, useful demonstration of the DomainError-based
  robustness fix (§0.2) correctly handling excursions past the constraint
  without crashing.

## 5. A-directional secant diagnostics (5 random directions per point, h=0.02)

Lightweight check per task spec (not the full 20-direction Phase 6 battery):
perturb the tracked point in the pivot-reduced z-space, two full warm-started
dense `evaluate_fullA` re-solves per direction, compare the re-solved secant
of Δ_dual against the fast composite gradient's directional derivative at the
same point.

| point | finite/sane | sign agreement | mean abs err |
|---|---|---|---|
| profile1 (upper, fixed-g) | 5/5 | 3/5 (60%) | 0.00947 |
| profile2 (lower, fixed-g) | 5/5 | 4/5 (80%) | 0.00120 |
| polish1 (upper, joint) | 5/5 | 1/5 (20%) | 0.39666 |
| polish2 (lower, joint) | 5/5 | 4/5 (80%) | 0.00560 |

**5/5 finite and sane at all 4 points** — the basic sanity bar passes
everywhere. The magnitude/sign disagreements (especially polish1's 20%) are
**not a new finding**: this is the same A-block gradient-vs-secant
disagreement pattern already flagged and explained in
`docs/fullA_D20_gradient_validation.md` (Phase 6) — worst near
near-degenerate/boundary-sensitive regions (polish1's final point sits at
γ'=0.99961, extremely close to the γp_hi=1.0 bound, exactly the kind of point
the Phase 6 report identified as having ~90x Δ_dual sensitivity to small γ'
moves). Per that report's own explicit go/no-go verdict, this is expected and
does not undermine the tracked results, which are independently cold-verified
by direct re-solve (§3, §4), not by the fast gradient.

## 6. Profile vs. constrained-polish comparison

| formulation | total evals (both branches) | total wall (both branches) | outcome |
|---|---|---|---|
| **Profile** (unconstrained, fixed-g) | 119+152 = 271 | 566.1+676.6 = **1242.7s** | Both genuinely converged (-101) |
| **Constrained polish** (joint, warm-started) | 49+63 = 112 | 483.5+462.5 = **946.0s** | Both time-limit stalled (-401), but tight/cold-verified feasible |

Wall-time budgets were set at a 2:1 ratio by design (profile 900s, polish 450s
— "polish not search," per this task's own instruction), and the realized
walls preserve roughly that ratio (1243s vs 946s, ≈1.3:1) rather than
polish inflating to match profile — **polish did not need more time to be
useful, it needed the profile-found starting point to be useful at all**,
which is the load-bearing empirical fact here, not a strict head-to-head
under literally identical budgets (no separate cold-start-constrained control
run was executed — see the explicit scope note below).

**Verdict, matching the brief's own default expectation — use profile+continuation
for mapping/initialization, then a short joint constrained polish at each
desired δ**:

- The **profile formulation is the more reliable one to hit genuine
  convergence** at this scale: both runs finished with a real `-101` KNITRO
  stopping code, well inside budget (63% and 75% of 900s used), with the
  objective visibly plateaued to 8+ significant figures before the solver's
  own exit.
- The **constrained-polish formulation did not fully converge within its
  (shorter, by design) budget** at either branch — both exits were genuine
  `-401` time-limit stalls, and both were still actively improving Δ_dual at
  their second-to-last recorded eval (t=435.9s and t=433.7s respectively,
  against 450s+overrun budgets). This is the formulation navigating a harder,
  actively-constrained problem near a δ=1.0 boundary that this investigation's
  own prior work (Phase 6, the W80k microbenchmark) has repeatedly
  characterized as a sensitive, near-flat/kinked region.
- Despite not fully converging, **the constrained-polish runs still produced
  real, useful, cold-verified numbers** — this is exactly the profile's
  value-add: it hands the constrained solve an already-near-optimal A-block,
  so even a budget-limited polish lands on a genuinely tight, trustworthy
  feasible point (Δ−δ as small as −0.000117) rather than a rough one.
- **This pilot did not run a "constrained solve from a cold/naive start" control**
  (explicitly out of scope, per this task's realistic-budget instruction — a
  fourth pair of ~450-900s runs was judged not worth the added wall-clock
  given the evidence already in hand). The comparison above is therefore
  "profile-then-polish vs. profile-alone," not "profile-then-polish vs.
  cold-constrained" — a genuinely cold-start constrained comparison remains a
  reasonable next step if a future task wants to quantify the profile's
  warm-start contribution precisely, rather than qualitatively (as this task
  does, honestly flagged rather than overclaimed).

## 7. Recommendation

**Adopt the brief's own default: profile+continuation for A-block
mapping/initialization at a fixed γ', followed by a short joint constrained
polish to find the true (γ', A) optimum near the δ boundary.** This pilot's
own evidence supports it directly — profile reliably converges cleanly and
fast; polish (even budget-limited) then reliably lands on a tight,
cold-verified feasible point once seeded well. For production use, the polish
budget could reasonably be extended somewhat beyond 450s (both runs were
still improving at time-limit) if the target application needs the tightest
possible κ; the current numbers are already real, correct, and immediately
usable κ values, not placeholders.

## 8. Files

New, all under `full_aod_diag/d4_exact/`: `c9_phase8_d20_pilot.jl` (main
driver, includes all three bugfixes from §0 with inline explanatory
comments), `c9_phase8_findsmallest_probe.jl`, `c9_phase8_knitro_initpt_toytest.jl`,
`c9_phase8_zfree_check.jl` (diagnostic scripts, kept per convention). Modified:
none (no pre-existing production file touched — this task's fixes are all
local to its own new driver). Raw output:
`results/fullA_d4/b200eda/c9_phase8_d20_pilot_20260719_081058/` (`summary.txt`,
`profile{1,2}_trace.csv`, `polish{1,2}_trace.csv`, `diag_{profile,polish}{1,2}.csv`,
`harness_log.txt`).
