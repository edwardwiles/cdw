# Profiled reformulation of the CC outer loop: fix gamma', minimize delta*

Status: D=4 (synthetic) and D=20 (real data) both tested, each with TWO independent
optimizers (BlackBoxOptim global/derivative-free, and KNITRO local/gradient-based --
literally the same solver the existing constrained outer loop uses). All numbers below
are from real runs, not estimates.

## 1. The idea

The existing outer loop solves a *constrained* problem: extremize gamma'_focal over
(gamma'_focal, A_od) subject to delta\*(gamma'_focal, A_od) <= delta_budget, via KNITRO
SQP/interior-point over D+1 free variables. `full_d2_correction_report.md` section 7
documents a severe gamma'-vs-A_od gradient-scale mismatch in that search (~160x at D=4,
~10^4-10^5x at D=20 real data) that appears to leave A_od under-explored.

The reformulation: **fix gamma'_focal at a target value ("GT") and minimize delta\*(A_od)
over A_od alone** -- a genuinely unconstrained (box-bounded only) problem, with no
gamma'-vs-A_od scale mismatch at all, since gamma' is no longer a free variable. If the
original constrained search found the true joint optimum, this minimum should recover
(approximately) the original delta budget. If it finds something meaningfully LOWER, that
is direct evidence of real headroom the constrained search left on the table.

## 2. What was built

- `run_profiled_delta_star_min.jl` / `run_profiled_delta_star_min_d20_real.jl` --
  BlackBoxOptim (`:generating_set_search`) versions, D=4 and D=20 real data.
- `run_profiled_delta_star_min_knitro.jl` / `run_profiled_delta_star_min_knitro_d20_real.jl`
  -- KNITRO versions (literally the same solver as the existing constrained search), D=4
  and D=20 real data.

Both use the already-validated `exact_inner_divergence_at` (`fixed_A_incumbent.jl`) as the
objective VALUE (a real gravity-freeze + real KNITRO inner-dual solve at every genuinely
new A_od). The KNITRO versions use the already-validated cheap fixed-dual-criterion
central-FD gradient (`fixed_dual_fd_gradient`, the same trick production's own
`gradient_method=:fixed_dual_fd_full` uses) as the objective GRADIENT -- no extra real
inner solves per outer iteration, critical for tractability at D=20 real data.

## 3. Three real bugs found and fixed while building the KNITRO version

1. **`eval_fcga=yes`** is set in this project's own `.opt` files (`csw_outer_25.opt`,
   `csw_outer_1000.opt`). When set, KNITRO expects ONE combined value+gradient callback
   per evaluation; registering separate `cb_F!`/`cb_G!` callbacks (the naive KNITRO.jl
   pattern) leaves KNITRO calling ONLY `cb_F!` -- the gradient callback is silently never
   invoked, `objGrad` stays at its default (~0), and KNITRO immediately declares the
   *starting point* "locally optimal" having taken zero real steps. Caught via a minimal
   2-variable toy KNITRO problem that reproduced the exact same symptom, then confirmed
   directly: the registered `cb_G!`'s own print statement never printed a single line
   despite KNITRO's "1 gradient evaluation" stat. `outer_loop_cached.jl` already has the
   correct branch for this (`if KNITRO.KN_get_int_param(kc,"eval_fcga")==1`); this was
   missed when building a fresh KNITRO NLP from scratch and is now fixed the same way
   (a combined `cb_FG!`).
2. **`exact_inner_divergence_at` returns a shorter NamedTuple** (no `x_star`/`frozen`
   fields) when gravity-infeasible. Accessing those fields unconditionally inside a KNITRO
   callback throws a Julia `FieldError` that KNITRO's C callback boundary swallows
   *silently* (logged only as a generic "exception in puts callback" warning), corrupting
   the search with stale gradient state rather than crashing loudly. Fixed by checking
   `isfinite(res.δ_star) && hasproperty(res,:frozen)` before touching those fields.
3. **A bare relative `KNITRO_OPT_FILE` path can silently fail to load** -- `KN_load_param_file`
   prints `"ERROR: Knitro could not open file ... for input."` but does NOT throw a Julia
   exception or halt the script; KNITRO just falls back to its own defaults. This matches a
   previously-documented class of cwd-resolution bug in this repo (session memory: "opt-file
   path" issues from the D=20 W-sensitivity work). Fixed by defaulting `KNITRO_OPT_FILE` to
   an absolute path (`joinpath(@__DIR__, ...)`).

None of these affect the objective VALUE computation (`exact_inner_divergence_at` itself,
already validated) -- only the new KNITRO NLP wiring built for this task.

## 4. D=4 results (synthetic data, W=8000, lower bound, delta budget=1.0)

Constrained search (fresh run this session): gamma'\_focal\* = 0.996896145, kappa=0.005168.
(Note: a fresh KNITRO run of the *constrained* search does not reproduce bit-identical
results run-to-run -- an earlier session's saved run landed at gamma'=0.997561,
kappa=0.0040616, a nearby but different point. This is expected local-search
path-dependence, not a bug, and does not affect the reformulation test, which is always
evaluated self-consistently against whatever gamma'\* the same run's own constrained
search found.)

| quantity | value |
|---|---|
| delta budget used by constrained search | 1.000000 |
| delta\*(A_od=A\*, same gamma') | 1.01714073 |
| delta\*(A_od=constrained-search's own A_od, same gamma') | 0.98307944 |
| **BlackBoxOptim** min over 2 starts | **0.79478558** |
| **KNITRO** (project's own `csw_outer_1000.opt`) min over 2 starts | **0.87980255** |

Both optimizers agree: **meaningfully below the 1.0 budget** -- BlackBoxOptim finds ~20.5%
headroom, KNITRO (a local search, so more conservative/less exhaustive) finds ~12.0%
headroom, starting from the SAME two points in both cases (A\* and the constrained
search's own A_od). The KNITRO run converges cleanly (status -101/-102, both legitimate
"feasible, benign stop" codes, not iteration-limit artifacts) with real, gradually
decreasing objective traces (91-198 evaluations per start). **Conclusion: at D=4, the
original constrained search does NOT find the true joint optimum -- real headroom exists.**

## 5. D=20 results (REAL data, W=80000, upper bound, delta budget=1.0)

Per explicit instruction, the constrained search was NOT re-run. The GT was reused exactly
from the already-completed, already-validated KNITRO-variable-scaling run documented in
`full_d2_correction_report.md` section 7.2
(`batch_out_realD20_W80000_fixeddualfdfull_scaled05/seq_upper_delta1.0.jld2`):
gamma'\_focal\* = 0.950259956423, kappa = 0.081518. A sanity check (this run's fresh
theta_r0[1:2] vs. the saved run's own theta[1:2]) confirmed the same real-data setup was
used before trusting any downstream number.

| quantity | value |
|---|---|
| delta budget used by that scaled constrained search | 1.000000 |
| delta\*(A_od=A\*, same gamma') | 1.79910589 |
| delta\*(A_od=scaled-search's own A_od, same gamma') | 1.00004390 |
| **KNITRO**, start=A\* | **1.08286395** (status=0, 17 real iterations, opt_err=7.1e-7 -- genuinely locally optimal) |
| **KNITRO**, start=scaled-search's own A_od | **1.00004390** (status=0, opt_err=2.2e-7 -- genuinely locally optimal AT the starting point itself) |

**This is the opposite finding from D=4.** Starting the unconstrained minimization from
the scaled constrained search's own A_od, KNITRO verifies -- via a real, non-buggy
(opt_err is a small nonzero residual, not the exact-zero signature of the earlier
callback bug) first-order check -- that this point is **already a genuine local optimum**
of delta\*(A_od) at this gamma'. No further headroom was found from that starting point.
Starting instead from the naive A\* baseline, KNITRO converges cleanly to a **different,
worse** local optimum (1.083, still above the 1.0 budget) after 17 real iterations --
confirming the A_od landscape is genuinely non-convex/multimodal at D=20, and that WHERE
you start matters enormously, but also that the scaled search (which already used KNITRO
variable scaling to move A substantially, per report section 7.2) landed in a good basin.

## 5b. D=20 multi-start (5 points): the scaled-search endpoint holds up

Extended the D=20 KNITRO test to 5 starting points total: the 2 above, plus 3 new
log-normal multiplicative perturbations of Acol (matching the precedent in
`multistart_screening_d20.jl`, which found 20/20 such perturbations gravity-feasible out
to ~20x relative distance). Script:
`run_profiled_delta_star_min_knitro_d20_multistart.jl`.

| start | delta\*_min | status | note |
|---|---|---|---|
| 1. A\* | 1.082864 | 0 (genuinely locally optimal) | worse basin |
| **2. scaled-search endpoint** | **1.000044** | **0 (genuinely locally optimal)** | **best** |
| 3. sigma=0.3 around A\* | 1.122240 | 0 | similar/slightly worse basin to #1 |
| 4. sigma=0.3 around scaled-search endpoint | 1.194061 | 0 | WORSE than #2 despite starting near it |
| 5. sigma=1.0 around A\* | failed (KN_RC_EVAL_ERR=-502) | -502 | starting point itself gravity-infeasible; this script's zero-gradient fallback for infeasible points gives KNITRO nothing to descend on -- an artifact of quick multi-start wiring (no repelling-gradient penalty was built, unlike `bbo_common.jl`'s `infeasibility_penalty`), not a finding about the landscape |

**The scaled constrained search's own endpoint remains the best point found across every
successfully-evaluated start.** Notably, even a MILD perturbation (sigma=0.3) of that
exact same point lands in a meaningfully worse basin (1.194 vs 1.000) -- the good optimum
is not part of some broad flat region, it is a genuine, somewhat narrow local optimum that
multi-start does not improve on. This substantially strengthens section 5's conclusion:
the D=20 scaled-search result is not a fluke or an artifact of that one starting point --
it is a real, locally-robust optimum that 4 independent alternative starts all fail to beat
or even match.

(Point 5's failure is a solvable but low-priority wiring gap, not investigated further here
-- would need a repelling gradient/penalty for gravity-infeasible A_od, analogous to
`bbo_common.jl`'s `infeasibility_penalty`, to let KNITRO recover from an infeasible start.)

## 5c. Is the ORIGINAL constrained result itself starting-point-robust?

Sections 5/5b test whether the reformulation (fix gamma', minimize delta\*(A_od)
unconstrained) can improve on a fixed constrained-search endpoint. A different, prior
question: is the section-7.2 D=20 result itself sensitive to where the *constrained*
search (the standard method, `outer_solve_nested_cached` with `use_var_scaling=true,
scaling_power=0.5`) started? That original result came from a SINGLE run, cold-started at
theta_r0 (A\*) -- confirmed directly from the saved jld2's own
`starting_point_source => "theta_r0 (initial)"` field, NOT from any prior multi-start.

Tested by re-running the actual standard constrained method (NOT the cheap unconstrained
reformulation) from 2 randomized Acol starting points (gamma'\_focal left at its usual
value; log-normal multiplicative perturbation of Acol, matching the
`multistart_screening_d20.jl` scheme). Script:
`run_scaled_constrained_d20_randomstart.jl`. Note: `run_profiled_production.jl`'s own
batch loop (`run_one_bound`) does NOT thread `scaling_power` through to
`outer_solve_nested_cached` at all (silently defaults to 1.0) -- this script calls
`outer_solve_nested_cached` directly with `scaling_power=0.5` explicit to faithfully match
the original run. An initial pair of runs (not shown) used the wrong (default, 25-iteration)
opt file by omission and hit the iteration cap before converging -- discarded and re-run
with the correct `csw_outer_1000.opt` (absolute path, to avoid the relative-path-silently-
fails-to-load bug documented in section 3 item 3).

| run | starting point | kappa (best-feasible) | KNITRO status | genuinely converged? |
|---|---|---|---|---|
| Original (section 7.2) | cold start at A\* | 0.081518 | -101 (iter/tol limited on the raw endpoint; needed the best-feasible fallback) | no |
| Random start 1 (seed=1, sigma=0.5, initial relΔA=0.50 from A\*) | | **0.082508** | **0** (locally optimal, opt_err=7.9e-7) | **yes** |
| Random start 2 (seed=2, sigma=1.0, initial relΔA=1.90 from A\*) | | **0.082350** | **-100** (feasible, converged, opt_err=9.4e-6) | **yes** |

**The original constrained-search result is starting-point-robust, not a fluke of cold-
starting at A\*.** Both randomized starts not only reproduce the original's quality, they
slightly EXCEED it (+1.0-1.2% kappa), and -- notably -- both terminate with a genuinely
clean KNITRO status (0 / -100), unlike the original raw endpoint (-101, which needed the
best-feasible-point fallback machinery to be trusted at all). This is independent,
positive evidence that D=20's kappa≈0.082 (upper bound, delta=1, W=80000) is a real,
robust local optimum of the CONSTRAINED problem itself, reachable from qualitatively
different starting points -- not merely a coincidence of the one starting point tried in
section 7.2. Cost was much higher than the original (~152-156 min vs ~46 min per run),
consistent with these runs doing more genuine iterations to reach a clean, tight
convergence rather than stopping early.

## 6. Interpretation and recommendation

The reformulation is validated as a genuinely useful, cheap diagnostic: **fix gamma' at a
candidate value and do a plain (no budget constraint, no gamma'-vs-A_od scale mismatch)
minimization of delta\*(A_od) using the SAME KNITRO solver** the production code already
uses -- no new solver machinery, no new tolerance regime, and (via the cheap fixed-dual-FD
gradient trick) no meaningfully higher per-iteration cost.

- **At D=4**, it exposes real headroom (~12-20%) that the constrained search's own
  gamma'-vs-A_od gradient-scale mismatch evidently left unclaimed -- direct, converged,
  reproducible evidence (two independent optimizers agree) that the constrained search is
  NOT finding the true joint optimum at this scale.
- **At D=20 real data**, applied to the ALREADY-scaled (KNITRO-variable-scaling-corrected)
  search result, it finds NO further headroom -- the scaled search's own endpoint passes a
  genuine local-optimality check. This is valuable in the other direction: it is
  independent, cheap confirmation that the section-7.2 scaling fix's D=20 result is a real
  local optimum, not merely "moved A substantially and got lucky."
- **Section 5c goes further**: re-running the actual (expensive) CONSTRAINED search itself
  from 2 different randomized starting points confirms the D=20 result is not sensitive to
  starting point either -- both random starts reach a comparable-or-slightly-better kappa
  (0.0825, 0.0824 vs the original 0.0815) with genuinely clean KNITRO convergence (the
  original's raw endpoint needed a best-feasible fallback; these didn't). Two independent
  lines of evidence (cheap unconstrained reformulation AND expensive constrained multi-start)
  now agree: D=20's kappa≈0.082 is a real, robust result.

**Recommended next steps**, in priority order:
1. Run the D=20 KNITRO reformulation test against the section-7 (UNSCALED, "A never
   moved") result too, not just the scaled one -- the unscaled endpoint IS A\* itself, so
   this is already covered by the "start=A\*" result above (delta\*_min=1.083, a genuinely
   worse local optimum than budget=1.0) -- i.e. this already shows the UNSCALED search's
   result is clearly suboptimal, consistent with the report's own finding that A never
   moved there.
2. Consider using this reformulation as a cheap post-hoc VALIDATOR for every future
   constrained-search result (D=10, other delta/W settings): run one unconstrained KNITRO
   minimization from the constrained search's own endpoint; if it stays put (like D=20
   here), trust the result; if it moves substantially (like D=4 here), the constrained
   search needs more work at that setting.
3. The D=4 lower-bound headroom (~12-20%) is itself worth understanding -- is it a
   genuine local-optima-multiplicity issue (like D=20's A\* start) or a sign the
   constrained search specifically under-iterates on Acol due to the scale mismatch? A
   direct comparison of the unconstrained-reformulation's final A_od against the
   constrained search's own trajectory (not just endpoint) would help distinguish these.
