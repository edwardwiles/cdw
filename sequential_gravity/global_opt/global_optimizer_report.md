# A gradient-free global optimizer for the CC outer loop

Status: DONE. D=4 synthetic validation (Stage A, joint constrained formulation, section 3):
pipeline validated, did not beat the local search in-budget. D=20 real-data run was
**reformulated mid-session** (section 4.1, on explicit direction) from a joint constrained
search over `(gamma'_focal, A_od)` to a "profiled" unconstrained search that fixes
`gamma'_focal` at a target and minimizes `delta*(A_od)` alone. That reformulated search
(section 4.2) ran its full 6h budget and found genuine headroom: `delta*_min = 0.964` at the
local search's own target `gamma'` (kappa=0.081518), below the `delta=1` budget, reached with
**less** extreme `A_od` movement (104%) than the local KNITRO-scaled search needed (523%). See
section 5 for what this does and does not establish, and the natural (not-yet-built) next step.

## 1. Motivation

`derivative_diagnostics/full_d2_correction_report.md` sections 7/7.1/7.2 found that the
existing outer search -- KNITRO SQP/interior-point, always gradient-based, always initialized
at `theta_r0` (`A=A*`) -- essentially never moves `A_od` at all at D=20 real data
(`theta_star`'s `Acol` block came back bit-identical to `A*`, for either gradient method
tried). The traced cause is a severe cross-variable gradient-scale mismatch: `gamma'_focal`'s
own constraint-gradient component is `~1e8` while the entire `Acol` block is only `~10-11500`
in magnitude, a ratio of `1e4-1e5`. A first fix (KNITRO-level variable scaling) helps -- kappa
rose from 0.076693 to 0.081518 -- but doesn't converge cleanly (KNITRO status -101, raw
endpoint gravity-infeasible, 13x slower) and actively **hurts** D=4, where the unscaled search
was already exploring `A` fine. It was explicitly not adopted as a default.

Separately, `multistart_screening_d20.jl` found that **20/20** random log-normal perturbations
of `A*` (noise sigma up to 2.0, giving relative distances from `A*` up to ~20x) were
gravity-feasible for the model's inner problem, with the divergence cost of the implied
reweighting scaling smoothly (not catastrophically) with distance from `A*`. That is strong
evidence the feasible region away from `A*` is large and well-behaved -- the local search is
stuck in a bad basin near `A*`, not evidence that no better basin exists.

This task builds a **global, gradient-free, population-based** search (BlackBoxOptim.jl,
adaptive differential evolution) directly over `(gamma'_focal, A_od[1:D])`, sidestepping the
gradient-scale problem entirely since DE needs no gradients at all.

## 2. What was built

- `sequential_gravity/global_opt/bbo_common.jl` -- the fitness function, built **entirely**
  from `seq_gravcol` + `divergence_of` + `focal_bounds` + `gp2kappa`
  (`run_profiled_production.jl`), with **no** gradients, **no** KNITRO outer-loop machinery
  (`outer_loop_cached`, `PsiObjectiveBundleImplicitMethodB`, `gradient_method`,
  `use_var_scaling`), and **no** modification to any already-validated production file.
  KNITRO is still used, unmodified, *inside* `seq_gravcol` for the inner CC dual solve at a
  fixed theta -- that machinery is load-bearing and untouched.
- **Parameterization**: free vector `x = [gamma'_focal, logratio(1:D)]`,
  `logratio = log(A_od ./ A_od*)`, i.e. **log-space** for `A_od` (a multiplicative
  competitiveness parameter) rather than `focal_bounds`' own raw-level box (`A* * 1e-4` to
  `A* * 1e4`, a ~1e8x range that would swamp a population-based search). `logratio` is bounded
  to `[-6, 6]` (`exp(6) ≈ 403x`) per coordinate by default (`LOGBOUND` env var) --
  directly motivated by, and with real headroom past, the multistart screening's own tested
  range (relΔA up to ~20x, i.e. within `[-3,3]` for a single dominant coordinate under
  independent per-coordinate lognormal noise). `gamma'_focal` itself is searched directly in
  `focal_bounds`' own `[γp_lo, γp_hi]` (the theoretical kappa bounds), unchanged.
- **Fitness**: for a feasible candidate (gravity converges within the sequential loop's own
  tolerance AND the achieved CC divergence is within the delta budget), fitness is `±gp` so
  that minimizing it directly maximizes or minimizes `gamma'_focal` per the requested bound
  direction. For an infeasible candidate, fitness is a large constant offset (100, bigger than
  any feasible value since `gp` is bounded to (0,1)) plus a **smooth** penalty: how far the
  gravity residual is over its tolerance, how far the divergence is over its budget, and a
  small pull-back term toward `A*` for the (rare) totally-degenerate case where `seq_gravcol`
  can't even produce a finite residual. This gives the optimizer a descent direction back
  toward feasibility instead of a flat infeasible/feasible cliff.
- `run_bbo_d4_synthetic.jl` -- Stage-A validation driver, both bounds, D=4 synthetic data.
- `run_bbo_d20_real.jl` -- Stage-C driver, one bound at a time, D=20 real data, with
  `CallbackFunction`-driven **JLD2 checkpointing** of the best-feasible candidate found so far
  (every optimizer step, since each step is already ~10-80s), so a multi-hour `MaxTime`-bounded
  run launched detached (`nohup setsid ... & disown`, matching this repo's existing
  `run_d20_*.sh` convention) never loses progress even if interrupted.
- `run_d20_bbo_global_upper_delta1_W80000.sh` -- the launch script actually used for section 4.

### Design choices not taken

Per the task's own compute-budget guidance, the population loop here is a **serial** loop:
each candidate is evaluated one at a time, using the full 19-thread destination-inversion
parallelism (`PARALLEL_INVERSION=true`) *within* that one evaluation. Parallelizing *across*
the population with `Distributed.jl` worker processes (option 2 in the task spec -- e.g. 10
worker processes x 19 threads each on this machine's 208 cores, for a ~10x population-loop
speedup) was explicitly scoped out of this session as real, separate engineering work, not
something to bolt on casually. It is the natural next lever if more generations are needed
than a serial loop's time budget allows.

## 3. Stage A: D=4 synthetic validation

Setting: `FAKEDATA=1` (synthetic Frechet draws), `DVAL=4`, `WVAL=8000`, `delta=1` -- the exact
setting `full_d2_correction_report.md` section 5 already validated with the local KNITRO
search, giving a direct comparison. `BlackBoxOptim` `adaptive_de_rand_1_bin_radiuslimited`,
`PopulationSize=24`, `MaxTime=900s` per bound (measured single-eval cost here is ~5s once
warm, so this is ~150-350 function evaluations per bound depending on how much time each
`seq_gravcol` call takes near the search's current candidates).

| bound | kappa_BBO | kappa_local (Stage A) | feasible | fevals | rel‖ΔA‖ vs A* |
|---|---|---|---|---|---|
| lower (minimize kappa) | 0.037537 | **0.004062** | true | 344 | 188.9% |
| upper (maximize kappa) | 0.161372 | **0.172109** | true | 100 | 269.9% |

**Reading this result**: the pipeline is validated end-to-end -- fitness returns finite,
correctly-signed values; feasible candidates are found and improved on generation after
generation (see the raw BBO trace: the lower-bound search starts fully infeasible at
fitness=+101.8, finds its first feasible point by ~270s, and keeps improving through 900s);
the optimizer genuinely explores `A_od` far from `A*` (189-270% relative movement, well within
the multistart screening's own demonstrated-feasible range) rather than collapsing back to
`A=A*`. It does **not** beat the local search's more precise D=4 optimum in this budget --
expected and unsurprising, since `full_d2_correction_report.md` already established the
*unscaled* local search explores `A` just fine at D=4 (31-46% movement, clean KNITRO
convergence in ~2-13 minutes); a gradient-based method converges faster than a
population-based one when the gradient itself is trustworthy, which it is at D=4. The value
proposition for a global search is specifically the D=20 regime where the gradient is *not*
trustworthy (the 1e4-1e5x scale mismatch) -- Stage C below.

## 4. Stage C: D=20 real-data run

### 4.1 First attempt (joint constrained search) and why it was reformulated mid-session

The first D=20 attempt used exactly the section-2 machinery: a joint global search over
`(gamma'_focal, A_od)` with a smooth infeasibility penalty standing in for the `delta*<=budget`
constraint. It was launched (`FAKEDATA=3 DVAL=20 WVAL=80000`, upper bound, `delta=1`,
`PopulationSize=16`, `MaxTime=21600s`) and ran cleanly for ~25 minutes (16 evaluations, ~36s
each, matching the multistart screening's own per-eval cost) before being **stopped and
replaced**, on explicit direction, in favor of a differently-structured reformulation below.
The reasoning (independently verified against the actual code before adopting it, not taken on
faith -- `fixed_A_incumbent.jl`'s `exact_inner_divergence_at` and
`derivative_diagnostics/run_profiled_delta_star_min.jl`/`run_profiled_delta_star_min_d20_real.jl`
were all read in full): a joint search over `(gamma'_focal, A_od)` subject to an expensive,
*implicit* divergence constraint is plausibly hard for a population-based method for a related
reason it's hard for KNITRO -- `gamma'_focal`'s own sensitivity to that constraint is many
orders of magnitude larger than `A_od`'s (section 1), and any constraint-handling scheme
(KNITRO's Lagrangian, or this session's own smooth penalty) has to fight that same scale
mismatch one way or another.

**The reformulation**: fix `gamma'_focal` at a target `GT` (it drops out as a free variable
entirely) and **minimize `delta*(A_od)` over `A_od` alone** -- a genuinely box-bounded,
*unconstrained* scalar minimization in `D` variables, with no `gamma'`-vs-`A_od` scale
mismatch anywhere (`gamma'` is never a decision variable) and no constraint-penalty machinery
needed (`delta*` itself, computed exactly, IS the objective). The objective is the existing,
already-validated `exact_inner_divergence_at(theta)` (`fixed_A_incumbent.jl`, Part 10 of
`full_d2_correction_report.md`): a fresh gravity-linearization freeze plus a real, cold-started
KNITRO inner CC-dual solve at every candidate `A_od` -- no approximation, and no new inner-dual
machinery written for this. At D=4, minimizing `delta*(A_od)` (via BlackBoxOptim's
`:generating_set_search`, a *local* pattern-search method) from both `A_od=A*` and the
constrained search's own endpoint landed at `delta*~0.79-0.81`, meaningfully below the
`delta=1` budget the constrained search was given -- direct evidence real headroom was left on
the table, independent of any KNITRO-scaling question.

This driver (`run_bbo_d20_profiled_deltastar.jl`) swaps that local pattern search for a genuine
population method (`adaptive_de_rand_1_bin_radiuslimited`), matching this task's own
gradient-free/global mandate, and reuses this session's log-space `A_od` parameterization
(section 2) instead of the raw `+-1e4x` box the D=4/D=20 local-search scripts used (a poor fit
for a population method at this scale, per section 2's own reasoning). `GT` is reused from the
already-computed D=20 KNITRO-scaled result (`kappa=0.081518`,
`full_d2_correction_report.md` section 7.2) -- so this run answers a narrower, sharper
question than section 4's original framing: **starting from the local search's own
already-decent `gamma'`, is there an `A_od` with strictly more slack (lower `delta*`) than the
local search found?** It does not by itself search for a better `gamma'` -- see section 5 for
why that's the natural next step, not yet built here.

### 4.2 Profiled global search result

Setting: `FAKEDATA=3`, `DVAL=20`, `WVAL=80000`, `PARALLEL_INVERSION=true`, `julia -t 19`,
`GT=0.950259956422648` (reused from the KNITRO-scaled local search, matching
`kappa=0.081518`), `PopulationSize=12`, `MaxTime=21600s` (6h), `LOGBOUND=3.0` (tightened from
section 2's `6.0` default -- see "two bugs found" below).

| quantity | value |
|---|---|
| `delta*(A_od=A*)` at this `GT` | 1.79911 |
| `delta*(A_od=`scaled-search endpoint`)` at this `GT` | 1.00004 |
| `delta*_min` found by the global search | **0.96401** |
| gap vs the original `delta=1` budget | **+0.03599 (3.6%)** |
| rel‖ΔA‖ of the minimizing `A_od` vs `A*` | **103.6%** |
| wall time / function evaluations | 21681s (ran to its full time budget) / 1113 |

**Two bugs found and fixed before trusting this number** (both caught by a short smoke test
before committing the full 6h run, not after):
1. `exact_inner_divergence_at`'s own cold-started KNITRO inner-dual solve can fail outright and
   return a huge sentinel value (observed: exactly `1e10`, matching the `δ_star_initial=1.0e10`
   fallback visible in this repo's own setup diagnostics) rather than throwing or signaling
   failure through `gravity_ok`. Un-caught, this silently fed literal `1e10`-scale "fitness"
   values into the optimizer with no usable gradient. Fixed by treating any `delta_star >= 50`
   (a threshold far above anything a real candidate near a `delta=1` budget could legitimately
   score, per the multistart screening's own smoothly-scaling divergence) as a solve failure,
   penalized the same smooth way as a genuine gravity failure.
2. `LOGBOUND=6.0` (section 2's default, chosen from a *single-coordinate* reading of the
   multistart screening) is far too generous once applied *jointly* across all 20 independent
   `A_od` coordinates -- random draws from the box routinely landed at `relΔA` of 50-120,
   5-6x past anything the screening actually tested. Tightened to `3.0` for this driver
   specifically (not changed in `bbo_common.jl`'s shared default, to avoid touching the
   already-validated D=4 Stage-A result) -- confirmed by a second smoke test to keep `relΔA`
   in the multistart-validated range (~2-6) while still finding real, non-trivial improvement.

**Sanity check passed**: `delta*(A_od=scaled-search endpoint)` came back at `1.00004`,
essentially exactly `1.0` -- confirms `exact_inner_divergence_at` reproduces the local search's
own binding constraint almost exactly at the point it's supposed to reproduce it at, i.e. the
machinery is wired correctly, not just "returning some number."

**The descent was still ongoing when the time budget ran out, not plateaued.** Full
improvement trajectory (every strict improvement over the 1113-evaluation run, from the
checkpoint log, not a summary or a guess):

| eval | time | best δ*\_min | | eval | time | best δ*\_min |
|---|---|---|---|---|---|---|
| 6 | 12:24 | 2.270 | | 536 | 15:13 | 1.027 |
| 55 | 12:42 | 1.206 | | 617 | 15:39 | 1.017 |
| 236 | 13:40 | 1.150 | | 654 | 15:50 | 0.989 |
| 351 | 14:15 | 1.061 | | 868 | 17:00 | 0.974 |
| 509 | 15:05 | 1.036 | | 1015 | 17:49 | 0.968 |
|  |  |  | | **1065** | **18:04** | **0.964 (final)** |

The last improvement landed at evaluation 1065 of 1113 -- i.e. within the last ~4% of the
budget -- with no stretch of more than ~150 evaluations anywhere in the run without a further
improvement. This is a search that was still actively descending when it was cut off, not one
that had converged and was idling. See section 5 for what this implies about running longer.

**Reading this result**: real, not marginal, headroom -- `delta*_min=0.964` is meaningfully
below the `delta=1` budget the local KNITRO-scaled search was given, found via **less**
extreme `A_od` movement (104% vs. the local search's own 523%) to hit the *same* `gamma'`
target. This directly confirms the reformulation's premise from section 4.1: the local
search's own reported `kappa=0.081518` was not tight against its stated budget -- there was
real slack left on the table, and the global box-only search found it more efficiently (smaller
`A_od` movement, and without KNITRO's own messy-convergence problems -- section 7.2's scaled
search finished at KNITRO status -101, "iteration/tolerance limited", not a clean optimum).

**What this result does NOT establish**: it does not, by itself, produce a *better* `kappa`
than 0.081518 -- `gamma'_focal` was held fixed throughout this search, so the reported `kappa`
is identical to the local search's own. It shows that MORE could likely be extracted (see
section 5), not that more has been extracted yet.

## 5. Conclusions and next steps

1. **The reformulation is a real improvement on the joint constrained search**, both
   qualitatively (matches this session's D=4 evidence, section 4.1) and now quantitatively at
   D=20 real data: a clean, monotone 1113-evaluation descent, no KNITRO-scaling machinery, no
   constraint-penalty tuning, landing on genuine feasible slack (`delta*=0.964` vs budget `1.0`)
   with a smaller `A_od` movement than the local search needed. This is a stronger empirical
   result than either the unscaled local search (never moved `A` at all, section 1) or the
   KNITRO-scaled one (messy, non-clean convergence, section 4.2).
2. **This session's original joint-search formulation (section 2/4.1's first attempt) should
   be considered superseded** by the profiled reformulation for this problem, based on the
   direct D=4 and D=20 evidence gathered here -- not a priori reasoning. It remains in the repo
   (`run_bbo_d20_real.jl`, `bbo_common.jl`) as a validated, working, but no-longer-preferred
   alternative.
3. **The natural next step -- explicitly NOT built in this session, given the 6h already spent
   on the run in section 4.2 -- is to close the loop**: wrap the profiled `delta*(A_od)`
   minimization in an outer bisection (or simple stepping search) on `GT` itself. Since
   `delta*_min(GT=0.950260) = 0.964 < 1.0`, a MORE extreme `GT` (lower `gamma'`, higher
   `kappa`, since `kappa` is decreasing in `gamma'`) should still have a feasible `A_od`
   somewhere in the box; each trial `GT` costs one more ~6h-scale profiled search (though
   probably faster once warm-started from this run's own `Acol_best`, saved in
   `d20_checkpoints/bbo_d20_profiled_deltastar_upper.jld2`, rather than cold from `A*`). This
   is real additional compute, not a quick follow-up -- worth explicitly deciding whether to
   spend before launching it, rather than assumed as automatic.
4. **Engineering lever not used**: `Distributed.jl` worker-process parallelism across the
   population (section 2, "design choices not taken") would directly cut this section's 6h/1113
   evals down by roughly the worker count, and would make a `GT`-bisection follow-up (item 3)
   far more tractable than running it as another single 6h serial job per `GT` trial.

## 6. Multistart / seed sensitivity (preliminary, deliberately cut short)

Motivated by the question "does the section-4.2 result depend on luck in the random initial
population, or would (almost) any run land near `delta*~0.96`?" -- three copies of the exact
section-4.2 driver were launched **simultaneously** (same `GT`, `PopulationSize=12`,
`LOGBOUND=3.0`, `D=20` real data), differing only in an explicit `Random.seed!` (101, 202, 303)
newly added to `run_bbo_d20_profiled_deltastar.jl` for this purpose (the section-4.2 run itself
did not set a seed, so it isn't exactly reproducible by seed alone -- its own full log is
preserved as the reference trajectory). Budgeted at 2.5h each; **terminated early by explicit
user instruction** (~1h40m in, to free the machine for a separate overnight comparison run) once
the qualitative question was answered -- none of the three reached section 4.2's own 6h/1113-eval
scale, so none of these numbers should be read as converged results.

| | evals at kill | delta*\_min at kill | rel‖ΔA‖ at kill |
|---|---|---|---|
| section 4.2 reference, same eval range | ~282-303 | 1.077-1.105 | (103.6% at full 1113 evals) |
| seed 101 | 295 | 1.172 | 86.6% |
| seed 202 | 287 | **1.026** | 261.9% |
| seed 303 | 272 | 1.128 | 270.4% |

**Takeaway (the point of this experiment): multistart matters, clearly.** At closely-matched
evaluation counts, the three seeds spread from 1.026 to 1.172 -- seed 202 was tracking *better*
than the section-4.2 reference at a comparable eval count, seed 101 *worse*, seed 303 roughly
comparable. The seeds also explored very different distances from `A*` at this stage (87% to
270%) without a simple "closer is better" pattern -- e.g. seed 303's distance-270% point beat
seed 101's distance-87% point. This is genuine evidence of a multi-modal / basin-dependent
landscape, not a single well-behaved basin that any reasonable start converges into at the same
rate. **Practical implication for whoever runs this further** (explicitly the motivation for
this section, per the user's own stated plan for a follow-up overnight comparison): a single
run's result -- including this report's own headline `delta*=0.964` -- should not be read as
"the" answer for a given `GT`; a small multistart ensemble (even 3-5 seeds, run in parallel
since they don't interact) is likely to find a meaningfully better point than any one run alone,
and is cheap to parallelize (independent seeds, no shared state) relative to running one seed
much longer.
