Prompt for a new Claude instance (paste this whole thing):

# Task: head-to-head comparison of 4 outer-loop search methods, D=20 real data

This is an ORCHESTRATION task. You will set up a fair, controlled comparison, then launch
FOUR independent, fully-detached, overnight background jobs (one per method, for isolation
-- no shared state, no contamination), then monitor them and produce a final comparison
report. The user may disconnect once you've launched everything; you must not get stuck
waiting for a decision at that point. **Read the whole prompt and do all your setup/design
work FIRST, asking the user anything genuinely ambiguous BEFORE launching anything** -- do
not discover a blocking question 6 hours into an unattended run.

## Where this lives

Repo: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf` (Julia project,
branch `feature/sequential-inversion-perf`). KNITRO only licenses on `demand.mit.edu`:

    export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
    export KNITRODIR=/opt/shared_sw/knitro/14.2.0
    export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
    export PATH="$HOME/.juliaup/bin:$PATH"
    cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
    julia --project=. <script>

**Read these, in full, before writing any code** -- this is not optional, all four methods
already have real, validated (or partially-validated) implementations you should reuse, not
rebuild from scratch:

1. `sequential_gravity/derivative_diagnostics/full_d2_correction_report.md` -- the master
   report for the corrected local-constrained method (sections 1-6), the D=20
   gradient-scale-mismatch problem and KNITRO-scaling fix (section 7/7.1/7.2), the
   delta-grid warm-start bug and fix (section 7.3), and the final independent
   re-verification (section 7.4).
2. `sequential_gravity/derivative_diagnostics/D20_METHOD_WRITEUP.md` -- a clean statement of
   exactly what the local-constrained method IS and the exact settings used, plus the full
   verification table for all 4 delta points (0.1, 1.0, 2.0, 5.0).
3. `sequential_gravity/derivative_diagnostics/profiled_reformulation_report.md` -- the
   local-UNCONSTRAINED method (fix gamma'_focal, minimize delta*(A_od) alone via KNITRO with
   the existing corrected gradient, no outer constraint at all). Includes a 5-point
   multistart at D=20 already.
4. `sequential_gravity/global_opt/global_optimizer_report.md` -- the GLOBAL (gradient-free,
   BlackBoxOptim) methods, both the original joint-constrained attempt (section 4.1, run for
   ~25 min then deliberately abandoned/superseded, NOT because it doesn't work but because a
   sibling reformulation looked more promising to pursue first) and the global-UNCONSTRAINED
   reformulation (section 4.2, run to a full 6h budget at D=20, found real headroom,
   `delta*_min=0.964` vs budget `1.0`, still improving when the budget ran out).
5. Memory entries (if visible to you -- this project uses a persistent memory system):
   `full-d2-winner-boundary-fix`, `profiled-reformulation-fix-gp-min-deltastar`,
   `feedback-match-optimizer-when-comparing-approaches` (**read this one carefully** -- it
   records that comparing a reformulation against a baseline must hold the OPTIMIZER choice
   fixed and vary only the problem shape, which is exactly why this task needs FOUR distinct
   method cells, not two).

## The 2x2 design

Two independent axes, both already explored separately but never compared head-to-head on
equal footing:

|  | **constrained** (extremize gamma'_focal s.t. delta\*<=budget) | **unconstrained/profiled** (fix gamma'_focal, minimize delta\*(A_od)) |
|---|---|---|
| **local** (gradient-based, KNITRO) | **LC**: `run_profiled_production.jl::outer_solve_nested_cached` (`gradient_method=:fixed_dual_fd_full`, `use_var_scaling=true`) | **LU**: `run_profiled_delta_star_min_knitro_d20_real.jl` machinery |
| **global** (gradient-free, BlackBoxOptim, population-based) | **GC**: `global_opt/run_bbo_d20_real.jl` / `bbo_common.jl` (revive and carry to completion -- see below) | **GU**: `global_opt/run_bbo_d20_profiled_deltastar.jl` machinery |

**Known real bugs/lessons from building these -- do not repeat them:**

- **A "solution" is whatever the search reports IS NOT reliable** -- the raw KNITRO
  endpoint (or a global optimizer's final reported point) can be, and often is at D=20,
  gravity-infeasible or divergence-budget-infeasible. EVERY method must track the best
  point seen ACROSS ALL evaluations that is verified feasible (gravity residual within
  tolerance AND, for constrained methods, divergence within budbudget) and report/warm-start
  from THAT, never the raw termination point. This is exactly the bug found and fixed in
  `run_profiled_production.jl`'s own delta-grid warm-start chain (`full_d2_correction_report.md`
  section 7.3) -- it silently chained TWO delta points to complete failure (zero feasible
  points found) by warm-starting from an infeasible raw endpoint. The SAME discipline must
  be built into all four methods' drivers here, including the two (LU, GU) that don't
  currently have a delta-grid/warm-start chain to get this wrong in yet.
- `run_one_bound` (the LC batch-loop entry point) has ANOTHER real bug, also just found and
  fixed: it never threaded a `scaling_power` keyword through at all, silently defaulting to
  1.0 regardless of any `SCALING_POWER` env var set at launch. Confirm you are ACTUALLY
  getting the scaling_power you intend (print/log it, don't just trust an env var was read)
  for LC's runs here.
- `exact_inner_divergence_at` (used by LU and GU as their core objective/evaluator) can fail
  outright and return a huge sentinel value (observed: exactly `1e10`) rather than signaling
  failure cleanly -- treat any returned `delta_star >= 50` (far above anything a legitimate
  candidate near these budgets should ever score) as an infeasible evaluation, not a real
  number.
- `A_od`'s natural free-variable range is enormous in raw level units (`focal_bounds` gives
  roughly `A* * 1e-4` to `A* * 1e4`) -- fine for KNITRO's own internal scaling, a poor fit for
  a population-based method's box constraints. The global methods (GC, GU) should search in
  LOG-space (`logratio = log.(A_od ./ A_od_star)`), bounded per-coordinate to something
  informed by `multistart_screening_d20.jl`'s own tested range (that screening found ALL 20
  random perturbations feasible up to relative distance ~20x from A*, but `global_optimizer_report.md`
  section 4.2 found that a bound generous enough for one coordinate (`+-6`, i.e. up to ~400x)
  becomes wildly too generous once applied INDEPENDENTLY across all 20 coordinates jointly
  (random draws routinely landing 50-120x from A*) -- that report tightened to `+-3` for its
  D=20 profiled run; start there, or size it from your own quick empirical check.
- Two rho (smoothing) conventions coexist in this codebase and are easy to mix up if you
  build any NEW independent verification: the focal destination's trade shares use a literal
  hard argmin (rho=0, `EK_moments_focal_norm_directgp!`, never smoothed); the D-1 omitted
  destinations are inverted under a smoothed softmax (rho=`global rho`=0.002,
  `invert_destination`'s internal convention throughout `seq_gravcol`). Getting this backwards
  (or applying one rho uniformly to both) produces convincing-looking but SPURIOUS
  discrepancies of 1e-3 to 1e-2 in a trade-share check -- this happened twice while producing
  `D20_METHOD_WRITEUP.md`'s verification table; read that document's section 5 for the full
  story before writing any share-matching check of your own.

## Experimental design

**Bound**: upper only (`find_smallest=true`, maximizes kappa).

**3 target points.** For the constrained methods (LC, GC), these are 3 divergence budgets:
`delta in {0.1, 1.0, 2.0}`. For the unconstrained methods (LU, GU), these are 3 FIXED
gamma'_focal targets, reused DIRECTLY from this session's own already-verified LC results at
those same 3 deltas (see `D20_METHOD_WRITEUP.md` section 5 -- use the EXACT saved values, not
the rounded ones the user mentioned):

| delta (LC/GC target) | gamma'_focal (LU/GU target) | kappa |
|---|---|---|
| 0.1 | 0.972866 (use the exact saved `best_feasible_gp`, currently 0.9728658401964491) | 0.044813 |
| 1.0 | 0.950260 (0.950259956422648) | 0.081518 |
| 2.0 | 0.944055 (0.9440553861575977) | 0.091491 |

(Pull these fresh from `sequential_gravity/batch_out_realD20_W80000_fixeddualfdfull_scaled05/seq_upper_delta{0.1,1.0,2.0}.jld2`'s
own `best_feasible_gp` field rather than retyping the numbers above, in case anything changes
before you start.)

**3 starting points per target, for every method:**
1. `A_od = A*` (the calibrated Frechet baseline).
2. A random perturbation of A* (log-normal multiplicative noise), call it `rand1`.
3. A SECOND random perturbation, `rand2`.

`rand1`/`rand2` must be **the exact same two vectors across all four methods** -- generate
them ONCE (fixed seed, e.g. matching `multistart_screening_d20.jl`'s own
`SEED=20260715` convention, sigma=0.5 for rand1 and sigma=1.0 for rand2 is a reasonable
default matching what's already been spot-tested elsewhere in this repo -- adjust if you
have good reason, but do not let each method's script independently draw its own "random
same-seed" points; save the two vectors to a single shared file (e.g. JLD2) that all four
method scripts load, so there is no possibility of an RNG-consumption-order mismatch making
them silently different).

**Warm-starting across targets, replacing ONE random start (per the user's explicit
request):**
- Target 1 (delta=0.1 / gamma'=0.972866): 3 starts = `{A*, rand1, rand2}` (all cold).
- Target 2 (delta=1.0 / gamma'=0.950260): 3 starts = `{A*, rand1, warm-from-target-1's
  BEST-FEASIBLE result}` (drop rand2).
- Target 3 (delta=2.0 / gamma'=0.944055): 3 starts = `{A*, rand1, warm-from-target-2's
  BEST-FEASIBLE result}` (drop rand2).

This gives exactly 9 solves per method (3 targets x 3 starts), 36 total, split into 4
separate processes/jobs (one per method) as the user explicitly requested, for isolation.

**What "warm-start from the best-feasible result" means precisely**: for LC/GC, the full
`theta` (gamma'_focal AND A_od) of whichever of the 3 starts at the PREVIOUS target achieved
the best (highest, since this is the upper bound) verified-feasible kappa. For LU/GU
(gamma'_focal is fixed at the NEW target throughout, not carried over), only the A_od part of
that same best-feasible result is reused as the new starting A_od.

## Fair-comparison requirements

- **Same per-solve compute budget across all four methods.** This is essential for the
  comparison to mean anything -- estimate wall-clock cost PER SOLVE for each method with a
  quick pilot (e.g. one D=20 evaluation timing, or reuse the existing reports' own timing
  data: LC solves have taken 45min-2.2h each in this session's own delta-grid runs; LU/GU
  are population/pattern-search methods whose cost scales with however many generations/
  evaluations you allow in a fixed time budget). Pick ONE wall-clock cap (e.g. 60-90 minutes)
  and apply it uniformly to every one of the 36 solves, regardless of method. **Compute the
  resulting total wall-clock estimate (36 solves x cap, per method, run as 4 concurrent
  jobs so it's ~9 x cap total, not 36 x cap) and tell the user this BEFORE launching** --
  if a 90-minute cap implies ~13.5 hours per method job, that's a real "next-day" number,
  not "ready by morning"; make sure the user is fine with whatever the real number is, or
  tighten the cap, BEFORE they disconnect.
- **Machine has 208 cores.** Each individual D=20 evaluation already parallelizes its own
  19-destination inversion across 19 threads (`PARALLEL_INVERSION=true`, `julia -t 19`).
  Running the 4 method-jobs concurrently costs `4 x 19 = 76` cores, leaving substantial spare
  capacity -- consider (but do not feel obligated to build, given the added complexity of an
  unattended run) parallelizing the 2 mutually-independent "cold" starts (A* and rand1) at
  each target against each other within a method's own job, since they don't have the
  warm-start dependency the third start does. If this feels like too much extra complexity
  for an overnight unattended run, a simple sequential 9-solve loop per method is a
  perfectly reasonable, more robust default -- your call, but state which you chose and why.
- **Every job must be fully detached** (survives the user disconnecting): the
  `nohup setsid <command> > log 2>&1 & disown` pattern used throughout this repo's existing
  `run_d20_*.sh` scripts and this session's own background launches. Write an ALLDONE sentinel
  file per job on completion. Checkpoint intermediate progress (e.g. best-feasible-so-far)
  after every solve at minimum, ideally more often for the global methods given their own
  report's precedent of per-evaluation JLD2 checkpointing (`global_optimizer_report.md`
  section 2) -- an unattended multi-hour job that loses all progress on an interruption is
  not acceptable here.
- **Pilot before committing.** Run ONE quick, cheap validation of each method's plumbing
  (a single solve, or even a D=4/synthetic-data smoke test where those already exist) BEFORE
  launching the full 9-solve overnight batch for that method. Several real bugs in the
  existing per-method code were only caught this way (see the "known bugs" list above, and
  `global_optimizer_report.md`'s own "two bugs found... before trusting this number" for
  precedent). Do this validation work WHILE the user is still available to answer questions,
  not after they've disconnected.

## Final comparison

For each of the 3 targets and each of the 4 methods, report the BEST result found across
its 3 starts, on a common footing:
- The gamma'_focal achieved (identical to the target for LU/GU by construction; the actual
  outcome for LC/GC).
- The kappa achieved.
- **The exact delta\* needed to achieve that gamma'_focal** -- for LU/GU this is the method's
  own direct output; for LC/GC, compute it via a POST-HOC audit using the ALREADY-BUILT
  `exact_inner_divergence_at` (`fixed_A_incumbent.jl`) at whatever gamma'/A_od the search
  actually landed on, exactly as `verify_d20_deltagrid.jl` already does for the existing LC
  results -- do not just report the nominal delta budget LC/GC were given, since (per
  `full_d2_correction_report.md` section 5's own D=4 audit) the true delta* at a constrained
  search's endpoint is not always exactly equal to its budget.
- Which starting point produced the best result (A*, rand1, rand2, or warm-start), and how
  much the winning `A_od` moved from `A*` (`rel‖ΔA_od‖`).
- Wall time and evaluation count actually used.

This lets you build a single table answering the user's real question -- "for a given
economic target, which method finds the cheapest (lowest delta\*) way to get there, and by
how much" -- across all 4 combinations of (local/global) x (constrained/unconstrained),
which is not directly comparable between constrained and unconstrained any other way (per
the user's own observation that you "can't literally compare constrained vs unconstrained
because they target different things").

Write up the full comparison (methodology, the table above, and your honest read of which
approach looks most promising for production use, including any important caveats about
compute cost / robustness / how close each method got to full convergence within its
budget) as a new markdown report in `sequential_gravity/derivative_diagnostics/` or
`sequential_gravity/global_opt/`, matching the style of the reports you were asked to read
above. Update the project's persistent memory with the key findings if you have access to
that system.

## If you have questions

Ask the user NOW, before launching anything long-running -- especially about: the per-solve
time budget (and the resulting total runtime this implies), the random-perturbation scale
for rand1/rand2, and whether to attempt within-method parallelism across independent starts.
Do not guess on something that would waste hours of unattended compute if wrong; do use your
own judgment (informed by the existing reports and a quick pilot) for anything you can
reasonably resolve yourself.
