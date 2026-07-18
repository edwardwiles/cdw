# Handover prompt for the next Claude session

Copy everything below this line into a fresh Claude Code session to continue this investigation.

---

# Continuation 4: wire the incremental hybrid gradient into a live solver, run the wall-clock frontier, and finish D=4

You are continuing the exact full-A(_{od}) investigation for the Gravity Robustness Paper. Read this
whole prompt before doing anything — it replaces re-deriving context from scratch. Preserve all
validated work; do not restart from production; do not make uncontrolled changes to the production
branch (`sequential-profiled-gravity` in `trade_robustness_modular`).

## 1. Repository state (verify, do not assume)

- Repo: `git@github.com:habibiscoding/Trade-Model-Robustness.git`.
- Diagnostic branch: `diag/fullA-d4-exact`.
- Worktree: `/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`.
- **Expected HEAD at handover time**: `b5c109d` (pushed to origin). Run `git fetch origin
  diag/fullA-d4-exact && git log origin/diag/fullA-d4-exact -1` and compare to local HEAD — if they
  differ, or if local HEAD differs from `b5c109d`, STOP and figure out why before proceeding (three
  background agents were dispatched from this branch state to isolated worktrees, on branches
  `diag/fullA-d4-exact-jach-audit`, `diag/fullA-d4-exact-smoothed-consistent`, and
  `diag/fullA-d4-exact-phase5-sequential` — see §3 below; check whether any have been merged since).
- Environment: `source .knitro_env.sh` first (puts `~/.juliaup/bin` on PATH, sets KNITRO env vars).
  KNITRO only licenses on host `demand.mit.edu` — check `hostname` before trusting any solve result.
  Julia 1.12.6, single-threaded by default (`Threads.nthreads()==1` unless `JULIA_NUM_THREADS` is
  set). 208 cores available.
- Run the mandatory smoke tests before any new work: `julia --project=.
  full_aod_diag/d4_exact/test_oracle.jl` and `test_oracle_profiled.jl` (both must print ALL PASS —
  if either fails, stop and diagnose drift before proceeding, per this investigation's standing
  discipline).

## 2. What is DONE and VALIDATED (read the docs, don't re-derive)

Read, in this order:
1. `docs/fullA_continuation3_resume_audit.md` — Phase 0 state verification + candidate registry.
2. `docs/fullA_performance_profile_v2.md` — **canonical** performance profile (supersedes
   `docs/fullA_performance_profile.md`, which is flagged SUPERSEDED at the top but kept for history).
   Corrected finding: `inner_solve` was mislabeled — 57.2% of a warmed evaluation is moment-matrix
   BUILD (unavoidable), only 5.6% is the actual CC dual optimization.
3. `docs/fullA_block_local_performance.md` — **the most important document for you**. Phase 2's
   incremental `L_fix` machinery: three tiers (`:block_local`, `:incremental`, `:incremental_o1`),
   all equivalence-tested to machine precision against the trusted full-rebuild `fixed_dual_L`.
   Realized speedups at D=4: 1.89x / 5.39x / 7.34x single-threaded, up to 44.5x at 16 threads.
4. `docs/fullA_d4_final_report.md`, `docs/fullA_d4_recommendation.md`, `docs/fullA_d4_code_audit.md`
   — the original investigation's findings (upper incumbent κ=0.17176461388430053 at
   γ'_focal=0.8930839180420251, classified `EXACT_FEASIBLE_CANDIDATE` + `H_BANDWIDTH_KKT_CANDIDATE`
   but explicitly NOT `ROBUST_LOCAL_CANDIDATE`; lower candidate is `BEST_FEASIBLE_STALLED`, far from
   the divergence budget; fixed-A benchmark κ=0.1439805232 as a sanity floor).
5. `docs/fullA_next_handoff.md` — running handoff log, most recent section at the top.

**Key code, all in `full_aod_diag/d4_exact/`**, all equivalence-tested before being trusted:
- `oracle.jl` / `oracle_fast.jl` — exact hard-value oracle (`evaluate_fullA` / `evaluate_fullA_fast`,
  the latter ~2x faster, bit-identical results).
- `winners.jl` / `winners_v2.jl` — winner/runner-up computation (fast version is allocation-free).
- `gravity_elimination.jl` — pivot-based exact gravity elimination (`build_pivot_elimination`,
  `pivot_expand`/`pivot_reduce`) — the reduced 16-dim coordinate `w = [γ'_focal; z_free[1:15]]` used
  throughout; `x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))` is the standard
  conversion, copy this pattern exactly, don't re-derive it.
- `three_way_derivatives.jl` — `BaseDualState`, `solve_base_state`, `fixed_dual_L` (the TRUSTED
  full-rebuild baseline), `frozen_adjoint_Q`, `optimized_Delta`.
- `lfix_incremental.jl` — **THE deliverable you build on**. `build_lfix_base_cache(x_free0, ctx,
  base)` builds an `LFixBaseCache` (self-validating — errors loudly if wrong); `lfix_incremental_at(
  cache, ctx, pe, w0, coord_idx, new_val; tier=:incremental_o1)` evaluates `L_fix` at a
  single-coordinate perturbation using the fastest validated tier. This gives you the A-BLOCK
  gradient (via central FD over `lfix_incremental_at`) at ~7-45x the cost of a full-rebuild `L_fix`
  gradient, itself already ~3x cheaper than `Delta_FD` (the original optimized-value ground truth).
- `run_d4_optimized_fd.jl` — the EXISTING live KNITRO outer-loop driver you should extend, not
  replace. It already has: reduced-coordinate setup, box bounds, a single divergence-budget
  inequality constraint, best-feasible incumbent tracking, `KN_set_cb_grad`-based custom gradient
  injection (`cb_G!`), and a mandatory exact-fresh-cold-recheck at the end. Study `cb_G!` and
  `eval_grad_central_fd` closely — your hybrid gradient replaces `eval_grad_central_fd`'s BODY, not
  the surrounding KNITRO scaffold.
- `candidate_registry.jl` — load any candidate (calibration, fixed-A benchmark, upper/lower
  incumbents) from its structured artifact. Never hand-transcribe a `w` vector into a new script;
  import it from here or extend this file.
- `derivative_methods.jl::should_refresh` — an existing, unwired policy function for the
  optimized-value-FD refresh trigger (iteration count + disagreement threshold). Use it, don't
  reinvent it.

## 3. Background work dispatched, may already be complete — CHECK FIRST

Three agents were launched from commit `b5c109d`, each in an isolated worktree, each instructed to
push to its OWN branch (not `diag/fullA-d4-exact` directly) and report back:

1. **`diag/fullA-d4-exact-jach-audit`** — audits and (if safe) eliminates the dense `jac_h`
   (draw×moment×outer-param) tensor allocation in the cached/Method-B full-A path where it's unused
   by the `L_fix`/hybrid gradient method. Expected deliverable: `docs/fullA_jach_audit.md`.
2. **`diag/fullA-d4-exact-smoothed-consistent`** — a genuinely consistent smoothed full-A solve
   (log-sum-exp values AND softmax allocation probabilities at the same temperature, throughout —
   not just a smoothed min), first fixing the known non-deterministic `smoothed_frozen_adjoint_Q`
   diagnostic bug. Expected deliverable: `docs/fullA_smoothed_consistent_experiment.md`.
3. **`diag/fullA-d4-exact-phase5-sequential`** — a properly-configured (maxit>>25, `eval_fcga=no`,
   genuine 5-start multistart) sequential/profiled production run on the identical synthetic economy,
   plus a full-A-matrix reconstruction and exact-feasibility verification, replacing the RETRACTED
   invalid comparison (κ=0.0779, see `docs/fullA_d4_final_report.md` §9.3 — do not resurrect that
   number). Expected deliverable: `docs/fullA_sequential_exact_comparison.md`.

**Before starting new work**: `git fetch origin` and check whether each of these three branches
exists and what it contains. If they're done: review each agent's changes (read their doc, spot-check
their equivalence tests actually ran and passed, do not merge blindly), merge the ones that pass
review into `diag/fullA-d4-exact` (or cherry-pick), and read their findings — the jac_h and
sequential-benchmark results in particular may change what Phase 3/6 below should do (e.g., if the
sequential reconstruction beats the current full-A incumbent, that reconstructed point becomes a
mandatory warm-start for everything downstream). If a branch doesn't exist or looks incomplete/still
running, don't block on it — proceed with Phase 3 below, which doesn't strictly depend on any of the
three, and revisit the merge once they land.

## 4. Non-negotiable workflow (unchanged from prior continuations)

1. Work only on `diag/fullA-d4-exact` (or a fresh isolated worktree branch, merged back carefully).
2. Additive/mirror-don't-modify for shared production code (`cc_algo/`, `moments/`); new diagnostic
   code goes in `full_aod_diag/d4_exact/`.
3. Equivalence-test every new evaluator against a trusted baseline BEFORE using it for anything else
   (this discipline caught 4 real bugs in Phase 2 — sign errors, indexing transposes, missing power
   transforms — that would otherwise have silently corrupted every downstream result).
4. Small, reviewable commits with detailed messages (see `git log --oneline -15` for house style).
5. Update `docs/fullA_next_handoff.md` after each phase: exact commit, completed tests, artifacts,
   failed attempts, next command.
6. Don't trust filenames for D/W/seed/δ/focal-country — open the artifact and verify fields directly
   (this investigation has been burned by stale-looking-valid result files before).
7. Fresh dedicated output directory for every production-worktree run.
8. Track the best exact-feasible point over the WHOLE callback history, never just the terminal
   iterate.
9. If context becomes tight, finish the current phase, commit, and write a complete handoff — don't
   shallowly start a later phase.

## 5. Required work order

### Phase 3 — composite hybrid gradient + live outer solver (the critical path, do this first)

Build `full_aod_diag/d4_exact/composite_gradient.jl`:

**Gamma component**: `dDelta/dgamma_focal_prime`, holding the solved dual fixed, via direct scalar
AD or an analytic derivative of the fixed-dual envelope restricted to coordinate 1. Recall (from
`lfix_incremental.jl`'s own header derivation) that γ'_focal touches ONLY the counterfactual column
— no winner-switching involved, genuinely smooth — so ForwardDiff over just this one scalar (or a
hand-derived closed form, since `cf_contrib_at` in `lfix_incremental.jl` already gives you the exact
closed-form dependence) should be cheap and exact. Validate against small-step `L_fix` differences
and optimized-value one-dimensional differences at multiple h, at several outer points.

**A-block**: central FD over `lfix_incremental_at(...; tier=:incremental_o1)`, one call per ± probe
per coordinate (15 A-block coordinates at D=4). Do NOT hard-code h=0.01 universally — implement a
documented bandwidth selector based on exact tie thresholds / a target switching-mass, slope
stability at h vs h/2, and a floor/ceiling. Save the chosen h and switching statistics per gradient
call (this is a real, previously-flagged gap — the existing `run_d4_optimized_fd.jl` uses a fixed
`FIXED_H = 0.01`, and h-sensitivity was found to matter a lot near the upper candidate — see
`docs/fullA_d4_final_report.md` §4 item 4, the multi-h KKT-residual drift finding).

**Optimized-value refresh policy**: wire `derivative_methods.jl::should_refresh` into a live decision
loop — full `Delta_FD` (optimized-value) gradient on trigger (periodic + rejected-step + poor
actual/predicted-reduction + large winner-hash change + failed random-directional check), cheap
composite gradient otherwise. Exact hard value/feasibility always determines acceptance and
incumbent tracking, regardless of which gradient drove the step.

**Live driver**: extend `run_d4_optimized_fd.jl`'s exact KNITRO scaffold (`KN_set_cb_grad`, best-
feasible tracking, exact cold recheck) with a `gradient_method` switch (`delta_fd | lfix_composite |
hybrid`) and a `hessopt` switch (`sr1 | lbfgs | bfgs | product_fd_control` — new `.opt` files under
`full_aod_diag/d4_exact/`, following the existing `csw_outer_phaseB_*.opt` naming pattern, all with
`eval_fcga=no` so the requested Hessian mode is genuinely honored, not silently downgraded — verify
from the KNITRO log each time, this has bitten the investigation before). Support a wall-clock limit,
full callback trace, refresh-reason logging.

**Before any long run**: validate the composite gradient at calibration, both upper/poll-improved
points, and the lower point. Report separately: gamma error; raw A-block cosine/norm ratio;
gravity-tangent A-block cosine/norm ratio (per `docs/fullA_performance_profile.md` §4's finding that
FULL-vector cosine is not a meaningful metric — the gamma component dominates the norm); A-only
random directional predictions; optimized-value reference h and switch mass.

### Phase 4 — fair wall-clock algorithm frontier at D=4/W=8000

Common starts, common exact settings, fixed wall-clock budgets (30s/60s/180s/~324s or closest
feasible). Compare AT LEAST: (1) optimized-value FD + product-FD Hessian-vector control (the
historical baseline); (2) optimized-value FD + SR1; (3) optimized-value FD + L-BFGS; (4) composite
hybrid + SR1; (5) composite hybrid + L-BFGS; (6) an exact-value derivative-free local poll/pattern-
search comparator near the best incumbent. Save best-exact-feasible-κ-vs-wall-time, vs-exact-value-
calls, inner solves, gradient calls/refreshes, rejected probes, terminal+best-feasible exact
rechecks. Do not rank by KNITRO's own reported optimality alone. Produce
`docs/fullA_algorithm_frontier_v2.md`.

**Minimum bar for this continuation if context runs short**: at least ONE wall-clock-matched
comparison between the existing product-FD/optimized-value control and hybrid-`L_fix`+SR1-or-LBFGS
— this was the previous continuation's own stated minimum target and was not reached; do not let a
third continuation also fall short of it without an explicit, documented reason.

### Phase 6 — complete D=4 through the gamma profile (do after Phase 3/4, using the validated hybrid)

`profile_Delta(g) = min_A Delta(g,A)` subject to exact gravity + genuine bounds, using the composite
hybrid for local A-minimization, optimized-value FD for refresh/final verification, warm continuation
across g, starting from calibration/fixed-A/current-incumbents/reliable-sequential-point (if Phase 5's
agent produced one — check §3). Coarse-to-fine over the theoretical g interval; do not assume
monotonicity; refine every crossing of `profile_Delta(g)=δ`, local-minimum exchanges, discontinuities.
Upper direction: polish the three archived poll-improved points (see
`results/fullA_d4/1b2a3a0/phaseA_upper_revalidation/step6_poll.csv`) plus the best profile crossing;
require a fresh cold recheck, multi-h optimized-value directional checks, no material improvement
under documented A-tangent/profile-direction polls. Lower direction: continue from g=1 and neighboring
profile points with a real wall-clock budget (the existing lower run is `BEST_FEASIBLE_STALLED`,
Δ−δ=−0.101, genuinely just out of iterations, not numerically broken). Classify every result:
`ROBUST_LOCAL_CANDIDATE` / `H_BANDWIDTH_KKT_CANDIDATE` / `BEST_FEASIBLE_STALLED` /
`INFEASIBLE_OR_UNRESOLVED`. Never call a local candidate a global bound.

### Phase 7 — nested-W stability (after Phase 6 produces stable candidates)

Deterministic W=80000 draw pool, prefix-nest W=8000/20000/80000 from it (the existing
`docs/fullA_d4_W_stability.md` used INDEPENDENT, not nested, draws — a documented, flagged
approximation, not the rigorous design; fix this properly here). Re-evaluate fixed-A benchmark,
current best upper/lower candidates, reliable-sequential point (if available) at each W: cold+warm
exact recheck, divergence/moment residuals, feasibility, winner-switch distribution, h-grid re-
selected per W to hold switching mass roughly constant. If a point loses feasibility at larger W,
continue/re-optimize from it there — do not call that a method failure.

### Phase 8 — staged D scaling, GATED (do not skip the gate)

D=6 pilot allowed ONLY if: block-local/incremental equivalence passes (✅ already true, Phase 2);
composite A-gradient passes tangent-block validation (Phase 3); the hybrid solver improves the
exact-feasible objective reliably at D=4 (Phase 4); no KNITRO option fallback occurs; projected wall
time is acceptable. One upper + one lower short pilot, full logging, optimized-value validation
refreshes, treated as a computational pilot not a headline bound. D=8/10 conditional on D=6 success.
D=20 explicitly deferred — do not attempt it this continuation.

## 6. Final questions your report must answer

1. Does the composite gradient provide correct gamma and tangent-A guidance without optimized-value
   FD every iteration? (Validate, don't assume — Phase 2's own bugs are a cautionary tale.)
2. At equal wall time, does hybrid SR1/L-BFGS beat product-FD Hessian-vector and optimized-value FD?
3. Do the gamma profile and continuation produce robust local upper AND lower candidates at D=4?
4. How do those candidates move under nested W=20000 and W=80000 draws?
5. Is a D=6 pilot justified, and what exact gates were passed?
6. What did the three background-agent workstreams (§3) find, and how did their findings change
   anything above (especially: does a properly-run sequential reconstruction beat the current full-A
   incumbent — if so, that's the single most important finding to act on first)?

Be explicit about uncertainty. Distinguish exact feasibility, bandwidth stationarity, robust local
optimality, and global optimality throughout. Preserve the exact full-A estimand — no positive-
temperature/smoothed solution or intermediate optimizer state is ever reported as a final bound.
