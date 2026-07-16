# Handoff: production consolidation done + 3 next-step experiments (2026-07-16)

Paste this whole file, or point a new Claude at it, to continue. Everything below "Current
state" is background; everything under "Next steps" is the actual work still to do.

## Current state (DONE, committed, pushed)

Repo: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf`, branch
`feature/sequential-inversion-perf`, commit `d6463f1` (pushed to
`origin/feature/sequential-inversion-perf`). KNITRO env (only 4 licenses total on
`demand.mit.edu` — see "Concurrency" below before launching anything):

    export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
    export KNITRODIR=/opt/shared_sw/knitro/14.2.0
    export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
    export PATH="$HOME/.juliaup/bin:$PATH"
    cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf

### What's in production now

- **`recover_lfd` nStatus bug fixed** in all 11 copies repo-wide (was silently accepting
  KNITRO-UNBOUNDED dual solves; see `head_to_head/HANDOFF_2026-07-16_recover_lfd_bug.md` for the
  full bug writeup and the 3 head-to-head points it corrupted).
- **`DUAL_WARM_MODE` (default `:persist`)** in `sequential_gravity/run_profiled_production.jl`:
  the inner CC dual solve (`recover_lfd`'s KNITRO call) now warm-starts across `seq_gravcol`'s
  within-theta re-solves AND across outer-loop theta points, via a global cache
  (`_DUAL_CACHE`/`_dual_warmstart_for`). Options: `:cold` / `:reset_per_theta` / `:persist`, set
  via `DUAL_WARM_MODE` env var or `DUAL_WARM_MODE[] = :mode` directly. Validated (controlled
  3-point test, `head_to_head/experiment_dual_warmstart_direct.jl`) to never change the accepted
  result and never increase KNITRO iteration count. Safe as a global (not task-local) only
  because nothing in this codebase calls `seq_gravcol`/`recover_lfd` concurrently across threads
  — see the comment at its definition if that ever changes.
- **Destination-share inversion now defaults to Optim.jl's `NewtonTrustRegion`**
  (`sequential_gravity/profiled_gravity.jl`, `invert_destination`'s `method=:trustregion`
  default) — done by a different session concurrently with this one; validated more reliable and
  3-15x faster than the old hand-rolled LM-damped Newton at tight tolerance. Old solver still
  available via `method=:handrolled`.
- **Hard-max (rho=0) independent post-verification**, `sequential_gravity/hardmax_verify.jl` —
  also done by a different session concurrently with this one. Wired into `run_one_bound` via
  `VERIFY_HARDMAX` (default **on** — adds ~1-3 min per saved checkpoint at D=20/W=80000). Do NOT
  remove or accidentally disable this; it's a real, validated check (rho-continuation homotopy
  against the TRUE hard-argmax model, not just the smoothed rho=2e-3 approximation the main
  pipeline optimizes against) and the user specifically flagged it as something not to lose.
- **`sequential_gravity/run_production.jl`** — new unified entry point. `METHOD` env var (default
  `LC`) dispatches to `run_profiled_production.jl` (LC, local-constrained/KNITRO — the winner of
  the corrected 4-method head-to-head comparison, see `head_to_head/FINAL_REPORT_2026-07-16.md`)
  or delegates to `head_to_head/run_lu.jl` / `run_gc.jl` / `run_gu.jl` for the other 3 methods.
  **Known limitation, not yet fixed**: LU/GC/GU's driver scripts run the SPECIFIC 3-target
  (T1/T2/T3), 3-start head-to-head design they were built for — they do NOT (yet) read
  `DELTA_GRID`/`BOUND` the way LC's own batch loop does. Unifying them onto a fully generic
  interface is real, undone work; flagging so it's not assumed to already work.
- Full D=20 real-data 4-method head-to-head comparison + $\delta^*(\gamma')$ schedule tooling +
  a large backlog of derivative/gradient-method diagnostics from earlier sessions were ALSO
  swept into this commit (they were sitting uncommitted in the working tree; see "Things you may
  have forgotten" below).

### RESOLVED: production defaults now match the config that won the head-to-head comparison

(Originally flagged as a discrepancy in this handoff's first version; fixed same-day on user
request.) `run_profiled_production.jl`'s defaults are now `GRADIENT_METHOD=fixed_dual_fd_full`,
`USE_VAR_SCALING=true`, `SCALING_POWER=1.0` -- matching the LC runs that actually won the
corrected head-to-head comparison (`FINAL_REPORT_2026-07-16.md`). The old `pointwise_ad` default
is still available (`GRADIENT_METHOD=pointwise_ad`) for A/B comparison only -- it's the OLD,
cheap outer-constraint gradient proved to silently drop the winner-boundary/Dirac term and report
false/premature convergence (re-confirmed THIS session: a pointwise_ad+no-scaling run converged
in 2 outer iterations to kappa=0.0377 vs fixed_dual_fd_full's genuine 0.0821 at the identical
delta=1.0 budget). Do not use it for anything reported. Task 1 below no longer needs to set these
3 env vars explicitly -- they're the default now -- but it doesn't hurt to set them anyway for
self-documenting run commands.

### Common-marginals restriction (CDW eq. 35/36) — now wired into production too

A separate concurrent session had implemented and validated this (D=4) in a SEPARATE, UNCOMMITTED
working copy (`/bbkinghome/edav/gravity_robustness/trade_robustness_common_marginals`, same
`origin` remote, branch `feature/common-marginals`, but genuinely uncommitted — not merely on an
unmerged branch). Ported into `trade_robustness_modular_perf/sequential_gravity/
run_profiled_production.jl` this session: `common_marginals_moments.jl` (the moment-construction
module, copied verbatim) plus the `CM_L`/`CM_REF`/`CM_EQ36` env vars and the `nCM`-threading edits
through `recover_lfd`/`seq_gravcol`/`make_stateful_moments`/`outer_solve_nested_cached`. Off by
default (`CM_L=0`) — smoke-tested to be an EXACT no-op vs the pre-CM baseline
(`seq_gravcol(θr0; δ=1.0)` gives the identical `R_mean=2.0682e-05` with or without this port).
`CM_L>0` smoke-tested to actually engage (57 extra moments at `CM_L=3`, `D=20`) and still solve
successfully. Also adopted 2 other real fixes from that same worktree: an opt-file absolute-path
fix in `recover_lfd` (bare relative filenames silently fail once anything in this codebase's own
setup chain `cd()`s elsewhere — see `setup/setwd.jl`, confirmed responsible for a wrong-directory
output earlier this session too) and a `BLAS_NUM_THREADS` shared-machine courtesy cap (default 19).

**⚠️ New, real gap found while porting, not present before**: `GRADIENT_METHOD=fixed_dual_fd_full`
(now the production default, see above) is **NOT compatible with `CM_ENABLED=true`** —
`make_seq_div_grad_fn_full!`/`full_fixed_dual_criterion.jl` (the corrected-gradient machinery)
hard-assumes exactly `D+2` moments and was never extended to the extra CM columns (it was written
against the `pointwise_ad` default that was in effect before this session flipped the default).
Attempting `CM_L>0` together with `fixed_dual_fd_full` now fails FAST with a clear, actionable
error (added this session, at the `outer_solve_nested_cached` dispatch site) rather than silently
producing a wrong gradient or a cryptic deep assertion — but it does still fail. **If task 1/2/3
below ever need `CM_L>0`, set `GRADIENT_METHOD=pointwise_ad` explicitly** (the only gradient path
currently validated with CM on), accepting that gradient's known weaknesses (see above), OR do the
real work of extending `full_fixed_dual_criterion.jl`'s frozen-moments construction to include the
CM block first. Tasks 1/2/3 as specified by the user do NOT mention CM_L, so this likely doesn't
block them — flagging so nobody is surprised by the crash if CM is turned on later.

### Things you may have forgotten (surfaced during a `git status` audit before committing)

The working tree had a large amount of uncommitted work from EARLIER sessions (not just this
one) sitting alongside this session's changes — all now committed together in `d6463f1`. Worth
knowing this is now IN git, in case any of it was meant to stay as scratch-only:
- `sequential_gravity/derivative_diagnostics/` (~45 `.jl` files + several `.md` reports) — the
  gradient-method/winner-boundary-derivative investigation series (Parts 1-5b, hardmax LP
  attempts, D=20 delta-grid verification, etc.)
- `sequential_gravity/global_opt/` — `bbo_common.jl` and the BlackBoxOptim D=4/D=20 drivers
- `diagnostics/` — an A-profile-optimality check
- A dozen or so root-level `run_d20_*.sh` one-off overnight-run launcher shell scripts
- `HANDOFF_D20_realdata_overnight_run.md`, `check_aod_basin.jl`, `dump_delta_star_csv.jl`,
  `dump_results.jl`

None of this was touched/modified by this session — it was pre-existing uncommitted work,
committed as-is alongside this session's changes because leaving it uncommitted risked losing it.
**Excluded from the commit** (via a new `.gitignore`): run logs (`*.log`, `*_run.log`),
`*_ALLDONE` sentinels, and JLD2 checkpoint/result directories (`batch_out*/`, `coldstart_*/`,
`d20_upper*/`, `out_lc/`, `out_lu/`, `out_gc/`, `out_gu/`, `out_lu_multistart50/`, `**/logs/`,
generic `*.jld2`) — these are regenerable outputs, not source, and would have bloated the repo.
Two exceptions force-added because they're seed/input data needed for reproducibility:
`head_to_head/shared_starts.jld2` (the Astar/rand1/rand2 starting points) and
`head_to_head/lu_multistart_points.jld2`.

## Next steps (NOT done — this is the actual handoff)

**Concurrency, confirmed by the user (2026-07-16): exactly 4 concurrent KNITRO instances, one
each for**: (1) task 1's upper bound, (2) task 1's lower bound, (3) task 2 (W-robustness), (4)
task 3 (full-A_od). I.e. task 1's own upper and lower bounds run as 2 SEPARATE concurrent
processes/instances, not one process doing both bounds in sequence -- run task 1's driver twice
(`BOUND=upper` and `BOUND=lower`, separately launched), not once with `BOUND=both`. This uses all
4 licenses at once; don't launch a 5th concurrent KNITRO process on top of these 4.

### Task 1: production-ready graph (W=80,000)

Run LC, **both bounds, as 2 separate concurrent instances** (`BOUND=upper` in one process,
`BOUND=lower` in another -- see concurrency note above, do NOT use `BOUND=both` in a single
process for this task), delta grid **0.1, 1.0, 2.0, 5.0**, **5 starts each** (one of the 5 =
Astar/`θr0`'s own A_od — i.e. `Acol_star`), W=80,000. For delta > 0.1, warm-start from the
**best-feasible solution of the previous delta** (not a cold restart at each delta) —
i.e. delta-chaining PER START (5 independent chains, each running 0.1→1.0→2.0→5.0), not one
chain per delta.

**This driver does not exist yet — it needs to be built**, combining two already-existing,
validated patterns:
- `run_profiled_production.jl`'s own batch loop (bottom of the file, `SKIP_BATCH_LOOP`-gated):
  already does exactly the delta-chaining-from-best-feasible logic needed, but only for ONE
  start (`θr0` cold) per bound direction.
  `head_to_head/run_lc.jl` (`lc_solve`/`lc_target_winner`): already does multi-start (3 starts:
  Astar/rand1/rand2, from `shared_starts.jld2`) and per-(target,start) checkpointing/resume, but
  only at ONE delta per target (T1/T2/T3, no cross-delta chaining) and upper-bound-only.
- You need: 5 starts × 2 bounds × 4 deltas = 40 solves, each individually checkpointed/resumable
  (follow `run_lc.jl`'s `load_done`/existing-file-skip pattern), with warm-starting BOTH within a
  chain (across delta, from that same start's own best-feasible endpoint) and NOT across starts
  (each of the 5 starts should be an independent chain — don't let start 2 warm-start from start
  1's endpoint, that would collapse the multi-start's whole point of covering the basin).
- Need 2 more starts beyond Astar/rand1/rand2 — `generate_shared_starts.jl` shows the existing
  pattern (log-normal perturbation of `Acol_star`, `Random.seed!(20260715)`, `sigma=0.5`/`1.0` for
  rand1/rand2). Either extend that script to also generate rand3/rand4 at new sigmas (keep them
  in the SAME shared JLD2 file, same seed continuation, so this stays reproducible and consistent
  with task 2/3 below reusing the identical 5 starts), or draw your own — but keep it consistent
  across tasks 1/2/3 so the comparison is fair, per the user's own framing for task 2/3.
- Remember the `GRADIENT_METHOD=fixed_dual_fd_full USE_VAR_SCALING=true SCALING_POWER=1.0`
  point above.
- `VERIFY_HARDMAX` stays on (default) — don't disable it to save time; it's part of what makes
  this "production-ready" rather than just a KNITRO-self-reported number.
- Output: presumably feeds a similar delta-vs-kappa graph to the one already built this session
  (see the published artifact from earlier in this conversation, or `head_to_head/
  FINAL_REPORT_2026-07-16.md`'s table) but with the FULL 4-point delta grid and BOTH bounds this
  time (the existing artifact only has upper-bound points at 3 deltas, from the single-chain
  official comparison).

### Task 2: W-robustness check (W=800,000)

Re-run task 1's design (same 5 starts, same delta grid 0.1/1.0/2.0/5.0, same warm-start
chaining) but **W=800,000** and **upper bound only** (the user's own words — task 2 is narrower
than task 1, lower bound not requested here). Separate KNITRO instance/process from task 1.
Reuse task 1's driver with `WVAL=800000 BOUND=upper` rather than writing a second driver.

Relevant prior finding to sanity-check against: project memory `d20-realdata-w-sensitivity` —
"W=8000 silently understates kappa for delta≥1 vs W≥80k" — so this comparison (80k vs 800k) is
specifically checking whether 80k itself is ALSO still under-resolved, one level up from that
earlier finding. If kappa values move meaningfully between W=80k and W=800k here, that's a real,
reportable finding, not noise — don't wave it away.

### Task 3: full-A_od-in-outer-loop robustness check (no sequential linearization)

Put all of A_od directly in the outer KNITRO loop (gravity becomes a normal outer EQUALITY
constraint, `has_gravity=true`) instead of the sequential/profiled linearization scheme
(`seq_gravcol`'s own iterative refinement). This existing, different formulation is already
implemented — do NOT reimplement from scratch:

- `full_aod_diag/run_fullA_D10_production.jl` (despite the "D10" name, it's D-configurable via
  `DVAL` — already has a `DELTA_GRID`/`BOUND` batch loop with delta-chaining, same pattern as
  `run_profiled_production.jl`'s own). Uses `PsiObjectiveBundleImplicitMethodB_fullA`
  (`full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl`) + analytic tariff-gravity gradient
  (`full_aod_diag/gravity_tariff.jl`).
- **Architecturally different from the sequential method — the recover_lfd bug does NOT apply
  here** (gravity is a direct outer constraint, not an LFD/dual reweighting via `recover_lfd`).
  Already warm-starts its own inner solve (`use_cached_x=true` baked in), so no analog of
  `DUAL_WARM_MODE` is needed here either.
- **Two real gaps to fix before this is usable for a fair D=20 real-data comparison** (checked
  this session, not yet fixed): `run_fullA_D10_production.jl`'s `params` tuple hardcodes
  `fakeData=1` (synthetic Fréchet data) and `W=8000`/`Jac_W=8000` — NOT read from `FAKEDATA`/
  `WVAL` env vars the way `run_profiled_production.jl` does. Add that env-var wiring first (small,
  mechanical change, same pattern as the other file) — otherwise this can't run on the same real
  D=20 data / W=80000 the other 3 tasks use, and the comparison wouldn't be fair.
- Needs the SAME 5-start, delta-chained-per-start design as task 1 for a fair comparison (the
  point of this task, per the user, is specifically to check whether the sequential
  linearization buys anything over just doing this directly — an apples-to-apples multi-start/
  delta-grid design matters more here than in tasks 1/2, since this IS the comparison). Same
  delta grid 0.1/1.0/2.0/5.0; confirm with the user whether they want both bounds here too (their
  message says "delta = 0.1, 0.2, 2.0, 5.0 etc" for task 3 — note "0.2" looks like a likely typo
  for "1.0" given task 1/2 both use 0.1/1.0/2.0/5.0; confirm rather than silently assuming).
- Separate KNITRO instance/process from tasks 1 and 2.

## Quick-start

```bash
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt KNITRODIR=/opt/shared_sw/knitro/14.2.0 \
  LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH PATH="$HOME/.juliaup/bin:$PATH"

# Confirm the consolidated state (should be d6463f1 or later, clean or only-new-scratch-outputs)
git log --oneline -1
git status --short   # should show nothing except possibly new *.jld2/*.log run outputs

# The unified entry point (LC default; see run_production.jl's own header for METHOD options)
METHOD=LC julia --project=. sequential_gravity/run_production.jl   # NOT yet multi-start -- see task 1

# The existing (single-start) LC batch loop directly, for reference on the delta-chaining pattern
# to replicate with 5 starts:
FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true GRADIENT_METHOD=fixed_dual_fd_full \
  USE_VAR_SCALING=true SCALING_POWER=1.0 DELTA_GRID=0.1,1.0,2.0,5.0 BOUND=both \
  julia -t 19 --project=. sequential_gravity/run_profiled_production.jl
```
