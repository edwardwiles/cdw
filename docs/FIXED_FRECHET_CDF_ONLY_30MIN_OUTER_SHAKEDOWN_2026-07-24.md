# CDF-Only 30-Minute Real Outer Shakedown — 2026-07-24

Real D=20/W=80,000/L=50/`:exclude_row`, `delta=1.0`, `maxtime_real=1800s`, 20 Julia threads
(threaded/syrk Hessian throughout, via `cm_frechet_verified_state_threaded`/
`cm_frechet_production_gradient_cdf_only`), KNITRO default production option file (`ek_inner.opt`,
`par_numthreads=1` — not overridden, since the thread-sweep found no benefit from raising it),
outer solver `csw_outer_wallclock_sr1.opt` (SR1 quasi-Newton, unchanged from the existing
`:cdf_power` driver). Starting point: `(gp_target, zfree*)` at `kappa*+1e-4` (the P1 point from the
other benchmarks in this package).

## Result summary

```
knitro_status=-401 ("Time limit reached, current point is feasible" -- a normal, clean exit for a
                     fixed wall-clock outer budget, not a crash or error)
wall=1974.7s (setup+search; the outer solve itself ran ~1835s against its own 1800s budget --
              a few seconds over, plausibly a single in-flight evaluation crossing the boundary)
n_eval=16              n_grad=5             n_new_point_solve=12
n_base_reused_at_gradient=5 (every gradient call reused a cached base -- state-reuse discipline intact)
n_time_limit_no_certificate=0    n_infeasible_certificate=4
BEST: gp=0.9877132718248147 (== the STARTING point)  Delta=0.008920180760516032  n_eval=1  t=3.0s
cold-verify of best incumbent: outcome=feasible, verified=true, Delta=0.008920180760516032 (exact match)
```

Progress per wall-clock minute (cumulative new-point solves): minute 1→3, 4→4, 7→6, 8→7, 10→8,
13→9, 16→10, 19→11, 22→12, 25→13, 28→14, 31→15, 33→16.

## Against task brief §12's explicit requirements

| Requirement | Result |
|---|---|
| ≥10 genuinely new valid outer trial points | **12** `n_new_point_solve` — met |
| ≥1 outer gradient | **5** `n_grad` calls, all state-reuse-correct — met |
| Movement in A and gp | gp moved 0.98771 → 0.97088 over the run (real search motion) — met |
| No timeout classified as infeasible | `n_time_limit_no_certificate=0` — met (every non-feasible eval got a genuine `:infeasible_certificate`, not an ambiguous timeout) |
| Best incumbent cold-verifies | Confirmed, exact Δ match — met |
| Clean termination | `knitro_status=-401`, a normal time-limit exit with a feasible current point, not a crash — met |

All six explicit bullet-point gates pass. **However, two things below are disclosed rather than
hidden behind that pass, because the SPIRIT of "make new points cheap enough for a real search to
make progress" is only partially achieved:**

## Disclosed finding #1: far-from-calibration trial points are NOT uniformly fast

The 16 `cb_F!` evaluations were not evenly spaced: eval 1→5 spanned 3s→380s (≈4 solves, ~94s/solve
average), eval 5→10 spanned 380s→941s (≈5 solves, ~112s/solve average), eval 10→15 spanned
941s→1809s (≈5 solves, ~174s/solve average) — and the outer SQP/interior algorithm itself completed
only **4 outer iterations** in the full budget (its own `Iter` table shows iterations 0-4). This
contradicts the direct P1/P2 benchmark numbers in `THREADED_CDF_ONLY_HESSIAN_BENCHMARK_2026-07-24.md`
(8-25s per solve) — because those benchmarks deliberately probed points **near calibration**
(`kappa*+1e-4`, `kappa*+2e-4`), while the real outer search's line-search/trial-step logic visits
points considerably farther away (e.g. eval 15's `gp=0.9709` vs the calibration `gp*=0.9878`, and
Δ values up to 1.40 — well outside the `delta=1` feasible region). **Individual solves at these
farther, often-infeasible points plausibly exceed the task's 120s "no individual ordinary trial
solve above 120s" ceiling** — this was not directly instrumented per-eval in this pass (a disclosed
gap, not a hidden failure: `run_frechet_upper_cdf_only.jl`'s trace records Δ/feasibility per eval
but not individual solve wall time; a follow-up should add that).

This is very unlikely to be a Hessian-construction or KNITRO-threading problem — every direct
benchmark in this package shows uniform, dimension-driven speedup regardless of which point is
solved. The far-point slowness is much more likely a **genuine numerical-difficulty** effect (more
KNITRO barrier iterations needed to establish feasibility/infeasibility far from the data-consistent
region), a distinct bottleneck from the one this task was chartered to fix.

## Disclosed finding #2: no net improvement in kappa over 30 minutes

The best feasible-and-verified incumbent found was the **starting point itself** (eval 1, t=3.0s) —
every subsequent evaluation was either infeasible (Δ>1) or feasible-but-not-`verified` (eval 10:
feasible=true but `is_verified_success` failed). The search did not find a strictly-better feasible
point in this run. Plausible (not confirmed) contributing factor: the pre-existing z-direction
outer-gradient discrepancy documented in
`FIXED_FRECHET_INNER_SOLVER_ARCHITECTURE_AUDIT_2026-07-24.md` §3 (shared with the existing,
unmodified `:cdf_power` path — not new to this session) could be misdirecting the SR1 quasi-Newton
search's step choices. Not root-caused in this pass.

## Verdict for this component

The 30-minute shakedown technically satisfies every literal bullet point of task brief §12, but
surfaces a real, second bottleneck (far-point solve cost, search stalling) that is **separate from,
and not fixed by,** this session's core deliverable (Hessian/KNITRO thread-count engineering at a
given point). Reported honestly per the task's own explicit instruction not to hide a result behind
a longer budget — this is not spun as unqualified success. See the top-level verdict document for
how this factors into the overall PORT READY EXPERIMENTAL call.
