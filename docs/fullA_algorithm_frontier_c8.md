# Continuation 8, workstream B: algorithm frontier RERUN (post compressed-live + winner-accelerator)

**CORRECTION (added by the coordinating session after merge):** this workstream branched from
`d6e3b05`, which predates Section 8's registry update. Every "canonical lower incumbent" comparison
below (notably the "**-14.6% relative improvement**" framing in §4) is against the OLD registered
`lower_lfixcomposite_fast_sr1_300s` (κ=0.005428799948779983), which Section 8
(`docs/fullA_d4_final_candidate_verification_c8.md`, merged before this report) already superseded
with **`lower_v2`, κ=0.004387827651021192**, now the registered headline lower incumbent in
`candidate_registry.jl`. Against that current baseline, this report's `lfixcomposite_lbfgs_compressed`
finding (κ=0.004634148517618675) is **not** an improvement — 0.004634 > 0.004388, i.e. it is a
*looser* (worse) lower bound than `lower_v2` by relative +5.6%. The finding is still worth keeping on
record (a genuinely different, LBFGS+compressed-found local optimum, independently cold-hard-
rechecked, that a plain multi-config frontier run reached from a generic start in ~22s without any
of Section 8's targeted warm-starting from the gamma-profile branch work) — it just is not the
current best-known lower candidate. All other numbers/comparisons in this report (the upper-direction
table, the JIT "stuck-at-15s" finding, fallback counts, stationarity diagnostics) are unaffected by
this correction and stand as reported.

---

**Supersedes `docs/fullA_algorithm_frontier_v2.md`.** That report predates two pieces of
infrastructure landed earlier in this same continuation-8 session on `diag/fullA-d4-exact`
(base commit `d6e3b05`, tip of the merged Wave 1 work):

- **Compressed live integration** (`docs/compressed_live_integration_report.md`): opt-in
  `moment_representation=:compressed` on `evaluate_fullA_fast`, wiring the O(W·D) compressed
  winner-form inner-dual FG callback into the live solver (default stays `:dense`).
- **Winner accelerator live wiring** (`docs/winner_accelerator_live_wiring.md`): coordinate-
  specialized top-3 update (now the **default**, `multi_method=:top3`) and the winner-margin
  certificate, both baked silently into every `composite_gradient_at`/`composite_gradient_at_fast`
  call this report's `lfix_composite`-family configs make — v2's own numbers were measured
  *before* these landed.

This rerun also adds a 5th method not in v2 at all: **consistent smoothed AD**
(`smoothed_consistent.jl`'s Method 1 — ForwardDiff of the fixed-dual envelope at a
freshly-solved smoothed inner-dual base), and runs **both directions at the full
15/30/60/120/300s checkpoint grid** (v2 covered upper only, at a single 60s budget).

## 1. Setup

D=4, W=8000, δ=1, common start `w0` (from `ctx.θ0_up`, identical construction to
`run_d4_optimized_fd.jl`/`run_phase4_frontier.jl` — same `d4_exact_setup()` determinism,
verified: every "stuck at 15s" row below reports the *identical* starting κ=0.06420809457817289
across all three lfix-family configs and both directions). Machine `demand.mit.edu`,
`JULIA_NUM_THREADS=20`. Subprocess-per-(config,budget) — reused unmodified from
`run_phase4_frontier.jl`'s own "clean KNITRO/obj state" design (new driver:
`full_aod_diag/d4_exact/run_d4_optimized_fd_c8.jl`; new orchestrator:
`full_aod_diag/d4_exact/c8_frontier_run.jl`).

**5 configs** (standing brief §6):

| label | gradient | hessian | moment representation |
|---|---|---|---|
| `lfixcomposite_sr1_dense` | `lfix_composite` | SR1 | dense |
| `lfixcomposite_sr1_compressed` | `lfix_composite` | SR1 | **compressed (new)** |
| `lfixcomposite_lbfgs_compressed` | `lfix_composite` | L-BFGS | **compressed (new)** |
| `deltafd_lbfgs_dense` | `delta_fd` (optimized-value FD) | L-BFGS | dense (reference control, unchanged from v2) |
| `smoothed_ad_sr1` | **consistent smoothed AD (new)** | SR1 | n/a |

`smoothed_ad_sr1` reuses `smoothed_consistent.jl`'s exact matched value+gradient construction
(re-solved smoothed inner dual for the value, `ForwardDiff.gradient` of `smoothed_fixed_dual_L`
for the gradient — "Method 1" in `docs/fullA_smoothed_consistent_experiment.md`, the same method
that document's own outer solve used), but as a **single fixed rho** for the whole wall-clock-
budgeted KNITRO solve, not the original 5-stage rho homotopy — a documented simplification
(driver header comment), required because this frontier's checkpoint structure is one continuous
wall-clock-budgeted solve per config, not a staged one. Rho is picked once per run via the exact
same "coarsest empirically-feasible-at-w0" scan `run_smoothed_homotopy.jl` validated, then a
decade finer.

**Include-order hazard found and documented** (driver header): loading both the
compressed_live/oracle_fast chain and the smoothed_consistent chain in the *same* Julia process
re-executes `module CounterfactualSensitivity`'s own definition, corrupting dispatch process-wide
(`UndefVarError`/export-ambiguity, confirmed empirically before committing to a design). Fixed by
loading exactly one chain per subprocess, selected from `D4X_GRADIENT_METHOD` before any
`include` happens — free, since every config already runs as its own subprocess.

**Both directions, full checkpoint grid, not trimmed**: KNITRO licenses up to 5 concurrent
instances (memory: `knitro-concurrency-limit-corrected`) and each orchestrator only ever runs one
subprocess at a time, so upper and lower were launched as two concurrent orchestrator processes —
2 concurrent KNITRO instances, well inside the validated bound — costing no extra wall-clock time
versus running one direction alone. Nothing was cut for time.

## 2. Headline table: best-feasible κ, method × checkpoint × direction

All κ are the tracked **best-feasible** value, independently re-verified via a **cold**
(`warm=false`) call to `evaluate_fullA` (the unmodified hard oracle) — never the raw KNITRO
terminal iterate, never a smoothed/compressed surrogate value. Every cell below is read directly
from that config's own `summary.txt`.

### Upper direction (find_smallest=true; larger κ = better)

| checkpoint | `lfixcomposite_sr1_dense` | `lfixcomposite_sr1_compressed` | `lfixcomposite_lbfgs_compressed` | `deltafd_lbfgs_dense` (control) | `smoothed_ad_sr1` |
|---|---|---|---|---|---|
| 15s | 0.064208 (stuck, outer=0) | 0.064208 (stuck, outer=0) | 0.064208 (stuck, outer=0) | 0.147523 | 0.167035 |
| 30s | 0.172214 | 0.172169 | **0.172345** | 0.170323 | 0.167035 |
| 60s | 0.172457 | **0.172495** | 0.172345 | 0.171774 | 0.167035 |
| 120s | 0.172457 | **0.172495** | 0.172345 | 0.171774 | 0.167233 |
| 300s | 0.172457 | **0.172495** | 0.172345 | 0.171774 | 0.167233 |

Winner at every checkpoint ≥30s: a `lfix_composite` variant. **`lfixcomposite_sr1_compressed`
wins outright from 60s on** (κ=0.1724951449325095), a small but genuine +0.022% relative
improvement over the canonical dense incumbent (κ=0.17245688540655113,
`docs/fullA_continuation7_handoff.md`) reproduced exactly here by `lfixcomposite_sr1_dense`
(γ'=0.8926359584642946, matching the documented γ'=0.8926359585 to the printed digits). Both
points independently cold-rechecked feasible (`Delta_minus_delta` = -7.6e-6 dense / -6.5e-5
compressed) with clean stationarity (`eta_nonneg=true`, `residual_relative`≈0.0021-0.0023,
`n_nonfinite_probes=0` — see §5).

### Lower direction (find_smallest=false; smaller κ = better/tighter)

| checkpoint | `lfixcomposite_sr1_dense` | `lfixcomposite_sr1_compressed` | `lfixcomposite_lbfgs_compressed` | `deltafd_lbfgs_dense` (control) | `smoothed_ad_sr1` |
|---|---|---|---|---|---|
| 15s | 0.064208 (stuck, outer=0) | 0.064208 (stuck, outer=0) | 0.064208 (stuck, outer=0) | 0.005682 | 0.005255 |
| 30s | 0.005429 | 0.005429 | **0.004634** | 0.005589 | 0.005067 |
| 60s | 0.005429 | 0.005429 | **0.004634** | 0.005589 | 0.005067 |
| 120s | 0.005429 | 0.005429 | **0.004634** | 0.005589 | 0.005067 |
| 300s | 0.005429 | 0.005429 | **0.004634** | 0.005589 | 0.005067 |

`lfixcomposite_sr1_dense`/`_compressed` reproduce the documented canonical lower incumbent
(κ=0.005428799948779983, `docs/fullA_continuation7_handoff.md`) essentially exactly (dense
0.005428799948779983 bit-identical; compressed 0.0054287999489507355, agrees to ~8e-14).
**`lfixcomposite_lbfgs_compressed` wins outright at every checkpoint ≥30s, and by a much larger
margin than the upper-direction win**: κ=0.004634148517618675, a **-14.6% relative** improvement
over the canonical lower incumbent. See §4 for the flag this deserves before being adopted as a
new headline number.

## 3. "Stuck at 15s" — a real, reported finding, not a bug

Every `lfix_composite`-family config shows **zero outer iterations and κ frozen at the starting
point's own value** (0.06420809457817289, both directions) at the 15s checkpoint, while
`deltafd_lbfgs_dense` (κ=0.1475/0.00568) and `smoothed_ad_sr1` (κ=0.1670/0.00526) make real
progress in the same 15s. Root cause, confirmed by direct inspection (not assumed): the
`lfix_composite`-family callback path is the FIRST-EVER call in a fresh process to
`composite_gradient_at`/`composite_gradient_at_fast` plus everything Wave 1 added underneath them
this session (winner-margin certificate, coordinate-specialized top-3 cache, compressed live
wiring) — JIT compilation of that newly-expanded call graph consumes nearly the entire 15s budget
before KNITRO gets a second outer iterate. `time_to_first_feasible_s` for these three rows is
~16-19s (i.e., *longer than the nominal budget* — the one-and-only F-evaluation recorded IS
already feasible, at the starting point, and its timestamp is essentially the whole wall clock).
This reverses completely by 30s (66-67 outer iterations for the composite family vs. 32 for
`deltafd_lbfgs_dense`), consistent with v2's own historical finding once compilation is
amortized. Reported explicitly so the 15s column isn't misread as "composite gradient is
slower" — it measures compile time at that checkpoint, not algorithm performance.

## 4. Notable finding: a materially better lower-direction candidate

`lfixcomposite_lbfgs_compressed`, lower direction, converges (KNITRO status -101, genuine
convergence, not a time/iteration-limit stop) by 25 outer iterations / 22.1s wall to:

```
gamma_focal_prime = 0.9972169282608725
kappa             = 0.004634148517618675
Delta_dual        = 0.9999985430459427   (Delta - delta = -1.46e-6, essentially tight)
gravity_value     = -9.00e-18
max_abs_moment_kkt_resid = 3.10e-15
mean_m_resid      = 8.88e-16
inner_status      = 0
cold_dense_fast_recheck: diff_vs_evaluate_fullA = 0.0   (exact agreement with the hard oracle)
```

This is a **-14.6% relative** improvement over the documented canonical lower incumbent
(κ=0.005428799948779983, `lower_lfixcomposite_fast_sr1_300s`). It is independently
cold-hard-rechecked feasible with excellent residuals — not a numerical artifact of compressed
mode (both sr1 configs, dense and compressed, land almost exactly on the OLD canonical value
instead, ruling out "compressed mode is just noisy"; this is a genuinely different, better local
optimum that only the LBFGS Hessian + compressed-mode combination found). **Flagged, not
adopted**: this report's external stationarity check (§5) is inconclusive at this point (as it
also is at the OLD canonical point — see §5's own caveat), so before this replaces the standing
canonical lower incumbent in `candidate_registry.jl`/memory, it should get the same multi-start
/ independent-verification treatment previous incumbent changes received (e.g.
`fullA-continuation6-tie-bug-fix`, `full-d2-winner-boundary-fix`). Reporting it here because it
fell directly out of this comparison and burying it would be inconsistent with this
investigation's own "never silently discard a stronger candidate" discipline.

## 5. Stationarity, exact value/gradient calls, inner solves, fallback counts

Per-config summary at the plateau checkpoint (300s; earlier checkpoints for configs that
converge before then give identical κ, see §2), both directions. `n_eval` = exact `Delta(w)`
value calls; `gradient_calls`/`n_cheap` = outer-loop gradient callback invocations (cheap =
lfix-family or smoothed_ad's own method, never `delta_fd`'s expensive central-FD refresh);
`inner_solves` = `CS.INNER_SOLVE_COUNT[]` consumed inside gradient calls only.
`compressed_fallback_count` (`COMPRESSED_FALLBACK_COUNT[]`) is **0 in all 50 runs** — no exact
price ties encountered anywhere in this sweep.

| direction | config | best κ (300s) | outer iters | n_eval | gradient_calls | inner_solves | fallback | knitro_status |
|---|---|---|---|---|---|---|---|---|
| upper | lfixcomposite_sr1_dense | 0.172457 | 113 | 474 | 114 | 114 | 0 | -103 (converged) |
| upper | lfixcomposite_sr1_compressed | 0.172495 | 150 | 543 | 151 | 151 | 0 | -103 (converged) |
| upper | lfixcomposite_lbfgs_compressed | 0.172345 | 69 | 344 | 70 | 70 | 0 | -103 (converged) |
| upper | deltafd_lbfgs_dense | 0.171774 | 54 | 1953 | 53 | 1696 | 0 | -102 (stall) |
| upper | smoothed_ad_sr1 | 0.167233 | 2474 | 9708 | 2475 | 2475 | n/a | -401 (time limit, every checkpoint) |
| lower | lfixcomposite_sr1_dense | 0.005429 | 51 | 374 | 50 | 50 | 0 | -102 (stall) |
| lower | lfixcomposite_sr1_compressed | 0.005429 | 50 | 359 | 49 | 49 | 0 | -102 (stall) |
| lower | lfixcomposite_lbfgs_compressed | 0.004634 | 25 | 158 | 26 | 26 | 0 | -101 (converged) |
| lower | deltafd_lbfgs_dense | 0.005589 | 41 | 1750 | 41 | 1351 | 0 | -101 (converged) |
| lower | smoothed_ad_sr1 | 0.005067 | 2275 | 11862 | 2276 | 2276 | n/a | -401 (time limit, every checkpoint) |

**Cold dense recheck** (`evaluate_fullA_fast(...; moment_representation=:dense, warm=false)`,
the mandated independent second check for every non-smoothed config): agrees with the
`evaluate_fullA` cold recheck to **exactly 0.0** absolute difference in every single one of the
40 non-smoothed runs (`diff_vs_evaluate_fullA` field in every `summary.txt`) — dense and
compressed searches both land on candidates the dense oracle reproduces bit-for-bit. Not
applicable for `smoothed_ad` (that path never loads `oracle_fast.jl`; its own `evaluate_fullA`
cold recheck is the authoritative check, reported directly, marked N/A rather than silently
skipped — see driver comment).

**Stationarity** (`external_stationarity_check_inline`, this driver's own copy of
`stationarity_check.jl`'s exact reduced-KKT methodology — copied rather than `include`d to avoid
this file's own documented include-order hazard, not re-derived): clean at the **upper**-direction
best points (`eta_nonneg=true`, `residual_relative`≈0.002, `n_active_bounds=0`,
`n_nonfinite_probes=0`). At **every** lower-direction best point checked (old canonical AND the
new §4 candidate alike), the central-FD probe for `grad_Delta` hits a non-finite value at exactly
one of the 16 coordinates (`n_nonfinite_probes=1`), which zeros that gradient component and
produces an uninformative `eta≈0`/`residual_relative=1.0` — this is a **limitation of the h=0.01
probe grid at these particular points** (same failure mode at both the old and new candidate,
ruling out this being evidence specific to the new one), not a red flag unique to §4's candidate.
Primary feasibility evidence (tiny `Delta_minus_delta`, `max_abs_moment_kkt_resid`~1e-15,
`mean_m_resid`~1e-15, `inner_status=0`, exact cold-recheck agreement) is what actually
establishes both points as genuine, well-converged feasible candidates; the stationarity check's
inconclusive lower-direction result is reported as a real limitation of this report's own
methodology, not smoothed over.

## 6. `smoothed_ad_sr1`: consistent, but underperforms and never converges within budget

At **every** checkpoint in both directions, `smoothed_ad_sr1` terminates with KNITRO status
`-401` (iteration/time limit) — it never reaches a genuine convergence status the way the
`lfix_composite` family and (at longer budgets) `deltafd_lbfgs_dense` do. κ plateaus early
(upper: 0.167035 from 15s through 60s, ticking up to only 0.167233 by 120s/300s despite outer
iterations growing from 85 to 2474; lower: flat at 0.005067 from 30s through 300s despite outer
iterations growing from 142 to 2275) — it is spending enormous additional effort (thousands of
extra inner solves) for essentially zero additional progress, and never catches the
`lfix_composite` family in either direction. This is consistent with, and a direct consequence
of, this report's own documented simplification (§1): the original smoothed-consistent
experiment needed a full 5-stage rho homotopy to get good results and found the SAME single
low-rho starting point that this frontier's rho-picker also lands on
(`docs/fullA_smoothed_consistent_experiment.md` §4's own coarse-rho infeasibility-wall finding);
running only the finest stage as a single fixed-rho solve, as this frontier does for wall-clock
comparability, forgoes the homotopy's own demonstrated basin-finding benefit. Reported honestly
as a real limitation of the single-rho simplification, not hidden — a genuine homotopy-based
smoothed AD entry remains a candidate for a future frontier iteration if wanted.

## 7. What's committed / reproduction

- `full_aod_diag/d4_exact/run_d4_optimized_fd_c8.jl` — extended per-config driver.
- `full_aod_diag/d4_exact/c8_frontier_run.jl` — orchestrator.
- `results/fullA_d4/57b2e14/c8_frontier_upper_20260718_163819.csv`,
  `c8_frontier_lower_20260718_163859.csv` — machine-readable summary tables (one row per
  (budget,config)).
- `results/fullA_d4/57b2e14/optfdc8_{upper,lower}_*` — 50 individual run directories
  (`summary.txt`, `callback_trace.csv`, `grad_log.csv`, `knitro.log` each), following the same
  convention as the historical `optfd_*`/`9e03706` run directories this report's own predecessor
  committed.
- `results/c8_frontier_std{out,err}_*.log` — per-subprocess console logs.

Reproduce (single config, e.g. the new lower-direction candidate from §4):
```
source .knitro_env.sh
D4X_GRADIENT_METHOD=lfix_composite D4X_HESSOPT=lbfgs D4X_MOMENT_REP=compressed D4X_MAXTIME_REAL=30 \
  JULIA_NUM_THREADS=20 julia --project=. full_aod_diag/d4_exact/run_d4_optimized_fd_c8.jl lower
```
Full grid (both directions, ~45 min each, safe to run concurrently):
```
julia --project=. full_aod_diag/d4_exact/c8_frontier_run.jl 15,30,60,120,300 upper &
julia --project=. full_aod_diag/d4_exact/c8_frontier_run.jl 15,30,60,120,300 lower &
```

## 8. Scope note: nothing was trimmed

The standing brief anticipated needing to trim the lower-direction sweep for time. That trimming
was **not needed**: both directions ran the full 5-checkpoint grid, concurrently (§1), at no
extra wall-clock cost versus running one direction alone. All "also report" items (time to first
feasible, outer iterations, exact value calls, gradient calls, optimized inner solves,
compressed/dense fallback counts, final cold dense recheck, stationarity diagnostics) are
reported above for every config in both directions, not a subset.
