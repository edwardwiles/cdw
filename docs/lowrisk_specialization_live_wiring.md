# Continuation 8, workstream 4: low-risk specialization live wiring

Branch `c8-lowrisk-wiring`, off `diag/fullA-d4-exact` @ `33c93ff`. Worktree
`gravity-fullA-d4-c8-lowrisk-wiring`, `demand.mit.edu`. All work additive in
new files (`live_defaults.jl`, `bench_gamma_profile_v2.jl`) plus one small
opt-in-flag addition to `gamma_profile.jl` (owned by this workstream). Did
NOT touch `oracle*.jl`, `context.jl`, `compressed_*.jl`,
`composite_gradient*.jl`, `lfix_incremental.jl`, `winner_certificate.jl`
(owned by parallel workstreams 2/3).

## 1. `enable_pow_cache!` — reconfirmed live-safe

Re-ran `verify_pow_cache_wiring.jl` (pre-existing, from
`docs/fullA_pow_cache_wiring.md`) unmodified against current HEAD (`33c93ff`):

- All 8 points (calibration, `upper_lfixcomposite_sr1_60s`, `lower_stalled`,
  5 random-feasible perturbations), both `evaluate_fullA` and
  `evaluate_fullA_fast` paths: **worst|Δ| = 0.000e+00 (bit-identical)**.
- Cache behavior: `n_recompute=1` then pure reuse (`n_reuse=13` on
  `evaluate_fullA`, `n_reuse=7` on `evaluate_fullA_fast`) — PASS.
- `ALL POW-CACHE WIRING EQUIVALENCE CHECKS PASSED`.

Still safe and unchanged since the original wiring report. Confirms it is
safe to rely on as a script-level default.

### Wiring: `live_defaults.jl` (new file)

Rather than touching `context.jl`/`d4_exact_setup()` (workstream 2's), added
`full_aod_diag/d4_exact/live_defaults.jl`: includes the three already-existing
opt-in helper files (`moments_fast.jl`, `autarky_cf.jl`, `autarky_cf_v2.jl`,
none modified) and adds one convenience wrapper:

```julia
enable_live_defaults!(ctx; pow_cache = true, autarky_cf_v2 = false)
```

`gamma_profile.jl` now calls this right after `ctx = d4_exact_setup(...)`,
gated by two env vars read once at load time (`GP_ENABLE_POW_CACHE` default
`"1"`, `GP_ENABLE_AUTARKY_CF_V2` default `"0"`). Both settings verified to
load and run cleanly (`GP_RUN_GRID=0` smoke test, and both flag combinations
under `GP_RUN_GRID=1`). `d4_exact_setup()` itself is unchanged; no other
script is affected unless it explicitly includes `live_defaults.jl`.

## 2. `enable_autarky_cf_v2!` in `gamma_profile.jl` — measured, not assumed

### Why the isolated 35.7x doesn't transfer here (established before measuring)

`gamma_profile.jl`'s per-eval cost is a **full `evaluate_fullA` call**: all
`D^2` moments, a full inner KNITRO dual solve, with **all 15 free A_od
entries** (not just A_dd) moving under a KNITRO local optimization at each
fixed g (`build_pivot_elimination`'s pivot is chosen by max |gravity
coefficient|, not pinned to the diagonal — z_free is the whole non-pivot
A-block). This is NOT the "CF-only sweep from raw Uσ" scenario the v2
report's 35.7x figure was measured on (`docs/autarky_cf_v2_cached_base.md`
benchmark A′) — in `gamma_profile.jl`, UσPow is materialized every call
regardless (needed for the factual O(W·D²) block), so v2's real saving
(skip a from-raw-Uσ σ-power) never applies. This predicts the "flat full
build" result (benchmark B in the v2 report), not the isolated CF-column
result — confirmed empirically below, twice, by two independent methods.

### Measurement 1: real `gamma_profile.jl` sweep, both configs

5 g-values (`GP_N_COARSE=5`), `GP_MAXTIME_PER_POINT=8.0`,
`GP_ENABLE_POW_CACHE=1` in both runs (only `GP_ENABLE_AUTARKY_CF_V2` varied),
`JULIA_NUM_THREADS=20`, otherwise identical settings/starting points:

| run | total wall | g=0.928424 (n_eval, wall) | g=0.964212 (n_eval, wall) |
|---|---|---|---|
| v1 only (pow_cache) | 77.5s | 94, 3.7s | 88, 3.6s (knitro_status=-101, converged) |
| v1+v2 (pow_cache+autarky_cf_v2) | 81.6s | 94, 3.8s | 232, 8.2s (knitro_status=-401, **time-limited**) |

At `g=0.928424` both runs land on **exactly the same n_eval=94** — the
trajectories are bit-identical up to that point (as expected: v2 is
documented bit-identical on `Delta_dual` downstream), giving one clean
apples-to-apples data point: **3.7s vs 3.8s, i.e. flat-to-slightly-slower**.

At `g=0.964212` the two runs **diverge**: run A converges naturally at
n_eval=88, run B is still iterating and hits the 8s wall-clock cutoff at
n_eval=232. This is a real, documented KNITRO artifact, not a bug: tiny
per-eval timing noise between configs changes how many SR1 quasi-Newton
iterations complete before `maxtime_real` fires, which can flip which basin
the optimizer is exploring in a genuinely-multi-basin objective (the
non-monotonicity this investigation already documented,
`docs/fullA_continuation7_handoff.md`). Total wall-clock across a
maxtime-limited multi-point sweep is therefore **not a clean metric** for
this comparison on its own — hence Measurement 2.

### Measurement 2: controlled trajectory replay (no time-cutoff confound)

New `bench_gamma_profile_v2.jl`: captures ONE real `gamma_profile.jl`-style
KNITRO local-optimization trajectory (same `profile_delta_at_gamma` logic,
generous 20s budget so it runs to natural completion, not a time cutoff) at
`g=0.9284`, giving a genuine 94-point sequence of `x_free` vectors (all 15
A_od entries + γ' moving, exactly `gamma_profile.jl`'s real access pattern).
Replays that **identical** sequence via direct `evaluate_fullA(...; warm=true)`
calls against two independently-built, freshly-warmed ctxs — same inputs,
only the moment-build implementation differs — with `@timed` evidence
(wall, allocation, GC time), not wall-clock alone:

| config | n | total | median/call | mean/call | alloc/call |
|---|---|---|---|---|---|
| pow_cache only (v1) | 94 | 2.579s | 0.0234s | 0.0274s | 15163.9 KB |
| pow_cache + autarky_cf_v2 | 94 | 2.692s | 0.0247s | 0.0286s | 14951.6 KB |

**Result: total-time ratio v1/v2 = 0.958 (v2 is ~4.4% SLOWER in total time,
~5.2% slower in median per-call time), despite saving 212.3 KB/call (~1.4%)
in allocation.** The allocation saving is real (matches the report's ~0.6%
CF-column-share expectation) but the call is inner-KNITRO-dual-solve-bound
(~15 MB/call, dominated by the factual block and the dual solve, not the CF
column), so the small allocation win doesn't show up as a wall-clock win —
if anything it's swamped by run-to-run solve-path noise in the other
direction.

**Both independent measurements agree: no positive effect, and mild evidence
of a small negative one.** This directly confirms — and extends to
`gamma_profile.jl`'s actual repeated-sweep access pattern — the v2 report's
own "flat in full build" finding; the 35.7x only exists for a CF-only build
from raw Uσ that `gamma_profile.jl` never performs.

## 3. `delta_star_schedule.jl` disposition

Searched the whole filesystem (`find /bbkinghome/edav/gravity_robustness
-iname "delta_star_schedule.jl"`), not just this worktree. Result: exists in
exactly two places, both **outside this worktree/investigation**:
`trade_robustness_cm_gradient_fix/sequential_gravity/delta_star_schedule.jl`
and `trade_robustness_modular_perf/sequential_gravity/delta_star_schedule.jl`
— a different repo/investigation (`sequential_gravity`) on different
branches, per the standing brief explicitly out of scope. It does **not**
exist inside `full_aod_diag/d4_exact/` on this branch. No action taken there.

Checked the other named/likely candidates inside `full_aod_diag/d4_exact/`
for the same CF-only-fixed-draws access pattern:
- `gamma_profile_multistart.jl` — `include`s `gamma_profile.jl` directly and
  reuses its `profile_delta_at_gamma` verbatim (same env-var wiring already
  covers it, no separate edit needed; same full-build access pattern, so the
  same negative/flat verdict applies — not run separately, per the brief's
  explicit "don't run the full multistart grid, that's other work").
- `check_multistart_isolate.jl`, `check_multistart_feasibility.jl`,
  `check_multistart_warmstart_rescue.jl` — all call `evaluate_fullA` in the
  same full-moment-build pattern (grep-confirmed). Not a CF-only fit.
- `h_sweep.jl` — does not call `evaluate_fullA`/`moments!` at all (uses
  `frozen_adjoint_Q`/`fixed_dual_L`/`optimized_Delta`/`compute_winners`, a
  different evaluation path). Not applicable.

**No script in `full_aod_diag/d4_exact/` has the narrow "CF-only, fixed
draws, only A_dd/γ' move" access pattern `enable_autarky_cf_v2!` targets.**
That pattern only exists where `delta_star_schedule.jl` lives, which is out
of scope for this worktree.

## 4. Recommendation

- **`enable_pow_cache!`: keep wired as the script-level default** (now live
  in `gamma_profile.jl` via `live_defaults.jl`, `GP_ENABLE_POW_CACHE=1`
  default). Unconditionally bit-identical, real (if modest) allocation/GC
  win, no measured or plausible downside.
- **`enable_autarky_cf_v2!`: do NOT wire as a default in `gamma_profile.jl`.**
  Two independent measurements (a real KNITRO sweep's matched-trajectory
  point, and a controlled no-confound trajectory replay) both show no gain
  and mild evidence of a ~4-5% per-call slowdown, consistent with the
  allocation saving being swamped by the inner-dual-solve-dominated call
  cost. The flag is wired and available (`GP_ENABLE_AUTARKY_CF_V2=1` /
  `enable_live_defaults!(ctx; autarky_cf_v2=true)`) for the day a genuine
  CF-only-from-raw-Uσ sweep script exists in this worktree, but should stay
  off by default here. This is a clean negative result, not a wiring
  failure — the specialization is correct and well-scoped, it is simply
  solving a cost that `gamma_profile.jl` doesn't have.

## Files

- `full_aod_diag/d4_exact/live_defaults.jl` (new) — `enable_live_defaults!`.
- `full_aod_diag/d4_exact/gamma_profile.jl` (modified) — opt-in env-var
  wiring, defaults `pow_cache=true, autarky_cf_v2=false`.
- `full_aod_diag/d4_exact/bench_gamma_profile_v2.jl` (new) — controlled
  trajectory-replay benchmark (Measurement 2 above), re-runnable.
