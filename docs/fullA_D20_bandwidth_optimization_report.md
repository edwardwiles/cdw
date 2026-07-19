# Full-A D=20 bandwidth-selection optimization (Continuation 9, Phase 5)

Same rigor/style as `docs/fullA_D20_W80k_microbenchmark.md`, whose §3D finding
this task targets directly: in the full 400-coordinate `L_fix` gradient
(`composite_gradient_at_fast`) at D=20/W=80,000, the `fd_and_bandwidth`
component (399 coords × bandwidth-select + 2 FD probes) cost **15.7s of a
~20.5s reconstructed gradient total** — the single largest piece, ahead of
`cache_build` (3.5s) and `base_state solve` (1.3s). Measured on
`demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, commit **`2573f3e`** (branch `c9-bandwidth`, forked from
`diag/fullA-d4-exact`). Context via `context_real_d20.jl`'s `d20_real_setup`
(France focal, natural-theta calibration point — the W80k doc's own Point 1).
Harness: `full_aod_diag/d4_exact/c9_bandwidth_optimization_benchmark.jl`
(main D=20 run) + `test_bandwidth_quantile.jl` (D=4 correctness suite). Raw
logs + CSVs: `results/fullA_d4/2573f3e/c9_bandwidth_optimization/`.

## 0. What was built

Four new/modified files, all additive (existing call sites unchanged unless
they opt into a new kwarg):

- **`bandwidth_quantile.jl`** (new): `select_bandwidth_quantile`, a
  closed-form replacement for `select_bandwidth`'s geometric-bisection search
  (method 3 of the task brief).
- **`bandwidth_cache_policy.jl`** (new): `BandwidthCachePolicy`, a
  staleness-detection wrapper around the pre-existing `h_mode=:cached`
  mechanism (methods 2 and 4 — same mechanism, different reuse-window
  tuning).
- **`composite_gradient_fast.jl`** (modified, additive): new
  `h_mode=:quantile` branch and new `validate_frac` kwarg (method 5,
  subsampled h/2 diagnostic). Every existing `h_mode`/kwarg is byte-identical
  in behavior to before this session (verified in `test_bandwidth_quantile.jl`
  Test 3).
- **`test_bandwidth_quantile.jl`** (new): D=4 correctness suite, run before
  any D=20 compute.

## 1. The closed-form derivation (method 3)

`select_bandwidth`'s bisection searches for an `h` that produces a target
switching-mass fraction, evaluating `count_winner_flips` (an O(W) pass) up to
7 times per coordinate. The derivation in `bandwidth_quantile.jl`'s header
shows this search is unnecessary: `price_and_pTsigma_cell` combined with
`aod_level_cell` shows `price[ω](h) = price0[ω] * exp(c·h)` **exactly** (no
small-h approximation) for a single changed origin at a destination, where
`c = -μ·slope_cell` is a single scalar shared by every draw (the slope comes
from `pivot_expand`'s affine map, read off directly: the "direct" cell always
has slope 1, the "pivot" cell has slope `-c[dir_lin]/c[piv_lin]`). Combined
with `update_winner_o1`'s own case analysis (a flip needs exactly one strict
inequality against a cached threshold price), every draw's exact flip
threshold `h_flip[ω]` solves in closed form — no iteration, one O(W) pass per
affected cell, then the target mass is read off as an order statistic of the
sorted `h_flip` array.

**Scope, honestly bounded**: this closed form only covers the case where the
coordinate's two affected cells (`dir_lin`, `piv_lin`) sit at *different*
destinations. When they collide (both at the pivot's own destination column —
exactly `D-1` of `D²-1` coordinates, 19/399 at D=20, confirmed in the D=20 run
below), two origins move simultaneously and the crossing-condition case
analysis is materially more complex (not derived here). `select_bandwidth_quantile`
detects this explicitly and falls back to the proven bisection for exactly
those coordinates — correctness preserved everywhere, the fast path only
claimed where actually derived.

**A real bug found and fixed before trusting any number below**: the first
version of this file targeted the *midpoint* of `target_mass_frac`'s band
`(lo,hi)=(0.003,0.03)`. `select_bandwidth`'s bisection, however, only needs to
*clear* `lo_frac` and stops as soon as it does (geometric doubling/halving
searching for *any* in-band point, not the center) — empirically it lands
close to `lo_frac`, not the midpoint. Targeting the midpoint produced a
gradient with A-block cosine similarity to the bisection baseline of only
**~0.58** at D=4 (see §3). Retargeting `lo_frac` directly fixed this — this is
reported as a real mid-session bug, not smoothed over.

## 2. D=4 correctness suite (`test_bandwidth_quantile.jl`, before any D=20 run)

Three claims checked at D=4 (`d4_exact_setup(find_smallest=true)`):

1. **Closed-form flip-threshold formula exactly predicts `count_winner_flips`**
   at several interior-h test points (avoiding the known strict-vs-non-strict
   boundary-tie ambiguity right at `h==h_flip`, which is a harmless, expected
   convention artifact, not a formula error — documented and worked around in
   the test). **PASS**, exact agreement at every non-collision coordinate.
2. **A-block gradient agreement, bisection vs. quantile**: cosine similarity
   **0.826** at D=4/W=8000 (after the lo_frac fix above; the pre-fix midpoint
   version scored 0.577). This is noisier than hoped, attributed to the FD
   estimate's own known h-sensitivity near winner-boundary kinks (a
   *pre-existing* property of this method, not introduced here — a naive
   `h=1e-5` "smooth" FD estimate at the same coordinates disagreed with
   *both* bisection and quantile by up to 100x in magnitude and sign,
   confirming the FD gradient genuinely depends on sampling enough kink mass,
   and W=8000/target-mass-fraction of 0.3-3% means only 24-240 draws worth of
   signal — a small, noisy sample). Flagged as an open question for the D=20
   run to resolve (§3).
3. **`validate_frac` subsampling never changes the returned gradient**, only
   which coordinates get the extra h/2 diagnostic probe. **PASS exactly**:
   `max|g diff|` across `validate_frac ∈ {1.0, 0.5, 0.0}` = 0.0 (bit-identical),
   validated-coordinate counts matched the requested fraction exactly
   (15/15, 7/15, 0/15 at D=4's 15 A-block coordinates).

## 3. D=20/W=80,000 results

### 3A. Memory safety (mandatory check before anything else)

`d20_real_setup(W=80000)`: 57.5s wall (cold JIT), **VmHWM 2.66 GB** —
consistent with the W80k doc's post-fix 2.60 GB, confirming
`needs_outer_moment_jacobian=false` is still the active default in this
worktree (`grep` confirmed before starting). Process VmHWM at the end of the
full benchmark (setup + all 5 parts below, ~5 minutes of KNITRO/gradient
compute): **5.87 GB** — safe, no memory alert triggered.

### 3B. Full 400-coordinate gradient wall-clock, threaded=true, N=4/config

| h_mode | median | speedup vs adaptive |
|---|---|---|
| **adaptive** (baseline, bisection) | 5.502s | 1.00x |
| **quantile** (NEW, closed-form) | 5.112s | **1.08x** |
| **fixed** (h0=0.01, pre-existing, no bandwidth search, no h/2 diagnostic) | 3.024s | **1.82x** |
| **cached, COLD** (empty Dict, every coord a miss) | 4.183s | 1.32x |
| **cached, WARM** (Dict pre-populated at the same point) | 2.958s | **1.86x** |

**Headline, reported plainly**: the closed-form quantile selector (method 3)
gives only a **modest 1.08x** speedup on the full gradient — far short of the
W80k doc's implied upside from "eliminate up to 7 bisection probes per
coordinate." The reason, diagnosed from this same data: `select_bandwidth`'s
bisection apparently converges in far fewer than 7 iterations in practice at
this real calibration point (consistent with the D=4 dev-debug session
observing `n_iter=2` on a representative coordinate) — so the *iteration
count* was never the dominant cost the W80k doc's phrasing suggested; the
per-probe O(W) work itself (shared by both bisection and the closed form) is
what's expensive, and the closed form does not reduce that.

The **real wins come from removing work outright, not searching for it
faster**: `h_mode=:fixed` (1.82x) and `h_mode=:cached` warm (1.86x) both skip
the h-vs-h/2 slope-stability diagnostic entirely (2 of the 4 per-coordinate
`a_block_fd_component` calls) *and*, for `:cached` warm, skip bandwidth
search too. The gap between `adaptive`/`quantile` (~5.1-5.5s) and
`fixed`/`cached_warm` (~3.0s) is dominated by that diagnostic, not by
bandwidth search — a finding this task did not anticipate going in, and one
the W80k doc's own component breakdown (which bundled bandwidth-select and
the 2 FD probes into one `fd_and_bandwidth` number) could not have separated.

### 3C. Winner-switch counts / weighted switching mass

| h_mode | mean mass | median mass | min | max |
|---|---|---|---|---|
| adaptive | 0.000591 | 0.000169 | 0.0 | 0.00576 |
| quantile | 0.000532 | 0.000169 | 0.0 | 0.00523 |

**A genuine D-scaling finding, not previously flagged**: both methods land
*well below* the configured target band `[0.003, 0.03]` — median mass
0.00017 is **18x below** the band's own floor. This means `h_floor=1e-4` is
binding for a large fraction of D=20's 399 coordinates (the search wants to
go lower than the floor allows to hit the target band, and stops there
instead) — the floor/ceiling and target-mass-band constants in
`select_bandwidth`'s signature were evidently tuned against D=4/D=10
conditions (per that function's own docstring, citing
`docs/fullA_d4_final_report.md`'s h=0.1 failure point) and were never
re-validated at D=20 scale. `select_bandwidth_quantile` fell back to the
proven bisection for 19/399 coordinates (the same-destination collision
case, exactly `D-1=19` as predicted by the derivation in §1).

### 3D. Per-coordinate gradient agreement, adaptive vs. quantile — the D=4 noise hypothesis confirmed

| | D=4/W=8000 (dev test) | D=20/W=80000 (this run) |
|---|---|---|
| cosine(g_adaptive, g_quantile), A-block | 0.826 | **0.9984** |
| sign agreement | noisy on several coords | **99.7%** (398/399) |

This directly confirms the hypothesis flagged in §2: the D=4 disagreement was
a small-W noise artifact of the FD's known h-sensitivity near winner-boundary
kinks, not a defect in the closed-form derivation. At D=20's 10x larger W
(more draws per target mass fraction, even with the floor binding per §3C),
the two selectors' independently-chosen bandwidths produce **near-identical**
gradients (median relative error 0.0, mean 0.007). **The closed-form
selector is validated as numerically trustworthy at the scale this
investigation actually cares about**, even though it does not deliver a large
wall-clock win there (§3B).

### 3E. `BandwidthCachePolicy` staleness-detection demo (10 simulated outer-loop points)

Small steps (`|step|=0.01` in reduced w-space) for 9 of 10 iterates, one
deliberate large jump (`step=0.5`) at iterate 7 to test the move-threshold
trigger (`max_move=0.05`, `max_iters_since_anchor=5`):

| iter | step | invalidated | reason | cache hits | wall |
|---|---|---|---|---|---|
| 1 | 0.01 | **true** | init | 0/399 | 4.01s |
| 2 | 0.01 | false | — | 398/399 | 3.09s |
| 3 | 0.01 | false | — | 399/399 | 3.06s |
| 4 | 0.01 | false | — | 399/399 | 3.08s |
| 5 | 0.01 | false | — | 399/399 | 2.94s |
| 6 | 0.01 | false | — | 399/399 | 2.89s |
| **7** | **0.50** | **true** | **move_threshold** | 0/399 | 4.00s |
| 8 | 0.01 | false | — | 399/399 | 2.98s |
| 9 | 0.01 | false | — | 398/399 | 2.88s |
| 10 | 0.01 | false | — | 399/399 | 2.96s |

**The staleness mechanism works exactly as designed**: the large jump at
iterate 7 is correctly detected and triggers a full cache invalidation
(reason `move_threshold`, matching the requirement "never silently use stale
bandwidths after large parameter moves"); every other iterate reuses the
cache and pays roughly the `:cached_warm`/`:fixed` price (~2.9-3.1s) instead
of `:adaptive`'s ~5.5s. Across this 10-point trajectory: 2 full-cost
iterates + 8 cheap ones averages to **~3.28s/iterate**, a **1.68x** reduction
versus paying `:adaptive`'s 5.50s on every iterate — and this ratio only
improves on a longer real outer-loop run, since the two full-cost iterates
are a one-time-per-invalidation cost, not a per-iterate one.

### 3F. A pre-existing thread-safety bug found (not introduced by this session)

Both `part1`'s `cached_warm` run (`all cache_hits=false` despite a
fully-pre-populated Dict) and `part4`'s demo (iterates 2 and 9 show 398/399
hits despite no invalidation) show occasional single-coordinate cache misses
that should be impossible if the Dict were correctly and fully populated.
Investigated with a targeted D=4 repro (`race_check.jl`, 200 trials of
"populate a fresh `Dict{Int,Float64}` via one `threaded=true`,
`h_mode=:cached` call, check `length(dict) == D²-1`"): **3/200 trials (1.5%)
produced a corrupted (under-populated) Dict.** Root cause: `composite_gradient_at_fast`'s
`h_mode=:cached` branch (`composite_gradient_fast.jl`, pre-existing
Continuation-5 code, not modified by this session beyond adding the
`:quantile`/`validate_frac` levers elsewhere in the same function) calls
`haskey`/`bandwidth_cache[k] = h` from inside `Threads.@threads for k in
2:D2` — a plain `Dict` is **not thread-safe for concurrent writes**, and a
cache-miss "populate" pass has every thread writing to the same Dict
concurrently. This is a genuine, reproducible, PRE-EXISTING bug in code this
task's `BandwidthCachePolicy` builds directly on top of.

**Recommendation** (not implemented in this session, flagged for whoever
adopts `:cached`/`BandwidthCachePolicy` in production): populate a fresh
`bandwidth_cache` with `threaded=false` (a one-time, already-cheap ~4s cost
per §3B's `cached_cold` row), then use `threaded=true` freely for the
subsequent read-only (all-hit) calls — reads without concurrent writes are
safe. Alternatively wrap the Dict writes in a `ReentrantLock` or switch to a
per-thread-Dict-then-merge pattern if populate-time threading is required.
This does not affect the wall-clock numbers reported in §3B/3E (those runs'
occasional 1-of-399 misses cost at most one extra cheap bandwidth-select
call, negligible against the ~3s totals) but IS a latent correctness risk
worth fixing before this mechanism is trusted unattended in a long production
run.

### 3G. Directional secant validation (5 random directions, h=0.02, matching W80k doc §3F exactly)

Same seed (31415), same methodology (2 full warm-started `evaluate_fullA`
re-solves per direction, bypassing the incremental machinery entirely) —
dir 1's secant (6.38e-5) reproduces the W80k doc's own dir-1 value exactly,
confirming this run's fidelity to that earlier benchmark.

| dir | secant | pred: adaptive | pred: quantile | pred: fixed | pred: cached_warm |
|---|---|---|---|---|---|
| 1 | 6.38e-5 | 9.18e-5 | 1.75e-4 | 2.94e-6 | 9.18e-5 |
| 2 | 7.41e-4 | 6.10e-4 | 5.48e-4 | 3.64e-4 | 6.10e-4 |
| 3 | 2.53e-4 | **-5.51e-5** | **-5.15e-5** | 1.92e-4 | **-5.51e-5** |
| 4 | 4.61e-4 | 8.54e-4 | 8.10e-4 | 4.97e-4 | 8.54e-4 |
| 5 | -5.94e-4 | -5.45e-4 | -5.92e-4 | -4.13e-4 | -5.45e-4 |

| method | mean abs err | median abs err |
|---|---|---|
| adaptive | 0.000182 | 0.000131 |
| quantile | 0.000192 | 0.000192 |
| **fixed** | **0.000143** | **0.000061** |
| cached_warm | 0.000182 | 0.000131 |

5/5 secants finite/sane for every method. **Direction 3 shows the SAME
A-block sign disagreement this investigation already flagged as open**
(memory: full-A continuation 8's Section 9 finding, "73-93% coordinate-sign
agreement in a near-flat region") — `adaptive`/`quantile`/`cached_warm` all
predict a *negative* secant where the true re-solved secant is positive;
`fixed` happens to get the sign right there. This is consistent, not a new
bug: `cached_warm` matches `adaptive` exactly on every direction (expected —
at this unmoved base point its cache was populated by an `adaptive` call, so
it returns the identical gradient), and `quantile` tracks `adaptive` closely
throughout (small residual differences, consistent with §3D's 0.9984
cosine). **`fixed` scores the lowest mean/median error in this small n=5
sample** — worth flagging honestly, but n=5 is far too small to conclude
`:fixed` is actually more accurate; it is more plausibly explained by
`:fixed`'s single global h=0.01 landing, by chance, in a locally more stable
region for these 5 particular directions. No method's predictions are
reliable enough in an absolute sense to treat any single directional
secant as validating the gradient beyond "same order of magnitude, mixed
sign accuracy" — consistent with this investigation's existing, unresolved
A-block sign-disagreement finding, not a new problem introduced here.

## 4. Recommendation

**Adopt `h_mode=:cached` wrapped in `BandwidthCachePolicy`, not
`h_mode=:quantile`, as the production lever for outer-loop gradient calls**,
with the thread-safety fix from §3F applied first (populate with
`threaded=false`). Rationale:

- The closed-form quantile selector (method 3, this session's main
  engineering deliverable) is **numerically validated and correct** at the
  D=20/W=80,000 scale this investigation cares about (§3D: 0.9984 cosine
  agreement with the trusted bisection baseline) — but delivers only a
  **modest 1.08x** wall-clock win, because bisection's real-world iteration
  count was already low at this calibration point. This is a genuine,
  reported negative result on the speed dimension specifically, not a
  correctness problem: the derivation and implementation are sound and could
  still be useful in a **regime where bisection needs more iterations**
  (e.g. further from calibration, or at a different target-mass band) —
  flagged as a follow-up, not claimed here.
- `h_mode=:cached` + `BandwidthCachePolicy`'s staleness detection delivers a
  **larger, more reliable win** (1.86x on a fully-warm call, ~1.68x averaged
  across a realistic 10-point trajectory with one deliberate large jump) by
  eliminating repeated bandwidth search entirely between nearby outer
  iterates, while still providing genuine per-coordinate adaptivity (unlike
  `:fixed`, which never adapts at all) and a concrete, working staleness
  guard (§3E) that satisfies the standing brief's explicit "never silently
  reuse stale bandwidths after large moves" requirement.
- `h_mode=:fixed` is the cheapest and, in this run's small secant sample,
  surprisingly not the least accurate — but it provides **no per-coordinate
  adaptivity at all**, reintroducing exactly the risk this investigation's
  own prior work flagged as a gap (`docs/fullA_d4_final_report.md` sec 4 item
  4: "h-sensitivity matters a lot near the upper candidate"). Not
  recommended as the sole production policy, though it remains a reasonable
  fast fallback when the outer solve is far from any KKT/tie boundary.
- Method 5 (`validate_frac` subsampling of the h/2 diagnostic) is
  implemented and exactly verified to leave the returned gradient unchanged
  (§2, Test 3) but was **not separately wall-clock-benchmarked at D=20** in
  this session (time budget) — its likely benefit is bounded by the gap
  between `adaptive` (5.50s, full diagnostic) and `cached_cold`/`:fixed`-like
  costs (~3.0-4.2s, no diagnostic), i.e. up to ~1.3-1.8x on top of whichever
  base `h_mode` it's layered onto. Flagged as a reasonable, low-risk
  complementary lever, not independently validated here.

## 5. What this report does NOT cover (explicitly out of scope, not overlooked)

- **Full closed-form derivation for the same-destination collision case**
  (§1, 19/399 coordinates at D=20) — falls back to bisection, correctness
  preserved, speed not improved for those coordinates specifically.
- **Re-tuning `h_floor`/`h_ceil`/`target_mass_frac` for D=20** (§3C's binding-floor
  finding) — flagged as a real, unexplored follow-up; this report only
  diagnoses the symptom (mass far below the target band), not a fix.
- **`validate_frac` wall-clock benchmarking at D=20** (§4) — implemented and
  correctness-verified at D=4 only.
- **Points closer to genuine infeasibility, or further from calibration** —
  all measurements here are at a single feasible point (Point 1 of the W80k
  doc); the closed-form method's relative advantage over bisection may differ
  where bisection needs more than 2-3 iterations to converge (not observed at
  this point).

## 6. Files

New this session, all under `full_aod_diag/d4_exact/`:
`bandwidth_quantile.jl`, `bandwidth_cache_policy.jl`,
`test_bandwidth_quantile.jl`, `c9_bandwidth_optimization_benchmark.jl`.
Modified (additive only): `composite_gradient_fast.jl` (`h_mode=:quantile`,
`validate_frac` kwarg). Raw logs + CSVs:
`results/fullA_d4/2573f3e/c9_bandwidth_optimization/`.
