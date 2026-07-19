# Full-A D=20 real-data (W=80,000): production bottleneck + memory microbenchmark

Continuation 9, Phase 2 (W=80,000 half of the standing brief; a companion W=800,000
memory-safety check was run separately by the coordinating session and is referenced
here only where it bears on the headline finding). Measured on `demand.mit.edu`,
`JULIA_NUM_THREADS=20` (default run) / swept 1/5/10/20, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` throughout, commit **`bd313a2`**. Harness:
`full_aod_diag/d4_exact/c9_w80k_microbenchmark.jl` (main 4-point profile) +
`c9_w80k_threadsweep.jl` (thread sweep) + `c9_w80k_memcheck.jl` (post-fix sanity
probe), all new this session, reusing `c8_perfprofile_harness.jl`'s `@prof`-timer /
warmed-process discipline directly (same `instrumentation.jl`, same component/scope
names) rather than reinventing it. Real-data context via `context_real_d20.jl`'s
`d20_real_setup` (ported+built in this session's Phase 1, France focal,
`baseIndex=2`, σ=2.5, μ estimated via gravity). Raw logs + CSVs + a consolidated
JSON: `results/fullA_d4/bd313a2/c9_w80k_microbenchmark/`.

## 0. Headline finding: a mis-defaulted flag, not an inherent D=20 cost

This benchmark run itself **found and fixed a real production bug** mid-session,
discovered because the first launch of the exact machinery this document profiles
triggered a server-wide memory alert on this shared 3TB machine.

**Root cause**: `d20_real_setup` (`context_real_d20.jl`) was built by mirroring
`context_scaled.jl`'s `d_exact_setup_scaled` signature verbatim, including its
default `needs_outer_moment_jacobian=true`. That default is harmless at the D=4-10
scales `context_scaled.jl` was written for (`jac_h` is a dense
`W x (nTotalMoments+2) x l_full` tensor, tiny at those sizes) but **not** at D=20:

```
jac_h size = W * (nTotalMoments+2) * l_full * 8 bytes
           = 80000 * 404 * 423 * 8 bytes
           = 109,027,584,000 bytes
           ≈ 109 GB
```

Both real production drivers (`run_fullA_D4_production.jl`, `run_fullA_D10_production.jl`)
already set this flag to `false` — their gradient path (ForwardDiff / Method B) never
touches an analytic outer moment Jacobian — but the diagnostic constructor's
inherited default was never overridden for D=20. This is exactly the kind of gap
Phase 3 of the standing brief is meant to audit (mandatory dense W-by-moment
materialization), and it is a substantially bigger offender than the plain
`W x nTotalMoments` matrix the brief anticipated (~257MB) — `jac_h` carries an
*extra* factor of `l_full` (≈423 at D=20), making it **~400x** worse than the
matrix everyone was watching for.

**Measured impact, before vs after the fix (commit `bd313a2`)**:

| | pre-fix (`needs_outer_moment_jacobian=true`) | post-fix (`=false`, now default) |
|---|---|---|
| VmHWM after `d20_real_setup(W=80000)` | **109.03 GB** | **2.60 GB** |
| VmHWM at end of the full 4-point benchmark | (run killed before reaching here) | **4.47 GB** |
| `d20_real_setup` wall, first call (JIT+build) | 102.11s | 53.1s |
| `d20_real_setup` wall, second call (steady-state) | 62.27s | 15.0s |
| W=800,000 (companion probe, coordinating session) | **killed after ~780GB and still climbing, headed past 1TB** | not re-tested (out of this task's scope; W=800k belongs to the separate memory-safety check) |

The fix also **more than doubled setup speed** (62.3s → 15.0s steady-state) — the
109GB tensor was not just sitting idle in memory, it was being *built* every call,
so removing it was a wall-clock win as well as a memory win. Every number in the
rest of this document is measured **post-fix**; the pre-fix run was killed mid-Part-1
and its partial output (a different, pre-fix `results/fullA_d4/5c1baca/...` directory)
is superseded by the complete post-fix run reported below.

**Process safety note**: this session's own first launch was one of the two
processes flagged in the server-wide alert (the other was the coordinating
session's independent W=800,000 probe). Both were killed, server memory was
confirmed to recover (`free -h`: used dropped from the alert level back to
~376GB/2.4TB-free by the end of this session), and every subsequent run in this
document was preceded by an explicit memory sanity check (`c9_w80k_memcheck.jl`:
one cold `evaluate_fullA` call, VmHWM asserted under 5GB) before being allowed to
proceed to the full benchmark or the thread sweep.

## 1. Context/setup (measured once, W=80,000)

| | value |
|---|---|
| D | 20 |
| n_free (`1+D^2`) | 401 |
| nTotalMoments | 402 |
| `d20_real_setup` wall, cold (JIT paid) | 53.1s |
| `d20_real_setup` wall, warm (JIT already paid) | 15.0s |
| `gc_live_bytes` delta during setup | 1791.8 MB |
| VmHWM after setup | 2.60 GB |
| `ctx.U` shape | **80000 × 20** (12.8 MB) |

**A second, smaller measurement-methodology finding**: `ctx.U`'s shape is
`W × D`, not `W × D²` as this document's own memory-ledger code initially assumed
(see §4) — `drawU.jl`'s `sizeU = UoModel == 1 ? D : D * D` branch is taking the
`UoModel==1` path even under this investigation's "γ_d≡1 is now the universal
gauge, `UoModel` should be vestigial" understanding (memory:
`uomodel-gamma-fwl-cleanup`). Not investigated further here (out of scope for a
microbenchmark), but flagged because it means the *design* draws are `D`-wide
(20 columns) even though the full-A_od outer loop's *free parameters* are `D²`-wide
(400 A-block entries) — these are two different `D`-indexed quantities in this
codebase and it is easy to conflate them, which is exactly what this document's
first draft did.

`gamma'_focal` (natural theta, calibration) = **0.9877618976237339**, bounds =
**[0.9307117545219048, 1.0]** — a tight, asymmetric-to-1.0 window; this matters for
§2's Point 3/4 magnitudes below.

## 2. The four outer points

All four were found feasible **on the first try** (no fallback perturbation search
needed) — a stronger result than the task anticipated:

| point | inner_status | Delta_dual | gamma' | note |
|---|---|---|---|---|
| 1. calibration (natural theta) | 0 | 0.0025909 | 0.987762 | primary benchmark point |
| 2. gravity-tangent perturbation | 0 | 0.0025761 | 0.987762 | step=0.02 in pivot-reduced z-space |
| 3. upper branch (gp0·1.01) | 0 | **0.230884** | 0.997640 | first offset tried (0.01) worked |
| 4. lower branch (gp0·0.99) | 0 | **0.129645** | 0.977884 | first offset tried (0.01) worked |

**Point 2 is gravity-EXACT, not gravity-tangent-to-first-order**: the pivot
elimination (`build_pivot_elimination`/`pivot_expand`) solves one A-block entry
per point to satisfy the linear gravity constraint identically, for *any* value of
the other `D²-1` free entries — so a random perturbation of the pivot-reduced
z-space is gravity-feasible exactly, confirmed here by `gravity_raw = -2.3e-15`
(machine zero) at Point 2. This is a stronger guarantee than the task's own
"-ish" allowance anticipated, and it is why this document did not need a separate
"gravity-tangent generator" — the existing D4-exact pivot-elimination machinery
already generalizes to D=20 with no changes.

**Points 3/4 (gamma' offsets)**: a striking finding worth flagging plainly — `Delta_dual`
jumps by **~90x** (0.0026 → 0.231) for just a **1% relative** move in gamma' above
calibration, and by ~50x for a 1% move below. This is a direct, mechanical
consequence of the tight `[0.9307, 1.0]` bound on gamma' at this real-data
calibration point: the calibration value (0.98776) already sits close to the upper
edge of a narrow window, so the model's divergence is extremely sensitive to gamma'
locally — a genuine feature of the D=20 real-data calibration, not a benchmark
artifact (both points solved cleanly, `inner_status=0`).

## 3. Part 1 — full breakdown at Point 1 (calibration)

### 1A. Value callback (`evaluate_fullA_fast`, dense mode — the "trusted reference"
path per `docs/fullA_canonical_performance_profile_c8.md`'s own framing; `evaluate_fullA`
itself, the plain oracle, has no fine `@prof` breakdown, only total/inner/post — see §3B)

**Warm-started (N=6), median ms**:

| component | median (ms) | % of TOTAL |
|---|---|---|
| **TOTAL** | **1862.1** | 100% |
| inner_moment_build | 1211.1 | 65.0% |
| winner_compute | 308.8 | 16.6% |
| moments_reuse | 144.0 | 7.7% |
| inner_knitro_dual_solve | 58.9 | 3.2% |
| inner_dual_fg_callback | 51.7 | 2.8% |
| kkt_residual_compute | 49.7 | 2.7% |
| moment_resid_compute | 49.0 | 2.6% |
| primal_weight_recovery | 27.6 | 1.5% |
| primal_divergence_compute | 0.89 | <0.1% |
| gravity_compute | 0.07 | <0.1% |
| reconstruct_full | 0.003 | <0.1% |

**`inner_moment_build` dominates a warm value call at D=20/W=80000** (65% of
TOTAL) — a qualitatively different bottleneck profile from D=4, where
`inner_moment_build` was a large but not overwhelming share (§1 of the D=4
canonical profile shows ~5.5ms of 11.0ms total, ~50%, at W=8000; here it is 65% of
a much larger absolute total). This is the dense `O(W·D²)`-ish moment matrix
build (`nTotalMoments=402` columns × `W=80000` rows), and it is the single
clearest target for a compressed-mode port at D=20 if this investigation wants to
optimize this path further (compressed mode was never ported to the D=20 context
in this session — see §7).

**Cold (`warm=false`, N=3), median ms**:

| component | median (ms) | % of TOTAL |
|---|---|---|
| **TOTAL** | **4719.5** | 100% |
| inner_knitro_dual_solve | 2895.9 | 61.4% |
| inner_moment_build | 1220.0 | 25.8% |
| inner_dual_hessian_callback (15 calls) | 508.0 | 10.8% |
| winner_compute | 315.3 | 6.7% |
| moments_reuse | 145.2 | 3.1% |
| inner_dual_fg_callback (18 calls) | 55.6 | 1.2% |
| kkt/moment_resid/primal_weight (combined) | 128.4 | 2.7% |

Cold solves take **2.5x longer** than warm (4.72s vs 1.86s) — dominated by
`inner_knitro_dual_solve` needing ~10-18 FG/Hessian callback iterations to
converge from a cold start vs 1 when warm-started, exactly the pattern the D=4
profile documented (there: cold 22.2ms vs warm 11.0ms, 2x, ~10 calls/solve both
scales) — this ratio is consistent across D, a useful cross-scale confirmation.

### 1B. `evaluate_fullA` (plain oracle), warm-started, N=5

| | median |
|---|---|
| total | 3.679s |
| inner (KNITRO solve) | 1.270s |
| post (moment/gravity/KKT bookkeeping) | 2.409s |

Notably, `evaluate_fullA`'s "post" phase (2.41s) is *larger* than its own inner
solve (1.27s) and larger than the entire `evaluate_fullA_fast` warm TOTAL (1.86s)
— `evaluate_fullA` (the plain, non-`_fast` oracle) rebuilds the full dense `K`/`G`
moment matrix a second time via `obj.moments!` after the inner solve (to compute
gravity/KKT/moment-residual diagnostics), essentially paying the
`inner_moment_build` cost twice. This is expected/by-design (the plain oracle
trades speed for a complete, independently-rebuilt diagnostic bundle) but is worth
naming explicitly: **use `evaluate_fullA_fast`, not plain `evaluate_fullA`, for
any performance-sensitive D=20 driver** — the ~2x cost difference at D=4 becomes a
~2x-on-a-much-bigger-number difference at D=20 (3.68s vs 1.86s).

### 1C. Full 400-coordinate `L_fix` gradient (`composite_gradient_at_fast`), N=4/config

| h_mode | threaded | median | reps (s) |
|---|---|---|---|
| adaptive | true | **6.435s** | 6.40, 6.47, 6.58, 6.36 |
| adaptive | false | 35.295s | 35.27, 35.27, 35.32, 35.34 |
| fixed | true | **4.377s** | 4.38, 4.38, 4.35, 4.39 |
| fixed | false | 11.433s | 11.41, 11.54, 11.29, 11.46 |

**Threading gives a 5.5x speedup for adaptive-h** (35.30s → 6.44s) and a **2.6x
speedup for fixed-h** (11.43s → 4.38s) at D=20 — both *larger* than the D=4
canonical profile's 3.4x/2.4x (§3, C1 of the D=4 doc), consistent with threading
mattering more as the coordinate count (`D²-1=399`) grows relative to fixed
per-call overhead. **Threading remains the dominant lever, exactly as at D=4** —
top3-vs-generic accounts for essentially nothing at the full-gradient level (see
§3E), threading accounts for 2.6-5.5x.

Extrapolating from D=4's 36ms (adaptive+threaded) to D=20's 6.44s is a **~180x**
wall-clock increase for a `D²` (400 vs 15, ~27x) times `W` (80000 vs 8000, 10x)
scaling — i.e. roughly consistent with an `O(D²·W)`-ish cost model (27×10=270,
same order of magnitude as the observed 180x), not a superlinear blowup, though
this is a two-point comparison across very different (D,W) pairs and should be
read as a sanity check, not a fitted scaling law.

### 1D. Gradient component decomposition (h_mode=adaptive, top3, N=3)

| component | median |
|---|---|
| base_state solve (shareable with value eval) | 1.273s |
| cache_build (`build_lfix_base_cache`) | 3.512s |
| gamma_analytic (closed-form) | 0.0001s |
| fd_and_bandwidth (399 coords × bandwidth-select + 2 FD probes) | 15.702s |
| **reconstructed total** | **20.487s** |

**Honest gap, reported rather than smoothed over**: this reconstructed total
(20.49s) does not match §3C's directly-measured serial-adaptive full-gradient time
(35.30s) — a ~15s discrepancy. Both use the same underlying primitives
(`build_lfix_base_cache`, `select_bandwidth`, `a_block_fd_component`) called in
the same order, so the gap is not an obvious double-count or omission; it was not
chased further given this task's time budget (matching this investigation's
standing "verify before causal claims" discipline — reported as unresolved rather
than assumed away). A plausible candidate is instrumentation overhead from
`time_ns()` calls nested inside the 399-iteration loop in the decomposition
helper (not present in the plain `composite_gradient_at_fast` call), but this was
not confirmed.

### 1E. Per-coordinate cost, 3 representative coordinates (N=8 reps each)

| coord | top3 (ms) | generic (ms) | ratio |
|---|---|---|---|
| k=2 | 72.32 | 70.45 | 0.974 |
| k=200 | 80.56 | 79.77 | 0.990 |
| k=399 | 78.89 | 81.60 | 1.034 |

**Noise-level at D=20, same as D=4's C1 finding** (there: 0.90-1.09x) — no
consistent top3-vs-generic direction at any of the three coordinates sampled.
Each single-coordinate probe costs ~70-82ms at D=20 (vs D=4's per-coordinate
average of ~2.4ms — driven by the W=8000→80000, 10x, scaling, since these probes
are dominated by O(W) work per the incremental machinery's own design), consistent
with §3C's per-coordinate arithmetic (6.44s / 399 coords ≈ 16ms mean, though the
threaded case divides work across 20 threads — the ~75ms single-probe numbers here
are measured *serially*, i.e. roughly `16ms × 20threads / (~4 probes-per-gradient-
step overhead factor)`-ish; not reconciled exactly, reported as measured).

### 1F. Optimized-value directional secants (NOT the incremental machinery)

5 random unit directions in the pivot-reduced (gravity-exact) z-space, h=0.02,
2 full warm-started `evaluate_fullA` solves per direction (matching the task's
explicit "2 optimized inner solves per direction" instruction — this deliberately
bypasses `composite_gradient_at_fast`'s O(1)/O(D) incremental machinery and
re-solves the inner CC dual problem from scratch at each perturbed point, the
"fully re-solved" `optimized_Delta` method per `three_way_derivatives.jl`):

| dir | secant | wall+ (s) | wall− (s) |
|---|---|---|---|
| 1 | 6.38e-5 | 5.14 | 5.21 |
| 2 | 7.41e-4 | 5.25 | 5.74 |
| 3 | 2.53e-4 | 5.79 | 5.73 |
| 4 | 4.61e-4 | 5.78 | 5.71 |
| 5 | -5.94e-4 | 5.74 | 5.75 |

**5/5 finite and sane** (all `inner_status=0` at both endpoints). Each
warm-started optimized solve costs ~5.1-5.8s — notably *more* than a plain
`evaluate_fullA_fast` warm call (1.86s, §3A) because `optimized_Delta` calls the
plain `evaluate_fullA` oracle (§3B's 3.68s total), not the fast path — consistent
with §3B's finding that the plain oracle should be avoided in performance-
sensitive code. Deliberately did **not** attempt a full 400-coordinate FD
gradient this way (out of scope; at ~5.5s/probe × 2 probes × 399 coords that would
be ~73 minutes, vs. the incremental machinery's 6.4s for the same information).

## 4. Part 2 — lighter checks at points 2-4

| point | value (warm) median | full gradient (top3,threaded,adaptive) median |
|---|---|---|
| 2 (gravity-tangent) | 1.841s | 6.246s |
| 3 (upper branch) | 1.832s | 6.341s |
| 4 (lower branch) | 1.770s | 6.083s |

All three closely match Point 1's numbers (1.86s value / 6.44s gradient) despite
very different `Delta_dual` values (0.0026 to 0.231) — **cost is essentially
independent of how close the point is to the calibration/feasibility boundary**,
at least across this modest range; no evidence here of the "near a KKT/tie
boundary, solves get harder" pattern this investigation has seen elsewhere (memory:
`hardmax-inversion-validation`'s winner-boundary discussion) — worth keeping in
mind as an open question for points *closer* to genuine infeasibility, which this
benchmark's 4 points did not probe (all were found feasible on the first offset
tried).

## 5. Thread-count sweep

Value callback and full 400-coordinate gradient, `JULIA_NUM_THREADS ∈ {1,5,10,20}`,
`OPENBLAS_NUM_THREADS=1` fixed throughout (per Continuation 8's standing finding —
not re-tested here), 4 separate processes (Julia thread count is not
runtime-configurable), each paying its own setup cost:

| threads | setup wall | value median | full gradient median | gradient speedup vs 1 thread |
|---|---|---|---|---|
| 1 | 59.7s | 2.486s | 36.982s | 1.00x |
| 5 | 55.1s | 1.951s | 12.186s | 3.03x |
| 10 | 53.3s | 1.898s | 8.378s | 4.41x |
| 20 | 56.6s | 1.758s | 6.767s | **5.47x** |

**The gradient scales well but sub-linearly with threads** (5.47x at 20 threads,
not 20x) — expected, since `composite_gradient_at_fast` only threads the
per-coordinate FD loop (399 of 400 dimensions; coordinate 1, gamma, is always
analytic/serial) and pays fixed serial overhead (`build_lfix_base_cache`,
`solve_base_state`) on every call regardless of thread count (§3D: ~4.8s of
serial-only cost). A simple Amdahl's-law back-of-envelope (serial fraction
≈4.8s/37.0s≈13%) predicts a max speedup around `1/0.13≈7.7x` as threads→∞, roughly
consistent with the observed diminishing returns from 10→20 threads (4.41x→5.47x,
not the ~2x a linear model would predict). **The value callback barely benefits
from threading at all** (2.486s→1.758s, only 1.41x at 20 threads) — expected,
since `evaluate_fullA_fast`'s dominant cost (`inner_moment_build`, §3A) is a
single dense BLAS operation, not something `composite_gradient_at_fast`'s
coordinate-level `Threads.@threads` loop touches.

**Setup wall time is flat across thread counts** (53-60s, no trend) — sensible,
since `d20_real_setup`'s cost is dominated by the two-way fixed-effects gravity
regression (`master_prestep.jl`, hundreds of `diffPreStep` convergence iterations
visible in the raw log) and moment-object construction, neither of which is
threaded in this codebase.

## 6. Memory ledger

| array | shape | theoretical | observed |
|---|---|---|---|
| **`jac_h` (the bug, §0)** | `80000 × 404 × 423` | **109 GB** | **109.03 GB confirmed** (pre-fix only; absent post-fix) |
| `ctx.U` (Frechet draws) | `80000 × 20` | 12.8 MB | 12.8 MB (confirmed exact) |
| inner K/G moment matrix (`evaluate_fullA`/`winners.jl`) | `80000 × 402` | 257.3 MB | not independently measured (transient, GC'd between calls) |
| `winners.jl` `gap` | `80000 × 20` | 12.8 MB | not independently measured |
| `lfix_incremental.jl` price/runnerup/third/contrib buffers (6×) | `6 × (80000 × 20)` | 76.8 MB | not independently measured |
| `compressed_moments.jl` `wval` | `80000 × 20` | 12.8 MB | not independently measured (compressed mode not exercised at D=20 this session, see §7) |
| gradient vector | `400` | 3.2 KB | negligible |
| **sum of all enumerable steady-state arrays** | | **~372 MB** | |
| **process VmHWM, post-fix, full 4-point benchmark** | | | **4.47 GB** |

**The single most important number in this ledger**: the enumerable steady-state
arrays sum to well under 1 GB, while the process's actual peak (4.47 GB) is
~12x that — the remainder is JIT/compile-time memory, KNITRO's own internal
workspace, GC fragmentation, and transient allocations from the many repeated
value/gradient calls this benchmark itself made (not a leak — `Base.gc_live_bytes()`
at the very end was only 1974.6 MB, well below VmHWM, confirming the peak was a
transient high-water mark, not a steady leak). **This entire steady-state picture
is dwarfed 24-260x by the single `jac_h` mis-default (§0)** — the headline
lesson of this memory ledger is "audit flags with hidden dimension-dependent
tensor sizes," not "the everyday arrays are large."

## 7. What this document does NOT cover (explicitly out of scope, not overlooked)

- **Compressed-mode moments at D=20**: never ported/tested here — `evaluate_fullA_fast`
  was only exercised in `:dense` mode. Given §3A's finding that `inner_moment_build`
  is 65% of a warm value call's cost, and the D=4-10 canonical profile's own finding
  that compressed mode's moment-build advantage *grows* with D (2.35x→3.64x, D=4→10),
  this is the most promising next optimization target for D=20 specifically — flagged,
  not investigated, per this task's explicit scope (dense-only, "trusted reference").
- **W=800,000**: entirely out of scope per the task's own framing (a separate
  coordinating-session memory-safety check); this document only references its
  killed-at-~780GB pre-fix trajectory in §0 because it directly corroborates the
  `jac_h` root-cause diagnosis (same bug, same formula, ~10x worse at 10x the W).
- **A full 400-coordinate finite-difference gradient cross-check**: explicitly
  out of scope per the task brief; §3F's 5-direction secant check is the
  substitute, and all 5 were finite/sane.
- **Points closer to genuine infeasibility**: all 4 points here were found feasible
  on the very first offset/perturbation tried — this benchmark says nothing about
  cost or reliability *near* a real feasibility boundary at D=20 (flagged as an
  open question in §4).

## 8. Files

New this session, all under `full_aod_diag/d4_exact/`: `c9_w80k_timing_probe.jl`
(initial scoping probe), `c9_w80k_memcheck.jl` (post-fix sanity probe),
`c9_w80k_microbenchmark.jl` (main 4-point harness), `c9_w80k_threadsweep.jl`
(thread sweep, 4 separate process launches). Fix: `context_real_d20.jl`
(`needs_outer_moment_jacobian` default `true`→`false`, commit `bd313a2`). Raw
logs + all CSVs + a consolidated `summary.json`:
`results/fullA_d4/bd313a2/c9_w80k_microbenchmark/`. No existing file besides
`context_real_d20.jl` was modified; `c8_perfprofile_harness.jl` itself was not
touched, only its timer pattern reused.
