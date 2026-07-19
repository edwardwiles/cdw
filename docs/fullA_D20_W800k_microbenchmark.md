# Full-A D=20 real-data (W=800,000): production-speed microbenchmark + memory safety

Continuation 9, Phase 2 (W=800,000 half), **completed** — supersedes the
earlier partial version of this document (memory-safety-only, deferred
scope). This run uses the **production-speed architecture** (compressed
moment mode, opt-in-only dense self-validation, winner-margin certificate,
staleness-aware `BandwidthCachePolicy`) established by Phases 3-5, NOT the
old dense-only path those sections profiled before those speedups existed.
Measured on `demand.mit.edu`, `JULIA_NUM_THREADS=20` (main run) / swept
1/5/10/20 (thread sweep), `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
throughout, commit **`7783ad3`** (branch `c9-phase2and9-w800k`, forked from
`diag/fullA-d4-exact`). Real-data context via `context_real_d20.jl`'s
`d20_real_setup` (France focal, `baseIndex=2`, σ=2.5, μ estimated via
gravity) — identical calibration to the W80k doc, only `W` differs.
Harness: `full_aod_diag/d4_exact/c9_w800k_timing_probe.jl` (scoping probe,
run first) + `c9_w800k_microbenchmark.jl` (main single-process 4-point
breakdown) + `c9_w800k_threadsweep.jl` (thread sweep) +
`c9_w800k_memsafety_probe.jl` (original safety-only probe, §0 below, kept
as historical context) + `c9_w800k_concurrent_memcheck.jl` (Phase 9
two-context concurrency check, cross-referenced from
`docs/fullA_D20_W800k_fallback_readiness.md`). Raw logs + CSVs:
`results/fullA_d4/7783ad3/c9_w800k_microbenchmark/`.

## Headline findings

1. **Compressed mode's advantage holds at W=800,000, and changes character**:
   warm speedup **1.685x** (vs W80k's 1.73x — essentially the same), but
   **cold now FAVORS compressed too (1.175x)** — reversing W80k's finding
   that cold was a slight net loss for compressed (0.923x, dense faster).
   Moment-build speedup **5.02x** (vs W80k's 5.74x, same order — the
   D-scaling trend 2.35x→5.74x, D=4→20, does not extend cleanly to a
   10x-more-draws axis at fixed D=20, reported honestly as "holds, roughly,
   not exactly").
2. **Threading benefit on the full gradient is LARGER at W=800,000 than at
   W=80,000, and grows with thread count**: the clean thread-sweep
   measurement (§7) gives **7.33x** at 20 threads (361.77s serial → 49.39s),
   vs W80k's own 5.47x — and the gap widens at every intermediate thread
   count too (5 threads: 3.49x vs 3.03x; 10 threads: 5.30x vs 4.41x). Value
   callback threading benefit stays modest (1.37x at NT=1→20, matching
   W80k's 1.41x).
3. **Memory is safe and stable across the full production-speed workload,
   not just the single-point safety probe**: peak VmHWM **21.07GB** across
   the full main-run (dense+compressed value in both modes, 3 gradient
   `h_mode`s, 5 directional secants, 4-point coverage) — only 4.5GB above
   the original safety probe's single-cold-eval 16.55GB, confirming the
   fuller production workload does not open a new memory hazard. `jac_h`
   (the original bug) confirmed absent (0 bytes observed vs its
   theoretical ≈1.09TB if the bad default were still active).
4. **All 4 outer points feasible on the first try** (matching W80k exactly)
   — calibration, gravity-tangent (gravity exact to `-2.34e-15`), upper
   branch (offset 0.01), lower branch (offset 0.01).
5. **Warm-start stability confirmed directly** (Phase 9 requirement): a
   cold `evaluate_fullA` (68.16s) immediately followed by a warm one
   (37.99s) at the same point gives a bit-identical `Delta_dual`
   (diff = 0.0) and a genuine 1.79x speedup — no crash, no numerical
   drift, at 10x the draws of the previously-validated W=80,000 case.
6. **5/5 directional secants finite and sane** at Point 1 — the same
   A-block sign-disagreement pattern this investigation has flagged before
   (memory: `full-A-continuation-8` §9) is visible here too (e.g. dir 1's
   secant is negative while `h_mode=:fixed`'s local prediction differs in
   sign in the W80k companion study) but is not new, not chased further
   here, consistent with prior reporting discipline.

---

## 0. Original memory-safety probe (historical context, kept as-is)

The very first attempt to characterize W=800,000 memory usage (pre-fix,
`needs_outer_moment_jacobian=true` default inherited from `context_scaled.jl`'s
diagnostic convention) was killed after climbing to **~780GB VmHWM and still
rising** — headed past 1TB on a shared 3TB machine — caught via a user-flagged
server-wide memory alert (23.5%+ and growing). Root cause and fix (commit
`bd313a2`) are documented in `docs/fullA_D20_W80k_microbenchmark.md` §0: `jac_h`
is a dense `W × (nTotalMoments+2) × l_full` tensor, `≈109GB` at W=80,000 and
`≈1.09TB` at W=800,000 at the bad default. The production convention
(`needs_outer_moment_jacobian=false`, matching `run_fullA_D4/D10_production.jl`)
removes this tensor entirely.

Script: `full_aod_diag/d4_exact/c9_w800k_memsafety_probe.jl`. One context build
+ one cold `evaluate_fullA` call at the natural-theta (calibration) point,
monitored externally via `/proc/<pid>/status` `VmHWM` polling with an automatic
200GB safety kill armed (never triggered).

| | value |
|---|---|
| `d20_real_setup(W=800000)` wall (cold, JIT paid) | 187.02s |
| `evaluate_fullA` cold eval wall | 73.49s |
| inner_status | 0 (clean convergence) |
| Delta_dual | 0.00026237 |
| gravity_raw | -3.38e-16 (machine zero) |
| **Peak process VmHWM (external, whole run)** | **16.55 GB** |

This established that W=800,000 is memory-safe for a single point, but left
open whether a fuller production-speed workload (compressed mode, multiple
`h_mode`s, repeated calls) stays safe — §1-6 below answer that directly.

---

## 1. Scoping probe (calibrates the main run's rep counts)

Script: `c9_w800k_timing_probe.jl`, run once before the main harness, per
this investigation's standing "small sanity check before every new code
path at W=800,000" discipline. Also the FIRST time `:compressed` mode was
timed at W=800,000 (Phase 3.1 only ported/timed it at W=80,000).

| step | wall (s) | VmHWM after (GB) |
|---|---|---|
| `d20_real_setup(W=800000)` | 172.5 | 16.96 |
| `evaluate_fullA` cold | 68.54 | 16.96 |
| `evaluate_fullA` warm (2nd call) | 38.3 | 18.13 |
| `evaluate_fullA_fast` dense, 1st call | 23.67 | 18.13 |
| `evaluate_fullA_fast` dense, 2nd call | 18.54 | 18.13 |
| `evaluate_fullA_fast` compressed, 1st call (NEW) | 21.91 | 18.13 |
| `evaluate_fullA_fast` compressed, 2nd call | 11.06 | 18.13 |
| `solve_base_state` | 19.22 | 18.13 |
| `composite_gradient_at_fast` adaptive, threaded | 52.14 | 18.13 |
| `composite_gradient_at_fast` cached, cold dict | 41.9 | 20.45 |
| `composite_gradient_at_fast` cached, warm dict | 29.03 | **26.87** |

`|Delta_dual diff dense vs compressed| = 5.96e-19` — exact agreement,
`COMPRESSED_FALLBACK_COUNT=0` (no price ties, as expected). This probe's
own peak (26.87GB, reached only after exercising 3 distinct gradient
`h_mode`s in one process) already showed memory would climb somewhat above
the single-point 16.55/16.96GB figures as more code paths run — motivating
close per-step VmHWM tracking in the main run below, not just a single
end-of-run number.

---

## 2. Part 0 — context/setup (main run, measured once)

| | value |
|---|---|
| D | 20 |
| n_free (`1+D^2`) | 401 |
| nTotalMoments | 402 |
| `d20_real_setup(W=800000)` wall (cold, JIT paid) | 176.75s |
| VmHWM after setup | 17.03 GB |
| `gc_live_bytes` delta during setup | 13563.1 MB |
| `ctx.U` shape | **800000 × 20** (128.0 MB) |

`gamma'_focal` (natural theta, calibration) = **0.9877618976237339**,
bounds = **[0.9307117545219048, 1.0]** — identical to the W80k doc (same
calibration point, only `W` differs), confirming apples-to-apples
comparability throughout this document.

## 3. Warm-start stability check (Phase 9 requirement)

A cold `evaluate_fullA` call immediately followed by a warm one at the
exact same point, before any other timed measurement:

| | value |
|---|---|
| cold wall | 68.16s |
| warm wall | 37.99s |
| **speedup** | **1.79x** |
| `|Delta_dual diff|` | **0.0** (bit-identical) |
| both crash-free and finite | **true** |

Confirms directly (not inferred from the smaller-W case) that inner warm
starts behave sanely at W=800,000: no crash, no numerical drift, a genuine
(if modest, since a fresh point still needs real KNITRO iterations) speedup.

## 4. The four outer points

All four feasible **on the first try**, exactly matching the W80k doc's own
finding:

| point | inner_status | Delta_dual | note |
|---|---|---|---|
| 1. calibration (natural theta) | 0 | 0.00026237076791181784 | reused from §3's warm-start check |
| 2. gravity-tangent perturbation | 0 | 0.0002779727070384637 | step=0.02, gravity_raw=-2.34e-15 (machine zero) |
| 3. upper branch (gp0·1.01) | 0 | 0.21690370705662493 | first offset (0.01) worked |
| 4. lower branch (gp0·0.99) | 0 | 0.12689170838791308 | first offset (0.01) worked |

Points 3/4's ~800x jump in `Delta_dual` from calibration reproduces the
W80k doc's own finding of a tight, asymmetric `[0.9307, 1.0]` bound on
`gamma'_focal` driving extreme local sensitivity — not re-derived here,
just confirmed present at W=800,000 too.

## 5. Part 1 — full breakdown at Point 1 (calibration)

### 5A. Value callback: dense AND compressed mode (the direct W=800,000 test)

Warm-started (N=4 dense, N=4 compressed), then cold (N=2 each), reusing
`c8_perfprofile_harness.jl`'s `@prof`-timer discipline directly (same
`instrumentation.jl`, same component-table format as the W80k doc).

**Dense, warm (N=4):**

| component | median (ms) | % of TOTAL |
|---|---|---|
| **TOTAL** | **18862.5** | 100% |
| inner_moment_build | 12259.9 | **65.0%** |
| inner_dual_hessian_callback (7 calls) | 5114.9 | 27.1% |
| winner_compute | 3222.0 | 17.1% |
| moments_reuse | 1443.4 | 7.7% |
| inner_dual_fg_callback (11 calls) | 554.7 | 2.9% |
| inner_knitro_dual_solve | 539.3 | 2.9% |
| kkt_residual_compute | 496.5 | 2.6% |
| moment_resid_compute | 467.2 | 2.5% |
| primal_weight_recovery | 278.9 | 1.5% |
| primal_divergence_compute | 7.4 | <0.1% |

`inner_moment_build` at 65.0% of TOTAL **exactly reproduces the W80k doc's
own 65.0% finding** — this bottleneck profile is stable across a 10x
change in `W` at fixed `D=20`, a useful cross-scale confirmation that it is
a genuine `O(W·D²)`-dominated cost, not a W80k-specific artifact.

**Compressed, warm (N=4)** — never timed at W=800,000 before this task:

| component | median (ms) |
|---|---|
| **TOTAL** | **11278.6** |
| inner_dual_hessian_callback_compressed (2 calls) | 6202.6 |
| winner_compute_compressed | 3208.2 |
| materialize_dense_for_postproc (NEW, compressed-only) | 2654.0 |
| inner_moment_build_compressed | 2441.6 |
| moments_reuse_compressed | 1434.6 |
| kkt_residual_compute_compressed | 496.3 |
| moment_resid_compute_compressed | 470.8 |
| primal_weight_recovery_compressed | 237.9 |
| inner_knitro_dual_solve_compressed | 88.0 |
| inner_dual_fg_callback_compressed (6 calls) | 79.1 |

**Headline dense-vs-compressed table (W=800,000):**

| | dense | compressed | speedup (dense/compressed) |
|---|---|---|---|
| warm TOTAL | 18.777s | 11.144s | **1.685x** |
| cold TOTAL | 41.523s | 35.340s | **1.175x** |
| warm `inner_moment_build` only | 12.260s | 2.442s | **5.021x** |

**This is the headline W80k-vs-W800k comparison the task asked for.**
Warm speedup (1.685x) matches W80k's 1.726x closely — the warm-callback
advantage is essentially W-independent at fixed D=20. **Cold speedup
flips sign**: W80k found cold compressed 0.923x (SLOWER than dense);
here cold compressed is 1.175x (FASTER). At W=80,000 the extra
`inner_knitro_dual_solve_compressed` cost (more KNITRO iterations needed
cold, per Phase 3.1's own diagnosis) outweighed the moment-build saving;
at W=800,000 the moment-build saving (12.33s→2.43s dense→compressed, an
absolute ~9.9s win) is now large enough in absolute terms to outweigh
whatever extra cold-iteration cost remains — a genuine, evidence-backed
finding that the W80k doc's cold-mode recommendation ("default to
compressed only for warm-heavy workloads") does **not** automatically
carry over to W=800,000, where compressed looks like the better default
unconditionally. Moment-build speedup (5.02x) is close to but below W80k's
5.74x — reported as "holds, same order of magnitude, not an exact
extension of the D-scaling formula," consistent with this investigation's
established discipline of not overfitting cross-scale comparisons.

Correctness: `|Delta_dual diff dense vs compressed| = 5.96e-19` (scoping
probe, §1) and `4.93e-32`-level agreement patterns consistent with the
W80k doc's own findings — not independently re-derived here since Phase
3.1 already established compressed-mode correctness at D=20 in general;
this task's contribution is the W=800,000 timing, not a new correctness
proof.

### 5B. `evaluate_fullA` (plain oracle), warm-started, N=3

| | median |
|---|---|
| total | 39.325s |
| inner (KNITRO solve) | 12.809s |
| post (moment/gravity/KKT bookkeeping) | 26.318s |

Post-processing (26.3s) dominates the inner solve (12.8s) by ~2x, matching
the W80k doc's own finding (there: post 2.41s vs inner 1.27s) — the plain
oracle's double moment-matrix rebuild remains the reason to avoid it in
performance-sensitive D=20 code, now confirmed at W=800,000 too.

### 5C. Full 400-coordinate `L_fix` gradient — three `h_mode`s, threaded, N=2/config

Per this task's explicit instruction to use "Phase 5's recommended
bandwidth policy," this run adds `h_mode=:cached` (wrapped in
`BandwidthCachePolicy`, primed once before timing) alongside `adaptive`
and `fixed`, instead of re-measuring the serial/threaded matrix (that data
point is supplied separately by the thread-sweep's own NT=1 process, §6).

| h_mode | threaded | median | speedup vs adaptive |
|---|---|---|---|
| adaptive | true | **48.418s** | 1.00x |
| fixed | true | 29.458s | 1.64x |
| cached (primed) | true | **28.969s** | **1.67x** |

`cached` (Phase 5's recommended production lever) is marginally faster
than `fixed` while retaining genuine per-coordinate adaptivity (unlike
`fixed`, which never adapts) — directly reproducing the W80k doc's own
`cached_warm`-beats-`fixed` finding (there: 2.958s vs 3.024s) at 10x the
draws. The relative adaptive-vs-fixed/cached gap (1.64-1.67x) is smaller
than W80k's 1.82-1.86x — a genuine, reported difference, plausibly because
the fixed per-coordinate overhead this gap comes from (the h/2 diagnostic
probe `adaptive`/`cached` both still pay on a cache miss, vs `fixed`'s
total skip) is a smaller fraction of a much larger per-coordinate O(W)
cost at W=800,000.

### 5D. Optimized-value directional secants — 5 random directions, h=0.02

Matching the W80k doc's §3F methodology exactly (2 full warm-started
`evaluate_fullA` re-solves per direction, bypassing the incremental
machinery entirely), but only 5 directions (not 20) per this task's
explicit scope — the thorough 20-direction validation is a separate,
concurrent Phase 6 task.

| dir | secant | wall+ (s) | wall− (s) |
|---|---|---|---|
| 1 | -5.731e-5 | 55.45 | 55.68 |
| 2 | -3.547e-6 | 58.20 | 57.24 |
| 3 | -7.400e-6 | 54.45 | 54.91 |
| 4 | 9.916e-5 | 54.21 | 54.40 |
| 5 | -2.064e-4 | 54.49 | 54.52 |

**5/5 finite and sane** (all `inner_status=0` at both endpoints). Each
warm-started optimized solve costs ~54-58s at W=800,000 (vs W80k's
~5.1-5.8s — a ~10x scaling, matching the 10x draw increase closely, a
cleaner O(W) signal than several other measurements in this document
since this path re-solves the inner CC dual from scratch and is dominated
by O(W) work with comparatively little fixed overhead). Direction-1's
secant magnitude (-5.73e-5) is the same order as the W80k doc's own dir-1
secant (6.38e-5, opposite sign) — different sign/magnitude is expected
since `drawU()`'s RNG stream does not nest across different `W` (documented
in this investigation's own `c8_nestedw_context.jl`), so W=80,000 and
W=800,000 see genuinely different noise realizations at "the same"
direction index, not a discrepancy to reconcile.

## 6. Part 2 — lighter checks at points 2-4

Value (dense, warm, N=2) + full gradient (`h_mode=:cached`, primed,
threaded, single rep), matching the W80k doc's own §4 scope:

| point | value (warm) median | full gradient (cached, threaded) |
|---|---|---|
| 2 (gravity-tangent) | 18.143s | 28.792s |
| 3 (upper branch) | 18.211s | 29.227s |
| 4 (lower branch) | 18.621s | 29.244s |

All three closely match Point 1's own numbers (18.777s dense-warm value /
28.969s cached gradient) despite `Delta_dual` ranging from near-zero
(points 1-2) to ~0.13-0.22 (points 3-4) — **reproducing the W80k doc's own
"cost is essentially independent of proximity to the feasibility
boundary" finding** at W=800,000, across all 4 of this document's points
(none of which probed genuine near-infeasibility, an open question this
document shares with the W80k doc).

## 7. Thread-count sweep

Value callback and full 400-coordinate gradient, `JULIA_NUM_THREADS ∈
{1,5,10,20}`, `OPENBLAS_NUM_THREADS=1` fixed throughout, 4 separate
processes (Julia thread count is not runtime-configurable), each paying
its own setup cost. **Value callback is the mandatory measurement per this
task's brief; the full gradient sweep is included too** (N=1 per thread
count given W=800,000's cost, vs the W80k doc's N=3 — an explicit,
reported scope reduction, not an omission).

| threads | setup wall | value median | full gradient (N=1) | value speedup vs 1 | gradient speedup vs 1 |
|---|---|---|---|---|---|
| 1 | 210.13s | 26.122s | 361.765s | 1.00x | 1.00x |
| 5 | 181.89s | 22.209s | 103.672s | 1.18x | **3.49x** |
| 10 | 175.14s | 19.514s | 68.204s | 1.34x | **5.30x** |
| 20 | 176.15s | 19.032s | 49.385s | 1.37x | **7.33x** |

**Cross-process consistency check**: the NT=20 gradient here (49.385s,
measured in a completely independent process) matches the main run's own
Part 5C `adaptive` measurement (48.418s) to within 2% — strong confirmation
that both numbers are measuring the same real quantity, not an artifact of
either harness.

**The threading benefit on the full gradient is consistently LARGER at
W=800,000 than at W=80,000, at every thread count**:

| threads | W80k gradient speedup | W800k gradient speedup |
|---|---|---|
| 5 | 3.03x | **3.49x** |
| 10 | 4.41x | **5.30x** |
| 20 | 5.47x | **7.33x** |

This directly answers this task's own headline question (§Headline finding
2): the gap widens monotonically with `W`, consistent with a growing
`O(W)`-per-coordinate cost dominating an increasingly small fixed
per-call/per-thread overhead as `W` grows — the more work each thread does
per dispatch, the closer the achieved speedup gets to the thread count
itself (7.33x at 20 threads is 37% of ideal linear scaling, vs W80k's
5.47x being 27% of ideal). **Value callback threading benefit stays
modest and roughly W-independent** (1.37x here vs W80k's 1.41x) — expected,
since `evaluate_fullA_fast`'s dominant cost (`inner_moment_build`, a single
dense BLAS `gemv!`) is not touched by `composite_gradient_at_fast`'s
coordinate-level `Threads.@threads` loop at all.

Setup wall time is flat-to-slightly-declining across thread counts
(210→176s, NT=1→20) — the same "setup cost is not meaningfully threaded"
pattern the W80k doc found (there: 53-60s, no trend), now confirmed at
W=800,000 too (a mild downward trend here is more plausibly cross-run
noise on a shared machine than a genuine setup-threading effect, since
`d20_real_setup`'s own dominant cost, per the W80k doc, is the untreaded
two-way FE gravity regression).

## 8. Memory ledger

| array | shape | theoretical | observed |
|---|---|---|---|
| `ctx.U` (Frechet draws) | `800000 × 20` | 128.0 MB | **128.0 MB** (confirmed exact) |
| inner K/G moment matrix (dense) | `800000 × 402` | 2572.8 MB | not independently measured (transient, GC'd between calls) |
| `winners.jl` `gap` | `800000 × 20` | 128.0 MB | not independently measured |
| `lfix_incremental.jl` price/runnerup/third/contrib buffers (6×) | `6 × (800000 × 20)` | 768.0 MB | not independently measured |
| `compressed_moments.jl` `wval` | `800000 × 20` | 128.0 MB | not independently measured |
| gradient vector | `400` | 3.2 KB | negligible |
| **`jac_h` (the original bug, §0)** | `800000 × 404 × l_full(~423)` | **≈1.094 TB** | **0.0 bytes confirmed absent** |
| **process VmHWM, main run (dense+compressed value, 3 gradient h_modes, 5 secants, 4 points)** | | | **21.07 GB** |
| **process VmHWM, scoping probe (also exercises 3 gradient h_modes, less coverage)** | | | 26.87 GB |
| **process VmHWM, original single-point safety probe** | | | 16.55 GB |

**Interpretation**: peak memory grows modestly (16.55GB → 21.07-26.87GB,
still well under 1% of this machine's 3TB) as more code paths are
exercised in one process — consistent with JIT/compile-time memory and
KNITRO's own internal workspace accumulating across distinct code paths,
not a leak (the main run's `Base.gc_live_bytes()` at the very end was
12630.7 MB, well below its own VmHWM, the same "transient high-water mark,
not steady leak" signature the W80k doc's own memory ledger documented).
**The single most important number remains the same as the W80k doc's own
finding**: every enumerable steady-state array here is under 3GB combined
(mostly the theoretical `inner K/G moment matrix` at 2.57GB, which itself
is transient), dwarfed by three-to-four orders of magnitude by what
`jac_h` would have cost had the original bug still been present — the
headline lesson stays "audit flags with hidden dimension-dependent tensor
sizes," now reconfirmed at 10x the draws.

## 9. What this document does NOT cover (explicitly out of scope, not overlooked)

- **A full 400-coordinate finite-difference gradient cross-check** — §5D's
  5-direction secant check is the substitute per this task's explicit
  scope; the thorough ≥20-direction validation is Phase 6, a separate
  concurrent task.
- **Points closer to genuine infeasibility** — all 4 points here (and in
  the W80k doc) were found feasible on the first offset tried; this
  document says nothing about cost or reliability near a real feasibility
  boundary at W=800,000.
- **Serial (`threaded=false`) full-gradient timing in the MAIN run** —
  deliberately not re-measured there (cost-prohibitive to duplicate);
  supplied instead by the thread-sweep's own NT=1 process (§7).
- **Two-context-concurrent memory testing** — covered separately in
  `docs/fullA_D20_W800k_fallback_readiness.md` (Phase 9's explicit
  "two-branch concurrency" requirement), not duplicated here.

## 10. Files

New this session, all under `full_aod_diag/d4_exact/`:
`c9_w800k_timing_probe.jl` (scoping probe), `c9_w800k_microbenchmark.jl`
(main 4-point harness), `c9_w800k_threadsweep.jl` (thread sweep),
`c9_w800k_concurrent_memcheck.jl` (Phase 9 concurrency check, see the
fallback-readiness doc). No existing file was modified — this task is a
pure measurement/documentation extension of the Phase 3-5 architecture,
consistent with those phases' own "additive only" discipline. Raw logs +
CSVs: `results/fullA_d4/7783ad3/c9_w800k_microbenchmark/`.
