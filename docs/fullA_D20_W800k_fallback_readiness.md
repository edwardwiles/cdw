# Full-A D=20 W=800,000 fallback readiness + ten-frontier-point cost projection

Continuation 9, Phase 9. Companion to `docs/fullA_D20_W800k_microbenchmark.md`
(Phase 2, "Part A" in the standing brief) — that document supplies every
timing/memory number cited here; this document answers the readiness
question directly ("could this investigation actually run W=800,000 as a
fallback if W=80,000 turns out insufficient?") and projects the cost of
doing so for ten frontier points. Measured/reasoned on `demand.mit.edu`,
commit `7783ad3` (branch `c9-phase2and9-w800k`).

## 1. Readiness checklist

| requirement | status | evidence |
|---|---|---|
| Context builds within safe peak memory | **CONFIRMED** | 16.55-27GB peak across every W=800,000 run this investigation has done (original safety probe, scoping probe, main 4-point run) — all <0.01% of this machine's 3TB. §2 below extends this to two CONCURRENT contexts. |
| No W-by-400 dense moment matrix mandatory | **CONFIRMED at W=800,000 specifically** | `:compressed` moment mode timed at W=800,000 for the first time this session (`fullA_D20_W800k_microbenchmark.md` §5A): 1.685x warm speedup, and — a genuinely new finding beyond the W=80,000 case — **1.175x speedup even cold** (W80k found cold compressed a net LOSS, 0.923x). Not merely assumed from the W=80,000 result; independently re-verified at 10x the draws. |
| One warm and cold exact value complete | **CONFIRMED** | Warm-start check (§3 of the microbenchmark doc): cold 68.16s, warm 37.99s, bit-identical `Delta_dual` (diff 0.0), 1.79x speedup, both crash-free. |
| One hard `L_fix` gradient completes | **CONFIRMED** | Three `h_mode`s completed cleanly at Point 1: adaptive 48.42s, fixed 29.46s, cached 28.97s (all threaded, N=2 each) — plus a fourth, independent serial (`threaded=false`) measurement from the thread-sweep's NT=1 process (361.77s), giving five total successful full-gradient completions across this task. |
| One selected optimized-value directional secant completes | **CONFIRMED** | 5/5 directional secants finite and sane at Point 1 (microbenchmark doc §5D), each requiring 2 full warm-started `evaluate_fullA` re-solves — 10 total successful optimized-value solves. |
| Inner warm starts are stable | **CONFIRMED** | Same warm-start check as above; additionally, every one of the 4 outer points (§4 of the microbenchmark doc) used `warm=true` after Point 1 with no failures, and the thread-sweep's own warm-started value callback (N=3/N=4 reps per config, 4 independent processes) never failed. |
| Checkpointing and process recovery work | **CONFIRMED APPLICABLE, pattern identified, not newly built** | §3 below. |

## 2. Two-context concurrency (real numbers, not extrapolation)

Per the standing safety discipline ("do not run multiple W=800,000 contexts
concurrently until one-context peak memory is known... you may run more
than one, but stay disciplined... don't jump straight to a 4-context
concurrent test without checking 2 first"), this task ran exactly 2
concurrent W=800,000 contexts (script:
`full_aod_diag/d4_exact/c9_w800k_concurrent_memcheck.jl`, `JULIA_NUM_THREADS=10`
each, `C9_CONCURRENT_TAG=A`/`B`, launched simultaneously), each doing its
own context build + one cold `evaluate_fullA` + one warm `evaluate_fullA`,
monitored externally via `/proc/<pid>/status` polling on BOTH PIDs with a
150GB combined-VmHWM safety kill armed.

| | process A | process B |
|---|---|---|
| setup wall | 194.79s | 179.11s |
| cold eval wall | 72.27s | 69.86s |
| warm eval wall | 39.37s | 39.68s |
| `Delta_dual` | 0.00026237076791181784 | 0.00026237076791181784 (bit-identical) |
| individual final VmHWM | **18.1 GB** | **17.92 GB** |

**Combined peak (approximate, both processes' individual VmHWM summed —
each process only reports its own): ≈36.0GB.**

**Headline: running 2 concurrent W=800,000 contexts costs almost nothing
extra per process.** Every timing (setup, cold, warm) for BOTH processes
falls within a few percent of the equivalent solo-run numbers measured
earlier in this task (solo: setup 172.5-210.1s, cold 68.16-73.49s, warm
37.99-38.3s) — there is no meaningful CPU/memory contention penalty at 2
concurrent contexts on this 208-core/3TB machine, even with each process
requesting 10 threads (20 total, a small fraction of the machine).
Combined memory (≈36GB) is trivial (1.2% of 3TB). **This directly answers
Phase 9's "two-branch concurrency" requirement with real numbers, not
extrapolation from the single-point 16.55GB figure**: 2 concurrent
W=800,000 processes work cleanly, at essentially solo-run speed each.

## 3. Checkpointing and process recovery

This investigation already has a working resumable-batch pattern, used in
production for the D=4 and D=10 driver scripts
(`full_aod_diag/run_fullA_D4_production.jl`,
`full_aod_diag/run_fullA_D10_production.jl`) — not built new for this task,
confirmed applicable by direct code inspection:

```julia
# result_path(bound_name, δval) -> one JLD2 file per (bound, delta) pair
function load_if_done(path)
    isfile(path) || return nothing
    d = try JLD2.load(path) catch; return nothing end  # corrupt/partial -> not done
    get(d, "done", false) === true ? d : nothing
end
...
for δval in DELTA_GRID   # ascending: warm-start chain runs small-delta-first
    existing = load_if_done(path)
    if existing !== nothing
        θcur = existing["theta_min_full"]; continue   # skip, resume from here
    end
    r, γp, κ = solve_bound(fs, δval, θcur)             # θcur = prior delta's solution
    JLD2.save(path, Dict(..., "done" => true))          # save BEFORE marking done
    θcur = r.θ_min_full
end
```

Three properties make this directly applicable at D=20/W=800,000, unchanged:

1. **Per-solve atomicity**: `"done"` is written only after a solve fully
   completes, as the LAST key in the saved dict — a crash mid-write (e.g.
   the machine's own memory-alert kill scenario earlier in this
   investigation) leaves a file that reads as `done=false`/nonexistent on
   the very next `load_if_done` call, never a false positive.
2. **Warm-start chain across delta**: `θcur` carries forward from one
   solved delta to the next SAME-bound-direction solve — this is exactly
   the "continuation savings" §5 below assumes, and it already exists in
   the codebase, not something to newly design for the ten-frontier-point
   run.
3. **Resume is a pure re-run**: restarting the same driver command after a
   kill/crash re-scans existing files and skips everything already
   `done=true` — no separate "recovery mode" flag needed.

**Does it need adaptation for D=20/W=800,000?** One genuine gap, flagged
honestly: `run_fullA_D10_production.jl`'s `solve_bound` calls
`CS.outer_loop_cached` with the ORIGINAL (non-`_fast`) gradient/value
machinery — it predates this investigation's Phase 3-5 compressed-mode and
cached-bandwidth-policy work. **Wiring `evaluate_fullA_fast`/
`composite_gradient_at_fast` (with `moment_representation=:compressed`,
`h_mode=:cached`+`BandwidthCachePolicy`) into a D=20-capable copy of this
driver is real, not-yet-done integration work** — the checkpoint/resume
SCAFFOLDING (JLD2 file-per-solve, `done` flag, warm-start chain) needs zero
changes, but the inner value/gradient calls it wraps do need to be swapped
to the production-speed path this task's whole benchmark validates, or the
ten-point run will pay the OLD (un-accelerated) per-call costs, not the
ones in §4-5 below. This is the single most important actionable item this
readiness check surfaces.

## 4. Per-call cost basis (from Part A, production-speed architecture)

| operation | cost at W=800,000 |
|---|---|
| Context setup (one-time per process) | 176.75s (main run) / 172.5-210.1s (range across all runs this task) |
| Value, compressed, warm | **11.144s** |
| Value, compressed, cold | 35.340s |
| Value, dense, warm | 18.777s (reference/fallback if compressed mode is ever distrusted) |
| Full gradient, `h_mode=:cached`, threaded (20) | **28.969s** |
| Full gradient, `h_mode=:adaptive`, threaded (20) | 48.418s |
| Full gradient, threaded (20) vs serial (1) | 7.33x speedup (measured, §7 of the microbenchmark doc) |

**Recommended production configuration for the ten-point run**:
`moment_representation=:compressed` for values (default-favorable at
W=800,000 in BOTH warm and cold regimes, per §1 of this table — a genuinely
stronger recommendation than the W=80,000 case's warm-only conditional
default), `h_mode=:cached` wrapped in `BandwidthCachePolicy` for gradients
(Phase 5's own recommendation, confirmed still favorable at W=800,000 in
`fullA_D20_W800k_microbenchmark.md` §5C), `JULIA_NUM_THREADS=20`,
`OPENBLAS_NUM_THREADS=1`/`MKL_NUM_THREADS=1`.

## 5. Expected value/gradient calls per point — basis and honest uncertainty

**Neither a genuine D=20 outer-loop call count nor `docs/fullA_D10_upper_gate.md`
was available at the time of writing** — checked directly: the sibling
worktree tracking Phase 7 (`gravity-fullA-d4-c9-phase7`, branch
`c9-phase7-d10gate`) was still at this task's own fork commit (`7783ad3`)
when checked, meaning that rerun had not yet landed. Per this task's own
explicit fallback instruction, the basis used instead is **this
investigation's own D=4 production frontier data**
(`results/c8_frontier_stdout_{upper,lower}_lfixcomposite_sr1_dense_*.log`,
a genuine converged `lfix_composite`+SR1+dense outer-loop run, the same
gradient-method class this task recommends for D=20 minus the compressed/
cached accelerations):

| direction | outer_iters | gradient calls | value calls (`Delta(w)` evals) | value:gradient ratio |
|---|---|---|---|---|
| upper (D=4) | 113 | 114 | 474 | 4.16 |
| lower (D=4) | 51 | 50 | 374 | 7.48 |

**This is explicitly a D=4 (16 free params) data point being extrapolated
to D=20 (401 free params, 25x more) — the single largest source of
uncertainty in this entire cost projection, flagged prominently rather
than smoothed over.** Two considerations bound a reasonable range:

- **Lower bound on iterations needed**: quasi-Newton curvature approximation
  (SR1/L-BFGS) does not literally need `O(n_free)` iterations to work well
  in practice — many real optimization runs on well-conditioned problems
  converge in far fewer iterations than the parameter count. If D=20 scaled
  as mildly as D=4 (e.g., a 2-3x iteration-count increase, not 25x), the
  ten-point projection would look close to the D=4 numbers scaled only by
  per-call cost.
- **Upper bound / genuine risk signal**: this investigation's own D=10
  gated pilot (`docs/fullA_d4_section10_dimension_scaling_c8.md`) found the
  upper direction **budget-stalled at a 120s wall-clock cap even at
  W=8,000** (cheap per-call cost) — D=10 has only 100 free params, a fifth
  of D=20's 401, and it already did not cleanly converge within a
  reasonable budget. This is real evidence that iteration counts (or at
  least wall-clock-to-convergence) grow non-trivially with dimension in
  this specific problem family, not just a generic quasi-Newton caveat.

**Working central estimate used below: 250 outer iterations per point**
(gradient calls ≈ outer iterations, ≈1:1 per the D=4 data; value calls ≈
250 × 5.8, the average of the two observed D=4 ratios) — presented as a
planning number, not a validated one. **Phase 7's D=10 rerun (in progress
in a concurrent worktree as of this writing) is the direct next step that
would replace this extrapolation with real D-scaled data** — this document
should be revisited once that lands.

## 6. Cost projection for ten frontier points (5 upper + 5 lower)

**Per-point cost (compute), central estimate (250 outer iterations)**:

| component | count | unit cost | subtotal |
|---|---|---|---|
| gradient calls | 250 | 28.969s (cached) | 7242s = 2.01h |
| value calls | ~1450 (250 × 5.8 avg ratio) | 11.144s (compressed warm) | 16159s = 4.49h |
| **per-point total** | | | **≈6.5 CPU-hours** |

**Continuation/warm-start savings** (per §3's existing chain-across-delta
pattern): the FIRST point in each branch pays the full 6.5h; **explicitly
NOT validated at the full-outer-loop level this session** (Part B's own
warm-start check only tested a single-call cold-vs-warm comparison, 1.79x,
not a full chained-outer-loop savings) — as a documented, flagged
assumption, this projection applies a conservative 30% per-point reduction
to the 4 SUBSEQUENT points in each branch (consistent with, but not
proven by, the single-call 1.79x/2.2x cold-vs-warm ratios measured
throughout this task):

| branch | point 1 (cold start) | points 2-5 (warm-started, ×0.7) | branch total |
|---|---|---|---|
| upper | 6.5h | 4×4.55h = 18.2h | **24.7h** |
| lower | 6.5h | 4×4.55h = 18.2h | **24.7h** |
| **all 10 points** | | | **≈49.4 CPU-hours** (vs. 65h with no continuation savings) |

**Two-branch concurrency**: upper and lower are independent bound
directions (never warm-start from each other, per the existing driver's
own convention, §3) — the natural, already-supported degree of real
concurrency is **2 processes**, giving:

- **Wall time (2-process, branch-concurrent)**: ≈24.7 hours (the longer of
  the two branches; they're symmetric here so both ≈24.7h) — **just
  over one day**, not the 49.4h serial figure.
- **CPU-hours**: unchanged at ≈49.4 regardless of concurrency (concurrency
  buys wall-clock, not total compute).

**How many W=800,000 contexts can run at once (real numbers, not just the
single-point 16.55GB extrapolation)**: §2 above measured 2 concurrent
contexts directly. Combined with this task's own single-process peaks
(16.55-27GB across every run), and this machine's 2.6TB free / 208 logical
cores:

- **Memory ceiling**: even 20 concurrent W=800,000 processes at the
  BUSIEST single-process peak observed (27GB) would total 540GB — 20% of
  free memory, comfortably safe. Memory is not the binding constraint at
  any plausible concurrency level this investigation would want.
- **CPU/thread ceiling (the real constraint on a SHARED machine)**: at
  `JULIA_NUM_THREADS=20`/process, only ~10 processes fit before
  oversubscribing this machine's 208 logical cores; at 10 threads/process
  (a reasonable compromise, since §7 of the microbenchmark doc shows
  threading gains are already 73% of ideal at 10 threads vs 37% at 20 for
  the gradient, so 10 threads/process sacrifices only a modest amount of
  per-process speed for headroom), **~15-20 concurrent processes fit**
  while leaving room for other users on this shared investigation machine.
- **Practical recommendation**: run the natural **2 branch-concurrent
  processes** (upper + lower) as the primary execution plan — this is the
  degree of concurrency the existing checkpoint/warm-start machinery
  already supports without modification, needs no continuation-savings
  sacrifice, and completes in ≈1 day wall-clock. If additional throughput
  is wanted (e.g., a second independent seed, or exploratory points outside
  the main delta grid), 2-4 MORE concurrent processes could be added
  safely given the memory/CPU headroom confirmed above — but running all
  10 points fully independently (sacrificing warm-start savings entirely)
  would only save ~18h of wall-clock (24.7h → ~6.5h) at the cost of ~15.6h
  MORE total CPU-hours (49.4h → 65h) and 10 concurrent processes' worth of
  shared-machine courtesy — not recommended as the default plan.

## 7. Summary numbers

| | value |
|---|---|
| Total CPU-hours, 10 points (with continuation savings) | **≈49.4h** |
| Total CPU-hours, 10 points (no continuation savings, fully parallel) | ≈65h |
| Wall time, 2-process branch-concurrent (recommended) | **≈24.7h (≈1 day)** |
| Wall time, 10-process fully-parallel (not recommended as default) | ≈6.5h |
| Memory per process (observed range across this task) | 16.55-27GB |
| Memory ceiling for concurrency (this machine) | not binding (≤20% of free memory even at 20 concurrent processes) |
| Recommended simultaneous processes | **2** (upper + lower branch), extensible to 4-6 for additional exploration given confirmed headroom |
| Single biggest open uncertainty | D=20 outer-iteration count (§5) — extrapolated from D=4 (25x fewer free params), not yet empirically measured; Phase 7's D=10 rerun is the direct next validation step |

## 8. What this document does NOT cover (explicitly out of scope, not overlooked)

- **An actual full outer-loop D=20/W=800,000 optimization run** — per this
  task's own framing ("the purpose is readiness, not necessarily a full
  optimization solve"), this document projects cost from validated
  building-block measurements (Part A) plus the best available call-count
  analog (D=4 production data), not from a genuine D=20 run, which does
  not yet exist anywhere in this repo family (confirmed, per Continuation
  9 Phase 1's own audit finding, still true as of this writing for
  W=800,000 specifically).
- **Wiring `evaluate_fullA_fast`/compressed/cached machinery into
  `run_fullA_D10_production.jl`'s D=20-capable copy** — flagged in §3 as
  necessary follow-up work, not performed here.
- **A validated (not assumed) continuation/warm-start savings factor for a
  full chained outer loop** — §6's 30% figure is a documented, flagged
  extrapolation from single-call cold/warm ratios, not a measured
  multi-point chain result.

## 9. Files

New this session: `full_aod_diag/d4_exact/c9_w800k_concurrent_memcheck.jl`
(2-context concurrency probe). No existing file modified. Cross-referenced
throughout: `docs/fullA_D20_W800k_microbenchmark.md` (Part A, the evidence
base for every per-call cost cited here).
