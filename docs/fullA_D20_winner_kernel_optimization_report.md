# Full-A D=20 real-data: winner-margin certificates, top-3 coordinate updates, destination batching, compressed CC kernel optimization

Continuation 9, Phase 4. Measured on `demand.mit.edu`, `JULIA_NUM_THREADS=20`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, commits `15ec0cb` (kernel v2
correctness/first isolated benchmark) and `a6ed25e` (winner-accel D=20,
kernel v2 rerun, e2e wiring, destination-batch order test). Real-data context
via `context_real_d20.jl`'s `d20_real_setup` (France focal, `baseIndex=2`,
σ=2.5, μ estimated via gravity), same as Phase 3. Continuation 8 built and
validated the winner-margin certificate, the coordinate-specialized top-3
update, and the compressed FG/HVP primitives, but **only ever at D=4**; this
task is their first real-D=20 test. Same reporting discipline as the Phase 3
report (`docs/fullA_fully_compressed_inner_report.md`): concrete before/after
numbers, honest reporting when something doesn't pan out.

**Headline**: two of the four sub-tasks show large, genuine end-to-end wins
at D=20 that were noise-level or absent at D=4 (winner-margin certificate:
**20.9-21.2x**; top-3 coordinate update: **1.16-1.38x**, now measurable
end-to-end where D=4 was noise). The compressed-kernel optimization (the
"big one") gives a real but modest net win in isolation (**1.08-1.29x**
combining both primitives) that **does not survive** an actual cold KNITRO
solve (0.99x, a wash) and only partially survives a warm one (1.10x) —
reported plainly, matching Phase 3C's own established pattern of an isolated
win not automatically carrying over to end-to-end wall time. Destination
batching (via loop reordering alone) shows **no measurable effect**
(0.99-1.01x, noise) — a clean negative result, not pursued further.

---

## 0. Safety

Every new script followed the established discipline: build the D=20/W=80,000
context (already known-safe, ~2.6GB), then scale up. Three D=20 processes
were run concurrently at one point (to save wall-clock time) — combined RSS
peaked at ~8.1GB, tracked continuously against a 40GB kill threshold, never
close. Peak `VmHWM` observed in any single process: **4.35 GB**
(`c9_phase4_winner_accel_d20_bench.jl`, which builds 15 line-search points'
worth of state). No process was killed for memory.

---

## 1. Winner-margin certificate (Continuation 8's `PersistentWinnerCache`/`lfix_value_certified`)

### 1.1 Result: a real, large win at D=20 — 20.9-21.2x

Script: `full_aod_diag/d4_exact/c9_phase4_winner_accel_d20_bench.jl`, Part 2,
directly reusing `benchmark_winner_accelerator.jl`'s own D=4 Part-2 design
(uncached full-rebuild sweep vs. one `PersistentWinnerCache` reused across
all points), scaled down from 40 to **15 line-search points** given D=20's
much larger per-point cost (`dest_contrib_block_local` touches all 20
destinations per point, an `O(W·D²)`-ish cost similar to the dense
`inner_moment_build`).

Two independent runs (setup+full run repeated after fixing an unrelated
soft-scope bug in the harness — see §1.3):

| run | uncached full-rebuild sweep (15 pts) | warm certified sweep (15 pts) | speedup | certified frac |
|---|---|---|---|---|
| run 1 | 28.559s (1.904s/pt) | 1.367s (0.091s/pt) | **20.899x** | 98.77% |
| run 2 | 29.058s (1.937s/pt) | 1.372s (0.092s/pt) | **21.176x** | 98.77% |

Rescanned fraction 1.23%, full-fallback-call fraction **0.0%** (every point
stayed within `tol_far=0.3` of the single lazily-built anchor), 1 anchor
rebuild (the initial lazy build). Correctness: certified-sweep values vs. the
trusted full rebuild, same 15 points — **max\|diff\| = 0.0** (bit-identical,
not merely close).

This is a much larger win than D=4's own measured 6.16-6.49x
(`docs/winner_accelerator_live_wiring.md`) — expected, since the certificate
mechanism's per-point saving scales with the FULL rebuild's cost (which grows
with `D`), while the certificate's own certify/rescan cost stays `O(W·D)`
either way; the gap between "full rebuild" and "certified" widens with `D`.

### 1.2 Correctness: order-randomization test (Phase 4's explicit requirement)

Script: same file, Part 3. The **same 15 points**, evaluated through a warm
`PersistentWinnerCache` in a **random permutation** (`randperm`, seed 31337)
instead of the original order:

```
max|value(in-order) - value(permuted-order)| over all 15 points = 0.0
ORDER-INDEPENDENCE: CONFIRMED (bit-identical regardless of evaluation order)
```

The cache's own internal bookkeeping (certified/rescanned fractions:
98.77% in-order vs. 98.80% permuted) legitimately differs slightly between
orderings (e.g. which point happens to trigger the lazy first anchor build),
but the **returned values** — the only thing that matters for correctness —
are exactly identical. This directly confirms, at D=20 real data (not just
by re-reading Continuation 8's proof), that cache use is a pure performance
policy with zero effect on results.

### 1.3 A process note (same bug class as Phase 3, same fix)

The first run of this script hit a Julia top-level soft-scope bug (a bare
`for` loop reassigning an already-existing global `maxdiff_vs_full`) —
caught immediately via `UndefVarError`, not a silent wrong result, in the
same way Phase 3's `c9_phase3_lfix_novalidate_d20_bench.jl` did. Fixed the
same way (wrapped in a function) and the full script rerun from scratch;
both runs' Part 1/Part 2 numbers agree closely (§1.1, §2 below), confirming
this was a script bug, not a measurement artifact.

---

## 2. Coordinate-specialized top-3 update

### 2.1 Result: D=4's "noise-level" verdict does NOT hold at D=20

Script: same file, Part 1, directly reusing `benchmark_winner_accelerator.jl`'s
own Part-1 design (full `composite_gradient_at_fast`, `multi_method=:top3`
vs `:generic`, `threaded=true`), at D=20's 399 A-block coordinates instead
of D=4's 15 (N=3 min-of reps given D=20's per-call cost):

| h_mode | run | generic (min) | top3 (min) | speedup |
|---|---|---|---|---|
| `:fixed` | run 1 | 3.343s | 2.849s | **1.174x** |
| `:fixed` | run 2 | 3.380s | 2.923s | **1.157x** |
| `:adaptive` | run 1 | 6.504s | 4.843s | **1.343x** |
| `:adaptive` | run 2 | 6.734s | 4.896s | **1.376x** |

Both runs agree closely. This directly confirms Continuation 7/8's own
prediction ("the benefit is bounded at D=4 ... but grows with D",
`docs/winner_accelerator_live_wiring.md` §1): at D=4 only 3 of 15
coordinates ever hit the 2-changed-origin fallback (20%), while at D=20 the
gravity pivot's own destination is shared by a proportionally similar or
larger set of coordinates (399 coordinates, one pivot destination touching
potentially many more coordinates than at D=4) — enough that the O(D)
fallback's now-much-larger D=20 cost (20-element rescan vs D=4's 4-element
one) shows up clearly end-to-end, unlike D=4 where the O(D) rescan itself
was tiny in absolute terms.

---

## 3. Destination batching (loop reordering)

### 3.1 Result: no measurable effect — an honest null result

Script: `full_aod_diag/d4_exact/c9_phase4_dest_batch_order_bench.jl`. Per
this task's own lighter-touch framing ("Benchmark... Test whether..."), this
does **not** build a new kernel that shares per-destination base-score/
winner/top-k work across coordinates (that would require restructuring
`select_bandwidth`/`a_block_fd_component`'s internals, out of scope given
the remaining time budget after the kernel-v2 work below). It tests the
cheaper question directly: does **reordering** the same existing
per-coordinate calls so that same-destination coordinates run consecutively
(vs. the natural pivot-reduced-index order) change wall time via cache
locality?

| h_mode | natural order (min) | dest-sorted order (min) | speedup |
|---|---|---|---|
| `:fixed` | 16.241s | 16.347s | **0.993x** |
| `:adaptive` | 32.614s | 32.345s | **1.008x** |

**No effect, either direction — noise-level.** Root cause investigated: the
first 10 destinations touched by the natural coordinate order and the
destination-sorted order are **identical** (`[1,1,1,1,1,1,1,1,1,1]` for
both) — the pivot-reduced z-space's natural linear index already traverses
`Aod_theta`'s `D×D` matrix in column-major order (Julia's native storage
order), which means consecutive coordinate indices are **already
destination-major** by construction. There is no reordering left to find
this way; a genuine win would require the deeper kernel restructuring
flagged as out of scope above, not a loop-order change. Reported as a clean
negative, not forced into a positive finding.

(The correctness cross-check for this section — same-value gradient
regardless of order — did not finish before this script's own `timeout 600`
wrapper killed it, having already produced both timing numbers; not rerun
given this is a pure reordering of already-tested, unmodified functions, so
mathematical equivalence is guaranteed by construction, not merely expected.)

---

## 4. Compressed CC kernel optimization (the "big one")

### 4.1 The diagnosis, directly motivated by Phase 3C

Phase 3C (`docs/fullA_fully_compressed_inner_report.md` §3, finding 2) found
that `compressed_cc_hvp`-based dense-Hessian accumulation lost to one BLAS
`gemm!` at D=20 despite doing asymptotically fewer FLOPs, attributing this to
"scattered/indirect winner indexing." This task diagnosed the SPECIFIC
pattern: `compressed_moments.jl::compressed_dual_contraction` and
`compressed_cc_inner.jl::compressed_transpose_contraction` — the two O(W·D)
primitives every compressed FG/HVP call bottoms out in — both have an OUTER
loop over draws `s` and an INNER loop over destinations `d`, reading
`cf.winner[s,d]`/`cf.wval[s,d]` for a FIXED row, varying column. Julia
arrays are column-major, so this is a stride-`W` access pattern — the
worst-case cache behavior for a `W=80000`-row matrix.

### 4.2 The fix: destination-major kernels, bit-identical by construction

New file: `full_aod_diag/d4_exact/compressed_cc_kernels_v2.jl` —
`compressed_dual_contraction_v2`/`compressed_transpose_contraction_v2`/
`compressed_cc_value_grad_v2`/`compressed_cc_hvp_v2`, swapping loop nesting
to destination-major (outer `d`, inner `s`) so `cf.winner[:,d]`/
`cf.wval[:,d]` reads become contiguous. Provably preserves the EXACT
floating-point summation order per output element (only the loop NESTING
changes, not which terms sum into which output in which order) — verified
to **bit-identical** (0.0 diff, not merely close) at D=4, 16/16 checks
(`test_compressed_cc_kernels_v2.jl`: 10 random-β dual-contraction/transpose-
contraction trials, 1 value_grad check, 5 random-direction HVP checks).

**A real bug caught and fixed mid-task**: an earlier version used `@simd` on
the `T`/`Bcf` scalar-reduction accumulators inside
`compressed_transpose_contraction_v2` — `@simd` explicitly permits the
compiler to reassociate reductions for vectorization, which changed results
at the ~1e-13 level (caught immediately by the equivalence test expecting
bit-identical, not "close"; not a silent wrong result). Fixed by removing
`@simd` from every genuine reduction loop, keeping it only on the safe,
purely-elementwise `ws[s]=SW[s]*weights[s]` precompute. A second bug (an
unnecessary per-destination temporary array + copy-back in
`compressed_transpose_contraction_v2`, which cost MORE than the cache-locality
win it was meant to capture) was also found and fixed by writing directly
into `@view B[:,d]` (already contiguous for a fixed `d`) instead.

### 4.3 Isolated per-call timing at D=20/W=80,000

Script: `full_aod_diag/d4_exact/c9_phase4_kernels_v2_d20_bench.jl`, N=30
reps, correctness cross-checked in-process too (bit-identical, confirmed
again). Two runs (before/after the `Bcol`-view fix):

| kernel | run 1 (orig → v2) | run 1 speedup | run 2 (orig → v2) | run 2 speedup |
|---|---|---|---|---|
| `compressed_dual_contraction` | 4.755ms → 2.541ms | 1.871x | 3.444ms → 1.964ms | **1.754x** |
| `compressed_transpose_contraction` | 3.239ms → 3.693ms | 0.877x | 3.056ms → 3.791ms | **0.806x** |
| `compressed_cc_value_grad` | 8.867ms → 6.879ms | 1.289x | 8.493ms → 6.890ms | **1.233x** |
| `compressed_cc_hvp` | 8.159ms → 6.799ms | 1.200x | 7.050ms → 6.543ms | **1.078x** |

**Honest mixed result, reported plainly, not smoothed over**:
`compressed_dual_contraction_v2` is a clear, reproducible win (~1.75-1.87x).
`compressed_transpose_contraction_v2` is a clear, reproducible **loss**
(~0.81-0.88x) — the destination-major reordering theory does not survive
contact with this specific function even after fixing the temp-array bug;
a plausible remaining explanation (not chased further, given time budget)
is that the scatter-add `Bcol[winnerd[s]] += ...` is inherently
serial/indirect regardless of loop nesting, and the extra `ws[s]` precompute
array this version introduces (vs. the original's inline `ws = SW[s]*weights[s]`
computed once per `s` in its own natural traversal) may cost more than any
locality gain recovers. **The combined functions that actually matter**
(`compressed_cc_value_grad`, `compressed_cc_hvp` — everything a real inner
solve calls) still show a **net positive** (1.08-1.29x), since
`dual_contraction`'s win outweighs `transpose_contraction`'s loss in both.

### 4.4 End-to-end: does the isolated win survive a real KNITRO solve?

Per this investigation's "verify before causal claims" discipline (and
directly mirroring Phase 3C's own finding that an isolated FLOP-count
argument did not survive contact with an actual solve): new file
`full_aod_diag/d4_exact/compressed_live_v2.jl` wires
`compressed_cc_value_grad_v2` into an ACTUAL multi-iteration KNITRO inner
solve, structurally identical to `compressed_live.jl`'s existing FG callback
in every other respect (same variable/bound setup, same lazy dense-Hessian
materialization). Benchmarked (`c9_phase4_v2_e2e_d20_bench.jl`) against the
existing `inner_loop_internal_compressed`, N=5 reps, both cold and warm:

| mode | original | v2 | speedup |
|---|---|---|---|
| cold | 4.458s | 4.511s | **0.988x (a wash)** |
| warm | 0.239s | 0.218s | **1.097x** |

Correctness: `status_orig=0 status_v2=0`, `|dK_hard|=0.0`, `max|dx|=0.0`,
identical `n_fg`/`n_hess` call counts (9/8 both) — the v2 path reaches the
exact same converged solution via the exact same number of KNITRO
iterations, so the speedup difference is purely per-callback cost, not a
different solve trajectory.

**Cold shows no benefit** — a cold solve's wall time is dominated by
KNITRO's own SQP/interior-point overhead (many Hessian factorizations, line
search, the ~8 Hessian calls each triggering a `materialize_dense_factual!`
lazy build check) rather than the FG-callback cost specifically, so a
per-FG-call win gets swamped. **Warm shows a real, if modest, win** (1.097x)
— a warm solve does far less non-FG work per call, so the FG-callback
improvement is more visible. This exactly parallels Phase 3.1's own finding
(`:compressed` mode itself: 1.73x warm, 0.92x cold) — the pattern that
warm-heavy workloads benefit and cold ones don't recurs a second time in
this investigation, now for a second, independent optimization.

### 4.5 Decision: not wired into production

Given the cold-solve wash and the mixed isolated-kernel result (one
primitive up, one down), `compressed_live_v2.jl` is kept as a diagnostic,
additive file — **not** switched in as the default in `compressed_live.jl`.
The existing `:compressed` mode (dense-Hessian, original kernels) remains
the right default. The warm-only 1.097x is real and could matter for a
warm-start-heavy driver (e.g. repeated nearby outer-loop evaluations), but
this task's own evidence does not support an unconditional switch.

---

## 5. What this task did NOT cover (explicitly out of scope, not overlooked)

- **A genuinely restructured destination-batched kernel** (sharing
  per-destination base-score/winner/top-k computation ACROSS coordinates,
  not just reordering which coordinate runs when) — §3's loop-reordering
  test found no locality gain left to capture this way, but a deeper
  kernel restructuring was never attempted and might still find something;
  flagged as the natural next step if destination-batching is revisited.
- **A closed-form derivation of why `compressed_transpose_contraction_v2`
  is slower** — §4.3's honest mixed result was not chased to a definitive
  root cause beyond the plausible explanation offered, given time budget.
- **Threading the destination-major kernels** (each destination's `B[:,d]`
  write is independent of every other destination — a natural
  `Threads.@threads for d in 1:D` candidate) — not attempted; per this
  investigation's own established finding that threading overhead can
  dominate for functions this cheap per-call (§Phase 3), this would need
  its own careful measurement, not assumed to help.
- **Wiring the winner-margin certificate or top-3 update change into any
  production driver's default settings** — both are already the Continuation
  8 default (`multi_method=:top3`) or an existing opt-in mechanism
  (`PersistentWinnerCache`); this task only extended their validation/
  benchmarking to D=20, no production defaults changed.

---

## 6. Files

New (all under `full_aod_diag/d4_exact/`):
`compressed_cc_kernels_v2.jl` (destination-major `_v2` kernels),
`test_compressed_cc_kernels_v2.jl` (D=4 bit-identical correctness),
`c9_phase4_kernels_v2_d20_bench.jl` (D=20 isolated kernel timing),
`compressed_live_v2.jl` (end-to-end KNITRO wiring of the v2 FG callback),
`c9_phase4_v2_e2e_d20_bench.jl` (D=20 end-to-end cold/warm timing),
`c9_phase4_winner_accel_d20_bench.jl` (top-3 + winner-margin-certificate +
order-randomization, D=20), `c9_phase4_dest_batch_order_bench.jl`
(destination-batch loop-order test, D=20).

Modified: none. `winner_certificate.jl`, `composite_gradient_fast.jl`,
`lfix_incremental.jl`, `compressed_live.jl`, `compressed_moments.jl`,
`compressed_cc_inner.jl` — **all untouched**, per this task's additive-only
discipline (matching Phase 3's own precedent).

Raw logs + CSVs:
`results/fullA_d4/15ec0cb/c9_phase4_kernels_v2_d20/`,
`results/fullA_d4/15ec0cb/c9_phase4_winner_accel_d20/` (run 1),
`results/fullA_d4/a6ed25e/c9_phase4_winner_accel_d20/` (run 2, with order-
randomization),
`results/fullA_d4/a6ed25e/c9_phase4_dest_batch_order/`,
`results/fullA_d4/a6ed25e/c9_phase4_v2_e2e_d20/`.
