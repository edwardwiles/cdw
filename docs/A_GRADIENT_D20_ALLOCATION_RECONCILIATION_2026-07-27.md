# A-Gradient D=20 Allocation Reconciliation — 2026-07-27

## Headline finding, stated plainly

**The prior audit's cited "756.5 MB/call pooled at real D=20/W=80,000" figure could not be
reproduced this session, because the function it refers to
(`composite_gradient_at_fast_pooled`) crashes with a `DimensionMismatch` at the actual current
real-D20 production default (`destination_sample=:exclude_row`, D=20 origins/19 destinations).**
See `SHARED_OUTER_A_GRADIENT_ARCHITECTURE_2026-07-27.md`'s "Bug 2" section for the root cause (a
hardcoded square `reshape(x_free0[2:end], D, D)` in both `composite_gradient_at_fast_buffered` and
`composite_gradient_at_fast_pooled`). That 756.5 MB number must date from a run under
`destination_sample=:all_legacy` (the pre-Part-A square mode) or from before the `:exclude_row`
default existed — it is not reproducible against the codebase's OWN current default context, and
this document does not attempt to force-reproduce it via `:all_legacy` (time did not allow a full
second D=20 campaign under the legacy mode; see "What was not done" below).

## What WAS reconciled, exactly, with real numbers

### D=4/W=8000 (exact, byte-for-byte)

`composite_gradient_at_fast_pooled`'s cold-bandwidth-cache total measures **19.1250 MB**. Isolating
each sub-piece with warmed `@allocated` calls on the exact same cache/point:

| Site | Bytes | % of total | Status |
|---|---|---|---|
| `build_lfix_base_cache` (fresh, once/call) | 4.4818 MB | 23.4% | NECESSARY_OUTPUT (per prior audit; not attacked this pass, see task §4 status) |
| `sum(select_bandwidth)`, all 15 A-block coordinates | 12.6449 MB | 66.1% | THE dominant site — `count_winner_flips`/`count_winner_flips_multi_top3`'s Dict + allocating `price_and_pTsigma_cell`. **ELIMINATED by this task's `select_bandwidth!`.** |
| `sum(a_block_fd_component_ws!)`, all 15 coordinates | 1.9900 MB | 10.4% | Isolated to EXACTLY 3 coordinates (k=14,15,16 — the ones sharing destination `d==baseIndex` with the pivot cell), each costing 0.6467 MB vs 0.0041 MB for the other 12 (single-origin) coordinates. This is the "2-changed-origins-in-1-destination" fallback `lfix_incremental_at_ws!` disclosed as allocating. **ELIMINATED by this task's `dest_contrib_incremental_top3!`/`lfix_incremental_at_ws2!`.** |
| **RECONCILED SUM** | **19.1167 MB** | | Matches the measured 19.1250 MB total to within 0.009 MB (rounding/GC bookkeeping noise) |

Raw log: `key_results_shared_a_gradient_2026-07-27/d4_allocation_reconciliation.log` (produced by
`bench_shared_a_gradient_reconciliation_d4.jl`, committed).

After this task's fix, the shared `economic_A_gradient!` measures **4.8807 MB cold-bandwidth-cache**
(74.5% below pooled) and **4.8009 MB warm** (25.8% below pooled's 6.4729 MB warm figure). The
warm-case residual (4.80 MB) is now dominated almost entirely by `build_lfix_base_cache`'s own
4.48 MB necessary-output allocation — i.e. **the coordinate-loop's own contribution to the warm
steady-state cost is now close to zero** (4.80 − 4.48 ≈ 0.32 MB, mostly `a_block_fd_component_ws2!`
and small per-call bookkeeping arrays like `h_used`/`cache_hits`).

### Real D=20/W=80,000 (the current production default, `:exclude_row`)

Since `composite_gradient_at_fast_pooled`/`_buffered` cannot run here (Bug 2), the comparison
actually available is **unbuffered (`composite_gradient_at_fast`) vs shared
(`economic_A_gradient!`)** — the only two functions that both execute correctly at this scale under
the current default:

| Backend | Cold bandwidth cache | Warm bandwidth cache (steady state) |
|---|---|---|
| Unbuffered (`composite_gradient_at_fast`) | 9296.14 MB | 4474.51 MB |
| Shared (`economic_A_gradient!`) | 651.24 MB | 614.83 MB |
| **Reduction** | **93.0%** | **86.3%** |

Both numbers are directly measured, warmed `@allocated` calls at a real solved calibration point
(`d20_real_setup(W=80000, δ=1.0, find_smallest=true)`, `:exclude_row` default), reported in
`key_results_shared_a_gradient_2026-07-27/d20_final_correctness_and_allocation_comparison.log`.
Correctness at the SAME point: `max|Δg| = 0.0` between `economic_A_gradient!` and
`composite_gradient_at_fast` (bit-identical), confirmed BEFORE the allocation numbers were trusted.

## Directionally consistent with the D=4 pattern

The D=20 warm-case residual for `economic_A_gradient!` (614.83 MB) is far larger in absolute terms
than the D=4 residual (4.80 MB) — expected, since `build_lfix_base_cache`'s own necessary W×D×Ddest
arrays scale as O(W·D·Ddest): at D=20/Ddest=19/W=80,000 that is ~250× the D=4/D=4/W=8000 element
count. This is consistent with (not independently re-measured this pass, but structurally
expected from) the pattern the D=4 reconciliation established: once the coordinate-loop's own
allocation is driven to ~0, the remaining cost is almost entirely `build_lfix_base_cache`'s
once-per-gradient-call necessary snapshot — exactly the target task §4 (persistent L-fix base
cache, NOT attempted this pass) would attack next.

## What was NOT done (honest gaps)

- **No `:all_legacy`-mode D=20 re-run** to directly reproduce a genuine (non-crashing)
  `composite_gradient_at_fast_pooled`/`_buffered` number at D=20 scale for a true 4-way
  apples-to-apples comparison against the ORIGINAL cited 756.5 MB figure. This would need a second
  full D=20 context build (`destination_sample=:all_legacy`) plus re-running the same battery —
  straightforward but not fit into this session's remaining time after the two bug
  investigations. Recommended as the first follow-up for whoever picks up task §4/§6 next, since
  it would also validate whether Bug 1 (now fixed) was ALSO silently active under `:all_legacy`
  (it would not have been, since `D==Ddest` there, but this should be independently confirmed, not
  assumed).
- **No profiler-based (`Profile.Allocs`/`--track-allocation`) line-level attribution** — all
  numbers here are warmed `@allocated` calls on isolated sub-pieces, sufficient to attribute bytes
  to a NAMED function but not to a specific line inside a large function without further manual
  isolation (which is what the per-coordinate k=2..16 breakdown above already did manually).
- **`build_lfix_base_cache`'s own internal breakdown** (which of its ~9 W×Ddest/W×D×Ddest arrays
  costs what) was not separately itemized this pass — it is reported as a single 4.48 MB
  (D=4)/large (D=20, not separately measured) NECESSARY_OUTPUT block, per the prior audit's own
  finding, not re-decomposed further.
