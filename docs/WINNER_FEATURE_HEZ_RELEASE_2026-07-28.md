# Winner-feature H_EZ threaded kernel — release notes (2026-07-28)

## What this is

`winner_pair_cross_hessian_zc_block_threaded!` (`threaded_cross_hessian.jl`), a threaded drop-in
replacement for the already-exact, already-shared `winner_pair_cross_hessian_zc_block!`
(`winner_pair_cross_hessian.jl`, merged pre-existing on `production/fullA-exact`). The exact
algebra (`H_EZ = E'SΦ - Esum*t'`, decomposed into a winner-conditioned scatter + rank-1 `pi_vec`
correction) is UNCHANGED — see `ZC_FEATURE_AND_CENTERING_ALGEBRA_2026-07-28.md`. This release is
purely about parallelizing the one loop that dominates its cost.

## Parallel design

The dominant cost is the winner-conditioned scatter:

```julia
for slot in 1:Ddest
    for w in 1:W
        v[w] = Snu[w]*y[w,slot]
    end
    for x in 1:nx, w in 1:W
        o = winner[w,slot]
        j = slot + (o-1)*Ddest
        HEZ[j+1,x] += v[w]*Z[w,x]
    end
end
```

For a FIXED `slot`, `j = slot + (o-1)*Ddest` only ever takes the `Ddest`-strided values `{slot,
slot+Ddest, slot+2*Ddest, ...}` — i.e. every possible origin `o` maps to a DIFFERENT `j`, but all
of them share the same residue `slot` mod `Ddest`. Partitioning the outer `slot` loop across
workers therefore gives each worker a **fully disjoint set of `HEZ` rows** — no two slots ever
write the same `j+1` row. This is output-row ownership, not draw-range ownership: each worker
re-scans all `W` draws but only for its own owned slots.

This is the SAME idiom `core_exact_hessian.jl::hessian_core_winner_pair!` already established in
production for H_EE (column/pair ownership, full draw rescan per owned output) — not a new pattern
introduced by this task, applied here to a different index axis (destination slot instead of
economic column).

No atomics, no reduction step: because each output row is owned by exactly one worker for the
whole call, and that worker accumulates over `w=1:W` in the SAME order the serial version does,
results are **bit-identical** to the serial kernel (verified directly, not assumed — see
`test_threaded_cross_hessian_d4.jl`, `maxdiff < 1e-12` checks, effectively machine-epsilon-tight).

Row 1 (ones/ζ) and the cf/gravity row are NOT winner-conditioned — left as single-thread BLAS
`gemv!` calls (`O(W*n_x)`, small relative to the `O(W*Ddest*n_x)` scatter for any `Ddest` bigger
than a handful).

## Allocation discipline

Persistent per-worker-slot scratch `ws.thread_scratch_ez[k]` (length-`W`, one per possible worker
slot up to `Threads.nthreads()`, built once at `WinnerZCCrossScratch` construction) replaces the
serial version's single shared `v` buffer — avoids a per-callback `Vector{Float64}(undef, W)`
allocation per task. `ws.tasks_ez` (persistent `Vector{Task}`) avoids a per-callback `Vector{Task}`
allocation. Zero allocation per call after warm-up (task construction/`Threads.@spawn` itself has
Julia-runtime overhead but no heap array allocation attributable to this kernel).

## Shared by

CM+ZC (`_fill_cm_HEE!`'s `HEM` block) and origin-ZC (`archA_partitioned_hess_cb_builder`'s `HER`
block) — the SAME function, gated behind `cctx.cross_hessian_threaded`/`octx.cross_hessian_threaded`
(default `false`, opt-in pending the performance gate's own verdict — see
`STRUCTURED_CROSS_HESSIAN_MASTER_REPORT_2026-07-28.md`).

## Correctness gates

`test_threaded_cross_hessian_d4.jl`: D=4, `K_mean∈{1,2}`, `K_pair∈{0,1}`, calibration + perturbed
points, workers ∈ {1,2,4}, both families — complete packed Hessian compared bit-for-bit
(`maxdiff < 1e-12`) against the serial path. Results: see
`STRUCTURED_CROSS_HESSIAN_MASTER_REPORT_2026-07-28.md`.

## Performance

See `CROSS_HESSIAN_WORKER_SWEEP_2026-07-28.csv` and
`CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28.csv` for the isolated-kernel and
sub-block-share numbers at real D=20/W=100,000.
