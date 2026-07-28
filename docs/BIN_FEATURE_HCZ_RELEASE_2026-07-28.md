# Bin-feature H_CZ threaded kernel — release notes (2026-07-28)

## What this is

`bin_zc_cross_hessian_fill_threaded!` (`threaded_cross_hessian.jl`), a threaded drop-in
replacement for the already-exact `bin_zc_cross_hessian_fill!` (`winner_pair_cross_hessian.jl`,
merged pre-existing). CM+ZC only (origin-ZC has no CM grid, so no H_CZ block at all). Algebra
unchanged (`H_CZ = (1/M) C'SZ`, built from the already-centered/`S`-weighted `ZcS` scratch shared
with H_ZZ's `:reference` backend — see `ZC_FEATURE_AND_CENTERING_ALGEBRA_2026-07-28.md`).

## Parallel design

```julia
for w in 1:W
    for x in 1:D
        b = Bidx[w,x]
        for j in 1:nz
            ZBinTab[x,j,b] += ZcS[w,j]
        end
    end
end
```

No winner-selection at all — `ZBinTab` is indexed purely by `(origin x, restriction column j, bin
b)`, independent of any per-draw winner outcome. Partitioning the origin axis `x in 1:D` across
workers gives each worker a disjoint set of `ZBinTab` ROWS directly (first index `x`) — the
simplest of the four kernels in this task, and the one the task brief itself anticipated exactly
("one CM origin per worker... a natural 20-thread decomposition" for real D=20).

Each worker re-scans all `W` draws for its own owned origins; no atomics, no reduction. Bit-
identical to the serial version for the same reason as H_EZ (same accumulation order per output
row, different worker executes it).

## Allocation discipline

`ws.tasks_cz` (persistent `Vector{Task}`, sized to `Threads.nthreads()` at `BinZCrossScratch`
construction) — no per-callback task-buffer allocation. No other per-worker scratch is needed (the
kernel writes directly into `ZBinTab`, no intermediate buffer).

## Shared by

CM+ZC only (this block does not exist for origin-ZC or plain flexible-CM/common-Frechet). If a
future common-Fréchet×ZC family is added, the task brief's own priority note says it should reuse
this same kernel rather than a new implementation — no such family exists yet in this codebase, so
this is a forward-looking note, not something gated here.

## Correctness gates

Exercised indirectly through the same D=4 gate as H_EC (both fire together inside
`hessian_cm_structured_v2!`'s per-`l` assembly loop for CM+ZC — see
`test_threaded_cross_hessian_d4.jl`'s `cm_meanzc` section, `cross_hessian_threaded` workers ∈
{1,2,4}, complete packed Hessian bit-exact).

## Performance

See `CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28.csv` (`H_CZ_prep(fill)` vs
`H_CZ_prep(fill,threaded)` rows, `cm_meanzc` family) and
`CROSS_HESSIAN_WORKER_SWEEP_2026-07-28.csv`.
