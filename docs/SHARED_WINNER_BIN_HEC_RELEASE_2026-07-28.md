# Shared winner-bin H_EC threaded kernel — release notes (2026-07-28)

## What this is

`winner_pair_cross_hessian_fill_threaded!` (`threaded_cross_hessian.jl`), a threaded drop-in
replacement for `winner_pair_cross_hessian_fill!` (`winner_pair_cross_hessian.jl`, merged
pre-existing, already exact/shared/dense-E-free). Per this task's own audit
(`STRUCTURED_CROSS_HESSIAN_PROVENANCE_2026-07-28.md`), this raw-table build is already the shared
H_EC backend for flexible-CM, common-Fréchet, and CM+ZC's true-economic sub-block (`ncore_core`
columns) — one function, three families, exactly the task brief's own "do not maintain separate
Fréchet and flexible-CM implementations" requirement, already satisfied before this task started.
This release's contribution is threading its two raw-table passes.

## Parallel design (two independent passes, two different ownership axes)

**Pass 1** (`NuTab`/`SOnlyTab`/`QCfTab`, indexed `(origin x, bin b)` only, no slot/column
dependence): partitioned by **origin** (`x in 1:D`) ownership — mirrors H_CZ's own scheme exactly.

**Pass 2** (`QTab`/`EsumEcon`, indexed `(economic column j, origin x, bin b)`, `j = slot +
(o-1)*Ddest` for the winner's origin `o`): partitioned by **destination slot** (`slot in
1:Ddest`) ownership — for a fixed slot, `j` only ranges over `Ddest`-strided values, so distinct
slots never touch the same `QTab` row. Same scheme as H_EZ's own kernel (see
`WINNER_FEATURE_HEZ_RELEASE_2026-07-28.md`) — deliberately reused rather than re-derived, since the
underlying index algebra is identical.

Both passes are followed by the SAME serial `O(D*L)`/`O(D*ncolI*L)` cumulative prefix sum the
serial version already used — left unthreaded (confirmed negligible next to the raw-table fill by
this task's own sub-block profile, not assumed).

Bit-identical to the serial kernel for the same reason as the other three kernels in this task (see
`WINNER_FEATURE_HEZ_RELEASE_2026-07-28.md`'s explanation, identical argument).

## Allocation discipline

`ws.tasks_ec` (persistent `Vector{Task}`, `WinnerBinCrossScratch`, sized to `Threads.nthreads()`)
reused sequentially across both passes (safe — pass 1 fully completes/fetches before pass 2
starts, no data race between the two passes' own task sets). No other new scratch needed.

## Audit of the existing (pre-task) implementation

Per the task brief's own Section 6 instruction ("audit the existing crossprep/winner-bin
implementation... its current final assembly is already negligible"): confirmed directly by this
task's sub-block profile (`H_EC_asm` rows) — the per-`l` block-slice assembly
(`winner_pair_cross_hessian_cm_block!`) is `O(NCORE*nO)` per call, small relative to the `O(W*D*Ddest)`
raw-table fill, and was NOT modified by this task (only the raw-table fill above was threaded).

## Shared by

flexible-CM, common-Fréchet, and CM+ZC's true-economic (`1:ncore_core`) columns — ONE function, all
three, gated behind `cctx.cross_hessian_threaded` (default `false`).

## Correctness gates

`test_threaded_cross_hessian_d4.jl` — `flexible_cm` section (workers ∈ {1,2,4}, calibration +
perturbed, complete packed Hessian bit-exact) and `cm_meanzc` section (same, exercising the
`ncore_core`-widened case together with H_CZ).

## Performance

See `CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28.csv` (`H_EC_prep(crossprep)` vs
`H_EC_prep(crossprep,threaded)` rows) and `CROSS_HESSIAN_WORKER_SWEEP_2026-07-28.csv`.
