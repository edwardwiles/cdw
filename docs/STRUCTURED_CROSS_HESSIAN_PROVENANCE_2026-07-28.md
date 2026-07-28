# Structured cross-Hessian (ZC/CM) provenance — 2026-07-28

## Isolation

- Repo: `github.com/edwardwiles/cdw` (local clone `/bbkinghome/edav/cdw`)
- Worktree: `/bbkinghome/edav/gravity_robustness/worktrees/optimize-structured-cross-hessian-zc-cm-2026-07-28`
- Branch: `optimize/structured-cross-hessian-ZC-CM-2026-07-28`
- Base: `origin/production/fullA-exact` @ `fb6ad2ea4b0e0e4fc730af4d431e74a5ddc7b744`
  ("Phase 1 reconciliation + fresh pre-merge regression gate reports", 2026-07-28 13:39:47 -0400)
- Working tree at branch creation: clean (fresh `git worktree add` from `origin/production/fullA-exact`)

**IMPORTANT correction to the task's own base-branch assumption**: the task brief instructed
starting from `production/fullA-exact` and warned the diagnostic archive
(`inner_timing_and_termination_diagnostic_2026-07-28.zip`) was captured against an *earlier*
snapshot (`93f26df7`, `release/no-moments-no-composite-G-production-2026-07-28`), not necessarily
current production HEAD. On investigation this local clone's own `production/fullA-exact` ref was
itself **139 commits stale** relative to `origin/production/fullA-exact` — the remote branch
already contains the full "true no-H OperatorPsiBundle" release AND substantially more
cross-Hessian work than the diagnostic snapshot reflects, including (already merged, already
production-default):

- `winner_pair_cross_hessian.jl`: exact, shared, allocation-free `H_EC` (`winner_pair_cross_hessian_fill!`
  / `_cm_block!`) and `H_EZ` (`winner_pair_cross_hessian_zc_block!`, shared verbatim by CM+ZC and
  origin-ZC) kernels, both derived from and validated against dense reference (D=4/D=20 gates:
  `test_flexible_cm_winner_bin_her_wiring_d4.jl`/`_d20.jl`, `test_cm_meanzc_winner_bin_hez_wiring_d4.jl`/`_d20.jl`,
  `test_originzc_winner_bin_her_wiring_d4.jl`/`_d20.jl`).
- `zc_restriction_operator.jl`: exact, shared `H_ZZ = (1/M) Z'SZ` primitive (`zc_restriction_gram!`),
  built directly from raw `Zraw_all`/`Zpairraw_all` feature matrices plus per-outer-point centering
  targets (never from a dense `obj.H` column read) — shared verbatim by CM+ZC's `HMM` block and
  origin-ZC's `HRR` block.
- `bin_zc_cross_hessian_fill!`/`_block!` (also in `winner_pair_cross_hessian.jl`): exact, shared
  `H_CZ = (1/M) C'SZ` primitive for CM+ZC's widened CM-grid cross block.
- `CM_CROSS_HESSIAN_BACKEND_DEFAULT[] = :winner_bin`, `ZC_CROSS_HESSIAN_BACKEND_DEFAULT-equivalent
  flags = :winner_bin` (production defaults, not opt-in) — confirmed by reading
  `cm_hessian_architectures.jl`'s `_cm_cross_hessian_wants_winner_bin` / `_originzc_zc_cross_hessian_wants_winner_bin`
  and their call sites.

So Sections 2–7 of the task's algebra/kernel spec are **already substantially implemented** on the
correct base branch — this was not visible from the diagnostic archive alone because that archive's
snapshot (`93f26df7`) predates a large amount of same-day work that has since landed on
`origin/production/fullA-exact`. The actual, measured, remaining gap (confirmed by direct code
reading, not assumption — see `CROSS_HESSIAN_PRECOMPUTATION_LIFECYCLE_AUDIT_2026-07-28.md`) is
narrower and more concrete than the task brief's own framing suggests: **all four raw-table-fill
routines behind these kernels are single-threaded** (`winner_pair_cross_hessian_fill!`,
`winner_pair_cross_hessian_zc_block!`, `bin_zc_cross_hessian_fill!` are plain serial loops;
`zc_restriction_gram!` is a single-BLAS-thread `gemm!` under this codebase's mandatory
`OPENBLAS_NUM_THREADS=1`), while `H_EE` (`core_exact_hessian.jl::hessian_core_winner_pair!`) and
the CM-grid bin-table construction (`cm_hessian_threaded.jl::build_bin_tables_threaded!`) already
have validated, production-default parallel implementations using an output-ownership
(no-atomics, full-draw-rescan-per-owned-output) idiom. This task's real scope is: **apply that
same established idiom to the four cross-block routines above**, which is exactly what the
diagnostic's own corrected finding (`00_READ_FIRST_CORRECTION_2026-07-28.md`) identified as the
dominant remaining unthreaded cost (39–79% of Hessian-callback time, depending on family).

## Active bundle / family configuration

- No-H `OperatorPsiBundle`: wired and gated for flexible-CM (`6bd7196`, `d7cedb3`); wiring for the
  other 3 restricted families is present in the same release train (`6a8fadc` "All five families:
  wire and gate the true no-H OperatorPsiBundle") — **not independently re-verified in this task**,
  out of scope per the task brief's explicit exclusion of "the true no-H bundle".
- Family block partitions confirmed by direct code reading (see `CMBinHessCtx`/`OriginZCCoreHessCtx`
  field layout, `cm_hessian_architectures.jl`):
  - `unrestricted`: `[E]` only, dense `hessian!`/generic FG.
  - `flexible_cm`: `[E | C]`, `CMBinHessCtx` with `ncore_core == NCORE` (no Z widening).
  - `common_frechet`: `[E | C | F]`, `CMBinHessCtx` + level-anchor block (`cm_frechet_hessian.jl`).
  - `cm_meanzc` (= task's "CM+ZC"): `[E | C | Z]`, `CMBinHessCtx` with `ncore_core < NCORE`
    (widened economic block folds `Z`'s mean/pair columns into `HEE`'s `HEM`/`HMM` corner) **plus**
    a separate `H_CZ` cross block against the CM grid.
  - `origin_zc` (= task's "ZC-only"): `[E | Z]`, `OriginZCCoreHessCtx`, no CM grid at all.
- Dimensions at real D=20/Ddest=19/W=100,000, `destination_sample=:exclude_row`, `L=50`,
  `K_mean=1,K_pair=0` (production default layout): measured directly in
  `CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28.csv` (Section 1 below), not assumed from
  the stale diagnostic snapshot's own dimension table.

## Environment

- Julia: `julia version 1.12.6` (via `juliaup`, **not** `/opt/shared_sw` — that copy is broken per
  standing project memory)
- KNITRO: `13.0.1` (`KNITRODIR=/opt/shared_sw/knitro/13.0.1`)
- Host: 208 logical CPUs (104 physical), 3.0 TiB RAM (same host as the diagnostic archive)
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for all primary Julia-threaded comparisons (project
  standing requirement)
- Sobol draw checksums: verified via `d20_real_setup`'s own `checksum_uniform`/`checksum_transformed`
  fields (SHA-256 of the recovered/transformed draw matrices) at each profiling/gate run — recorded
  per-run in the corresponding CSV/log, not hand-copied from the diagnostic archive (a different W
  or `destination_sample` would produce a different checksum, so the diagnostic archive's own
  checksum is not directly comparable here; this run's own checksum is what matters for
  reproducibility of *this* task's results).

## Clean status

Worktree created directly from `origin/production/fullA-exact` via `git worktree add`, no
uncommitted changes at task start. All work in this task lands as new commits on
`optimize/structured-cross-hessian-ZC-CM-2026-07-28`, never by editing history on
`production/fullA-exact` directly.
