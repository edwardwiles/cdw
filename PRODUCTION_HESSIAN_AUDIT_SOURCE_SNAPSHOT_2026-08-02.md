# Production Hessian Audit — Source/Runtime Snapshot (2026-08-02)

## Repository and worktree

- Canonical repo: `/bbkinghome/edav/cdw` (origin `git@github.com:edwardwiles/cdw.git`)
- Audit worktree: `/bbkinghome/edav/gravity_robustness/worktrees/audit-production-all-hessian-allocation-efficiency-2026-08-02`
- Branch: `audit/production-all-hessian-allocation-efficiency-2026-08-02`
- Base: `production/fullA-exact`

## Production SHA (confirmed live)

```
commit 21fa6ec3b5e07dc09c3076404899aa0139f2f027
Merge: 377f48e 57f15ce
Author: Edward Wiles <emmabenzvi@gmail.com>
Date:   Sat Aug 1 23:33:02 2026 -0400

    Merge ZC Hessian backend production integration (surgical, gated)
```

Matches the task-provided production HEAD exactly (`21fa6ec3b5e07dc09c3076404899aa0139f2f027`).
ZC production tag `zc-hessian-optimized-production-release-2026-08-01` confirmed present in `git tag -l`.

Per this merge commit's own message, the current production defaults are:

```
CM+ZC:      H_ZZ=blas_syrk   H_CZ=draw_chunk_reordered   H_EZ=drawmajor_v2
Origin-ZC:  H_ZZ=blas_syrk   H_EZ=drawmajor_v2           (H_CZ not applicable)
Julia threads=10, BLAS threads=8 (auto-selected for cm_meanzc/origin_zc)
```

— matching the task's stated intended ZC backends exactly.

## git status (fresh worktree)

```
On branch audit/production-all-hessian-allocation-efficiency-2026-08-02
nothing to commit, working tree clean
```

Untracked files: none (`git ls-files --others --exclude-standard` empty).

## Julia / KNITRO / BLAS

- Julia: `1.12.6` (via juliaup at `~/.juliaup/bin`; NOT `/opt/shared_sw` — that install is known
  broken, see memory `julia-toolchain-use-juliaup-not-shared-sw`)
- `Project.toml` deps include: KNITRO, Enzyme, Mooncake, ForwardDiff, ReverseDiff, JuMP, Sobol,
  HiGHS, DifferentiationInterface, SparseDiffTools/SparseConnectivityTracer, JLD2
- KNITRO: pinned to **13.0.1** via `.knitro_env.sh` (`KNITRODIR=/opt/shared_sw/knitro/13.0.1`).
  14.0.0/14.2.0 are installed on host but NOT covered by the site Ziena license
  (`KN_new()` → `-520`) — do not switch versions without confirming licensing.
- BLAS: `LBTConfig([ILP64] libopenblas64_.so)` (OpenBLAS, ILP64)
- Default (unconfigured) `BLAS.get_num_threads()` = 104, `Threads.nthreads()` = 1 — i.e. neither
  Julia threads nor BLAS threads default to the production policy; both must be set explicitly
  per-run (`julia -t 10`, `OPENBLAS_NUM_THREADS=8`) as production scripts already do.

## CPU / NUMA

- Intel(R) Xeon(R) Platinum 8270 CPU @ 2.70GHz
- 4 sockets x 26 cores x 2 threads/core = 208 logical CPUs
- 4 NUMA nodes, each spanning all 4 sockets in a striped pattern (node0 = {0,4,8,...}, node1 =
  {1,5,9,...}, etc.) — i.e. NUMA nodes are interleaved across logical CPU ids, not contiguous
  blocks. This matters for `taskset` core-range assignment in the ten-by-ten gate (§19): a
  contiguous `taskset -c N-M` range spans multiple NUMA nodes rather than staying local.
- `numactl` not installed on this host — NUMA-aware affinity must go through `taskset` CPU id lists
  directly; cross-node effects will need to be inferred/documented as a limitation per §11 if
  hardware counters aren't available either.
- Current shell affinity: unrestricted (`0-207`).

## Existing reusable infrastructure (found, not yet verified)

The prior ZC production integration (`zc-hessian-optimized-production-release-2026-08-01`, closed
out 2026-08-01) already built substantial infrastructure directly relevant to this audit's scope
for the **ZC-only and CM+ZC** families specifically:

- `full_aod_diag/d4_exact/campaign_cm_family_runner.jl` — canonical production include stack
- `full_aod_diag/d4_exact/gate5_ten_by_ten_resource_gate_mixed_2026-08-01.sh` +
  `gate5_single_solve_worker_2026-08-01.jl` — mixed cm_meanzc/origin_zc ten-by-ten smoke at
  W=100k/K=3, BLAS=8/Julia=10, disjoint taskset ranges (reusable pattern for §19, needs extending
  to all 5 families)
- `docs/ZC_HESSIAN_PRODUCTION_INTEGRATION_2026-08-01.md` — the 6-gate integration process or ZC
  backends, including the exact bugs found (2 missing dispatch branches + 1 gate methodology flaw)
- `docs/CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28_cmzc_originzc.csv` and
  `docs/CMZC_ORIGINZC_D20_PROFILE_AND_HZZ_BENCHMARK_2026-07-28.md` — prior block-level timing for
  ZC families at D20 (starting point for §6, needs re-verification against current HEAD and
  extension to the required exact-operation label granularity)
- `docs/ZC_COMPILE_FREE_BACKEND_AB_2026-08-01.csv` — prior compile-free backend A/B methodology
  (reusable pattern for §15's one-change A/B)

No equivalent artifact set was found yet for **unrestricted / flexible CM / common Fréchet** at
this resolution — those three families will need the block-timing/allocation/type-stability work
built fresh in this audit (§4-§13), following the same conventions as the ZC work rather than
duplicating its methodology from scratch.

## Not yet recorded (pending harness build, §4)

- Live dual dimensions, block dimensions, packed-Hessian lengths per family (require constructing
  each context — deferred to harness build so numbers come from one canonical code path per the
  task's own requirement, not ad hoc re-derivation here)
- Workspace sizes per family
- Confirmed thread settings per family as actually selected at runtime (vs. documented defaults)
