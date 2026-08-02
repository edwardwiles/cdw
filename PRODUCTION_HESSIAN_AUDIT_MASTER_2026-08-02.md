# Production Hessian Allocation, Type-Stability, and Inefficiency Audit — Master Report (2026-08-02)

**Branch**: `audit/production-all-hessian-allocation-efficiency-2026-08-02`
**Base**: `production/fullA-exact@21fa6ec` (verified live, matches task brief exactly)
**Worktree**: `/bbkinghome/edav/gravity_robustness/worktrees/audit-production-all-hessian-allocation-efficiency-2026-08-02`
**Status at time of writing**: W=100k gates complete; **W=500k and ten-by-ten gates not yet run**
(checkpointed with the user per this task's own risk/cost profile — see §18/§19 below). Not yet
merged to `production/fullA-exact`; not yet tagged; not yet pushed to Dropbox.

## 1. What this audit covers

All 5 full-(A), gamma-normalized families (unrestricted, flexible common marginals, common Fréchet
marginals, ZC-only/origin-ZC, common marginals+ZC), scoped strictly to: Hessian-callback
allocation, type-stability, workspace lifecycle, repeated work, memory-access patterns, backend
dispatch, and BLAS/thread policy. Per the task brief's explicit exclusions, this audit did NOT
touch reduced/profiled economic moments, outer gradients, dual-bank warm starts, outer solver
behavior, scientific calibration, moment definitions, or restriction definitions. No production
campaign was launched.

## 2. Source/runtime snapshot

See `PRODUCTION_HESSIAN_AUDIT_SOURCE_SNAPSHOT_2026-08-02.md` for the full record. Summary: Julia
1.12.6 (juliaup), KNITRO 13.0.1 (pinned, license-gated), OpenBLAS (ILP64), Intel Xeon Platinum 8270
(4 sockets x 26 cores x 2 threads, 208 logical CPUs, 4 interleaved NUMA nodes), production HEAD and
ZC tag confirmed matching the task brief exactly.

## 3. Canonical audit harness

`full_aod_diag/d4_exact/production_all_hessian_audit_harness_2026-08-02.jl` — merges the two
previously-SEPARATE production include stacks (`campaign_cm_family_runner.jl` for 4 families,
`campaign_unrestricted_runner.jl` for unrestricted; no single file covered all 5 before this audit).
Exposes: context construction, first valid FG (via each family's own `*_base_state` function, which
IS the true-cold inner solve when it's the first solve call in a fresh process), one frozen-state
Hessian callback, N repeated frozen-state callbacks, and (opt-in, `AUDIT_BLOCK_TIMING=1`) block-
level sub-timing. Uses the real production scientific config throughout: sigma=3, own-trade and
Brazil→Korea excluded from gravity, `destination_sample=:exclude_row`, focal country France,
randomized Sobol draws, **K=3 for the ZC families (not the K=1 the existing
`campaign_cm_family_runner.jl` "profile" default silently uses)** — confirmed live via each run's
own `backend_info`/dimension printout, not assumed.

Validated across all 5 families at W=20,000 and W=100,000, both pre- and post-fix.

## 4. Findings

### 4a. Type stability — `TYPE_STABILITY = pass`

The one historically-documented bug class (`fetch(::Task)` → `Any` → boxed accumulation) is
confirmed already fixed at this HEAD (`core_exact_hessian.jl:758`, the only assignment-form
`fetch()` in the audited directories, already `::Tuple{Float64,Float64,Float64}`-annotated). No new
instance found. Full per-block `@code_warntype` sweep (generator expressions, SubArrays,
heterogeneous tuples individually) not exhaustively completed — see
`PRODUCTION_HESSIAN_TYPE_STABILITY_AUDIT_2026-08-02.md` for the precise scope of what was and
wasn't checked.

### 4b. Recurring allocations per callback (post-fix, W=100,000, bytes)

```
unrestricted:     12,034.16
flexible_CM:      41,631.68
common_Frechet:    41,689.8   (was 561,683.76 pre-fix -- 13.3x reduction)
ZC_only:           48,094.64
CM_plus_ZC:        70,807.0   (was 1,996,201.84 pre-fix -- 28.2x reduction)
```

### 4c. Material inefficiencies found (2 fixed, several flagged not fixed)

**Fixed** (commit `8f1151e`, both verified bit-identical against pre-fix at the real D20/W=20,000
calibration point — see `PRODUCTION_HESSIAN_CORRECTNESS_GATES_2026-08-02.md`):

1. **cm_meanzc**, `cm_hessian_architectures.jl::_fill_cm_HEE!`: `@views HEE[ncore+1:NCORE,
   1:ncore] .= transpose(HEM)` allocated a full temporary (96.6% of the family's per-callback
   bytes) because the destination and source views share the same parent array — Julia's broadcast
   aliasing-defensive-copy path materializes a temporary even at genuinely disjoint index ranges.
   Fixed with an explicit loop (confirmed zero-allocation via isolated repro).
2. **common_frechet**, `cm_frechet_hessian.jl::_fill_frechet_level_blocks!`: `cctx.R' *
   Hraw_cmlevel` allocated a fresh vector on every one of a 2500-iteration (`L*L`, `L=50`) loop
   (89% of the family's per-callback bytes) — the same defect class `block_ec`'s own fix
   (elsewhere in this codebase) already addressed for a sibling block, never applied here. Fixed
   with a persistent buffer + `mul!`.

**Flagged, not fixed** (out of this audit's surgical, single-change scope; each documented with
its own reasoning for deferral):

- Unrestricted has NO sub-block profiling instrumentation at all (block-timing map, §4e).
- No runtime call counters exist for the ZC-specific H_ZZ/H_EZ/H_CZ backend dispatchers (only the
  shared H_EE core backend has one) — a real telemetry gap (backend-dispatch gate, §4f).
- The repo's own D4 dense-truth correctness harness (`test_shared_core_hessian_d4_gates.jl`) is
  broken independent of this audit — `:dense_reference` requires `obj.H`, but the current
  `:operator` production default builds an H-less `OperatorPsiBundle`; confirmed via a true pre-fix
  checkout that this failure predates and is unrelated to this audit's changes (correctness gates,
  §4d).
- `Bidx`'s `W x D` column-major layout makes the per-draw bin-table-build inner loop over D origins
  strided rather than contiguous — plausible (not hardware-counter-confirmed) memory-access
  finding, not implemented given its cross-cutting blast radius (memory-access audit, §4g).

### 4d. Correctness gates

```
Gate 1 (exact bit-identity, real point):  PASS (both fixes, max|diff|=0.0 over the complete packed Hessian)
Gate 2 (D4 dense-truth):                  BLOCKED -- pre-existing, unrelated harness defect (see above)
Gate 3 (D20 fixed-state reference):       PASS (subsumed by Gate 1)
Gate 4 (true-cold inner solve):           PASS -- nStatus unchanged (0, both families, both arms);
                                           cold_solve_s unchanged within noise (expected -- dominated
                                           by O(W) KNITRO iteration compute, not callback allocation)
```
Full detail: `PRODUCTION_HESSIAN_CORRECTNESS_GATES_2026-08-02.md`.

### 4e. Block timing map (4 of 5 families instrumented)

Corrected for the profiler's own documented nested-timer double-counting; coverage 99.78-99.98%
across all 4 instrumented families (meets the >=99% requirement). Headline: cm_meanzc's ZC-specific
blocks (H_CZ_prep+H_ZZ+H_ER+H_ER_prep) are 67.1% of its callback, NOT the shared H_EE/H_EC/H_CC
core; flexible_cm/common_frechet are both dominated by `bintables_prep` alone (~55%, untouched by
either accepted fix); origin_zc is 94.4% H_ZZ_gram+H_EZ_fill. Unrestricted has no sub-block
instrumentation (confirmed gap). Full detail: `PRODUCTION_HESSIAN_BLOCK_TIMING_MAP_2026-08-02.md`.

### 4f. Backend-dispatch proof

All 5 families' selected backends confirmed live (config match) via each harness run's own
`backend_info`; the one existing runtime counter (`CORE_HESSIAN_COUNTERS`, H_EE core backend)
confirmed `dense_core_fallback_calls=0` and `winner_pair_parallel_calls` matching the expected call
count exactly. **No equivalent runtime counter exists for the ZC-specific H_ZZ/H_EZ/H_CZ
dispatchers** — flagged as a real gap (see §4c), not silently assumed passing. Full detail:
`PRODUCTION_HESSIAN_BACKEND_DISPATCH_GATE_2026-08-02.csv`.

### 4g. Workspace lifecycle, repeated work, memory access, BLAS/thread policy

- Workspace lifecycle: confirmed the codebase's existing persistent-scratch discipline
  (`Hraw_EC`/`block_ec`/`tls`/`ensure_*_scratch!`) is correctly applied everywhere EXCEPT the two
  sites this audit fixed. Full detail: `PRODUCTION_HESSIAN_WORKSPACE_LIFECYCLE_AUDIT_2026-08-02.md`.
- Repeated work: no case found of two blocks computing the same quantity under silently different
  conventions (the specific trap the task brief warns against); both accepted fixes ARE this
  section's finding, viewed from a different angle (allocation-per-copy, not duplicated
  arithmetic). Full detail: `PRODUCTION_HESSIAN_REPEATED_WORK_AUDIT_2026-08-02.md`.
- Memory access: `Bidx` strided-access finding (§4c); hardware counters unavailable on this host
  (documented limitation, not silently omitted). Full detail:
  `PRODUCTION_HESSIAN_MEMORY_ACCESS_AUDIT_2026-08-02.md`.
- BLAS/thread policy: real sweep (BLAS=1/4/8, the 2 families the current policy governs) confirms
  the production `BLAS_THREADS=8` default is genuinely justified — cm_meanzc 24.5% faster,
  origin_zc 41.4% faster vs BLAS=1, consistent with each family's own BLAS-threaded block share
  from the block-timing map. Full detail: `PRODUCTION_HESSIAN_BLAS_THREAD_POLICY_2026-08-02.md`.

### 4h. Regression safeguards (implemented, not just proposed)

`full_aod_diag/d4_exact/test_hessian_allocation_regression_2026-08-02.jl` — a real, passing
(28/28) `@testset` suite: per-family allocation ceiling (2x margin over this audit's own measured
post-fix bytes), live dual-dimension assertion, selected-backend field assertions, and
`CORE_HESSIAN_COUNTERS.dense_core_fallback_calls==0`. Tolerance ceilings throughout, no fragile
exact-wall-time assertions, per the task brief's own explicit guidance.

## 5. Performance tables

`PRODUCTION_ALL_FAMILY_HESSIAN_PERFORMANCE_AB_2026-08-02.csv` — full baseline/accepted-fixes/
combined table, all 5 families, W=100,000.

## 6. What was NOT done (explicit, not silently skipped)

- Full 6-point BLAS-thread grid ({2,6,10}) and the 3 families outside the current BLAS policy.
- Full per-block `@code_warntype` sweep beyond the fetch/Task check.
- Fixing the pre-existing broken D4 dense-truth harness (a real finding, not this audit's to fix
  unilaterally — needs an architectural decision about `OperatorPsiBundle`-compatible dense-truth
  testing).
- Implementing the `Bidx` transpose memory-layout candidate (cross-cutting, needs its own gates).
- Adding runtime backend counters for the ZC-specific H_ZZ/H_EZ/H_CZ dispatchers (flagged as
  recommended future work, same reasoning).
- Sub-block instrumentation for unrestricted (flagged, low practical value given that family's
  single-block structure and already-small ~30ms callback).
- **W=500,000 confirmation (task §18) and the ten-by-ten mixed-process resource gate (task §19) —
  explicitly checkpointed with the user before running, given their real compute/host-contention
  cost; not yet executed as of this document's writing.**
- Merge to `production/fullA-exact`, tag, or Dropbox push — all gated on the two items above per
  the task's own merge policy (§21: "After all gates... merge only accepted surgical changes").

## 7. Final verdict (current state — W=500k/ten-by-ten pending)

```
TYPE_STABILITY = pass

RECURRING_ALLOCATIONS (bytes/callback, W=100,000, post-fix) =
    unrestricted:12034.16
    flexible_CM:41631.68
    common_Frechet:41689.8
    ZC_only:48094.64
    CM_plus_ZC:70807.0

MATERIAL_INEFFICIENCIES_FOUND =
    cm_meanzc: view-aliasing broadcast defensive-copy in _fill_cm_HEE! (FIXED)
    common_frechet: per-iteration allocating R'*vector product in _fill_frechet_level_blocks! (FIXED)
    unrestricted: no sub-block profiling instrumentation (flagged, not fixed)
    ZC families: no runtime backend counters for H_ZZ/H_EZ/H_CZ dispatch (flagged, not fixed)
    all CM families: pre-existing broken D4 dense-truth harness vs :operator bundle default (flagged, not fixed)
    flexible_cm/common_frechet/cm_meanzc: Bidx column-major strided access in bin-table build (flagged, not fixed)

ACCEPTED_OPTIMIZATIONS =
    cm_meanzc_explicit_loop_fix: 28.2x bytes/callback reduction (1,996,201.84->70,807.0), ~15.8% callback-time speedup (concurrent-run measurement, not independently isolated)
    common_frechet_persistent_buffer_mul_fix: 13.3x bytes/callback reduction (561,683.76->41,689.8), 2.8% callback-time speedup (isolated measurement)

REJECTED_OPTIMIZATIONS =
    none proposed and rejected -- every candidate investigated was either implemented (2) or explicitly deferred as out-of-scope (4, listed above), not proposed-then-rejected on its merits

BACKEND_DISPATCH = pass_all_families (config-match proof, all 5; H_EE core runtime-counter proof, all 5; ZC-block-specific runtime counters do not yet exist -- see MATERIAL_INEFFICIENCIES_FOUND)

W100K_COLD_SPEEDUP (true-cold complete inner solve, cold_solve_s) =
    unrestricted:~1.00x (unaffected, no shared code path)
    flexible_CM:~1.00x (unaffected, no shared code path)
    common_Frechet:~1.00x (0.9995x measured -- expected: fix affects callback allocation, not O(W) KNITRO compute)
    ZC_only:~1.00x (unaffected, no shared code path)
    CM_plus_ZC:~1.00x (1.0033x measured -- same expectation as common_Frechet)

W500K_GATE = not_yet_run (checkpointed with user before proceeding)

TEN_BY_TEN_GATE = not_yet_run (checkpointed with user before proceeding)

SCIENTIFIC_EQUIVALENCE = pass_all (Gate 1 exact bit-identity + Gate 4 nStatus-unchanged both families; Gate 2 D4 dense-truth blocked by a pre-existing, unrelated harness defect, not a finding against the fixes themselves)

PRODUCTION_MERGE = not_ready_pending_W500k_and_ten_by_ten_gates

CAMPAIGN_LAUNCHED = false
```
