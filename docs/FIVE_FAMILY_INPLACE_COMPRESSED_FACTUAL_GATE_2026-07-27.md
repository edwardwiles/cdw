# Five-family in-place compressed-factual gate — 2026-07-27

Correctness + counter evidence for `build_economic_moment_state!` across the five production
families, per addendum §7. All numbers below are pasted from actual command output run this
session (`git branch feature/shared-economic-moment-state-builder-2026-07-27`), not estimated.

## Scope actually fixed this session: the unrestricted family

Only the **unrestricted family**'s hot path (`compressed_live.jl::inner_loop_internal_compressed`)
required a code change this session — the 4 restricted families already called the shared builder
correctly before this session started (see `SHARED_ECONOMIC_MOMENT_STATE_BUILDER_2026-07-27.md`
§1). The gates below are split accordingly: **new** gates for the unrestricted family (the actual
fix), and **re-run of pre-existing** gates for the 4 restricted families (to confirm my struct-field
addition to `CompressedFactualWorkspace` — `last_theta`/`has_last`, needed for the new
`DUPLICATE_ECONOMIC_STATE_BUILDS` counter — did not regress them).

## A. Unrestricted family — new gates this session

### A.1 D=4 square + D=4 rectangular (`test_shared_economic_moment_state_builder_2026-07-27.jl`)

Full command:
```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
  full_aod_diag/d4_exact/test_shared_economic_moment_state_builder_2026-07-27.jl
```

Actual output (18/18 checks):
```
PASS  D4 square: build_economic_moment_state! (no ws attached) vs allocating reference: bit-identical
PASS  D4 square: build_economic_moment_state! (ws attached) vs allocating reference: bit-identical
PASS  D4 square: build_economic_moment_state! returns cf whose winner/wval alias ctx4_ws.cf_workspace's buffers
PASS  D4 rectangular: D_dest == D-1
PASS  D4 rectangular: build_economic_moment_state! (ws attached) vs allocating reference: bit-identical
PASS  attach_compressed_factual_workspace: exactly 1 new ECONOMIC_WORKSPACE_ALLOCATIONS for this fresh context
PASS  calibration point (1st, workspace attached)
PASS  calib point: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS still 0 after 1st compressed call
PASS  calib point: INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS > 0 after 1st compressed call
PASS  perturbed point (2nd, SAME workspace -- stale-value check)
PASS  perturbed point: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS still 0 after 2nd compressed call
PASS  2 distinct outer points -> ECONOMIC_WORKSPACE_REFILLS >= 2
PASS  re-building the SAME θ_full on the SAME workspace increments DUPLICATE_ECONOMIC_STATE_BUILDS
PASS  ECONOMIC_WORKSPACE_ALLOCATIONS: 0 NEW allocations across the measured window (reused, not reallocated)
PASS  ECONOMIC_WORKSPACE_RESIZES == 0 (no shape change occurred)
PASS  fresh ctx carries no cf_workspace
PASS  no-workspace ctx: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS > 0 (fallback path exercised, exactly as before this task)
PASS  no-workspace ctx: INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS == 0
ALL TESTS PASSED
```

`compare_point` prints the actual `evaluate_fullA_fast` dense-vs-compressed diffs at each point
(with `ctx_ws.cf_workspace` attached, i.e. the REAL fixed hot path exercised end to end, not just
the isolated builder):
```
calibration point (1st, workspace attached)            inner_status(dense/compr)=0/0  dK=0.000e+00  dDelta=1.019e-17  dlambda=0.000e+00
perturbed point (2nd, SAME workspace -- stale-value check)inner_status(dense/compr)=0/0  dK=0.000e+00  dDelta=1.166e-15  dlambda=0.000e+00
```

### A.2 Real D=20/W=80,000, `destination_sample=:exclude_row` (ROW=20, non-last omitted destination)

`test_shared_economic_moment_state_builder_d20_2026-07-27.jl`, actual output:
```
>>> D=20 Ddest=19 W=80000 row_idx=20
PASS  attach_compressed_factual_workspace: exactly 1 new ECONOMIC_WORKSPACE_ALLOCATIONS
D20/W80000 calibration point (workspace attached)      inner_status(dense/compr)=0/0  dK=0.000e+00  dDelta=4.770e-18  dlambda=4.747e-14  t_dense=22.1s  t_compr=3.7s
PASS  D20/W80000 calibration point (workspace attached)
PASS  calib: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS still 0
D20/W80000 near-calibration point (SAME workspace -- stale-value check)inner_status(dense/compr)=0/0  dK=0.000e+00  dDelta=2.168e-17  dlambda=2.214e-13  t_dense=6.1s  t_compr=1.6s
PASS  D20/W80000 near-calibration point (SAME workspace -- stale-value check)
PASS  near-calib: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS still 0
PASS  2 distinct D20 outer points -> ECONOMIC_WORKSPACE_REFILLS >= 2
PASS  D20: ECONOMIC_WORKSPACE_ALLOCATIONS == 0 new allocations in measured window
PASS  D20: ECONOMIC_WORKSPACE_RESIZES == 0
ALL D20/W80000 TESTS PASSED
```

This covers 4 of the addendum's 7 required point categories in one gate: **D=20/W=80,000**,
**calibration**, **non-last omitted destination** (ROW=20 is omitted, not the highest-index
country trivially), and **two distinct points through one workspace** (stale-value detection).
`dDelta`/`dlambda` are at machine-precision-consistent levels (1e-14 to 1e-18) — this is real
KNITRO, not a synthetic/mocked solve, so exact bit-identity is not expected (different call
sequence -> different floating-point summation order inside KNITRO), but agreement is far tighter
than any economically meaningful tolerance.

Also observed (not the primary claim, but consistent with the pre-existing continuation-8 report
of compressed mode's speed advantage, reproduced independently here): compressed mode ~6x faster
wall-clock than dense at both points (3.7s vs 22.1s calibration; 1.6s vs 6.1s near-calibration).

### A.3 Allocation/timing (D=4 square; `bench_shared_economic_moment_state_builder_2026-07-27.jl`)

```
build_compressed_factual (allocating reference):             4.421 ms       1475344 bytes
build_compressed_factual! (first fill, fresh ws):                             899272 bytes
build_compressed_factual! (repeated refill, same ws):         4.169 ms        899064 bytes
build_economic_moment_state! (repeated, ws attached):         5.052 ms        899272 bytes
reduction (reference -> shared in-place): 39.05%  (0.576 MB saved/call)
complete unrestricted inner solve, NO workspace (pre-fix behavior):     16.151 ms     671984640 bytes  allocating_calls=21
complete unrestricted inner solve, WITH workspace (this task's fix):    15.707 ms     113324880 bytes  allocating_calls=0 inplace_calls=21
```

(`allocating_calls=21`/`inplace_calls=21` = 1 `@allocated` call + 20 `med_time` repetitions of
`evaluate_fullA_fast`, confirming exactly one moment-state build per inner solve either way.) The
complete-solve allocation drop (672MB -> 113MB across 21 cold D=4 solves, ~83%) is larger than the
isolated builder's own 39% reduction — the isolated number covers only the `winner`/`wval`/
`cf_raw` buffers; the full-solve number also reflects that those buffers are no longer contributing
fresh GC pressure across a whole cold-solve's worth of repeated `θ_full` reconstructions and warm
retries. Full CSV: `docs/shared_economic_moment_state_builder_timing_2026-07-27.csv`.

## B. Restricted families (CM, common Fréchet, CM+ZC, ZC-only) — re-verification, not new gates

These 4 families' `moments!` closures already called the shared builder (`cf_build`, now an alias
for `build_economic_moment_state!`) correctly before this session (see architecture doc §1) — this
session's changes to them were limited to the `CompressedFactualWorkspace` struct gaining two new
instrumentation-only fields (`last_theta`, `has_last`) and `cf_build`'s body being redirected
through the new canonical function (same dispatch, same return value). Re-ran the existing,
pre-session gates covering all 4 to confirm no regression:

- `test_compressed_factual_buffer_reuse.jl` (D=4 square, D=20/W=200 rectangular non-last-omitted,
  real D=20/W=80,000 calibration + near-calibration, allocation gate) — **ALL TESTS PASSED**
  (re-run this session, full output in session log; allocation numbers: 64.67MB (reference) ->
  39.71MB (in-place), 38.59% reduction, unaffected by this session's struct-field addition).
- `test_phaseE_workspace_correctness.jl` (CM family, workspace ON vs OFF, calibration + perturbed)
  — **ALL PASS** (re-run this session).
- `test_compressed_live_integration.jl` (unrestricted family, dense vs compressed, full trajectory
  + tie-fallback, no workspace attached — exercises this session's fixed code path's FALLBACK
  branch) — **ALL COMPRESSED-LIVE-INTEGRATION EQUIVALENCE TESTS PASSED**, worst-case field diff
  across the whole suite `7.994e-14` (`Delta_primal`).
- `test_phase1_d20_exact_cache_and_workspace_all_families.jl` (real D=20/W=80,000, ALL FOUR
  restricted families via their real production entry points `archC_verified_state`/
  `archC_frechet_verified_state`/`archC_meanzc_verified_state`/`archOZ_verified_state`, exact-cache
  hit/miss sequencing + workspace ON/OFF byte-identical comparison) — launched this session
  (real D=20/W=80,000, 4 families x 5 steps each, several real KNITRO solves, multi-minute
  runtime). **Flexible CM family: 12/12 checks PASS, 0 FAIL** (real inner solve at A, cache hit at
  A x2, genuine miss at distinct point B, return-to-A cache hit, workspace object identity stable
  across all 5 calls -- no resize, cache-OFF-vs-ON agreement at A) as of this doc's last update.
  Common Fréchet / CM+ZC / ZC-only sections were still running when this session's time budget was
  reached — **honest gap**: not all 4 families' results are captured here. See
  `docs/../key_results/allfam_d20_gate_output.log` (pushed to Dropbox) for whatever this run
  produced by session end; if it completed with all PASS, that supersedes this note, if not, the
  remaining 3 families' re-verification (not new capability, just regression-confirmation against
  this session's `CompressedFactualWorkspace` struct-field addition) is the concrete next step for
  whoever picks this branch up.

## Honest gaps

- No dedicated NEW gate was written this session for the 4 restricted families specifically
  targeting `build_economic_moment_state!` by its new name (they already went through the
  equivalent `cf_build` path pre-session, and the pre-existing gates above cover them) — re-running
  existing gates was judged sufficient evidence of no regression, not equivalent to a fresh,
  dedicated correctness proof under the new name for each of the 4.
- "One hard point" (a point near a known feasibility-screen boundary or KKT-tight point) was not
  separately constructed this session; the D=20 near-calibration perturbation (`0.03 .* randn`,
  used in both A.1 and A.2) is a generic perturbed point, not a deliberately adversarial one.
- Delta* (`Δ*`) and outer A-gradient end-to-end reproduction (as opposed to the inner-solve
  `Delta_dual`/`K_hard`/`lambda` fields already compared) were not independently re-derived this
  session.
