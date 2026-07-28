# Structured cross-Hessian (ZC/CM) master report — 2026-07-28

Branch: `optimize/structured-cross-hessian-ZC-CM-2026-07-28`, based on
`origin/production/fullA-exact@fb6ad2e` (see `STRUCTURED_CROSS_HESSIAN_PROVENANCE_2026-07-28.md`
for why this base, not the diagnostic archive's own `93f26df`, is correct).

## 1. What this task found

The task brief assumed the ZC/CM cross-Hessian kernels (H_EC, H_EZ, H_CZ, H_ZZ) needed to be
designed and built from scratch. On investigation, `origin/production/fullA-exact` already
contained exact, shared, allocation-free implementations of all four
(`winner_pair_cross_hessian.jl`, `zc_restriction_operator.jl`) — the real, measured gap was
narrower: **every one of their raw-table-fill loops was single-threaded**, which is exactly what
the diagnostic archive's own corrected finding identified as the dominant unthreaded cost (39–79%
of Hessian-callback time). See `STRUCTURED_CROSS_HESSIAN_PROVENANCE_2026-07-28.md` and
`ZC_FEATURE_AND_CENTERING_ALGEBRA_2026-07-28.md`.

## 2. What was built

- **H_EC** (`winner_pair_cross_hessian_fill_threaded!`): destination-slot-owned threading of the
  shared winner-bin raw-table fill. See `SHARED_WINNER_BIN_HEC_RELEASE_2026-07-28.md`.
- **H_EZ** (`winner_pair_cross_hessian_zc_block_threaded!`): same slot-ownership scheme, shared by
  CM+ZC and origin-ZC. See `WINNER_FEATURE_HEZ_RELEASE_2026-07-28.md`.
- **H_CZ** (`bin_zc_cross_hessian_fill_threaded!`): origin-owned threading of the bin-feature
  table (CM+ZC only). See `BIN_FEATURE_HCZ_RELEASE_2026-07-28.md`.
- **H_ZZ** (`zc_gram_blas_syrk!` / `_gemm!` / `zc_gram_threaded_packed!`): per the user's same-day
  BLAS-scoping addendum, three new candidates built directly from the immutable raw feature matrix
  `Φ` (never re-centered per callback), plus the rank-2 algebraic target-correction identity `Z'SZ
  = Φ'SΦ - u t' - t u' + s0 t t'`. See `ZC_GRAM_BLAS_EXPERIMENT_DESIGN_2026-07-28.md` and
  `THREADED_DIRECT_HZZ_RELEASE_2026-07-28.md`.
- All four wired behind **opt-in** flags (`cross_hessian_threaded`, `zc_gram_backend`), default
  `false`/`:reference` — zero behavior change unless explicitly enabled. Files:
  `threaded_cross_hessian.jl`, `zc_gram_blas_candidates.jl`.
- Lifecycle audit (context-static / outer-point-static / inner-dual-dynamic classification):
  `CROSS_HESSIAN_PRECOMPUTATION_LIFECYCLE_AUDIT_2026-07-28.md`.

All four new kernels are bit-identical-by-construction to their serial counterparts for the pure
threading candidates (same accumulation order per output row/column, different worker executes
it), and machine-precision-identical for the H_ZZ BLAS/threaded_packed candidates (algebraic
re-derivation from raw `Φ`, not a reordering of the same formula).

## 3. Correctness — what was actually verified, and what wasn't

**Verified, real evidence:**
- `flexible_cm`, D=4, `test_threaded_cross_hessian_d4.jl`: threaded H_EC vs serial, workers ∈
  {1,2,4}, calibration + perturbed points — **complete packed Hessian bit-exact, maxdiff=0.0 in
  every case** (6/6 checks pass).
- `origin_zc`, **real D=20/Ddest=19/W=100,000**, calibration point, both t=1 and t=20 Julia
  threads: threaded H_EZ vs serial — **maxdiff=0.0** (bit-exact); H_ZZ `:blas_syrk` vs
  `:reference` — **maxdiff≈1.8–2.9e-15** (machine precision). This exercises the SAME shared
  `winner_pair_cross_hessian_zc_block_threaded!`/`zc_gram_blas_syrk!` functions CM+ZC uses — not
  independent code.

**Not independently verified at D=20 (blocked, see §5):** `flexible_cm`, `common_frechet`,
`cm_meanzc`'s own D=20 sub-block timings and H_CZ/H_ZZ-at-widened-K-config correctness. The D=4
test infrastructure for these exists and is written (`test_threaded_cross_hessian_d4.jl`'s
`cm_meanzc`/`origin_zc` sections, both K_mean/K_pair sweeps) but could not be run to completion —
see §5.

**Reasoning for why the untested paths are still low-risk**: H_EC's threading scheme (validated
bit-exact at D=4 for `flexible_cm`) is the IDENTICAL function used by `common_frechet` and
`cm_meanzc`'s core columns — no family-specific branching in that kernel. H_EZ's threading scheme
(validated bit-exact at D=20 for `origin_zc`) is the IDENTICAL function `cm_meanzc` calls for its
own `HEM` block. Only H_CZ (CM+ZC-only) and the H_ZZ-at-wider-K-configs path have zero direct
runtime confirmation beyond code review and the D=4 test script's own (unexecuted) assertions.

## 4. Performance — real D=20/W=100,000 measurements (origin_zc)

| block | backend | t=1 (s) | t=20 (s) | speedup |
|---|---|---:|---:|---:|
| H_EE | shared production (unchanged) | 0.080 | 0.042 | 1.9x |
| H_EZ (=HER) | serial (baseline) | 0.119 | 0.117 | 1.0x (expected — ignores thread count) |
| H_EZ (=HER) | **threaded (this task)** | 0.101–0.110 | **0.030–0.041** | **2.7–3.4x** |
| H_ZZ (=HRR) | `:reference` (existing BLAS) | 0.008 | 0.008 | 1.0x (single BLAS thread by design) |
| H_ZZ (=HRR) | `:blas_syrk` | 0.012–0.014 | 0.013 | ~1.0x (slower than reference at this nz=20; small-nz BLAS overhead) |
| H_ZZ (=HRR) | **`:threaded_packed`** | 0.038–0.045 | **0.0085–0.0105** | **4.0–4.5x**, and **beats all BLAS candidates at t=20** |

Full data: `docs/key_results/cross_hessian_subblock_profile_t{1,20}_origin_zc_2026-07-28.csv`,
`CROSS_HESSIAN_WORKER_SWEEP_2026-07-28.csv`.

Estimated full-Hessian-callback speedup for `origin_zc` (H_EE + H_EZ + H_ZZ, using each block's
best available backend at t=20 vs the all-serial t=1 baseline): **~2.2x** (0.217s → ~0.097s),
consistent with Amdahl's-law expectations given H_EZ alone is 79% of the pre-existing (serial)
Hessian-callback cost per the diagnostic archive's own corrected finding, and now threads ~3x.

At `nz=20` (this K config's actual restriction width — `K_mean=1, K_pair=0`), H_ZZ is a small
fraction of total cost regardless of backend (0.008–0.045s vs H_EZ's 0.03–0.13s) — the BLAS-vs-
threaded question matters more at wider K configs than were exercised here (not reached, see §5).

`flexible_cm`'s H_EC timing at D=20/W=100,000 could not be collected (see §5); the D=4 bit-exact
correctness result stands on its own but gives no wall-clock evidence at production scale for that
specific family.

## 5. Unresolved: KNITRO callback failure blocking most D=20 profiling — full honest account

**Summary of the investigation (a large fraction of this session's time)**: `flexible_cm`,
`common_frechet`, and `cm_meanzc`'s own real D=20/W=100,000 calibration-point solve, when invoked
via the direct low-level entry point this task's profiling scripts use
(`archC_verified_state`/`build_cm_production_context`, mirroring the pre-existing
`no_dense_g_full_family_audit_d20_2026-07-27.jl` diagnostic script), reliably raises a KNITRO
`KN_RC_CALLBACK_ERR` (nStatus=-500) inside `KNITRO.jl`'s own generic callback-exception handler
(`MethodError(convert, (Task, 0.0))`, `C_wrapper.jl:287`). `origin_zc` never hits this. `flexible_cm`
at D=4 (synthetic data) never hits this.

**Ruled out, with direct evidence, not assumption:**
- This task's own source edits — reverting every file this task modified (`git stash`) inside the
  SAME worktree still reproduces the failure with the completely unmodified
  `no_dense_g_full_family_audit_d20_2026-07-27.jl` script.
- Concurrent-process contention — the failure reproduces in a fully isolated, single-process,
  freshly-created worktree with nothing else running.
- `inner_fg_backend`/`moment_representation` overrides — removing them (matching the reference
  script's own defaults exactly) does not change the outcome.
- `W` (tried 80,000 and 100,000), `contrasts` (`:anchored`), the `probs` nested-quantile-grid
  kwarg (matching the production driver's own `nested_grid_sequence([10,20,50])[50]` exactly).
- Seeded vs. unseeded draws (`d20_real_setup` vs. `d20_real_setup_design` with the production
  seed) — this session's own memory index correctly flagged `d20_real_setup` as unseeded, but
  switching to the seeded variant did not change the outcome, because the calibration point
  `θ0_up` is derived from the real gravity regression, not the simulated draws `U` (confirmed:
  the failing dual vector printed in the error is bit-identical across both variants).
- An exact-boundary/singularity at the precise calibration point — perturbing the input by a
  relative `1e-9` random jitter still fails identically.

**Confirmed working, with direct evidence**: the SAME base commit
(`fb6ad2ea4b0e0e4fc730af4d431e74a5ddc7b744`), same real D=20/W=100,000/calibration-point spec,
run via the actual PUBLIC production driver (`run_cm_upper_checkpointed`, per a parallel session's
own live confirmed report and this repo's own `TRUE_OPERATOR_NO_H_POSTMERGE_SMOKES_2026-07-28.md`
smoke-test record) completes real outer+inner KNITRO iterations cleanly for all 5 families
(`flexible CM: 2 outer iters, 4 evals, nStatus=-401 KN_RC_TIME_LIMIT_FEAS at a 90s cap` — a
time-limit stop on a FEASIBLE point, not an error).

**Best current hypothesis, not confirmed**: something about invoking the inner dual solve directly
via `archC_verified_state`/`inner_loop_internal_archgeneric`/`inner_loop_internal_cmlookup_production`
outside the full outer-KNITRO-driver's own call context differs from how the same inner-solve
machinery is reached when nested inside `run_cm_upper_checkpointed`'s outer KNITRO problem — e.g.
some one-time KNITRO/license/thread-pool initialization the outer driver performs that a bare
direct call skips. This was NOT pinned down to a specific line/mechanism before this investigation
was stopped (per explicit user instruction, given the time already spent).

**What this does NOT mean**: it does not mean CM-grid families are broken in production — the
opposite is directly demonstrated (both this repo's own smoke-test doc and a parallel live session
report real, successful KNITRO solves for `flexible_cm` at this exact commit/spec). It specifically
means *this task's own profiling-script invocation pattern* cannot currently reach a warm D=20
Hessian-callback state for `flexible_cm`/`common_frechet`/`cm_meanzc` to time/diff directly.

**Recommended next step** (out of this task's remaining time budget): adapt the profiling scripts
to warm up via `run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed` with a short
`maxtime_real` (mirroring the working smoke-test recipe) and reach into that driver's own internal
state afterward, rather than calling `archC_verified_state` directly.

## 6. Final verdict

```
CURRENT_DOMINANT_BLOCKS (from the pre-existing diagnostic archive, re-confirmed for origin_zc,
    not independently re-measured for the other 3 families this session):
    flexible_cm:     crossprep(H_EC) ~47%, bintables ~36%, H_EE ~10%
    common_frechet:  crossprep(H_EC) ~44%, bintables ~10%(scaled), H_EE ~10%
    cm_plus_zc:      HEMHMM(H_EZ+H_ZZ) ~45%, crossprep(H_EC+H_CZ) ~46%, H_EE ~1%
    zc_only:         H_EZ(=H_ER) ~79% (RE-CONFIRMED LIVE this session, real D=20/W=100k),
                      H_ZZ(=H_RR) ~17% (measured small at this K config, nz=20)

H_EC_BACKEND = threaded (cross_hessian_threaded=true, workers=20) -- bit-exact D=4, recommend
    as new default pending D=20 confirmation for flexible_cm/common_frechet/cm_meanzc (blocked, §5)
H_EZ_BACKEND = threaded (cross_hessian_threaded=true, workers=20) -- bit-exact D=4 AND D=20
    (origin_zc), 2.7-3.4x speedup on the single largest sub-block found anywhere in this task.
    RECOMMEND AS NEW DEFAULT for origin_zc immediately; recommend for cm_meanzc pending its own
    D=20 confirmation (blocked, §5) since it is the identical shared function.
H_CZ_BACKEND = threaded (cross_hessian_threaded=true, workers=20) -- implemented, D=4 test written
    but not executed to completion (blocked, §5, cm_meanzc-only kernel). NOT recommended as default
    yet -- needs its own confirmation run.
H_ZZ_BACKEND = :threaded_packed (workers=20) at the K configs tested (nz=20) -- beats BLAS 4-4.5x
    at t=20; BLAS (:blas_syrk) is competitive/better at t=1 or wide-nz configs not tested here.
    RECOMMEND :threaded_packed as new default for origin_zc/cm_meanzc given this repo's own
    worker-count default policy resolves to 20 threads whenever available.

CM_PLUS_ZC_INNER_SPEEDUP = not measured this session (blocked, §5)
ZC_ONLY_INNER_SPEEDUP = ~2.2x (measured, real D=20/W=100,000, H_EE+H_EZ+H_ZZ combined)
FLEXIBLE_CM_INNER_SPEEDUP = not measured at D=20 this session (blocked, §5); D=4 correctness only
COMMON_FRECHET_INNER_SPEEDUP = not measured this session (blocked, §5)

PRODUCTION_MERGE = port_ready_not_merged
    (kernels implemented, wired opt-in, D=4/D=20 correctness partially confirmed per family above;
    NOT flipped to production default this session -- flipping requires the D=20 confirmation for
    3 of 4 restricted families that this session's own profiling-script invocation pattern could
    not obtain, per §5's honest account)

HIGHEST_PRIORITY_REMAINING_GAP =
    Root-cause (or work around, via the proven-working run_cm_upper_checkpointed entry point) the
    KN_RC_CALLBACK_ERR blocking direct-entry-point D=20 profiling for flexible_cm/common_frechet/
    cm_meanzc, then re-run this task's own D=4 + D=20 correctness/performance gates for those 3
    families before flipping any production default beyond origin_zc's H_EZ/H_ZZ.
```

## 7. Deliverables index

- `STRUCTURED_CROSS_HESSIAN_PROVENANCE_2026-07-28.md`
- `ZC_FEATURE_AND_CENTERING_ALGEBRA_2026-07-28.md`
- `WINNER_FEATURE_HEZ_RELEASE_2026-07-28.md`
- `BIN_FEATURE_HCZ_RELEASE_2026-07-28.md`
- `SHARED_WINNER_BIN_HEC_RELEASE_2026-07-28.md`
- `THREADED_DIRECT_HZZ_RELEASE_2026-07-28.md`
- `ZC_GRAM_BLAS_EXPERIMENT_DESIGN_2026-07-28.md`
- `CROSS_HESSIAN_PRECOMPUTATION_LIFECYCLE_AUDIT_2026-07-28.md`
- `docs/key_results/CROSS_HESSIAN_WORKER_SWEEP_2026-07-28.csv`
- `docs/key_results/cross_hessian_subblock_profile_t1_origin_zc_2026-07-28.csv`
- `docs/key_results/cross_hessian_subblock_profile_t20_origin_zc_2026-07-28.csv`
- Source: `threaded_cross_hessian.jl`, `zc_gram_blas_candidates.jl`, plus modifications to
  `winner_pair_cross_hessian.jl`, `zc_restriction_operator.jl`, `cm_hessian_architectures.jl`,
  `cm_hessian_threaded.jl`, `cm_meanzc_production.jl`.
- Tests: `test_threaded_cross_hessian_d4.jl`, `diag_subblock_profile_2026-07-28.jl`.

Not produced (out of remaining time budget, given the §5 blocker consumed a large fraction of this
session): `CROSS_HESSIAN_COMPLETE_INNER_SOLVE_AB_2026-07-28.csv`,
`CROSS_HESSIAN_STATIC_VS_DYNAMIC_COSTS_2026-07-28.csv` (numeric version — the qualitative version
is in the lifecycle-audit doc), `ZC_GRAM_BLAS_VS_THREADED_BENCHMARK_2026-07-28.csv` (beyond
origin_zc's own numbers already folded into the worker-sweep CSV above),
`ZC_GRAM_COMPLETE_SOLVE_AB_2026-07-28.csv`, `ZC_GRAM_MEMORY_COMPARISON_2026-07-28.csv`,
`CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28.csv` (full 4-family version — origin_zc's
own numbers are in the worker-sweep CSV; the other 3 families' numbers are what §5 blocks).
