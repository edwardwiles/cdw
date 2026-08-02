# ZC Hessian backend production integration — final report (2026-08-01)

Controlled, surgical integration of the validated ZC Hessian backend candidates into
`production/fullA-exact`, per explicit task instructions — NOT a verbatim merge of the exploratory
optimization branch. Branch: `integrate/zc-hessian-backends-production-2026-08-01`, based directly
on `origin/production/fullA-exact@377f48e` (the actual current production tip; merge-base with the
prior exploratory work is `58bae94`, one commit back). Worktree:
`integrate-zc-hessian-backends-2026-08-01`.

## Integrated defaults

```text
CM+ZC:      H_ZZ=blas_syrk   H_CZ=draw_chunk_reordered   H_EZ=drawmajor_v2
Origin-ZC:  H_ZZ=blas_syrk   H_EZ=drawmajor_v2           (H_CZ not applicable)
Julia threads=10 (documented + warned-on-violation in campaign_cm_family_runner.jl)
BLAS threads=8   (auto-selected for cm_meanzc/origin_zc specifically, siblings untouched)
```

Reference backends (`:reference`, `:draw_chunk_thread_local`, `:winner_bin`) remain selectable,
unremoved. `hzz_chunked_syrk_candidate_2026-08-01.jl` is tracked but NOT wired into any dispatch —
diagnostic-only per instruction, not separately integrated/gated as a selectable backend.

## What changed (surgical, not verbatim)

- **New tracked files**: `hcz_reordered_candidate_2026-08-01.jl`, `hez_drawmajor_candidate_2026-08-01.jl`,
  `hez_drawmajor_v2_candidate_2026-08-01.jl`, `hzz_chunked_syrk_candidate_2026-08-01.jl`.
- **`cm_hessian_architectures.jl`**: self-guarded includes for all four candidate files (mirroring
  the file's own existing idiom for `zc_gram_blas_candidates.jl`) + new `OriginZCCoreHessCtx.zc_ez_backend`/
  `zc_drawmajor` fields/dispatch + a previously-missing `:drawmajor_v2` branch for cm_meanzc's own
  H_ER dispatch (see bugs below).
- **`hcz_drawchunk_candidate_2026-07-29.jl`**: default flipped to `:draw_chunk_reordered` + a
  previously-missing dispatch branch for it (see bugs below).
- **`zc_gram_blas_candidates.jl`**: default flipped to `:blas_syrk`, new `ZC_GRAM_BLAS_THREADS_DEFAULT[]=8`.
- **`cm_checkpoint.jl`**: `run_cm_upper_checkpointed` auto-selects `ZC_GRAM_BLAS_THREADS_DEFAULT[]`
  for cm_meanzc specifically (via the existing `family_tag` computation), leaving flexible_cm/
  common_frechet's own zero-behavior-change default (`nothing`) untouched — this function is SHARED
  across three families, so the BLAS-thread default could not simply be changed unconditionally.
- **`cm_originzc_checkpoint.jl`**: `run_originzc_upper_checkpointed`'s own `blas_threads` kwarg
  default changed directly to `ZC_GRAM_BLAS_THREADS_DEFAULT[]` — safe, this function is origin-ZC-only.
- **`production_backend_manifest.jl`**: both `resolve_flexible_cm_manifest` (cm_meanzc branch) and
  `resolve_origin_zc_manifest` now record `zc_gram_backend`/`hcz_prep_backend`/`zc_ez_backend`
  (read directly off the live `cctx`/`octx`), alongside the pre-existing `julia_threads`/`blas_threads`.
- **`campaign_cm_family_runner.jl`**: usage doc updated to recommend `-t 10`/`OPENBLAS_NUM_THREADS=8`
  for cm_meanzc/origin_zc; a runtime warning fires if either family runs with `Threads.nthreads()<10`.

## Three real bugs found and fixed during this integration's own gate process

This is exactly why the task specified gates before merge rather than a verbatim merge — each of
these would have shipped a real defect (two crashes, one false-positive test) had they not been
caught:

1. **`hcz_prep_dispatch!` never had a `:draw_chunk_reordered` branch.** Only `:origin_owned`/
   `:draw_chunk_thread_local` were recognized; setting the new default threw
   `KN_RC_CALLBACK_ERR` (nStatus=-500) inside the real KNITRO Hessian callback the FIRST time Gate 3
   (a real-driver run) hit it. Fixed by adding the missing dispatch arm.
2. **cm_meanzc's own H_ER dispatch never had a `:drawmajor_v2` branch** (only `:drawmajor`, v1) —
   this bug PRE-DATES this integration; it was already present in the exploratory closeout branch's
   own `cm_hessian_architectures.jl`. Any cctx with `zc_ez_backend=:drawmajor_v2` silently fell
   through to the `:cross_hessian_threaded` branch (byte-identical to `:winner_bin`) for cm_meanzc
   specifically — meaning the exploratory session's own "validated" cm_meanzc speedup numbers never
   actually exercised drawmajor_v2 for H_EZ at all (origin-ZC's own octx dispatch was correct
   throughout; only cm_meanzc's cctx dispatch was missing the branch). Fixed by mirroring origin-ZC's
   existing branch. See memory `zc-hessian-backend-closeout-2026-08-01` (corrected) and
   `feedback-solve-timing-jit-thread-warmstart-pitfalls-2026-08-01`.
3. **This integration's own first-draft Gate 2 never primed a real `CompressedFactual`** (never
   called `obj.moments!` at a real economic point before the Hessian comparison), so both arms
   silently fell back to the dense-reference path — a trivial dense-vs-dense pass that never
   exercised the bin/threaded backend dispatch it was meant to test. Fixed by priming properly and
   adding an explicit counter-based check (`NO_DENSE_G_COUNTERS[].winner_cross_hessian_calls`)
   confirming the real path is genuinely reached.

Bug 1 was caught immediately (a hard crash). Bugs 2 and 3 were SILENT — no error, no crash, just a
quietly wrong/trivial test result — and were only caught because Gate 3's crash prompted surfacing
the real underlying exception (`DIAG_LAST_HESS_EXCEPTION` temporary instrumentation) rather than
accepting the KNITRO-level error code at face value, which then led to re-examining Gate 2's own
methodology once bug 2 raised the question of whether other silent gaps existed.

## Gate results (from a clean checkout, all six required before merge)

1. **Tracked + loaded via canonical production stack**: PASS. A test script using EXACTLY
   `campaign_cm_family_runner.jl`'s own include list (no manual candidate includes) resolves every
   candidate symbol and shows both `build_cm_meanzc_production_context`/`build_originzc_production_context`
   defaulting to `blas_syrk`/`draw_chunk_reordered`/`drawmajor_v2` with zero explicit kwargs.
2. **Complete packed-Hessian correctness** (`GATE2_PRODUCTION_DEFAULTS_COMPLETE_HESSIAN_2026-08-01.csv`):
   PASS, 3/3 trials each family, genuinely exercising the winner-pair/bin-structured dispatch
   (confirmed via counter, not just no-error). cm_meanzc: `max|Δ|` 4.7e-9 to 2.0e-8 against scale
   ~7.2e6-8.5e6. origin_zc: `max|Δ|` 6.5e-9 to 2.0e-8 against scale ~8.4e6-1.9e7. Both comfortably
   tolerance-level (relative error ~1e-15).
3. **Corrected compile-free genuine-cold inner A/B, W=100,000 and W=500,000**
   (`ZC_COMPILE_FREE_BACKEND_AB_2026-08-01.csv`): PASS, all 18 rows (5 cm_meanzc arms + 4 origin-ZC
   arms × 2 W values), `nStatus=0` throughout, matching `Delta_dual` to 6+ significant figures and
   near-identical KNITRO iteration counts within each (family, W). `all_optimized` fastest at every
   point:

   | Family | W | reference | all_optimized | speedup |
   |---|---:|---:|---:|---:|
   | cm_meanzc | 100,000 | 24.01s | 15.21s | 36.7% faster |
   | cm_meanzc | 500,000 | 106.34s | 77.06s | 27.5% faster |
   | origin_zc | 100,000 | 19.05s | 11.14s | 41.6% faster |
   | origin_zc | 500,000 | 87.30s | 61.34s | 29.7% faster |

4. **Both public entry points select new backends through the live context, no explicit kwargs**:
   PASS (folded into Gate 1's own script — `build_cm_meanzc_production_context`/
   `build_originzc_production_context` called with zero backend kwargs both resolve to the new
   defaults).
5. **Mixed-family ten-by-ten resource smoke** (`GATE5_MIXED_TEN_BY_TEN_RESOURCE_GATE_2026-08-01.md`):
   PASS. 5 cm_meanzc + 5 origin_zc concurrent fresh processes × 10 Julia threads each (100 of this
   host's cores, disjoint `taskset` affinity), all `all_optimized`, W=100,000: all 10 completed in
   142s wall, all `nStatus=0`, no throughput collapse, no swap pressure, 2.8TiB memory still free.
6. **Manifest recording**: PASS (folded into Gate 1's own script's printed
   `zc_gram_backend`/`hcz_prep_backend`/`zc_ez_backend` fields off the live `cctx`/`octx`, and the
   `production_backend_manifest.jl` resolver changes above).

## Production merge verdict

```text
PRODUCTION_DEFAULTS =
    CM_plus_ZC:
        H_ZZ: blas_syrk (full-workspace, W500K default per instruction; chunked_syrk available but
              not wired/gated as a selectable backend this integration)
        H_CZ: draw_chunk_reordered
        H_EZ: drawmajor_v2
    origin_ZC:
        H_ZZ: blas_syrk
        H_EZ: drawmajor_v2
        (H_CZ: not applicable)

GATE_1_TRACKED_AND_LOADED = pass
GATE_2_COMPLETE_HESSIAN_CORRECTNESS = pass
GATE_3_COMPILE_FREE_COLD_AB_W100K_W500K = pass
GATE_4_ENTRY_POINTS_SELECT_DEFAULTS = pass
GATE_5_MIXED_TEN_BY_TEN_RESOURCE = pass
GATE_6_MANIFEST_RECORDS_BACKENDS_AND_THREADS = pass

BUGS_FOUND_AND_FIXED_DURING_INTEGRATION = 3 (2 dispatch gaps, 1 gate methodology flaw)

PRODUCTION_MERGE = ready

CAMPAIGN_LAUNCHED = false
```
