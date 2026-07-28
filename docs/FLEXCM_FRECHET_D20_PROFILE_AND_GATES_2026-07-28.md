# flexible_cm / common_frechet real D=20 Hessian sub-block profile + threaded-H_EC correctness gate (2026-07-28)

Branch `agent/d20-profile-flexcm-frechet-2026-07-28`, based on
`optimize/production-structured-CM-ZC-hessian-2026-07-28@410c154`
(`origin/production/fullA-exact@5b4f9da`). Scope: real D=20/Ddest=19/W=100,000 profiling and
correctness gates for **flexible_cm** and **common_frechet** ONLY (cm_meanzc/origin_zc/H_ZZ backend
sweep are a sibling agent's separate worktree, not duplicated here).

## 1. What was built

- `cm_hessian_subblock_profiling.jl` (new): `CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED` (`Ref{Bool}`,
  default `false`), the `@cmhess_prof` timing macro (reuses `instrumentation.jl`'s
  `prof_record!`/`PROF_TIMES`/`prof_summary()` machinery), an opt-in per-Hessian-callback dual-point
  capture buffer (`CM_HESSIAN_CAPTURED_X`), and a live-`pcx` stash (`CM_LIVE_PCX_STASH`) written by
  `run_cm_upper_checkpointed` the instant its `(cctx,obj)` are built. Zero overhead when the flag is
  off (single `Ref` check, no `time_ns()` call) -- self-include guards added to
  `cm_hessian_architectures.jl`/`cm_hessian_threaded.jl`/`cm_frechet_hessian.jl`/`cm_checkpoint.jl`
  so none of this repo's other ~181 callers of those files needed their own include list edited.
- `@cmhess_prof` labels wired into the SHARED `hessian_cm_structured_v2!` (the production
  `use_threaded_bins=true` path both families dispatch through): `ddpsi`, `misc_bookkeeping`,
  `H_EE`, `bintables_prep`, `H_EC_prep`, `H_EC_asm`, `H_CC`, `packing`; into
  `_fill_frechet_level_blocks!` (common_frechet only): `level_table_prep`, `H_EF`, `H_CF`, `H_FF`;
  and into `_prep_dual_index_for_archC!` (shared): `hessw_operator_prep` (the "shared
  `operator_hessian_weights!`/Ψ'' pass" -- split into this label plus `ddpsi` since they live in two
  different functions).
- `profile_flexcm_frechet_d20_2026-07-28.jl` (new): real D=20/W=100,000 driver, adapted from
  `smoke_delta1_flexcm.jl`/`smoke_delta1_frechet.jl`, that calls **only**
  `run_cm_upper_checkpointed` (never a low-level entry point). Sets
  `CROSS_HESSIAN_THREADED_DEFAULT[]`/`CROSS_HESSIAN_WORKERS_DEFAULT[]` before the call, writes the
  sub-block profile CSV from `prof_summary()`-backed data, and (in `threaded` mode) runs an offline
  correctness gate by reaching into `CM_LIVE_PCX_STASH[]` and re-invoking
  `hessian_cm_structured_v2!` directly (a plain Julia call, no second KNITRO invocation) at real
  captured dual-solve points (`CM_HESSIAN_CAPTURED_X`) under `cross_hessian_workers` in
  `{1,4,8,10,20}` vs the serial reference (`cross_hessian_threaded=false`).
- **Bug found and fixed** (not introduced by this task, blocking the gate entirely):
  `winner_pair_cross_hessian.jl`'s `WinnerBinCrossScratch(ncolI,D,L)` constructor passed its last
  two positional arguments in the wrong order relative to the struct's own field declarations
  (`tasks_ec::Vector{Task}` then `EsumEcon::Vector{Float64}`, but the constructor passed
  `zeros(ncolI)` then `Vector{Task}(...)` -- the reverse). Any FRESH build of this scratch struct
  (`cctx.cross_scratch === nothing`) threw `MethodError: Cannot convert an object of type Float64 to
  an object of type Task`, 100% reproducibly. A live production run's own calibration-point Hessian
  callbacks happened not to need a fresh build (reused an already-built instance from an earlier
  successful construction), masking the bug there; this task's own gate hit the fresh-build path
  directly and surfaced it immediately. Fixed with a two-argument swap (see commit `bbda590`) --
  confirmed correct by the gate's own subsequent 20/20 bit-exact PASS for both families.

## 2. KNITRO callback-error investigation (honest account)

Before the fix above, both an unmodified control run (`smoke_delta1_flexcm.jl`, via `git stash`) and
this task's own instrumented script reproducibly hit `KN_RC_CALLBACK_ERR` (nStatus=-500,
`MethodError(convert, (Task, 0.0))` inside KNITRO.jl's C wrapper) at the very first inner solve, at
the exact calibration point, identically across separate process runs and Julia thread counts
(`-t 1`, `-t 8`, `-t 10`, `-t 20` all reached or were consistent with the same failure signature).
**This is the same symptom class the master report (§5) attributes to the direct-low-level-entry
KNITRO trap -- but it reproduced here through the confirmed-working public driver
(`run_cm_upper_checkpointed`) too**, contradicting that report's optimistic "public driver
confirmed working" conclusion (which was based on a different session's run, evidently
host/timing-dependent). After the `WinnerBinCrossScratch` fix, all four real runs
(flexible_cm x {threaded,serial}, common_frechet x {threaded,serial}) completed cleanly with real
outer+inner KNITRO progress -- so this specific instance of the symptom was very likely the
`WinnerBinCrossScratch` bug itself surfacing as a generic KNITRO-callback-exception catch-all
(a Julia exception thrown inside a nested inner-solve invoked from the outer KNITRO callback
degrades to KNITRO.jl's generic `MethodError(convert,(Task,0.0))` handler regardless of the
exception's real cause) -- not a separate, still-open KNITRO/threading race. Not fully proven (the
fix could coincidentally have also perturbed timing enough to dodge a genuinely separate race), but
the most parsimonious explanation given the evidence, and not worth further time given
`CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED`-off/on both now succeed reliably.

## 3. Sub-block timing profile (real D=20/Ddest=19/W=100,000, L=50, calibration + non-calibration + solver-derived points)

All runs: `-t 20`, `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1`, `destination_sample=:exclude_row`,
`cm_gradient_backend=:cplus`, `marginal_restriction` per family, `maxtime_real=90`. Full CSV:
`docs/CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28_flexcm_frechet.csv`; raw per-run CSVs
and solver-status text under `results/profile_flexcm_frechet_2026-07-28/`.

Each run's Hessian callback fired across a REAL KNITRO trajectory (not one synthetic point) --
`flexible_cm`: 76 real Hessian callbacks (both modes), spanning n_eval=4 distinct outer points, from
the calibration outer point through 3 further outer iterates before the wall-clock cap.
`common_frechet`: 100 callbacks (threaded, n_eval=4) / 53 callbacks (serial, n_eval=3). The gate's
"calibration"/"non_calibration"/"solver_derived_late" points below are real captured dual vectors
from this same trajectory (indices 1, ~25%, ~60%, and last of the captured-point buffer).

### flexible_cm (mean callback time; production-candidate = threaded, workers=20)

| block | serial (s) | threaded (s) | share of callback (threaded) |
|---|---:|---:|---:|
| bintables_prep | 0.1705 | 0.1710 | 55% |
| H_EC_prep | 0.1970 | 0.0609 | 20% |
| H_EE | 0.0434 | 0.0464 | 15% |
| H_CC | 0.0153 | 0.0161 | 5% |
| packing | 0.0063 | 0.0070 | 2% |
| H_EC_asm | 0.0065 | 0.0065 | 2% |
| ddpsi | 0.0010 | 0.0010 | <1% |
| misc_bookkeeping | 0.0028 | 0.0026 | <1% |
| hessw_operator_prep | 0.0005 | 0.0005 | <1% |
| **complete_hessian_callback** | **0.4433** | **0.3121** | 100% |

**Dominant block for flexible_cm: `bintables_prep`** (the shared winner/bin/feature-table
construction, ALREADY threaded by the pre-existing `threaded_bins=true` production default,
independent of this task's `cross_hessian_threaded`) -- 55% of the callback once H_EC is also
threaded, up from being roughly co-dominant with H_EC_prep (39% vs 44%) when H_EC is serial. This
matches the master report's qualitative estimate ("crossprep(H_EC)~47%, bintables~36%") closely at
the serial baseline.

**H_EC threaded speedup (serial vs threaded@workers=20): 3.23x** (0.1970s -> 0.0609s). Complete-
callback speedup: **1.42x** (0.4433s -> 0.3121s), consistent with Amdahl's law given H_EC_prep was
~44% of the serial callback.

### common_frechet (mean callback time; production-candidate = threaded, workers=20)

| block | serial (s) | threaded (s) | share of callback (threaded) |
|---|---:|---:|---:|
| bintables_prep | 0.1457 | 0.1484 | 51% |
| H_EC_prep | 0.1770 | 0.0580 | 20% |
| H_EE | 0.0439 | 0.0428 | 15% |
| H_CC | 0.0149 | 0.0135 | 5% |
| H_CF | 0.0061 | 0.0042 | 1% |
| packing | 0.0069 | 0.0066 | 2% |
| H_EC_asm | 0.0063 | 0.0056 | 2% |
| level_table_prep | 0.0032 | 0.0030 | 1% |
| H_FF | 0.0021 | 0.0021 | <1% |
| H_EF | 0.0012 | 0.0012 | <1% |
| ddpsi | 0.0009 | 0.0009 | <1% |
| misc_bookkeeping | 0.0027 | 0.0027 | <1% |
| hessw_operator_prep | 0.0014 | 0.0005 | <1% |
| **complete_hessian_callback** | **0.4125** | **0.2897** | 100% |

**Dominant block for common_frechet: `bintables_prep`** (51%), same as flexible_cm -- the three
Fréchet-only level blocks (H_EF/H_CF/H_FF) together are under 2% of the callback, negligible
relative to H_EE/H_EC/bintables (confirms the harmonization task's own claim that common_frechet's
level-anchor extension is cheap on top of the shared CM machinery).

**H_EC threaded speedup (serial vs threaded@workers=20): 3.05x** (0.1770s -> 0.0580s). Complete-
callback speedup: **1.42x** (0.4125s -> 0.2897s).

**Caveat**: the `threaded-H_EC speedup at workers=20 vs workers=1` (both threaded, different worker
counts) requested by the parent task's report-back was NOT separately timed -- only
serial-vs-threaded@workers=20 timing, and workers-in-{1,4,8,10,20}-vs-serial CORRECTNESS (§4,
timing-blind) were measured, given the session's remaining time budget after the KNITRO/bug
investigation in §2. The origin_zc sibling profile (already in this repo,
`cross_hessian_subblock_profile_t{1,20}_origin_zc_2026-07-28.csv`) found a comparable serial-vs-
threaded ratio (2.7-3.4x) for its own H_EZ block, so a workers=1-threaded point is expected to sit
close to the serial number reported here, but this was not directly confirmed for H_EC.

## 4. D=20 correctness gate: threaded H_EC vs serial, workers in {1,4,8,10,20}

For each family: 4 real points (calibration + 2 non-calibration + 1 solver-derived-late, all real
captured dual vectors from the live trajectory above) x 5 worker counts = **20 checks per family,
40 total**. Method: direct re-invocation of `hessian_cm_structured_v2!` on the SAME live
`(cctx,obj)` state at each point, comparing the packed upper-triangular Hessian under
`cross_hessian_threaded=true,cross_hessian_workers=wk` against `cross_hessian_threaded=false`
(serial reference) -- pure Julia function calls, no second KNITRO invocation anywhere in this
section (per the master report's own recommended workaround).

**Result: 40/40 PASS, maxdiff = 0.0 EXACTLY (bit-exact, not merely close) in every single check**,
for both flexible_cm and common_frechet, at every worker count and every point type. Full data:
`docs/key_results/correctness_gate_flexible_cm_2026-07-28.csv`,
`docs/key_results/correctness_gate_common_frechet_2026-07-28.csv` (copied from
`results/profile_flexcm_frechet_2026-07-28/`).

## 5. Solver-behavior invariance (KNITRO status/iterations/kappa, threaded vs serial)

| family | mode | knitro_status | n_eval | n_grad | kappa |
|---|---|---|---|---|---|
| flexible_cm | serial | -401 (TIME_LIMIT_FEAS) | 4 | 3 | 0.05060338369820083 |
| flexible_cm | threaded | -401 (TIME_LIMIT_FEAS) | 4 | 3 | 0.05060338369820083 |
| common_frechet | serial | -401 (TIME_LIMIT_FEAS) | 3 | 3 | 0.03037888823166357 |
| common_frechet | threaded | -401 (TIME_LIMIT_FEAS) | 4 | 4 | 0.0435273382990532 |

`flexible_cm`: **identical** status/n_eval/n_grad/kappa to full precision between serial and
threaded -- clean confirmation. `common_frechet`: same terminal status class (both a feasible
time-limit stop, not an error) but threaded reached **one more outer eval** (n_eval=4 vs 3) within
the identical 90s wall-clock budget, hence a different final `kappa` -- **this is the expected
consequence of comparing two different-speed runs under a WALL-CLOCK time limit, not a correctness
difference**: threaded's ~1.4x faster callback lets KNITRO's outer loop complete more work in the
same real-time budget, so the two runs' outer trajectories diverge in HOW FAR they get, not in what
each computed VALUE is (already proven bit-exact and separately, at fixed points, in §4). A
same-iteration-count comparison (removing the wall-clock confound) was not run given remaining time
budget -- flagged as the one open item if a stricter apples-to-apples solver-trajectory comparison is
wanted later.

## 6. Verdict

```
FAMILIES PROFILED (real D=20/W=100,000): flexible_cm, common_frechet -- BOTH SUCCEEDED
DOMINANT_BLOCK:
    flexible_cm:     bintables_prep (55% of threaded callback), H_EC_prep second (20%)
    common_frechet:  bintables_prep (51% of threaded callback), H_EC_prep second (20%)
H_EC_THREADED_SPEEDUP (serial -> threaded@workers=20):
    flexible_cm:     3.23x on H_EC_prep alone; 1.42x complete-callback
    common_frechet:  3.05x on H_EC_prep alone; 1.42x complete-callback
CORRECTNESS_GATE: 40/40 PASS, maxdiff=0.0 bit-exact, workers in {1,4,8,10,20}, both families,
    calibration + 2 non-calibration + 1 solver-derived real D=20 point each
SOLVER_BEHAVIOR_INVARIANCE: flexible_cm identical (n_eval/n_grad/kappa exact match); common_frechet
    same status class, differs in outer progress made within a shared wall-clock budget (expected,
    not a correctness issue -- see §5)
BUG_FOUND_AND_FIXED: WinnerBinCrossScratch constructor argument-order bug (winner_pair_cross_hessian.jl),
    pre-existing (structured-cross-hessian task), blocked ANY fresh build of that scratch struct --
    fixed, commit bbda590
WARMUP_THEN_DIRECT_CALL_TRICK: NOT used -- all profiling/capture/gate state was reached exclusively
    via the instrumentation-inside-the-driver approach (CM_LIVE_PCX_STASH/CM_HESSIAN_CAPTURED_X),
    never a second/direct low-level KNITRO entry point call
PRODUCTION_MERGE: profiling/gate work is diagnostic-only (no production default changed by this
    task); the H_EC threading itself (cross_hessian_threaded/cross_hessian_workers, from the parent
    structured-cross-hessian task) is now ALSO D=20-confirmed correct+beneficial for flexible_cm and
    common_frechet specifically (previously only origin_zc had real D=20 confirmation) --
    recommend flipping CROSS_HESSIAN_THREADED_DEFAULT[] to true for these two families pending the
    user's own review of this report, still port_ready_not_merged per this task's own no-push rule
```

## 7. Deliverables index

- `docs/CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28_flexcm_frechet.csv`
- `docs/key_results/correctness_gate_flexible_cm_2026-07-28.csv`
- `docs/key_results/correctness_gate_common_frechet_2026-07-28.csv`
- `docs/key_results/subblock_profile_{flexible_cm,common_frechet}_{serial,threaded}_2026-07-28.csv`
- Source: `cm_hessian_subblock_profiling.jl` (new), `profile_flexcm_frechet_d20_2026-07-28.jl` (new),
  modifications to `cm_hessian_architectures.jl`, `cm_hessian_threaded.jl`, `cm_frechet_hessian.jl`,
  `cm_checkpoint.jl`, `winner_pair_cross_hessian.jl` (bugfix).
