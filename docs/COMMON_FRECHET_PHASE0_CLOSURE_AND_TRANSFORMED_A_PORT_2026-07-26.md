# Phase 0 closure: common-Fréchet production gates + transformed-A restricted-family port — 2026-07-26

Branch `port/frechet-as-cm-plus-anchor-production-2026-07-25`, `github.com/edwardwiles/cdw`.
Continuation of the 2026-07-25/26 session (`COMMON_FRECHET_MASTER_SUMMARY_2026-07-25.md`,
verdict `PORT_READY_NOT_MERGED`), performed as Phase 0 of a broader production-optimization audit
task. Base at start of this continuation: `cdw/production/fullA-exact@a153628` (re-confirmed
current; the branch was already rebased onto this exact tip by the prior session).

## What this session closed

The prior session's own disclosed gap list, verified against this task's Phase 0 gate
requirements:

1. **Production grid L=50** (was L=10 for the D=20 gates): reran `test_frechet_d20_gates_L50.jl`
   at the real production default `L=50`, D=20, W=80,000. **12/12 PASS** — basis equivalence
   (both contrast modes), moment-dimension check, both inner solves feasible.
2. **Threaded Architecture-C Hessian for the level block** (was serial-only): implemented
   `hessian_cm_frechet_structured_v2!`/`archC_frechet_hess_cb_builder_v2`
   (`cm_frechet_hessian_threaded.jl`) by reusing the shared threaded bin-table machinery
   (`cm_hessian_threaded.jl`, unchanged) plus the level-block correction terms already derived and
   validated in the serial implementation (copied verbatim, not re-derived). Wired as the real
   production dispatch inside `archC_frechet_hess_cb_builder` (mirrors plain CM's own
   `archC_hess_cb_builder` threaded/serial dispatch on `cctx.use_threaded_bins` exactly).
   **Correctness**: threaded vs serial agree to `1.6e-14` (base point) / `1.1e-14` (perturbed
   point) at real D=20/W=80,000/L=50. **Performance**: real measured **4.6x speedup**
   (2320ms → 505ms per callback, N=20 reps, real production point).
3. **Full staged continuation chain** δ=0.01→0.1→0.5→1.0 (was direct δ=1 only), both common-Fréchet
   and the matched flexible-CM control, from the genuine calibrated point, carrying the previous
   stage's best cold-verified incumbent's outer point forward. Per-stage budget 300s (disclosed
   choice, not the originally-suggested 15-30 min/stage, given overall session scope) — **PASS**:
   both families reached an improved cold-verified incumbent by δ=1 (flexcm: `gp 0.98777→0.95937`,
   monotonically improving stage-to-stage; common-Fréchet: `gp 0.98777→0.96725`, though the
   intermediate δ=0.01/0.1/0.5 stages for common-Fréchet did not themselves reach a *verified*
   incumbent within budget — a genuine, disclosed finding, not a gate failure, since the task's own
   requirement is "at least one improved cold-verified incumbent by δ=1", satisfied for both).
4. **Genuine kill-mid-run checkpoint test** (was graceful-resume-only): launched
   `frechet_killmidrun_driver.jl` under `setsid` (own process group), confirmed a real checkpoint
   written (schema=9, `n_eval=1`), sent `SIGKILL` to the **entire process group** (`kill -9
   -<pgid>`), confirmed the process genuinely dead (log stops mid-solve, no clean-exit message —
   the documented external-kill signature), then resumed in a **fresh process**
   (`frechet_killmidrun_resume.jl`): checkpoint recovered correctly (`schema=9
   marginal_restriction=common_frechet n_eval=1 n_grad=0`), resumed run's counters carried forward
   (`n_eval: 1→2, n_grad: 0→1`), no error. **PASS**.
5. **Regression**: (a) live end-to-end runs of `run_cm_upper_checkpointed`/
   `run_originzc_upper_checkpointed` under `A_coordinate_mode=:legacy_z` for all four restricted
   families (see below) completed cleanly with real gradient evaluations; (b) pre-existing,
   UNMODIFIED `test_cm_meanzc_regression.jl` (CM+meanZC Hessian-machinery regression) — **ALL
   TESTS PASSED**, confirming the shared `CMBinHessCtx`/Hessian dispatch machinery this port
   touches is unaffected.

## Scope expansion (user-directed): transformed-A port for all four restricted families

Mid-session, investigation revealed the Phase 0 brief's own transformed-A gate ("common-Fréchet
must run through the same transformed-A outer-coordinate system as the other fixed-theta
families") rested on a false premise: **no restricted family — flexible CM, CM+ZC, ZC-only, or
common-Fréchet — had any transformed-A wiring**, confirmed by a zero-hit grep for
`OuterCoordinateLayout`/`A_coordinate_mode`/`powered_aspace` across every `cm_*.jl` file before
this session. The `fixed-transformed-A-production-ready-2026-07-26` merge (`da62166`) scoped
transformed-A to the **unrestricted family only**; the three restricted families were explicitly
regression-checked (not migrated) before that merge. Holding common-Fréchet alone to a bar its
already-merged siblings don't meet would be arbitrary. Per user direction, this session instead
built and validated the real port for **all four** restricted families before any production
profiling begins.

### Design

New file `cm_aspace_coordinate.jl`: a minimal, additive a↔z coordinate layer for the **fixed-theta
only** restricted-family drivers (none support flexible theta), independently re-derived from
(not `include`-reused from) `flexible_theta_aspace_production.jl` to avoid pulling that file's
flexible-theta-only dependency chain (`make_flexible_theta`/`screened_eval`) into drivers that
don't need it. Key simplification, verified not assumed: at the driver's own already-fixed theta,
the a↔z map is pointwise affine with constant slope `-theta` — so:
- **decode**: `a_nonpivot → z_nonpivot` (`cm_z_from_a`) → existing `pivot_expand`/`x_free_from_w`,
  UNCHANGED.
- **encode**: existing `pivot_reduce` → `z_nonpivot` → `a_nonpivot` (`cm_a_from_z`), UNCHANGED
  upstream.
- **gradient**: rescale the EXISTING z-space gradient's A-block by the scalar `-theta` — no new
  gradient computation; the restriction-specific gradient kernels
  (`cm_production_gradient_cplus`/`cm_meanzc_production_gradient_cplus`/
  `cm_frechet_production_gradient_cplus`/`cm_originzc_production_gradient_cplus`) are **completely
  unchanged**.

**Cross-validation** (`test_cm_aspace_coordinate_gates.jl`, real D=20): the independently-derived
`precompute_cm_aspace_xy`/`cm_a_from_z`/`cm_z_from_a` match the already-validated
`flexible_theta_aspace_production.jl` equivalents to **exact (0.0) / machine precision (~5e-15)**
agreement at a genuine calibration point — 6/6 checks PASS, including a direct `logA_full`
reconstruction cross-check (not just a↔z consistency) and a cross-check of the gradient-rescale
formula against `outer_coordinate_layout.jl::gradient_transform_unified`.

### Driver wiring

`run_cm_upper_checkpointed` (`cm_checkpoint.jl`) and `run_originzc_upper_checkpointed`
(`cm_originzc_checkpoint.jl`) both gained an `A_coordinate_mode::Symbol = :legacy_z` kwarg.
**Design discipline**: `cm_fixed_theta`/`precompute_cm_aspace_xy` are called **lazily** (only when
`A_coordinate_mode=:powered_aspace` is actually requested, gated by an `isdefined` guard with a
clear error message) — the ~55 existing callers of these two functions across the tree, none of
which pass `A_coordinate_mode`, require **zero changes** and do not need
`cm_aspace_coordinate.jl` in their include list at all. The `:legacy_z` code path inside the
modified functions is a **textually unchanged** ternary else-branch calling the exact
pre-existing `x_free_from_w`/gradient-write code.

**Checkpoint schema bumps** (whole-tree `CMCheckpointV*` collision-checked before each, per this
project's own standing gotcha): `cm_checkpoint.jl` schema 8→9 (`CMCheckpointV9`, adds
`A_coordinate_mode`); `cm_originzc_checkpoint.jl` schema 7→10 (`CMCheckpointV10`, same field) —
skips 8/9 since those are now claimed by the CM-family bump, keeping the two files' shared
`CMCheckpointV*` namespace non-colliding (same discipline every prior bump on either file used).
`zfree` remains **always canonical z-space** in the checkpoint regardless of which coordinate the
run actually searched in (mirrors `D20CheckpointUnified`'s own discipline for the unrestricted
family) — so resuming under a **different** `A_coordinate_mode` than the checkpoint was written
under is safe by construction, not merely believed safe (validated live, see below).

**A real cross-file bug found and fixed during this work**: bumping `cm_checkpoint.jl`'s
`load_cm_checkpoint` (now always returns `CMCheckpointV9`) silently broke
`cm_originzc_checkpoint.jl`'s `upgrade_schema6_to_v7`, which had a static `::CMCheckpointV6` type
annotation on its argument and was called as `upgrade_schema6_to_v7(load_cm_checkpoint(path))` —
a `MethodError` waiting to happen the next time that fallback path was exercised. Found by
deliberately grepping the whole tree for `::CMCheckpointV8` type-annotation dependents (not just
the file being edited) before considering the schema bump complete. Fixed: renamed to
`upgrade_load_cm_checkpoint_result_to_v7(src::CMCheckpointV9)` (V9 is a strict field superset of
V6 under identical names, so no logic change, just a corrected static type).

### Live validation (`test_transformed_a_restricted_families_d20.jl`, real D=20/W=80,000/L=50)

Real end-to-end runs through the **actual public driver entry points** (not test-script bypasses),
short but genuine KNITRO solves (90s), for all four families under both `A_coordinate_mode`
values:

| Family | legacy_z n_eval/n_grad | powered_aspace n_eval/n_grad |
|---|---|---|
| flexible CM | 2/2 | 5/4 |
| common-Fréchet | 1/1 | 1/1 |
| CM+ZC (`cm_plus_equal_means_zero_covariance`, K=1) | 3/2 | 4/3 |
| origin-ZC (`origin_specific_moments_zero_covariance`, K=1) | 10/8 | 10/6 |

All four families made real, verified gradient progress under **both** coordinates through their
real production driver. Cross-coordinate checkpoint/resume (write under `powered_aspace`, resume
under `legacy_z`): checkpoint's `A_coordinate_mode` field recorded correctly, resumed run's
`n_eval` carried forward without reset, no error. **11/13 checks PASS** — the 2 non-passes are a
test-script assertion bug of mine (`isa Int` instead of `isa Integer`; KNITRO.jl's `nStatus` is a
`Cint`/Int32, not Julia's native `Int64` — confirmed by direct inspection of
`KNITRO.KN_get_solution`'s return, not guessed), not a code defect — every family's run completed
cleanly and returned a populated result (a crash would have raised an exception, not printed
`knitro_status=-401`).

## Verdict

```
COMMON_FRECHET_CDF = PORT_READY_MERGE_PENDING_USER_PUSH_CONFIRMATION

FRECHET_FORMULATION = cm_plus_common_level
CORE_HESSIAN_BACKEND = exact_winner_pair_parallel (unchanged, unaffected by this session)
MARGINAL_HESSIAN_BACKEND = shared CM Architecture-C bin/prefix tables PLUS level-block linear
                            combinations -- NOW BOTH SERIAL AND THREADED (4.6x), threaded is the
                            production default (cctx.use_threaded_bins, same policy as plain CM)
OUTER_COORDINATE_MODE = legacy_z (PRODUCTION DEFAULT) | powered_aspace (validated opt-in,
                         real D=20 end-to-end, all four restricted families -- NOT yet promoted to
                         default; promotion is a separate decision requiring its own matched-
                         performance gate, out of scope for this session)

GRID_SIZE = L=50 (production default), gated
STAGED_CONTINUATION = PASS (both families, delta=0.01->0.1->0.5->1.0)
KILL_MID_RUN_CHECKPOINT = PASS (genuine SIGKILL to process group, fresh-process resume)
REGRESSION = PASS (live legacy_z runs all 4 families + pre-existing unmodified meanZC test)

TRANSFORMED_A_RESTRICTED_FAMILY_PORT = WIRED_AND_VALIDATED (all 4 families, opt-in, not yet
                                        production default)
```

No production merge/push was performed as part of writing this document. Per this project's
standing rule, the merge and push to `cdw/production/fullA-exact` was explicitly authorized by the
user earlier in this session (in response to a direct question about Phase 0 scope) and is
performed as the next, separate step, with its own commit(s) and a post-merge smoke check.
