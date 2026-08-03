# REDUCED inner readiness — pre-fix baseline — 2026-08-03

Recorded at the canonical REDUCED tip (`prototype/profiled-destination-scales@7ec5c6c`), before any
edit on `fix/profiled-inner-readiness-2026-08-03`. Sourced from `reduced_postcleanup_audit_2026-08-03.zip`
(pulled from `dropbox:Gravity robustness/Analysis/Server Output/`, per this repo's standing rule that
task-referenced zips live on Dropbox, not locally) plus this session's own independent re-derivation via
direct code reading (not grep alone) in `/bbkinghome/edav/cdw_worktrees/profiled-inner-readiness-2026-08-03`.
Per task §3, this baseline is evidence-gathering only — no benchmarking of broken code.

## 1. Inner FG callback method table — `_callbackEvalFG_inner_profiled!`

Two top-level definitions with the historically-identical, fully-untyped 5-arg signature
`(kc, cb, evalRequest, evalResult, userParams)`:

| File | Line | `userParams` meant to be | Body |
|---|---|---|---|
| `oracle_fast.jl` | 102 | one of the legacy callable `PsiObjectiveBundle{Explicit,Implicit,Delta}` types (`cc_algo/PsiObjectiveBundle.jl`) — calls `obj(x, evalResult.objGrad)` as a functor | unclamped |
| `profiled_operator_bundle_2026-08-01.jl` | 74 | `ProfiledCBState` (this file's own struct; wraps `obj::OperatorPsiBundle`) | unclamped, **no `obj.lower_limit` reference anywhere** |

Because both were fully untyped, Julia's method table treated them as **the same method** — whichever
file's `function` statement executed last in a given process replaced the other, process-globally, not
two independently-dispatched methods. `oracle_fast.jl` is `include`d by ~280 files across the repo
(overwhelmingly the FULL/legacy-bundle test and benchmark suite); of those, 19 files also `include`
`profiled_operator_bundle_2026-08-01.jl` later — in every one of those 19, the ProfiledCBState-oriented
definition silently won, including for any of `oracle_fast.jl`'s own registration sites
(`oracle_fast.jl:145`, `cm_hessian_architectures.jl:1794`, `chunked_hessian.jl:136`) that happened to run
afterward in the same process. In practice this never manifested as a crash because no real script both
(a) drives the legacy 1-arg `inner_loop_KNITRO_profiled(obj::<legacy bundle>)` path and (b) includes
`profiled_operator_bundle_2026-08-01.jl` in the same process — but it meant the REDUCED/`ProfiledCBState`
path's own callback (`profiled_operator_bundle_2026-08-01.jl:74`) computed
`evalResult.obj[1] = sum(Psi_q) / M + ζ` **unconditionally**, with no reference to `obj.lower_limit` at all.

`_callbackEvalH_inner_profiled!` (Hessian callback) has the identical collision shape at
`oracle_fast.jl:112` / `profiled_operator_bundle_2026-08-01.jl:108` (no clamp logic relevant here, but
the same untyped-signature collision).

## 2. `lower_limit` — root cause is a struct default, not (only) the collision above

`OperatorPsiBundle` (`operator_psi_bundle.jl:75`, `@with_kw mutable struct`) declared
`lower_limit::Float64 = -KNITRO.KN_INFINITY` as its own default. Of the 13 `OperatorPsiBundle(...)`
construction call sites in the whole repo, **8 already pass `lower_limit` explicitly** (`cm_frechet_level.jl`
×2, `cm_meanzc_moments.jl` ×2, `cm_production_bundle.jl` ×2, `compressed_live.jl`, both
`test_operator_no_H_bundle_equivalence_unrestricted*.jl` files) — all FULL-side or shared. The remaining
**5, all REDUCED-family bundle constructors, omitted it**, so each silently received the struct default
`-KN_INFINITY` instead of the real value (`-50` everywhere it's set at a context root, e.g.
`context_real_d20.jl:198`, `context_scaled.jl:83`):

| Family | Constructor | File:line (pre-fix) |
|---|---|---|
| unrestricted | `build_profiled_operator_bundle` | `profiled_operator_bundle_2026-08-01.jl:57` |
| common_frechet | `build_reduced_frechet_operator_bundle` | `profiled_reduced_frechet_lookup_kernels_2026-08-02.jl:157` |
| flexible_CM | `build_reduced_cm_operator_bundle` | `profiled_reduced_lookup_kernels_2026-08-02.jl:209` |
| cm_meanzc | `build_reduced_meanzc_operator_bundle` | `profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl:193` |
| origin_zc | `build_reduced_originzc_operator_bundle` | `profiled_reduced_originzc_lookup_kernels_2026-08-02.jl:112` |

Because `f <= -Inf` is never true for any real evaluated `f`, the clamp condition was permanently inert
for all 5 REDUCED families regardless of the collision above — a genuinely-unbounded trial point ran the
full KNITRO iteration budget (100 by production default) instead of exiting fast via native
`nStatus=-300`. Measured pre-fix cost (salvage package, `diagnostic-profiled-outer-fastfail-nu-ab-2026-08-02`,
D4-gated, not independently re-run this session before the fix): 33-38s → target ~1.5-1.7s per rejection
at W=20,000; 77-90s → target ~7.0-7.2s at W=100,000.

## 3. `threaded_bins` — per-family settings at every production/test construction site

Kernel-level parity for `threaded_bins=true` vs `false` under the **profiled/reduced** layout is already
gate-tested and passing in canonical for two families:
- `test_flexcm_threaded_profiled_d4_2026-08-02.jl` (flexible_CM): `hessian_cm_structured_v2!` threaded
  vs serial, max|Δ| < 1e-10.
- `test_zc_lane_cmzc_threaded_profiled_d4_2026-08-02.jl` (CM+ZC): same check, same tolerance.

Despite that, every actual production/driver construction site (`run_coldsolve_*_w100k_2026-08-02.jl`,
`run_prodscale_*_2026-08-02.jl`, `run_outer_flexcm_reduced_constrained_2026-08-02.jl`, and nearly all
`test_profiled_reduced_*`/`test_*_hessian_dense_truth_audit*` files) hardcodes
`build_cm_bin_ctx(...; threaded_bins = false, ...)` for flexible_CM's own `cctx`. Two in-repo comments
(`cm_hessian_threaded.jl:215`, `cm_hessian_architectures.jl:1487`) independently state
`threaded_bins=false` "is required for every profiled/reduced cctx in this codebase" — but both comments
describe the state *before* a same-day (2026-08-02) fix ported the missing `cctx.profiled_layout` branch
into the threaded twin (`cm_hessian_threaded.jl`), which the gate tests above already exercise
successfully. Whether that stale-comment / still-`false`-everywhere gap is genuine caution or simply an
unflipped default has NOT been independently re-verified this session yet (task §6 requires this per
family, not assumed from one).

`common_frechet` and `cm_meanzc` have no analogous threaded-vs-serial D4 gate test found in canonical —
their `threaded_bins` status remains `AMBIGUOUS`, a separate, still fully open issue from flexible_CM's.

## 4. Hessian backend selection (ZC-lane)

`cm_hessian_architectures.jl` (H_EM, drawmajor_v2 path) and the ZC dispatch-proof tests
(`test_zc_lane_cmzc_dispatch_proof_2026-08-02.jl`, `test_zc_lane_originzc_dispatch_proof_2026-08-02.jl`)
show `drawmajor_v2`/`draw_chunk_reordered`/`blas_syrk` dispatch counters already exist and are already
gate-tested with positive dispatch + zero fallback checks for both origin_ZC and CM+ZC under the REDUCED
path — this appears more mature than the audit's "NOT_INDEPENDENTLY_VERIFIED_THIS_PASS" framing suggested;
re-running these existing gates (not writing new ones) is the right first step for task §8.

## 5. common_frechet / cm_meanzc Hessian allocation regressions (confirmed live, unfixed as of baseline)

- `common_frechet`: `_fill_frechet_level_blocks!` (FULL path, `cm_frechet_hessian.jl:197`) uses
  `mul!(ext.block_cmlevel, cctx.R', Hraw_cmlevel)` (persistent buffer). The REDUCED/profiled sibling
  `_fill_frechet_level_blocks_profiled!` (`cm_frechet_hessian.jl:331`) instead allocates
  `cctx.R' * Hraw_cmlevel` fresh every call. Confirmed present, unfixed, at baseline.
- `cm_meanzc`: `_fill_cm_HEE!`'s fixed branch (`cm_hessian_architectures.jl:1224-1226`) uses an explicit
  `@inbounds` mirror loop for both triangles; the `profiled_layout` branch (`cm_hessian_architectures.jl:1080`)
  instead does an allocating transpose-broadcast into a view. Confirmed present, unfixed, at baseline.

## 6. D20/W20k cold-solve and fast-rejection status at baseline

Not independently re-run this session before the fix (per task §3, no benchmarking of known-broken code).
Per audit evidence: unrestricted and flexcm/CM-grid REDUCED cold solves both reach a symptom consistent
with the missing `lower_limit` clamp (slow `nStatus=-400`-style timeouts rather than fast `nStatus=-300`
rejections). flexible_CM's specific "eval18" D20/W=100,000 stall is independently resolved as diagnosis
already (memory `eval18-forensic-verdict-genuinely-unbounded-2026-08-02`): genuinely unbounded, Hessian
conditioning explodes to ~1e10-3.8e11 along the trajectory, and raising `maxit` 100→1000 lets it correctly
cross the `-50` clamp at iteration 299 and exit `nStatus=-300` — not itself the same defect as the
missing-clamp method-collision bug fixed in this task (that defect meant the clamp could never fire at
ANY iteration, not merely too slowly).
