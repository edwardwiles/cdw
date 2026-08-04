# Prior-agent claim/evidence audit — 2026-08-03

Source archives (pulled from `dropbox:Gravity robustness/Analysis/Server Output/`, per this repo's
standing rule that task-referenced zips live on Dropbox):

- `profiled_inner_readiness_2026-08-03.zip` → task-1 report ("REDUCED inner correctness and
  kernel-parity closeout"), branch `fix/profiled-inner-readiness-2026-08-03`, merged/tagged
  `profiled-inner-ready-2026-08-03@b80dd48`.
- `profiled_outer_production_readiness_2026-08-03_continuation.zip` → task-2 session-2 report,
  branch `feature/profiled-outer-production-readiness-2026-08-03@781eb65`, not merged.

Both extracted to
`/bbkinghome/edav/repo_scratch/profiled-functional-readiness-closeout-2026-08-03/prior_archives/`.
Current source checked in worktree
`/bbkinghome/edav/cdw_worktrees/profiled-functional-readiness-closeout-2026-08-03`
(`fix/profiled-functional-readiness-closeout-2026-08-03`, branched from
`origin/feature/profiled-outer-production-readiness-2026-08-03@781eb65` — confirmed via
`git log --oneline -1 origin/feature/profiled-outer-production-readiness-2026-08-03` at session
start, matches the task brief's recorded HEAD exactly).

## Provenance discrepancy — resolved, not a real contradiction

The outer continuation's `MASTER.md` §Provenance states `HEAD: 5e2c262` (10 commits ahead of
`b80dd48`); its own `provenance.txt` states `HEAD: 781eb65fcaf77a0efbfb5a807905f6fa80f7c2af`. These
are not in conflict: `git log` on that branch shows `781eb65` ("Update MASTER.md: full honest
status...") is the very next commit on top of `5e2c262` — i.e. `MASTER.md`'s own prose records the
HEAD *as of the moment its content was drafted*, one commit before the commit that actually adds
that file to the tree. `provenance.txt` (generated mechanically, after the commit) and the real
`origin` ref both agree on `781eb65` as the true, current tip. **No stale/wrong-branch risk** — this
task's worktree is confirmed built from the correct, most-recent commit.

## Claim/evidence table

| # | Claim | Source report | Archived log | Current-source confirmation | Status |
|---|---|---|---|---|---|
| 1 | `_callbackEvalFG_inner_profiled!`/`_callbackEvalH_inner_profiled!` method collision fixed via `userParams::ProfiledCBState` type dispatch | inner MASTER.md §1 | n/a (code-level claim) | Confirmed live: `profiled_operator_bundle_2026-08-01.jl:87,132` both typed `userParams::ProfiledCBState`; `oracle_fast.jl:102,112` remain untyped/generic (legacy bundle path) — two genuinely separate methods, not a collision | VERIFIED |
| 2 | `OperatorPsiBundle.lower_limit` has no struct default (required field) | inner MASTER.md §1-2 | n/a | Confirmed live: `operator_psi_bundle.jl:81` declares `lower_limit::Float64` with no `=` default | VERIFIED |
| 3 | All 5 REDUCED bundle constructors propagate `lower_limit` explicitly | inner MASTER.md §2 | n/a | Confirmed live via grep: all 5 files (`profiled_operator_bundle_2026-08-01.jl`, `profiled_reduced_frechet/lookup/meanzc/originzc_lookup_kernels_2026-08-02.jl`) pass `lower_limit = ref_obj.lower_limit` | VERIFIED |
| 4 | common_frechet uses persistent `mul!` buffer (`cm_frechet_hessian.jl:331`) | inner MASTER.md §3 | n/a | Confirmed live: line reads `mul!(ext.block_cmlevel, cctx.R', Hraw_cmlevel)` with an inline comment documenting the fix | VERIFIED |
| 5 | cm_meanzc uses explicit H_EM mirror loop, not allocating transpose-broadcast | inner MASTER.md §3 | n/a | Confirmed live: `cm_hessian_architectures.jl:1080` region now has an explicit `@inbounds` loop with a comment documenting the aliasing-defensive-copy defect it replaces | VERIFIED |
| 6 | Independent verification wired into all 4 restricted REDUCED evaluators | inner MASTER.md §4 | n/a | Confirmed live: `profiled_restricted_family_adapters_2026-08-02.jl` calls `verify_inner_solution_reduced_cm!`/`_cm_frechet!`; `profiled_zc_lane_point_evaluators_2026-08-02.jl` calls `verify_inner_solution_reduced_originzc!`/`_cmzc!`; all 4 functions exist in `reduced_restricted_family_verification_2026-08-03.jl` / `reduced_originzc_verification_2026-08-02.jl` | VERIFIED |
| 7 | `threaded_bins=true` flipped for flexible_CM's 4 real production drivers | inner MASTER.md §6 | n/a | Confirmed live: `run_coldsolve_flexcm_w100k`, `run_prodscale_flexcm`, `run_prodscale_flexcm_frechet_ab` (flexcm line only — frechet line correctly stays `false`), `run_outer_flexcm_reduced_constrained` all pass `threaded_bins = true` for the flexcm cctx | VERIFIED |
| 8 | `inner_fg_backend = :dense_reference` kwarg in these same drivers does NOT mean dense G/H materialization on the REDUCED path | not an explicit prior claim, but load-bearing for the "no dense reduced" rule | `run_coldsolve_flexcm_w100k_2026-08-02_tail.txt`: `dense_economic_G_materializations=0 dense_CM_G_materializations=0 (0 expected)`, `FLEXCM_W100K_COLD_SOLVE_RESULT: PASS` | Archived log's own counters prove zero dense materializations at real W=100,000; the `:dense_reference` symbol is a legacy kwarg name whose actual dispatch is gated on `profiled_layout` being set, independently checked via `NO_DENSE_G_COUNTERS[]` in the same script (`brute_force_verify`'s own intentional dense recompute is correctly excluded from the pre-verify snapshot) | VERIFIED (flagged and resolved this session — not misleading, but worth recording since the kwarg name alone would suggest a violation of the repo's zero-tolerance dense-reduced rule) |
| 9 | common_frechet/cm_meanzc `threaded_bins` deliberately left ambiguous/untouched | inner MASTER.md §6, §9 | n/a | Confirmed live: no `threaded_bins=true` override found for these two families in any production driver; task's own §3.2 requires closing this | VERIFIED (as a statement of remaining scope, not a completed gate) |
| 10 | ZC optimized backends (`blas_syrk`, `drawmajor_v2`, `draw_chunk_reordered`) dispatch with zero fallback | inner MASTER.md §5 | not independently pulled this session | not re-run this session yet | SOURCE_ONLY (pending re-run under §3 of this task) |
| 11 | Genuine-cold W=100,000 KNITRO solves, `nStatus=0`, for all 5 families | inner MASTER.md §8 | `run_coldsolve_{cmzc,flexcm,frechet,originzc}_w100k_2026-08-02_tail.txt` — all 4 present and show `nStatus=0` with KKT residuals 6.6e-13 to 8.2e-14; unrestricted's own genuine-cold result is quoted in MASTER.md text (not a separate archived tail file, but backed by `test_unrestricted_stall_sentinel_w100k_2026-08-03.jl`'s own archived summary) | not re-run this session yet | SOURCE_ONLY (logs present and internally consistent; no fresh re-run performed this session — scheduled under §3.3 of this task) |
| 12 | unrestricted stall resolved: adversarial points reject in ~1.3s via `nStatus=-300` (vs. pre-fix 77-90s `nStatus=-400`) | inner STALL_REASSESSMENT.md | `test_stall_sentinel_summary.txt` (archived) | table matches the MASTER.md text (7.14s/1.56s feasible, 1.33s/1.34s adversarial `-300`) | VERIFIED (as reported); **not yet independently re-derived against a historically-captured point this session** — the reassessment doc itself says these are freshly-constructed adversarial points, not a byte-for-byte historical replay. Task §4 requires locating and replaying the actual historical stall point — still open, see below |
| 13 | flexible_CM eval18 = `resolved_as_diagnosis`, genuinely unbounded | inner MASTER.md §7, §10 (`RESIDUAL_INNER_STALL: flexible_CM: resolved_as_diagnosis`) | none archived in either zip this session — text says "not re-derived, relies on prior conclusion" | prior conclusion itself lives in memory `eval18-forensic-verdict-genuinely-unbounded-2026-08-02`, not in either supplied archive | **UNRESOLVED per this task's own explicit instruction** ("Do not preserve that classification without evidence... the result was not re-derived and relies on a prior conclusion that the user did not accept"). Requires fresh forensic replay — see §4 of this task, tracked separately, not closed by this audit alone |
| 14 | flexible_CM W=100,000 fast-rejection sentinel inconclusive (both attempts solved to optimality instead of rejecting) | inner MASTER.md §9 | implied by `test_flexcm_fastreject_w100k_summary.txt` | not independently re-derived this session | SOURCE_ONLY — honestly reported as a known gap by the prior agent, not a false claim; task §3.3 requires constructing a genuinely infeasible point systematically |
| 15 | `scientific_manifest/` full suite 173/173 passing (ScientificManifest=29, RunManifest=43, FamilyRegistry=70, ABComparability=31) | outer MASTER.md | `test_scientific_manifest_output.txt`, `test_run_manifest_output.txt`, `test_family_registry_output.txt`, `test_ab_comparability_output.txt` (all 4 archived) | not re-run this session yet | SOURCE_ONLY (pending re-run under §3 of this task) |
| 16 | D4 checkpoint/resume round trip 18/18 checks PASS for `run_profiled_upper_constrained` (origin_zc) | outer MASTER.md §5 | `d4_checkpoint_resume_output.txt` **ends in `signal 15: Terminated`**, mid-JIT-compile inside the Hessian callback (`inner_loop_KNITRO_reduced_originzc` → `KN_solve`), never reaches a pass/fail line | Re-run in progress this session (see below) | **LOG_MISSING / not creditable as-is** — per this task's explicit instruction, the terminated log alone does not support the 18/18 claim. A fresh clean re-run was launched this session (`d4_checkpoint_resume_rerun_2026-08-03.txt`); see the checkpoint/resume section of `MASTER.md` for its outcome, which supersedes this row |
| 17 | FamilyRegistry symbol-mapping finding: FULL uses `:unrestricted`/`:flexible_cm`/`:common_frechet`/`:origin_zc`/`:cm_meanzc`, REDUCED's `family_kind` uses `:unrestricted`/`:flexible_CM`/`:common_Frechet`/`:ZC_only`/`:CM_plus_ZC` — two different symbol sets for the same 5 families | outer MASTER.md §4 | n/a (grep-based finding) | Independently spot-checked this session: `scientific_manifest/FamilyRegistry.jl` contains `REDUCED_FAMILY_KIND_TO_CANONICAL`; both symbol sets appear as claimed | VERIFIED |
| 18 | Section 6 (free eta_nu) is investigation-only, not implemented; blocked on `evaluate_profiled_originzc_point`'s nu being closed-over/fixed, not a function argument | outer MASTER.md §6, `section6_free_nu_investigation.md` | n/a | Confirmed live: `profiled_zc_lane_point_evaluators_2026-08-02.jl:70-74` calls `evaluate_profiled_originzc_point` with no `nu` positional/keyword argument in its signature at the call site consistent with the report | VERIFIED |
| 19 | Section 7 coordinate mode `:profiled_pivot_anchor_relative` round-trip verified at D4 (10/10 checks) but gradient chain-rule NOT verified | outer MASTER.md §7 | `d4_coordinate_roundtrip_output.txt` (archived) | not re-run this session yet | SOURCE_ONLY (pending re-run); gradient chain-rule verification remains explicitly open, folded into this task's §8 |
| 20 | `bin/run_profiled_model.jl` / unified CLI runner NOT built | outer MASTER.md §4, verdict `CANONICAL_RUNNER = fail_registry_done_unified_cli_not_built` | n/a | Confirmed live: no `bin/` directory exists in the worktree; no unified runner found by name search | VERIFIED (as an honest gap, correctly not claimed as done) |
| 21 | `production_ready=false` for all 5 REDUCED families in FamilyRegistry | outer MASTER.md verdict block | n/a | Not yet independently re-read this session (deferred to §3/§9 gate work) | SOURCE_ONLY |

## Net assessment

The two prior reports are **honest and internally consistent** with each other and with the current
source in every case checked this session — no `LOG_CONTRADICTS_REPORT` or fabricated claim was
found. The two genuine problems this task's brief anticipated are both real and both confirmed
independently this session:

1. **Row 16**: the checkpoint/resume "18/18 PASS" claim's own archived evidence is a terminated
   process, not a passing run — correctly not creditable until re-run. A fresh re-run was launched
   this session; see `MASTER.md` for the resolved outcome.
2. **Row 13**: flexible_CM's eval18 classification (`resolved_as_diagnosis`) is explicitly
   self-described by its own author as not re-derived and rests on a prior finding the user did not
   accept — correctly treated as unresolved by this task, not inherited as settled.

No other claim in either archive was found to be `LOG_MISSING`, `LOG_CONTRADICTS_REPORT`, or
`STALE_AFTER_LATER_COMMITS` as of this session's read of current source at
`fix/profiled-functional-readiness-closeout-2026-08-03` (branched from `781eb65`).
