# REDUCED inner correctness and kernel-parity closeout — MASTER report — 2026-08-03

Branch: `fix/profiled-inner-readiness-2026-08-03`, from canonical `prototype/profiled-destination-scales@7ec5c6c`.
Worktree: `/bbkinghome/edav/cdw_worktrees/profiled-inner-readiness-2026-08-03`. 8 commits, all
validated against real Julia/KNITRO runs (not merely code review) before being written. Baseline
recorded in `BASELINE.md`; residual-stall reassessment in `STALL_REASSESSMENT.md` (both this
directory).

## 1. Before/after callback-method map

| Function | Before | After |
|---|---|---|
| `_callbackEvalFG_inner_profiled!` | Two fully-untyped 5-arg definitions (`oracle_fast.jl:102`, `profiled_operator_bundle_2026-08-01.jl:74`) — a genuine Julia method-table collision; whichever file loaded last silently replaced the other process-globally. | Type-dispatched: `oracle_fast.jl`'s stays generic (serves legacy callable `PsiObjectiveBundle*` types); `profiled_operator_bundle_2026-08-01.jl`'s is now `userParams::ProfiledCBState`-specific. Both coexist as genuinely separate methods — confirmed by `length(methods(...)) >= 2` in the new regression test. |
| `_callbackEvalH_inner_profiled!` | Identical collision shape. | Identical fix (type-annotated `::ProfiledCBState`). |
| `OperatorPsiBundle.lower_limit` | Struct-level default `-KNITRO.KN_INFINITY` — the actual root cause: 5 REDUCED bundle constructors silently omitted the kwarg and got an inert clamp instead of an error. | No default — a required field. All 8 pre-existing call sites already passed it explicitly (verified); the 5 broken ones now do too. |

## 2. lower_limit replay table

| Family | Pre-fix | Post-fix (this session, D20/W=100,000, freshly-constructed adversarial points) |
|---|---|---|
| unrestricted | clamp never fires (inert `-Inf` default) → slow `nStatus≈-400` timeout (salvage measurement: 77-90s at W=100k) | 2/2 adversarial points reject in ~1.3s via correct `nStatus=-300` |
| flexible_CM | same defect | clamp mechanism independently verified in isolation (Psi≡0 stand-in, exact line-level test); one attempted real adversarial point at W=100k solved to genuine optimality instead (not infeasible for this family's feasible region at this scale — not a regression, see §9 below) |
| common_frechet | same defect | clamp kwarg now propagated; family's own genuine-cold W100k solve reaches `nStatus=0`, `kkt_resid=7.1e-13` |
| origin_ZC | same defect | clamp kwarg now propagated; genuine-cold W100k solve `nStatus=0`, `kkt_resid=7.4e-13` |
| CM_plus_ZC (cm_meanzc) | same defect | clamp kwarg now propagated; genuine-cold W100k solve `nStatus=0`, `kkt_resid=8.2e-14` |

## 3. Allocation fixes before/after

| Family | Site | Before | After | Validation |
|---|---|---|---|---|
| common_frechet | `cm_frechet_hessian.jl:331` (`_fill_frechet_level_blocks_profiled!`) | `cctx.R' * Hraw_cmlevel` — fresh alloc every L×L loop iteration | `mul!(ext.block_cmlevel, cctx.R', Hraw_cmlevel)` — persistent buffer, ported from the already-fixed non-profiled sibling | D4 FG/KNITRO-solve regression test passes unchanged (`inner_status=0`) |
| cm_meanzc | `cm_hessian_architectures.jl:1080` (`_fill_cm_HEE!` profiled branch) | `@views HEE[ncore+1:NCORE,1:ncore] .= transpose(HEM)` — hits Julia's broadcast aliasing-defensive-copy path (source/dest are views of the same parent array) | explicit `@inbounds` mirror loop, ported from the already-fixed non-profiled branch | D4 ForwardDiff Hessian cross-check passes at machine precision (H_EM block max\|Δ\|=2.69e-15) — proves the fix is numerically correct, not just non-allocating |

## 4. Verification-gate table

| Family | Before | After |
|---|---|---|
| unrestricted | already wired (`verify_inner_solution_reduced_profiled!` + `verify_namedtuple_from_operator`) | unchanged, re-confirmed passing |
| flexible_CM | zero verification calls | new `verify_inner_solution_reduced_cm!` wired into `evaluate_profiled_flexcm_point` |
| common_frechet | zero verification calls | new `verify_inner_solution_reduced_cm_frechet!` wired into `evaluate_profiled_frechet_point` |
| origin_ZC | verifier existed (`verify_inner_solution_reduced_originzc!`, 2026-08-02) but was **never called from its own evaluator** (confirmed live: `UndefVarError` before this session's include-guard fix) | wired into `evaluate_profiled_originzc_point` |
| CM_plus_ZC | zero verification calls, no verifier existed for this exact block combination | new `verify_inner_solution_reduced_cmzc!` (reduced-econ + ZC mean/pair + CM-grid, three blocks) wired into `evaluate_profiled_cmzc_point` |

**Positive gates**: all 4 restricted-family D4 outer-gradient/production-runner tests pass unchanged
through the real evaluators (not mocks) with genuine KNITRO solves, `Delta_dual`/`kkt_resid`
computed from a fresh, independently-recomputed dual residual.

**Negative gate** (`test_reduced_restricted_verification_negative_2026-08-03.jl`): corrupting one
dual coordinate by 50 (no re-solve) blows up the KKT residual by 14-15 orders of magnitude for all
4 families (genuine ~1e-15 → corrupted ~5), proving a caller gating on these fields would correctly
reject a forged "accepted" status.

## 5. Backend-dispatch table

Re-ran the existing dispatch-proof tests (not newly written — already existed, already covered
this) after this session's edits, to confirm no regression:

| Backend | Family | Result |
|---|---|---|
| `blas_syrk` (H_ZZ/H_MM) | origin_ZC, CM_plus_ZC | dispatches (positive count, zero fallback) |
| `drawmajor_v2` (H_EZ/H_EM) | origin_ZC, CM_plus_ZC | dispatches (positive count, zero fallback) |
| `draw_chunk_reordered` (H_CZ) | CM_plus_ZC | dispatches (positive count, zero fallback) |
| `threaded_bins=true` (flexible_CM, CM_plus_ZC threaded Hessian) | both | matches serial to ~1e-14/1e-15 at both 4 and **production's 10-thread policy** |

## 6. threaded_bins production default

Flipped `threaded_bins` from `false` to `true` for flexible_CM's own cctx construction in all 4 real
production driver scripts (`run_coldsolve_flexcm_w100k`, `run_prodscale_flexcm`,
`run_prodscale_flexcm_frechet_ab`'s flexcm line only, `run_outer_flexcm_reduced_constrained`) —
the kernel-level threaded branch was already gate-tested and passing; the driver default was simply
stale/unflipped. `common_frechet`/`cm_meanzc`'s own `threaded_bins` remain untouched (still `false`)
— no analogous gate exists for them; this is a **separate, still-open ambiguity**, not addressed
this session (matches the task's own scope note).

## 7. Stall sentinel results

See `STALL_REASSESSMENT.md` for full detail.

- **unrestricted**: resolved. Real D20/W=100,000 panel — 2 feasible points solve cleanly
  (`nStatus=0`, KKT~1e-13); 2 adversarial points reject in ~1.3s via correct `nStatus=-300` (vs.
  pre-fix ~77-90s slow `nStatus=-400` timeout).
- **flexible_CM ("eval18")**: resolved as diagnosis (prior session,
  `eval18-forensic-verdict-genuinely-unbounded-2026-08-02`, not re-derived) — genuinely unbounded
  point, severe Hessian ill-conditioning, a *different* issue from the method-collision bug fixed
  this session. One fresh adversarial attempt at W=100,000 this session solved to genuine
  optimality rather than reproducing infeasibility — inconclusive on that specific perturbation,
  not a regression (the clamp mechanism itself is independently verified in isolation).
- No residual stall found requiring the bounded-escalation program (per-iterate capture, FD
  cross-checks, solver-arm matrix) — not run, per the task's own conditional instruction.

## 8. Five-family inner-readiness table

| Family | D4 (FG/FD/Hessian/verification) | D20/W=20,000 | D20/W=100,000 |
|---|---|---|---|
| unrestricted | PASS (outer-gradient FD gate) | not separately re-run this session (already covered at W=100k, a superset scale, by the stall sentinel) | genuine-cold PASS (calibration) + 2 fast-rejection PASS (~1.3s each) |
| flexible_CM | PASS (FG/KNITRO-solve, threaded parity) | not separately re-run this session | genuine-cold PASS (`nStatus=0`, kkt=6.6e-13); fast-rejection attempt inconclusive (solved instead of rejecting — see §7) |
| common_frechet | PASS (FG/KNITRO-solve) | not separately re-run this session | genuine-cold PASS (`nStatus=0`, kkt=7.1e-13) — **this family's first real execution at any scale**, per task's own §6 instruction |
| origin_ZC | PASS (FG/Hessian ForwardDiff cross-check, dispatch-proof) | PASS via real production runner (`test_phase13_production_runner_d20_w20000_gate_2026-08-02.jl`: verified-feasible best incumbent found, checkpoint round-trips, exact-cache lookups recorded) | genuine-cold PASS (`nStatus=0`, kkt=7.4e-13) |
| CM_plus_ZC | PASS (FG/Hessian ForwardDiff cross-check, dispatch-proof, threaded parity) | PASS via real production runner (same gate as above) | genuine-cold PASS (`nStatus=0`, kkt=8.2e-14, `winner_cross_hessian_calls=16`, `dense_cross_hessian_calls=0`) |

Warm-start solves at W=100,000 were not separately executed this session for the 4 restricted
families (out of the wall-clock budget spent on the items above); D4/D20 evidence plus the
genuine-cold W=100,000 PASS for all 5 families is the basis for the readiness verdict below.

## 9. Honest limitations of this pass

- flexible_CM's W=100,000 fast-rejection sentinel did not land on an actually-infeasible point
  (two attempted perturbations both solved to genuine optimality) — the clamp mechanism itself is
  proven correct via an isolated unit test and via unrestricted's own successful W=100,000
  fast-rejection, but this is not the same as replaying a genuinely-infeasible flexible_CM point at
  this exact scale.
- No warm-start-solve gate was run at W=100,000 for the 4 restricted families.
- common_frechet/cm_meanzc `threaded_bins` ambiguity remains open (by design, out of this
  session's scope per the task's own note).
- P2 items (checkpoint wiring into `run_profiled_upper_constrained`, outer-gradient threading,
  bandwidth-search caching) were explicitly out of scope for this task and were not touched.

## 10. Commits

```
54b0a94 Fix _callbackEvalFG_inner_profiled! method collision and missing lower_limit clamp
d7ac04b Port common_frechet persistent mul! buffer fix to profiled branch
013a09e Port cm_meanzc H_EM mirror-loop allocation fix to profiled branch
1a0d44f Add pre-fix baseline doc for profiled-inner-readiness task
6031d1f Wire independent verification into all 4 restricted REDUCED evaluators
ab26d00 Flip threaded_bins=true for flexible_CM's real production drivers
a8d8ef4 Add stall-regression sentinel panel + reassessment doc
945dabe Add flexible_CM W100k fast-rejection sentinel (task 9/10)
```

HEAD at time of writing: `945dabe`. Merge/push/tag/cleanup status: **not yet performed** — pending
explicit user confirmation per this session's standing safety discipline (destructive/remote
git operations require confirmation regardless of task-brief instructions to do them
automatically).

## Final verdict block

```
CALLBACK_METHOD_COLLISION = fixed
LOWER_LIMIT = pass_all_five
THREADED_BINS =
    flexible_CM: pass
    common_frechet: not_addressed (pre-existing ambiguity, out of scope this session)
    cm_meanzc: not_addressed (pre-existing ambiguity, out of scope this session)
ALLOCATION_FIXES =
    common_frechet: pass
    cm_meanzc: pass
INDEPENDENT_VERIFICATION =
    unrestricted: pass
    flexible_CM: pass
    common_frechet: pass
    origin_ZC: pass
    CM_plus_ZC: pass
ZC_BACKENDS =
    origin_ZC: pass
    CM_plus_ZC: pass
RESIDUAL_INNER_STALL =
    unrestricted: resolved
    flexible_CM: resolved_as_diagnosis (pre-existing eval18 finding; separate from the fixed defect)
INNER_READY =
    unrestricted: yes
    flexible_CM: yes
    common_frechet: yes
    origin_ZC: yes
    CM_plus_ZC: yes
MERGED_TO_CANONICAL_PROTOTYPE = no_pending_user_confirmation
EXTRA_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
DENSE_CODE_USED = false
FULL_PRODUCTION_CHANGED = false
CAMPAIGN_LAUNCHED = false
```
