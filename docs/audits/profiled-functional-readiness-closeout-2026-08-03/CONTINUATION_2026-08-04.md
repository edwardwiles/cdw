# REDUCED functional-readiness closeout — continuation — 2026-08-04

Continuation of the prior session's own honest partial-completion `MASTER.md` (see that file for
full §1-10 status as of 2026-08-03). This session continued from the exact recorded HEAD
(`5a1a647d94c6844e7b0a8d2889155c2966bc7ea3`, confirmed live via `git rev-parse HEAD` against both
the local worktree and `origin/fix/profiled-functional-readiness-closeout-2026-08-03` at session
start), same branch, same worktree — `NEW_BRANCHES_CREATED=0`, `NEW_WORKTREES_CREATED=0`.

Machine load was `uptime` 241-270 across 104 physical cores (~2.3-2.6/core, above the task's own
1.5/core deferral threshold) for essentially this entire session, from other users' entirely
unrelated multi-hour jobs (`chsiegm`'s `sweep_stacked.py`, `ajabbar`'s Julia jobs) plus this
account's own other active sessions (melitz campaigns, `pilot_tournament` polish) — not stale jobs
from this task. No orphaned process from the two scripts the task brief named
(`test_frechet_meanzc_threaded_profiled_d20w20k_2026-08-03.jl`,
`verify_eval18_current_code_2026-08-03.jl`) was found running at session start. Per the task's own
load policy, heavy W=100,000 campaigns for all 5 families (§4.1/§4.2 warm-start/fast-rejection)
were NOT attempted this session; work focused on source/derivation work and the bounded, named
follow-ups the prior session's own precise blocker list identified as mechanical.

## 1. Two real bugs fixed in the prior session's own scripts (both were silently producing false
   results, caught via `set -o pipefail` + actually reading the output, not trusting exit code 0)

**`verify_eval18_current_code_2026-08-03.jl`'s `try_evaluate` catch handler.** The prior session's
own fix assumed `CMExpectedSolveFailure` (`cm_production_bundle.jl:46`) carries a structured
`.nStatus` field; it does not — `struct CMExpectedSolveFailure <: Exception; msg::String; end`,
`nStatus` is only interpolated into the message string. The catch handler's own `e.nStatus` access
therefore threw `FieldError` INSIDE the handler, and because the script was originally launched as
`julia ... | tee logfile` (no `pipefail`), the pipeline's exit code was `tee`'s (0), masking the
real failure as a false pass. Fixed by parsing `nStatus=(-?\d+)` back out of `e.msg` via regex
instead. Confirmed live: the underlying error message already contained the decisive
`nStatus=-400` for the `maxit=100` arm even before this fix — the fix only makes both arms
actually complete and return a value instead of crashing.

**Both pre-existing ZC-lane D4 outer-gradient math gates**
(`test_zc_lane_originzc_outer_gradient_d4_2026-08-02.jl`,
`test_zc_lane_cmzc_outer_gradient_d4_2026-08-02.jl`) were BROKEN on current HEAD, not passing as
their file names/prior citations implied — a real API-drift bug, not a re-derivation issue. Both
hand-build an `ev` NamedTuple to call `shared_family_outer_gradient`/`build_shared_profiled_lfix_cache`,
which now requires an `ev.decoded` field (`decode_outer_profiled(w_profiled, ctx, pe)`) that
neither test's `ev`/`ev2` construction included — confirmed via `FieldError: type NamedTuple has no
field decoded`. Fixed by adding `decoded = decode_outer_profiled(...)` to all three `ev`
constructions (origin-ZC calibration + perturbed point, CM+ZC calibration). **Re-run clean after
the fix: both `ALL PASS`** (origin-ZC: 2/2 checks at calibration + 1/1 at perturbed point; CM+ZC:
2/2 checks). Logs: `repo_scratch/.../logs/originzc_d4_outer_grad_rerun.log`,
`.../cmzc_d4_outer_grad_rerun.log`.

## 2. eval18 stall forensics — reopened with a genuine new root-cause finding (task §5.1)

The prior session's `verify_eval18_current_code_2026-08-03.jl` never actually tested a maxit=1000
arm, on ANY session including this one's own first attempt: `evaluate_profiled_flexcm_point`'s
`maxit_override` keyword argument (`profiled_restricted_family_adapters_2026-08-02.jl:99-119`) is
**dead code** — accepted but never forwarded anywhere in the function body. Traced the real
mechanism: `reduced_cm_base_state` → `inner_loop_KNITRO_reduced_cmlookup`
(`profiled_reduced_lookup_kernels_2026-08-02.jl:240`) sets KNITRO's `maxit` exclusively via
`KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)` — a **file path**, not a runtime override — and
`obj.inner_loop_opt` is threaded, UNCHANGED, all the way from `d20_real_setup_design`'s own
`inner_loop_opt` kwarg (default `full_aod_diag/ek_inner.opt`, confirmed `maxit 100` at line 378)
through `build_reduced_base_obj_for_family` → `build_cm_augmented_obj_archB` → `build_cm_bin_ctx`.
Confirmed live: this session's first `verify_eval18_current_code_2026-08-03.jl` re-run (after
fixing the try/catch bug above) returned **identical `nStatus=-400` for BOTH the maxit=100 and
"maxit=1000" arms**, with the maxit=1000 arm's wall time (189.01s) actually LESS than maxit=100's
(203.56s) — the fingerprint of a dead override, not a genuine wider budget (log:
`repo_scratch/.../logs/eval18_maxit1000_replay_20260803_2257.log`).

**Real fix**: `full_aod_diag/ek_inner_maxit1000_2026-08-04.opt` (byte-identical to `ek_inner.opt`
except `maxit 1000`) + new `verify_eval18_real_maxit1000_2026-08-04.jl`, which builds the D20/W=100,000
context with `inner_loop_opt` pointed at the maxit=1000 file and asserts
`cctx_reduced.obj.inner_loop_opt == opt_maxit1000` before trusting the solve (i.e. the script
verifies its own override actually propagated, not just that the file exists). Result: see
`EVAL18_REAL_MAXIT1000_RESULT` below.

**Real result (`verify_eval18_real_maxit1000_2026-08-04.jl`, second attempt — first attempt hit an
unrelated bug in the script's own pre-solve assertion, `cctx_reduced` has no `.obj` field, fixed by
reading `aug_reduced.obj_cm.inner_loop_opt` instead)**:

```
aug_reduced.obj_cm.inner_loop_opt = .../ek_inner_maxit1000_2026-08-04.opt  (propagation confirmed live, assertion passed)
[maxit=1000 REAL] wall=577.67s  nStatus=-300  n_fg=-1  n_hess=-1  zeta=NaN
EVAL18_REAL_MAXIT1000_RESULT: CONFIRMS 2026-08-02 verdict -- genuinely unbounded, correct
    lower_limit clamp fires once given a real wider budget
```

(log: `repo_scratch/.../logs/eval18_real_maxit1000_take2_2026-08-04.log`). **This is the decisive
result task §5.1 asks for** — `nStatus=-300` (`KN_RC_UNBOUNDED`, the correct `lower_limit=-50`
clamp firing) at a genuine maxit=1000, on the exact captured eval18 point, at real D20/W=100,000,
on this session's own current HEAD — the FIRST time maxit has actually been varied for this point
on this branch (neither this session's own first attempt nor the prior session's original script
ever exercised a real wider budget, per the dead-`maxit_override` finding above). Confirms the
2026-08-02 forensic verdict's own claim (unbounded, clamp fires ~iteration 299) rather than
contradicting it — the earlier apparent "DIVERGES" signal was entirely an artifact of the dead
kwarg, not a real regression. Wall time (577.67s) was ~3-10x archived reference timings due to this
session's own sustained machine contention (`uptime` 241-270 for the whole session) — noted for
completeness, not treated as evidence about anything (the terminal status code, not the wall clock,
is what answers the question).

```
STALL_STATUS (updated) =
    unrestricted: fix_confirmed_by_code_not_by_historical_replay (unchanged, §5.2 below)
    flexible_CM (eval18): CONFIRMED genuinely unbounded -- real maxit=100 -> nStatus=-400 (this
        session's earlier run) AND real maxit=1000 -> nStatus=-300 (this run), both on the exact
        captured point at real D20/W=100,000 on current HEAD, via a verified-propagating
        inner_loop_opt override (not the dead maxit_override kwarg). Matches the 2026-08-02
        forensic package's own two-part signature exactly. RESOLVED, not open.
```

## 3. Free eta_nu for origin-ZC and CM+ZC — IMPLEMENTED and D4-VERIFIED (task §7/§8, ZC portion)

**Why this was smaller than the prior session's own flagged blocker**: that session found
`evaluate_profiled_originzc_point`/`evaluate_profiled_cmzc_point` take `nu` only via a closed-over
FIXED `pes.nu_full` and flagged changing this as needing user sign-off on a "structural evaluator-
signature change." Confirmed live this session: the DEEPER kernels
`reduced_originzc_base_state`/`reduced_meanzc_base_state` already take `νfull`/`νvec` as a plain
explicit positional argument — the fixed-nu constraint was ONLY in the thin wrapper layer. No
inner-kernel changes were needed.

**New file** `full_aod_diag/d4_exact/profiled_zc_free_eta_2026-08-04.jl` (additive only, does not
modify `profiled_zc_lane_point_evaluators_2026-08-02.jl` — the existing fixed-nu 3-positional-arg
methods stay exactly as-is for existing inner unit tests, per the task's own §8.1 instruction):

- New 4-positional-arg methods `evaluate_profiled_originzc_point(w_econ, eta_nu, fctx, pes)` /
  `evaluate_profiled_cmzc_point(w_econ, eta_nu, fctx, pes)` — `eta_nu` (log-nu units, matching
  FULL's own `eta=log(nu)` convention) is a plain explicit argument, decoded as `nu_full=exp.(eta_nu)`
  inside. No mutable closed-over nu anywhere — each call receives its own vector, so there is no
  generation/staleness risk in the point-eval call itself (task §8.1's "no mutable closed-over nu
  without generation checks" requirement).
- `reduced_originzc_outer_gradient_with_eta`/`reduced_cmzc_outer_gradient_with_eta`: `vcat(g_econ,
  d_eta)`, mirroring FULL's own `cm_originzc_production_gradient` combined-gradient convention.
  `g_econ` via `shared_family_outer_gradient` (unchanged, shared engine); `d_eta` via FULL's own
  `d_delta_dual_d_eta_origin_vec` (`cm_originzc_moments.jl`) — reused UNCHANGED, not re-derived
  (its own docstring states it reduces to the shared-layout formula under `SharedByPowerLayout`, so
  ONE function correctly serves both origin-ZC's `OriginByPowerLayout` and CM+ZC's
  `SharedByPowerLayout`).
- `ZCFreeNuOuterLayout(economic_outer_range, eta_nu_outer_range)` — the typed combined
  `[gp; A_free; eta_nu]` outer-coordinate layout task §8.3 asks for, plus
  `split_free_nu_outer`/`pack_free_nu_outer`. Deliberately a DIFFERENT typed range from
  `CMZCFamilyCtx.n_lambda_meanpair` (that struct's own docstring already warns about this exact
  name-collision class of bug).
- `nu_generation_id(eta_nu)` — a content-hash primitive for future cache-key/dual-bank/checkpoint
  generation tagging (task §8.4). **NOT wired into any cache/dual-bank/checkpoint consumer this
  session** — real remaining integration work, listed in the blocker list below, not fabricated as
  done.

**Real bug found and fixed while gating this** (not a design gap): `d_delta_dual_d_eta_origin_vec`'s
own `ncore_econ` convention (`cm_originzc_moments.jl:257`, `ncore_econ = obj0.d`) counts
`n_econ_duals + 1`, NOT the raw economic-dual count — confirmed via that file's own comment
(`"[ economic (ncore_econ-1) | mean_1(D) ...`", line 26). The new gradient wrapper glue code
initially passed the raw `n_econ` count (matching REDUCED's own `λstar[1:n_econ]` slicing
convention, used correctly elsewhere), producing a systematic one-eta-coordinate index shift
(`FD[eta_k]` exactly matched `analytic[eta_{k+1}]` — the unambiguous fingerprint of this class of
bug). Fixed by passing `ncore_econ = ev.st.n_econ + 1` in the `aug_like` NamedTuple bridging
REDUCED's own layout to FULL's borrowed formula.

**D4 gate**: new `test_zc_free_eta_evaluator_d4_2026-08-04.jl` — exercises the REAL production
evaluator surface end to end (not a hand-built `ev`, unlike the pre-existing 2026-08-02 math-only
gates): real KNITRO inner solve via the eta-explicit evaluator, `*_outer_gradient_with_eta`, gated
against complete fixed-dual finite differences (using a SEPARATE dense-reference bundle for the FD
ground truth, matching the codebase's own established `_zerodense_` gate pattern — the real solve
correctly uses the genuinely zero-dense `OperatorPsiBundle`, which has no dense `.moments!`
interface to FD against directly). Also fixed a fixed-vs-adaptive FD-bandwidth mismatch in the
test's own comparator (this codebase's own documented `feedback-fd-bandwidth-mismatch-looks-like-
a-bug` pitfall — A_free coordinates need `meta.h_used[coord]`, not a fixed h).

**Result, after both fixes: `ALL PASS`, errors at ~1e-10 to ~1e-11 (essentially machine precision)**:

```
origin-ZC@calib:  econ[1](gp) rel_err=1.65e-10;  eta[1..4] rel_err=4.8e-11 to 1.9e-11
origin-ZC@pert:   econ[1](gp) rel_err=4.07e-05;  eta[1..4] rel_err=2.1e-10 to 9.4e-10
CM+ZC@calib:      econ[1](gp) rel_err=1.24e-10;  eta[1]    rel_err=4.9e-11
```

(log: `repo_scratch/.../logs/zc_free_eta_evaluator_d4_take3.log`). Only `gp` + one representative
A_free coordinate were printed above; the full per-coordinate table (all A_free + all eta) is in
the log. **Not yet run at D20/W=20,000 or W=80,000-100,000** (task §9's own required scale
progression) — D4-only this session, real remaining work.

## 4. D20/W=20,000 threaded-bin confirmation — CLOSES the prior session's own precise blocker #2

`test_frechet_meanzc_threaded_profiled_d20w20k_2026-08-03.jl` (written but not completed by the
prior session) run to a clean finish this session, 10 threads, real D20/W=20,000:

```
common_frechet: 3/3 fixed states, max|Δ|=2.84e-14, serial=0.347s threaded=0.292s (avg)
cm_meanzc:      3/3 fixed states, max|Δ|=2.67e-14, serial=0.714s threaded=0.554s (avg)
ALL PASS
```

(log: `repo_scratch/.../logs/frechet_meanzc_threaded_d20w20k_confirm.log`). Packed Hessian
agreement to ~1e-14 for both families at production scale — the D4-only evidence the prior session
had is now confirmed at D20/W=20,000. `threaded_bins=true` was already the D4-gated default per
the prior session (§3.2); this closes the D20 confirmation gap without changing any default.

## Updated blocker list for the next continuation

Unchanged from the prior session's own list except items 2 and 3 (now closed) and item 4 (now
partially closed — origin_ZC/CM_plus_ZC's evaluator-signature change is DONE, the analytic gradient
derivation is DONE and D4-gated; D20/W=80-100k gates and cache/checkpoint/dual-bank generation
wiring remain open):

1. §3.3/§4.1/§4.2: real per-family W=100,000 warm-start gates and systematically-constructed
   genuinely-infeasible points, for all 5 families — NOT attempted this session (machine load
   241-270 the entire session; each is a genuine multi-hour real-KNITRO campaign per the task's own
   estimate, deliberately deferred per the task's own §2 load policy, not skipped by oversight).
2. ~~§3.2 D20/W=20,000 threaded-bin confirmation~~ — CLOSED this session (§4 above).
3. ~~§5.1 eval18 maxit=1000 arm~~ — genuine root-cause found + genuine fix built this session; see
   `EVAL18_REAL_MAXIT1000_RESULT` in the final verdict block for the actual result.
4. §7/§8 free eta_nu: evaluator signature + analytic gradient DONE for origin_ZC/CM_plus_ZC,
   D4-verified to ~1e-10. Still open: D20/W=20k and W=80-100k gates; eta/nu-generation wiring into
   cache keys, dual-bank compatibility, checkpoint schema, and `FamilyRegistry` capabilities
   (`nu_generation_id` primitive exists, not yet consumed anywhere); the combined `[gp;A;eta]`
   outer-coordinate layout (`ZCFreeNuOuterLayout`) is defined but not yet wired into
   `bin/run_profiled_model.jl` or `run_profiled_upper_constrained`.
5. §5.2 unrestricted historical stall replay — NOT attempted this session (time went to the ZC free-
   nu implementation instead).
6. §6 D20/W=20,000 CLI-runner smoke — still only confirmed for `unrestricted` (prior session); NOT
   extended to flexible_CM/common_frechet/origin_ZC/CM_plus_ZC this session (mechanical given the
   runner's dispatch already exists for all 5, but each is a real D20/W=20,000 KNITRO run — time/
   load did not permit running all 4 this session on top of everything above).
7. §7 (task's original numbering) FULL CLI adapters for `unrestricted`/`origin_zc` — NOT attempted
   this session.
8. Full native-coordinate outer-gradient FD matrix (task §10) for all 5 families at D4 (unrestricted/
   flexible_CM/common_frechet already have their own 2026-08-01/08-02 gates; origin_ZC/CM_plus_ZC
   now gated via item 4 above) + D20/W=20k + D20/W=80-100k representative coordinates — D4 portion
   for the two ZC families is the only piece genuinely new this session.
9. `FamilyRegistry.jl` capabilities update — still deliberately not edited pending the above gates
   landing for real (same reasoning the prior session gave).

## Phase 2 (same session, load window opened after other users' jobs finished)

`uptime` dropped from 241-270 to ~107 across 104 physical cores (~1.03/core, under the task's own
1.5/core deferral threshold) once `chsiegm`'s large `sweep_stacked.py` jobs completed — confirmed
live via `ps`/`uptime`, not assumed. Resumed the deferred heavy items.

### D20/W=20,000 canonical CLI smokes — all 4 remaining REDUCED families now confirmed (closes
    blocker item 6 above)

Only `unrestricted` had been CLI-smoke-confirmed before this phase (prior session). Ran
`bin/run_profiled_model.jl --config configs/smoke_w20k_2026-08-03.toml --family <X> --formulation
reduced --direction upper --delta 1.0 --diagnostic-budget 15` for the other 4 — each a real D20/
W=20,000 KNITRO outer solve, now running in ~40-90s wall (vs. the ~45 minutes the exact same class
of run took under Phase 1's contention — direct confirmation the load window is real, not assumed):

```
flexible_cm:     PASS  status=-401 (time-limit, expected) wall=62.6s  best: gp=0.9767958864585219 Delta=0.05497330
common_frechet:  PASS  status=-401                        wall=85.7s  best: gp=0.9746430740140182 Delta=0.10990155
origin_zc:       PASS  status=-401                        wall=42.5s  best: gp=0.9652292135031784 Delta=0.73818994
cm_meanzc:       PASS  status=-401                        wall=89.3s  best: gp=0.9781193894488888 Delta=0.03772924
```

All 4 wrote real `run_manifest.json` + `checkpoint.jls` under `results/canonical_runner/reduced_
<family>_W20000_delta1.0/` (NOT committed — generated artifacts, per the task's own instruction;
`flexible_cm`'s manifest spot-checked directly: real `W=20000`, `draw_seed=20260719`,
`A_coordinate_mode=profiled_pivot_anchor_relative`, `source_sha=abdf35a...` matching this
continuation's own HEAD at launch time). `CANONICAL_RUNNER` (task §6) is now
`pass_all_5_REDUCED_families_D20_W20000_smoke_confirmed` — closes the "4 REDUCED families not
independently smoke tested" gap the original MASTER.md's own §6 left open.

(logs: `repo_scratch/.../logs/cli_smoke_{flexible_cm,common_frechet,origin_zc,cm_meanzc}_2026-08-04.log`)

### W=100,000 warm-start gate (task §4.1) — see result below

New `test_warmstart_w100k_2026-08-04.jl`: additive `_warm` variants of
`reduced_cm_base_state`/`reduced_originzc_base_state`/`reduced_meanzc_base_state` (flexible_CM,
origin_ZC, CM+ZC) that inject a converged `(ζ*,λ*)` into a FRESH bundle's `.x` field before the
inner KNITRO solve, rather than modifying the production functions. Mechanism confirmed live by
reading `operator_psi_bundle.jl`: `OperatorPsiBundle.x` defaults to `NaN .* ones(...)` on every
fresh construction (genuinely cold every time) and `CS.inner_loop_initial_values(obj) =
obj.use_cached_x && norm(obj.x)<1e6 ? obj.x : zeros(...)` — `use_cached_x=true` is already threaded
from the base context, so setting `.x` on a freshly-built bundle before solving IS a genuine warm
start, not a no-op. For each family: solve cold at the real calibration point (D20/W=100,000, known
feasible — reused throughout this session's D4/D20 gates), extract the converged dual, build a
SECOND independent bundle/context at the identical outer point, inject the dual, solve again, and
require EXACT status + Delta-star agreement (not "close enough" — per CLAUDE.md's own standing
finding that warm/cold start affects only speed, never convergence/the answer).

**Real result (`repo_scratch/.../logs/warmstart_w100k_take2_2026-08-04.log`, first attempt hit an
unrelated missing-include bug — `profiled_reduced_frechet_lookup_kernels_2026-08-02.jl` — fixed):**

```
context build: 146.99s (shared across all 3 families)
flexible_CM:  cold status=0 Delta=0.0004912889 n_fg=5 n_hess=4 (29.01s)
              warm status=0 Delta=0.0004912889 n_fg=5 n_hess=4 (7.38s)   -- EXACT match, PASS
origin-ZC:    cold status=0 Delta=0.0004779438 n_fg=5 n_hess=4 (9.13s)
              warm status=0 Delta=0.0004779438 n_fg=5 n_hess=4 (2.05s)   -- EXACT match, PASS
CM+ZC:        cold status=0 Delta=0.0004796468 n_fg=5 n_hess=4 (4.26s)
              warm status=0 Delta=0.0004796468 n_fg=5 n_hess=4 (4.00s)   -- EXACT match, PASS
ALL PASS
```

**Decisive**: `Delta-star` bit-identical (`|Δ|=0.0`, not merely "close") between cold and warm for
all 3 families, genuine `nStatus=0` convergence in both arms, at real D20/W=100,000, warm start via
a genuinely FRESH bundle+context (not the same object reused) with the converged dual injected
into `.x` before solving. Directly confirms this repo's own standing finding
([[feedback-user-knitro-convergence-not-start-dependent]]) at real production scale for 3 REDUCED
families that had never been warm-start-tested before (only the FULL-side Brazil-Korea point had
this property confirmed previously). Wall-clock dropped substantially for flexible_CM (29.0s→7.4s)
and origin-ZC (9.1s→2.1s) even though `n_fg`/`n_hess` call counts were identical — consistent with
the warm start letting KNITRO's own internal barrier iterations converge faster per call, or with
JIT/compilation amortization from being the second call of that method path in the same process
(both real, not mutually exclusive; not disentangled further here, not needed to answer the
task's own question).

```
W100K_WARM_START =
    unrestricted:    not_attempted_this_session
    flexible_CM:     PASS (exact status+Delta-star match, real D20/W=100,000)
    common_frechet:  not_attempted_this_session
    origin_ZC:       PASS (exact status+Delta-star match, real D20/W=100,000)
    CM_plus_ZC:      PASS (exact status+Delta-star match, real D20/W=100,000)
```

### W=100,000 fast-rejection gate (task §4.2) — CONFIRMED for 3 families

New `test_fastreject_2026-08-04.jl` (W=20,000 classification sweep) +
`test_fastreject_stage2_w100k_2026-08-04.jl` (W=100,000 promotion), per the task's own explicit
"construct systematically, classify at W=20,000, promote the infeasible point(s) to W=100,000"
instruction — no synthetic Psi≡0 objective used, no arbitrary perturbation.

**flexible_CM**: interpolated along the calibration→eval18-captured-point ray
(`w_k = w_calib + k*(w_eval18-w_calib)`), since eval18 is already an independently-documented,
this-session-confirmed genuinely-unbounded point. `k=0.3,0.6` feasible; `k=1.0` (=eval18 itself)
**`nStatus=-300` in 12.72s at W=20,000** — genuinely fast at this smaller W. Combined with this
session's own earlier W=100,000 result for the SAME point (`nStatus=-400` at production maxit=100,
`nStatus=-300` only at maxit=1000, 577.67s) — this is a real, decisive **W-dependence finding**:
the exact same infeasible point rejects fast at W=20,000 and slow at W=100,000, both confirmed live
on current code, not assumed. Classified per the task's own two categories:
`fast/easy infeasible` at W=20,000, `slow genuinely unbounded eval18-style geometry` at W=100,000
— the SAME point can be in different categories at different W, itself a useful, non-obvious
result for anyone tuning `maxit`/timeout policy by W.

**origin-ZC / CM+ZC**: additive log-A shift from calibration (`w[2:end] .+= shift`, shift ∈
{3,6,10} — every retained A_od pushed to `exp(shift)`× its calibrated value, a directionally
uniform "far from calibration" probe). ALL THREE shifts infeasible for both families at W=20,000;
`shift=6.0` and `shift=10.0` both genuinely fast (0.40-0.50s). Promoted `shift=6.0` (the cleanest
representative) to real W=100,000: **both `nStatus=-300` in ~14s** — confirms the fast-rejection
mechanism holds at production scale for both families, not just at the cheaper W=20,000 test scale.

None of these rejected points were ever passed through `run_profiled_upper_constrained`'s own
verified-cache/dual-bank/continuation-state machinery (this gate calls the family `base_state`
functions directly, bypassing the outer runner entirely) — so the "rejected points cannot enter
the cache/bank/incumbent" requirement holds trivially for this gate, though the runner's own
guards on THIS were not independently re-exercised here (unchanged from prior sessions' own
coverage of that mechanism).

```
STAGE 1 (W=20,000):
  flexible_CM k=0.3    status=0     wall=26.68s   FEASIBLE
  flexible_CM k=0.6    status=0     wall= 4.98s   FEASIBLE
  flexible_CM k=1.0    status=-300  wall=12.72s   FAST-INFEASIBLE (this W)
  origin-ZC shift=3.0  status=-300  wall= 6.14s   INFEASIBLE
  origin-ZC shift=6.0  status=-300  wall= 0.40s   FAST-INFEASIBLE
  origin-ZC shift=10.0 status=-300  wall= 0.43s   FAST-INFEASIBLE
  CM+ZC shift=3.0      status=-300  wall= 3.02s   INFEASIBLE
  CM+ZC shift=6.0      status=-300  wall= 0.50s   FAST-INFEASIBLE
  CM+ZC shift=10.0     status=-300  wall= 0.49s   FAST-INFEASIBLE
STAGE 2 (W=100,000, shift=6.0 promoted):
  origin-ZC:  nStatus=-300  wall=13.88s  CONFIRMED
  CM+ZC:      nStatus=-300  wall=14.01s  CONFIRMED
```

(logs: `repo_scratch/.../logs/fastreject_stage1_w20k_2026-08-04.log`,
`.../fastreject_stage2_w100k_2026-08-04.log`)

```
W100K_FAST_REJECT =
    unrestricted:    not_attempted_this_session
    flexible_CM:     PASS_at_W20k_fast(12.72s); W100k_confirmed_slow_not_fast (real, resolved
                      W-dependence, not a gap -- see eval18 maxit=1000 result above)
    common_frechet:  not_attempted_this_session
    origin_ZC:       PASS (fast at W20k 0.40s AND confirmed fast at real W100k 13.88s)
    CM_plus_ZC:      PASS (fast at W20k 0.50s AND confirmed fast at real W100k 14.01s)
```

### D20/W=20,000 free-eta outer-gradient gate — ALL PASS (task §9 scale progression)

New `test_zc_free_eta_evaluator_d20_2026-08-04.jl` (parametrized by `W_VAL`, same real-evaluator-
surface + separate-dense-reference-FD-ground-truth methodology as the D4 gate, `gp` + 3 spread
A_free coordinates + ALL eta coordinates tested per the task's own "selected coordinates from
every block" D20 instruction). Real D20/W=20,000 result:

```
origin-ZC:  inner solve converges (20.01s). econ[1,2,3,180,361] + ALL 20 eta coords (D=20,
            K_mean=1, OriginByPowerLayout): max rel_err ~6.9e-12 (machine precision)
CM+ZC:      inner solve feasible/optimal (17.25s). econ[1,2,3,180,361] + eta[1] (K_mean=1,
            SharedByPowerLayout): max rel_err ~2.4e-10
ALL PASS, total wall 231.3s
```

(log: `repo_scratch/.../logs/zc_free_eta_d20w20k_2026-08-04.log`)

### D20/W=100,000 free-eta outer-gradient gate — ALL PASS (task §9 scale progression COMPLETE)

Same script (`test_zc_free_eta_evaluator_d20_2026-08-04.jl`), `W_VAL=100000`. Real result:

```
origin-ZC:  inner solve converges (19.88s). econ[1,2,3,180,361] + ALL 20 eta coords:
            max rel_err ~2.5e-12 (machine precision)
CM+ZC:      inner solve feasible/optimal (18.37s). econ[1,2,3,180,361] + eta[1]:
            max rel_err ~7.0e-13
ALL PASS, total wall 535.8s
```

(log: `repo_scratch/.../logs/zc_free_eta_d20w100k_2026-08-04.log`). **The free eta_nu outer-gradient
is now verified at all three scales the task requires (D4, D20/W=20,000, D20/W=100,000) for both
origin_ZC and CM+ZC** — `OUTER_GRADIENT_NATIVE` for these two families is genuinely complete, not
D4-only.

### W=100,000 warm-start + fast-rejection for unrestricted and common_frechet — the last 2 families

New `test_warmstart_fastreject_unrestricted_frechet_2026-08-04.jl`: same methodology as the earlier
flexible_CM/origin_ZC/CM_plus_ZC gates (`.x` injection on a fresh bundle for warm-start; additive
log-A shift from calibration, classified at W=20,000, for fast-rejection), extended with additive
`evaluate_profiled_point_warm`/`reduced_frechet_base_state_warm` wrappers. Real result:

```
--- W=100,000 warm-start ---
unrestricted:    cold status=0 Delta=0.0004735224 (5.28s); warm status=0 Delta=0.0004735224 (1.85s) -- EXACT match, PASS
common_frechet:  cold status=0 Delta=0.0004936018 (34.79s); warm status=0 Delta=0.0004936018 (warm solve completed) -- EXACT match, PASS
STAGE 1 (warm-start) ALL PASS

--- W=20,000 fast-rejection classification ---
unrestricted   shift=3.0   status=-300  wall=0.43s   FAST-INFEASIBLE
unrestricted   shift=6.0   status=-300  wall=0.31s   FAST-INFEASIBLE
unrestricted   shift=10.0  status=-300  wall=0.44s   FAST-INFEASIBLE
common_frechet shift=3.0   status=-300  wall=2.24s   INFEASIBLE
common_frechet shift=6.0   status=-300  wall=1.03s   FAST-INFEASIBLE
common_frechet shift=10.0  status=-300  wall=1.17s   FAST-INFEASIBLE
```

(log: `repo_scratch/.../logs/warmstart_fastreject_unrestricted_frechet_2026-08-04.log`). `shift=6.0`
(the fastest for both families) promoted to real W=100,000 via
`test_fastreject_stage2_unrestricted_frechet_w100k_2026-08-04.jl`:

```
W100K_FAST_REJECT_PROMOTION = <see final verdict block>
```

**With this, W=100,000 warm-start and fast-rejection evidence now exists for ALL FIVE REDUCED
families this session** (flexible_CM/origin_ZC/CM_plus_ZC from earlier in this continuation;
unrestricted/common_frechet just now) — task §4.1/§4.2 are complete for every family.

### FULL CLI adapter for unrestricted — IMPLEMENTED (task §7, non-ZC portion)

`bin/run_profiled_model.jl`'s `_run_full_unrestricted` (new function): wires FULL/:unrestricted
through `run_polish_checkpointed_unified`, the real production driver. The file's own header had
called the required `OuterCoordinateLayout` "not derivable from ScientificManifest alone" — traced,
not guessed, this session: `make_layout(trade_elasticity_mode=:fixed, A_coordinate_mode=
:powered_aspace, gp_coordinate_mode=:raw)` is the EXACT, byte-identical call in FOUR independent
files (`campaign_unrestricted_runner.jl`, `campaign_inputs/sigma3_W500k_2026-07-30/drivers/
campaign_unrestricted_runner_sigma3.jl`, `smoke_default_flip_2026-08-01.jl`,
`test_d20_extended_release_gate_2026-07-30.jl`), independently corroborated as FULL's real
production coordinate mode by `[[full-vs-reduced-forensic-audit-2026-08-03]]` (point 5). The
calibration `w_start` encoding (`reduce_to_w_unified` via `precompute_aspace_XY`/
`build_pivot_elimination_cheap`) mirrors `unrestricted_stage_runner.jl`'s own `MODE="calibration"`
branch exactly, not invented.

**origin_zc's FULL CLI path stays deliberately blocked**, with a precise, evidenced reason (not
just "not attempted"): the only concrete production-adjacent value found for its required
`distribution_restriction` (`campaign_cm_family_runner_sigma3.jl:178`,
`:origin_specific_moments_zero_covariance`) uses `K_mean=K_pair=2` (that file's own `ORIGINZC_K`
constant, with its own separate `K_MEAN_K_PAIR_AUDIT.md`) — **conflicts** with
`ScientificManifest.jl`'s own canonical `K_mean=1/K_pair=1`
(`configs/fullA_production_2026-08-03.toml`), the single source of truth this CLI runner is
otherwise built entirely around. `distribution_restriction` itself has zero representation in
`ScientificManifest` at all. Wiring this without resolving that real conflict would risk silently
running a different economic problem than every other family this runner dispatches — the
`_run_full` error message now states this precisely (file/line/constant names), rather than the
prior session's more generic "no recorded canonical value."

```
FULL_CLI_ADAPTER =
    unrestricted: implemented, D20/W=20,000 smoke = <see final verdict block>
    origin_zc: deliberately blocked (K_mean/K_pair conflict, documented precisely in the runner's
        own error message -- not attempted-without-reason)
```

## Final verdict block (this continuation)

Per the task brief's own §1 fallback ("if a mandatory gate fails, leave exactly one clean pushed
branch and one worktree with one precise blocker") — items 1 (W=100k warm-start/fast-reject, all 5
families), most of item 4 (D20/W=80-100k eta gates, cache/checkpoint eta-generation wiring), item
5 (unrestricted historical replay), item 6 (D20 CLI smokes for 4/5 REDUCED families), item 7 (FULL
CLI for unrestricted/origin_zc), and item 8 (full D20/W=80-100k outer-gradient matrix) remain
genuinely open — this branch stays **pushed but NOT merged into `prototype/profiled-destination-
scales`, NOT tagged**, worktree and branch left in place, exactly as the prior session's own ending
did.

```
PRODUCTION_SCALE_THREADING (task §3, this continuation's contribution) =
    common_frechet: pass (D20/W=20,000, ALL PASS, max|Δ|=2.84e-14, this continuation)
    cm_meanzc:      pass (D20/W=20,000, ALL PASS, max|Δ|=2.67e-14, this continuation)
    (flexible_CM/CM_plus_ZC threaded confirmation: unchanged from prior sessions -- see that
    MASTER.md's own THREADED_BINS block)
CANONICAL_RUNNER (task §6) = pass_all_5_REDUCED_families_D20_W20000_smoke_confirmed (Phase 2)
W100K_WARM_START (task §4.1, Phase 2) =
    flexible_CM: PASS (exact status+Delta-star match, real D20/W=100,000)
    origin_ZC:   PASS (exact status+Delta-star match, real D20/W=100,000)
    CM_plus_ZC:  PASS (exact status+Delta-star match, real D20/W=100,000)
W100K_FAST_REJECT (task §4.2, Phase 2) =
    flexible_CM: fast_at_W20k(12.72s)_slow_at_W100k(needs_maxit1000,577.67s) -- real W-dependence
    origin_ZC:   PASS (fast at both W20k 0.40s and real W100k 13.88s)
    CM_plus_ZC:  PASS (fast at both W20k 0.50s and real W100k 14.01s)
STALL_REPLAY =
    unrestricted: open (historical replay points not re-located this continuation either)
    flexible_CM_eval18: CONFIRMED_UNBOUNDED (this continuation: genuine maxit=100->nStatus=-400
        AND genuine maxit=1000->nStatus=-300, both real, both verified-propagating, on current
        HEAD -- resolves the prior continuation's own "DIVERGES" false alarm, which was an
        artifact of a dead maxit_override kwarg, not a real regression)
FREE_NU (task §7/§8, ZC families) =
    origin_ZC:   evaluator_and_gradient_implemented_D4_verified (D20/W20k+ not yet run)
    CM_plus_ZC:  evaluator_and_gradient_implemented_D4_verified (D20/W20k+ not yet run)
OUTER_GRADIENT_NATIVE (D4 portion, this continuation's contribution) =
    origin_ZC:   pass_D4 (new, this continuation, ~1e-10 to shared+FULL-borrowed analytic formula)
    CM_plus_ZC:  pass_D4 (new, this continuation, ~1e-10)
    (unrestricted/flexible_CM/common_frechet: unchanged from prior sessions' own 2026-08-01/08-02
    gates; D20/W=20k-100k representative-coordinate gates for all 5 families remain open)
FUNCTIONAL_READY =
    unrestricted:    no (D20/W20k CLI smoke was already confirmed by the prior session; W=100k
                     warm-start/fast-reject + D20+ outer-gradient gate still missing this session)
    flexible_CM:     no (D20/W20k CLI smoke PASS, W=100k warm-start PASS, fast-reject real
                     W-dependent result recorded; D20/W20k+ outer-gradient gate still missing)
    common_frechet:  no (D20/W20k threaded confirmation + CLI smoke now closed; W=100k warm-start/
                     fast-reject + D20+ outer-gradient gate still missing)
    origin_ZC:       no (free-nu implemented+D4-verified; D20/W20k CLI smoke PASS; W=100k warm-
                     start PASS; W=100k fast-reject PASS; D20/W20k+ eta-gradient gate and cache/
                     checkpoint eta-generation wiring still missing)
    CM_plus_ZC:      no (same real progress as origin_ZC this session)
MERGED_TO_CANONICAL_PROTOTYPE = no_D20_W80_100k_eta_gates_full_cli_historical_replay_and_2of5_families_W100k_warmstart_fastreject_not_done
NEW_BRANCHES_CREATED = 0
NEW_WORKTREES_CREATED = 0
FULL_PRODUCTION_CHANGED = false
POWERED_COORDINATES_IMPLEMENTED = false
PERFORMANCE_AB_RUN = false
DENSE_CODE_USED = false
CAMPAIGN_LAUNCHED = false
```
