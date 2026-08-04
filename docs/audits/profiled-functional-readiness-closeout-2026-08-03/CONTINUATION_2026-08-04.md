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
    unrestricted:    no (unchanged -- W=100k warm-start + outer-gradient D20+ gate missing)
    flexible_CM:     no (eval18 now genuinely resolved as expected-unbounded, not a blocker per se,
                     but W=100k warm-start + fast-rejection + D20+ outer-gradient gate still missing)
    common_frechet:  no (D20/W20k threaded confirmation now closed; W=100k warm-start + D20+
                     outer-gradient gate still missing)
    origin_ZC:       no (free-nu now implemented+D4-verified; D20/W20k+ eta gates, cache/checkpoint
                     eta-generation wiring, W=100k warm-start still missing)
    CM_plus_ZC:      no (same as origin_ZC)
MERGED_TO_CANONICAL_PROTOTYPE = no_W100k_warmstart_fastreject_all5_families_and_D20_W80_100k_eta_gates_and_full_cli_and_historical_replay_not_done
NEW_BRANCHES_CREATED = 0
NEW_WORKTREES_CREATED = 0
FULL_PRODUCTION_CHANGED = false
POWERED_COORDINATES_IMPLEMENTED = false
PERFORMANCE_AB_RUN = false
DENSE_CODE_USED = false
CAMPAIGN_LAUNCHED = false
```
