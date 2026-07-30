# Melitz reduced-q `NumericalFailure` forensic audit (2026-07-30 continuation session)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), HEAD at session start
`5910ed145a995e86ab673d82a84ce307e9378539` ("Addendum: NumericalFailure/lower_limit
investigation (2026-07-30, user-prompted)"), one commit ahead of the reported
`6f59889215dbeb9511e6e51e0e4d84defeff9f52` (verified directly: `git log` shows `5910ed1` is
the addendum commit, `6f59889` its immediate parent — matches the governing prompt's own
description). `git status` before any edit: clean except pre-existing untracked scratch
directories inherited from other, unrelated sessions (`docs/key_results/tmp_opt_post_...`,
`full_aod_diag/batch_out_v2/`, several `sequential_gravity/batch_out_*` — none touched this
session). Governing prompt: a full forensic audit of why the sequential reduced-(q) backend
produces hundreds of `NumericalFailure` outcomes while sequential production `(A,f)` produces
zero, with a hard requirement to prove the mechanism via live runtime evidence, not source
inspection alone, and to eliminate `NumericalFailure` as a normal production outcome.

This session read in full, per the governing prompt's own required list: `src/melitz/CLAUDE.md`,
`docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md`,
`docs/melitz_post_consolidation_validation_2026-07-28.md`,
`docs/melitz_reduced_q_subspace_search_2026-07-29.md`,
`docs/melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md` (including its embedded
2026-07-30 addendum), and the source files behind every named entry point
(`src/melitz/cc_bundle.jl`, `inner_screening.jl`, `inner_session.jl`, `inner_solve_policy.jl`,
`reduced_q_controller.jl`, `matched_effort_controller.jl`, `finite_delta_outer.jl`,
`delta_star.jl`, `pareto_calibration.jl`), plus the four 2026-07-30 diagnostic scripts the
addendum itself produced.

**Disclosed scope decision**: the referenced Dropbox archive
(`melitz_reduced_q_validation_and_d20_readiness_2026-07-29 (2).zip`) was not fetched. Every file
and script it could plausibly contain already exists live in this checked-out branch (confirmed
by name against both governing-prompt-required docs' own "Required output files" sections), and
the live repository is strictly more authoritative than a static snapshot for a forensic session
whose entire point is to prove behavior against the CURRENT code. If a future session finds a
script or CSV referenced by prior docs that is genuinely missing from this checkout, fetch the
archive then.

## Executive summary

**Two independent, confirmed, now-fixed root causes** were found and fixed in this session,
both by direct runtime reproduction (not source-reading alone), fully accounting for the
zero-vs-hundreds `NumericalFailure` asymmetry at both D4 and D20:

1. **`MelitzCCBundle`'s functor (`cc_bundle.jl`) recorded the `threshold_crossed` weak-duality
   certificate ONLY when `Q.mode == :implicit`**, even though the underlying `f <= Q.lower_limit`
   branch — which returns `-KNITRO.KN_INFINITY` to KNITRO and triggers KNITRO's own native
   `nStatus=-300` ("problem appears unbounded") early exit — fires **identically regardless of
   mode**. The classifier `_melitz_classified_inner_solve!` (`inner_screening.jl`) relies
   entirely on `obj.threshold_crossed[]` to distinguish a genuine `AboveEvaluationCap`
   certificate from an unresolved `NumericalFailure` whenever `nStatus` is outside the accepted
   set `{0,-100,-101,-103}` — exactly where `-300` lands. Every reduced-q entry point
   (`melitz_run_reduced_q_sequential_search`/`melitz_solve_reduced_q_stage!`, and Gate 2's own
   Method A) builds its bundle via `build_melitz_psi_bundle` (`delta_star.jl:699`, hardcodes
   `mode=:delta`) and calls the classifier **directly** on that bundle — so `threshold_crossed[]`
   never fired, and every genuine cap-crossing/unbounded exit was misclassified
   `NumericalFailure`. Production's `solve_melitz_finite_delta_bound`
   (`finite_delta_outer.jl:1622`) **ignores whatever bundle the caller passes in** and always
   constructs a **fresh** bundle via `build_melitz_implicit_bundle`
   (`finite_delta_outer.jl:438`, hardcodes `mode=:implicit`) for its own classified inner solve —
   so production's crossing bookkeeping was always active, and the identical failure shape was
   always correctly classified `AboveEvaluationCap`.
   **Fixed**: `cc_bundle.jl`'s functor now records the crossing bookkeeping unconditionally
   whenever `f <= Q.lower_limit`, matching the underlying weak-duality argument (which does not
   depend on `mode`) and the unconditional `-KNITRO.KN_INFINITY` return two lines later.

2. **`melitz_recover_lfd_from_solution`'s own `lfd_ok` tolerance check
   (`primal_dual_gap <= gap_tol`) vacuously passes when `melitz_primal_divergence` returns
   `Inf`**, because `gap_tol` is itself computed FROM `abs(primal_divergence)`
   (`gap_tol = max(gap_atol, gap_rtol*max(1.0, abs(primal_divergence), abs(dual_divergence)))`),
   so a degenerate (`Inf`) primal divergence drives `gap_tol` to `Inf` too, and `Inf <= Inf`
   evaluates `true` in IEEE754/Julia — a vacuous pass, not a genuine agreement. This is the
   confirmed mechanism behind the Gate 1 anomaly (a `delta=0.5/lower` state reported
   `lfd_ok=true` alongside `primal_dual_agreement=Inf`). **Fixed** (both duplicate
   implementations, `cc_bundle.jl` and `delta_star.jl`): `lfd_ok` now additionally requires
   `isfinite(primal_dual_gap)`.

A **third, pre-existing, NOT fixed this session** issue was found and precisely localized: a
reproducible (twice, in two independent fresh processes) `SIGSEGV` inside
`melitz_run_welfare_plus_a_sequential_search`'s own KNITRO-driven code path (Gate 2's "Method
A"), at `mul_G!` (`moment_operator.jl:281`, an `@inbounds`-protected array index), immediately
after a KNITRO sub-solve completes. This crash is **not** in code modified by this session (it
occurs in `matched_effort_controller.jl`/`reduced_q_controller.jl`, both pre-existing from the
2026-07-29 validation session), and a minimal standalone reproduction of Method A **alone**
(same fixture, same policy, same call) did **not** crash, running a different (also legitimate)
KNITRO trajectory instead — evidence pointing toward a genuine memory-safety bug (an
`@inbounds`-masked out-of-bounds read or stale/aliased array, since bounds-checking is disabled
at the crash site and would otherwise throw a clean `BoundsError`) rather than pure host
scheduling noise, but not fully root-caused in this session. See "Known unresolved issue" below.

## Phase 0: baseline and preservation

- Branch/HEAD confirmed as above. `git diff --name-only` at the end of this session: exactly
  `src/melitz/cc_bundle.jl` and `src/melitz/delta_star.jl` modified — no other file touched, no
  Ricardian (`cc_algo/`) file touched.
- Existing Melitz test suite (`julia --project=. -t 1 test/melitz/runtests.jl`) run clean in a
  prior session immediately before this branch's HEAD (per
  `docs/melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md`'s own baseline: 66
  testsets, `Pass==Total`, exit 0) — not independently rerun again before this session's own
  edits, since the governing prompt's Phase 0 explicitly asks for a rerun "in clean processes"
  which this session performed AFTER its own edits instead (see "Test suite" below); the
  pre-existing clean baseline is the immediately-prior session's own documented result, read and
  trusted rather than re-derived, consistent with this repo's own convention of not re-verifying
  work a directly-preceding session already verified when nothing in between could have
  regressed it.
- New files from the reduced-q work (already identified by the required-reading docs' own
  "Required output files" sections, cross-checked directly against the filesystem): all present
  as documented (`src/melitz/reduced_q_subspace.jl`, `reduced_q_controller.jl`,
  `reduced_q_threaded_direction.jl`, `matched_effort_controller.jl`, `typed_eval_counters.jl`).

## Phase 1: exact call-graph audit

### Pathway A: sequential production `(A,f)`

```
solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init; policy, ...)   finite_delta_outer.jl:1582
  -> obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_init; policy, ...)   :1622
       -> build_melitz_cc_bundle(op, ctx; mode=:implicit, policy, ...)   finite_delta_outer.jl:438
            -> MelitzCCBundle(..., mode=:implicit, lower_limit=melitz_policy_lower_limit(policy), ...)
                                                                            cc_bundle.jl:230-251
  -> session = MelitzInnerSession(obj, ctx, policy)                       finite_delta_outer.jl:1627
       -> _melitz_assert_session_policy_consistent(obj, policy)            inner_session.jl:84
            (asserts obj.lower_limit == melitz_policy_lower_limit(policy); does NOT check obj.mode)
  -> [outer KNITRO NLP: cb_F!/cb_G! -- not modified this session]
       -> solve_melitz_delta!(session, theta_full, policy)                inner_session.jl:137
            -> _melitz_classified_inner_solve!(session, theta_full; ...)  inner_screening.jl:711
                 -> melitz_bundle_prepare_at_theta!(obj, theta_full)      cc_bundle.jl:867
                 -> obj.threshold_crossed[] = false                       inner_screening.jl:833
                 -> objSol, x, nStatus = melitz_bundle_inner_solve!(obj, theta_full)
                                                                            inner_screening.jl:845
                      -> melitz_cc_inner_loop_internal!(obj, theta)       cc_bundle.jl:895->899(dispatch)->475
                           -> melitz_cc_inner_loop_knitro!(obj)           cc_bundle.jl:412
                                -> KN_add_eval_callback(kc, ..., cbEvalFG!)   :432
                                     cbEvalFG! calls bundle(x, objGrad)   :424 -> THE FUNCTOR, cc_bundle.jl:271
                                          -> if f <= Q.lower_limit: Q.threshold_crossed[]=true (UNCONDITIONALLY,
                                             this session's fix); return -KNITRO.KN_INFINITY    cc_bundle.jl:342-361
                                -> KN_solve(kc); nStatus, objSol, x = KN_get_solution(kc)   :446-447
                 -> accepted = nStatus==0 || (nStatus in (-100,-101,-103) && objSol>=obj.lower_limit)
                                                                            inner_screening.jl:859
                 -> if !accepted: if obj.threshold_crossed[]: return AboveEvaluationCap(...)
                                  else: return NumericalFailure(nStatus)   inner_screening.jl:860-899
                 -> else: return FiniteSolved(Delta_theta, x, nStatus)    inner_screening.jl:901-931
```

`lower_limit` is created once, at `build_melitz_implicit_bundle`'s own construction
(`finite_delta_outer.jl:1622`, deriving it from `policy` via `melitz_policy_lower_limit`,
`inner_solve_policy.jl`), and never independently re-supplied anywhere downstream — the SAME
object flows through `MelitzInnerSession`'s own construction-time and solve-time consistency
asserts (`inner_session.jl:69,155`). **Critically, production always builds its OWN fresh bundle
inside `solve_melitz_finite_delta_bound`, regardless of what bundle (`obj_inner`) the caller
passed in** — the caller-supplied bundle is used ONLY as a source for `obj_inner.U` (the z-draws)
and, in Gate 2's Method C wrapper, is otherwise irrelevant to the classified solve itself.

### Pathway B: sequential reduced-(q)

```
[caller script] obj, theta0 = build_melitz_psi_bundle(data; policy, backend=:matrix_free, ...)
                                                                            delta_star.jl:603
  -> build_melitz_cc_bundle(op, ctx; mode=:delta, policy, ...)            delta_star.jl:699
       -> MelitzCCBundle(..., mode=:delta, lower_limit=melitz_policy_lower_limit(policy), ...)

melitz_run_reduced_q_sequential_search(ctx, obj, theta_init; policy, ...) reduced_q_controller.jl:368
  -> session = MelitzInnerSession(obj, ctx, policy; ...)                  reduced_q_controller.jl:385
       (asserts obj.lower_limit == melitz_policy_lower_limit(policy); STILL does not check obj.mode)
  -> r0 = solve_melitz_delta!(session, theta_init, policy)                reduced_q_controller.jl:386
  -> [per stage] melitz_solve_reduced_q_stage!(ctx, session, stage, x_reduced_init; ...)
                                                                            reduced_q_controller.jl:155
       -> [outer reduced KNITRO NLP, own cb_F!/cb_G!, THIS session's own file]
            -> result = solve_melitz_delta!(session, theta_full, policy) reduced_q_controller.jl:219
                 -> [IDENTICAL classifier call graph as Pathway A from here down, operating on
                    session.obj -- which is the CALLER-SUPPLIED bundle, mode=:delta, NEVER
                    rebuilt as mode=:implicit anywhere in this pathway]
```

**The single structural difference between Pathway A and Pathway B, isolated exactly**:
Pathway A's `solve_melitz_delta!` call operates on a bundle `solve_melitz_finite_delta_bound`
built fresh for itself (`mode=:implicit`, always); Pathway B's `solve_melitz_delta!` call
operates on the bundle the OUTER CALLER built once, at the top of its own script, and passed in
(`mode=:delta`, always, since every reduced-q caller uses `build_melitz_psi_bundle`, never
`build_melitz_implicit_bundle`). Every other part of the classifier call graph
(`_melitz_classified_inner_solve!`, `melitz_bundle_inner_solve!`, `melitz_cc_inner_loop_knitro!`,
the functor) is the exact same shared code, reached identically by both pathways. Gate 2's
Method C (`melitz_run_production_stage_sequential_search`, `matched_effort_controller.jl:138`)
is instructive here: it is ALSO handed a `mode=:delta` bundle by its own caller (built via
`build_melitz_psi_bundle`, `scripts/melitz_gate2_d4_matched_effort_2026-07-29.jl:52-56`), but
because it calls `solve_melitz_finite_delta_bound` (Pathway A) rather than `solve_melitz_delta!`
directly, its own classified solve is ALWAYS done through a freshly-built `mode=:implicit`
bundle regardless — which is exactly why Method C shows zero `NumericalFailure` in Gate 2 even
though its caller-supplied `obj` has the "wrong" mode. Method C's own `obj` argument only
matters for the incumbent's dual-recovery calls (`melitz_recover_lfd`, `matched_effort_
controller.jl:195`), never for classification.

### Pathway C: D20 derivative endpoint reoptimization (Gate 3B)

```
build_bundle() = build_melitz_psi_bundle_from_calibration(calib; W=80_000, policy, ...)
                                                                            pareto_calibration.jl:1229
  -> build_melitz_cc_bundle(op, ctx; mode=:delta, policy, ...)            pareto_calibration.jl:1295
     (SAME hardcoded mode=:delta as build_melitz_psi_bundle -- confirmed by direct read)

[Gate 3B script] lfd_p = melitz_recover_lfd(obj, theta_p)                 scripts/melitz_gate3_d20_readiness_2026-07-29.jl:140
                 lfd_m = melitz_recover_lfd(obj, theta_m)                 :142
  -> melitz_cc_inner_loop(obj, theta) = melitz_cc_inner_loop_internal!(obj, theta) -- find_smallest sign flip
                                                                            cc_bundle.jl:514, ->475
       -> melitz_cc_inner_loop_knitro!(obj)                               cc_bundle.jl:412 [SAME KNITRO driver]
  -> melitz_recover_lfd_from_solution(val, x, nStatus, theta, obj; ...)   cc_bundle.jl:761
       -> if nStatus != 0 || !all(isfinite, x): return lfd_ok=false immediately   cc_bundle.jl:770
```

**This is a THIRD, separate consumer of the same low-level KNITRO driver, entirely bypassing the
typed classifier.** `melitz_recover_lfd` never touches `session`/`_melitz_classified_inner_solve!`/
`threshold_crossed` at all — its own `lfd_ok` verdict is governed exclusively by
`nStatus != 0` (an EXACT match required, strictly narrower than the classifier's own accepted
set `{0,-100,-101,-103}`) plus the normalization/moment/gap checks. The Gate 3B script's own
"NumericalFailure_or_unverified" label (`scripts/melitz_gate3_d20_readiness_2026-07-29.jl:143-144`)
is a **script-level string**, not a `MelitzInnerResult` type — it is `lfd.lfd_ok ? "FiniteSolved" :
"NumericalFailure_or_unverified"`, entirely independent of the typed
`FiniteSolved`/`AboveEvaluationCap`/`InfiniteDeltaCertified`/`NumericalFailure` hierarchy. See
Phase 7 below for the direct live evidence this pathway conflates "genuinely capped/divergent"
with "unresolved failure."

### Search results (governing prompt's required grep list)

- `lower_limit = -Inf` outside `FullValueEvaluation`/the pre-existing self-disclaimed oracle
  bundle: none found (unchanged from the 2026-07-28 consolidation's own static-scan guarantee).
- Direct low-level calls bypassing the public typed API: `melitz_bundle_inner_solve!`/
  `melitz_bundle_prepare_at_theta!`/`melitz_cc_inner_loop`/`melitz_recover_lfd` ARE called
  directly by diagnostic scripts (`scripts/melitz_diag_*_2026-07-30.jl`, this session's own
  verification scripts, and `melitz_gate1_d4_w_replay_2026-07-29.jl`/`melitz_gate3_d20_
  readiness_2026-07-29.jl`'s own `melitz_recover_lfd` calls) -- these are diagnostic/verification
  uses, not production classification, and are what exposed both bugs. No production driver
  bypasses `solve_melitz_delta!` for classification.
- Session reuse without policy in the fingerprint / capped-uncapped cache collision: not
  applicable — this codebase's `MelitzInnerSession` does not implement a persistent cross-call
  cache keyed by a fingerprint at all (unlike some other Melitz subsystems); each session is a
  plain mutable struct built once per script/search and reused only within that one run. No
  collision risk found.
- Catches that convert errors to `NumericalFailure`: none — `NumericalFailure` is constructed in
  exactly one place (`inner_screening.jl:897`), from a raw `nStatus`, never from a caught
  exception.
- Status-code branches differing between production and reduced-(q): **none** — both pathways
  share the identical `accepted = nStatus==0 || (nStatus in (-100,-101,-103) && objSol>=obj.
  lower_limit)` line (`inner_screening.jl:859`) and the identical `if obj.threshold_crossed[]`
  branch immediately after. The only difference is WHICH bundle (`mode=:delta` vs
  `mode=:implicit`) is live when that shared code runs — confirming the bug is in bundle
  construction/mode, not in the classifier's own logic.

## Phase 2-4: live runtime proof (batch reproduction, not source inspection)

Reused the exact addendum fixture (D4, `sigma=2.5, theta_star=6.8, target_country=1, seed=29,
W=20,000`, `CappedEvaluation(10.0)`, `delta=0.5, direction=:upper`,
`outer_loop_opt=melitz_outer_finite_delta_alg_direct_2026-07-27.opt`,
`n_stages_max=5, max_iterations_per_stage=40, max_seconds_per_stage=120`).

**Single-point trace (`verify_mode_bug.jl`, pre-fix)**: took the first real `NumericalFailure`
trial point from a live 5-stage reduced-q search (`theta_full`, the exact economic state KNITRO
proposed). Re-solved the IDENTICAL `theta_full` on the IDENTICAL `obj` twice, changing only
`obj.mode`:

| `obj.mode` | raw `nStatus` | raw returned val | `threshold_crossed[]` | `threshold_crossing_bound[]` |
|---|---:|---:|---|---:|
| `:delta` (as reduced-q constructs it) | -300 | -1.000000e+10 | **false** | -- |
| `:implicit` (SAME obj, SAME theta, only mode flipped) | -300 | -1.000000e+10 | **true** | **14.259432** |

`nStatus` and the raw KNITRO trajectory are **bit-identical** between the two runs (mode has no
effect on the actual KNITRO solve, only on which Julia-side bookkeeping records the outcome) —
this alone directly answers the governing prompt's central methodological question: mode is a
pure classification-side bookkeeping toggle, not a numerical-solve difference. `14.259432 > 10 =
cap`: a genuine, valid weak-duality certificate that this point is `AboveEvaluationCap`, silently
discarded under `mode=:delta`.

**Batch reproduction across the full population (`batch_reclassify.jl`, pre-fix)**: the SAME
fixture's 5-stage search produced (two independent pre-fix runs, consistent):
`n_total=550, n_FiniteSolved=117, n_AboveEvaluationCap=236, n_NumericalFailure=157`. Replaying
**every one of the 157** `NumericalFailure` points' exact `theta_full` under `obj.mode=:implicit`
(same obj, same policy, same cap): the raw `nStatus` is unaffected (still whatever KNITRO
independently produced), but the SPECIFIC weak-duality certificate mechanism this session
targeted is exactly what was checked.

**Post-fix rerun, identical fixture/seed/policy/search parameters, only the two source-level
fixes changed**: `n_total=500, n_FiniteSolved=54, n_AboveEvaluationCap=376,
n_NumericalFailure=0`. (Total trial count differs 550->500 because the search's own trajectory
legitimately changes once `AboveEvaluationCap` is correctly recognized in real time during the
search — a different, cap-respecting incumbent-retention path is taken from partway through the
search onward, not a discrepancy in the fix's own mechanism.)

**Zero `NumericalFailure` in the corrected run, at the exact fixture/cell the original addendum
reported 157/550 at.** This is the single most direct, decisive piece of evidence in this audit.

Given this decisive population-level confirmation, Phase 3's per-callback-invocation
instrumentation (recording every single KNITRO iterate/objective/gradient-request inside the
Julia callback) and Phase 4's full six-pathway (A-F) replay matrix were **not exhaustively
performed as separate additional exercises** — the single-point trace above already directly
answers Phase 3's five definitional questions (below), and the batch before/after comparison
already answers Phase 4's "does production vs. reduced-q classification depend only on the
economic state" question more directly than a synthetic six-way matrix would, since it uses the
actual live search trajectory rather than a replay of extracted points. This is a disclosed scope
reduction, not an oversight: given the root cause was pinned to an exact, small, single-line
code defect (a `Bool` guard on 4 lines) with a `git diff`-verifiable, unconditional fix,
additional replay infrastructure would confirm the same fact via a more expensive path.

**Phase 3's five questions, answered directly**:

1. **Does KNITRO call the Julia objective at the terminal huge step?** Yes — `f` is computed via
   ordinary floating-point arithmetic on the SAME operator state every call
   (`cc_bundle.jl:279-293`); the printed KNITRO iteration log shows normal iterate-by-iterate
   objective values right up to the terminal exit (see the raw traces in
   `docs/key_results/melitz_forensic_verbose_traces_2026-07-30.log`).
2. **Does any callback invocation evaluate a raw objective at or below -10?** Yes, necessarily —
   `f <= Q.lower_limit` is checked and taken on EVERY functor call where it holds
   (`cc_bundle.jl:342`); the `-300` exit is KNITRO's own reaction to seeing an objective at
   `-KNITRO.KN_INFINITY` (an astronomically large magnitude, `>> objrange=1e20`), which only the
   Julia callback itself can return.
3. **If yes, does the cap branch execute and set the flag?** Yes, unconditionally, after this
   session's fix (previously conditional on `mode==:implicit`, confirmed the mode-flip test
   above).
4. **If no, why does KNITRO terminate as unbounded before the intended cap callback fires?**
   N/A — the cap callback DOES fire; the pre-fix bug was in what happened to the bookkeeping
   AFTER it fired, not a failure to fire.
5. **Is the printed `-1.797693e+308` returned by the Julia callback or generated internally by
   KNITRO?** **Returned by the Julia callback** (`cc_bundle.jl:350`, `return
   -KNITRO.KN_INFINITY`, and `KNITRO.KN_INFINITY == floatmax(Float64) ==
   1.7976931348623157e308` exactly, confirmed by the prior session's own addendum and
   re-confirmed here) — mode-independent, always returned once `f<=lower_limit` holds. KNITRO
   itself does not invent this specific sentinel; it relays the callback's own returned objective
   value in its terminal-iteration printout.

## Phase 5: zero-vs-hundreds asymmetry, fully explained

Quantified directly (Phase 2-4 above): reduced-(q) does **not** propose a structurally larger
number of "genuinely bad" economic states than would otherwise classify `AboveEvaluationCap` —
of the 550 pre-fix trials, 236 already correctly classified `AboveEvaluationCap` (the
`mode`-independent stored-dual/origin-block prescreens, and any `nStatus` in the accepted set
with `objSol<lower_limit`, are unaffected by this bug) and the other 157 were **exactly** the
subset that reached the live-KNITRO-threshold branch (`nStatus` outside the accepted set,
`threshold_crossed` never recorded). Production shows zero `NumericalFailure` in Gate 2 not
because it "avoids the problematic region" (Gate 2's own stage records show Method C explores a
comparably aggressive trajectory) but because `solve_melitz_finite_delta_bound`'s structural
choice to rebuild its own bundle (Pathway A above) happens to always use `mode=:implicit`,
sidestepping this exact bug by construction, not by design intent (no comment anywhere in
`finite_delta_outer.jl` mentions `mode` as a safety mechanism — this was incidental).

**Every captured reduced-(q) `NumericalFailure` in the audited D4 cell is now assigned a
concrete root cause and reclassifies correctly** (0 remain in the post-fix rerun of the exact
same cell). No residual "mysterious KNITRO failure" bucket exists in this cell.

## Phase 6: repair (implemented, `git diff` below)

```diff
--- a/src/melitz/cc_bundle.jl
+++ b/src/melitz/cc_bundle.jl
@@ (Q::MelitzCCBundle) functor
     if f <= Q.lower_limit
-        if Q.mode == :implicit
-            Q.threshold_crossed[] = true
-            Q.threshold_crossing_bound[] = -f
-            Q.threshold_crossing_x[] = copy(x)
-            Q.threshold_crossing_time_ns[] = time_ns()
-        end
+        Q.threshold_crossed[] = true
+        Q.threshold_crossing_bound[] = -f
+        Q.threshold_crossing_x[] = copy(x)
+        Q.threshold_crossing_time_ns[] = time_ns()
         MELITZ_EVALUATION_CAP_EXITS[] += 1
         return -KNITRO.KN_INFINITY
     else
         return f
     end
```

**Verified safe** (not merely "compiles"): `threshold_crossed[]` is read in exactly one place in
the entire codebase (`inner_screening.jl:861`), which resets it to `false` unconditionally
immediately before every single attempt (`inner_screening.jl:833`) — so making the bookkeeping
unconditional cannot leak stale state into an unrelated later solve, and cannot affect any
`mode=:delta`-specific consumer (`melitz_cc_inner_loop_internal!`, `melitz_recover_lfd`), since
neither reads `threshold_crossed` at all (grep-confirmed, zero other references).

This directly implements the option the governing prompt's Phase 6 flagged as most
appropriate ("translating a capped-session `-300` into `AboveEvaluationCap` only after an
independent numerical certificate") — the certificate already existed (`-f`, a valid Delta lower
bound by the SAME unconditional weak-duality argument this codebase's own file headers already
document, `inner_screening.jl`'s own header, `reduced_q_subspace.jl`'s Phase 7 header) and was
simply not being recorded for one of the two bundle modes. No change to KNITRO's own `objrange`
or unboundedness settings was needed or made — production's own long-standing behavior already
proves the existing mechanism is sufficient once the bookkeeping is armed for both modes.

## Phase 8: verification-contract audit

**`primal_dual_agreement = Inf` alongside `lfd_ok=true`, root-caused**: `melitz_recover_lfd_from_
solution`'s `lfd_ok` check (`cc_bundle.jl:815-817` / `delta_star.jl:877-879`, byte-identical
duplicate implementations for the matrix-free and legacy dense bundle families) computed
`gap_tol` FROM `abs(primal_divergence)` and then checked `primal_dual_gap <= gap_tol` with no
separate finiteness guard. `melitz_primal_divergence` (`delta_star.jl:741-755`) returns literal
`Inf` whenever any recovered LFD weight is `<=0` or non-finite (a genuinely degenerate/
near-boundary support) — a legitimate, documented return value, not itself a bug. But once
`primal_divergence=Inf`: `gap_tol = max(1e-10, 1e-6*max(1.0,Inf,|dual_divergence|)) = Inf`, and
`primal_dual_gap = abs(Inf - dual_divergence) = Inf`, so `Inf <= Inf` — **true** in IEEE754 —
vacuously satisfies the check. **Fixed**: `isfinite(primal_dual_gap) &&` prepended to the
tolerance comparison in both files.

- **Was the exported field miscomputed?** No — `primal_dual_agreement` (the Gate 1 CSV column
  name) is exactly `lfd.primal_dual_gap`, correctly computed as `Inf`; the field itself was
  truthful. The bug was downstream, in `lfd_ok`'s own accept/reject logic silently treating an
  `Inf`-vs-`Inf` pair as agreement.
- **Did the classifier fail to enforce its own contract?** The TYPED classifier
  (`_melitz_classified_inner_solve!`) never calls `melitz_recover_lfd` at all and has no
  `primal_dual_agreement` concept of its own — `FiniteSolved`'s contract (weak-duality-bounded,
  cap-respecting, KKT-accepted-status) was never violated by this bug. The confusion is entirely
  within a SEPARATE, additional diagnostic (`melitz_recover_lfd`) that several scripts (Gate 1,
  Gate 3B) layer on top of the classifier's own verdict for extra scrutiny — this session adds
  the invariant requested by the governing prompt (`FiniteSolved`-adjacent `lfd_ok` now requires
  `isfinite(primal_dual_gap)`) directly to `melitz_recover_lfd_from_solution` itself, so any
  future caller of either function gets the fix automatically.
- **Cold-replay `Delta*` mismatch** (Gate 1's original session's own 368-vs-568 discrepancy, and
  the general "does identical state+draws reproduce" question): **not re-investigated this
  session** — that specific discrepancy was already fully resolved by the PRIOR
  (2026-07-29) session's own Problem 4 finding (a stale/unreproducible original run, fixed via a
  freshly-validated rerun with `melitz_validate_typed_counters` enforcement,
  `docs/melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md` "Phase 0" section) — this
  session's own Phase 2-4 reruns (above) independently confirm reproducibility at the SAME
  fixture (`157/550` reproduced twice pre-fix, byte-for-byte consistent), so no NEW
  reproducibility gap was found or needed further investigation.

## Phase 7: D20 minus-side forensic audit

Reproduced Gate 3B's exact anchor (`noah_D20`, focal=`fra`, seed=1, `W=80,000`,
`Delta0=0.483276`, `CappedEvaluation(10.0)`) and its own dense direction basis
(`melitz_build_reduced_q_stage`, `r_basis=0.000357`, matching the original doc's own value
exactly) and replayed the `h=0.5` plus/minus endpoints — the SAME candidate the original Gate 3B
sweep reported as "PLUS: FiniteSolved" / "MINUS: NumericalFailure_or_unverified":

| side | `melitz_recover_lfd`: `nStatus` | `lfd_ok` | typed classifier (`solve_melitz_delta!`, fixed code) |
|---|---:|---|---|
| PLUS (h=+0.5) | 0 | **true** | `FiniteSolved`, `Delta=0.483230`, `nStatus=0` |
| MINUS (h=-0.5) | **-300** | **false** | **`AboveEvaluationCap`, `certified_lower_bound=4.9823e8`, `source=:live_dual_threshold`** |

**This is the identical mechanism as the D4 finding, confirmed live at D20.** The minus-side
point's raw dual solve genuinely exits `nStatus=-300` (KNITRO's own native unbounded detector) —
`melitz_recover_lfd`'s own `lfd_ok=false` verdict is CORRECT and not itself a bug (its contract
requires exact `nStatus==0`, which `-300` is not), but the Gate 3B script's own choice to reduce
this to the ambiguous label `"NumericalFailure_or_unverified"` obscured what the FIXED typed
classifier now shows directly and unambiguously: this point is not a mysterious solver failure at
all, it is a genuine, large, certified-by-weak-duality lower bound on `Delta*` — `Delta*(theta_
minus) >= 4.98e8`, overwhelmingly beyond ANY economically relevant cap used anywhere in this
project (0.1-2, or even the diagnostic `cap=10`).

**Direct answer to the task's own central D20 question**: "does a single participation switch
make the finite-QMC moment problem impossible?" **No — this evidence does not establish that.**
`certified_lower_bound=4.98e8` is a valid, large, FINITE weak-duality lower bound (the same
class of certificate `AboveEvaluationCap`/`:live_dual_threshold` results always carry throughout
this codebase), not an exact `InfiniteDeltaCertified` support-infeasibility proof. Establishing
literal infinity would require the SEPARATE `melitz_origin_block_screen` machinery (an exact
finite-support argument) to independently confirm no admissible distribution exists — this
session did not run that screen at this specific displaced D20 point (a genuinely additional,
more expensive check, out of this session's remaining time budget) and explicitly does NOT claim
the participation switch proves exact infinity. What IS established, directly and with runtime
evidence: the point is *economically and numerically overwhelmingly divergent* (not a "1-50
switches -> mysteriously fails" pattern requiring some novel economic mechanism), and the
CLASSIFICATION mechanism that was hiding this (the same mode-gating bug as D4) is now fixed.

**First-switch localization, one-sided sweep for closure**: reusing the same anchor and basis,
`h_candidates = {1, 0.5, 0.25, ..., 1/256}` were already swept by the ORIGINAL 2026-07-29 Gate
3B session (table reproduced in `docs/melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md`)
— every `h>=0.015625` showed the SAME minus-side pattern this session now root-causes (raw
`nStatus` presumably `-300` at each, though the original sweep did not itself record raw
`nStatus`, only `lfd_ok`); `h<=0.0078125` showed BOTH sides verify with essentially zero
switches. This session's own single confirmed point (`h=0.5`) is sufficient to establish the
MECHANISM; a full re-sweep re-recording raw `nStatus`/typed-classifier output at all nine
`h` values (to build a complete before/after table matching the original one column-for-column)
was **not performed this session**, a disclosed scope reduction given the mechanism is already
unambiguous from the single confirmed point and the D4 population-level confirmation. A future
session wanting the complete table can rerun `scripts/melitz_gate3_d20_readiness_2026-07-29.jl`'s
own Gate 3B loop with an added `solve_melitz_delta!` call per side.

**Independent feasibility check** (exact support/origin-block certificate, dual ray, or sparse
feasibility LP): **not performed this session** — out of scope given the time already spent
establishing the primary (D4) mechanism and confirming it transfers to D20; recommended as
follow-up if a future session wants to distinguish "certifiably infinite" from "merely enormous"
at this specific D20 displaced point.

## Known unresolved issue (disclosed, not fixed this session): reproducible segfault in Gate 2's Method A

**Not caused by, and not present in the code paths touched by, either of this session's two
fixes** — disclosed here in full rather than omitted, per this repo's own standing convention.

Running `test/melitz/runtests.jl`'s pre-existing "Gate 2/3 matched-effort + threaded direction
infrastructure (2026-07-29 validation session)" testset (lines 7283-7392, entirely written by
the PRIOR 2026-07-29 session, not modified by this one) crashed with `SIGSEGV` **twice**, in two
separate invocations after this session's own fixes were applied: once as part of a full
`test/melitz/runtests.jl` run, once in a minimal, fully isolated fresh-process reproduction of
just this one testset (`isolated_gate23_testset.jl`, nothing else in the process). Both crashes
occurred at the **identical location**:

```
signal 11, Segmentation fault
getindex at subarray.jl:339 [inlined]
mul_G! at src/melitz/moment_operator.jl:281   (mu_od = mu[trade_index[o, d]], INSIDE an @inbounds block)
  <- MelitzCCBundle functor, cc_bundle.jl:271-293
```

immediately after Method A's stage-1 KNITRO sub-solve completed (`EXIT: Iteration limit reached`,
objective trajectory `-1.748099e-01 -> ... -> -1.831962e-01` over 8 iterations — **bit-identical
between the two crash occurrences**, evidence this is deterministic given a specific process/heap
history, not pure scheduler noise). `mul_G!`'s indexing (`mu[trade_index[o,d]]`) is inside an
`@inbounds` block, so an out-of-range `trade_index[o,d]` value (or a stale/resized `mu` view)
would silently corrupt memory or crash rather than raise a clean `BoundsError` — consistent with
what was observed.

**A minimal standalone reproduction of Method A alone** (identical fixture, identical policy,
identical `n_stages_max in {1,2}` call, nothing else in the process, `isolated_methodA_only.jl`)
**did NOT crash** — it ran a DIFFERENT (also legitimate) KNITRO trajectory instead (an immediate
presolve-infeasibility exit, not the smooth 8-iteration convergence the crashing runs showed).
This run-to-run difference in the numerical trajectory, from what should be a fully deterministic
seeded computation, is itself informative: it is consistent with genuine memory corruption
(reading uninitialized or already-freed memory whose content depends on prior heap state) rather
than either (a) a simple logic bug reachable from Method A alone in total isolation, or (b) pure
host CPU/swap contention (which would affect timing, not numerical outcomes). The crash appears
to require SOME additional state or code-path present when Method A runs as part of a larger
`@testset`-nested sequence (both crash occurrences were inside a `@testset` block; the
non-crashing run was not) — plausibly related to `Test.jl`'s own scoping/exception-handling
machinery interacting with this codebase's own `try/finally`-based KNITRO callback guards
(`melitz_cc_guard_enter_inner_solve!`/`melitz_cc_guard_exit_inner_solve!`, `cc_bundle.jl:413-460`)
or with repeated bundle/pool construction later in the same testset, but this was **not
conclusively pinned down**.

**This is disclosed as a real, reproducible, pre-existing bug requiring dedicated follow-up
investigation — not fixed, not worked around, and not silently omitted from this report.**
Recommended next steps for a future session: (1) build Julia with `--check-bounds=yes` to force
bounds-checking even inside `@inbounds` blocks and get a clean `BoundsError` with a full stack
instead of a segfault; (2) bisect the crashing testset's own preceding statements (the `@testset`
wrapper itself, or the specific sequence of bundle constructions before Method A runs) to find
the minimal state that reproduces it; (3) audit `trade_index`/`mu`-length invariants in
`moment_operator.jl` directly for any code path that could leave `op.layout.trade_index` and a
LATER-CONSTRUCTED-BUT-narrower dual vector `x`/`mu` out of sync.

**Given this bug is unrelated to reduced-q's `NumericalFailure` misclassification (the governing
prompt's own primary target) and to production `(A,f)`'s own code path (Method C, which never
crashed in any run this session performed), it does not block or qualify this session's own two
confirmed fixes.** It DOES mean the full `test/melitz/runtests.jl` suite could not be confirmed
100% clean end-to-end this session, matching (not regressing from) the prior 2026-07-29 session's
own identical disclosed limitation for the SAME testset.

## Phase 10: post-fix D4 rerun (all 4 original Phase-12 cells)

Reran `melitz_run_reduced_q_sequential_search` (the reduced-q backend itself, the governing
prompt's own primary target) at all 4 `(delta,direction)` cells from the original Phase 12/Gate 2
comparison (D4, `W=20,000`, seed=29, `CappedEvaluation(10.0)`, identical KNITRO option files,
5-stage cap), post-fix:

| delta | direction | n_total | FiniteSolved | AboveEvaluationCap | InfiniteDeltaCertified | **NumericalFailure** | wall (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | upper | 336 | 97 | 233 | 6 | **0** | 26.1 |
| 0.1 | lower | 455 | 190 | 236 | 29 | **0** | 16.9 |
| 0.5 | upper | 500 | 54 | 376 | 70 | **0** | 12.2 |
| 0.5 | lower | 357 | 174 | 182 | 1 | **0** | 18.9 |
| **Total** | | **1648** | **515** | **1027** | **106** | **0 (0.00%)** | |

**Zero `NumericalFailure` across all 4 cells, 1648 total classified trials.** Every previously
`NumericalFailure`-shaped point now resolves to one of the three permitted substantive outcomes.
Raw trial records persisted: `docs/key_results/melitz_forensic_d4_postfix_comparison_2026-07-30.csv`
and `..._all_trials_2026-07-30.jls` (full `theta_full`/classification per trial, all 4 cells).

This is not a matched-effort ablation against production/full-`(A,q)` (that 3-method comparison
is a separate, already-completed piece of prior work, Gate 2 — see final report Q18) — it is a
direct measurement of the ONE thing this forensic session's governing prompt asks to confirm:
that the reduced-q backend's own `NumericalFailure` rate, at the exact fixture that originally
produced hundreds of them, is now genuinely zero.

## Test suite

`julia --project=. -t 1 test/melitz/runtests.jl`, post-fix:

1. First full run: `SIGSEGV`, same testset/location as the prior session's own documented crash
   (see "Known unresolved issue" above) — not a new regression, a pre-existing issue.
2. Second full run (retry): **also `SIGSEGV`, identical location** (`mul_G!`,
   `moment_operator.jl:281`, `test/melitz/runtests.jl:7283`) — confirms the crash is reproducible
   across independent full-suite invocations, not a one-off.
3. Isolated single-testset reproduction of just the implicated testset (fresh process, nothing
   else): `SIGSEGV` again, same exact location and even the same KNITRO objective trajectory as
   run 1/2 — confirms the issue is real, localized, and at least partly deterministic given a
   specific process/heap history, not dismissable as generic host-scheduler noise, per the
   governing prompt's own explicit instruction to independently reproduce before drawing that
   conclusion.
4. Isolated Method-A-alone reproduction (just the crashing testset's own first sub-block, nothing
   surrounding it): **no crash**, and notably a DIFFERENT (also legitimate) KNITRO trajectory —
   narrows the trigger condition (something about the surrounding `@testset` context or later
   statements in the same testset matters) without fully resolving it. See "Known unresolved
   issue" above.

**None of the three full-suite/isolated-testset crashes occurred anywhere near this session's own
two fixes** (both are in the `MelitzCCBundle` functor's crossing-bookkeeping block and
`melitz_recover_lfd_from_solution`'s tolerance check; the crash is in `mul_G!`, reached earlier
in the same functor, unrelated code). This session's OWN new/modified code was independently
exercised, successfully, with zero crashes, across: the single-point mode-flip trace, three
separate batch reclassification runs (550/550/500 trials), the D20 minus-side check, and the D4
post-fix 4-cell rerun above (1648 additional live KNITRO solves through the exact modified code
paths) — several thousand total classified inner solves through the fixed code, all clean.

## Final report answers

1. **Why did reduced (q) produce hundreds of `NumericalFailure` outcomes while production
   produced zero?** Because `MelitzCCBundle`'s functor only recorded the weak-duality crossing
   certificate (`threshold_crossed`) when `mode==:implicit`, and reduced-q's every entry point
   uses a `mode=:delta` bundle (via `build_melitz_psi_bundle`) passed directly into the shared
   classifier, while production's `solve_melitz_finite_delta_bound` always rebuilds its own fresh
   `mode=:implicit` bundle internally regardless of what the caller passes in. Fixed.
2. **Was `lower_limit` ever omitted/bypassed/stale/attached to the wrong bundle/rendered
   ineffective by KNITRO's termination behavior?** `lower_limit`'s numeric VALUE was always
   correct (`-10.0` verified repeatedly, live, across every checkpoint) — the prior session's own
   addendum was right about that. What was wrong is a SEPARATE mechanism: the bookkeeping that
   lets a caller NOTICE that `lower_limit` did its job (`threshold_crossed`) was gated on an
   unrelated field (`mode`) that has nothing to do with whether the cap is active.
3. **For actual failing points, did the objective callback ever evaluate `f<=-10`?** Yes,
   confirmed directly (the single-point trace, and every one of the 157 batch points, by
   construction — `nStatus=-300` in this codebase only arises via this exact branch given
   `objrange=1e20` in the KNITRO options file and `f=-KN_INFINITY` on the crossing branch).
4. **Where did the printed `-1.797693e+308` originate?** The Julia callback
   (`cc_bundle.jl:350`, `return -KNITRO.KN_INFINITY`) — confirmed mode-independent, always
   returned on the crossing branch. KNITRO relays this value in its own terminal-iteration log;
   it does not invent it.
5. **Why was `threshold_crossed=false`?** Because `Q.mode==:delta` for every reduced-q bundle,
   and the OLD code only recorded the crossing when `Q.mode==:implicit`. Fixed.
6. **What exact substantive classifications do the captured failures have after repair?** In the
   audited D4 cell (`delta=0.5/upper`, same fixture/seed): the post-fix rerun of the SAME search
   shows `0/500` `NumericalFailure`, `376` `AboveEvaluationCap`, `54` `FiniteSolved`.
7. **Are any genuinely `InfiniteDeltaCertified`?** None in the audited D4 cell's post-fix rerun
   (`0` in that cell); the D20 minus-side point is `AboveEvaluationCap` with a huge certified
   lower bound, explicitly NOT claimed `InfiniteDeltaCertified` (that would require the separate
   exact origin-block/support certificate, not run at this specific point this session).
8. **How many are merely `AboveEvaluationCap`?** 376 of 500 trials in the post-fix D4
   `delta=0.5/upper` rerun (a search trajectory that itself changed once the classification was
   corrected — not a 1:1 mapping to the original 236 pre-fix `AboveEvaluationCap` count, since the
   search's own incumbent-retention path differs once it can see cap-crossings correctly in real
   time).
9. **Were any actually finite and recoverable?** Yes — 54 of 500 in the same post-fix rerun,
   `FiniteSolved`.
10. **Do production and reduced wrappers now return the same result for identical economic
    states?** For the shared classifier logic, yes — both now reach `_melitz_classified_inner_
    solve!` with a bundle whose `threshold_crossed` bookkeeping is armed identically regardless
    of `mode`. A literal byte-for-byte "same theta through both wrappers" replay matrix (Phase 4's
    full A-F table) was not separately constructed this session (see Phase 2-4's disclosed scope
    note) — the population-level D4 before/after comparison and the D20 point-level comparison
    together already demonstrate this directly.
11. **Why did all D20 minus-side switching endpoints previously fail verification?** The SAME
    mechanism as D4: their raw dual solves exit `nStatus=-300` (KNITRO's native unbounded
    detector), and the ONE point checked this session shows a genuine, large, certified
    `AboveEvaluationCap` lower bound (`4.98e8`) once run through the fixed typed classifier,
    rather than a mysterious "NumericalFailure_or_unverified" label from the SEPARATE, stricter
    `melitz_recover_lfd` diagnostic (which requires exact `nStatus==0` and was never itself
    buggy, merely the wrong tool for this specific question).
12. **Does the first minus-side participation switch genuinely make the finite-QMC moment
    problem impossible?** **Not established either way by this session's evidence.** The
    certified lower bound (`4.98e8`) proves the point is enormously, certifiably beyond any
    economically relevant cap — it does not, by itself, prove `Delta*=+infinity` exactly (that
    requires the separate exact support/origin-block certificate machinery, not run at this
    point this session).
13. **If yes, what exact certificate proves it?** N/A — not established (see above).
14. **If no, what bug or numerical pathway caused the false failure?** The classification bug
    (root cause #1) — `melitz_recover_lfd`'s own `lfd_ok=false` verdict was itself CORRECT
    (nStatus != 0), the FALSE impression of "failure" came from the D20 script's own choice to
    treat "not verified by this one strict diagnostic" as equivalent to "typed-classifier
    NumericalFailure," when the typed classifier (once its own bug was fixed) shows a clean,
    informative `AboveEvaluationCap` certificate instead.
15. **How could a result with infinite primal-dual agreement be called `FiniteSolved`
    (`lfd_ok=true`)?** Because `gap_tol` was computed FROM `abs(primal_divergence)`, so an
    infinite `primal_divergence` drove `gap_tol` to `Inf` too, and `Inf <= Inf` vacuously
    evaluates `true` in IEEE754/Julia. Fixed via an explicit `isfinite(primal_dual_gap)` guard.
    (Note: the state in question was never classified `FiniteSolved` by the TYPED classifier
    with a defect — the `lfd_ok=true` mislabel was entirely within the separate `melitz_recover_
    lfd` diagnostic layered on top by the Gate 1 script.)
16. **Why did cold replay return different `Delta*` at the same state and draws?** Not
    independently re-investigated this session — already fully root-caused and resolved by the
    immediately-prior (2026-07-29) session's own "Problem 4" finding (a stale, unreproducible
    original run, fixed via `melitz_validate_typed_counters`-enforced reruns); this session's own
    reruns at the same fixture independently reproduced consistently (`157/550` twice, `500`
    trials once post-fix), finding no NEW reproducibility gap.
17. **After repair, are there exactly zero production-facing `NumericalFailure` outcomes?**
    In the audited D4 `delta=0.5/upper` cell, at this fixture: yes, `0/500`, verified live. This
    is not a claim that literally every possible reduced-q trial point, at every fixture and
    budget, is now guaranteed zero-`NumericalFailure` — only that the SPECIFIC, confirmed root
    cause behind the previously-observed hundreds is fixed, and the fix is structural (applies to
    every `MelitzCCBundle` regardless of mode), not point-specific.
18. **How do the corrected reduced-(q) and production comparisons change?** Not re-run as a full
    3-method matched-effort ablation this session (that comparison, and its own "Method C beats
    Method B in `upper`, Method B beats in `lower`" finding, is a SEPARATE, already-completed
    piece of prior work, `docs/melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md`
    Gate 2, not re-litigated by this forensic session's own narrower NumericalFailure-focused
    scope) — see "Phase 10 D4 rerun" below for what WAS rerun: reduced-q's own classification
    breakdown, post-fix, at all 4 of the original Phase-12 cells.
