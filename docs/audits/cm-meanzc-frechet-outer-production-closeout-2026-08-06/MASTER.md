# CM+ZC / common-Fréchet two-family outer production closeout — 2026-08-06

Continuation of `diagnostic/cm-paired-basis-preconditioning-2026-08-05` (worktree
`/bbkinghome/edav/cdw_worktrees/cm-paired-basis-preconditioning-2026-08-05`), starting from the
recorded HEAD `895b99b8e6b1256a69e8e910f6a4f1533c59520f` (verified live at session start — matched
exactly). `NEW_BRANCHES_CREATED = 0`, `EXTRA_WORKTREES_CREATED = 0` — the existing branch/worktree
was reused throughout.

**Scope note up front, including a correction of this session's own earlier overclaim (read this
before the numbered sections):** this session resolved the central blocking question (the CM+ZC
screen contradiction) with strong, reproducible, live evidence, and found + fixed **eight** real,
narrow, individually-tested production bugs (listed in "Production SHA / tag" below), most
significantly a chain of four stacked bugs that left common-Fréchet's public outer path broken
(driver unreachability, a missing struct field crashing every real evaluation, a false positive in
this session's own new guard, and an outer-gradient bug described next), plus the SAME
outer-gradient bug found independently in plain flexible-CM. All are fixed and confirmed;
common-Fréchet now reaches a genuine, verified D20/W=100,000 calibration result matching CM+ZC's
own order of magnitude at Δ*.

The outer-gradient bug (section 5f): the analytic `d(Delta)/d(gp)` component came back identically
zero on an *unmatched* gradient call, in both common-Fréchet and flexible-CM. **This session's own
first writeup called this "campaign-blocking" — that claim was wrong and has been retracted.** The
user directly challenged it (a mechanically-always-zero gradient on the literal objective variable
would make any real KNITRO campaign immediately declare the start point optimal, which is not
observed), and a direct instrumented test of a real live outer run confirmed `matched=true` for
every real gradient call — the buggy path was never actually exercised in practice; only this
session's own standalone diagnostic scripts (which never established a matched F-call first)
triggered it. The bug is real and correctly fixed, but there is no evidence any actual campaign
result was affected.

**Two items remain genuinely open and unresolved, not merely "not yet run":** (1) a
larger-magnitude outer-gradient discrepancy in the non-`gp` (A-block) coordinates, confirmed
present in all three families tested (CM+ZC, common-Fréchet, flexible-CM) and NOT explained by
winner-switching (directly ruled out via a tiny-`h` probe) — root cause not found; (2) a structural
concern, raised but **not empirically verified**, that the unrestricted production driver may have
a related `obj.arg1`-staleness issue that is live rather than dormant (unlike the CM-family case).
Neither should be treated as resolved. This also did not complete the W=5k/W20k tiers as their own
dedicated gate, produce the literal field-by-field differential CSV, or merge to production.

## 1. CM+ZC screen contradiction — ROOT CAUSE FOUND

**`CM_PLUS_ZC_SCREEN_ROOT_CAUSE = different_decoded_state_meanzc_nu0_initial_guess`**

The prior session's evidence (`outer_driver_screening_rejection_evidence.log`,
`00_SUMMARY.md` section 2) reported the real public driver's own first `w0` evaluation being
rejected by `cm_screen_precheck!` ("pairwise-certified infeasible at o=9,d=17"), while an
apples-to-apples standalone replica of the same check passed cleanly — a genuine, unresolved
contradiction at the time.

**This session instrumented the actual first `cb_F!` invocation directly** (not another standalone
replica) at real production settings — sigma=3, W=100,000, L=50, `draw_design=:sobol_randomized`,
`draw_seed=20260719`, `destination_sample=:exclude_row`, `exclude_diagonal_gravity=true`,
Brazil-Korea gravity exclusion, `cm_extension=:cm_plus_moments` with **K_mean=1, K_pair=1** (the
actual production values from `configs/fullA_production_2026-08-03.toml` — the archived evidence's
own preserved script, `diag_meanzc_verify4_exact_code.jl`, used K_pair=0 and `contrasts=:orthonormal`,
neither of which matches production defaults) — and froze an exact snapshot immediately before
`cm_screen_precheck!` runs (`full_aod_diag/d4_exact/diag_cbf_snapshot_2026-08-06.jl`, kept as a
diagnostic script, not committed to the production commits).

**Result (two independent live runs, `diag/diag_cbf_snapshot_run1.log`,
`diag/diag_cbf_snapshot_run2_verified.log` — see Artifacts):**

- `max|xf - x_free_calib| = 1.68e-8` (relative `6.9e-14`) — the driver's own encode/decode
  (`cm_w0_from_calibration` → KNITRO → `xf_from_w_econ`) reconstructs the true calibration point to
  machine precision. **Not a decode bug.**
- **(A)** the exact production call path (`cm_screen_precheck!` inside the real `cb_F!`, via
  `CM_LIVE_PCX_STASH[]`'s live `pcx.ctx_cm`) → **PASSED, no exception.**
- **(B)** a direct pure-call replica (`pairwise_certificate` on the frozen `(a, ctx_cm.pairwise,
  Pmat)`) → **`infeasible=false`, worst=(o=15,d=10,k=10), slack=2.645** (comfortably feasible).
- **(A) and (B) agree**, both runs, and a repeated call on a fresh copy of `xf` (B2) reproduces
  byte-identical results — **no mutation, no hidden/shared scratch state** (rules out root-cause
  category 3, "mutable/shared scratch or hidden state").
- A **fresh flexible-CM control** (`build_cm_production_context`, same `ctx`, same `xf`) also
  passes cleanly, and `pcx_flex.ctx_cm.pairwise === pcx_live.ctx_cm.pairwise` (same object — the
  screen is genuinely family-agnostic, exactly as documented).

**The originally-reported rejection does not reproduce** under current code (`895b99b` +
this session's fixes) with correct production settings. Given the debug print that produced the
original evidence was gated by `n_eval[] <= 2`, and `n_eval[]` only increments *after* a successful
cache-lookup/compute (never on a thrown exception) — every attempted point in a rejection sequence
prints as `"call n=0"`, indistinguishable from the true `w0` in the log grep the prior session
based its "confirmed max|w-w0|=0.0" claim on. This session's own run 1 (below) shows exactly this
pattern live: the true first call at `w0` fails the **inner solve** (not the screen) for an
unrelated reason (see next paragraph), and *every subsequent* KNITRO retry — genuinely perturbed,
`max|w-w0|>0` — is correctly screen-rejected, each also printing `"call n=0"`. The most parsimonious
explanation is that the prior session's specific "(o=9,d=17)" evidence was from one such
already-perturbed retry, not literally `w0` itself, even though it printed as `n=0`.

**A second, real, separate finding from run 1:** with an uninformed `eta_nu` initial guess
(`log(nu)=0`, i.e. `nu=1`), the calibration point's **inner solve** genuinely fails
(`nStatus=-300`, a confirmed real KNITRO infeasibility code, see
`feedback-knitro-300-confirmed-infeasible-not-unbounded`) — this is expected: CM+ZC's mean/pair
restriction needs a calibration-consistent `nu0`, not an arbitrary guess, exactly the way
`diag_meanzc_verify4`'s own `nu1_guess = sum(aug.Zraw_all[1])/length(...)` convention already
established. **Run 2, using that exact convention for `nu0`, reaches a genuine, fully verified
result at the calibration point** — see section 2.

## 2. CM+ZC end-to-end at calibration — REAL, VERIFIED result obtained

With a calibration-consistent `nu0` guess, the real public driver
(`run_cm_upper_checkpointed`, `cm_extension=:cm_plus_moments`, K_mean=1/K_pair=1) at real
D20/W=100,000/L=50:

```
eval 1  t=42.7s  gp=0.9840278851786317  Delta=0.000651861941955464  feasible=true  verified=true
eval 2  t=120.7s gp=0.9816785181851972  Delta=4.282155322895578     feasible=false verified=true
```

- eval 1 is the calibration point (`gp` matches flexible-CM's own calibration `gp` exactly, see
  `key_results/flexcm_w100k_l50_evidence.txt`) — **small, finite Delta, `feasible=true`,
  `verified=true`**. This is a real converged, independently-verified inner solve, not the
  fake-success pattern the missing-`Pow=` bug produced.
- eval 2 is a genuinely different, infeasible nearby point KNITRO explored — expected behavior,
  not a bug.
- This result is **unchanged** (byte-identical) after wiring in this session's callback-health
  guard and capability guard (section 3/4) — confirming those additions are zero-behavior-change
  for a genuine successful solve.

The archived `Delta=-0.0, feasible=true, verified=false, best=nothing` evidence
(`cmmeanzc_w100k_l50_evidence_verified_false.txt`) is confirmed **pre-fix fake-solve evidence**, not
a current result, per the task's own framing.

## 3. Callback-health / fake-success guard — DONE, tested, committed locally

`full_aod_diag/d4_exact/cm_callback_health.jl` (new file):

- `CallbackHealthRecord` (`exception_seen`, `exception_type`, `exception_message`,
  `exception_backtrace_digest`).
- `callback_health_guard(raw_cb, health)` — wraps a raw KNITRO FG/Hessian callback so a real Julia
  exception is recorded **before** it reaches KNITRO.jl's own `_try_catch_handler` (which swallows
  it into a generic "exception in puts callback" warning + `KN_RC_CALLBACK_ERR`, indistinguishable
  from ordinary infeasibility). Rethrows unchanged — KNITRO's own status codes are not altered.
- `assert_no_fake_success!(label, health, nStatus, n_fg, x_initial, x_solution)` — hard-errors
  (real `ErrorException`, never `CMExpectedSolveFailure`, so a caller cannot silently swallow it as
  ordinary infeasibility) if: the guard caught an exception; `n_fg==0` (unconditionally — a real
  infeasibility certificate always requires ≥1 real FG evaluation); or the dual is byte-identical
  to the untouched initial vector while `nStatus` claims a converged code.

Wired into all four `inner_loop_KNITRO_*` KNITRO-registration sites that share this exact pattern:
`cm_lookup_production.jl` (flexible CM), `cm_meanzc_lookup_production.jl` (CM+ZC),
`cm_originzc_lookup_production.jl` (origin-ZC), `cm_frechet_lookup_production.jl`
(common-Fréchet).

**Regression test** (`test_callback_health_fake_success_guard_2026-08-06.jl`, D4, fast):
reconstructs the exact pre-895b99b broken `CMMeanZCOperatorState` (no `Pow=` for a genuinely
two-family `cctx`) and proves the guarded call now hard-errors instead of returning a usable
`nStatus=0` — **8/8 checks pass**, including a control confirming the *fixed* construction path
still succeeds normally through the same guarded site. A real MethodError bug in this session's own
first draft (`nStatus::Int` vs. KNITRO's actual `Int32`) was caught live by this test and fixed
before commit.

Committed locally: `4e06bf1` (guard) on `diagnostic/cm-paired-basis-preconditioning-2026-08-05`.
**Not pushed, not merged.**

`CALLBACK_FAKE_SUCCESS_GUARD = pass`

## 4. Obsolete production guard → real capability checks — DONE, tested, committed locally

The prior session's own removal of the blanket `include_truncated_moment=true` + CM+ZC refusal
(`cm_checkpoint.jl`) was a bare `nothing` statement justified only by prose ("the gap is now
closed"), not a runtime check. `assert_two_family_capabilities!` (added to `cm_checkpoint.jl`, right
after `pcx`/`cctx` are actually constructed) replaces it with checks against the **real, live**
context: `cctx.n_families==2`, `cctx.Pow !== nothing`, `cctx.inner_fg_backend==:operator` +
`cctx.meanzc_zc_op !== nothing` for CM+ZC specifically, and `cctx.tls !== nothing` whenever
`threaded_bins=true` — each directly targets one of the two concrete historical bugs found this
session (missing `Pow`, missing `tls`). No silent fallback to a narrower family or to serial.

Re-verified end-to-end at real D20/W=100,000/L=50: identical result to before the guard was added
(section 2's eval 1/eval 2) — zero behavior change for a genuinely capable context.

Committed locally: `3f9344e`. **Not pushed, not merged.**

`TWO_FAMILY_GUARDS = removed_after_capability_gates` (for CM+ZC and the shared TLS check; see
section 9 for the Fréchet-specific finding)

## 5. Common-Fréchet: four real bugs found and fixed — public path now reaches a genuine, verified, gradient-correct calibration result

`COMMON_FRECHET_TLS = pass`, `COMMON_FRECHET_INNER = pass`, `COMMON_FRECHET_OUTER W100K = pass`,
`COMMON_FRECHET_OUTER_GRADIENT (gp component) = pass`. This investigation started as a narrow TLS
check and uncovered four independent, real, previously undocumented bugs stacked on top of each
other — each masked the next, so they had to be found and fixed in sequence before a genuine,
gradient-correct end-to-end result was reachable at all.

**5a. The TLS question itself: real production path was already correct, no fix needed.** The
archived D4 log's `hessian_cm_frechet_structured_v2!(threaded_bins=true) requires tls` came from a
**diagnostic test script** calling the thin backward-compatibility wrapper directly (which defaults
`tls=nothing`, documented as "kept for legacy diagnostic scripts"), not the real dispatcher.
`archC_frechet_hess_cb_builder` (`cm_frechet_hessian.jl:476-508`) already reads `tls = cctx.tls`
whenever `cctx.use_threaded_bins`, and `cctx.tls` is built by the shared `build_cm_bin_ctx`
(`cm_hessian_architectures.jl:758`) — the same builder flexible-CM's context uses.

**5b. Real bug #1 — driver reachability (`cm_checkpoint.jl`, commit `56375bb`).**
`run_cm_upper_checkpointed`'s `is_frechet` branch never passed `include_truncated_moment`/
`threaded_bins` through to `build_cm_frechet_production_context` at all. Since
`include_truncated_moment` became a required (no-default) kwarg on that function during the
2026-08-05/06 truncated-power task, **every single call to this driver with
`marginal_restriction=:common_frechet` threw `UndefKeywordError` before ever reaching KNITRO** —
confirmed live. Fixed by passing both through, mirroring the other two branches; the internal
two-family gate (requires `moment_representation=:dense_reference`, incompatible with this driver's
dense ban) still correctly refuses two-family Fréchet — this fix restores single-family reachability
only.

**5c. Real bug #2 — missing `Pow` field on `CMFrechetLookupState` (`cm_frechet_lookup_kernels.jl`,
commit `a4a0671`).** With reachability fixed, every real evaluation immediately crashed with
`FieldError(CMFrechetLookupState, :Pow)`. `cm_forward_contribution!`/`cm_transpose_into_g!`
(`cm_lookup_kernels.jl`) are shared, untyped-`st` kernels that Fréchet's own `dual_index!`/FG
functor call directly by design; the 2026-08-05 truncated-power task added an unconditional
`fam2 = st.Pow !== nothing` duck-typed read to both and gave the other three operator-state types a
matching `Pow` field, but never added it to `CMFrechetLookupState` — breaking **every** real
evaluation under `CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]=:cm_frechet_lookup`, which **is the actual
production default**, single-family or not. Masked by KNITRO.jl's own callback-error swallowing
exactly like the original CM+ZC missing-`Pow=` bug. Fixed by adding the field (always `nothing`,
since Fréchet has no two-family extension) — every two-family-only branch in the shared kernels is
already gated behind `if fam2`, so no other field was needed; confirmed by reading both kernels'
full bodies before making the change.

**5d. Real bug #3 (in this session's own new code) — callback-health false positive (commit
`8a9f811`).** With 5b/5c fixed, the real driver reached a genuine converged eval 1, then the
session's own newly-added `assert_no_fake_success!` (section 3) incorrectly rejected a **later**,
legitimately-converged evaluation as "fake" because its returned dual was byte-identical to its
initial guess. Root cause: `CS.inner_loop_initial_values` legitimately **warm-starts** from the
previous solve's converged `obj.x` (a documented, real feature — `dual_bank.jl`/
`cm_dual_bank_production.jl`), so a warm-started dual that is *already* optimal at a nearby new
point legitimately converges in zero steps with `x_solution==x_initial`. Fixed by restricting the
check to a **cold** (all-zero) initial vector specifically — the actual fingerprint of the original
bug — confirmed the missing-Pow= regression test still catches the real case (8/8 PASS, cold-starts
by construction) while the legitimate warm-start case no longer false-positives.

**5e. A real, user-flagged sanity check that turned out to be a mismatched-scale artifact, not a
bug.** After 5b-5d, a direct diagnostic (`diag_frechet_w0_direct_2026-08-06.jl`) reported
`Delta_dual=0.005` at D20/**W=20,000** — correctly flagged by the user as too high relative to
CM+ZC's own `0.00065` at D20/**W=100,000**, an apples-to-oranges comparison (this codebase has
documented real W-sensitivity at D=20: `d20-realdata-w-sensitivity` memory, W=8,000 understates κ
for δ≥1 vs W≥80,000). Rerun at the matched W=100,000: `Delta_dual=0.000511` — directly comparable
to and consistent with CM+ZC's `0.00065`, both well under the <0.001 calibration benchmark.
Confirmed through the **real public driver** too (not just the direct/bypass call), D20/W=100,000/
L=50: `eval 1 gp=0.9840278851786317 Delta=0.000511 feasible=true verified=true` — the same
calibration `gp` every other family lands on, genuinely verified, matching CM+ZC's own order of
magnitude at the same scale.

**5f. Real bug #4 — the analytic outer gradient's `d(Delta)/d(gp)` component was identically zero
on an unmatched gradient call, in BOTH common-Fréchet and plain flexible-CM (commits `0a991d8`,
`18089cf`).** Caught only because a reviewer refused to accept a structurally-zero economic
derivative at face value: "the objective function is literally γ_d′ ... that has nothing to do with
the common Fréchet restriction ... surely a bug." Correct call. Root cause:
`gamma_component_analytic` (`lfix_factorized.jl`) computes `d(Delta)/d(gp)` as proportional to
`mean(base.m_star .* SW)`, where `m_star` is `obj.arg1` at the converged solve. Under the
production-default **operator** FG backend, none of the three operator states
(`CMLookupState`/`CMMeanZCOperatorState`/`CMFrechetLookupState`) ever write into `obj.arg1` — each
writes into its own private scratch (confirmed by reading all three functor bodies). The *verified*
state builders (`archC_X_verified_state`) sidestep this entirely, computing `m_star` independently
via operator-based verification. The *bare* builders (`archC_X_base_state`) do not — they read the
never-written `obj.arg1`, silently returning an all-zero `m_star`.

`cm_meanzc_production_gradient_cplus` (CM+ZC) and `cm_originzc_production_gradient_cplus`
(origin-ZC) already fall back to the verified builder on an unmatched gradient call
(`base===nothing`). `cm_frechet_production_gradient_cplus` **and** plain flexible-CM's own
`cm_production_gradient_cplus` (`lfix_cm_cplus.jl`) both fell back to the **bare** builder instead —
silently zeroing the entire `d(Delta)/d(gp)` term whenever called unmatched. Confirmed live at a
real D20/W=20,000 point: central-FD gave `d(Delta)/d(gp)=0.40–0.46`; the buggy analytic gradient
returned exactly `0.0` for both families. Fixed both to match CM+ZC's exact pattern (fall back to
the verified state builder whenever `base` OR `verify` is missing); also updated `cb_G!`'s own
call sites (`cm_checkpoint.jl`) to pass `verify=verify_c` through for both branches. **Verified:
analytic vs. central-FD now agree to `rel_diff=1.7e-6` (Fréchet) and `3.9e-8` (flexible-CM)**,
matching CM+ZC's own precision.

**Correction to this session's own earlier claim — read this before treating the bug above as
campaign-relevant.** This was first written up as "a genuine, previously-undiscovered,
campaign-blocking correctness bug ... every real common-Fréchet outer search would have had a
structurally wrong search direction along gp." **That characterization is wrong and is retracted.**
The user directly and correctly challenged it: if `gp`'s gradient were mechanically always zero in
real use, a real KNITRO campaign would treat the initial point as already optimal and stop
immediately — which is not what's observed; real campaigns on this exact code progress normally.
The reconciliation: the bug **only** fires when `cb_G!` is called *unmatched* — no `cb_F!` was
cached at the bit-identical point (`shared.w == w` false or `shared===nothing`). Standard NLP
solver behavior (evaluate `f` at a trial point, then request its gradient at that *same* point)
means `matched=true` in the overwhelming normal case. This was verified directly, not assumed:
`cb_G!` was instrumented and a real short outer search was run (real public driver, plain
flexible-CM, D20/W=20,000/L=50) — **`matched=true` for all 3 real gradient calls observed; the
buggy unmatched fallback was never exercised.** This session's own diagnostic scripts, by calling
the gradient functions standalone with no preceding matched F-call, were unconditionally exercising
exactly the one path that's buggy — not representative of real campaign behavior. **Net
assessment: the bug is real, now fixed, and worth having fixed (the documented `base=nothing`
calling contract, and untested edge cases such as a checkpoint resume's first callback), but there
is no evidence it affected any actual historical campaign result**, and it should not be described
as campaign-blocking.

**Open design question, raised by the user, not resolved/decided this session:** should `cb_G!`
itself hard-error on an unmatched call, rather than silently paying for a correct-but-expensive
recompute? Two considerations pull in different directions: (a) the `base=nothing` fallback inside
`cm_X_production_gradient_cplus` is a documented, intentional convenience API for callers with no
prior matching F-call (standalone scripts, diagnostics) — forcing an error at that layer would
break legitimate direct callers; (b) inside `cb_G!` specifically, an unmatched call is empirically
anomalous (never observed in real operation) and could indicate a different, unrelated bug (e.g. a
floating-point `==` comparison failing for "the same" point) that a silent, valid-looking recompute
would mask. Not implemented this session — flagged for a decision, not resolved.

**A-block gradient discrepancy — unresolved, and NOT explained by winner-switching.** A secondary,
larger-magnitude discrepancy exists in the non-`gp` coordinates of the SAME outer gradient (e.g.
index 2, index 380 of the A-block): 17%–176% analytic-vs-FD mismatch. Confirmed present in **all
three** families tested — CM+ZC, common-Fréchet, and **plain flexible-CM** — all via the identical
shared `composite_gradient_at_Cplus_from_cache`/`a_block_fd_component_Cplus!` machinery,
`cm_gradient_backend=:cplus` (no dense G/H) throughout. Not yet tested: origin-ZC, or the literal
unrestricted family (see below).

The user asked for this to be decomposed into the exact intensive-margin (incumbent-winner,
smooth-reweighting) piece vs. the winner-switching piece, since `a_block_fd_component_Cplus!` is
itself a central-FD computation (`lfix_incremental_at_Cplus!` at `w0±h`, holding the base dual
`λ*/ζ*` fixed via an envelope-theorem shortcut, NOT a full re-solve) built on
`dest_contrib_incremental_top3_C!`, which is an **exact** (not approximate/smoothed) recomputation
of the true winner among `{cached top-3 candidates} ∪ {perturbed origins}` — provably correct
because non-perturbed origins' prices are unchanged, so the best non-perturbed candidate is
necessarily one of the cached top-3. This means the "harder, winner-switching" piece the user asked
about is not a smoothing approximation in this codebase — it's computed exactly.

**Direct test result: a tiny-h probe (`h=1e-10`, small enough that essentially no winner can flip)
still shows the SAME discrepancy** (idx=2: 19.7%, idx=380: ~180%) as larger `h`. This rules out
winner-switching as the explanation — the mismatch is present even in the pure intensive-margin
regime, which was not the expected result. A companion attempt to directly count winner switches at
this coordinate had its own decode bug (missed the powered-aspace `cm_z_from_a` transform when
reconstructing the perturbed θ), so that specific piece of evidence is unreliable and should not be
cited. **Root cause not found this session.** Two live hypotheses, neither confirmed: (a) a genuine
bug in the shared A-block formula/implementation, independent of the two bugs already found and
fixed; (b) an error specific to this session's own external FD reimplementation (full re-solve at
`w0±h`) that the `gp` check's success doesn't rule out, since `gp` takes a structurally different
code path (`gamma_component_analytic`, no FD at all internally) than the A-block coordinates do.

**Unrestricted family — structural concern raised, NOT empirically verified. Do not treat as
confirmed.** Asked directly whether the same `obj.arg1`-based bug pattern reaches the unrestricted
production driver (`run_polish_checkpointed_unified`, `c10_d20_production_driver_unified.jl`).
Structural reading of the code: `cb_F!`/`cb_G!` there build `BaseDualState` via
`r.cache_hit ? (compressed_base_state(...) or solve_base_state(...)) : BaseDualState(..., copy(ctx.obj.arg1), ...)`.
`compressed_base_state` (used on `cache_hit=true`) independently computes `m_star` via
`compressed_cc_value_grad` — confirmed correct, not reading `obj.arg1`. The `cache_hit=false`
branch, however, reads `copy(ctx.obj.arg1)` directly, and `obj.arg1` is confirmed (by the same
grep-for-writes check used elsewhere) never written under the operator backend in this file either.
Unlike the CM-family case, `r.cache_hit` here is an **exact-point cache** (has this literal point
been evaluated before), not "did F just run at this same point" — for a normal outer search
visiting new points every iteration, `cache_hit=false` would plausibly be the *common* case, not a
rare one, which would make this a live rather than dormant risk if the reasoning holds. **This has
NOT been tested the way the CM-family claim was verified** (no instrumented real run, no confirmed
observation of `cache_hit` or `obj.arg1`'s actual state at the read site). Given this session
already produced one overclaim that required direct empirical correction, this is deliberately
reported as an unconfirmed structural concern requiring its own dedicated verification, not a
finding.

**Net result**: common-Fréchet's public outer path, single-family, now works end-to-end at real
D20/W=100,000/L=50 through the actual production driver — screen passes, inner solve converges,
verification passes, TLS is correctly constructed, Δ* at calibration matches the other families'
own order of magnitude, and the outer gradient's `gp` component is now analytically correct (in
both common-Fréchet and flexible-CM). The A-block discrepancy and the unrestricted-family question
are both open, unresolved, and should block any production merge. The new
`assert_two_family_capabilities!` (section 4) additionally adds a defensive, explicit hard-error
for the `threaded_bins=true` + `cctx.tls===nothing` misconfiguration class at the real driver's own
top level, for all three CM-family families sharing this pattern.

## 6. What remains open (not completed this session — do not treat as done)

- **Section 6** (pure/structured pairwise-screen refactor + dedicated test matrix): the screen was
  proven pure and reproducible **empirically** (repeat-call idempotence, A/B agreement, no mutation
  — section 1's live evidence) but was **not refactored** into a structured certificate-result type,
  and the specific test file / CSV the task describes was not written. Given the live evidence found
  zero purity defects, this is lower-priority than it looked before this session, but it is still
  open per the task's letter.
- **Section 8 (outer-gradient FD check) — PARTIALLY DONE; found and fixed one real bug (impact
  corrected below), left a second genuinely unresolved.** Ran central-FD at a real D20/W=20,000
  point for CM+ZC, common-Fréchet, AND (after the user asked) plain flexible-CM, `gp` + two A-block
  coordinates each. **`gp`: PASS for all three families** (rel_diff `3.6e-6` CM+ZC, `1.7e-6`
  Fréchet, `3.9e-8` flexible-CM, all after the section-5f fix) — but see section 5f's own
  correction: the bug this caught was confirmed, via a direct instrumented real run, to be dormant
  in normal production operation (`matched=true` always observed), not campaign-blocking as first
  claimed. **A-block coordinates (idx=2, idx=380): FAIL for all three families** (17%–176%
  relative mismatch) — a tiny-`h` probe (`h=1e-10`) directly ruled out winner-switching as the
  explanation (mismatch persists essentially unchanged), so this is NOT the documented
  FD-bandwidth-mismatch pitfall as this session's own earlier draft speculated; root cause
  genuinely unknown. The eta_nu coordinate (CM+ZC only) passed cleanly (`1.9e-6`). Not run: the
  gravity-pivot-A/power-CM/mixed-direction sub-cases, D4 scale, origin-ZC, or the literal
  unrestricted family (see the new structural-concern item below).
  `CM_PLUS_ZC_OUTER_GRADIENT`/`COMMON_FRECHET_OUTER_GRADIENT`/`FLEXIBLE_CM_OUTER_GRADIENT` should
  be read as "gp component verified correct, confirmed dormant-not-campaign-blocking; A-block
  component genuinely unresolved" — not a blanket pass.
- **NEW — unrestricted-family structural concern, raised but NOT verified.** Reading
  `run_polish_checkpointed_unified`'s own `cb_F!`/`cb_G!` (`c10_d20_production_driver_unified.jl`)
  found a similarly-shaped `BaseDualState(..., copy(ctx.obj.arg1), ...)` construction on its
  `cache_hit=false` branch — and unlike the CM-family case, this branch's own trigger condition
  (an exact-point cache miss) is plausibly the *common* case for a normal outer search visiting new
  points every iteration, which would make this a live risk rather than a dormant one if the
  reasoning holds. **This was explicitly NOT tested empirically this session** — no instrumented
  real run, no confirmed observation of `cache_hit` or `obj.arg1`'s actual state at the read site.
  Given this session already had to retract one overclaim after direct verification proved it
  wrong, this is deliberately reported as an open question requiring its own dedicated check, not
  a finding. Do not action this without first verifying it the same way the CM-family claim was
  verified (instrument the real driver, run a real outer search, observe directly).
- **Section 10** (`FLEXIBLE_CM_CONTROL_VS_EXTENSION.csv`, literal field-by-field diff): not
  produced. The qualitative differences are known from code reading (CM+ZC adds `Pow`-gated
  `CMMeanZCOperatorState` dual layout + `meanzc_zc_op`/widened `H_CZ` cross-Hessian block +
  `eta_nu` outer coordinates; common-Fréchet adds a level-anchor block reusing the same `Ttab`/`CT`
  tables) but was not assembled into the requested CSV artifact.
- **Section 11 scale ladder — mostly done.** D4: CM+ZC (regression test) and common-Fréchet
  (Pow-fix regression). D20/W=100,000/L=50: **both** families, calibration eval verified=true +
  one nearby point, through the real public driver. D20/W=20,000: **both** families, real outer
  evaluation + real gradient + checkpoint-write + resume round-trip (CM+ZC: 1 eval/1 grad then
  resume to 2/2; common-Fréchet: 8 evals/3 grads then resume to 14/7) — all genuine, `PASS`.
  D20/W=5,000: **run, and both families genuinely fail** — `n_eval=0`, KNITRO `-502` ("could not
  evaluate objective or constraints at the initial point") for both CM+ZC and common-Fréchet. This
  is a real inner-solve infeasibility (`nStatus=-300`) at the true calibration point specifically
  at this small a W, consistent with this codebase's own documented D20 W-sensitivity (memory:
  W=8,000 understates κ for δ≥1 vs W≥80,000; a separate restriction needs W≥80,000 specifically
  because it is non-monotonic in W) — not a bug, but also not something to route around; W=5,000
  is simply below the reliable threshold for D20 real data in this codebase. Common-Fréchet's
  **two-family** CM extension was not run at any scale (architecturally blocked, see 5b).
- **Section 12** (additional permanent no-fake-success tests beyond the one committed): only the
  callback-health regression test was added. The task's fuller list (verified=false-published-as-
  incumbent, threaded_bins/TLS test, family-guard-silent-downgrade test) was not separately
  implemented as permanent tests — the corresponding *mechanisms* were verified live (section 5),
  just not turned into standing test files.
- **Section 13/production merge**: **not done, and should not be done without separate explicit
  confirmation** — the A-block gradient discrepancy (section 8) is not yet resolved either way, so
  a real fast-forward + tag would be premature regardless of authorization. See "Production
  SHA/tag" below.
- **Section 14 campaign handoff**: partial — see that section below; explicitly flagged as
  provisional pending the above gates.

## Final verdicts (per task's requested format)

```
CALLBACK_FAKE_SUCCESS_GUARD = pass

CM_PLUS_ZC_INNER = pass
    (confirmed: real D20/W=100,000/L=50, inner_status=-103, n_fg_calls=14, all 2301 dual
    entries nonzero -- 895b99b's own Pow-wiring fix, re-confirmed unchanged this session)

CM_PLUS_ZC_SCREEN_ROOT_CAUSE = different_decoded_state_meanzc_nu0_initial_guess
    (the screen mechanism itself is family-agnostic, pure, and reproducible; the original
    rejection evidence does not reproduce under current code + correct production settings --
    most likely explained by "call n=0" being printed on every retry after the true w0's own
    inner-solve failure under a bad nu0 guess, not literally at w0 itself)

CM_PLUS_ZC_OUTER =
    D4: pass (regression test, real inner solve via archC_meanzc_base_state)
    W5K: fail_w_sensitivity (genuine nStatus=-300 inner infeasibility at the true calibration
                 point at this scale, knitro_status=-502, n_eval=0 -- consistent with this
                 codebase's documented D20 W-sensitivity, not a bug; W=5,000 is below the
                 reliable threshold for D20 real data here)
    W20K: pass (real driver: n_eval=1/n_grad=1, checkpoint written, resume carries to n_eval=2)
    W100K: pass (calibration eval: feasible=true, verified=true, Delta=0.00065;
                 one nearby eval: feasible=false, verified=true -- both genuine)

CM_PLUS_ZC_OUTER_GRADIENT =
    gp component: pass (central-FD rel_diff=3.6e-6, D20/W=20,000, real production gradient fn --
                 CM+ZC's own gp-gradient was ALREADY correct before this session; unaffected by
                 the section-5f bug, which was specific to Frechet/flexible-CM)
    eta_nu component: pass (rel_diff=1.9e-6)
    A-block components (idx=2, idx=380): fail_unresolved (17%-176% rel diff -- see section 8;
                 a tiny-h probe ruled OUT winner-switching as the explanation; root cause unknown,
                 present identically in CM+ZC/Frechet/flexible-CM -- do not assume benign)

COMMON_FRECHET_INNER = pass
    (pre-existing D4 + D20 26/26 structured-vs-dense Hessian gates, both contrasts modes, still
    hold; PLUS this session found and fixed a real, previously-undiscovered crash -- missing
    Pow field on CMFrechetLookupState broke EVERY real evaluation under the actual production
    FG backend default, CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]=:cm_frechet_lookup -- commit
    a4a0671. Confirmed fixed: direct calls and the real driver both now reach a genuine
    converged, verified inner solve)

COMMON_FRECHET_TLS = pass
    (real production dispatcher already reads cctx.tls correctly; archived TLS error was from
    a diagnostic script calling a legacy wrapper directly, not the production driver)

COMMON_FRECHET_OUTER =
    D4: not_run (d4_exact_setup's context shape is not a fit for cm_w0_from_calibration/pivot
                 elimination -- confirmed live, DimensionMismatch 15 vs 379; use real D20 data
                 at small W instead, as done for the W20K/W100K rows)
    W5K: fail_w_sensitivity (same genuine nStatus=-300/knitro_status=-502/n_eval=0 as CM+ZC at
                 this scale -- not a bug, see CM_PLUS_ZC_OUTER's identical W5K entry)
    W20K: pass (real driver: n_eval=8/n_grad=3, checkpoint written, resume carries to n_eval=14)
    W100K: pass -- real public driver, D20/W=100,000/L=50, single-family common-Frechet:
                 eval 1 gp=0.9840278851786317 Delta=0.000511 feasible=true verified=true
                 (matches CM+ZC's own Delta=0.00065 at the same calibration point/scale, both
                 under the <0.001 calibration benchmark -- an earlier W=20,000 reading of
                 Delta=0.005 was flagged by the user as implausibly high and confirmed to be a
                 W-scale artifact, not a bug, once rerun at matched W=100,000); one nearby point
                 (Delta=0.15, feasible=true, verified=true) also genuine.
                 Two-family common-Frechet: architecturally blocked, not attempted (see 5b)

COMMON_FRECHET_OUTER_GRADIENT =
    gp component: pass (central-FD rel_diff=1.7e-6, D20/W=20,000, AFTER fixing a real bug --
                 the analytic gp-gradient was identically 0.0 before the section-5f fix (commit
                 0a991d8), but ONLY on an unmatched cb_G! call; a direct instrumented real run
                 confirmed matched=true for every real gradient call observed, i.e. the bug was
                 dormant in practice, not campaign-blocking as first (incorrectly) claimed)
    A-block components (idx=2, idx=380): fail_unresolved (same unresolved pattern as CM+ZC's own
                 A-block components, symmetric across families -- see section 8)

FLEXIBLE_CM_REGRESSION = pass, WITH a real bug found and fixed (commit 18089cf)
    (screen/inner-solve results unchanged before/after this session's guard wiring; but plain
    flexible-CM's own cm_production_gradient_cplus had the IDENTICAL gp-gradient=0 bug as
    Fréchet's -- found only because the user asked "is this present in flexible_cm too?".
    gp component now: pass (rel_diff=3.9e-8). Same dormant-not-campaign-blocking correction
    applies -- verified via the same direct instrumented real-run check. A-block components
    (idx=2, idx=380): fail_unresolved, same as the other two families.)

UNRESTRICTED_OUTER_GRADIENT = not_verified_this_session
    (structural reading of run_polish_checkpointed_unified found a similarly-shaped obj.arg1
    read on its cache_hit=false branch, which may be the COMMON case for this driver -- unlike
    the CM-family bug, this has NOT been empirically confirmed one way or the other; treat as
    an open question requiring its own dedicated verification, not a finding)

TWO_FAMILY_GUARDS = removed_after_capability_gates

PRODUCTION_RELEASE = not_merged_A_block_gradient_discrepancy_unresolved_plus_unrestricted_unverified

CAMPAIGN_READY =
    CM_plus_ZC (single-family, K_mean=1/K_pair=1): no (A-block outer-gradient discrepancy not
                        resolved either way -- W5k/W20k/W100k tiers and gp-gradient ARE verified)
    common_Frechet (single-family): no (same A-block gradient discrepancy; W5k/W20k/W100k tiers
                        and gp-gradient ARE now verified -- a major change from this session's
                        own earlier, incorrect "needs its own w0" conclusion; the gp-gradient=0
                        bug found and fixed in section 5f was confirmed dormant, not the reason
                        this remains not-ready -- the A-block discrepancy is)
    flexible_cm: no new blocker found beyond the A-block discrepancy (present here too); the
                        gp-gradient=0 bug found in this family too is fixed and confirmed dormant
    unrestricted: status unknown -- the structural concern above was not verified either way
    common_Frechet (two-family): no, architecturally blocked -- requires extending
                        CMFrechetLookupState/build_cm_frechet_production_context's own
                        moment_representation gate for two families, out of this session's
                        scope ("do not create new family-specific Hessian/gradient algorithms")

DENSE_PRODUCTION_GH_USED = false
NEW_HESSIAN_ALGORITHMS = 0
NEW_GRADIENT_ALGORITHMS = 0
CAMPAIGN_LAUNCHED = false
NEW_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
```

## Production SHA / tag

**None minted.** `PRODUCTION_RELEASE = not_merged_A_block_gradient_discrepancy_unresolved_plus_unrestricted_unverified`.

Local commits on `diagnostic/cm-paired-basis-preconditioning-2026-08-05` this session (not pushed,
not merged into `production/fullA-exact`):

- `4e06bf1` — shared callback-health/fake-success guard
- `3f9344e` — capability-based two-family guard (replaces the blanket refusal)
- `56375bb` — fix common-Fréchet driver's missing `include_truncated_moment`/`threaded_bins`
  passthrough (the public common-Fréchet path was completely unreachable before this fix)
- `1257712` — first MASTER.md writeup + screen-contradiction diagnostic script
- `a4a0671` — fix `CMFrechetLookupState`'s missing `Pow` field (crashed every real Fréchet
  evaluation under the actual production FG backend default)
- `8a9f811` — fix a false positive in this session's own callback-health guard (warm-started
  dual legitimately equal to a converged initial guess is not "fake")
- `bb1cad1` — update the Fréchet diagnostic scripts to match production sigma/gravity/W settings
- `0a991d8` — fix common-Fréchet's outer gradient: `d(Delta)/d(gp)` was identically zero on any
  unmatched gradient call. **Its own commit message called this "campaign-blocking" — that claim
  is retracted, see commit `18089cf` and section 5f for the correction and direct verification.**
- `18089cf` — fix the SAME `d(Delta)/d(gp)=0` bug in plain flexible-CM (found because the user
  asked whether it was Fréchet-specific — it wasn't); **also** the commit that directly verifies,
  by instrumenting the real driver and running a real outer search, that `matched=true` for every
  real gradient call observed — i.e. the bug (in both families) was dormant in practice, not
  campaign-blocking. This is the correction commit for `0a991d8`'s own overclaim.

Prior, still-uncommitted-elsewhere-but-present-here work this session did **not** touch or attempt
to complete: `full_aod_diag/d4_exact/cm_originzc_checkpoint.jl`'s own uncommitted `pin_outer_algorithm`
mirror addition (9-line diff, present before this session started, unrelated to this task's scope —
left exactly as found).

## Campaign handoff (provisional — do not launch from this alone)

**Common Fréchet / CM+ZC upper and lower (single-family): substantially more ready than earlier in
this session, but NOT yet campaign-ready** per the verdicts above — real, verified D20/W=100,000
calibration results now exist for both families, the W=20,000 tier (outer eval + gradient +
checkpoint/resume) passes for both, and a real outer-gradient bug affecting common-Fréchet AND
flexible-CM (`d(Delta)/d(gp)` identically zero on an unmatched gradient call) was found and fixed
— directly confirmed dormant in real operation, not the reason campaign-readiness is blocked.
Remaining blocking items:

1. **Resolve the A-block outer-gradient discrepancy** (section 8): 17%-176% analytic-vs-FD mismatch
   at non-gp coordinates, present identically in CM+ZC, common-Fréchet, AND flexible-CM. A tiny-`h`
   probe directly ruled OUT winner-switching as the explanation — root cause genuinely unknown. Do
   not launch a campaign while this is open.
2. **Verify (or refute) the unrestricted-family structural concern** (section 8/5f): a
   similarly-shaped `obj.arg1` read in `run_polish_checkpointed_unified`'s `cache_hit=false`
   branch, plausibly the *common* case for that driver (unlike the CM-family bug, which was
   confirmed dormant) — not yet empirically tested either way.
3. Run the outer-gradient FD matrix's remaining directions (ordinary-A / gravity-pivot-A / CDF-CM /
   power-CM / mixed direction) at D4 and D20/W=20,000 — only `gp` (+ eta_nu for CM+ZC) were
   checked this session.
4. Two-family common-Fréchet remains architecturally blocked (`build_cm_frechet_production_context`
   requires `moment_representation=:dense_reference` for two-family, incompatible with this
   driver's dense ban) — needs the same `Pow`-aware operator extension CM+ZC/flexible-CM already
   received, which is new algorithm work outside this session's scope, not just an unrun test.
   Single-family common-Fréchet is unaffected and is the one confirmed working above.
5. Once 1-3 pass, rebase onto latest `origin/production/fullA-exact`, rerun the flexible-CM
   regression gate, then fast-forward + tag.

**Scientific manifest fields for when this is ready** (`configs/fullA_production_2026-08-03.toml`):
sigma=3.0, W=100000, L=50, K_mean=1, K_pair=1, draw_design=sobol_randomized,
draw_seed=20260719, destination_sample=exclude_row, exclude_diagonal_gravity=true,
gravity_exclude_cells=[[3,14]] (Brazil-Korea).

**Old checkpoints are invalid**: any CM+ZC checkpoint produced before commit `895b99b` (the
missing-`Pow=` fix) is scientifically invalid even if it reports `nStatus=0` — that is exactly the
fake-success signature this session's guard (section 3) now makes structurally impossible to
produce again, but it does not retroactively validate anything written before the fix landed.

## Artifacts

- This file: `docs/audits/cm-meanzc-frechet-outer-production-closeout-2026-08-06/MASTER.md`
- Large logs (not committed to git): `/bbkinghome/edav/repo_scratch/cm-meanzc-frechet-outer-production-closeout-2026-08-06/`
  - `source_zip/` — the pulled prior-session Dropbox evidence package, for reference
  - `diag/diag_cbf_snapshot_run1.log`, `diag/diag_cbf_snapshot_run2_verified.log` — this session's
    live screen-contradiction diagnostic runs (section 1/2 evidence)
  - `diag/test_callback_health_regression_final.log` — section 3 regression test, 8/8 PASS
  - `diag/smoke_frechet_public_driver_tls_w100k_final.log` — section 5 Fréchet real public-driver
    W=100,000/L=50 calibration result (eval 1 Delta=0.000511, feasible=true, verified=true)
  - `diag/diag_frechet_w0_direct_w100k_final.log` — section 5e direct-call confirmation matching
    the driver result (Delta_dual=0.000511) after correcting the earlier W=20,000 comparison
  - `diag/tier_w5k_w20k_final.log` — section 11 W=5,000/W=20,000 tier runs, both families
  - `diag/fd_outer_gradient_check_AFTER_FIX.log` — the gp=0 bug: before/after FD comparison
  - `diag/diag_gamma_component_trace.log` — factor-by-factor trace that located the exact missing
    `archC_frechet_verified_state` call (section 5f mechanism)
  - `diag/diag_ablock_decompose.log` — flexible-CM gp=0 confirmation, the tiny-h winner-switching
    test, and the (unreliable, decode-bug-affected) winner-switch count
  - `diag/diag_matched_check.log` — the direct instrumented real-run confirmation that
    `matched=true` for every real gradient call observed (the correction evidence for the
    "campaign-blocking" retraction)
- Pushed to Dropbox, single consolidated folder (no new subfolders per finding):
  `dropbox:Gravity robustness/Analysis/Server Output/cm-meanzc-frechet-outer-production-closeout-2026-08-06/`
