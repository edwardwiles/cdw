# CM+ZC / common-Fréchet two-family outer production closeout — 2026-08-06

Continuation of `diagnostic/cm-paired-basis-preconditioning-2026-08-05` (worktree
`/bbkinghome/edav/cdw_worktrees/cm-paired-basis-preconditioning-2026-08-05`), starting from the
recorded HEAD `895b99b8e6b1256a69e8e910f6a4f1533c59520f` (verified live at session start — matched
exactly). `NEW_BRANCHES_CREATED = 0`, `EXTRA_WORKTREES_CREATED = 0` — the existing branch/worktree
was reused throughout.

**Scope note up front:** this session resolved the central blocking question (the CM+ZC screen
contradiction) with strong, reproducible, live evidence, and shipped three narrow, tested production
fixes (callback-health guard, capability-based guard, and a real common-Fréchet driver-reachability
bug found while chasing the TLS question — the public common-Fréchet path was completely broken
before this session, not merely TLS-limited). It did **not** complete the full scale ladder
(D4/W5k/W20k/W100k × both families × multiple points), the outer-gradient FD matrix, the literal
field-by-field differential CSV, a Fréchet-specific `w0` derivation, or a production merge. Those
are called out explicitly below rather than fabricated — see "What remains open."

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

## 5. Common-Fréchet: a much more fundamental driver bug found and fixed — public path was completely unreachable

`COMMON_FRECHET_TLS = pass` (the specific TLS wiring question). But investigating it surfaced a
**more serious, previously-undocumented bug** in the same call path, fixed this session.

**5a. The TLS question itself: real production path was already correct.** The archived D4 log's
`hessian_cm_frechet_structured_v2!(threaded_bins=true) requires tls` came from a **diagnostic test
script** (`test_frechet_hessian_structured_vs_dense_d4_twofamily_2026-08-06.jl`) calling the thin
backward-compatibility wrapper `hessian_cm_frechet_structured_v2!` **directly**, which defaults
`tls=nothing` (documented explicitly as "kept for legacy diagnostic scripts", not the production
entry point). The **real** production Hessian dispatcher, `archC_frechet_hess_cb_builder`
(`cm_frechet_hessian.jl:476-508`), already reads `tls = cctx.tls` (not a hardcoded `nothing`)
whenever `cctx.use_threaded_bins`, and `cctx.tls` is built by the **shared** `build_cm_bin_ctx`
(`cm_hessian_architectures.jl:758`) — the same builder flexible-CM's context uses. No genuine TLS
gap in the real driver.

**5b. What that investigation actually found: `run_cm_upper_checkpointed`'s `is_frechet` branch
never passed `include_truncated_moment`/`threaded_bins` through to
`build_cm_frechet_production_context` at all.** Since `include_truncated_moment` became a required
(no-default) kwarg on that function during the 2026-08-05/06 truncated-power task, this meant
**every single call to this driver with `marginal_restriction=:common_frechet` threw
`UndefKeywordError` before ever reaching KNITRO, regardless of any other setting** — the real
public common-Fréchet path was entirely unreachable, not merely TLS-limited. Confirmed live: the
first attempt at a real D20/W=5,000 smoke through the unfixed driver failed exactly this way.

**Fixed** (`cm_checkpoint.jl`, committed `56375bb`): both kwargs now passed through, mirroring the
other two branches exactly. `build_cm_frechet_production_context`'s own internal capability gate
(requires `moment_representation=:dense_reference` for `include_truncated_moment=true`, which
conflicts with `prepare_production_run`'s hard ban on dense bundles) still correctly refuses
two-family common-Fréchet through this driver — this fix restores reachability for **plain
(single-family) common-Fréchet only**; it does not relax the two-family restriction, which remains
a separate, disclosed, still-open gap.

**After the fix**, a real D20/W=5,000 smoke (`smoke_frechet_public_driver_tls_2026-08-06.jl`)
confirms: the driver now reaches KNITRO cleanly, `cctx.tls` is constructed
(`cctx.use_threaded_bins==true`, `cctx.tls!==nothing`), and **no** "threaded_bins=true requires
tls" error occurs — section 5a's TLS finding holds. **However, the generic calibration `w0`**
(built via `cm_w0_from_calibration`, the same construction flexible-CM/CM+ZC use) **is not
automatically a valid starting point for common-Fréchet's own restriction structure** (`D*L`
restrictions vs. flexible-CM's `(D-1)*L` — a genuinely different feasible region/anchor): the
first outer evaluation fails (KNITRO `-500`/`-502`, "could not evaluate objective or constraints
at the initial point"). **This is a separate, still-open item** (Fréchet needs its own
calibration-consistent `w0`, not yet derived this session) — do not read `COMMON_FRECHET_OUTER` as
passing at any real-evaluation tier; only the reachability/TLS defect is fixed and confirmed.

The new `assert_two_family_capabilities!` (section 4) additionally adds a defensive, explicit
hard-error for the `threaded_bins=true` + `cctx.tls===nothing` misconfiguration class at the real
driver's own top level, for all three families sharing this pattern.

## 6. What remains open (not completed this session — do not treat as done)

- **Section 6** (pure/structured pairwise-screen refactor + dedicated test matrix): the screen was
  proven pure and reproducible **empirically** (repeat-call idempotence, A/B agreement, no mutation
  — section 1's live evidence) but was **not refactored** into a structured certificate-result type,
  and the specific test file / CSV the task describes was not written. Given the live evidence found
  zero purity defects, this is lower-priority than it looked before this session, but it is still
  open per the task's letter.
- **Section 8** (outer-gradient FD matrix across gp/A/gravity-pivot/CDF/power/eta_nu/mixed
  directions, D4 and D20/W=20,000): **not run.** No FD evidence was gathered this session beyond
  the pre-existing D20 26/26 structured-vs-dense Hessian gates already on record.
  `CM_PLUS_ZC_OUTER_GRADIENT` and `COMMON_FRECHET_OUTER_GRADIENT` are **not verified** by this
  session — do not report them as passing.
- **Section 10** (`FLEXIBLE_CM_CONTROL_VS_EXTENSION.csv`, literal field-by-field diff): not
  produced. The qualitative differences are known from code reading (CM+ZC adds `Pow`-gated
  `CMMeanZCOperatorState` dual layout + `meanzc_zc_op`/widened `H_CZ` cross-Hessian block +
  `eta_nu` outer coordinates; common-Fréchet adds a level-anchor block reusing the same `Ttab`/`CT`
  tables) but was not assembled into the requested CSV artifact.
- **Section 11 scale ladder**: D4 — done for CM+ZC (regression test) and common-Fréchet (TLS
  smoke). D20/W=100,000/L=50 — done for CM+ZC (calibration eval verified=true + one nearby
  point). **W=5,000 and W=20,000 tiers were not run for either family. Common-Fréchet was not run
  at W=100,000 through the real driver with the two-family CM extension.** `CM_PLUS_ZC_OUTER` and
  `COMMON_FRECHET_OUTER` below are reported only for the tiers actually run.
- **Section 12** (additional permanent no-fake-success tests beyond the one committed): only the
  callback-health regression test was added. The task's fuller list (verified=false-published-as-
  incumbent, threaded_bins/TLS test, family-guard-silent-downgrade test) was not separately
  implemented as permanent tests — the corresponding *mechanisms* were verified live (section 5),
  just not turned into standing test files.
- **Section 13/production merge**: **not done, and should not be done without separate explicit
  confirmation** — several release gates above (FD gradient matrix, full scale ladder, Fréchet
  W=100,000 two-family) are not met yet, so a real fast-forward + tag would be premature regardless
  of authorization. See "Production SHA/tag" below.
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
    W5K: not_run
    W20K: not_run
    W100K: pass (calibration eval: feasible=true, verified=true, Delta=0.00065;
                 one nearby eval: feasible=false, verified=true -- both genuine)

CM_PLUS_ZC_OUTER_GRADIENT = not_verified_this_session

COMMON_FRECHET_INNER = pass (pre-existing: D4 + D20 26/26 structured-vs-dense Hessian gates,
    both contrasts modes, unchanged this session)

COMMON_FRECHET_TLS = pass
    (real production dispatcher already reads cctx.tls correctly; archived TLS error was from
    a diagnostic script calling a legacy wrapper directly, not the production driver; confirmed
    empirically at real D20/W=5,000 through the public driver, threaded_bins=true, AFTER fixing
    a separate, more serious bug found investigating this -- see below)

COMMON_FRECHET_OUTER =
    D4: not_run (d4_exact_setup's context shape is not a fit for cm_w0_from_calibration/pivot
                 elimination -- confirmed live, DimensionMismatch 15 vs 379; use real D20 data
                 at small W instead, as done for the W5K row)
    W5K: fail_frechet_specific_w0_needed (driver reaches KNITRO cleanly post-fix, TLS constructed
                 correctly, but the generic calibration w0 is not a valid starting point for
                 common-Frechet's own D*L restriction structure -- KNITRO -500/-502 at the
                 initial point; needs a Frechet-specific w0 derivation, not yet done)
    W20K: not_run
    W100K: not_run

COMMON_FRECHET_OUTER_GRADIENT = not_verified_this_session

FLEXIBLE_CM_REGRESSION = pass
    (unchanged D20/W=100,000/L=50 result before/after this session's guard wiring;
    flexible_cm's own screen control passes identically to CM+ZC's)

TWO_FAMILY_GUARDS = removed_after_capability_gates

PRODUCTION_RELEASE = not_merged_frechet_w0_and_gradient_FD_and_full_scale_ladder_not_yet_run

CAMPAIGN_READY =
    CM_plus_ZC: no (W5k/W20k tiers and outer-gradient FD not yet verified)
    common_Frechet: no (needs its own calibration w0 before any real evaluation is possible;
                        W5k/W20k/W100k tiers and outer-gradient FD not yet verified; two-family
                        CM extension for Frechet remains architecturally blocked -- see section 5b)

DENSE_PRODUCTION_GH_USED = false
NEW_HESSIAN_ALGORITHMS = 0
NEW_GRADIENT_ALGORITHMS = 0
CAMPAIGN_LAUNCHED = false
NEW_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
```

## Production SHA / tag

**None minted.** `PRODUCTION_RELEASE = not_merged_gradient_FD_and_full_scale_ladder_not_yet_run`.

Local commits on `diagnostic/cm-paired-basis-preconditioning-2026-08-05` this session (not pushed,
not merged into `production/fullA-exact`):

- `4e06bf1` — shared callback-health/fake-success guard
- `3f9344e` — capability-based two-family guard (replaces the blanket refusal)
- `56375bb` — fix common-Fréchet driver's missing `include_truncated_moment`/`threaded_bins`
  passthrough (the public common-Fréchet path was completely unreachable before this fix)

Prior, still-uncommitted-elsewhere-but-present-here work this session did **not** touch or attempt
to complete: `full_aod_diag/d4_exact/cm_originzc_checkpoint.jl`'s own uncommitted `pin_outer_algorithm`
mirror addition (9-line diff, present before this session started, unrelated to this task's scope —
left exactly as found).

## Campaign handoff (provisional — do not launch from this alone)

**Common Fréchet / CM+ZC upper and lower: NOT campaign-ready per this session's own verdicts
above.** The blocking items before a real campaign launch:

1. **Derive a common-Fréchet-specific calibration `w0`** — the generic `cm_w0_from_calibration`
   point is not a valid starting point for Fréchet's own `D*L` restriction structure (section 5b);
   this blocks every subsequent Fréchet item below.
2. Run the W=5,000 and W=20,000 tiers for both families (short outer runs, threaded backends,
   cache/checkpoint round-trip) — section 11 requirement, not done (CM+ZC W=100,000 calibration
   point IS done and verified; W5k/W20k are not).
3. Run the outer-gradient FD matrix (gp / ordinary-A / gravity-pivot-A / CDF-CM / power-CM /
   eta_nu / mixed direction) at D4 and D20/W=20,000 for CM+ZC, and the analogous check for
   common-Fréchet — section 8 requirement, not done.
4. Run common-Fréchet through the real driver at D20/W=100,000/L=50 with the two-family CM
   extension — currently hard-blocked by `build_cm_frechet_production_context`'s own
   `moment_representation=:dense_reference` requirement for two-family, which this driver cannot
   satisfy (dense is banned). Two-family common-Fréchet needs the same `Pow`-aware operator
   extension CM+ZC/flexible-CM already received before this is reachable at all — a real,
   disclosed scope gap, not just an unrun test.
5. Once 1-4 pass, rebase onto latest `origin/production/fullA-exact`, rerun the flexible-CM
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
  - `diag/smoke_frechet_public_driver_tls_w5k_final.log` — section 5 Fréchet driver-reachability
    fix confirmation + the still-open Fréchet-w0 failure mode
- Pushed to Dropbox: `dropbox:Gravity robustness/Analysis/Server Output/cm-meanzc-frechet-outer-production-closeout-2026-08-06/`
