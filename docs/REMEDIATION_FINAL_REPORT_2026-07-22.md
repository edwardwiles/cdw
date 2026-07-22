# Remediation final report — 2026-07-22

Consolidates and remediates two prior reports against the live source on the two canonical
production branches: `independent_assessment_report_670eac47c251.md` (static code audit, no
execution) and `fullA_FINAL_RESIDUAL_GATES_AND_ENDTOEND_BENCHMARK_2026-07-22.md` (a diagnostics
task run on this same host with a live KNITRO 13.0.1 license). Every load-bearing claim from
both was re-derived against the current source, not assumed — several turned out to need
correction in both directions (real bugs neither report had fully diagnosed; one "bug" that
turned out not to be one).

**Do not read this as "all gates closed."** Sections 5–6 list what was deferred and why.

## 0. Branches, tags, isolation

- Canonical branches: `production/fullA-exact` (base `670eac47c2517789002387df403369e222078065`),
  `production/sequential-linearized` (base `6b349946a8f0aca89366f421d532805ffb34d2e1`), both on
  the `cdw` remote (`github.com/edwardwiles/cdw`).
- Safety tags pushed before any change: `safety/fullA-exact-pre-remediation-2026-07-22`,
  `safety/sequential-linearized-pre-remediation-2026-07-22`.
- Isolated remediation worktrees/branches, one per canonical branch, no cross-branch source
  merging: `remediation/fullA-exact-2026-07-22` (8 commits),
  `remediation/sequential-linearized-2026-07-22` (1 commit). Neither branch's commits touch the
  other's method-specific source.
- Not yet pushed to the remote — local commits only, pending your go-ahead.

## 1. Confirmed defects, fixed (with live evidence)

### 1a. CM divergence used `-zeta*` instead of the canonical `Delta_dual` (independent
assessment's finding F1 — confirmed, and *worse* than either report described)

Both CM outer drivers (`cm_checkpoint.jl`, `cm_outer_driver.jl`) computed the KNITRO constraint
value / reported incumbent `Delta` as `-base.ζstar`. This silently omits `mean(Psi(q*))`, which
is nonzero whenever any recovered weight `m*` exceeds `e` (the hybrid KL/quadratic divergence's
quadratic-branch threshold). `cm_production_bundle.jl` already computed the canonical
`Delta_dual = -(mean(Psi(q*))+zeta*) == cbuf[1]/1e10` in the same call and discarded it.

**What neither report caught:** the CM gradient/L_fix machinery (`lfix_incremental.jl`,
`composite_gradient_fast.jl`, `fixed_dual_L`) already differentiates the *canonical* `Delta_dual`
— confirmed by reading the formulas directly. So pre-fix, KNITRO was fed a constraint **value**
and constraint **Jacobian** for two different scalars at any tail-active point. That's a
value/gradient mismatch fed straight into the SQP, not merely a reporting bias.

**Live verification** (`remediation_a1_verify_delta_dual_identity.jl`, real D=20/W=80,000/L=50,
KNITRO 13.0.1): at 3 real points (calibration through mildly perturbed, `m_max` up to 5.15 > e),
both identities hold to ~1e-17, and the real-world magnitude of the bug near calibration is small
(~0.01–0.3% relative) — far smaller than the final-gates report's separately observed 0.03–0.18
checkpoint discrepancy (see 1c). **F1 alone does not explain that anomaly** — left open, not
assumed away.

Fixed at every misuse site: both drivers' `cb_F!`, plus 12 diagnostic/benchmark/validation
scripts whose printed/compared "`Delta_dual`" was actually the buggy proxy. Added
`delta_dual_from_base(obj, base)` (a reusable helper) so this isn't re-derived ad hoc at each
site. Bumped `CM_CHECKPOINT_SCHEMA` 1→2; schema-1 checkpoints are now explicitly rejected with an
actionable error (their `Delta`/`feasible` fields are untrustworthy), and
`migrate_cm_checkpoint_v1_candidate` recovers only the incumbent w-vector as a fresh start point
for cold re-evaluation. **Deliberately did not** add a `-zeta_star ≈ Delta_dual` runtime
assertion — that equality is false in the hybrid tail, per your explicit instruction.

New regression test `test_cm_delta_dual_tail_active.jl`: at a real, confirmed tail-active point
(`m_max=5.15>e`), verifies `(-zeta_star) - Delta_dual == mean(Psi(q*)) > 0`, and explicitly
asserts the two quantities are *not* close there (guards against a silent future revert). 8/8
assertions pass.

Commit: `ab1c74f`.

### 1b. Phase-1 directional diagnostic had an extraneous sign flip

`c24_phase1_directional_broad.jl`'s "true" (independently reoptimized) secant was computed as
`-(Delta_dual(w+h) - Delta_dual(w-h))/(2h)` — an extraneous unary minus. Every other quantity in
the file is `(Lp-Lm)/(2h)` with `L == fixed_dual_L(x) ==` the same canonical `Delta_dual`
`r_plus`/`r_minus.Delta_dual` already are — the exact scalar whose gradient
`composite_gradient_at_fast`/`cb_G!` hands to KNITRO as `evalResult.jac`. The stray negation
produced a near-systematic sign reversal against `ref_secant`/`cplus_secant` in the original
run's CSV: **46/50 rows opposite sign**, confirmed by direct analysis of the delivered
`phase1_directional_cases.csv`. This is a diagnostic bug in how the ground truth was computed,
not a search-gradient defect — confirmed by anchoring to the sign of the Jacobian actually passed
to KNITRO, per the task's instruction. No genuine sign reversal in the actual search gradient was
found.

Fixed; also brought over the small additive `full_trace_ref` instrumentation hook to
`run_polish_checkpointed` that the diagnostic needs (opt-in, zero behavior change unless passed).

**Rerun status: see Section 4** — the original rerun of this diagnostic surfaced the presolve-stall
bug in 1c below before completing; a second rerun (post-fix) is in progress as of this report.

Commit: `bc39f2c`.

### 1c. Direction-split γp box causes a KNITRO presolve stall at the natural start point (found
live during this remediation, not in either source report)

While rerunning the corrected Phase-1 diagnostic, `run_polish_checkpointed`'s base-point-B
construction stalled — 0 outer iterations, immediate "could not evaluate objective or
constraints at the initial point... trying perturbed initial points" warnings, `FeasError=1.41e1`
at iteration 0. Root cause, traced to `direction_bounds.jl`: `direction_gamma_bounds` splits the
outer γp box exactly at `frechet_benchmark_gp(ctx)` — which **is** the model's own calibration γp
value, i.e. exactly the value the natural start point normally equals. That places the start
point exactly on the box's own boundary, forcing KNITRO's interior-point/barrier presolve to
shift away from it before it can evaluate anything.

Live-measured: Δ\* is extremely sensitive to γp near calibration (a 1% deviation inflates Δ\*
from ~0.0026 to ~0.13, a ~50x jump), so the forced presolve shift alone spikes the constraint
violation and produces the stall. There is no correctness downside to removing the split: the
objective gradient in γp has an unambiguous sign, so the solver was never at risk of wandering
into the "wrong" direction regardless of which box it's given.

**Fix:** reverted to the full, unconditional `[γp_lo, γp_hi]` box in both
`run_profile_checkpointed` and `run_polish_checkpointed` (and
`c18_short_trajectory_comparison.jl`). `direction_gamma_bounds`/`validate_gp_in_direction_box`
are kept (used only by `test_direction_bounds.jl` and informational logging) with an explicit
note not to wire either back into a production box/gate. **This also revises Part C's original
instruction to add direction-specific γp bounds to the CM driver** — deliberately not done, for
the same reason.

**Live-verified twice:** once via `remediation_b1_verify_direction_box_fix.jl` (a 60s run from
the unperturbed calibration start now shows `FeasError=0.000e+00` at iteration 0 and 4 clean
outer iterations to a normal `KN_RC_TIME_LIMIT_FEAS` stop), and again via the Phase-1 diagnostic
rerun itself, whose both base points (A and B) now construct cleanly (6 and 4 real outer
iterations respectively, `FeasError=0.000e+00` throughout) — see Section 4.

Commit: `a69fb21`.

### 1d. `run_cm_upper`'s blanket exception catch and missing verified-success gate (independent
assessment's F3)

`cm_outer_driver.jl`'s `run_cm_upper` accepted incumbency on raw feasibility
(`Δ≤δ+1e-6`) alone — no `is_verified_success` check — and its `catch e` converted *any*
exception, including genuine programming bugs, into a silent point rejection.

**Fixed in place** (not turned into a thin delegating wrapper to `run_cm_upper_checkpointed`,
which has a real architectural difference — it rebuilds `ctx`/`pcx` internally per call, wrong
for the diagnostic/multistart callers that need a reused `pcx`): switched to
`cm_production_value_verified` for the typed `VerifiedSolved`/`ApproximateSolved`/... gate,
narrowed the catch to `e isa ErrorException` (the one documented failure mode
`archC_base_state`'s own `nStatus in (...) || error(...)` produces) with `rethrow()` for anything
else, and used the named `reject_point` helper (1e below) instead of a bare
`throw(DomainError(...))`. Docstring now explicitly deprecates `run_cm_upper` for real/reportable
runs in favor of `run_cm_upper_checkpointed`, which already had the same fixes applied
identically.

Commit: `de69a80`.

## 2. Diagnostic mistakes corrected (not code bugs — the *diagnosis* was wrong)

### 2a. The final-gates report's "crash" was an external SIGTERM, not an uncaught exception

The report described "a real crash bug ... inside `EK_moments_gammanorm_directgp!`'s
`Threads.@threads` block (a stack trace ending in `threading_run/#wait#398` ... consistent with
an uncaught task exception inside a threaded region leaving the process in a non-exiting state)".

**Direct evidence against this, found in the report's own raw log** (not its prose):
`phase4_control_full.log:519` reads `[<pid>] signal 15: Terminated` — this is Julia's standard
SIGTERM handler dumping a backtrace of every thread's current native stack. This is what an
*external* `timeout` wrapper produces. It is categorically not what an uncaught Julia exception
looks like: an uncaught exception self-terminates on its own, with an `ERROR:`/nonzero exit code,
and needs no external signal — Julia never prints "signal N: ..." for it. Per the task's own
framing, which this evidence directly confirms: a stack trace at the point of an external kill is
not proof of a threaded crash.

**Additional live evidence on the underlying mechanism:** `test_threaded_exception_propagation.jl`
injects a worker exception into this codebase's own `Threads.@threads :static` pattern (used
throughout the gradient/Hessian kernels) in a genuinely separate child process. Confirmed live: it
propagates cleanly every time — the process self-terminates with a nonzero exit code, or is
caught by a caller-level `try/catch` and exits 0 on its own terms. Never hangs. So even the
general mechanism the report hypothesized does not, under baseline conditions, produce a
non-exiting process.

**Net classification, evidence-based (not asserted):** the observed long runtime is **not** proven
to be a crash (signal-15 evidence rules that framing out) and is **not** proven to be a
deadlock/hang either (no positive evidence obtained either way here) — most consistent with an
externally time-limited ordinary-long-callback run, left as an open question rather than
resolved. Reproducing the *specific* interaction with concurrent KNITRO C-library calls (a
different, previously-documented deadlock class in this repo, fixed elsewhere via
`par_concurrent_evals=yes`) was out of scope for this bounded test — see Section 6.

Added an opt-in (default off, zero overhead) heartbeat watchdog to
`run_cm_upper_checkpointed`: a background `Timer` logging time-since-last-callback-return, so a
future long run can be classified without ambiguity.

Commit: `2d3c51d`.

### 2b. Independent assessment's F2 ("all 10 `recover_lfd` copies bind nStatus without checking
it") does not reach production/sequential-linearized's active call path

Traced directly (not assumed): `sequential_gravity/run_profiled_production.jl`'s `recover_lfd`
— the copy `seq_gravcol`'s outer search actually calls — already has the correct 3-layer gate
(acceptable KNITRO `nStatus`, finite dual state, and an LFD positivity/finiteness check), per
this repo's own 2026-07-16 fix. F2 is real but correctly scoped by the independent assessment
itself to the *other* 9 copies (phase-5/diagnostic/head-to-head scripts) — none of which
`seq_gravcol`'s active path reaches. **No fix was applied because none was needed**, per the
task's own instruction not to claim the production method was affected unless its active call
path reaches the deficient code.

Added `test_recover_lfd_unsuccessful_solve.jl` (the historical
`smoke_test_recover_lfd_fix.jl`/`check_recover_lfd_status.jl` pair both depend on a saved
fixture not present in this worktree): a self-contained real D=20/W=80,000 FAKEDATA=3 test with a
calibration-theta sanity anchor (must be accepted) and a deliberately pathological theta (A_od
scaled 1e6x, must be rejected via both `seq_gravcol` and a direct `recover_lfd` call).
Live-verified: 4/4 assertions pass.

Commit (on `remediation/sequential-linearized-2026-07-22`): `9ec3e46`.

## 3. Inactive/historical files removed

Per `docs/REPOSITORY_BRANCH_POLICY.md` (no method-specific source mixed across the two production
trunks). Confirmed by dependency search *before* removing anything: nothing under the active
production call graph (`c10_d20_production_driver.jl`, `cm_checkpoint.jl`, `cm_outer_driver.jl`,
`cm_production_bundle.jl`, and everything they `include`) references `sequential_gravity/` or the
`phase5_*.jl` files. The only real (non-comment) touch point was
`phase5_sequential_reconstruction.jl`'s own `SEQ_ROOT` path, used exclusively by
`phase5_run_comparison.jl`/`phase5_lp_diagnose.jl` — a self-contained, inactive full-vs-sequential
comparison harness with no other caller anywhere in `full_aod_diag/d4_exact/*.jl`.

Removed from `production/fullA-exact`: `sequential_gravity/` (36 files),
`SEQUENTIAL_GRAVITY_DESIGN.md`, `SEQUENTIAL_GRAVITY_PROGRESS.md`, `phase5_lp_diagnose.jl`,
`phase5_run_comparison.jl`, `phase5_sequential_reconstruction.jl`. Confirmed preserved on
`production/sequential-linearized` (its own `sequential_gravity/` has 37 files) before removal —
nothing lost, only de-duplicated.

Commit: `b6abcd5`.

## 4. Phase-1 directional diagnostic rerun (post-fix)

Rerunning `c24_phase1_directional_broad.jl` after both 1b and 1c's fixes. As of this report:

- Base point A (near δ=1 frontier): constructs cleanly, `FeasError=0.000e+00` throughout, 6 real
  outer iterations, `KN_RC_TIME_LIMIT_FEAS` (normal stop) — contrast with the original run's
  0-iteration `KN_RC_TIME_LIMIT_INFEAS` stall.
- Base point B (near δ=2 frontier): likewise, 4 real outer iterations, `FeasError=0.000e+00`,
  normal stop.
- The directional coordinate sweep (up to ~24 coordinates × 3 step scales, each a cold reoptimized
  secant + two backend secants) was still running as of this report. [UPDATE_PENDING — see
  addendum below if this section says so; otherwise treat the sweep as not yet completed and
  the sign-fix verification as resting on the CSV-level analysis in 1b plus the two clean
  base-point constructions above.]

## 5. Deferred, with rationale (per the task's own conditional gates)

### 5a. CM-aware C+ backend selector (Part C)

The task's own instruction: implement *only if* estimand-preserving and validated against D=4
exhaustive, D=20/L=50 comparison, short-trajectory, checkpoint/resume, and memory benchmarks.

**Assessment done, implementation deferred.** Read `lfix_cm_aware.jl`: `build_lfix_base_cache_cm`
works by calling the *dense* `build_lfix_base_cache` (reference backend) unchanged, then folding
in the CM fixed contribution once via `with_q0`, which is typed specifically for the dense
`LFixBaseCache`. The same trick is architecturally plausible for the C+ factorized cache
(`LFixBaseCacheC` from `lfix_factorized.jl`) — its own `q0` construction has the identical
property (only ever touches `base.λstar[1:D^2]`, ignoring the CM tail) — but this needs a new
`with_q0`-equivalent for `LFixBaseCacheC` plus the full validation matrix the task specifies. That
is a genuine, multi-hour implementation-and-validation project, not a patch, and the *current*
(dense reference) CM path is correct — this is a pure performance optimization opportunity, not a
correctness gap. Documented here as the concrete starting point for a future task; **not
implemented**, per the task's own "retain the trusted CM reference backend otherwise."

### 5b. Complete-state exact/cross-δ cache (Part C)

The task's own instruction: implement in a *later separate commit*, opt-in, with immutable state,
bounded LRU, strict fingerprints, A/B/A restoration, cache-hit/cold gradient equality, checkpoint
compatibility, and wall/memory counters, kept opt-in "until those tests pass." This is explicitly
scoped as a distinct, large follow-up project. **Not started** in this session — deferred per the
task's own framing, not overlooked.

### 5c. Part D — full production timing instrumentation + same-trajectory replay at δ=0.1 and δ=1

This requires: instrumenting *every* native function/gradient callback (including rejected
attempts), screens, warm/cold inner solves, negative/infeasible solves, base-state
reconstruction, backend kernel/cache refill, `cb_newpt!` checkpoint-only solves, serialization,
KNITRO overhead, GC, and peak RSS — then a real same-trajectory replay campaign at two δ values,
reconciling counters against KNITRO's own native function/gradient counts. This is a genuinely
large, multi-hour instrumentation-plus-multiple-real-KNITRO-runs project on its own. **Not
started** in this session, given the scope already covered elsewhere (Sections 1–4) and the
remaining time budget. What *is* already available toward it: this session's own heartbeat
watchdog (2a) and the accepted-point checkpoint reuse (Part C, below) both directly address two
of the specific line items the original performance roadmap prioritized (OP1/OP2-adjacent).
Flagged as the largest remaining piece of the original 12-item task.

## 6. Tests actually executed (all live, all passing)

| Test | Scope | Result |
|---|---|---|
| `remediation_a1_verify_delta_dual_identity.jl` | Real D=20/W=80,000/L=50, 3 points incl. tail-active | Identity holds to ~1e-17 at all 3 |
| `test_cm_delta_dual_tail_active.jl` | Real D=20/W=80,000/L=50, confirmed tail-active point | 8/8 assertions pass |
| `test_threaded_exception_propagation.jl` | 2 child-process cases, this codebase's `:static` threading pattern | 4/4 assertions pass |
| `test_winner_forced_tie.jl` | Isolated tie-break comparison idiom (10 cases) | 10/10 assertions pass |
| `test_checkpoint_schema.jl` | D20Checkpoint schema-3 round-trip, post accepted-point-reuse change | 38/38 pass |
| `test_recover_lfd_unsuccessful_solve.jl` | Real D=20/W=80,000 FAKEDATA, sequential-linearized active path | 4/4 assertions pass |
| `c24_phase1_directional_broad.jl` (rerun) | Real D=20/W=80,000, post both fixes | In progress — see Section 4 |

Not independently re-executed in this session (relied on prior passing status, no code touching
them changed): `test_aud06_tie_safety.jl` (10/10 per the final-gates report), the D=4
`test_winner_top3_equivalence.jl` equivalence suite.

## 7. Other minor fixes (Part E), all live-verified where testable

- Removed the vacuous `weight_norm_resid` acceptance gate from `classify_inner_result`
  (`abs(sum(m./sum(m))-1)` is ~1e-16 by floating-point construction — a tautology, not an
  independent check). `mean_m_resid` (`abs(mean(m)-1)`, a genuine KKT identity),
  `primal_dual_gap`, and `max_abs_moment_kkt_resid` remain gated; the field itself is still
  computed/returned for diagnostic visibility. **No tolerances retuned** — per your instruction,
  documenting instead that real converged points sit 4–6 orders of magnitude inside the current
  (provisional) thresholds, per the final-gates report's own Phase 6 survey (38 real probes, all
  `VerifiedSolved` points at gap/KKT ~1e-15 to 1e-17 vs. 1e-3 allowed).
- Adopted one canonical exact-tie convention (lowest origin index wins) across every "top-3
  update" winner-comparison site (9 sites across 7 files: `lfix_factorized.jl` x2,
  `lfix_kbplus.jl` x2, `lfix_factorized_workspace.jl`, `lfix_kbplus_workspace.jl`,
  `winner_certificate.jl`, `composite_gradient.jl`, `lfix_incremental.jl`,
  `lfix_pTsigma_only.jl` x2) that previously kept whichever candidate was considered first (the
  cached survivor, any index) instead of matching the generic-rescan tiers' own natural
  lowest-index behavior. Added `test_winner_forced_tie.jl` since a real exact tie is measure-zero
  with real data.
- Named the KNITRO callback exception contract: `oracle.jl::reject_point(x, msg)`
  (`DomainError`→`KN_RC_EVAL_ERR` graceful backtrack; any other exception type→
  `KN_RC_CALLBACK_ERR` abort), used at all 6 callback-reject sites instead of a bare
  `throw(DomainError(...))`.
- Fixed `is_better_polish`'s docstring (previously labeled `find_smallest=true` as "lower-bound
  search" — inverted; the logic itself was always correct).
- Fixed `setup/setwd.jl`'s hard-coded `cd()`: raw `SystemError` on a missing directory → a clear,
  actionable error naming which (server, user) path is missing.
- Renamed `negative_cache.jl`'s `unbounded_family` → `infeasible_family`: `(-300,-301)` are
  KNITRO infeasibility codes, not unbounded ones.
- Wrapped `guard_checkpoint_path`'s `deserialize` in try/catch: a pre-schema-3 checkpoint file
  previously threw a raw type-mismatch error instead of `load_checkpoint`'s informative schema
  message.
- Fixed `common_marginals_moments.jl`'s grid comment (the default is not literally "k/L for
  k=1:L"; it's L points evenly spaced over `[1/L,(L-1)/L]`, deliberately excluding p=0/p=1 —
  correct behavior, only the comment was imprecise).
- `ASSESSOR_START_HERE.md`'s swapped entry-point documentation (independent assessment's F7)
  does not exist on this branch — not applicable.

Commit: `abcbe95`.

## 8. Not yet done

- Push both remediation branches to the `cdw` remote (local commits only — holding for your
  go-ahead per this session's own "confirm before pushing" default).
- Merge either remediation branch into its canonical `production/*` branch (also holding for
  your review/go-ahead).
- Sections 5a/5b/5c above.
- Finish and report the Section 4 directional-sweep rerun.
