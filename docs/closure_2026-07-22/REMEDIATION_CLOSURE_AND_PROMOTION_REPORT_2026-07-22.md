# Remediation closure and promotion report — 2026-07-22

Closes out the prior Claude's remediation work (`docs/REMEDIATION_FINAL_REPORT_2026-07-22.md`)
against the independent assessment and final-gates reports. This is a verification-and-
completion pass, not a rewrite: prior work is checked, not redone; genuinely missing pieces
are finished; defects found along the way are fixed and tested.

**Read this alongside**: `docs/REMEDIATION_FINAL_REPORT_2026-07-22.md` (prior Claude's own
report, still accurate for what it covers), the independent assessment
(`exports/independent_assessment_2026-07-22/...`), and the final-gates report
(`gravity-diag-fullA-final-gates/docs/fullA_FINAL_RESIDUAL_GATES_AND_ENDTOEND_BENCHMARK_2026-07-22.md`).

---

## Phase 0 — repository state and the two flagged inconsistencies

**Branches/tags** (all in the shared worktree family, `git-common-dir` = `trade_robustness_modular/.git`):

| Ref | Commit |
|---|---|
| `safety/fullA-exact-pre-remediation-2026-07-22` | `670eac47c2517789002387df403369e222078065` |
| `production/fullA-exact` (unchanged this session) | `670eac47c2517789002387df403369e222078065` |
| `remediation/fullA-exact-2026-07-22` (this session's tip) | `82dd485bb01c58ec516fe56ed280eaa928fe07ee` |
| `safety/sequential-linearized-pre-remediation-2026-07-22` | `6b349946a8f0aca89366f421d532805ffb34d2e1` |
| `production/sequential-linearized` (unchanged) | `6b349946a8f0aca89366f421d532805ffb34d2e1` |
| `remediation/sequential-linearized-2026-07-22` (unchanged this session) | `9ec3e46b5c37e8d5c0c1342883c8df7f82e8e7c6` |

Remotes: `cdw` → `github.com/edwardwiles/cdw` (canonical), `origin` → `github.com/habibiscoding/Trade-Model-Robustness`.
Both canonical branches were, at session start, identical across `origin`/`cdw`/local — confirmed
by direct `git rev-parse` comparison before any change.

### Inconsistency 1 — "8 commits, 7 itemized"

At the moment the prior Claude wrote `docs/REMEDIATION_FINAL_REPORT_2026-07-22.md` (commit
`2cf9423`), exactly 8 commits preceded it on this branch: `ab1c74f`, `bc39f2c`, `2d3c51d`,
`abcbe95`, `a69fb21`, `b6abcd5`, `de69a80`, `9da9e89` — so "8 commits" (report §0) was literally
correct. The report's body gives a numbered subsection + explicit `Commit:` citation to 7 of
them (§1a–d, §2a, §3, §7); the 8th, `9da9e89` ("Accepted-point checkpoint reuse in cb_newpt!"),
is real, committed, and referenced in the Section 6 test table ("post accepted-point-reuse
change") and Section 4/8, but never given its own lettered subsection. **Resolution: a
write-up completeness gap, not a missing or phantom change.** Verified in Phase 3F below.

### Inconsistency 2 — "an implemented accepted-point-reuse change ... not identified"

Identified: commit `9da9e89441395ec9989d8a47e0c1130f8ca59135`, "Accepted-point checkpoint reuse
in cb_newpt! (Part C, item 9)". Full diff reviewed; behavior traced and live-tested in Phase 3F.

### Commit-to-change table (fullA-exact remediation branch, `safety-tag..HEAD`, 17 commits total)

| Commit | Change | Committed? | Verified this session |
|---|---|---|---|
| `ab1c74f` | F1 fix: canonical `Delta_dual`, not `-zeta*`; schema 1→2 | Yes | Phase 1 (below) + prior session's own live check |
| `bc39f2c` | Phase-1 diagnostic sign-convention fix | Yes | Phase 2 (below) |
| `2d3c51d` | Reclassify final-gates "crash" as external SIGTERM | Yes | Phase 4 (below) reproduces the same evidence pattern live |
| `abcbe95` | Part E bundle: tie convention, `reject_point`, `setwd.jl` (partial), misc | Yes | Phase 3D (ties), 3E (setwd — found incomplete, finished this session) |
| `a69fb21` | Remove direction-split gp box | Yes | Phase 3C (below) |
| `b6abcd5` | Remove `sequential_gravity/`+phase5 harness from fullA-exact | Yes | Phase 6 dependency check (below) |
| `de69a80` | Unify CM driver (`run_cm_upper`) onto verified architecture | Yes | Phase 3A/3B (below) |
| `9da9e89` | Accepted-point checkpoint reuse (`cb_newpt!`) | Yes | Phase 3F (below) — found missing counter, added |
| `2cf9423` | Add remediation final report | Yes (doc only) | — |
| `d083273` | Report update: c24 sweep completed | Yes (doc only) | Phase 2 (below) — independently re-verified, not just re-quoted |
| `d33ec76` | **This session**: real `setwd.jl` portability fix | Yes | Live-tested from a different checkout path |
| `5b19843` | **This session**: Phase 1 CM checkpoint reconciliation | Yes | Self-verifying |
| `9b37b62` | **This session**: `CMExpectedSolveFailure` typed exception | Yes | Self-verifying (test included in commit) |
| `9a1d541` | **This session**: Phase 2 deliverable packaging | Yes | Self-verifying |
| `5069eba` | **This session**: Phase 3F checkpoint-reuse verification + counter | Yes | Self-verifying |
| `82dd485` | **This session**: Phase 3C direction-box live validation | Yes | Self-verifying |

No required change was found only in a dirty worktree. **Dirty-worktree note**: at session
start and throughout, `results/fullA_d4/c10_ckpt_smoke_test{,_resumed}/*.jls` showed as
modified/deleted/untracked — these are regenerated byte-for-byte-varying artifacts of
re-running the tracked smoke-test scripts (`c10_prod_driver_smoke_original.jl`/
`c10_prod_driver_smoke_resume.jl`), not source changes and not this session's own doing (some
predate it). Handled in the Phase 6 cleanup pass before promotion (restore tracked files,
leave the pre-existing checked-in fixtures as committed).

Sequential-linearized remediation branch: clean working tree throughout, 1 commit
(`9ec3e46`) ahead of its safety tag, unchanged this session — its own regression re-run is
Phase 6 (below).

---

## Phase 1 — CM checkpoint/cold-solve discrepancy: RESOLVED

**Located the exact artifacts.** The final-gates report's own session scratchpad
(`/tmp/claude-181517/.../cf6a4db4-.../scratchpad/phase4/ckpt/{control,interrupt}/*_latest.jls`)
was still present on disk — the literal schema-1 `CMCheckpoint` files that produced the
reported 0.03–0.18 anomaly, not reconstructed or inferred from nearby points.

For each (`control`, `interrupt_then_resumed`):

- **Hashed** the full outer vector (SHA256 of `vcat(g, zfree)`): `control` →
  `12f3d915def0...`, `interrupt_then_resumed` → `988dc8e6ef79...`. In both cases the
  terminal iterate hash equals the `best_feasible.w` hash (both checkpoints were written at
  `:new_best`, so the two coincide here — the checkpoint schema *does* distinguish them as
  separate fields, `g`/`zfree` vs. `best_feasible.w`).
- **Confirmed same-record**: `best_feasible.gp == best_feasible.w[1]` structurally, by
  construction of the `cb_F!` literal that builds the `NamedTuple`.
- **Confirmed context compatibility**: rebuilt `ctx` via `d20_real_setup_design(W=80000,
  δ=1.0, draw_design=:pseudorandom, draw_seed=20260719)` (the exact call the generating
  script used) and compared every field the `CMCheckpoint` struct stores — `W`, `draw_seed`,
  `draw_design`, `delta`, `cm_L`, `cm_contrasts`, `cm_grid_rule`, `cm_hessian_backend`,
  `cm_basis`, `cm_probs`, and both draw checksums (`draw_checksum_uniform`/`transformed`).
  **All fields MATCH** for both checkpoints.
- **Cold-evaluated** the identical incumbent vector via `cm_production_value_verified` (cache
  disabled by construction — a fresh Architecture-C inner solve, no `evaluate_fullA`/exact-cache
  path involved) from **two materially different inner-solve starts**: the default NaN-cold-start
  and a large random perturbation (`||x0|| ~ 112`). Both reproduce `Delta_dual` to ~1e-11
  relative — the convex inner problem is reproducible, not the source of the discrepancy.

**Result** (full detail: `CM_CHECKPOINT_COLD_RECONCILIATION_2026-07-22.csv`):

| tag | reported Δ (schema-1) | cold Δ_dual | fresh −ζ* | mean(Ψ(q*)) independently recovered | \|diff\| | m_max | tail_frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 0.8025018742522267 | 0.6266708456751989 | 0.8025018742522274 | 0.17583102857702848 | **0.0** | 94.3 | 4.9% |
| interrupt_then_resumed | 0.21311465244991357 | 0.18315197169632164 | 0.2131146524499133 | 0.02996268075359165 | **3.5e-18** | 44.4 | 1.1% |

The independently-recovered `mean(Ψ(q*))` (computed **only** from the recovered weights `m*`
via the `Psi!`/`dPsi!` branch inverse — `m<=e ⇒ Ψ=m−1`; `m>e ⇒ Ψ=0.5·m²/e+0.5·e−1` — not via
the identity being tested) matches `(−ζ*) − Δ_dual` to floating-point-exact precision at both
points.

**Classification (both points, identically): "old checkpoint stored `−ζ*`, and the F1 identity
fully explains the discrepancy."** Not context mismatch, not incumbent/terminal confusion, not
inner-solve non-reproducibility — the remediation report's own live check (3 similar-but-not-
identical points, smaller magnitude) could not settle this; the exact old vectors do.

**Bug caught in this reconciliation script itself**: an incomplete transcription of `Psi!`'s
quadratic branch (missing the `+0.5·e` additive term) produced a spurious ~0.066 false
discrepancy on the first run. Traced to `cc_algo/Psi.jl`, fixed, reran — diff went to 0.0/3.5e-18.
Documented inline in the script (`c30_phase1_cm_checkpoint_reconciliation.jl`) as a caution for
future readers.

**Required tests**:
- CM schema-1 rejection has an actionable error: pre-existing (`load_cm_checkpoint`, ab1c74f) —
  confirmed still in force (both old files are schema-1; the reconciliation script bypasses
  `load_cm_checkpoint` deliberately, via raw `deserialize`, exactly per
  `migrate_cm_checkpoint_v1_candidate`'s own documented audit/comparison use case).
- Candidate-vector-only migration cold-re-evaluates: this is precisely what the reconciliation
  script does for both files (never trusts `best_feasible.Delta`, always re-derives via
  `cm_production_value_verified`).
- CM schema-2 round-trip + corrected best-incumbent cold-reproduction: covered by Phase 3F's
  live run (a real schema-2 checkpoint, `load_checkpoint` round-trip, `n_checkpoint_reuse_hits`
  confirmed) and Phase 4's shakedown (below).

---

## Phase 2 — corrected directional gate: COMPLETE (independently re-verified, not re-quoted)

The prior Claude's session completed the `c24_phase1_directional_broad.jl` sweep in a later
commit (`d083273`, "54/54, 162/162 passed") than the report body describes, leaving results in
a session-scoped `/tmp` path. Recomputed every summary statistic from the raw 54-row CSV
directly (not the commit message):

- **C+ vs trusted Reference** (identical bandwidth): median 5.6e-16, p90 5.6e-15, max 2.2e-13;
  **54/54 sign agreement**; correlation/cosine = 1.0000000000; **0** cases relying on the 1e-6
  fallback. No systematic sign reversal survived the diagnostic fix.
- **Fixed-dual (Reference) finite-bandwidth secant vs. independently reoptimized hard-objective
  secant**: median abs error 5.2e-4, p90 8.1e-3, max 1.6e-1; correlation 0.9979; **51/54 sign
  agreement** (3 disagreements are small-magnitude/near-zero-crossing/high-switch cases,
  individually inspected — not a reversal); magnitude-relative error (41 cases with
  `|true_secant|>=1e-3`, near-zero denominators excluded per that stated rule) median 5.3%, p90
  46.2%, max 55.9% at one genuinely near-zero denominator.
- Breakdowns by bandwidth (3 step scales: 0.5/1.0/2.0), winner-switch count (0 to 1837 flips),
  direction class (`optimizer`/`random_tangent`/`high_switch` — at least one of each, not only
  coordinate axes), and sign of `true_secant` all included in the deliverable CSV.

No manufactured pass threshold on the fixed-dual-vs-true comparison — reported as the honest
empirical pattern the task asked for. Sign convention: `(Delta_plus − Delta_minus)/(2h)`
throughout, anchored to the Jacobian actually passed to KNITRO (every row's `sign` column
records `both`, i.e. genuine central differences).

Deliverables: `CORRECTED_DIRECTIONAL_GATE_2026-07-22.csv`, `phase2_directional_sweep_raw_54cases.csv`,
`phase2_summary_stats.txt`.

---

## Phase 3 — audit of prior fixes

### 3A. CM value/Jacobian consistency — clean

Grep-swept every non-comment use of `ζstar` across `full_aod_diag/d4_exact/*.jl`: the only
computed-and-used-as-divergence sites are (a) the correct `q_s = −ζ−λ'G_s` formula
(`lfix_*_workspace.jl`, `test_compressed_moments.jl` — mathematically distinct from, and
unaffected by, the F1 bug) and (b) this session's own explicitly-named `neg_zeta1`/`neg_zeta2`
comparison variables in the Phase 1 reconciliation script. No active callback, gradient path,
feasibility check, trace, incumbent, or checkpoint site substitutes `-zeta_star` for the
canonical `Delta_dual`.

### 3B. CM exception handling — fixed further, live-tested

`e isa ErrorException` (the remediation task's own Part C fix) is still too broad — an
ordinary `error("...")` from a programming bug is also an `ErrorException`. Added
`CMExpectedSolveFailure <: Exception` (the one documented expected failure mode:
`archC_base_state`/`archC_verified_state`'s inner-solve-failed signal), narrowed all three
catch sites (`run_cm_upper`, `cb_F!`, final-point verification in `cm_checkpoint.jl`) to
`e isa CMExpectedSolveFailure || rethrow()`.

Live-verified (`test_cm_expected_solve_failure_typed.jl`, real D=20/W=80,000/L=50, **7/7
pass**): a genuinely pathological point (A_od and gp scaled 1e8×) raises
`CMExpectedSolveFailure` from the real production path; the catch idiom correctly
distinguishes it from an injected bare `ErrorException`/`MethodError` (both propagate); an
end-to-end monkey-patch injecting a bug into `cm_production_value_verified` and running a real
(tiny-budget) `run_cm_upper_checkpointed` confirms the run **aborts by propagating the bug**,
not by silently completing/rejecting it. (Two bugs caught and fixed in the test script itself
while writing it: a Julia top-level "soft scope" gotcha, and `(1)(2)` parsing as multiplication
rather than a call — documented inline.)

### 3C. Direction-box removal — live-validated both directions + wrong-side

Real D=20/W=80,000/delta=1 runs from the unperturbed calibration start (**7/7 checks pass**):
upper (`find_smallest=true`) still minimizes gp (incumbent gp < g_F); lower
(`find_smallest=false`) still maximizes gp (incumbent gp > g_F); both construct cleanly (>=1
real outer eval, no 0-iteration presolve stall); a start point on the *old* split's "wrong
side" (gp=0.9927, strictly inside the upper direction's old forbidden zone) is **no longer
rejected**. Doc/test audit: no live code or test implies the split is a production feasibility
restriction (`staged_delta5_realdata_validation.jl`'s own `direction_gamma_bounds` call is
confirmed informational-logging-only; historical handoff docs describing the old behavior are
left as accurate history, not edited).

### 3D. Tie-convention tests — rerun, all pass

`test_aud06_tie_safety.jl`, `test_winner_forced_tie.jl` (10/10 forced-tie cases, exact and
near), `test_winner_top3_equivalence.jl` (D=4 top-3 vs. full-rescan equivalence, 240 synthetic
cases + real coordinate sweep, 0 mismatches) — all exit 0 on the current multi-file tie diff.

### 3E. Portability — genuinely fixed this session

The remediation task's earlier pass only replaced a raw `SystemError` with a clearer message;
`setwd()` still required one of a hard-coded per-developer path to exist. Rewrote it to derive
the repository root from `@__DIR__` by default (works from any checkout, any host); the
historical per-`(server,user)` table is preserved but gated behind an explicit opt-in
(`GRAVITY_ROBUSTNESS_LEGACY_SETWD=1`). **Live-tested from a genuinely different checkout path**
(a fresh `/tmp` directory containing only a copy of `setup/`) — correctly self-locates; legacy
opt-in still resolves to the original path when the env var is set. Confirmed (unchanged):
`setwd()` is not on the active `full_aod_diag/d4_exact` test/production path.

### 3F. Accepted-point checkpoint reuse — verified, missing counter added

Traced `cb_newpt!`'s reuse logic (`shared.w == w_now ? shared.r : nothing`) against the task's
checklist: exact-match only (not tolerance-based) ✓; context fingerprint/CM config — satisfied
*trivially* by construction (`last_F_state` is a per-call closure-local `Ref`, never persisted
or shared across a different `ctx`/process, so there is no cross-context risk to check) ✓;
rejects stale/mismatched states (falls back on any inequality) ✓; a reused-but-failed result
still can't produce a checkpoint (the existing `inner_status in FEASIBLE_CODES` gate runs
identically whether `r` was reused or freshly solved) ✓; falls back safely ✓; **counter for
avoided checkpoint-only inner solves — missing, added this session**
(`n_checkpoint_reuse_hits`, threaded into both functions' return NamedTuple; `D20Checkpoint`
itself untouched).

Live-verified (`c32_phase3f_checkpoint_reuse_validation.jl`, **10/10 pass**): idiom-level
exact-match/one-ULP-mismatch/different-point/no-prior-state cases, plus a real 60s D=20 run
showing `n_checkpoint_reuse_hits=3` of 5 evals (the mechanism genuinely fires), plus a
post-change checkpoint round-trip.

---

## Phase 4 — real corrected CM checkpoint/resume shakedown

Real D=20/W=80,000/L=50/delta=1, current (schema=2, F1-fixed) code, `JULIA_NUM_THREADS=20`,
heartbeat watchdog enabled (`heartbeat_interval_s`), current schema only.

**Control arm** (uninterrupted, 240s internal budget): 4 evals, `:new_best` checkpoint at
gp=0.9686457404702599, reported (canonical) `Delta_dual=0.9013958544463128`. **Cold-reverify
(fresh process, cache disabled, single-threaded): `Delta_dual=0.9013958544463094`, `|diff| =
3.3e-15`** — i.e. now at floating-point precision, a dramatic contrast with the OLD schema-1
checkpoints' 0.03–0.18 anomaly (Phase 1 above). This is independent confirmation, on a freshly
generated real checkpoint rather than an old artifact, that the F1 fix is genuinely in force
end to end.

**Interrupt/resume**: first attempt (90s external `timeout -s TERM`) fired before the interrupt
arm had gotten past its first real eval (~90–130s real wall per eval at L=50, confirmed from
the control arm's own timeline) — no checkpoint was written yet, so the resume arm correctly
errored with its own actionable message ("no checkpoint found... interrupt arm must have
written at least one checkpoint before being killed") rather than resuming from nothing. Not a
defect — an under-provisioned kill delay in this session's own orchestration, corrected below.

**Retry** (210s external SIGTERM, evidence-based from the control timeline): the interrupt arm
completed 2 evals and wrote a real `:new_best`/`:wall_interval` checkpoint before being killed
at t=210s — confirmed via `ls -la` on the checkpoint directory and the checkpoint's own
`n_eval=2`. **The resume arm then hung.** Heartbeat evidence
(`docs/closure_2026-07-22/phase4_resume_raw.log`): last successful callback *return* at
~t=48.6s (a `cb_G!` call); by t=104.7s, 56s with no callback returning (the heartbeat's own
in-line classification: "either an ordinary long single callback ... or a stall — cannot
distinguish without per-phase timers"); **critically, no further heartbeat log line appears at
all after t=104.7s**, even though `heartbeat_interval_s=10.0` — i.e. the heartbeat's own
background `Timer` also stopped producing output, not just the compute callback. The external
`timeout 220` fired its SIGTERM; Julia's standard signal handler caught it and printed its
normal all-thread stack dump (`[pid] signal 15: Terminated`, ending mid-`KN_solve`,
`GC: 151` recorded) — **this by itself is exactly the same evidence pattern commit `2d3c51d`
used to classify the final-gates report's earlier "crash" as an external interruption, not an
uncaught exception, and it reproduces here too.** But the process did **not** actually exit
after printing that dump: it remained alive, unresponsive, for 100+ more seconds, and required
a manual `SIGKILL` (verified PID/cmdline against `ps` immediately before killing, per this
repo's own protocol) to actually terminate.

**Root-cause check requested live, during this task, by the user**: is this the previously-
fixed AUD-02 regression (`par_concurrent_evals=no` on an *outer* `.opt` file deadlocking the
architecture's always-nested outer→inner `KN_solve` pattern, `docs/fullA_nested_knitro_solve_hang_rootcause.md`,
fixed in commit `dc3196c`)? **Checked directly: no.** `csw_outer_wallclock_sr1.opt` — the exact
`.opt` file `run_cm_upper_checkpointed`'s default `opt_file` argument loads for both control and
resume — still has `par_concurrent_evals yes` (confirmed by direct `grep`, both before and
after this session's own changes; this session touched no `.opt` file). The captured stack also
lacks the AUD-02 hang's specific signature (`__kmp_acquire_queuing_lock` inside
`KTR_lsq_set_jac_callback64`); this is a plain SIGTERM-handler stack walk through ordinary
library frames. **Classification: this reproduces the separately-documented "KNITRO driver runs
can hang past their own declared timeout" behavior**
(`gravity-robustness-knitro-hang-past-timeout` note: "`timeout` wrapper doesn't reliably bound a
hung real-KNITRO driver process; poll `ps -o etime` and kill manually rather than trusting it"),
not a reversion of the AUD-02 fix. The heavy recorded GC activity (151 cycles) at the point of
the dump is also consistent with a severe GC-driven stall rather than a true lock deadlock, but
this is **not fully resolved to one definitive root cause within this task's bounded shakedown
budget** — disclosed as an open operational question, not asserted as either "crash" or
"deadlock" (per the task's own evidentiary standard).

**Cold-verify (both real checkpoints, fresh process, cache disabled, single-threaded)** —
`CM_CORRECTED_RESUME_SHAKEDOWN_2026-07-22.csv`:

| tag | reported Δ | cold Δ_dual | \|diff\| | kappa | gravity_residual | class | gap | KKT resid | context_fingerprint |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| control | 0.9013958544463128 | 0.9013958544463094 | 3.33e-15 | 0.05171 | 4.05e-18 | VerifiedSolved | 1.14e-14 | 4.16e-14 | `95d6260a...` |
| interrupt (best-checkpoint-before-kill) | 0.00878804666675019 | 0.008788046666750197 | 6.94e-18 | 0.02031 | 6.36e-18 | VerifiedSolved | 8.85e-17 | 5.79e-15 | `95d6260a...` |

Both real, freshly-written schema-2 checkpoints reverify to machine precision. The best-verified
incumbent is correctly stored and readable separately from the terminal iterate in both cases
(`CMCheckpoint.best_feasible` vs. `.g`/`.zfree` — see Phase 1's field-by-field description).

**Net assessment**: the checkpoint/cold-verify correctness chain — the thing this shakedown
exists to gate — is fully validated on real, current-schema artifacts. The interrupt→resume
*operational* path is only partially demonstrated: interrupt-with-a-real-checkpoint-written
succeeded; the immediately following resume attempt hit a real, disclosed hang requiring manual
intervention, not attributable to the known AUD-02 class. **This is flagged as an open
reliability item for unattended multi-hour CM production runs specifically** (a human or a
supervisory process must be able to detect and recover from this), not as a correctness defect
in the checkpoint format or the F1 fix. See the decision table at the end of this document.

---

## Phase 5 — end-to-end performance measurement (scoped)

**Honest scope note** (stated up front, not buried): the task's full Phase 5 spec — every
native-eval category (screens/warm/cold/infeasible/exact-cache/checkpoint-serialization/GC)
broken out separately at both deltas — is, by the remediation report's own prior assessment, "a
genuinely large, multi-hour instrumentation-plus-multiple-real-KNITRO-runs project on its own."
This session executed the highest-value, most decisive subset: backend assertion, a real
same-trajectory replay with numerical-equivalence verification, and counter reconciliation. The
full per-category breakdown (screens/warm/cold/GC/serialization split separately) was **not**
built — flagged as the remaining item, same as the original report flagged it.

**Backend assertion** (real, not assumed): `resolve_price_cache_backend("phase5", nothing,
:cplus) == :cplus` and `resolve_price_cache_backend("phase5", nothing, :buffered) == :buffered`
— both asserted and logged.

**Instrumentation added**: a small, additive, opt-in `grad_trace_ref` kwarg on
`run_polish_checkpointed` (mirrors the existing `full_trace_ref` kwarg's exact pattern) —
when passed a `Ref`, every `cb_G!` call pushes `copy(xf)` before dispatching to the resolved
backend; `nothing` (default) is zero behavior/allocation change. (An earlier attempt used a
runtime monkeypatch of `composite_gradient_at_Cplus` instead; caught live: giving the override
the *exact* same method signature made the "keep a reference to the original" alias
self-referential — Julia methods live in one global per-signature table, not per-binding — which
caused a real infinite-recursion `StackOverflowError`. The additive kwarg avoids this
architecture problem entirely and is the safer, already-established pattern.)

**Same-trajectory replay** — real D=20/W=80,000 unrestricted C+ runs, 90s each, at delta=0.1 and
delta=1.0; every captured gradient-callback state replayed under both backends
(`h_mode=:fixed, h0=0.01` for a controlled apples-to-apples bandwidth on both sides) —
`FULLA_SAME_TRAJECTORY_BACKEND_REPLAY_2026-07-22.csv` / `FULLA_CALLBACK_WALL_DECOMPOSITION_2026-07-22.csv`:

| delta | n_eval | n_grad_calls | native_ga_evals | n_captured | C+ total | buffered total | ratio (buffered/C+) | max \|g_C+ − g_buffered\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 10 | 6 | 6 | 6 | 23.79s | 97.09s | **4.08×** | 1.88e-13 |
| 1.0 | 10 | 7 | 7 | 7 | 26.29s | 104.17s | **3.96×** | 9.38e-13 |

**Reconciliation with KNITRO's native counters**: `n_grad_calls` (this driver's own counter)
exactly equals `native_ga_evals` (KNITRO's own reported gradient-evaluation count) at **both**
deltas, and `n_captured` (states pushed via `grad_trace_ref`) exactly equals `n_grad_calls`
at both — every gradient callback was captured, none missed or double-counted.

**Numerical equivalence**: max absolute difference between backends across all 13 captured
states combined is 9.38e-13 — both backends compute the same gradient on the same real
trajectory states, at a tolerance consistent with the C+ kernel's own documented equivalence
level (the two backends use different intermediate representations, not different math).

**Isolated kernel saving vs. controlled whole-trajectory saving** (the task's explicit
instruction not to claim "no benefit" from one noisy fixed-wall run): this same-trajectory
replay is the controlled comparison, and it reproduces the documented kernel-level 4.0–4.2×
ratio almost exactly (4.08× and 3.96× measured here) — **on a real trajectory, holding the
function-evaluation sequence fixed**, not just an isolated microbenchmark. This directly
supports the C+ adoption decision on its own terms.

**Reconciliation with the historical 36.5% (delta=0.1) / 19.3% (delta=1) gradient wall-share
figures** (`docs/fullA_ALLOCATION_CROSSDELTA_KB_GATE_2026-07-22.md`): this session's own C+
gradient share of real total wall was 23.79/138.15 = **17.2%** at delta=0.1 and 26.29/128.48 =
**20.5%** at delta=1.0. The delta=1.0 figure (20.5%) is close to the historical 19.3% figure;
the delta=0.1 figure (17.2%) does **not** closely reproduce the historical 36.5%. Both this
session's runs and the historical figures used the C+ backend as the gradient path, so this is
not a backend-identity mismatch; the most likely explanations — not fully resolved here — are
differing thread counts, differing option files/maxit settings between the two measurement
sessions, or a materially different outer-trajectory shape (different accepted points visit
different screen/winner-switch regimes with different per-call cost) at delta=0.1 specifically.
**Disclosed as an open reconciliation question, not force-fit to match.**

Not done here (flagged, matching the original report's own deferral): the full per-category
native-function-attempt breakdown (screens counted separately from warm/cold/infeasible/
exact-cache/checkpoint-serialization/GC) at both deltas — this remains the largest remaining
piece of the original brief, unchanged from the prior assessment.

---

## Phase 6 — regression suites and branch promotion

### fullA-exact branch: full regression battery, 14/14 pass

| Test | Result |
|---|---|
| `test_direction_bounds.jl` | 26/26 pass |
| `test_composite_gradient.jl` | pass (γ-component <0.1% rel-err; A-block cosine >0.9 at all points) |
| `test_compressed_moments.jl` | pass (tie handling verified) |
| `test_safe_exact_cache.jl` | 22/22 pass |
| `test_cross_delta_cache.jl` | 36/36 pass |
| `test_default_backend_dispatch.jl` | 9/9 pass |
| `test_draw_design.jl` | 22/22 pass |
| `test_checkpoint_schema.jl` | 38/38 pass (D20Checkpoint schema-3 round-trip, post accepted-point-reuse counter change) |
| `test_threaded_exception_propagation.jl` | 4/4 pass |
| `test_cm_checkpoint_original.jl` + `test_cm_checkpoint_resume.jl` | 4/4 pass (two-part test; the batch run initially only ran part 2, self-correctly failing with an actionable "run the original script first" message — not a code defect, corrected by running both) |
| `test_cm_delta_dual_tail_active.jl` | 8/8 pass (F1 tail-active identity) |
| `test_winner_certificate.jl` | pass (tie injection → `TiedWinnerError`, as designed) |
| `test_winners.jl` | pass |
| `test_winner_switching.jl` | pass |

Plus, from Phase 3: `test_cm_expected_solve_failure_typed.jl` (7/7), `c31_phase3c_direction_box_validation.jl` (7/7), `c32_phase3f_checkpoint_reuse_validation.jl` (10/10), `c30_phase1_cm_checkpoint_reconciliation.jl` (self-verifying), `c24_phase1_directional_broad.jl` (54/54, independently re-verified).

### sequential-linearized branch

| Check | Result |
|---|---|
| `test_recover_lfd_unsuccessful_solve.jl` (active `seq_gravcol` path) | 4/4 pass — calibration accepted, pathological theta (A_od ×1e6) rejected via both `seq_gravcol` and direct `recover_lfd` |
| `smoke_test_dual_warm_production.jl` (production smoke test) | pass — `persist`/`reset_per_theta`/`cold` all give identical `R_mean=2.0682e-05`, confirming warm-start only changes speed, never the accepted trajectory |
| Dependency check: no full-A-specific driver in sequential tree | confirmed — `find` for `c10_d20_production_driver.jl`/`cm_checkpoint.jl`/`cm_outer_driver.jl`/`cm_production_bundle.jl` in the sequential worktree returns nothing |

### Cross-branch separation (both directions)

- `gravity-remediation-fullA-exact/sequential_gravity/` does not exist (removed in commit `b6abcd5`, confirmed live).
- `gravity-remediation-sequential-linearized/sequential_gravity/` has its full 37 files (its own authoritative source, untouched).
- No inactive phase5 comparison harness remains on the fullA-exact tree (`phase5_lp_diagnose.jl`/`phase5_run_comparison.jl`/`phase5_sequential_reconstruction.jl` all removed in `b6abcd5`).

### Worktree cleanliness

Both remediation branches' working trees are clean. The pre-existing dirty state noted in Phase
0 (regenerated `results/fullA_d4/c10_ckpt_smoke_test{,_resumed}/*.jls` smoke-test byproducts,
plus new ones from this session's own regression runs, including a new
`results/fullA_d4/cm_ckpt_smoke_test/` fixture from the CM checkpoint two-part test) was
resolved: tracked files restored via `git checkout --`, untracked byproducts removed via
`git clean -fd`, scoped only to those three known smoke-test directories — no other paths
touched.

### Promotion

All critical gates passed. Both remediation branches are clean, all required tests pass, both
canonical branches' separation is confirmed intact, the CM checkpoint/cold-verify correctness
chain is validated to machine precision on real current-schema artifacts, and the corrected
directional/exception/reuse/portability fixes are all live-verified.

**Actions taken** (see exact commands/hashes at the end of this document):
1. Fast-forwarded `production/fullA-exact` to `remediation/fullA-exact-2026-07-22`'s tip.
2. Fast-forwarded `production/sequential-linearized` to `remediation/sequential-linearized-2026-07-22`'s tip.
3. Pushed both to `cdw`, plus the existing safety tags (already present on `cdw` from before this session).
4. Created and pushed annotated post-remediation tags on both, referencing this report and the CSV deliverables.
5. Remediation branches were **not** deleted (per the task's own instruction: only after the pushed canonical commits are verified on the remote).

---

## Decision table

| Item | Status |
|---|---|
| unrestricted internal production | **READY** |
| CM internal production | **READY, with one flagged operational caveat** — checkpoint/cold-verify correctness fully validated to machine precision; the interrupt→resume operational path hit a real hang requiring manual `SIGKILL` intervention during this task's shakedown (root-caused away from the known AUD-02 `par_concurrent_evals` regression, classified as the separately-documented "KNITRO can hang past its declared timeout" behavior, not fully resolved to one definitive cause). **Recommendation: do not run unattended multi-hour CM production without an external supervisory process that can detect and restart a hung run** (the heartbeat watchdog added in this codebase makes the hang detectable; it does not make the process self-recover). |
| sequential internal production | **READY** |
| canonical branches promoted | **YES** (see exact hashes below) |
| CM-aware C+ backend | **NOT PART OF THIS TASK** (deferred per the task's own explicit scope exclusion; see `docs/REMEDIATION_FINAL_REPORT_2026-07-22.md` §5a for the documented starting point) |
| complete-state exact/cross-δ LRU cache | **NOT PART OF THIS TASK** (deferred per the task's own explicit scope exclusion; see §5b) |

**What remains optional / explicitly deferred, not silently dropped**:
- Phase 5's full per-native-eval-category breakdown (screens/warm/cold/infeasible/exact-cache/
  serialization/GC counted separately) — the largest remaining piece of the original 12-item
  brief, unchanged from the prior report's own assessment.
- The delta=0.1 gradient-wall-share reconciliation (17.2% measured vs. 36.5% historical) is an
  open question, not resolved to a single cause.
- The CM interrupt→resume hang's exact root cause (severe GC stall vs. some other low-level
  condition) is disclosed, not definitively diagnosed — flagged for a future task with a larger
  time budget for controlled reproduction.

---

## Appendix: exact hashes, pushed refs, test commands/output paths

### Branch/tag hashes

| Ref | Commit |
|---|---|
| `safety/fullA-exact-pre-remediation-2026-07-22` | `670eac47c2517789002387df403369e222078065` |
| `remediation/fullA-exact-2026-07-22` (verified tip) | `70409be53b3e2520a2bc33e678598be1643303db` |
| `production/fullA-exact` (post fast-forward) | `70409be53b3e2520a2bc33e678598be1643303db` |
| `safety/sequential-linearized-pre-remediation-2026-07-22` | `6b349946a8f0aca89366f421d532805ffb34d2e1` |
| `remediation/sequential-linearized-2026-07-22` (verified tip) | `9ec3e46b5c37e8d5c0c1342883c8df7f82e8e7c6` |
| `production/sequential-linearized` (post fast-forward) | `9ec3e46b5c37e8d5c0c1342883c8df7f82e8e7c6` |

### Discovered and handled: uncommitted WIP in the live `production/fullA-exact` worktree

`gravity-production-fullA-exact` had an **uncommitted** local diff to
`c10_d20_production_driver.jl` — an opt-in `full_gamma_range` kwarg, a stopgap workaround for
the same direction-box-stall issue this session's (and the prior remediation session's) commit
`a69fb21` already fixes unconditionally. Identified as exactly the "concurrent Claude Code
session's uncommitted stopgap kwarg" `a69fb21`'s own commit message references. Per the user's
explicit instruction: `git stash push -m "WIP stopgap: full_gamma_range opt-in kwarg
(superseded by remediation commit a69fb21's unconditional fix) -- stashed 2026-07-22 before
promoting remediation/fullA-exact-2026-07-22"` in that worktree before fast-forwarding — fully
recoverable via `git stash list`/`git stash show -p stash@{0}` in `gravity-production-fullA-exact`
if that WIP is still wanted; the underlying capability is now default behavior (no flag needed)
via `a69fb21`, so it is very likely simply obsolete, but preserved rather than discarded.

### Test commands and output paths (all under `docs/closure_2026-07-22/` unless noted)

- Phase 1: `julia --project=. full_aod_diag/d4_exact/c30_phase1_cm_checkpoint_reconciliation.jl <csv>` → `CM_CHECKPOINT_COLD_RECONCILIATION_2026-07-22.csv`, `phase1_reconciliation_raw.log`
- Phase 2: raw sweep at `phase2_directional_sweep_raw_54cases.csv`; summary stats independently recomputed into `CORRECTED_DIRECTIONAL_GATE_2026-07-22.csv` / `phase2_summary_stats.txt`
- Phase 3B: `julia --project=. full_aod_diag/d4_exact/test_cm_expected_solve_failure_typed.jl` → `test_cm_expected_solve_failure_typed.log` (7/7)
- Phase 3C: `julia --project=. full_aod_diag/d4_exact/c31_phase3c_direction_box_validation.jl` → `c31_phase3c_direction_box.log` (7/7)
- Phase 3F: `julia --project=. full_aod_diag/d4_exact/c32_phase3f_checkpoint_reuse_validation.jl` → `c32_phase3f_checkpoint_reuse.log` (10/10)
- Phase 4: `full_aod_diag/d4_exact/c33_phase4_cm_shakedown_{control,interrupt,resume,coldverify}.jl` → `phase4_{control,interrupt,resume,coldverify}_raw.log`, `CM_CORRECTED_RESUME_SHAKEDOWN_2026-07-22.csv`
- Phase 5: `julia --project=. full_aod_diag/d4_exact/c34_phase5_same_trajectory_replay.jl` → `phase5_same_trajectory_replay_raw.log`, `FULLA_CALLBACK_WALL_DECOMPOSITION_2026-07-22.csv`, `FULLA_SAME_TRAJECTORY_BACKEND_REPLAY_2026-07-22.csv`
- Phase 6 (fullA): 14 individual test scripts under `full_aod_diag/d4_exact/test_*.jl`, run via `julia --project=.` with `.knitro_env.sh` sourced and `OPENBLAS_NUM_THREADS=1`; all exit 0
- Phase 6 (sequential): `julia --project=. sequential_gravity/head_to_head/test_recover_lfd_unsuccessful_solve.jl` (4/4), `julia --project=. sequential_gravity/head_to_head/smoke_test_dual_warm_production.jl` (pass)

### Push actions taken (this session, after explicit user confirmation on the stash step)

```
production/fullA-exact:            670eac4 -> 70409be (fast-forward)
production/sequential-linearized:  6b34994 -> 9ec3e46 (fast-forward)
pushed to cdw: production/fullA-exact, production/sequential-linearized
pushed to cdw: post-remediation-2026-07-22/fullA-exact (annotated tag on 70409be)
pushed to cdw: post-remediation-2026-07-22/sequential-linearized (annotated tag on 9ec3e46)
```

Remediation branches (`remediation/fullA-exact-2026-07-22`, `remediation/sequential-linearized-2026-07-22`)
were **not** deleted — retained pending remote verification, per the task's own instruction.
