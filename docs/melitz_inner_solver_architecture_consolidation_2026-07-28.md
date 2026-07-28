# Melitz inner-solver architecture consolidation (2026-07-28)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), starting HEAD
`dcfe25441795de0e4682ebf7562a6701eaf76d62` (one commit ahead of the reported
`a8c2e898bed3bf0ff124ea1bc60fe0bc38b0c963` -- the extra commit, `dcfe254`, is Phase 6b of the
prior anomaly session, a doc-only addition answering a user follow-up question; verified
before any edit, working tree otherwise clean apart from pre-existing untracked scratch
directories inherited from other sessions), 30 commits ahead of `cdw/melitz/fullD-delta-star`,
not pushed. Governing prompt: a bounded architectural consolidation of the Melitz inner-solve
interface, closing the entire class of bug the prior session's
`docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md` diagnosed and
partially (Phase 5 there) patched, per that document's own Phase 6b: "the genuine structural
fix... not done this session, disclosed as a scoped-out recommendation."

## 1. Old fragmented call graph (confirmed by inventory, before this session)

```
CONSTRUCTORS (each independently decided lower_limit, several defaulting to uncapped)
├─ build_melitz_cc_bundle(...)                    -- lower_limit: REQUIRED Float64, no default
│     (a 2026-07-26 session's own partial fix; still a bare Float64, not a typed policy)
├─ build_melitz_implicit_bundle(...)               -- delta_evaluation_cap/inner_solve_config
│                                                      both Union{Nothing,...}=nothing -> uncapped
├─ build_melitz_psi_bundle(...)                    -- inner_solve_config=nothing -> uncapped
└─ build_melitz_psi_bundle_from_calibration(...)   -- inner_solve_config=nothing -> uncapped
      (used by scripts/melitz_regression_fixtures_2026-07-28.jl's build_realD20_fixture/
       build_d4_fixture -- THE shared fixture behind every 2026-07-27/07-28 phase script,
       and the CONFIRMED ROOT CAUSE of the 1.510118e14 FiniteSolved-above-cap anomaly)

DRIVERS
├─ solve_melitz_finite_delta_bound      -- delta_evaluation_cap::Real=10.0, always active
│                                          (safe by construction, but its OWN cap value was
│                                          independent of build_melitz_implicit_bundle's)
├─ solve_melitz_nuisance_min_delta      -- inner_solve_config mandatory (safe by construction)
├─ melitz_fixed_point_probe             -- built its OWN obj via build_melitz_implicit_bundle
│                                          WITH NEITHER cap kwarg -> permanently uncapped;
│                                          CONFIRMED LIVE-UNSAFE, used directly by
│                                          test/melitz/runtests.jl Test D/D'
└─ (no driver -- a script builds a fixture and calls melitz_classified_inner_solve directly)
      -- the anomaly's own actual path

     ↓ every one of the above eventually calls ↓

melitz_classified_inner_solve(obj, theta, ctx; delta_evaluation_cap, bank, ...)
   -- delta_evaluation_cap was a PER-CALL argument gating only the two cheap pre-solve
      screens; the live KNITRO-native lower_limit threshold was a property of obj, FIXED AT
      CONSTRUCTION TIME by whichever of the four constructors built it -- completely
      decoupled from the delta_evaluation_cap value passed here. No check anywhere that the
      two agreed.
   -- own docstring OVERCLAIMED ("every early-abort threshold in this function... is now
      gated on delta_evaluation_cap ONLY") -- true only for the two screens, not the live
      KNITRO threshold on 4 of 5 entry paths -- a documented, concrete contributing cause of
      the anomaly recurring across sessions (prior doc's own Phase 6b).
```

Five independent routes into the same low-level KNITRO call; two ad hoc, caller-scoped fixes
(`build_melitz_cc_bundle`'s no-default `lower_limit`, `solve_melitz_finite_delta_bound`'s own
`@assert isfinite(obj.lower_limit)`) each closed exactly one of the five, leaving the other
four -- confirmed live as the mechanism behind the `1.510118e14` anomaly and the
`melitz_fixed_point_probe` gap.

## 2. New single-entry call graph

```
inner_solve_policy.jl (NEW)
  abstract type MelitzInnerSolvePolicy end
  struct CappedEvaluation <: MelitzInnerSolvePolicy   (cap, max_iterations, max_seconds)
  struct FullValueEvaluation <: MelitzInnerSolvePolicy (max_iterations, max_seconds)
  melitz_policy_lower_limit(policy) -> Float64   -- -cap or -KN_INFINITY
  melitz_policy_cap(policy) -> Float64           -- cap or Inf
  melitz_apply_policy_to_knitro!(kc, policy)     -- maxit/maxtime_real, matrix-free path only

Every bundle constructor now takes policy::MelitzInnerSolvePolicy as a MANDATORY keyword
(no default, no Union{Nothing,...}):
  build_melitz_cc_bundle, build_melitz_implicit_bundle,
  build_melitz_psi_bundle, build_melitz_psi_bundle_from_calibration
  -- lower_limit is ALWAYS melitz_policy_lower_limit(policy), never an independent Float64.

inner_session.jl (NEW)
  mutable struct MelitzInnerSession
      obj::Any; ctx::Any; policy::MelitzInnerSolvePolicy; bank::MelitzDualBank
  end
  -- constructor asserts obj.lower_limit == melitz_policy_lower_limit(policy)
     (_melitz_assert_session_policy_consistent, re-checked again at solve time)

  solve_melitz_delta!(session, theta, policy; ...) -> MelitzInnerResult
  -- THE ONE PUBLIC AUTHORITATIVE ENTRY POINT. Asserts policy==session.policy, re-asserts
     the lower_limit invariant, times the attempt, archives it if pathological (Section 13),
     and delegates to the now-INTERNAL classifier.

inner_screening.jl
  _melitz_classified_inner_solve!(session, theta; ...) -> MelitzInnerResult   [INTERNAL]
  -- reads obj/ctx/bank/cap OFF session (delta_evaluation_cap = melitz_policy_cap(session.policy))
  -- never called directly by production code any more; only by solve_melitz_delta!

cc_bundle.jl
  melitz_bundle_inner_solve!(obj, theta), melitz_cc_inner_loop_knitro!(bundle)   [INTERNAL]
  -- MelitzCCBundle now carries its own `policy::MelitzInnerSolvePolicy` field, applied to
     the live KNITRO instance via melitz_apply_policy_to_knitro! (maxit/maxtime_real)
```

Every solve now flows through exactly one node (`solve_melitz_delta!`) that reads `obj` and
`policy` off the SAME `MelitzInnerSession` object -- there is no remaining code path where the
live KNITRO threshold and the classifier's own cap-based screening could be built from two
independently-suppliable numbers.

## 3. All removed unsafe defaults

| site | before | after |
|---|---|---|
| `build_melitz_cc_bundle` | `lower_limit::Float64` (required, but a bare number) | `policy::MelitzInnerSolvePolicy` (required, typed) |
| `build_melitz_implicit_bundle` | `delta_evaluation_cap::Union{Nothing,Real}=nothing` + `inner_solve_config::Union{Nothing,MelitzInnerSolveConfig}=nothing` | `policy::MelitzInnerSolvePolicy` (mandatory, no default) |
| `build_melitz_psi_bundle` | `inner_solve_config::Union{Nothing,MelitzInnerSolveConfig}=nothing` | `policy::MelitzInnerSolvePolicy` (mandatory, no default) |
| `build_melitz_psi_bundle_from_calibration` | `inner_solve_config::Union{Nothing,MelitzInnerSolveConfig}=nothing` | `policy::MelitzInnerSolvePolicy` (mandatory, no default) |
| `scripts/melitz_regression_fixtures_2026-07-28.jl`'s `build_d4_fixture`/`build_realD20_fixture` | called the above with no cap kwarg at all -- silently uncapped, THE confirmed root cause | `policy::MelitzInnerSolvePolicy=CappedEvaluation(10.0)` -- an explicit, named, capped default (not a `nothing` sentinel); every existing phase script calling these with no `policy` argument is now capped by default |
| `melitz_fixed_point_probe` | built its own `obj` via `build_melitz_implicit_bundle` with NEITHER cap kwarg -- confirmed live-unsafe | `policy::MelitzInnerSolvePolicy=CappedEvaluation(10.0)` (matches `solve_melitz_finite_delta_bound`'s own default) |
| `solve_melitz_nuisance_min_delta` | `inner_solve_config::MelitzInnerSolveConfig` (mandatory; safe, but the old type) | `policy::MelitzInnerSolvePolicy` (mandatory, same requiredness, new type) |
| `melitz_classified_inner_solve` | `delta_evaluation_cap::Real` independent argument | REMOVED -- the renamed internal function derives cap from `session.policy` only |
| `src/melitz/inner_solve_config.jl` | `MelitzInnerSolveConfig`/`melitz_configure_lower_limit`/`melitz_assert_evaluation_cap_active` (symbol-mode config) | FILE DELETED -- fully superseded by `inner_solve_policy.jl` |

## 4. Every migrated caller

`src/melitz/`: `cc_bundle.jl`, `delta_star.jl`, `finite_delta_outer.jl`, `inner_screening.jl`,
`nuisance_profile.jl`, `pareto_calibration.jl`, `predictor_corrector.jl`, `include_melitz.jl`
(updated), plus two new files `inner_solve_policy.jl`/`inner_session.jl`. One deleted file
(`inner_solve_config.jl`).

`scripts/`: `melitz_regression_fixtures_2026-07-28.jl` (the one shared production fixture
this session's scope covers) and `run_melitz_delta_star_fake.jl` (a durable, reusable,
non-dated reproduction runner referenced by `docs/melitz_delta_star.md`, not one of the ~80
dated one-off diagnostic scripts) -- see Section 6 disclosure below for why the other ~80
dated diagnostic/phase/anomaly scripts under `scripts/` are NOT migrated.

`test/melitz/standalone_no_cc_algo.jl`: the separate cc_algo-independent subprocess script
`runtests.jl` itself launches and asserts against -- migrated (two call sites).

`test/melitz/runtests.jl`: every call site touching the old API -- roughly 230 individual
call-site edits across the file (mechanically verified: zero remaining
`inner_solve_config=`/`MelitzInnerSolveConfig(`/`melitz_classified_inner_solve(` call-syntax
anywhere in `src/melitz/` or the test file; the ~28 remaining `melitz_classified_inner_solve(`
occurrences repo-wide are ALL in the disclosed-out-of-scope historical script corpus). Include
list updated to match (`inner_solve_config.jl` -> `inner_solve_policy.jl`, `inner_session.jl`
added after `inner_screening.jl`). One dedicated testset
("`MelitzInnerSolveConfig` + `melitz_assert_evaluation_cap_active`") rewritten to test the new
types' equivalent behavior; the old `outer_delta`-equals-cap soft-warning test has no
replacement (disclosed -- see Section 6).

`Project.toml`: added `Serialization` as a direct dependency (Section 13's archiving; already
resolved transitively in `Manifest.toml`, no new download).

## 5. Exact anomaly regression

The `1.510118e14` point (`docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md`
Phase 1's exact reconstruction) can no longer be constructed the way it originally arose:
`build_realD20_fixture()` (the exact function the anomaly traced the bug to) now defaults to
`policy=CappedEvaluation(10.0)`, so the SAME no-argument call that produced the anomaly now
produces a properly-capped `obj.lower_limit=-10.0` from the start -- the KNITRO-native
mid-solve bailout is armed, `nStatus=-300`/`-103`-style runaway pursuit is intercepted by the
existing hardened accept-gate (`objSol >= obj.lower_limit`, prior session's Phase 5 fix,
retained unchanged inside `_melitz_classified_inner_solve!`) or by `melitz_origin_block_screen`
if enabled, and the `FiniteSolved`-above-cap output invariant (also retained) makes a silent
`Delta=1.51e14` return structurally impossible regardless.

The prior session's own reconstruction script
(`scripts/melitz_anomaly_phase1_reconstruct_2026-07-28.jl`) is NOT updated to the new API (it
is part of the disclosed-out-of-scope historical script corpus, Section 6) -- it is retained,
unmodified, as the historical record of the original incident's exact reproduction steps. This
session's own regression coverage of the same underlying point is `test/melitz/runtests.jl`'s
existing Phase I.1/Phase I.5/Phase I.6/Section 12 testsets (all migrated to the new
`MelitzInnerSession`/`solve_melitz_delta!` API, Section 4 above), which directly exercise
`CappedEvaluation`-governed solves at both feasible and certified-bad points against the SAME
underlying classifier logic the anomaly point exercised, plus the NEW
"Section 11: architecture consolidation" static test (Section 8 below) proving the specific
decoupled-cap code shape that produced the anomaly can no longer exist in production source.

## 6. Disclosed scope decisions (not oversights)

1. **The ~80 pre-existing, dated, one-off diagnostic/phase/anomaly scripts under `scripts/`
   are NOT migrated to the new mandatory-`policy` API.** These are frozen historical lab
   artifacts from prior sessions (phase-N campaigns, anomaly reconstructions, benchmark
   sweeps), not part of this codebase's production call graph, and migrating ~80 files'
   worth of ad hoc call sites is exactly the "broad campaign" this session's own governing
   prompt scopes out ("Do not run outer campaigns, parameterization comparisons, or
   performance tuning"). They will fail with `UndefKeywordError`/`MethodError` if re-run
   as-is under the new API -- a real, disclosed, accepted consequence of closing the bug
   class at its structural root, not a silent gap. The ONE script this session's scope
   explicitly does cover -- `scripts/melitz_regression_fixtures_2026-07-28.jl`, the shared
   fixture builder behind the confirmed anomaly and reused by dozens of those same historical
   scripts -- IS migrated (Section 3), so every script that only calls `build_d4_fixture()`/
   `build_realD20_fixture()` with no arguments (the common case) is now safely capped by
   default even without being touched directly.
2. **Section 4's literal "fixture/calibration constructors must return an economic
   problem/context only, never bake in solver state" is satisfied at the POLICY level, not
   via a full teardown of `MelitzCCBundle`/the legacy dense bundles into separate
   context-only and session objects.** `MelitzCCBundle.op` (`MelitzMomentOperator`) already
   IS the separated economic state (no solver-state field lives on it); the legacy dense
   bundles (`PsiObjectiveBundleDelta`/`PsiObjectiveBundleImplicit`) are `cc_algo`
   (Ricardian-owned) types this session cannot restructure at all. `MelitzInnerSession` is
   the new object that owns the actual missing piece -- the POLICY decision and the
   warm-start bank -- decoupled from every fixture/calibration constructor's own defaults.
   Fully splitting `MelitzCCBundle` into a bare-economic-context type plus a wrapping session
   would require rewriting dozens of consumer functions across `direct_gradient.jl`,
   `sorted_crossing_gradient.jl`, `touched_row_gradient.jl`, `delta_star.jl`,
   `nuisance_profile.jl`, and `finite_delta_outer.jl` that all currently read fields directly
   off `obj` -- out of this session's bounded scope, and a materially higher-risk rewrite of
   numerically-validated production code than this specific bug class requires.
3. **`CappedEvaluation`/`FullValueEvaluation`'s `max_iterations`/`max_seconds` are only
   DYNAMICALLY applied to the live KNITRO instance for the Melitz-owned matrix-free path**
   (`melitz_cc_inner_loop_knitro!`, via `melitz_apply_policy_to_knitro!`). The legacy dense
   bundles' KNITRO instance is constructed inside `cc_algo/inner_loop_functions.jl`
   (Ricardian-owned, never touched by this session, per the absolute Ricardian boundary) --
   `max_iterations`/`max_seconds` are NOT dynamically enforceable there; only the static
   `inner_loop_opt`/`outer_loop_opt` `.opt` file's own `maxit`/`maxtime_real` govern that path,
   exactly as before this session. The cap (`lower_limit`, the actual root-cause mechanism)
   applies uniformly to BOTH bundle families regardless, since `lower_limit` is a plain
   mutable field on both.
4. **A genuine per-iteration "objective trace" (Section 13) is not implemented.** Recording
   one would require threading a callback into the hot KNITRO objective/gradient/Hessian
   functor -- exactly what "without affecting ordinary hot-path performance materially" warns
   against modifying casually. `MelitzPathologicalSolveRecord` instead captures a cheap
   two-point proxy (objective at the starting dual and at the final iterate), computed only
   on the rare slow-path branch, never inside the hot callback.
5. **The old `MelitzInnerSolveConfig`'s `outer_delta`-equals-cap soft warning has no
   replacement.** The new policy types have no `outer_delta` concept at all -- `policy` is
   constructed and threaded independently of any particular outer-budget value now, so the
   soft "did you mean to set the cap equal to the budget" heuristic no longer has a natural
   home. Not reintroduced; a purely diagnostic nicety, not a correctness invariant.

## 7. Result-verification rules (retained, unchanged in substance)

`FiniteSolved`/`AboveEvaluationCap`/`InfiniteDeltaCertified`/`NumericalFailure`'s own semantics
and the two Phase-5 hardening invariants from the prior anomaly session
(`objSol >= obj.lower_limit` cross-check on approximate-status acceptance; the unconditional
`Delta_theta <= delta_evaluation_cap + tol` assert before constructing `FiniteSolved`) are
UNCHANGED -- this session's contribution is eliminating the possibility that
`delta_evaluation_cap`/`obj.lower_limit` ever independently disagree in the first place, not
re-deriving the classification rules themselves, which were already correct given a
consistent cap.

## 8. Feasibility-screen decision

`melitz_origin_block_screen` remains opt-in (`origin_block_screen::Bool=false` default on
`solve_melitz_delta!`/`_melitz_classified_inner_solve!`), unchanged from the prior session --
this session did not re-benchmark it (out of scope: "Do not run... performance tuning"). The
prior session's own Phase 4 finding stands: it is the one mechanism that classifies a
genuinely infinite point cheaply and correctly where the live KNITRO threshold cannot (an
unbounded pursuit can terminate via `nStatus=-300` without ever crossing `f<=lower_limit`
cleanly enough for the crossing flag to fire).

## 9. Static-test results

New testset `"Section 11 (architecture consolidation): static scan for unsafe defaults/direct
low-level calls"` (`test/melitz/runtests.jl`), scanning `src/melitz/*.jl` plus
`scripts/melitz_regression_fixtures_2026-07-28.jl`:

- no `inner_solve_config` reference anywhere in production source: **PASS** (confirmed by
  direct grep during this session: zero hits outside historical scripts/comments).
- no raw `lower_limit=-KNITRO.KN_INFINITY` default outside the two sanctioned exceptions
  (`inner_solve_policy.jl`'s own `FullValueEvaluation` implementation;
  `matrix_free_dual_solve.jl`'s pre-existing, already self-disclaimed
  diagnostic-only oracle bundle): **PASS**.
- no direct call to `_melitz_classified_inner_solve!`/`melitz_bundle_inner_solve!`/
  `melitz_cc_inner_loop_knitro!` outside their own sanctioned files
  (`inner_screening.jl`/`inner_session.jl`/`cc_bundle.jl`): **PASS**.
- `melitz_classified_inner_solve` (old name) no longer defined; `solve_melitz_delta!`/
  `_melitz_classified_inner_solve!` both defined: **PASS**.

Full assertion counts for these four checks and the rest of the suite: see Section 10 below
(filled in after the full-suite run this session performed).

## 10. Full-suite run

`julia --project=. -t 1 test/melitz/runtests.jl` (KNITRO 13.0.1, `.knitro_env.sh` pinned,
`OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1`), plus the separate standalone subprocess this
suite itself launches (`test/melitz/standalone_no_cc_algo.jl`, exercised via its own "Standalone
(no cc_algo) subprocess" testset):

**189,262 / 189,262 individual assertions passed, 0 failed, 0 errored, 0 broken, across 63
top-level testsets, exit code 0.** Includes every migrated entry-path testset (Section 4/9),
the new "Section 11 (architecture consolidation): static scan for unsafe defaults/direct
low-level calls" testset (6/6 passed), and the standalone cc_algo-independent subprocess
(2/2 passed).

This did NOT pass on the first attempt -- three real, disclosed rounds of fallout, each fixed
before the next run:

1. **`run_melitz_inner_delta`** (`delta_star.jl`, a thin `kwargs...`-forwarding wrapper around
   `build_melitz_psi_bundle`) was missed by the direct-call-site inventory (it does not
   literally contain the constructor's name at ITS OWN call sites) -- two call sites
   (`test/melitz/runtests.jl`, `scripts/run_melitz_delta_star_fake.jl`) needed
   `policy=FullValueEvaluation()` added.
2. **Two pre-existing tests (`Test D'`, `test/melitz/runtests.jl`'s "Phase I.6: finite raw
   dual captured into the bank on NumericalFailure") asserted `NumericalFailure` on a
   perturbed point that, under the OLD decoupled-cap bug, could only ever reach
   `NumericalFailure` because the live KNITRO threshold was silently disabled.** Once this
   session's fix genuinely arms the threshold, that SAME point (confirmed identical
   construction: `theta0_20 .+ 0.5.*randn(MersenneTwister(1),...)`) correctly crosses it and
   classifies `AboveEvaluationCap(:live_dual_threshold)` instead -- the intended, correct
   post-fix behavior (a cheaper, earlier, equally-valid rejection), not a regression. Both
   tests were rewritten to use `policy=FullValueEvaluation()` explicitly, matching their own
   documented intent ("no certificate of any kind" / "the branch where a real KNITRO attempt
   happens and fails without a certificate") -- a genuinely uncapped session is required for
   that branch to be the one actually reached; a properly-armed capped session, correctly,
   no longer reaches it for this point. Both changes are disclosed inline in the test file's
   own comments, not silently patched.
3. **`test/melitz/standalone_no_cc_algo.jl`** (a separate script, not `include`d into
   `runtests.jl`, launched as its own subprocess) had two more unmigrated call sites
   (`build_melitz_psi_bundle`, `solve_melitz_finite_delta_bound`) -- missed by the initial
   sweep because it is not itself under `test/melitz/runtests.jl`'s own text; found via the
   subprocess testset's own failure once the rest of the suite reached it.
4. **The new Section 11 static-scan testset itself had two false positives** on its first run:
   a bare-substring check for `inner_solve_config` and a bare-regex check for
   `lower_limit=-KNITRO.KN_INFINITY` both matched this session's OWN backtick-quoted historical
   narrative inside docstrings (e.g. "the old `inner_solve_config::Union{Nothing,...}=nothing`
   pair" -- prose explaining what was fixed, required by Section 12's own documentation
   mandate, not live code). Tightened to the specific dangerous code shapes
   (`inner_solve_config\s*=\s*nothing`, `inner_solve_config::`,
   `lower_limit\s*=\s*-KNITRO\.KN_INFINITY`) with a backtick-precedes-match exclusion (this
   repo's own convention: every historical inline-code mention in a docstring is
   backtick-quoted; live code never is) -- verified directly against the five originally-flagged
   files to confirm all were prose, zero were real.

No other fallout. `git status`/`git diff --name-only` confirms the fix set stayed within the
same 13 files (11 modified `src/melitz/`+`scripts/`+`test/melitz/` files, one deleted, two new)
throughout all four rounds -- no Ricardian file was ever touched, including during this
debugging.

## 11. Complete files changed

```
Modified:
  Project.toml
  scripts/melitz_regression_fixtures_2026-07-28.jl
  scripts/run_melitz_delta_star_fake.jl
  src/melitz/cc_bundle.jl
  src/melitz/delta_star.jl
  src/melitz/finite_delta_outer.jl
  src/melitz/include_melitz.jl
  src/melitz/inner_screening.jl
  src/melitz/nuisance_profile.jl
  src/melitz/pareto_calibration.jl
  src/melitz/predictor_corrector.jl
  test/melitz/runtests.jl
  test/melitz/standalone_no_cc_algo.jl

New:
  src/melitz/inner_solve_policy.jl
  src/melitz/inner_session.jl
  docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md   (this document)

Deleted:
  src/melitz/inner_solve_config.jl
```

**Zero diff in `cc_algo/`, `full_aod_diag/`, or any other Ricardian path** (`production/fullA-exact/`
does not exist in this repo) -- confirmed via `git status`/`git diff --name-only` at the end of
this session.

## Acceptance criteria

1. One public Melitz inner-solve API: **met** (`solve_melitz_delta!`).
2. Every solve has a mandatory explicit typed policy: **met** (all four bundle constructors +
   the two drivers that build their own bundle now require `policy::MelitzInnerSolvePolicy`,
   no default resolves to uncapped except the explicitly-named `FullValueEvaluation()`).
3. No capped solve can be constructed with lower_limit disabled: **met** --
   `MelitzInnerSession`'s own constructor (and `solve_melitz_delta!`'s re-check) assert
   `obj.lower_limit == melitz_policy_lower_limit(policy)`.
4. No fixture silently bakes in an uncapped solver: **met for the confirmed root cause**
   (`build_d4_fixture`/`build_realD20_fixture` now default to `CappedEvaluation(10.0)`,
   named, not a `nothing` sentinel) -- see Section 6.1 disclosure for the ~80 historical
   scripts not migrated.
5. Capped and full-value solves use separate compatible session state: **met** --
   `MelitzInnerSession` docstring documents never sharing one session/bank across a policy
   switch; each construction path in this session's migrated code builds a fresh session per
   policy.
6. Low-level solve functions are not called directly from production scripts: **met** for
   `src/melitz/` and the one migrated script; see Section 6.1 for the disclosed historical-
   script exception.
7. `FiniteSolved` requires complete primal-dual verification: **met, unchanged** (prior
   session's Phase 5 hardening retained verbatim).
8. The exact anomaly cannot return `FiniteSolved`: **met** (Section 5).
9. Every former entry path is covered by tests: **met** -- outer FC/GA
   (`solve_melitz_finite_delta_bound`), fixed-point probe (`melitz_fixed_point_probe`),
   nuisance profile (`solve_melitz_nuisance_min_delta`), calibration fixture
   (`build_melitz_psi_bundle_from_calibration`), test fixture (`build_melitz_psi_bundle`),
   diagnostic helper (`_melitz_classified_inner_solve!`/`solve_melitz_delta!` direct tests)
   all have migrated, passing coverage in `test/melitz/runtests.jl`.
10. Static tests reject future unsafe defaults/direct calls: **met** (Section 9).
11. Misleading documentation corrected: **met** (Section 12 of the governing prompt --
    `inner_screening.jl`'s and `finite_delta_outer.jl`'s docstrings both corrected in place,
    old overclaim retained only as explicitly-marked historical narrative, not live guidance).
12. No Ricardian code changes: **met** (Section 11 above).
