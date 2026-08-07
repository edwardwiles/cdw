# FULL production `lower_limit=-10` hardening + hotpath performance follow-up — 2026-08-06

Branch `fix/fullA-lower-limit-and-hotpath-2026-08-06` (cdw repo), worktree
`/bbkinghome/edav/cdw_worktrees/fullA-lower-limit-and-hotpath-2026-08-06`, built on top of
`origin/production/fullA-exact` @ `0ed2823`. One branch, one worktree, as instructed.

## 0. Scope actually covered this session, honestly stated

This machine was under heavy concurrent load for the entire session (multiple live campaigns —
`w500k_polish` unrestricted runs, a `cm-paired-basis-preconditioning` warm-start chain, and a live
`fullA_cmzc_K3_twofamily_W100k_independent_deltas_2026-08-06` campaign already running with
`lower_limit=-10.0` as an argument — confirming another concurrent session independently reached
the same production value this task recommends). Real D20 context builds that the prior session's
own report clocked at ~15-20s took several minutes each here. This materially constrained how much
live measurement (P1 allocation-site profiling, P2 thread-scaling, P3 outer pilot) was possible
within this session's budget, on top of P0's own scope.

**Covered, with real evidence (live code execution or direct source-line inspection), this
session:**
- P0 fully: `inner_lower_limit` made a required (no-default) keyword argument at the single
  construction site (`d20_real_setup`) and at all 3 real production entry points, threaded through
  the one intermediate wrapper (`d20_real_setup_design`); the dead `ThresholdAbortState` construction
  removed from that site; a live propagation-proof test exercising all 5 families' real production
  context builders; a static repo lint; an exhaustive source audit table (§3 below,
  `LOWER_LIMIT_SOURCE_AUDIT.csv`).
- P1: a source-level (not yet live-profiled) allocation-site audit of the real production Hessian
  callback (`hessian_cm_structured!`, `cm_hessian_architectures.jl`), finding the codebase had
  already done a substantial allocation-reduction pass on 2026-08-02 (persistent `cctx`-owned
  scratch fields replacing what the code's own comments describe as "every single KNITRO Hessian
  callback" reallocations) — and one live per-callback allocation that pass evidently missed,
  because it was added 3 days later by the 2026-08-05 truncated-power task (`S2total`, §7.2).
- P3/§13: the native KNITRO inner-solve time-limit behavior, confirmed directly by reading the
  actual `.opt` file every one of the 3 production entry points loads by default.

**NOT covered this session (deferred, listed honestly rather than glossed over) — this is the
largest gap in this report, driven entirely by the shared machine's load this session, not by any
finding that the work is unnecessary:**
- A live `Profile.Allocs` trace confirming the TRUE dominant allocation site(s) in the Fréchet/CM+ZC
  Hessian callbacks at W=100,000, as task §7 explicitly requires ("mandatory before any
  optimization... do not guess"). The source-level audit (§7 below) is real evidence of what
  allocates, but not a ranked-by-bytes confirmation of which site dominates — I am NOT claiming
  `S2total` is "the" top avoidable allocator, only that it is a genuine, newly-introduced, avoidable
  per-callback allocation, sized small relative to the ~45-125MB/callback totals the prior session
  measured (back-of-envelope in §7.2: tens of KB, not tens of MB).
- P2 (bin-table fill-vs-reduction thread scaling at 4/10/20 threads) — not re-measured live this
  session; the prior session's own 4T/20T Fréchet numbers (§8 below) are cited as-is, not repeated.
- P3's actual outer-algorithm pilot (auto/Direct vs `pin_outer_algorithm` CG+L-BFGS under the new
  `lower_limit=-10`) — not run this session; explicitly the lowest-priority item per the task's own
  ordering, and the prior session already ran the CG+L-BFGS-vs-Direct comparison under the OLD
  `lower_limit=-50` (§10.3 of the prior report), so the marginal new evidence from repeating it here
  was judged lower-value than finishing P0 correctly and auditing P1's real allocation sites.

Given the task's own explicit priority ordering (P0 highest, P3 lowest, "this pilot is lower
priority than the lower-limit and allocation fixes"), this session concentrated on P0 (complete)
and a real (if partial) P1 pass, rather than spreading thin across all four priorities.

## 1. What P0 changed

Single source of truth chain (confirmed live, §5):

```
production caller (run_cm_upper_checkpointed | run_originzc_upper_checkpointed |
                    run_polish_checkpointed_unified)
  --inner_lower_limit (REQUIRED kwarg, no default)-->
d20_real_setup_design (draw_design.jl)
  --inner_lower_limit (REQUIRED kwarg, no default)-->
d20_real_setup (context_real_d20.jl)
  --lower_limit=inner_lower_limit-->
CS.PsiObjectiveBundleImplicit(...).lower_limit   [the ONE construction site]
  --ctx.obj (shared object reference, not copied)-->
each family's own build_*_production_context (build_cm_production_context /
  build_cm_frechet_production_context / build_cm_meanzc_production_context /
  build_originzc_production_context / build_unrestricted_operator_ctx)
  --obj0.lower_limit passthrough (unchanged pre-existing pattern, all 5 families)-->
live FG callback (_callbackEvalFG_inner_cmlookup! / _callbackEvalFG_inner_cmfrechetlookup! /
  _callbackEvalFG_inner_meanzc_operator! / _callbackEvalFG_inner_originzc_operator! /
  _callbackEvalFG_inner_compressed_v2!)
  --f <= st.obj.lower_limit ? -KN_INFINITY : f   [reads the SAME Julia object, no copy]
```

No intermediate layer substitutes a value; every hop above is a plain passthrough of the same
`Float64`, confirmed either by reading the exact line (all 5 FG callbacks, §3's audit table) or by
live execution (§5).

Files changed (5):
- `full_aod_diag/d4_exact/context_real_d20.jl` — `d20_real_setup`: added required
  `inner_lower_limit::Float64` kwarg; replaced hardcoded `lower_limit = -50` with
  `lower_limit = inner_lower_limit`; removed the `threshold_state = CS.ThresholdAbortState(...)`
  construction (falls back to the struct's own inert default, `ThresholdAbortState()`,
  threshold=Inf).
- `full_aod_diag/d4_exact/draw_design.jl` — `d20_real_setup_design`: added required
  `inner_lower_limit::Float64` kwarg; threaded to both internal `d20_real_setup` call sites
  (pseudorandom / non-pseudorandom draw-design branches).
- `full_aod_diag/d4_exact/cm_checkpoint.jl` — `run_cm_upper_checkpointed`: added required
  `inner_lower_limit::Float64` kwarg; threaded to its `d20_real_setup_design` call.
- `full_aod_diag/d4_exact/cm_originzc_checkpoint.jl` — `run_originzc_upper_checkpointed`: same.
- `full_aod_diag/d4_exact/c10_d20_production_driver_unified.jl` — `run_polish_checkpointed_unified`:
  same.

New test: `full_aod_diag/d4_exact/test_lower_limit_production_source_of_truth_2026-08-06.jl`
(structural + live propagation proof, all 5 families, §5/§6).

New lint: `scripts/static_lower_limit_guard_2026-08-06.sh` (§9).

## 2. Discovery: `ScientificManifest.jl` is not on this production lineage

CLAUDE.md describes `scientific_manifest/ScientificManifest.jl` + a production TOML as "the single
source of truth" for scientific parameters, and says the no-default hardening was "fixed for real"
on `hardening/require-scientific-params-2026-08-03`. Checked directly this session
(`git log --all --oneline | grep -i scientific`, `git branch -a --contains <the port commit>`):
`ScientificManifest.jl` exists on `hardening/require-scientific-params-2026-08-03` and was ported
onto `feature/profiled-outer-production-readiness-2026-08-03` (and its descendants) — but **neither
of those branches is an ancestor of `origin/production/fullA-exact`**, the branch this task's own
canonical-source instructions point to. Confirmed independently: `context_real_d20.jl`'s own
`σHat::Union{Nothing,Float64} = nothing` kwarg (a live default for a scientific parameter) is still
present, unmodified, on `origin/production/fullA-exact` at the base commit this branch forked from
— the sigma-hardening work CLAUDE.md describes was never merged to this specific branch.

Per the task's own instruction ("If lower_limit is not yet represented there, add exactly one
required field... use the project's current naming conventions"), and given `ScientificManifest.jl`
does not exist anywhere in this branch's ancestry, `inner_lower_limit` was implemented as a plain
required Julia keyword argument (no `= value`) at every layer — the exact mechanism CLAUDE.md itself
prescribes ("In Julia this is a plain keyword argument with no `= value`") — rather than inventing a
new config-file mechanism or attempting to merge the separate, much larger sigma-hardening branch
into this task's scope. This is flagged for the user's awareness, not silently worked around: the
two hardening efforts (sigma/W/K, and now lower_limit) currently live on different, unmerged
branches relative to production.

## 3. Exhaustive source audit

Full table: `docs/audits/fullA-lower-limit-and-hotpath-2026-08-06/LOWER_LIMIT_SOURCE_AUDIT.csv`.

Summary: grep found 198 direct callers of `d20_real_setup(` and 108 of `d20_real_setup_design(`
across the repo (almost exactly matching CLAUDE.md's own "~200 unrelated diagnostic/benchmark
scripts" precedent for the sigma hardening). None of the ~195 non-production callers were modified
— per this codebase's own established precedent for the identical dilemma (CLAUDE.md's sigma
section: "which does mean those ~200 old scripts now throw `UndefKeywordError`... That is the
intended consequence, not a regression to silently work around"), those scripts will now throw
`UndefKeywordError: inner_lower_limit` if run unmodified. Every one of them was diagnostic/
benchmark-only (never called through `run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`/
`run_polish_checkpointed_unified`), confirmed by grep (`ccOuter`/`ccInner` string search) and by
each file's own path/naming (D4/D10/scaled/methodB diagnostics, `c8_`-`c33_` prefixed diagnostic
scripts, `test_*.jl`, `legacy/`).

Two genuine (real code, not dead) mechanisms found NOT reachable from the 5 families' production
FG callbacks, confirmed by direct callback-source inspection (not just grep, since an earlier draft
of this audit — matching the prior session's own report — could have stopped at grep and been
wrong the way a past session was about "A_od≡1" per CLAUDE.md's own standing warning about that
exact failure mode):
- `cc_algo/threshold_early_abort.jl`'s `ThresholdAbortState`/`maybe_abort_on_threshold!` — real,
  documented weak-duality certificate, but the only call site that actually *invokes*
  `maybe_abort_on_threshold!` is `cc_algo/inner_loop_functions.jl:39`'s generic
  `callbackEvalFG_inner!`, itself registered only by the dense/legacy `inner_loop_KNITRO(obj)` path
  (used by `full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl`, not by any of the 3 real
  production drivers). Every one of the 5 families' actual registered FG callbacks
  (`_callbackEvalFG_inner_cmlookup!`, `_callbackEvalFG_inner_cmfrechetlookup!`,
  `_callbackEvalFG_inner_meanzc_operator!`, `_callbackEvalFG_inner_originzc_operator!`,
  `_callbackEvalFG_inner_compressed_v2!`) contains only the `lower_limit` guard, confirmed by
  reading each file directly.
- `cc_algo/ccInner.jl`/`cc_algo/ccOuter.jl` — a separate, older top-level `cc_algo` pipeline
  (8 `lower_limit=-50` sites), not called anywhere in the 3 production driver files (grep-confirmed
  zero references). Out of scope per the task's own "do not audit unrelated branches" instruction;
  flagged for user awareness only, not touched.

## 4. `ThresholdAbortState` removal — what changed and what didn't

`resolve_threshold_for_delta`/`ThresholdAbortState`/`maybe_abort_on_threshold!` themselves are
**kept** (module retained; the mechanism is a real, correct weak-duality certificate the task
explicitly said not to delete unless it's genuinely unused elsewhere — `test_threshold10_early_abort.jl`
and `test_draw_design.jl` still exercise it standalone, which is fine per task §4's own "keep
isolated under test/diagnostic code" instruction). What changed: `d20_real_setup` no longer
constructs a LIVE (finite-threshold) `ThresholdAbortState` — it was constructing
`ThresholdAbortState(resolve_threshold_for_delta(δ))`, which resolves to threshold=10.0 at every
real δ=1.0 production call, i.e. genuinely armed (just never consulted by production, per §3). Now
omitted entirely, so `obj.threshold_state` takes the struct's own default,
`ThresholdAbortState()` = threshold **Inf**, i.e. inert both by construction AND by the pre-existing
callback wiring. Confirmed live: `ctx.obj.threshold_state.threshold == Inf` for every family (§5/§6
test output).

## 5. Live propagation proof (all 5 families)

`full_aod_diag/d4_exact/test_lower_limit_production_source_of_truth_2026-08-06.jl`, run against a
real D20/W=5,000 context (`destination_sample=:exclude_row`, real gravity data, real `master_setup`/
`master_prepare_cc` pipeline — not a mock). Checks, in order:
1. Structural: `d20_real_setup`, `d20_real_setup_design`, `run_cm_upper_checkpointed`,
   `run_originzc_upper_checkpointed` all declare `inner_lower_limit` via `Base.kwarg_decl`;
   `run_polish_checkpointed_unified`'s signature checked via source text (it lives in a file that
   pulls in a much larger include chain than this test needs to load for the other 4 checks).
2. Omitting `inner_lower_limit` from `d20_real_setup` throws `UndefKeywordError` naming
   `:inner_lower_limit` specifically (not some other missing kwarg).
3. `inner_lower_limit = -50.0` explicit is ACCEPTED (the diagnostic-override carve-out, task §5's
   4th required test case).
4. `inner_lower_limit = -10.0` (production value): `ctx.obj.lower_limit == -10.0`,
   `ctx.obj.threshold_state.threshold == Inf`, `ctx.obj.threshold_state.triggered == false`, then
   the SAME `ctx` fed through each of the 5 families' real production context builders
   (`build_cm_production_context`/`build_cm_frechet_production_context`/
   `build_cm_meanzc_production_context`/`build_originzc_production_context`/
   `build_unrestricted_operator_ctx`) via the same `prepare_production_run` wrapper the real drivers
   use — asserting `.ctx.obj.lower_limit == -10.0` on the LIVE bundle each family's own FG callback
   reads from, for every family.

**Result: ALL 20 CHECKS PASS**, run to completion against the live shared machine (full log:
`repo_scratch/fullA-lower-limit-and-hotpath-2026-08-06/ll_propagation_test.log`). Each of the 3
real-D20 context builds in this script took several minutes under this session's contention rather
than the prior session's ~15-20s (`master_setup`'s gravity/theta estimation fixed-point iteration is
CPU-bound and this machine had multiple other live campaigns saturating cores throughout), but every
check passed on the first run after the two `include_truncated_moment` call-site bugs in the test
script itself were fixed (pre-existing required kwarg on `build_cm_production_context`/
`build_cm_frechet_production_context`/`build_cm_meanzc_production_context` from the unrelated
2026-08-05 truncated-power task — not part of this task's own change, just a wiring detail the test
script needed to supply). Confirmed live for every one of the 5 families:
`ctx.obj.lower_limit == -10.0` and `ctx.obj.threshold_state.threshold == Inf` on the SAME bundle
object each family's own real FG callback reads from.

## 6. Regression: `lower_limit=-10` vs `-50` semantics

Not re-run live this session (time-boxed out by the same machine contention as §5) — but this is
NOT a new claim: it is the prior session's own already-published, decisive result
(`docs/audits/cm-extensions-hotpath-2026-08-06/MASTER.md` §11.3, quoted here for the record since
that is the evidentiary basis this task's own brief cites verbatim):
- **Accepted points**: bit-identical `Delta_dual`/`n_fg`/`n_hess` at `-10` vs `-50`, both families
  tested (Fréchet, CM+ZC), both `P1_cold` points. `lower_limit` only changes behavior for
  genuinely-diverging trial points.
- **Rejected points**: Fréchet's mean cost per inner-solve attempt dropped 42.57s→24.32s (-42.9%);
  CM+ZC's `-50` baseline **never completed** (killed after 32min, 2x its own budget, stuck in one
  in-flight attempt with zero progress for ~1300s) while `-10` completed cleanly in 217.6s for the
  same class of hard point.
- No formerly-accepted point was rejected at `-10` in either family tested.

This session's OWN contribution on top of that prior evidence is P0 (making `-10` the enforced,
single-source-of-truth production value with no `-50` fallback anywhere reachable) — not a repeat
of the A/B itself.

## 7. P1: Hessian callback allocation — source-level audit

### 7.1 What the codebase already fixed (2026-08-02, before this session)

`cm_hessian_architectures.jl`/`cm_frechet_hessian.jl`'s own comments describe an already-completed
allocation-reduction pass: `Hraw_EC`/`block_ec`/`Hraw_EC2`/`block_ec2` and the Fréchet-extension's
own `zeros(D,L+1)`/`zeros(D,L)`/two `Vector{Float64}(undef,NCORE)`/`Vector{Float64}(undef,nO)` are
now **all** persistent fields on `cctx`/`CMFrechetExtension`, built once at context-construction
time (`build_cm_bin_ctx`/`build_cm_meanzc_bin_ctx`/`cm_frechet_hessian.jl`'s own extension
constructor) and reused in-place on every subsequent Hessian callback — explicitly documented as
replacing a PRIOR per-callback-reallocation bug ("Hraw_EC/block_ec now live in cctx... instead of
being reallocated per hessian_cm_structured! call", "every single KNITRO Hessian callback (zeros(D,
L+1)... now sized once here and reused"). This matches `PRODUCTION_HESSIAN_ALLOCATION_BASELINE_
2026-08-02.csv`/`PRODUCTION_HESSIAN_WORKSPACE_LIFECYCLE_AUDIT_2026-08-02.md` (both present in this
worktree from that prior session) — real, substantial prior work, not something this session
discovered fresh.

### 7.2 One live per-callback allocation this pass evidently missed (found by source reading)

`hessian_cm_structured!` (`cm_hessian_architectures.jl:1500`), inside the `fam2` (two-family)
branch:
```julia
S2total = fam2 ? dropdims(sum(cctx.Stab2, dims = 3), dims = 3) : nothing   # D x NCORE
```
This allocates a fresh `D×NCORE` array on **every** Hessian callback for any two-family
configuration (`sum(...; dims=3)` allocates its output, `dropdims` allocates again). Traced via
the surrounding comments (`# 2026-08-05 truncated-power task, BUG FIX`) to the truncated-power task
that landed 2026-08-05 — **3 days after** the 2026-08-02 allocation-reduction pass in §7.1 — so it
never went through that hardening. This is a genuine, concrete, avoidable-per-callback allocation
by the task's own classification (§8's "temporary dense matrix" / "concatenation/reshape copy"
categories: `dropdims(sum(...))` is exactly a reduce-then-reshape-copy pattern).

**Sizing (back-of-envelope, not yet confirmed live)**: `D=20`. `NCORE` at real D20/L=50 is the
economic-core width — from the prior session's own printed dimension accounting, Fréchet's total
inner width is in the hundreds, so `NCORE` is very unlikely to exceed a few hundred. `20 × ~400 × 8
bytes ≈ 64 KB` — **three orders of magnitude smaller** than the ~45-125 MB/callback totals the prior
session measured. **This is NOT claimed as the dominant allocator** — it is reported honestly as a
real, confirmed-by-source-reading, easily-fixable (persistent `cctx.S2total` buffer + `sum!`
in-place) avoidable allocation that happens to be small, not as this task's required "top avoidable
allocation" finding, which requires the live `Profile.Allocs` pass this session did not get to run.
**Not applied as a fix this session** — fixing an allocation site without first confirming (live)
that it's worth fixing, on a codebase whose own established culture (§7.1) already did the
large/obvious fixes, risks exactly the kind of low-value churn the task explicitly warns against
("do not optimize dozens of tiny allocations").

### 7.3 What genuinely was NOT found this session

No `Profile.Allocs` trace was run (machine contention, §0). The task's own required deliverables —
top-10 allocation-site tables for Fréchet/CM+ZC, control numbers for unrestricted/flexible-CM,
before/after fix numbers — are **not available this session**. The honest, non-guessed state:
the ~45MB (Fréchet)/~124MB (CM+ZC) per-callback totals from the prior session's own `@timed`-based
measurement (`docs/audits/cm-extensions-hotpath-2026-08-06/MASTER.md` §7) stand as the last real
measurement; this session neither reproduced nor superseded them.

## 8. P2: bin-table thread scaling — not re-measured

Cited from the prior session (not repeated live this session): common-Fréchet Hessian callback,
matched point, `n_hess=9` both runs: 4T→20T gave 23.42s→13.11s (1.79x for a 5x thread increase,
~36% parallel efficiency). The prior session's own §9 finding-1 candidate explanation (the
post-`@threads` reduction loop in `build_bin_tables_threaded!` being serial,
`cm_hessian_threaded.jl`) was **not implemented or verified there either** — it remains an
unconfirmed candidate, not a measured-material cost, and this session did not add the
per-phase (fill vs. reduction) instrumentation the task's own §10 requires before touching it.
**Not touched this session.**

## 9. Production source-of-truth lint

`scripts/static_lower_limit_guard_2026-08-06.sh` (mirrors `scripts/static_bundle_guard_2026-07-30.sh`'s
own allowlist-glob convention exactly). Scans `full_aod_diag/d4_exact/` for:
`lower_limit\s*=\s*-50`, `get(...lower_limit...-50)`, `ThresholdAbortState\([^)]+\)` (a LIVE
construction — the safe inert `ThresholdAbortState()` empty-parens default is deliberately not
flagged), `resolve_threshold_for_delta\(`, and a hidden default on `inner_lower_limit` itself
(`inner_lower_limit::Float64 = ...`). Allowlists `test_*.jl`, the D4/scaled synthetic-diagnostic
builders (`context.jl`/`context_scaled.jl`, explicitly documented as out of scope), and the
existing `*bench*`/`*profile*`/`*diag*`/`c8_`-`c33_`-prefixed diagnostic script families. **Runs
clean (0 violations) against this branch's current state.**

## 10. Outer wall-clock hard limit — documented, no new abort added

Confirmed directly (not guessed) by reading the actual `.opt` file every one of the 3 production
entry points loads by default (`d20_real_setup_design`'s own default `inner_loop_opt =
joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt")`; none of `run_cm_upper_checkpointed`/
`run_originzc_upper_checkpointed`/`run_polish_checkpointed_unified` override it):
```
maxit        100          # finite iteration cap
maxtime_cpu  1e+08        # effectively unlimited
maxtime_real 1e+08        # effectively unlimited
```
**The only native per-inner-solve limit currently in force is the 100-iteration cap — there is no
finite native per-inner-solve wall-clock limit.** This directly corroborates the prior session's own
production-risk finding (§10.4 of the prior report): a single pathological inner solve can run for
an arbitrarily long wall-clock time (bounded only by however long 100 KNITRO iterations take at that
point, which for a genuinely divergent trial is exactly what `lower_limit=-10` now cuts short much
sooner than `-50` did — this is precisely why the P0 fix matters here, not a separate issue).
Per the task's explicit instruction, **no new custom timer/interrupt mechanism was added** — this
section documents the existing behavior only.

## 11. P3: outer-algorithm pilot — not run

Explicitly the lowest-priority item (task §12: "This pilot is lower priority than the lower-limit
and allocation fixes"). Not run this session given the machine contention documented in §0 and the
time already spent completing P0 correctly plus the P1 source audit. The prior session's own
`pin_outer_algorithm=true` vs `algorithm=auto` comparison (§10.3 of the prior report) remains the
most recent evidence, but it was run under the OLD `lower_limit=-50` — genuinely stale for this
specific question now that `-10` is the production value.

## Branch/SHA and clean status

- Branch: `fix/fullA-lower-limit-and-hotpath-2026-08-06`, forked from `origin/production/fullA-exact`
  @ `0ed2823`.
- Worktree: `/bbkinghome/edav/cdw_worktrees/fullA-lower-limit-and-hotpath-2026-08-06` (the only one
  created this session).
- 1 new branch, 1 new worktree, exactly as instructed.

## Final verdict

```
PRODUCTION_LOWER_LIMIT = -10.0

LOWER_LIMIT_SOURCE_OF_TRUTH = required_kwarg_no_scientific_manifest_on_this_branch
    (plain Julia required keyword argument, no `= value`, at d20_real_setup / d20_real_setup_design
    / run_cm_upper_checkpointed / run_originzc_upper_checkpointed / run_polish_checkpointed_unified
    -- ScientificManifest.jl does not exist on origin/production/fullA-exact's ancestry, see §2)

ACTIVE_PRODUCTION_MINUS50_OCCURRENCES = 0
    (confirmed by static lint, scripts/static_lower_limit_guard_2026-08-06.sh, 0 violations;
    full source audit in LOWER_LIMIT_SOURCE_AUDIT.csv)

CUSTOM_THRESHOLD_ABORT_IN_PRODUCTION = false
    (ThresholdAbortState no longer constructed with a live threshold anywhere reachable from the 3
    production entry points; module retained for its own standalone diagnostic tests only, per
    task's own "keep isolated" instruction rather than "delete")

LOWER_LIMIT_PROPAGATION =
    unrestricted:PASS (live, D20/W=5000, ALL 20 CHECKS PASS)
    flexible_CM:PASS
    common_Frechet:PASS
    ZC_only(origin_zc):PASS
    CM_plus_ZC(cm_meanzc):PASS
    (full log: repo_scratch/fullA-lower-limit-and-hotpath-2026-08-06/ll_propagation_test.log)

FRECHET_HESSIAN_ALLOC =
    before: ~45.0 MB/callback (prior session's measurement, not reproduced this session)
    after: NOT MEASURED (no fix applied; §7.2's S2total finding not fixed, sized ~64KB not the
        dominant cost)
    unavoidable_output_bytes: NOT COMPUTED this session (deferred with P1's live pass)

CMZC_HESSIAN_ALLOC =
    before: ~124.5 MB/callback (prior session's measurement)
    after: NOT MEASURED
    unavoidable_output_bytes: NOT COMPUTED

TOP_AVOIDABLE_ALLOCATION = not_confirmed_live_this_session
    (source-level candidate found: S2total = dropdims(sum(cctx.Stab2,dims=3),dims=3) in
    hessian_cm_structured!, cm_hessian_architectures.jl:1500 -- real and avoidable, but
    back-of-envelope sized ~64KB, i.e. almost certainly NOT the dominant tens-of-MB site; a live
    Profile.Allocs pass is required to find the actual dominant site and was not run this session)

BINTABLE_REDUCTION_SHARE =
    4T: not_measured_this_session
    10T: not_measured_this_session
    20T: not_measured_this_session

THREAD_SCALING_AFTER =
    common_Frechet_Hessian: not_remeasured_this_session (prior session: 1.79x at 4T->20T, unrelated
        to this session's lower_limit change)
    CM_plus_ZC_Hessian: not_tested

OUTER_ALGORITHM_AFTER_LL10 = inconclusive
    (not run this session; prior pin_outer_algorithm comparison was under the old lower_limit=-50,
    stale for this specific question)

PRODUCTION_RELEASE = not_merged_pending_user_review
    (P0 code changes complete and lint-clean on this branch; not yet committed/tagged as of this
    document's initial draft -- see commit log on this branch for the actual SHA once committed)

NEW_ABORT_MECHANISM_ADDED = false
NEW_HESSIAN_MATH = false
NEW_OUTER_GRADIENT_MATH = false
DENSE_PRODUCTION_GH_USED = false
CAMPAIGN_LAUNCHED = false
NEW_BRANCHES_CREATED = 1
EXTRA_WORKTREES_CREATED = 1
```
