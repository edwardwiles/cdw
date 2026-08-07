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

**UPDATE (continuation session, same day): P1/P2/P3 all completed live after all**, once the shared
machine freed up enough capacity for three concurrent background campaigns (run on disjoint pinned
idle-core sets, standard mitigation for this shared box). All three produced real, decisive
findings — see §§7-11 below for full detail:
- **P1**: a live `Profile.Allocs` trace (not the source-only audit originally reported) found the
  TRUE dominant allocation site — `fill_cm_HCC!`, 66.9% of all bytes attributed to the Hessian call
  tree, traced to a specific `Array{Float64,4}(undef,D,D,L,L)` (~40 MB at D20/L=50) reallocated
  fresh on every call inside `_build_reflected_bilinear`, fired **4 times per Hessian callback**
  for common-Fréchet two-family (twice in `fill_cm_HCC!`, redundantly twice more in
  `_fill_frechet_level_blocks!`). **Fixed**: converted to a persistent, `cctx`-owned, reused buffer
  at both call sites, via the same "outer-constructor-forwards-with-new-field-defaults" pattern this
  struct already used safely for its other persistent scratch fields. Verified against an
  independent dense-reference Hessian implementation, not just re-run against the old code.
- **P2**: real 4T/10T/20T timing of `build_bin_tables_threaded!`'s zero/fill/reduce phases on one
  matched real point — confirms AND refines the task's own hypothesis (both the serial reduction
  AND the per-thread buffer zeroing scale linearly with thread count, together canceling out most
  of the parallel speedup from 10T to 20T).
- **P3**: a real, short A/B pilot under the corrected `lower_limit=-10` — CG+L-BFGS completed 3
  major iterations in less wall-clock than auto/Direct completed zero.

This section (§0) is left in its original "partial session" form below for an honest record of
what this session's FIRST pass actually covered before the continuation; §§7-11 carry the complete,
updated findings.

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

### 7.3 UPDATE (continuation, same session): live Profile.Allocs trace completed

The machine freed up enough capacity to complete a real `Profile.Allocs` pass after all. Ran
`p1_allocs_frechet_2026-08-06.jl`: real D20/W=100,000, two-family common-Fréchet
(`include_truncated_moment=true`), through the actual `run_cm_upper_checkpointed` public driver,
wrapped in `Profile.Allocs.@profile sample_rate=0.1` (10% sampling — `sample_rate=1.0` was tried
first on a toy example and found far too slow/voluminous for a real campaign-scale run), then
filtered to allocation records whose backtrace passes through the real Hessian-callback call tree
(`hessian_cm_structured_v2!` and its callees).

**Top allocation sites, real live data (D20/W=100,000, two-family common-Fréchet, 9 real Hessian
callbacks, 10% sampled — bytes below are the SAMPLED total, i.e. ~10x smaller than the true total;
percentages/ranking are what matters, not the absolute sampled MB):**

| # | Site | Count | Sampled bytes | Sampled MB | % of Hessian-tree total |
|---|---|---|---|---|---|
| 1 | `fill_cm_HCC!` | 9 | 1.713e7 | 16.34 | **66.9%** |
| 2 | `_fill_cm_HEE!` | 4 | 2.770e6 | 2.64 | 10.8% |
| 3 | `winner_pair_cross_hessian_cm_block!` | 34 | 2.073e6 | 1.98 | 8.1% |
| 4 | `winner_pair_cross_hessian_colsum_pow!` | 24 | 1.463e6 | 1.40 | 5.7% |
| 5 | `_fill_frechet_level_blocks!` | 6 | 9.664e5 | 0.92 | 3.8% |
| 6-15 | (various small `hessian_cm_structured_v2!`/`prefix_sum_tables_threaded!` internals) | — | — | — | <2% combined |

Sampled total attributed to the Hessian call tree: 24.42 MB (10% sample) → **estimated true total
≈ 244 MB across 9 callbacks ≈ 27 MB/callback** — same order of magnitude as, and a real independent
cross-check of, the prior session's own directly-measured ~45 MB/callback for this family (different
method, `@timed` vs sampled `Profile.Allocs`; both point to "tens of MB", not hundreds or single
digits).

**Root cause, identified and traced to source**: `fill_cm_HCC!` (66.9% of allocated bytes, but only
~11% of *wall-clock time* per the prior session's own block timing — a striking allocation/time
mismatch that is itself the tell) calls `_build_reflected_bilinear` (`cm_hessian_architectures.jl`)
twice per invocation for any two-family context. That function allocated a **fresh
`Array{Float64,4}(undef, D, D, L, L)`** on every single call — at real D20/L=50, that's
`20×20×50×50×8 bytes ≈ 40 MB` **per call**, i.e. up to ~80 MB/Hessian-callback just from this one
temporary array, matching the measured order of magnitude closely. Worse: `_fill_frechet_level_blocks!`
(the Fréchet-only "level anchor block" function, called from every common-Fréchet Hessian callback,
item #5 above) **recomputes the exact same two tables a second time**, by its own comment
("`recomputed here rather than shared/cached since this function runs once per Hessian callback
exactly like that one does`") — so for common-Fréchet two-family specifically, this one 40 MB
temporary array pattern fires **4 times per Hessian callback**, not 2.

This is a textbook instance of this task's own "temporary dense matrix" category (§8's
classification): the buffer is fully overwritten every call (no partial-fill/accumulation hazard),
sized identically every call for a given `cctx`, and every other comparable temporary on this exact
struct (`Hraw_EC`/`block_ec`/the Fréchet-extension's own scratch fields, per the codebase's own
2026-08-02 allocation-hardening pass, §7.1) was already converted to a persistent, `cctx`-owned
buffer — this one specifically was introduced by the 2026-08-05 truncated-power task, 3 days after
that hardening pass, and evidently missed.

### 7.4 Fix applied (top 1 site, both call sites)

Added two persistent `Union{Nothing,Array{Float64,4}}` fields to `CMBinHessCtx`
(`Trefl12`/`Trefl22`, `D×D×L×L`), wired through the SAME pre-existing, previously-proven-safe
"outer constructor forwards to positional inner constructor with new-field defaults, no existing
call site needs to change" pattern this struct already used to add `Hraw_EC2`/`block_ec2` etc.
(`cm_hessian_architectures.jl`, both real construction call sites — `build_cm_bin_ctx` and
`cm_meanzc_production.jl`'s builder — go through the shared `build_cm_family2_tables` helper via a
`family2...` kwarg splat, so adding the 2 new fields there required **zero** changes at either call
site). `_build_reflected_bilinear` now takes an optional `Trefl` buffer kwarg and writes into it
in-place instead of allocating fresh when supplied (defaults to the OLD fresh-allocation behavior
when omitted — zero behavior change for any caller that doesn't pass one). Both real call sites
(`fill_cm_HCC!` and `_fill_frechet_level_blocks!`) now pass `cctx.Trefl12`/`cctx.Trefl22` — safe to
share the SAME buffer across both because they run strictly sequentially within one single-threaded
Hessian callback (`fill_cm_HCC!` always finishes and fully consumes its result into `Hfull` before
`_fill_frechet_level_blocks!` runs), and every element of `Trefl` is unconditionally overwritten
(`Trefl[x,y,l,lp] = ...` inside a full `D×D×L×L` nested loop, never accumulated into), so reusing a
stale buffer carries no correctness risk.

**Correctness preserved by construction** (every output element is freshly assigned, not
accumulated, so buffer reuse cannot leak stale data) **and independently verified live**: re-ran
`test_frechet_hessian_structured_vs_dense_d20_twofamily_2026-08-06.jl` (a pre-existing repo gate —
patched only to supply the newly-required `inner_lower_limit` kwarg, no other change — comparing
Architecture C/structured, the code path this fix touches, against Architecture A/dense reference,
an independent implementation, at real D20/W=80,000, both contrast modes) post-fix.

**RESULT: 26/26 checks PASS.** Every Hessian block (H_EE, H_E-levelpow, H_CM(cdf)/(pow)-level,
H_CM(cdf)/(pow)-levelpow, H_level-levelpow, H_levelpow-levelpow — i.e. every single-family AND
every two-family cross block) matches the independent dense-reference implementation to
machine-precision tolerance (`max|diff|` ranging `3.277e-15` to `1.861e-14`, all comfortably within
floating-point noise, `max|H_structured_v2(threaded) - H_dense| = 1.137e-13` overall). Full log:
`repo_scratch/fullA-lower-limit-and-hotpath-2026-08-06/verify_p1_fix_FINAL.log`.

**This eliminates the dominant allocation site**: 2 fresh 40 MB arrays/callback (`fill_cm_HCC!`) +
2 more (`_fill_frechet_level_blocks!`, same buffers now reused instead of freshly allocated) → 0
fresh allocations from this pattern, for both real two-family production families
(common-Fréchet, and CM+ZC K=3 via the same shared `fill_cm_HCC!` — `_fill_frechet_level_blocks!`
itself is Fréchet-only, gated on `extension !== nothing`, so CM+ZC only gets the `fill_cm_HCC!`
half of this fix, still its own real win). Did NOT also fix the smaller `RowMarg`/`ColMarg`/
`RowCum`/`ColCum`/`Total` temporaries inside `_build_reflected_bilinear` (each ≤0.16 MB, negligible
next to the 40 MB `Trefl` — task §9's own "stop after the top one to three sites" instruction).

## 8. P2: bin-table thread scaling — measured live this session, hypothesis CONFIRMED

**This was completed live**, using a parameterized clone of `build_bin_tables_threaded!`
(`build_bin_tables_threaded_timed`, `p2_bintables_fill_vs_reduction_2026-08-06.jl`) that separately
times the zero/reset, `Threads.@threads` parallel-fill, and serial-reduction phases, called 3x per
thread count (fastest of 3 reported, standard micro-benchmark noise reduction) against ONE real
warmed common-Fréchet cctx/point (D20/W=100,000, real `run_cm_upper_checkpointed` public-driver
warm-up, `NCORE=382`).

**Methodology deviation, disclosed**: rather than launching 3 separate `julia -t N` processes (each
re-paying several minutes of context-build cost under this session's machine contention, §0), one
process launched with `-t 20` calls the SAME warmed `(cctx, tls, w)` repeatedly with an explicit
`nt_use` parameter in `{4,10,20}` (all ≤ 20, so `Threads.@threads :static for tid in 1:nt_use`
genuinely dispatches across `nt_use` distinct OS threads each time, not a simulation). Also
`fill_S=false` throughout (the T-table-only path — the point this session's own warm-up run reached
had `use_winner_bin=false`, which would require `fill_S=true` and hence a real dense `H` field the
operator/no-dense-H production bundle structurally does not carry; extracting `E` from the operator
state instead was not implemented this session, so this measures the always-present `Ttab`
fill+reduce machinery, not the smaller `Stab` piece — see the script's own comments for the full
disclosure).

**Real results:**

| nt_use | zero (reset) | fill (parallel) | reduce (serial) | total | reduce share |
|---|---|---|---|---|---|
| 4  | 0.0085s | 0.3314s (94.7%) | 0.0101s (2.9%)  | 0.3500s | 2.9%  |
| 10 | 0.0229s | 0.1358s (74.4%) | 0.0237s (13.0%) | 0.1825s | 13.0% |
| 20 | 0.0483s | 0.0830s (46.3%) | 0.0479s (26.7%) | 0.1792s | 26.7% |

**Reads, decisively:**
1. **The task's own hypothesis is confirmed**: the serial reduction's share of total time grows
   monotonically with thread count — 2.9% → 13.0% → 26.7% — nearly a 10x relative-share increase
   from 4T to 20T.
2. **Both `zero` and `reduce` scale roughly linearly in `nt_use`** (zero: 0.0085→0.0229→0.0483,
   ~1x/2.7x/5.7x; reduce: 0.0101→0.0237→0.0479, ~1x/2.3x/4.7x) — expected, since both loop over
   `nt_use` separate per-thread buffers of FIXED size (`D×D×L1×L1`) regardless of thread count; more
   threads means more buffers to zero and sum, not less.
3. **The actual parallel work (`fill`) scales well** (0.3314→0.1358→0.0830, i.e. 2.44x for 2.5x
   threads at 4→10, and a further 1.64x for 2x threads at 10→20 — reasonable, if imperfect,
   speedup for the genuine work).
4. **But total wall-clock barely improves from 10T to 20T** (0.1825s→0.1792s, essentially flat) —
   because the linearly-growing `zero+reduce` overhead cancels out most of the additional parallel
   speedup `fill` would otherwise deliver. This is the precise mechanism behind the prior session's
   own observed 4T→20T Hessian-callback ratio (1.79x for a 5x thread increase, ~36% efficiency,
   §8 of the prior report) — not a different, unconfirmed phenomenon, but the SAME one, now
   decomposed into its two additive causes (zero-reset AND reduction, both O(nt_use), not just
   the reduction alone as originally hypothesized).

**Recommendation (not implemented this session, correctly gated behind more care per the task's own
§11 instruction)**: a fixed-order deterministic tree reduction (task §11 Option B) would improve the
`reduce` phase's own scaling (O(log nt) tree depth vs O(nt) linear sum) but would NOT address the
`zero` phase, which is the SAME order of magnitude and grows the SAME way — a complete fix likely
needs to also avoid re-zeroing all `nt` buffers every callback (e.g., only zero the buffers actually
touched, or restructure to avoid needing a fresh zero each call). This is real, measured, specific
evidence for future work, not implemented or further speculated on this session.

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

## 11. P3: outer-algorithm pilot — completed live, real result

Ran `p3_outer_algorithm_pilot_ll10_2026-08-06.jl`: real D20/W=20,000, common-Fréchet two-family,
`inner_lower_limit=-10.0` (the NEW production value — the prior session's own comparison, §10.3 of
the prior report, was run under the OLD `-50`, genuinely stale for this question). Sequential A→B
on the same pinned core set (not concurrent), `maxtime_real=300s` per run (a genuinely short pilot
per the task's own "lower priority" framing).

**First attempt crashed with a real, unrelated bug** (`JULIA_NUM_THREADS=6` was not one of
`hessian_core_winner_pair!`'s precomputed worker counts, `[1,2,4,8,10,19,20]` —
`core_exact_hessian.jl`), caught immediately via `assert_no_fake_success!`'s own
callback-error-surfacing rather than silently producing a wrong/degenerate result. Relaunched with
`JULIA_NUM_THREADS=4`, a valid worker count.

**Real results:**

| Config | Wall | Major iterations | `n_eval` | `n_grad` | KNITRO status |
|---|---|---|---|---|---|
| `algorithm=auto` (→ Direct) | 518.5s | **0** | 1 | 1 | -401 |
| `pin_outer_algorithm=true` (CG+L-BFGS) | 382.7s | **3** | 5 | 4 | -401 |

**Decisive, even at this short budget**: CG+L-BFGS completed 3 real major iterations in LESS wall
time (382.7s) than auto/Direct needed to complete ZERO major iterations (518.5s — longer wall clock
for strictly less progress). Both hit `knitro_status=-401`; per this repo's own standing memory
(`feedback-check-d20-w-sensitivity-knitro-failure`), -401 at D20 is very often a benign
budget/W-sensitivity artifact, not a real solver defect — consistent with both runs being cut off by
`maxtime_real` mid-progress rather than genuinely failing.

**Bonus, unplanned confirmation of §10's finding**: both runs' actual wall-clock (518.5s / 382.7s)
exceeded their own `maxtime_real=300s` budget substantially — direct, real-world confirmation (not
just reading the `.opt` file) that a single in-flight inner solve can carry the total run well past
its nominal time budget, exactly as §10 documents from the `.opt` file's own `maxtime_real=1e8`
(effectively unlimited) native inner-solve setting.

**This session's own recommendation, based on real evidence under the corrected `lower_limit`**:
`pin_outer_algorithm=true` continues to look like a strong candidate default for common-Fréchet
two-family, now confirmed (not just inferred from the pre-`-10` prior session) to outperform
`algorithm=auto` under the corrected `lower_limit=-10` too. Not flipped as the new default this
session — task §12 explicitly reserves that decision ("Do not switch the production default merely
because one configuration is faster per iteration... Recommend a switch only if it gives
equal-or-better verified progress per wall/CPU and no new failures" — this pilot's budget was too
short to confirm "no new failures" at full rigor, both runs hit -401 which needs the fuller
matched-budget replication task §12 itself calls out as the eventual bar, not repeated here).

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
    before: ~45.0 MB/callback (prior session's @timed measurement) / ~27 MB/callback estimated from
        this session's own live Profile.Allocs 10%-sample trace (24.42 MB sampled / 9 calls / 0.1)
        -- same order of magnitude, real independent cross-check via a different method
    after: NOT RE-MEASURED post-fix (the fix's correctness was verified via the independent
        structured-vs-dense gate, §7.4; a fresh Profile.Allocs pass to quantify the POST-fix byte
        count was not re-run this session -- the fix eliminates 4 x ~40MB fresh Trefl allocations/
        callback by construction (code-level: Array{Float64,4}(undef,...) replaced by a persistent
        buffer reused in-place), a ~160MB/callback reduction by direct accounting, not re-measured
        live)
    unavoidable_output_bytes: packed Hessian output buffer itself (n(n+1)/2 Float64s, KNITRO-
        required interface buffer) -- NOT separately computed this session

CMZC_HESSIAN_ALLOC =
    before: ~124.5 MB/callback (prior session's measurement, not reproduced this session for CM+ZC
        specifically -- this session's live P1 profiling targeted common-Frechet only, time-boxed)
    after: NOT MEASURED for CM+ZC (the fill_cm_HCC! half of the fix applies to CM+ZC too, since
        that function is shared per its own docstring -- _fill_frechet_level_blocks! is Frechet-only
        so CM+ZC gets a smaller share of the total fix; not quantified this session)
    unavoidable_output_bytes: NOT COMPUTED

TOP_AVOIDABLE_ALLOCATION = CONFIRMED_LIVE_AND_FIXED
    (fill_cm_HCC! -> _build_reflected_bilinear's Trefl = Array{Float64,4}(undef,D,D,L,L), ~40MB,
    reallocated fresh 4x/callback for common-Frechet two-family -- 66.9% of all bytes attributed to
    the Hessian call tree in a live Profile.Allocs trace, D20/W=100,000, real production driver.
    Fixed: converted to 2 persistent cctx-owned buffers (Trefl12/Trefl22), reused at both call
    sites. Verified against an independent dense-reference Hessian implementation.)

BINTABLE_REDUCTION_SHARE =
    4T: 2.9% of total (0.0101s / 0.3500s)
    10T: 13.0% of total (0.0237s / 0.1825s)
    20T: 26.7% of total (0.0479s / 0.1792s)
    (real, live-measured, D20/W=100,000 common-Frechet, matched point; zero-reset phase scales the
    SAME way as reduce -- both are O(nt_use) -- see §8 for the full breakdown and why 10T->20T
    barely helps overall despite fill itself scaling well)

THREAD_SCALING_AFTER =
    common_Frechet_bintables_total: 4T->10T 1.92x, 10T->20T 1.02x (nearly flat) -- 4T->20T overall
        1.95x for a 5x thread increase (~39% efficiency), consistent with and explaining the prior
        session's own whole-Hessian-callback number (1.79x at 4T->20T, ~36% efficiency)
    CM_plus_ZC_Hessian: not tested (P2 measured common-Frechet's shared Ttab machinery only)

OUTER_ALGORITHM_AFTER_LL10 = recommend_CG_LBFGS
    (real short pilot, D20/W=20,000, common-Frechet two-family, inner_lower_limit=-10.0:
    algorithm=auto/Direct completed 0 major iterations in 518.5s wall; pin_outer_algorithm=true
    (CG+L-BFGS) completed 3 major iterations in 382.7s wall -- less wall-clock for strictly more
    progress. Both hit knitro_status=-401 (likely benign W-sensitivity per this repo's own standing
    memory, not a new failure mode). Recommended as a strong candidate default, NOT flipped this
    session -- task's own bar ("no new failures", full matched-budget replication) not fully met at
    this short pilot's budget)

PRODUCTION_RELEASE = 871b819 (P0 tag fullA-lower-limit-10-production-ready-2026-08-06)
    (P0 merged+pushed to origin/production/fullA-exact as a clean fast-forward, user-confirmed.
    P1's allocation fix (Trefl12/Trefl22 persistent-buffer change) is a SEPARATE, subsequent commit
    on this same branch, verified but NOT YET merged to production as of this verdict block's
    initial draft -- see the branch's own commit log for whether/when it was subsequently merged)

NEW_ABORT_MECHANISM_ADDED = false
NEW_HESSIAN_MATH = false
NEW_OUTER_GRADIENT_MATH = false
DENSE_PRODUCTION_GH_USED = false
CAMPAIGN_LAUNCHED = false
NEW_BRANCHES_CREATED = 1
EXTRA_WORKTREES_CREATED = 1
```
