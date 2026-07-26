# Production Optimization Stack Audit — Baseline Findings + Remediation Sizing

**Date:** 2026-07-26
**Status:** Phase 0 + static source-level audit COMPLETE. Runtime instrumentation and the five
300-second profiling runs are **explicitly paused**, at the user's request, until fix decisions
are made on the findings below. Nothing in production has been modified. This document plus its
sizing section are the basis for that decision.

**Scope of this document:** the original task (`audit the production optimization stack across
all counterfactual families`) asked for a live-evidence 5-family x 7-area kernel matrix, a
14-item supporting-plumbing matrix, opt-in runtime counters, and five real 300s KNITRO profiling
runs with wall-clock/allocation attribution. This document covers the **static source-level**
portion of that work only (Phase 0 ancestry, entry-point/manifest verification, and the full
7-area + supporting-plumbing inventory across all five families) plus a sizing analysis for the
fixes under consideration. The runtime instrumentation and profiling phases have not started.

---

## 1. Phase 0 — production state (verified)

- Worktree: `/bbkinghome/edav/gravity_robustness/worktrees/audit-production-5x7-2026-07-26`,
  checked out clean at `production/fullA-exact`.
- Local tip `f1fa8e770759c62b3f96c1024dd310f235ea463e` == origin tip (bidirectional
  `git merge-base --is-ancestor` both directions confirmed). Working tree clean.
- Tag at tip: `common-frechet-cdf-cm-plus-level-production-ready-2026-07-26`.
- **Common-Fréchet CDF is genuinely canonical production** — this confirms the prior session's
  addendum: it is NOT `PORT_READY_NOT_MERGED`. All five families are live production rows, not
  four.
- Ancestry re-verified for every cited release tag against current HEAD. All confirmed ancestors
  except one **naming trap, not a real gap**: the tag
  `shared-winner-pair-core-hessian-production-ready-2026-07-25` (`32fce7f`) points to a pre-merge
  candidate that is **not** an ancestor of HEAD — but the actual merge landed via a
  differently-named branch, `release/shared-winner-pair-final-merge-2026-07-25` (`61a3bd6`),
  which **is** confirmed an ancestor (`61a3bd6` -> `a153628` -> `f1fa8e7`). The feature is
  genuinely in production; don't grep for it by the pre-merge tag.

## 2. Entry points and manifest (verified from live source)

| Family | Entry point | File | Key production defaults |
|---|---|---|---|
| UNRESTRICTED (task-designated) | `run_polish_checkpointed_unified` | `c10_d20_production_driver_unified.jl:129` | `layout` required (no default); `destination_sample=:exclude_row`; `price_cache_backend=:cplus` |
| FLEXIBLE_CM | `run_cm_upper_checkpointed` | `cm_checkpoint.jl:588` | `marginal_restriction=:common_flexible`, `cm_extension=:cm_only`, `cm_gradient_backend=:cplus`, `A_coordinate_mode=:legacy_z`, `L=10` (pass `L=50` explicitly for production-grid profiling) |
| COMMON_FRECHET_CM | same function | same file | `marginal_restriction=:common_frechet`, `cm_extension=:cm_only` (guarded — cannot combine with meanzc) |
| FLEXIBLE_CM_PLUS_ZC | same function | same file | `cm_extension=:cm_plus_equal_means_zero_covariance`, `K_mean=1,K_pair=1` |
| ZC_ONLY | `run_originzc_upper_checkpointed` | `cm_originzc_checkpoint.jl:451` | `distribution_restriction` required (no default), `K_mean=1,K_pair=0` |

Two concrete manifest gaps found (Section 4, items B5 and C1) and one entry-point ambiguity found
that changes what "the production driver" even means for UNRESTRICTED (Section 3, item A).

## 3. (A) UNRESTRICTED entry-point ambiguity — dated and root-caused

The task designates `run_polish_checkpointed_unified` as the UNRESTRICTED entry point. The
**only committed CLI/process-launchable stage-runner script** for unrestricted,
`unrestricted_stage_runner.jl`, instead calls `run_profile_checkpointed`
(`c10_d20_production_driver.jl:518`) — a different, older function.

**Root cause, confirmed via `git log`:**
- `unrestricted_stage_runner.jl` was authored 2026-07-24 10:58 (commit `f2da001`), explicitly to
  give unrestricted CLI/process-group-supervisor parity with the CM/origin-ZC stage runners. At
  that time `run_polish_checkpointed_unified` **did not exist yet**.
- `c10_d20_production_driver_unified.jl` (containing `run_polish_checkpointed_unified`) was first
  added 2026-07-25 09:55 (commit `2127b3b`) — the next day.
- `unrestricted_stage_runner.jl` was touched again 2026-07-25 21:20 (commit `a30c402`, ~11 hours
  after the unified driver existed) for unrelated fixes (public-driver runtime counters), but was
  **never repointed** to the new driver.

This is very likely simple staleness, not a deliberate design choice — but it has real
consequences: `run_profile_checkpointed` has no `A_coordinate_mode`/`trade_elasticity_mode`
concept at all (pure legacy z-space, fixed-theta only) and **hardcodes `algorithm=3`**
(Active-Set/SLQP) whenever `pin_outer_algorithm=false` (its own default,
`c10_d20_production_driver.jl:741-744`) — every other driver in this audit leaves KNITRO's
`algorithm=auto`. It is also very plausibly the exact "fixed-gp minimum-divergence profile
routine" the original task brief warns not to accidentally profile — the name match is not a
coincidence.

**Sizing the fix** (repoint `unrestricted_stage_runner.jl` at `run_polish_checkpointed_unified`):
- **Includes**: `c10_d20_production_driver_unified.jl` itself hard-asserts its dependencies via
  `isdefined(Main, ...)` checks (`outer_coordinate_layout.jl`, `flexible_theta.jl`,
  `flexible_theta_aspace_production.jl`, `production_backend_manifest.jl`,
  `knitro_outer_algorithm.jl`) — these all need to be included before it, in addition to the
  existing `c10_d20_production_driver.jl` include (which `_unified` itself also requires already
  be loaded).
- **w0 construction**: not a from-scratch problem — a template already exists.
  `matched_comparison_three_arm.jl:51-59` shows the established pattern:
  `layout = make_layout(trade_elasticity_mode=..., A_coordinate_mode=..., gp_coordinate_mode=:raw)`
  then `w_start = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout)` then
  `run_polish_checkpointed_unified(label, find_smallest, w_start; layout, ...)`. The stage
  runner's current `g0`/`zfree0` calibration derivation would need to feed into
  `reduce_to_w_unified` rather than being passed as two separate arguments.
- **Checkpoint-format break**: this is the real cost. `run_profile_checkpointed` writes
  `D20CheckpointV4` (legacy schema-4); `run_polish_checkpointed_unified` writes
  `D20CheckpointUnified` (`CHECKPOINT_SCHEMA_UNIFIED=1`, a completely separate schema/type). Any
  currently-committed checkpoints from real campaigns launched under the current script would
  **not** be resumable by a version of the script pointed at the unified driver without an
  explicit migration path or a dual-mode resume branch.
- **Net size estimate**: a contained, single-file change (roughly 40-80 lines: new includes,
  layout construction, w0 construction swapped in for the current g0/zfree0 pair, resume-path
  handling) plus a new smoke/byte-identity gate confirming that `A_coordinate_mode=:legacy_z,
  trade_elasticity_mode=:fixed` under the new driver reproduces the old script's results exactly
  before anyone relies on the new options (`:powered_aspace`, flexible theta) it would newly
  unlock. Moderate, not trivial — the main real risk is the checkpoint-format break for any
  in-flight campaign, not the code change itself.

## 4. (B) Fixable gaps — sizing

### B1. Dense per-iterate inner FG callback in the four restricted families — HIGHEST PAYOFF, user already flagged this

All four restricted families register the same generic `_callbackEvalFG_inner_profiled!`
(`oracle_fast.jl:102-110`) for their inner KNITRO dual solve, whose objective callable performs a
**dense `BLAS.gemv!` against a fully materialized `H` matrix on every single inner-solve
iterate** — reachable under ordinary default settings, gated behind no opt-in flag. Only
UNRESTRICTED's inner FG loop is genuinely O(W·D) winner-gather/scatter
(`compressed_dual_contraction`/`compressed_transpose_contraction`).

**A validated, already-built, faster kernel exists**: `cm_lookup_kernels.jl` /
`cm_lookup_live_knitro.jl`'s `_callbackEvalFG_inner_cmlookup!` — explicitly O(W·(D-1)), grid-
resolution-independent, and per its own header comment "already validated to machine precision
against the dense `obj(x,g)` callable in `c12i_validate_lookup_fg.jl`". It is wired into **zero**
production entry points — reachable only from `c12i_*`/`c13_*`/`c14_*`/`bench_*` benchmark and
gate scripts.

**Important correction to the initial framing, found during sizing**: this kernel is CM-grid
specific — `CMLookupState` (`cm_lookup_kernels.jl:194-212`) carries `ncm`/`L`/`bins`/`R` fields
that only exist for families with a CM threshold grid. **It therefore applies to three of the
four restricted families — FLEXIBLE_CM, COMMON_FRECHET_CM, and FLEXIBLE_CM_PLUS_ZC — but not
ZC_ONLY**, which has no CM grid at all (only a small, fixed-size dense mean/pair restriction
block, ~210 columns at D=20/K_mean=1/K_pair=1). ZC_ONLY's dense-FG cost may matter far less in
relative terms given its much smaller restriction dimension, but that is an open runtime
question, not yet measured.

**The integration is more than a one-line callback swap**, but is still well-contained:
- `cm_lookup_live_knitro.jl`'s own `inner_loop_KNITRO_cmlookup` bundles a **dense** Hessian
  callback (`_callbackEvalH_inner_cmlookup!`, `KN_DENSE_ROWMAJOR`, calling the standard dense
  callable) for isolated microbenchmark purposes — it does **not** use the production
  Architecture-C/winner-pair structured Hessian (`archC_hess_cb_builder` /
  `archC_frechet_hess_cb_builder`). Wiring the fast FG kernel into production means building a new
  registration function that uses the lookup FG callback but keeps the **existing** production
  Hessian-callback builder, not the microbenchmark's own dense one.
- The existing production inner-loop registration, `inner_loop_KNITRO_archgeneric`
  (`cm_hessian_architectures.jl:672`), is **already parameterized** on `hess_cb_builder` — this is
  exactly how `archC_hess_cb_builder`/`archC_frechet_hess_cb_builder`/
  `archA_partitioned_hess_cb_builder` get plugged in today per family, which makes decoupling the
  FG-callback choice from the Hessian-callback choice a natural fit, not a redesign.
  - Genuine wrinkle: KNITRO attaches one `userParams` per callback pair (FG+Hessian share it). The
    existing Hessian-callback closures expect `userParams == obj` (e.g. `archC_hess_cb_builder`'s
    closure does `o = userParams`); the lookup FG path needs `userParams == st::CMLookupState`.
    Because `CMLookupState` already carries a `.obj` field, the fix is a thin adapter — either
    change the handful of existing Hessian-closure bodies from `o = userParams` to
    `o = userParams.obj`, or wrap each `hess_cb_builder`'s returned closure once with a generic
    unwrapper. Low risk, mechanical, a few lines.
- **Call sites needing the swap** (confirmed via grep of `inner_loop_internal_archgeneric(`):
  `cm_production_bundle.jl` (flexible CM), `cm_frechet_cplus.jl` (common-Fréchet),
  `cm_meanzc_production.jl` (CM+ZC) — three call sites, one per applicable family. (ZC-only's own
  two call sites in `cm_originzc_production.jl` are excluded per the CM-grid-only scope above.)
- **Net size estimate**: moderate. A new `inner_loop_KNITRO_cmlookup_production`-style function
  (~40 lines, closely mirroring `inner_loop_KNITRO_archgeneric`), a small Hessian-closure adapter
  reused across the three existing `hess_cb_builder`s, and updating three call sites — gated
  behind a new opt-in kwarg (e.g. `inner_fg_backend::Symbol = :generic | :cmlookup`, matching the
  codebase's own established `:reference`/`:cplus`-style backend-switch pattern) so it can be
  validated against the current dense-BLAS path end-to-end (ideally at real D=20/L=50/W=80,000
  scale, machine-precision agreement, matched call counts) before any default is flipped. The
  microkernel-level numerical validation groundwork already exists
  (`c12i_validate_lookup_fg.jl`); what's missing is the production-Hessian-compatible wiring and
  an end-to-end gate at production scale for each of the three families.

### B2. Exact cache not wired into any of the four restricted families

`SafeExactCache{CMEvalKey}` / `cm_production_value_v2` (`cm_config.jl:198-243`) is a complete,
ready-to-use implementation — proper cache key (`CMEvalKey`, including `ctx_fingerprint` and CM
config so it can never collide with an unrestricted or differently-configured CM cache), a
cache-hit shortcut that bypasses `inner_loop_KNITRO_archgeneric` entirely, and a
feasibility-enforcing store contract. It is simply never called from `cm_checkpoint.jl` or
`cm_originzc_checkpoint.jl` (confirmed by exhaustive grep, zero hits) — `cb_F!`
(`cm_checkpoint.jl:1002`) calls a non-cached value function instead. **Sizing**: small,
low-risk per family — construct one `SafeExactCache{CMEvalKey}` at driver setup (mirroring the
unrestricted driver's own `SafeExactCache()` construction), thread it into the CM value call
inside `cb_F!` (and the ZC-only analog), and add a `use_exact_cache::Bool=true` kwarg to each
driver's public signature. The caching primitive itself needs no new development — only the
wiring.

### B3. Dual-bank warm starts not wired into any of the four restricted families

Same shape of gap as B2. `DualBank`'s selection logic (`select_warm_start`, layout-aware
`dual_bank_zfree`, `dual_bank.jl`) is generic and already handles coordinate-mode/flexible-theta
correctly (it was specifically fixed for theta-blindness for the unrestricted family). Wiring it
into CM/ZC would need: instantiate `DualBank(dual_bank_size)` at driver setup; call
`select_warm_start` where the inner-solve initial dual guess is currently set unconditionally
from `ctx.obj.x`; record each accepted solve back into the bank. **Sizing**: similar order of
effort to B2, slightly larger since it touches before/after-solve hooks rather than a single
lookup-and-return.

### B4. Stale docstring (trivial)

`cm_originzc_checkpoint.jl:448` states this arm "has no CM-grid block and always uses Architecture
A" — no longer true for H_EE (which dispatches to the shared `exact_winner_pair_parallel` backend
by default; only H_ER/H_RR remain dense). One-line comment fix, zero behavior change.

### B5. Manifest-label fix (trivial, ~1 line)

`resolve_flexible_cm_manifest`'s `restriction_hessian_backend` field
(`production_backend_manifest.jl:129`) reports `:cm_bin_prefix_plus_congruence` whenever
`is_meanzc` is true, regardless of whether R-congruence is actually engaged (it's gated on
`contrasts=:orthonormal`, while `run_cm_upper_checkpointed`'s own default is `contrasts=:anchored`
for both branches — meaning the label is very likely wrong under default settings today). Fix:
key the label on the real `cctx.R !== nothing` condition instead of `is_meanzc`. Reporting-only
fix; the Hessian computation itself is already correct either way.

## 5. (C) Lower-stakes real findings — fine to carry as baseline, not urgent to fix pre-profiling

- Common-Fréchet's startup manifest is print-only (`print_frechet_startup_manifest`,
  `cm_frechet_level.jl:322`) — no structured/JSON-able resolver like the other three families
  have in `production_backend_manifest.jl`. Real gap for the "machine-readable manifest"
  deliverable; no runtime-correctness impact.
- `price_cache_backend` kwarg on the unified driver has no live effect on `cb_G!` — vestigial
  dead configuration surface (only gates whether an unused workspace gets built).
- CM+ZC's CM-grid columns come from a persistent dense copy (`precalc_common_marginals_cdf`)
  rather than the bin-recompute (`fill_cm_columns_from_bins!`) plain flexible-CM uses — a minor
  architectural inconsistency between the two CM arms, both still upstream of the same generic
  dense FG contraction (B1) either way.
- Kill-mid-run test coverage is uneven: common-Fréchet has a genuine SIGKILL/process-group test
  (`frechet_killmidrun_driver.jl`/`_resume.jl`); flexible-CM has a weaker SIGTERM-only test
  (`c33_phase4_cm_shakedown_interrupt.jl`); UNRESTRICTED, CM+ZC, and ZC-only have none in this
  worktree.
- No `scripts/` directory or shell scripts exist in this specific worktree (they exist in ~30
  other sibling worktrees) — supervisor-script audit for this worktree is limited to the
  Julia-level stage-runner scripts.
- Restricted families' economic-core builders call the allocating `build_compressed_factual`
  (not the `!`-workspace variant) fresh on every single `moments!` call — no workspace-pooling
  equivalent to `CompressedFactualWorkspace` (which exists, wired only into unrestricted's
  DualBank-scoring branch) is wired into any of the four families' hot moment-build path.
  Allocation site confirmed by source; magnitude requires profiling.
- The envelope pre-solve screen's fixed-theta assumption is provably self-gating
  (`fast_range_screen.jl:167-168,754-762` — catches `EnvelopeUnsupportedContext` once at
  ctx-build time under flexible theta, degrading gracefully). Not a bug; the static
  `production_screen_stack(:unrestricted)` manifest string just always lists "envelope(...)"
  regardless of whether it's actually live — same flavor of imprecision as B5.

## 6. (D) Requires runtime verification, not resolvable from source — do not treat as decided

- Whether `algorithm=auto` resolves to the same real KNITRO algorithm across families once
  instrumented (`knitro_outer_algorithm.jl`'s own docstring says it resolves differently by
  problem structure even off one shared `.opt` file) — this compounds with finding (A): if
  UNRESTRICTED's real launch path is `run_profile_checkpointed` (hardcoded `algorithm=3`) while
  CM/ZC leave `auto`, the outer solves may be running genuinely different algorithms by default,
  confounding any cross-family evaluation-count comparison exactly as the original task brief
  warns.
- Whether the CM `build_bin_tables!`/`Ttab`/`Stab` construction pass is still the dominant
  Hessian-callback cost now that both H_EE moved to the shared winner-pair backend and the
  table-build itself was threaded (a prior, pre-both-changes profile found this pass at 80.2% of
  callback wall time — that figure is stale and needs remeasurement, not reuse).
- Actual dense-inline-fallback frequency for H_EE across the four winner-pair-sharing families
  under ordinary runs (reachable only via rare per-point conditions or explicit override per
  source, real hit rate unknown).
- Actual resolved Julia/BLAS/KNITRO thread counts at launch time (policy is deterministic given
  `Threads.nthreads()`, but the real number is a launch-environment fact).

## 7. Recommended sequencing

Given the sizing above:
1. **(A)** and **B1** are the two decisions the user already flagged as wanting more detail on.
   (A)'s fix is moderate-effort and its main cost is the checkpoint-format break for in-flight
   campaigns, not the code change. B1 is moderate-effort, well-scoped, and very likely the highest
   wall-clock payoff available in this audit — but only closes the gap for 3 of 4 restricted
   families (not ZC-only).
2. B2/B3 (exact cache, dual bank) are smaller, same-shape, low-risk wiring jobs — good candidates
   to bundle with B1 if the restricted-family drivers are already being touched.
3. B4/B5 are one-line, zero-risk, no reason not to include in any remediation pass.
4. Section 5 (C) items and Section 6 (D) items should be carried into the final baseline matrix
   as-is; (D) items specifically require the runtime instrumentation/profiling phase that is
   currently paused.

No code has been changed in the production worktree. All of the above is sized against the
current `production/fullA-exact @ f1fa8e7` source, read-only.
