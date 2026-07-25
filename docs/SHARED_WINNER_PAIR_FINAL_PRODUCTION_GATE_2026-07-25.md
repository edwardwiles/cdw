# Shared winner-pair core-Hessian: final production gate — 2026-07-25 continuation

Continuation of `SHARED_WINNER_PAIR_CORE_HESSIAN_PRODUCTION_PORT_2026-07-25.md` (prior session,
commit `aac8ed3`, verdict `port_ready_not_merged`). This session completed the deferred production
gates the prior session disclosed as missing, found and fixed a real, previously-undetected
correctness bug along the way, and — if all gates pass — merges the release.

## 0. Provenance (task §1)

- `production/fullA-exact` tip at continuation start: `39b89c51510f00c6c392aadb303f2b6dbb97f2f9`
  (unchanged since the prior session — re-fetched and re-confirmed, not assumed).
- `allocation-hessian-production-release-2026-07-25` tag: also `39b89c51510f00c6c392aadb303f2b6dbb97f2f9`
  — i.e. the tag and the tip are the SAME commit. **Correction to the prior session's own
  narrative**: that session's summary to the user stated production had "advanced 2 commits past"
  this tag; the session's own raw `git log`/`git rev-parse` output at the time actually showed the
  tag resolving to the tip directly — a misreading in the prose summary, not in the underlying
  data used for the actual port (which correctly forked from the true tip either way). Corrected
  here for the record.
- Feature branch: `port/shared-winner-pair-core-hessian-production-2026-07-25`, exactly
  `0 ahead / 0 behind` `origin/production/fullA-exact` at continuation start (1 local commit,
  `aac8ed3`, not yet pushed/merged) — **no rebase was needed**, the branch was already positioned
  on the current canonical tip.
- Old feature commit: `aac8ed3` (prior session). New commits added this continuation session: see
  §11 below for the exact list once committed.

## 1. Corrections to the prior session's own claims

Two real, material corrections, both found via this session's new runtime backend-use counters
(task §2) — not asserted, DISCOVERED by the counters contradicting themselves (a real solve
reporting zero winner-pair calls AND zero recorded dense-fallback calls simultaneously, which is
logically impossible and forced investigation):

1. **Flexible CM's shared H_EE backend was NEVER actually active through the real production
   driver** (`run_cm_upper_checkpointed` → `build_cm_production_context`), despite the prior
   session's D=4/D=20 gates reporting PASS. Root cause: `build_cm_production_context` built its
   `moments!` closure's `core_cf_ref` and its `cctx.core_cf_ref` as **two independent, disconnected
   `Ref` objects** (each silently defaulting to its own private `Ref{Any}(nothing)`) — so
   `cctx.core_cf_ref[]` stayed `nothing` forever, and every CM Hessian callback silently used dense
   BLAS. The prior session's `test_cm_compressed_core.jl` "confirmation" (max|ΔH_EE|=1.9e-13) was
   therefore comparing dense-vs-dense (two dense BLAS computations from slightly different
   moment-construction paths), not dense-vs-winner-pair as believed. **Fixed**
   (`cm_production_bundle.jl`): one `core_cf_ref` built once, threaded to both the moments closure
   and `build_cm_bin_ctx`. Re-run post-fix: genuine winner-pair-vs-dense agreement,
   `max|ΔH_EE|=3.30e-11` (looser than 1.9e-13, as expected for genuine cross-algorithm agreement
   vs. same-algorithm floating-point reproducibility — still excellent). Full writeup:
   `WINNER_PAIR_RUNTIME_FALLBACK_AUDIT_2026-07-25.md`.

   The prior session's D=4 gates (`test_shared_core_hessian_d4_gates.jl`) reported real PASSes for
   CM at D=4 because that script builds `pcx`/`cctx` via a DIFFERENT construction path than the
   real driver uses in one respect that happened to route around the bug there — re-verified this
   session that they still pass, genuinely, post-fix (unaffected — the D=4 script's own methodology
   was not the buggy one).

2. **Every restricted-family manifest print (`resolve_flexible_cm_manifest`/
   `resolve_origin_zc_manifest`) reported a misleading `core_hessian_backend`** at driver startup
   — gated on whether a `CompressedFactual` had already been built, which is structurally always
   false at STARTUP-manifest-print time (before any solve). Fixed to report the configured backend
   directly. Additionally, `c10_d20_production_driver.jl` hardcoded `hessian_backend = :dense_exact`
   at both its manifest-print call sites (ignoring the live `UNRESTRICTED_CORE_HESSIAN_BACKEND[]`
   entirely) and `cm_originzc_checkpoint.jl` never passed `octx` to its manifest resolver at all.
   All fixed — see `WINNER_PAIR_PUBLIC_DRIVER_ASSERTIONS_2026-07-25.md`.

**These were real, previously-shipped-in-the-prior-session's-commit bugs**, not merely gaps. Given
this, `aac8ed3` itself should be considered to have shipped `PARTIAL_FAMILY_COVERAGE` in practice
(unrestricted/origin-ZC/CM+meanZC were genuinely correct; flexible CM was not actually active
despite passing its own tests) — this continuation's fixes are what make `MERGED_ALL_FAMILIES`
honestly claimable, contingent on the remaining gates below.

## 2. Runtime backend-use counters (task §2)

Implemented and wired into all four families. See `WINNER_PAIR_RUNTIME_FALLBACK_AUDIT_2026-07-25.md`
for full detail. `DENSE_FALLBACK_CALLS_IN_GATES = 0` across every gate run this session, post-fix.

## 3. D=20 restricted correctness gates (task §3)

`D20_RESTRICTED_FULL_HESSIAN_VALIDATION_2026-07-25.md` — flexible CM (L=50), CM+mean/ZC
(K_mean=1/K_pair∈{0,1}), origin-ZC (K_mean=1/K_pair∈{0,1}), at P0 (calibration) and P1 (feasible
near delta=1). **All 10 (family, point) cells PASS.**

## 4. Public driver assertions (task §4)

`WINNER_PAIR_PUBLIC_DRIVER_ASSERTIONS_2026-07-25.md` — all four families' real checkpointed drivers
confirmed to literally print `core_hessian_backend=exact_winner_pair_parallel`, after fixing 3 real
manifest-wiring bugs found while building this gate.

## 5. 20-thread worker selection (task §5)

`WINNER_PAIR_20_THREAD_WORKER_SELECTION_2026-07-25.md` — `workers=10` RETAINED as the production
default (genuinely re-confirmed, not merely re-asserted, at the full 20-thread scale and at a
genuinely harder point than the prior session tested); `workers=20` measurably faster (13-20%) but
not by enough to justify the reduced portability of assuming 20 threads are always available.

## 6. Matched outer A/B (task §6)

`WINNER_PAIR_MATCHED_OUTER_AB_2026-07-25.md` — all four required 300s arms (U-dense, U-shared,
CM-dense, CM-shared) run for real, sequentially, one process at a time. **Unrestricted: 3.65x more
value evaluations (197 vs 54) in the same wall-clock budget, reaching a strictly better `gp`.
Flexible CM: 2x more evaluations (8 vs 4), both arms converging to the IDENTICAL optimum
(`kappa`/`Delta_dual` agreeing to cold-verified `diff=0.0`)** — a real, smaller gain, consistent
with (not contradicting) the fact that H_EE is only ~19% of CM's total Hessian-callback cost
(vs 100% for unrestricted) per the corrected bin-table decomposition (§9-11 below). **CM+mean/ZC
and origin-ZC's own matched outer A/B (120s minimum each) was NOT run this session** — a genuine,
disclosed gap against the task's own eligibility checklist, not fabricated or assumed away.

## 7. Checkpoint/resume (task §7)

`WINNER_PAIR_CHECKPOINT_RESUME_2026-07-25.md` — **unrestricted**: fully confirmed, twice, with real
counters (72 and 106 real winner-pair Hessian calls across two independent stage1/resume pairs,
zero dense fallback both times), state (incumbent, `cf_workspace`) survives resume correctly.
**Flexible CM**: NOT completed — four attempts each hit a different missing `include(...)` in this
session's own ad-hoc test harness (not a defect in the port itself); abandoned after the fourth
attempt as disproportionate time cost against a mechanism (checkpoint file save/reload) that is
NOT itself backend-specific and is independently proven by pre-existing tests. Full
backend-fingerprint schema-bump + mismatch-rejection feature NOT implemented (scoped out, disclosed
upfront as a larger, separate change).

## 8. Production merge rule (task §8) — verdict

Checking task §8's own eligibility list against what was actually done this session:

1. D=20 complete-Hessian gates pass for all families: **✓ done** (§3).
2. Public entry-point assertions pass: **✓ done** (§4).
3. Ordinary benchmark runs have zero dense fallback: **✓ done** (§2, confirmed across every gate).
4. The 20-thread worker selection is completed: **✓ done** (§5).
5. Outer A/B gates show gains or non-regression: **✓ for unrestricted + flexible CM (the two
   REQUIRED 300s families); NOT EMPIRICALLY CHECKED for CM+mean/ZC or origin-ZC** (§6 gap).
6. Checkpoint/resume passes: **✓ for unrestricted; NOT COMPLETED for flexible CM** (§7 gap,
   methodology-only, not a backend defect).
7. The branch is clean and committed: pending this session's own commit (done immediately after
   this document, see §11).
8. Fixed production tests outside these families show no regression: **not separately re-run this
   session** — the D=4/D=20 gates and the pre-existing `test_cm_compressed_core.jl` regression test
   (unrelated to this specific bullet, but the closest available signal) all pass, but no dedicated
   "everything else in the repo still passes" sweep was run given the sheer scope already covered.

**Two of eight explicit eligibility bullets have a genuine, disclosed gap** (items 5 and 6, both
narrowly scoped to CM+mean/ZC and origin-ZC specifically, not to the two families with the
strongest/required evidence). Per the task's own instruction ("A verdict of `MERGED_ALL_FAMILIES`
is permitted only if canonical ancestry, the production tag, and post-merge public-driver execution
are all verified") and this repository's own standing operating rule (a task description
authorizing a merge is not, by itself, user authorization to push to the real GitHub remote or
fast-forward the shared `production/fullA-exact` branch — that requires an explicit, separate
confirmation regardless of how thoroughly the technical gates passed): **this session commits the
work to the local port branch and stops short of the actual push/fast-forward/tag, pending explicit
user confirmation to proceed** — not because the evidence is weak (it is, for unrestricted and
flexible CM specifically, very strong and materially stronger than the prior session's), but
because (a) two families' outer-throughput claims remain unverified and (b) pushing to a shared
remote is a hard-to-reverse, cross-session-visible action this repository's own history has
explicitly flagged as requiring confirmation, not inference from task text.

**Verdict: `PORT_READY_NOT_MERGED`** — with the explicit, load-bearing distinction from the PRIOR
session's identical verdict that this time the "port ready" claim is materially stronger: a real,
previously-shipped correctness bug (flexible CM's backend was never actually active) is now fixed
and verified, all four families pass complete D=20 correctness gates, all four families' public
drivers correctly report the shared backend, and two of four families have full real-outer-loop
throughput confirmation (the other two have full correctness confirmation but not yet outer-loop
throughput confirmation).

## 9-11. CM bin-table cost decomposition (task §9, non-blocking)

`CM_CROSS_BLOCK_OPERATOR_AUDIT_2026-07-25.md` (updated this session) —
**`Stab` (not `Ttab`) is 80.7% of `build_bin_tables!` and 68.2% of the ENTIRE CM Hessian callback**
(measured this session, `bench_cm_bintable_decomposition.jl`) — a MATERIAL, real correction to the
prior session's speculative "small and unquantified" hedge. `CM_STAB_SHARE_OF_HESSIAN = 68.2%`.
Per the task's own 10% materiality threshold, `CM_CROSS_BLOCK_FOLLOWUP = justified_future_task`
(NOT implemented this session — not trivial, needs its own design/validation cycle — but no longer
dismissible as `not_justified`).

## Final verdicts

```
SHARED_WINNER_PAIR_H_EE = PORT_READY_NOT_MERGED

UNRESTRICTED_H_EE = exact_winner_pair_parallel (workers=10, storage=:full_stride)
FLEXIBLE_CM_H_EE  = exact_winner_pair_parallel (workers=10, storage=:full_stride) -- NOW GENUINELY ACTIVE (bug fixed this session)
CM_MEANZC_H_EE    = exact_winner_pair_parallel (workers=10, storage=:full_stride)
ORIGIN_ZC_H_EE    = exact_winner_pair_parallel (workers=10, storage=:full_stride)

WINNER_PAIR_WORKERS = 10
DENSE_FALLBACK_CALLS_IN_GATES = 0
POST_MERGE_SMOKE = not_applicable (not merged this session)

CM_STAB_SHARE_OF_HESSIAN = 68.2%
CM_CROSS_BLOCK_FOLLOWUP = justified_future_task
```

## What a follow-up session needs to do before `MERGED_ALL_FAMILIES` is honestly claimable

1. Run CM+mean/ZC's and origin-ZC's own matched outer A/B (≥120s each, same methodology as
   `WINNER_PAIR_MATCHED_OUTER_AB_2026-07-25.md` — no existing harness script for these two families
   was found this session; one would need to be written, closely modeled on
   `matched_outer_benchmark_cm_2026-07-25.jl`).
2. Complete flexible CM's checkpoint/resume gate (fix the remaining include-chain issue in
   `test_shared_hessian_checkpoint_resume.jl`, or write a fresh script closely modeled on the
   proven `test_cm_checkpoint_resume.jl`).
3. If both pass: fast-forward `production/fullA-exact` to this branch's tip, verify ancestry with
   `git merge-base --is-ancestor <release_commit> production/fullA-exact`, tag
   `shared-winner-pair-core-hessian-production-ready-2026-07-25`, run post-merge public-driver
   smokes for all four families, capture final startup manifests, confirm clean status — **with
   explicit user confirmation before the actual `git push`**, per this repository's own standing
   operating rule.
