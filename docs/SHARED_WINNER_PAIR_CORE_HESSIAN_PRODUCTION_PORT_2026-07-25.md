# Shared winner-pair core-Hessian production port — master summary — 2026-07-25

## Goal

One shared exact core-Hessian backend computing the common economic/trade-share (H_EE) block for
every current production restriction family (unrestricted, flexible CM, CM+mean/ZC, origin-specific
ZC) — not merely enabling winner-pair for unrestricted.

## Release state reached

**`WIRED_IN_ALL_PUBLIC_FAMILIES` → `VALIDATED_THROUGH_PUBLIC_ENTRY_POINTS` (D=4, all four families;
D=20 real-data for unrestricted directly and for flexible CM incidentally via a pre-existing
regression test). NOT `FAST_FORWARDED_TO_CANONICAL_PRODUCTION` — see verdicts below for why.**

## 0. Provenance

- Canonical production tip at session start: `origin/production/fullA-exact` (remote `cdw`,
  `git@github.com:edwardwiles/cdw.git`) @ `39b89c5` — 2 commits past the
  `allocation-hessian-production-release-2026-07-25` tag (which the handoff cited as the merged
  allocation/Hessian release commit `6f1d2c5`; production had genuinely advanced, exactly as the
  task anticipated).
- Validated-but-unmerged winner-pair branch: `diag/compressed-hessian-operator-audit-2026-07-25`
  @ `3fceb71`, base `b7435ee` — confirmed via `git merge-base` to be EXACTLY the stated base,
  and via `git diff b7435ee..HEAD -- compressed_moments.jl` to be byte-identical to current
  production's `CompressedFactual` definition (no adaptation needed for the port).
- Port branch: `port/shared-winner-pair-core-hessian-production-2026-07-25`, created as a git
  worktree off `origin/production/fullA-exact` @ `39b89c5` (0 commits ahead/behind at branch
  creation — a clean fork point, not a stale one).

## 1. What was built

`full_aod_diag/d4_exact/core_exact_hessian.jl` (new, ~640 lines): ports the diag branch's serial
correctness-oracle kernel (`winner_pair_hessian!`) and its validated destination-pair-owned
parallel kernel (`hessian_core_winner_pair!`, `port_ready_10_workers`) essentially verbatim —
preserving, exactly as the task required: the REAL sampling weights `ν=SW` (not an assumed
all-ones vector — verified against `compressed_moments.jl`'s own formula both by the diag branch
and independently re-confirmed by this session's research); the live winner contribution `x_{ωd}`;
the ζ dual column; the optional counterfactual price-index column; the low-rank target
corrections; the hybrid-divergence curvature weights (consumed via `obj.ddPsi!`, never
re-derived); KNITRO's packed row-major upper-triangle convention; rectangular omit-ROW indexing
including the non-last-omitted-destination case the diag branch specifically stress-tested. The
draw-partitioned `_threaded`/`_threaded_nomacro` variants (unresolved correctness bug once
`nthreads()>1`, per the diag branch's own docs) were deliberately NOT ported.

Added on top (the one genuine gap the diag branch's own gap-analysis docs flagged as unbuilt):
`CoreExactHessianWorkspace` + `fill_core_hessian_upper!`, the ONE shared entry point every family
now calls — see `CORE_HESSIAN_PACKED_ASSEMBLY_VALIDATION_2026-07-25.md` for the design (dense-view
insertion into each family's own pre-existing dense scratch, rather than a hand-rolled
local-to-global packed-index table) and its validation.

## 2. Where it's wired (see `CORE_HESSIAN_FAMILY_COVERAGE_TABLE_2026-07-25.md` for full detail)

| Family | Resolved H_EE backend |
|---|---|
| Unrestricted | `exact_winner_pair_parallel` (workers=10, `:full_stride`), called directly on the packed KNITRO output — no dense round-trip |
| Flexible CM | `exact_winner_pair_parallel`, via shared `_fill_cm_HEE!` (one helper, both serial and production-default-threaded Architecture-C callers) |
| CM+mean/ZC | Same shared backend, reused verbatim (task §4.3's explicit "do not implement a second winner-pair variant") — `_fill_cm_HEE!` additionally partitions this family's WIDER "economic" block into the true core sub-block (winner-pair) plus core×mean/pair and mean/pair×mean/pair corners (dense BLAS, unchanged) |
| Origin-ZC | Partitioned via new `archA_partitioned_hess_cb_builder`: H_EE (winner-pair), H_ER (dense, computed once), H_RR (dense) — H_RE never independently computed |

A `CompressedFactual` is now built for CM/CM+meanZC/origin-ZC's core columns too (previously only
unrestricted did this) — each family's own `moments!` closure builds it once per outer point and
publishes it into a shared `core_cf_ref::Ref{Any}` box (the SAME pattern this codebase already
uses to pass FG-computed state to a Hessian callback, e.g. `compressed_live.jl`'s
`obj.arg0 .= q`), which the Hessian callback reads and rebuilds its `CoreExactHessianWorkspace`
from only when the `cf` object identity changes (a new outer point) — not on every Hessian call.
Every family retains a `:dense_reference` fallback (task §5's "acceptable to preserve dense BLAS
as a named fallback") for whenever no compressed core is available for a point (a caught
`TiedWinnerError`) or an explicit anti-regression comparison.

## 3. Correctness gates run (task §7)

`full_aod_diag/d4_exact/test_shared_core_hessian_d4_gates.jl` — D=4, all four families, 40/40
PASS (`docs/full_correctness_log_2026-07-25.txt`):
- Unrestricted: serial + parallel(workers=2, `:full_stride`) + parallel(workers=4,
  `:direct_packed`) vs `CS.hessian!`, at 6 dual points (zero, 4 random, 1 real KNITRO-solved).
- Flexible CM: H_EE-block and full-assembled-Hessian agreement, both serial and production-default
  threaded-bins Architecture C, plus real inner-solve dual-point agreement.
- CM+mean/ZC: K_mean=1/K_pair∈{0,1}, true-core-sub-block and full-Hessian agreement.
- Origin-ZC: K_mean=1/K_pair∈{0,1}, H_EE-block and full-Hessian agreement (confirming the
  H_EE+H_ER+H_RR partition reconstructs the original monolithic dense contraction exactly).

Two real bugs were found and fixed BY these gates (both disclosed, neither hidden):
1. A docstring-placement Julia parse error (cosmetic — a doc-comment landed between an existing
   docstring and its function during an `Edit`), caught by the FIRST gate run failing to even
   load.
2. An off-by-one in this session's OWN `_dense_reference_core_hessian!` test-comparison helper
   (manually faked a ζ ones-column instead of reading `H`'s own column 2, shifting every real core
   column by one) — caught because Section A (unrestricted) showed ~100% relative error while
   Sections B/C/D (which reuse the identical winner-pair kernel via a DIFFERENT code path) passed
   at machine precision, isolating the bug to the test helper, not the production kernel/wiring.
   Confirmed NOT a production-path bug: `_callbackEvalH_inner_compressed!`'s own `:dense_reference`
   branch calls the real, untouched `CS.hessian!`, never this helper.

Additionally, the pre-existing (unmodified) `test_cm_compressed_core.jl` regression test — written
before this port, for a different purpose (compressed-core moments, not Hessian backend) — now
incidentally exercises the winner-pair H_EE backend as its default and passed all 7 of its own
sections at real D=20/W=80,000/L=50 scale (`docs/cm_d20_regression_log_2026-07-25.txt`),
independent confirmation at production scale.

D=20/W=80,000 real-data smoke+timing check for unrestricted (task §6, scoped — see
`WINNER_PAIR_POST_REBASE_THREAD_SELECTION_2026-07-25.md`): complete inner solve, P0 calibration
point, through the real production entry point — **9.96x speedup** (workers=10) over the
byte-identical `:dense_reference` path, identical `nStatus`/`n_fg`/`n_hess`/`objSol`/dual solution.

## 4. What was genuinely NOT completed this session (disclosed, not fabricated)

- The full task-brief-specified matched real outer A/B campaign (300s/120s per family, P0/P1/P2,
  cold-verified kappa, verified progress/minute) — see `SHARED_H_EE_MATCHED_OUTER_AB_2026-07-25.md`.
- The full D=20 correctness-gate matrix across P0/P1/hard-point for every family (only unrestricted
  P0 was directly re-run at D=20 this session; CM got incidental D=20 coverage via the pre-existing
  regression test; CM+meanZC and origin-ZC were only validated at D=4 this session).
- P1/P2 worker-count sweeps, and any per-family (non-unrestricted) worker-count sweep.
- Running the pre-existing `test_backend_manifest_unrestricted.jl`/`test_backend_manifest_cm_
  originzc.jl` public-entry-point assertion tests against the newly-extended manifest fields (see
  `CORE_HESSIAN_PUBLIC_ENTRY_POINT_ASSERTIONS_2026-07-25.md`).
- The CM winner-bin cross operator candidate (Phase B, §10-11) — audited and NOT implemented,
  decision documented in `CM_CROSS_BLOCK_OPERATOR_AUDIT_2026-07-25.md` (current H_EC is already
  0.2% of Hessian callback time; the real opportunity is inside `build_bin_tables!`'s shared
  Ttab+Stab accumulation pass, not isolated/measured finely enough this session to justify
  implementing).
- Production merge/tag/post-merge-smoke (see verdicts below).

## 5. Git state

- Branch: `port/shared-winner-pair-core-hessian-production-2026-07-25`.
- Base: `origin/production/fullA-exact` @ `39b89c5` (exact fork point, 0 ahead/behind at creation).
- Diff: 8 files changed, 420 insertions, 62 deletions (`core_exact_hessian.jl` +
  `test_shared_core_hessian_d4_gates.jl` + `bench_shared_core_hessian_unrestricted_d20.jl` new;
  `cm_hessian_architectures.jl`/`cm_hessian_threaded.jl`/`cm_meanzc_moments.jl`/
  `cm_meanzc_production.jl`/`cm_originzc_moments.jl`/`cm_originzc_production.jl`/
  `compressed_live.jl`/`production_backend_manifest.jl` modified — no unrelated files touched).
- Not pushed to any remote, not merged, not fast-forwarded into `production/fullA-exact` this
  session (see repo convention: production pushes/merges require explicit user confirmation, not
  inferred from a task brief's instructions alone).

## 6. Final verdicts

```
SHARED_WINNER_PAIR_H_EE = port_ready_not_merged
    -- correctness validated (D=4 all 4 families, D=20 unrestricted + incidental D=20 CM);
       full matched-outer-A/B campaign (task §8) and complete D=20 gate matrix (task §7) not run
       this session -- both genuinely required before a merged_all_families verdict, per the
       task's own "Production eligibility requires" list.

UNRESTRICTED_H_EE = exact_winner_pair_parallel (workers=10, storage=:full_stride)
FLEXIBLE_CM_H_EE  = exact_winner_pair_parallel (workers=10, storage=:full_stride)
CM_MEANZC_H_EE    = exact_winner_pair_parallel (workers=10, storage=:full_stride)
ORIGIN_ZC_H_EE    = exact_winner_pair_parallel (workers=10, storage=:full_stride)

CM_H_EC = retained_bin_prefix
ORIGIN_ZC_H_ER = retained_dense

POST_MERGE_SMOKE = not_applicable (not merged this session)
```

## 7. Recommended next steps for a follow-up session

1. Run the deferred matched-outer-A/B campaign (§4) — the actual production-eligibility gate.
2. Extend D=20 correctness gates to CM+mean/ZC and origin-ZC (D=4 only this session).
3. Run the pre-existing public-entry-point manifest assertion tests against the new fields.
4. Only then: fast-forward `production/fullA-exact`, tag
   `shared-winner-pair-core-hessian-production-ready-2026-07-25`, run post-merge smokes.
