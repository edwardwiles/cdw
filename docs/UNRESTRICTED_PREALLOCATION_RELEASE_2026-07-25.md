# Unrestricted preallocation release — 2026-07-25

Task §1.1/§3.1–3.3. Status: **MERGED** (ancestor of `production/fullA-exact`, tagged
`unrestricted-preallocation-production-ready-2026-07-25`).

These fixes were implemented and validated on the source feature branch
(`port/production-allocation-and-hessian-optimizations-2026-07-25`), then cherry-picked onto this
release branch unchanged (commits `3d208e6`, `85b2370`, `693b809`) and re-verified against real
licensed KNITRO throughout this session (11/11 public entry-point assertions, the P0/P1/P2 BLAS
sweep, the matched 300s before/after run — all exercised this exact code). This document restates
and re-verifies what the source branch established; it does not re-derive it from scratch.

## What was fixed

- **`CompressedFactualWorkspace`** (`3d208e6`): reusable buffer for `build_compressed_factual!`,
  wired into the *actual* production hot path (`screened_eval`'s dual-bank scoring block) — the
  source branch's own investigation found and corrected a mis-identified call site
  (`build_compressed_factual` via `inner_loop_internal_compressed`, which turned out to be
  unreachable from either real driver; see `feedback-verify-hot-path-not-just-grep-found` in
  project memory). 24.96 MB/call → ~0 at the corrected site.
- **`fast_range_screen.jl` dense K/G copy removal** (`85b2370`): the single largest unrestricted
  allocation site the source audit's own dynamic trace found. 245.56 MB → 304 bytes (100%).
- **`hard_score_B` cache + `canonical_price_precompute` workspace** (`693b809`): 25.6 MB → 0 and
  38.4 MB → 0.02 MB (99.95%) respectively.

## What was NOT fixed (explicitly, not silently)

`constCons_matrix`, `build_winner_ref!`, `mass_at`, `pivot_expand`, `compressed_dual_contraction` —
named by the original audit but smaller and deprioritized. The independent follow-up audit
(`UNRESTRICTED_REMAINING_ALLOCATION_AUDIT_2026-07-25.md`, `audit-unrestricted-allocation-gap-2026-07-25`
worktree, snapshot `2a555fb`) found these collectively account for a modest slice of a real
`run_polish_checkpointed` window's allocation (~9.6% combined: `constCons_matrix` 3.1%, `mass_at`
0.9%, `pivot_expand` 0.4%, `compressed_dual_contraction` 5.1%) — real and correctly attributed,
but small relative to that audit's headline finding.

**That audit's headline finding, carried forward here unchanged**: ~80% of a real
`run_polish_checkpointed` call's total allocation is a **one-time context/screen/gradient-pool
build cost**, paid once per call regardless of outer-iteration count — not a recurring per-eval
hot-path problem at all. The highest-leverage remaining fix is enabling `reuse=` context caching
for `destination_sample=:exclude_row` (currently hard-restricted to `:all_legacy` in
`build_fullA_context`, `reusable_context.jl:39-48`). This is **not implemented in this release** —
see `omit_row_context_reuse = future_task` below.

## Re-verification this session

- 11/11 public entry-point assertions (`test_backend_manifest_unrestricted.jl`) confirm
  `run_polish_checkpointed`/`run_profile_checkpointed` both reach `core_moment_representation=
  compressed`, the representation these fixes target.
- Matched 300s before/after (`MATCHED_300S_BEFORE_AFTER_2026-07-25.md`): **30.262 GB → 13.418 GB
  user allocation (−55.6%)** in a real 300s window at real D=20/W=80,000 scale, with 46 vs 37 outer
  evals completed and cold-verify agreement to 1.22e-15.

## Follow-up (not this release)

```
omit_row_context_reuse = future_task
```

Expected benefit: eliminating the ~7.8 GB/~70-83s one-time setup cost for any campaign that calls
`run_polish_checkpointed` fresh more than once under `:exclude_row` (per-stage continuation,
restart). Risk: correctness — `reuse_matches` already checks `destination_sample` compatibility
generically (`reusable_context.jl:82-88`); the remaining work is verifying
`build_fullA_context` builds a `destination_sample`-aware context correctly for the non-`:all_legacy`
case, not inventing new reuse machinery. Not implemented this release given time budget, per this
session's own explicit instruction not to delay the present release for context reuse.
