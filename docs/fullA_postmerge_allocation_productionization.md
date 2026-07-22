# Full-A_od allocation/cache-cleanup productionization

Branch: `audit/fullA-postmerge-correctness` (correctness-hardening work, all 14 independent-audit
findings resolved — see `docs/fullA_independent_audit_remediation.md`), with
`perf/fullA-allocation-cache-cleanup` merged in on top.

**Status: in progress.** This document is being filled in incrementally. Sections marked TODO
are not yet complete — do not treat this as a final report.

## 1. Active-call-path map (source present vs actually wired)

Corrected premise: the brief states the allocation/cache-cleanup branch "has now been merged
into the main production branch." This was checked directly with git and found to be **false** at
session start: `integration/fullA-final-production-merge` @ `2620097` was an *ancestor* of
`perf/fullA-allocation-cache-cleanup` @ `1279ed6` (the perf branch forked forward from
production and was never merged back) — the reverse of "merged in." None of the four target
symbols existed anywhere on the production branch. User-confirmed: merged
`perf/fullA-allocation-cache-cleanup` into this branch as a prerequisite (clean merge, 9 purely
additive files, zero conflicts with the correctness fixes).

| Feature | Source present | Called by production driver | Persistent across callbacks/stages | Tests |
|---|---:|---:|---:|---|
| `composite_gradient_at_fast_pooled` / `GradWorkspacePool` | Yes (`gradient_workspace.jl`, now merged) | **No** — `c10_d20_production_driver.jl` still calls `composite_gradient_at_fast_buffered` in both `cb_G!`s | N/A (not constructed anywhere in the driver) | Only `test_gradient_workspace.jl` (own file); production-driver wiring test does not exist yet |
| `CrossDeltaExactCache` | Yes (`cross_delta_cache.jl`, now merged) | **No** — `staged_delta5.jl`/`run_polish_checkpointed` still pass `exact_cache::SafeExactCache` | N/A | Only `test_cross_delta_cache.jl` (own file) |
| `run_cm_upper_checkpointed` | Yes (`cm_checkpoint.jl`, now merged) | **No** — no caller anywhere references it outside its own test files | N/A | `test_cm_checkpoint_original.jl`, `test_cm_checkpoint_resume.jl` |

**Bugs found in the newly-merged code before any wiring was attempted** (both are exactly the
class of defect the independent audit's AUD-04/AUD-08 findings describe — found by applying the
same scrutiny, not assumed absent because the code is new):

1. **`CrossDeltaExactCache`'s `FullAInnerKey` did not carry a context fingerprint.** It stripped
   `FullAEvalKey` down to `(x_free, find_smallest, inner_loop_opt, mode)` for the δ-independence
   design (correct and well-reasoned on its own terms — see the file's own header derivation),
   but doing so also silently dropped the `ctx_fingerprint` field this session's AUD-08 fix had
   just added to `FullAEvalKey` — reintroducing the exact cross-context aliasing risk AUD-08 was
   written to close. **Fixed**: `FullAInnerKey` now also carries `ctx_fingerprint`.
2. **`run_cm_upper_checkpointed`'s incumbent/checkpoint gate has no AUD-04-equivalent residual
   check.** `is_new_best` in `cm_checkpoint.jl` only checks `isfinite(Δ) && Δ <= delta + 1e-6` —
   status-level feasibility is implicitly enforced (an infeasible inner solve throws in `cb_F!`'s
   try/catch before `is_new_best` is ever evaluated), but there is no equivalent of
   `is_verified_success` (finite primal-dual gap, weighted moment/KKT residual, normalization,
   conjugate-domain check). The final `:stage_complete` checkpoint is also written
   unconditionally, the same AUD-10 pattern already fixed in `c10_d20_production_driver.jl`.
   **Not yet fixed** — `cm_production_value`/`archC_base_state` return a bare `BaseDualState`
   (`x_free0, θ_full0, ζstar, λstar, m_star, inner_status`), which does not carry the richer
   diagnostic fields (`primal_dual_gap`, `weight_norm_resid`, `mean_m_resid`,
   `max_abs_moment_kkt_resid`, `m_min`) `classify_inner_result`/`is_verified_success` need.
   Building those out for the Architecture-C CM path is new diagnostic work, not a simple
   wire-up, and was not attempted in this pass to avoid introducing an unvalidated change to an
   already-working gradient/value path under time pressure. **Recommendation: do not route
   production CM campaigns through `run_cm_upper_checkpointed` until this gate exists** — per the
   brief's own standard ("After this passes, route production CM campaigns through the
   checkpointed wrapper"), it has not yet passed.

## 2. Remaining work (not yet done)

- Wire `GradWorkspacePool` into the production gradient driver behind a reference flag (old
  buffered path retained); matched exact-equality tests at D=4/D=20/nearby/difficult points; A/B
  benchmark (bytes, GC, wall, RSS, serial + 20-thread) against the ~3.7-4.5 GB -> 756.5 MB prior
  session number.
- Persistent `LFixBaseWorkspace` for `build_lfix_base_cache`'s ~590-650 MB/gradient (price0/pTσ0
  tensors) — not started.
- Wire `CrossDeltaExactCache` into `staged_delta5.jl`'s continuation (now that its context-
  fingerprint gap is fixed) and benchmark stage-transition hit rates / end-to-end wall savings.
- Build the AUD-04-equivalent verified-success gate for the CM Architecture-C path before
  recommending `run_cm_upper_checkpointed` as production, then run the interrupted/resume
  campaign the brief specifies (D=20, W=80,000, L=50, delta=1).
- Fresh line-level `Profile.Allocs` audit, two-tensor representation experiment (A-D), dense
  post-solve materialization audit, cross-δ continuation benchmark, final matched benchmarks —
  all not started.

## 3. What this document does NOT yet claim

No allocation, GC, wall-time, or peak-RSS numbers have been measured on this branch yet. The
"~3.7-4.5 GB -> 756.5 MB" and "16-29x cache speedup" figures referenced above are from the PRIOR
session's own work (now merged in) — they have not been independently re-verified on this branch
and should not be cited as re-confirmed until they are.
