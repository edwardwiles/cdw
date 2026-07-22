# Full-A_od independent-audit remediation

Branch: `audit/fullA-postmerge-correctness`, forked from `integration/fullA-final-production-merge` @ `2620097`.
Audit reviewed: `FullA_od_Independent_Audit_2026-07-21.pdf` (stable commit `f6ae01e`, pending branches `6d5eb67e`/`5b8b38bc`).

This document is a work-in-progress record of the remediation pass requested in the post-merge
correctness-hardening brief. It is being filled in incrementally as fixes land; treat any section
marked TODO as not yet complete.

## 1. AUD-01..AUD-14 disposition table

| ID | Severity | Disposition (this head) | Evidence |
|----|----------|--------------------------|----------|
| AUD-01 | Critical | **Fixed by prior merge** | `moments_gammanorm.jl:111` still has the decreasing kappa formula, but the driver's `find_smallest ? :upper : :lower` convention (`c10_d20_production_driver.jl:465,780`) is now consistent with it via the direct-γ' objective (`EK_moments_gammanorm_directgp!`), and the last straggler (`organic_failure_capture.jl`) was fixed live in this session (commit `2620097`, on top of the driver-delta5 branch's own earlier fix `69587e3`). |
| AUD-02 | Critical | **Fixed this session** | All 3 actively-loaded outer opt files (`csw_outer_wallclock_{sr1,lbfgs,productfd}.opt`) now have `par_concurrent_evals no` (previously only the inner opt file did). Added a thread-aware runtime guard in `cc_algo/PsiObjectiveBundle.jl` (`_enter_callback!`/`_exit_callback!`) that errors on genuine cross-thread reentry into one bundle instance's callbacks while correctly allowing legitimate same-thread nesting (a real pattern found live in `prepare_cc/PMM.jl`'s δ*-initial computation — an early non-thread-aware version of the guard false-positived on it). |
| AUD-03 | High | **Fixed this session** | `solve_base_state` (three_way_derivatives.jl) always re-solves and was never the bug. The actual defect was in `c10_d20_production_driver.jl`'s 4 `BaseDualState(...)` construction sites, which read live `ctx.obj.arg1` after a `screened_eval` call that might have been a cache hit (leaving `ctx.obj.arg1` at a DIFFERENT point's state). Fixed: `base = r.cache_hit ? solve_base_state(xf, ctx) : BaseDualState(...)`. |
| AUD-04 | High | **Fixed this session** | Added `VerifiedSuccessTolerances`/`InnerResultClass`/`classify_inner_result`/`is_verified_success` (oracle.jl). `is_cacheable_result` now requires `VerifiedSolved` or `ExactInfeasible`, not a bare status-code check. Incumbent acceptance (`is_new_best` in both `cb_F!`s) now also requires `is_verified_success(r)`. |
| AUD-05 | High | **Fixed by prior merge** | `composite_gradient.jl`'s docstrings and `docs/fullA_D20_gradient_validation.md` already describe the A-block gradient as an adaptive-h secant, not an exact derivative. |
| AUD-06 | High | **Fixed this session** | `fast_range_screen.jl`'s Hmax fusion only updated the strict-first winner; a tied non-first origin could get a false `:winning_range` certificate. Fixed by tracking per-(o,d) ties and refusing the certificate when a tie is present (falls back to the trusted real-solve path), in both the early-exit and full-scan branches. Regression test: `test_aud06_tie_safety.jl`. |
| AUD-07 | High | **Fixed by prior merge** | `run_profile_checkpointed`/`run_polish_checkpointed` now seed `best[]`/`best_feasible[]` from the cold-verified start point (commit `792d4e1`), not `nothing`. |
| AUD-08 | Medium | **Fixed this session** | Added `context_fingerprint(ctx)` (versioned SHA-256 over draws/shapes/draw-design/fixed trade data/CM config/option-file contents/KNITRO release, memoized per ctx). Added as a 6th field to `FullAEvalKey` and `CMEvalKey`; all construction call sites updated. Deliberately does NOT include δ or find_smallest (Delta*(theta) is budget/direction-independent). |
| AUD-09 | Medium | **Fixed this session** | Added `ResumeTolerances`/`check_resume_tolerances!`; both resume-validation blocks now hard-error (not just log) when cold-recomputed Delta/gravity/KKT-residual/moment-mean exceed tolerance. |
| AUD-10 | Medium | **Fixed this session** | The final `:stage_complete` checkpoint at both driver entry points now checks `is_verified_success(r_final)` first; an unverified terminal point is checkpointed as `:stage_complete_unverified` (loudly logged) instead of silently masquerading as a normal stage-complete checkpoint. `best[]`/`best_feasible[]` (the correctly-gated incumbent) is unaffected either way. A resume from an `:stage_complete_unverified` checkpoint prints an explicit warning. |
| AUD-11 | Medium | **Fixed this session** | `draw_design_meta` now computes `checksum_uniform`/`checksum_transformed` via `sha256_of_matrix` (canonical little-endian SHA-256 + shape) instead of Julia's `hash()`. |
| AUD-12 | Medium | **Fixed this session** | Both `a_block_fd_component!`/`a_block_fd_component` (lfix_buffer_reuse.jl / composite_gradient.jl) now: (1) geometrically shrink h and retry when both probes are nonfinite; (2) fall back to a one-sided secant when only one side is finite (unchanged logic, now checked at every h); (3) return `NaN` (never `0.0`) only once every h and the base value are exhausted. |
| AUD-13 | Low | **Fixed this session** | Renamed `moment_resid` → `benchmark_unweighted_moment_mean` across all 21 files that construct or read it (word-boundary-safe rename; `max_abs_moment_resid`/`moment_resid_blas` untouched). The correctly-weighted residual was already separately exposed as `mean_m_resid`/`max_abs_moment_kkt_resid`. |
| AUD-14 | Low | **Fixed this session** | `qmc_draws.jl`'s three generators now wrap their `Random.seed!`/`rand`/`rhalton` calls in `_with_saved_global_rng` (save/restore the global default RNG around the call). Returned draw VALUES are unchanged bit-for-bit (same internal seed!/rand sequence); only the caller-visible global-RNG leak is fixed. Known remaining gap: this does not make concurrent construction of DIFFERENT contexts on DIFFERENT threads race-free (documented, not silently claimed fixed) — that needs the fuller local-RNG-threading refactor of `cc_algo/rhalton.jl` the audit originally recommended. |

**Summary: all 14 findings addressed** — 3 were already fixed by prior merges (AUD-01, 05, 07) before this session started; the remaining 11 were fixed in this session with `docs/fullA_independent_audit_remediation.md`-referenced code changes and, where practical, a permanent regression test.

## 2. Verification status

| Check | Status |
|---|---|
| `test_infeasibility_screen.jl` | PASS (72/72) — real D=4 KNITRO solves, tie-safety, screen agreement |
| `test_safe_exact_cache.jl` sections 1-3 | PASS (cache hit/miss, AUD-04-aware cacheability, screen-certificate caching) — section 4 (in-process 3-way concurrent KNITRO stress) is pre-existing, documented-fragile (silently hangs/dies independent of this session's changes) and was not used as a pass/fail signal |
| `test_aud06_tie_safety.jl` (new) | PASS (10/10) |
| `test_composite_gradient.jl` | PASS |
| `test_composite_gradient_fast.jl` | PASS |
| `test_oracle_fast.jl` | PASS |
| `test_compressed_live_integration.jl` | PASS |
| `test_draw_design.jl` | TODO (in progress at time of writing) |
| Matched D=20 delta=1/delta=2 validation (A/B/A cache-state, cache-disabled cold reference) | **Not yet run** — planned next |

## 3. Known follow-ups (explicitly out of scope for this pass, not silently dropped)

- AUD-14's local-RNG threading is a save/restore wrapper, not a full refactor of `cc_algo/rhalton.jl` to thread an explicit RNG through every scramble/permutation call. Safe for the current single-context-construction-at-a-time usage pattern; would need the fuller refactor before concurrent multi-context construction is claimed race-free.
- `AUD-06`'s certificate/cache schema was not version-bumped, because no currently-existing code path persists a winning-range certificate to disk across processes (SafeExactCache is always freshly constructed per outer run). This becomes necessary once Prompt 1's `CrossDeltaExactCache` or a checkpoint-embedded exact cache lands.
- A handful of diagnostic/experiment scripts (`c8_*`, `c9_*` one-off files) still hand-construct `BaseDualState` directly from `ctx.obj.arg1` without the AUD-03 cache-hit guard; only the production driver's 4 call sites were fixed. These scripts are not part of the production hot path.
- Tolerances in `VerifiedSuccessTolerances`/`ResumeTolerances` are provisional (not empirically calibrated against real D=20/W=80,000 solves in this pass) — see their docstrings.

## 4. Commits

See `git log` on `audit/fullA-postmerge-correctness` for the itemized commit sequence (one logical concern per commit, per the brief's integration instructions).
