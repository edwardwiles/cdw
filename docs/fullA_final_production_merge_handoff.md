# Final production merge: CM/parallel + driver/δ5 consolidation

**Date:** 2026-07-21
**Verdict: `PRODUCTION MERGE COMPLETE — READY FOR RUNS`**

## 1. Branch / commit map

| Role | Branch | Worktree | Tip before this task | Status |
|---|---|---|---|---|
| Maintained production | `diag/fullA-d4-exact` | `gravity-fullA-d4` | `08b07d0` | **now at `3855430`** |
| CM/parallel stream | `integration/fullA-cm-parallel-production` | `gravity-fullA-cm-parallel-production` | `08b07d0` (== production tip already) | subsumed, archival |
| Driver/δ5 stream | `diag/fullA-driver-delta5` | `gravity-fullA-driver-delta5` | `c3e2b72` | subsumed (by content, see §3), archival |
| Negative-cache audit | `integration/fullA-negative-cache-audit` | `gravity-fullA-negative-cache-audit` | `81f5afb` | already merged into `08b07d0` prior to this task |
| This task's integration branch | `integration/fullA-final-production-merge` | `gravity-fullA-final-production-merge` | branched from `08b07d0` | merged into production at `3855430`, kept for reference |

**Rollback tags:** `pre-consolidation-2026-07-21` (`98983bd`, predates CM-parallel/negative-cache work) and `pre-final-merge-2026-07-21` (`08b07d0`, the state immediately before this task's port). To roll back this merge: `git reset --hard pre-final-merge-2026-07-21` on `diag/fullA-d4-exact` (only if no one has built on `3855430` yet — prefer `git revert -m 1 3855430` otherwise).

**Merge commit:** `3855430` on `diag/fullA-d4-exact` (25-file, clean, no-conflict `--no-ff` merge of `integration/fullA-final-production-merge`).

## 2. Key finding before any work started: stream 1 was already merged

`diag/fullA-d4-exact` and `integration/fullA-cm-parallel-production` were **the same commit** (`08b07d0`) at task start — a merge made earlier the same day with parents `bfcf54b` (CM/parallel tip) and `81f5afb` (negative-cache-audit tip). The negative-cache work is real (`negative_cache.jl`, `docs/fullA_negative_cache_audit.md`) but opt-in/default-off (`use_neg_cache=false`) with its own audit evidence (10/10 LP certificates, 0/30 production false positives). Per explicit user decision, this was accepted as already-reviewed; the integration branch was built from `08b07d0` as-is rather than rolling it back. **No re-merge of stream 1 was needed** — the actual remaining work was porting stream 2 (driver/δ5) onto this base.

## 3. Conflict-resolution log (driver/δ5 port)

Ported via `git cherry-pick f6ae01e..c3e2b72` (7 commits) rather than a branch merge, since `diag/fullA-driver-delta5` diverged from `08b07d0` 49 commits back and a merge would have needed the same file-level resolution anyway. Only one file was touched by both streams: `c10_d20_production_driver.jl` (checkpoint schema / KNITRO-version-check / negative-cache vs. incumbent-seeding / direction-box / diagnostics / reusable-context / dual-bank-AB code). Every conflict was of the same shape — **both sides independently added something, nothing needed to be chosen over the other** — resolved by keeping both:

- `include(...)` lines for `negative_cache.jl` + `incumbent_logic.jl` (2×), then + `dual_bank_ab_harness.jl`
- cold-retry `exact_cache=` bugfix (HEAD) + `organic_failures` capture hook (branch) in `cb_F!`
- `use_neg_cache`/`neg_cache_code_version` kwargs (HEAD) + `reuse`/`organic_failures` kwargs (branch) on `run_polish_checkpointed`
- print-statement diagnostics: native KNITRO status fields (branch) + neg-cache confirmation counts (HEAD)
- `neg_cache`/`ab_stats` kwargs on `screened_eval`'s signature + the negative-cache short-circuit block

The **schema-3 `D20Checkpoint` struct itself never conflicted** — it was untouched by the driver branch's own commits (confirmed: `git log f6ae01e..08b07d0 -- <driver-branch files>` shows zero overlap outside the one shared driver file), so "preserve the newer CM/parallel schema" required no explicit action beyond the include/kwarg merges above.

All 7 ported commits, in order: `792d4e1` (incumbent-seeding fix), `1b2cf82` (KNITRO status decoder + diagnostics), `5c0d51a` (reusable context + organic capture), `52bd0fa` (dual-bank A/B), `1f8a498` (handoff doc), `45f3baf` (direction/box audit fix), `6e5b0f6` (Section 17.4 real-data results).

## 4. Direction / gamma-box derivation and verification

Verified directly from the merged production code, not just the brief's claim: `κ = 1 - gp^(σ/(σ-1))` (`c10_d20_production_driver.jl:862` et al.), σ=2.5 in the real D=20 context ⇒ exponent 5/3 > 1 ⇒ κ strictly decreasing in `gp`. `direction_gamma_bounds(ctx, find_smallest)` (`direction_bounds.jl:90-93`) constructs `find_smallest=true` ("upper") → `[γp_lo, γ_f^F]`, `find_smallest=false` ("lower") → `[γ_f^F, γp_hi]`, sourced from the pre-existing `theoretical_gammaprime_bounds`. The rejected `lambda_ff^(1/(σ-1))` formula is not implemented anywhere (only quoted-and-rejected in comments).

**Real-data verification table** (W=80000, δ=2.0, draw_seed=20260719, from `gp_kappa_table.jl`):

| gp | κ | relationship |
|---|---|---|
| 0.930712 (=γp_lo) | 0.112792 | — |
| 0.944974 | 0.090017 | decreasing |
| 0.973499 | 0.043776 | decreasing |
| 0.987762 (=γ_f^F, Frechet) | 0.020314 | decreasing |
| 0.990821 | 0.015251 | decreasing |
| 0.996940 | 0.005094 | decreasing |
| 1.000000 (=γp_hi) | 0.000000 | decreasing |

Known real Continuation-8 candidates: upper (gp=0.89264, κ=0.17246) lies below γ_f^F ✓; lower (gp=0.99674, κ=0.00439) lies above γ_f^F ✓. `validate_gp_in_direction_box` hard-errors (no silent clamp) on any out-of-box point — fired correctly (and expectedly) three separate times during this session's own script-writing mistakes (bad synthetic start points), confirming the guard actually works end to end.

**Checkpoint `direction` labels**: `find_smallest ? :upper : :lower` confirmed correct at both `D20Checkpoint`-construction sites in `c10_d20_production_driver.jl`. **One real leftover bug found and fixed**: `organic_failure_capture.jl` (added by commit `af9be85`, which predates the driver branch's own direction-fix commit `69587e3`) still had the **old, inverted** `find_smallest ? :lower : :upper` in both its JLD2 record and its JSON summary writer. Caught live: the organic-failure fire-test's first captured record showed `"direction": "lower"` for a `find_smallest=true` (real upper) run. Fixed in both spots, committed (`2620097`), and the rest of the codebase was swept for the same pattern — no other instances found (all other `direction=` sites either correctly derive `find_smallest = direction == "upper"` from an already-correct input, or are documentation/comments citing the historical bug).

## 5. Full test results — 193/193 pass

| File | Result | Notes |
|---|---|---|
| `test_incumbent_seeding.jl` | 18/18 | pre-existing, unchanged |
| `test_direction_bounds.jl` | 26/26 | pre-existing, unchanged |
| `test_knitro_status.jl` | 77/77 | pre-existing, unchanged |
| `test_per_solve_counters.jl` | 12/12 | pre-existing, unchanged |
| `test_parallelism_guards.jl` | 12/12 | **backfilled this session** — no prior coverage of the inner-solve/coord-pool mutual-exclusion invariant |
| `test_checkpoint_schema.jl` | 38/38 | **backfilled this session** — schema-3 round-trip, `knitro_version` field, schema-mismatch fail-fast, `guard_checkpoint_path` |
| `test_cm_interval_equivalence.jl` | 13/13 | **backfilled this session** — wraps the existing `c13_validate_interval_native_archC.jl` script in a real `@testset` (worst diff 8.4e-16) |
| `test_lfix_buffer_fix.jl` | 7/7 | **backfilled this session** — formalizes `c14_verify_lfix_buffer_fix.jl`; fixed a stale docstring citation in `lfix_incremental.jl` pointing at a testset that never existed |

All 8 files re-run and reconfirmed passing **directly from the production worktree** (`gravity-fullA-d4`) after the merge, at commit `3855430`.

## 6. Hard CM threaded-Hessian benchmark (revalidation)

Rerun `c14_cm_hessian_benchmark.jl` post-merge (`-t 10`, using the fixture already committed at `08b07d0`) and compared against the pre-merge committed `results/fullA_d4/c14_parallel_prod/cm_hessian_benchmark.csv`:

| Point | n_hess | REF wall | D_blas10 wall | Speedup | max｜Δ_dual − REF｜ |
|---|---|---|---|---|---|
| hard_cm_point | 12 | 44.4s | 9.63s | **4.615x** | 4.0e-15 |
| near_infeasible_cm_point | 14 | 49.2s | 12.0s | **4.098x** | 7.1e-15 |
| calibration | 11 | 38.1s | 8.9s | **4.257x** | 2.4e-15 |
| cm_trajectory_stage_L50 | 10 | 33.7s | 8.7s | **3.857x** | 3.3e-16 |

All 4 points in the required 7-14 Hessian-callback range; speedups **3.86x–4.62x**, matching/slightly exceeding the pre-merge baseline (3.45x–4.6x); correctness agreement at machine precision throughout; no races or nondeterminism observed. `Delta_dual` at the two CM points also matches the pre-merge committed CSV to ~11 significant figures (1.6984210658660936→1.698421065866; 6.722758555627821→6.722758555628), serving simultaneously as fixed-point regression evidence (§8).

## 7. Allocation-fix revalidation

`build_lfix_base_cache` (via `price_and_pTsigma_cell!`), real D=20/W=80000/δ=1.0:

| Point | Allocation | vs. ~1078MB baseline |
|---|---|---|
| calibration | 650.3 MB | −40% |
| nearby_perturbed | 589.6 MB | −45% |

Bit-for-bit identical to the original allocating `price_and_pTsigma_cell` at both points (`price_ref == price_new`, `pTσ_ref == pTσ_new`, `max_abs_diff=0.0`). `validate_dense=true`'s independent dense self-check passed at both points.

## 8. Staged canonical-checkpoint validation (flagship live run)

The original canonical δ=2 checkpoint (`d2_startA_canon_stage_complete_neval158.jls`) was written under **schema 1** (23 fields, no `draw_design`/checksums/`knitro_version`) and could not be loaded via the current `load_checkpoint` (schema-3 fail-fast). Decoded by defining a matching schema-1 struct locally (bypassing the merged branch's schema-3 type) and extracting `g`/`zfree` directly — this is itself a nice concrete confirmation of why the fix matters: the decoded record showed `branch=:lower` for `find_smallest=true`, i.e. it was mislabeled by the very bug this merge fixes.

Ran `run_staged_delta5_continuation("canon_upper", g=0.9505276120136134, zfree, find_smallest=true, delta_stages=[2,3,4,5], stage_maxtime_real=90s, W=80000, draw_seed=20260719, reuse_context=true)`:

| Stage | δ | wall | κ | best_gp | n_rejected | knitro_status |
|---|---|---|---|---|---|---|
| 1 | 2.0 | 114.7s | 0.080683 | 0.950778 | 0 | −411 (TIME_LIMIT_INFEAS) |
| 2 | 3.0 | 119.1s | 0.082907 | 0.949397 | 1 | −401 (TIME_LIMIT_FEAS) |
| 3 | 4.0 | 110.6s | 0.083168 | 0.949235 | 2 | −401 |
| 4 | 5.0 | 117.0s | 0.083168 | 0.949235 | 3 | −401 |

**Invariants hold exactly**: best gp non-increasing across stages (0.950778 → 0.949397 → 0.949235 → 0.949235) and κ non-decreasing (0.080683 → 0.082907 → 0.083168 → 0.083168, flat at stage 4 where no improvement was found — expected, not a failure). This is a **genuine outer-search improvement** over the seed (unlike the driver branch's own earlier validation, which only reached a flat/no-improvement result under adverse conditions) — the strongest available evidence that the incumbent-seeding and direction-box fixes work correctly together on a real, previously-buggy checkpoint.

**Context-reuse equivalence**: a `reuse_context=false` stage-1 arm gave **bit-identical** κ (0.0806831259203773 both arms). Wall-time comparison in this run showed only 1.9% savings (both arms ran in the *same* process, so the "rebuild" arm benefited from JIT/disk-cache warmth left over from the reuse arm immediately before it — a real confound, not a regression; the previously-established ~10-12% savings from a genuinely separate-process A/B in the driver branch's own handoff remains the more trustworthy figure). **Documented honestly as a measurement limitation, not re-litigated with more compute given time budget.**

## 9. Organic `-300` fire-test

A short direct δ=5 run (`find_smallest=true`, started from the staged run's own δ=5 terminal point) produced **3 real organic `-300` (KN_RC_UNBOUNDED) captures**, each with a full JLD2 record (exact outer vector, reconstructed A, draw/config hashes, warm state, status diagnostics) and a JSON summary. This is where the direction-label bug (§4) was actually caught — a genuine, unplanned payoff of live-firing this feature rather than only structurally reviewing it. Capture infrastructure confirmed opt-in (`organic_failures=nothing` by default, zero behavior change) and working end to end.

## 10. Successful-dual-bank A/B

Ran `dual_bank_ab_trajectory` on a real 8-point trajectory reconstructed from the staged run's own checkpoints (δ=2 stage). **Correctness-neutral**: identical inner-solve statuses at every point in both arms (8/8 feasible), zero harmful/beneficial points. **Wall-time**: bank ON was *slower* on this particular short/already-warm 8-point trajectory (−14.5%, scoring overhead ~1.6s total across 8 calls, `mean_n_candidates=4.5`). Reported honestly per the brief's own framing ("useful validation but not a blocker") — this is diagnostic instrumentation, not a production-path change; `use_dual_bank=true` remains the existing production default, unaffected by this result.

## 11. Fixed-point numerical regression — 5 points

| Point | Result |
|---|---|
| Unrestricted calibration | Delta_dual=0.230884149003, gravity≈6.4e-18, KKT-resid≈1.7e-12 — finite, consistent |
| Latest unrestricted candidate (staged run's own δ=2 terminal point, re-evaluated at δ=1) | Delta_dual=2.011767222564, gravity≈6.0e-18 — finite, consistent |
| Nearby unrestricted perturbation | Delta_dual=0.230052889037, gravity≈6.8e-18 — finite, consistent |
| Hard CM L=50 candidate | Delta_dual matches pre-merge CSV to ~11 sig. figs (§6) |
| Near-infeasible CM point | Delta_dual matches pre-merge CSV to ~11 sig. figs (§6) |

Exact-cache hit/miss confirmed correct (first eval populates, second identical-point eval hits with `Delta_dual` match). Full gradient at calibration: length 400, all finite, norm 97.68.

## 12. Fresh-process smoke tests

**Unrestricted**: short δ=1 upper continuation from `gp0` (exactly the Frechet boundary, valid for either direction) — 45s budget, real KNITRO run, found a feasible best (Delta_dual=0.00242). Checkpoint save → resume round trip: schema=3, `knitro_version="Knitro 13.0.1"` populated correctly, resume continued from `n_eval=2`, direct re-evaluation at the checkpointed point matched the checkpoint's own recorded value to ~15 significant figures (tiny cold-vs-warm KNITRO-path difference, within floating-point-reduction tolerance).

**CM (L=50)**: reused the existing `c14_smoke_cm_l50.jl` (short `run_cm_upper` continuation, 180s budget) — real D=20 context built (68.0s), continuation ran (κ=0.036, status=−401 as expected for a short budget), checkpoint round trip (its own simpler NamedTuple scheme, per the disclosed gap in §13) confirmed OK, KNITRO-version invariant confirmed.

**Config printed**: KNITRO 13.0.1 (fail-fast verified), Julia via juliaup (1.12.6 — see note below), BLAS threads configurable via `BLAS.set_num_threads`, CM `threaded_bins=true` + `syrk!` + BLAS10 as the validated production recommendation, draw design `:pseudorandom` seed `20260719`, upper/lower gamma boxes as in §4.

**Toolchain note (non-blocking but worth flagging loudly)**: this repo's real Julia is the **juliaup-managed** interpreter (`PATH="$HOME/.juliaup/bin:$PATH"`, resolves to 1.12.6), per every worktree's own `.knitro_env.sh`/`README.md`. The tempting `/opt/shared_sw/julia/1.10.11/bin/julia` on PATH is a decoy for this codebase — `Base.StaticData`/`Base.ReinferUtils` are both undefined in that build, so `PrecompileTools` (a transitive dependency of nearly everything) throws on any fresh precompile, presenting as confusing depot-corruption-looking errors. Cost real time to diagnose this session; saved as [[julia-toolchain-use-juliaup-not-shared-sw]] in memory.

## 13. Remaining nonblocking follow-ups

- **`q_bufs`/`psi_bufs` full-gradient allocation work** — flagged in the CM/parallel handoff as not yet fixed (~3997MB/call at one site); explicitly out of scope for this merge per the brief.
- **CM checkpoint-schema unification** — `run_cm_upper` (`cm_outer_driver.jl`) still uses an ad-hoc `Ref{Any}` NamedTuple checkpoint, not the schema-3 `D20Checkpoint` path. Real, disclosed gap (confirmed again via `c14_smoke_cm_l50.jl`'s own header comment). Document as a blocker before any *unattended long CM production run* specifically (per brief §13); does not block unrestricted production runs or this merge.
- **`fast_range_screen.jl` missing `@prof` labels** — sites 1/2/7 of the allocation audit could not be decomposed; flagged, not fixed, in the CM/parallel handoff.
- **Dual-bank wall-time savings on short trajectories** — this session's A/B found a *negative* result on one particular 8-point same-process trajectory; the mechanism is correctness-neutral (verified) but its net timing benefit likely depends on trajectory length/warmth. Not a regression — `use_dual_bank=true` stays the default — but worth a longer, separate-process A/B if the dual-bank timing claim needs to be nailed down further.
- **Context-reuse wall-savings measurement confound** (§8) — this session's same-process reuse-vs-rebuild comparison undercounted the rebuild cost due to JIT/disk-cache warmth; the ~10-12% figure from the driver branch's own separate-process A/B remains the trustworthy one.

## 14. Cleanup / archival

`gravity-fullA-cm-parallel-production` and `gravity-fullA-driver-delta5` (and their branches) are subsumed by this merge and marked archival — not deleted, since this doc and the merge commit are the reference for what they contributed. `gravity-fullA-final-production-merge` (the integration worktree) is likewise kept for reference, already merged.

---

**PRODUCTION MERGE COMPLETE — READY FOR RUNS**
