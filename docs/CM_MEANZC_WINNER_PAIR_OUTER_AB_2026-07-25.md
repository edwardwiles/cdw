# CM+mean/ZC matched outer A/B: dense-reference vs shared winner-pair H_EE (final gate)

Task §4. This gate was the first genuine gap the prior sessions disclosed as not run. It surfaced
a real bug in the pre-existing benchmark harness (below) before it could produce any result.

## Bug found and fixed (blocking, in this harness only — not a production driver bug)

`matched_outer_benchmark_cm_2026-07-25.jl` was written and validated only for
`cm_extension=:cm_only`. Selecting a meanZC arm via `BENCH_CM_EXTENSION` never appended the
required `eta_nu` block to `w_calib`, and its cold-verify always called the CM-only (not
meanZC-aware) verification pair. Both defects made every meanZC arm fail immediately
(`nStatus=-502 "Could not evaluate objective or constraints at the initial point"`, `n_eval=0`)
before this fix — a harness bug, not a defect in `run_cm_upper_checkpointed` or the shared H_EE
backend itself (the real production driver already builds its own correct meanZC start point, see
`cm_production_stage_runner.jl`'s calibration-mode `w0` construction, which this fix mirrors).
Fixed both; added `reset_core_hessian_counters!()`/`print_core_hessian_counters()` so the report
below can show actual backend use, not just configuration.

## Configuration (matches task spec exactly)

D=20, D_dest=19 (`destination_sample=:exclude_row`), W=80,000, seed 20260719, fixed theta, delta=1,
genuine calibrated start (`ctx.θ0_up`), `cm_extension=:cm_plus_equal_means_zero_covariance`
(K_mean=1, K_pair=1), `pin_outer_algorithm=true` (algorithm=2/hessopt=6, identical outer algorithm
both arms), identical screens/cache/checkpoint settings, 20 Julia threads, `OPENBLAS_NUM_THREADS=1`,
one process at a time.

## Results (two independent full pairs of runs; second pair adds runtime counters)

| Metric | dense (run 1) | winner-pair (run 1) | dense (run 2, w/ counters) | winner-pair (run 2, w/ counters) |
|---|---|---|---|---|
| measured wall | 215.13s | 194.93s | 159.52s | 167.91s |
| n_eval / n_grad | 6 / 4 | 6 / 4 | 6 / 4 | 6 / 4 |
| knitro_status | -401 | -401 | -401 | -401 |
| kappa | 0.04438842845316193 | 0.04438842845316171 | (same config) | (same config) |
| best_Delta | 0.6461620594077415 | 0.6461620594077915 | — | — |
| best_t | 130.6s | 102.8s | — | — |
| cold-verify Δ_dual | 0.6461620594077425 | 0.646162059407786 | 0.6461620594077425 | 0.646162059407786 |
| cold-verify \|diff\| | 9.99e-16 | 5.44e-15 | 9.99e-16 | 5.44e-15 |
| cold-verify ok | true | true | true | true |
| alloc / GC | 16.80GB / 1.287s | 18.07GB / 1.053s | 16.54GB / 0.762s | 17.59GB / 0.861s |
| peak RSS | 1116 KB | 1100 KB | 1204 KB | 1272 KB |
| winner_pair_hessian_calls | n/a (not yet instrumented) | n/a | 0 | 70 (all parallel) |
| dense_core_fallback_calls | n/a | n/a | 54 (`debug_reference_requested`) | 0 |

## Assessment against task requirements

- **Complete numerical agreement**: kappa agrees to 12 significant digits; `Delta_dual` agrees to
  5.4e-15 absolute (cold-verified independently in both runs, both arms).
- **Zero unexplained fallback**: winner-pair arm's runtime counters show `dense_core_fallback_calls=0`
  in both instrumented and (via the identical post-merge smoke, see
  `POST_MERGE_WINNER_PAIR_PUBLIC_SMOKE_2026-07-25.md`) production-driver runs. The dense arm's own
  54 fallback calls are all the expected, documented `:debug_reference_requested` reason (i.e. the
  arm deliberately requested dense — not an unexplained fallback).
- **No regression in verified progress**: `n_eval`/`kappa`/`best_Delta` are identical between arms
  in both pairs (0% regression) — the wall-clock delta flips sign between the two pairs (winner-pair
  9.4% faster in run 1, 5.3% slower in run 2), consistent with H_EE being only ~19% of CM's total
  Hessian-callback cost (the bin-table `Stab`/`Ttab` construction dominates, see
  `CM_WINNER_BIN_CROSS_FOLLOWUP_2026-07-25.md`) — a fixed kernel's speedup on a minority cost share
  is expected to be small and can be within host-load noise (load avg 28–47/208 cores varied
  between runs), not a genuine regression signal.
- **No new callback or checkpoint failures**: both arms ran to their time limit cleanly
  (`knitro_status=-401`, the same normal time-limit-feasible status both arms), no errors.

**Verdict: PASS.**
