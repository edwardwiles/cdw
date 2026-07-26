# Origin-ZC matched outer A/B: dense-reference vs shared winner-pair H_EE (final gate)

Task §5. The second genuine gap the prior sessions disclosed as not run. New harness
(`winner_pair_outer_ab_originzc_2026-07-25.jl`), adapted from the restricted-immutable-workspace
port's own origin-ZC A/B harness (`restricted_workspace_outer_ab_originzc.jl`) — same real public
driver (`run_originzc_upper_checkpointed`), unmodified, K_mean=K_pair=1, same resource/screen/cache
policy; the only change between the two arms is the global `ORIGINZC_CORE_HESSIAN_BACKEND_DEFAULT[]`
Ref, set once before the driver call.

## Configuration

D=20, D_dest=19 (`destination_sample=:exclude_row`), W=80,000, seed 20260719,
`distribution_restriction=:origin_specific_moments_zero_covariance`, K_mean=1, K_pair=1, delta=1,
genuine calibrated start, `cm_gradient_backend=:cplus`, 20 Julia threads,
`OPENBLAS_NUM_THREADS=1`, one process at a time, 150s measured budget.

## Results

| Metric | dense-reference | winner-pair |
|---|---|---|
| measured wall | 193.072s | 182.327s |
| n_eval / n_grad | 11 / 8 | 11 / 8 |
| knitro_status | -411 | -411 |
| kappa | 0.04897846802396799 | 0.04897846802396799 (**identical**) |
| best_Delta | 0.8460550137406051 | 0.8460550137406035 |
| best_n_eval | 6 | 6 |
| best_t (time to best) | 81.679s | 64.908s |
| cold-verify Δ_dual | 0.8460550137406057 | 0.8460550137406044 |
| cold-verify \|diff\| | 5.55e-16 | 8.88e-16 |
| cold-verify ok | true | true |
| alloc / GC | 13.41GB / 0.868s (49 GCs) | 15.22GB / 1.236s (53 GCs) |
| peak RSS | 1224 KB | 1192 KB |
| winner_pair_hessian_calls | 0 | 151 (all parallel) |
| dense_core_fallback_calls | 119 (`debug_reference_requested`) | 0 |
| compressed_core_rebuilds | 0 | 14 |

## Assessment against task requirements

- **Complete numerical agreement**: `kappa` is bit-identical between arms. `Delta_dual` agrees to
  8.9e-16 absolute, cold-verified independently in both arms via `cm_originzc_production_value_verified`.
- **Zero unexplained fallback**: winner-pair arm's `dense_core_fallback_calls=0`; the dense arm's
  119 fallback calls are all the expected `:debug_reference_requested` reason.
- **No regression in verified progress**: identical `n_eval`/`n_grad`/`kappa`/`best_n_eval`; winner-pair
  reaches the SAME best point 20.6% sooner (64.9s vs 81.7s) and finishes the measured window 5.6%
  faster overall — a clean win, not merely a non-regression.
- **No new callback or checkpoint failures**: both arms terminate at `knitro_status=-411` (the same
  normal status both arms), no errors.

**Verdict: PASS** (clean win, not just non-regression).
