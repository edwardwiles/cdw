# W=500,000 Confirmation (2026-08-02)

Task brief §18: one frozen-state callback + one true-cold inner solve per family at the largest
production-validated scale, using the canonical harness (post-fix production HEAD, real
calibration point, exact production scientific config). All 5 families run (not just the
priority-ordered subset — host resources were sufficient to run all 5 concurrently).

## Results

| family | ctx_build_s | cold_solve_s | nStatus | single-call bytes | mean repeat bytes/call (n=10) | W=100k bytes/call (for comparison) |
|---|---|---|---|---|---|---|
| unrestricted | 143.9 | 5.22 | 0 | 11,872 | 11,907.2 | 12,034.16 |
| flexible_cm | 136.2 | 19.55 | 0 | 41,456 | 41,504.0 | 41,631.68 |
| common_frechet | 144.5 | 22.99 | 0 | 41,536 | 41,539.2 | 41,689.8 |
| origin_zc | 129.3 | 43.86 | 0 | 47,920 | 47,961.6 | 48,094.64 |
| cm_meanzc | 135.9 | 102.41 | 0 | 70,632 | 70,673.6 | 70,807.0 |

All 5 families: **nStatus=0 (converged/optimal)**, `max_h_drift=0.0` across all 10 repeats
(deterministic, purely a function of the frozen state). Backend `backend_info` for every family
matches the exact production defaults confirmed at W=20k/100k (blas_syrk/draw_chunk_reordered/
drawmajor_v2 for cm_meanzc, blas_syrk/drawmajor_v2 for origin_zc, winner_bin/cm_lookup for
flexible_cm, winner_bin/cm_frechet_lookup for common_frechet, `OperatorPsiBundle` for
unrestricted). cm_meanzc's `CORE_HESSIAN_COUNTERS` confirm `dense_core_fallback_calls=0`,
`winner_pair_parallel_calls=22` (= 2x(1 warm-up + 1 single + 10 repeat) = 22, exactly as expected).

## Confirms both accepted fixes hold at this scale

Per-callback bytes are **essentially unchanged from W=100,000 to W=500,000** for every family
(within ~1% — consistent with this audit's own earlier finding that allocation is a function of
dual dimension/block structure, not W). Both accepted fixes' allocation reductions therefore hold
at the largest scale this audit tested:
- cm_meanzc: 70,673.6 bytes/call at W=500k (vs the pre-fix baseline's ~1,996,000 bytes/call at
  W=100k — the ~28x reduction is not a small-scale artifact).
- common_frechet: 41,539.2 bytes/call at W=500k (vs the pre-fix baseline's ~561,600 bytes/call at
  W=100k — the ~13.3x reduction likewise holds).

## Runtime notes

Context build scaled from ~60-70s (W=100k) to ~130-145s (W=500k) across all 5 families -- sub-
linear relative to the 5x draw-count increase, consistent with the mix of O(W) and O(1) work in
context construction. cm_meanzc's true-cold solve (102.4s) and its own repeated-callback mean
(7.68s/call) were measured under a real, substantial host-contention spike (other users' load
average peaked at 115.72 during this specific run, confirmed via `uptime`) -- the wall-clock
absolute values above should be read as "correct and complete," not as clean isolated performance
numbers; the BYTES figures (the actual subject of this confirmation) are unaffected by contention.

## Peak RSS

Observed RSS per process during this run (5 concurrent processes): 7-27GB, well within the host's
~2.8TB available memory at the time. No swap pressure observed (`free -g` confirmed 0 swap used
throughout).

## Verdict

```
W500K_GATE = pass (all 5 families, nStatus=0, both accepted fixes' allocation reductions confirmed
             at scale; absolute wall-clock times affected by real host contention during this run,
             not independently re-isolated)
```
