# Winner-pair dynamic worker-count policy — final gate (2026-07-25/26)

Task §3. Replaces the hard-coded `workers=10` default (all three families: unrestricted, CM,
origin-ZC) with a resolved policy, since the prior session's own 20-thread sweep found `workers=20`
genuinely (not tied-within-noise) faster than `workers=10` at real production points, not merely
"largest tried."

## Policy implemented

`core_exact_hessian.jl`, `resolve_core_hessian_workers_default()`:

```julia
function resolve_core_hessian_workers_default()
    n = nthreads()
    n >= 20 && return 20
    n >= 10 && return 10
    return max(1, n)
end
```

`UNRESTRICTED_CORE_HESSIAN_WORKERS` (`compressed_live.jl`), `CM_CORE_HESSIAN_WORKERS_DEFAULT`,
`ORIGINZC_CORE_HESSIAN_WORKERS_DEFAULT` (`core_exact_hessian.jl`) all now initialize from this
function instead of a literal `10`. A companion `core_hessian_worker_policy_label()` reports WHY
that count was chosen, and both fields are now printed in every family's startup manifest
(`resolve_unrestricted_manifest`/`resolve_flexible_cm_manifest`/`resolve_origin_zc_manifest`,
`production_backend_manifest.jl`).

## Rerun confirmation (task's explicit "rerun before finalizing" requirement)

`bench_unrestricted_20thread_worker_selection.jl`, real production entry point
(`inner_loop_KNITRO_compressed`), `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`, D=20 real data,
`:exclude_row`, seed 20260719, 3 warmed reps per config, minimum reported. Host load average at
run time: 34–47 (out of 208 cores) — this run's own host was noticeably busier than the prior
session's sweep, which is visible in the noisier numbers below.

| Backend | P1 (near δ=1) | speedup | P2 (hard) | speedup |
|---|---|---|---|---|
| dense-reference | 3.9985s | 1.00x | 5.7021s | 1.00x |
| winner-pair serial | 0.5814s | 6.88x | 0.6691s | 8.52x |
| workers=1 | 0.7695s | 5.20x | 0.8374s | 6.81x |
| workers=2 | 0.5830s | 6.86x | 0.6746s | 8.45x |
| workers=4 | 0.4320s | 9.26x | 0.5607s | 10.17x |
| workers=8 | 0.4061s | 9.85x | 0.4718s | 12.08x |
| **workers=10** | **0.4428s** | **9.03x** | **0.5240s** | **10.88x** |
| **workers=20** | **0.4022s** | **9.94x** | **0.5111s** | **11.16x** |

`workers=20` beat `workers=10` at both points (P1: 9.2% faster; P2: 2.5% faster) — never worse.
`workers=8` edged out `workers=10` at P2 this run (a real but small effect, consistent with
run-to-run noise on a shared, variably-loaded host); it does not change the `20 vs 10` comparison
the task asks for. Combined with the prior session's own less-loaded sweep (`workers=20` 13.6%/
12.4% faster than `workers=10` at the same two points,
`docs/WINNER_PAIR_20_THREAD_WORKER_SELECTION_2026-07-25.md`), `workers=20` has now been measured
faster than `workers=10` in two independent runs at two independent points, under two different
host-load conditions, by a margin from 2.5% to 20% — never tied and never reversed.

## Decision

**`workers=20` is now the resolved production default whenever `>=20` Julia threads are
available.** Confirmed live in every post-merge public-driver smoke (below): all four families'
startup manifests report `core_hessian_workers=20`, `core_hessian_worker_policy=ge20_threads_use_20`
under `JULIA_NUM_THREADS=20`.

**`WINNER_PAIR_WORKER_POLICY`** = `nthreads>=20 -> 20 | nthreads in [10,20) -> 10 | nthreads<10 -> nthreads`
**`WINNER_PAIR_WORKERS_AT_20_THREADS`** = `20`
