# CDF+POWER Status — Experimental, Not Requested — 2026-07-24

Per task addendum §1/§10: `frechet_feature_set=:cdf_power` (the combined CDF + truncated-`(1-σ)`-
power restriction, the "full paper spec") was the previous port-prep session's default, carried
over from a historical "reproduce the paper draft" instruction that the addendum clarifies was
never part of the user's actual current request. This session did **not** spend further
engineering time making `:cdf_power` production-feasible, per explicit instruction.

```
cdf_power_status = experimental_not_requested
```

## What remains true of `:cdf_power` after this session

- Its correctness tests (`test_frechet_power_hessian_d4_gates.jl`, gates P0-P3) are untouched and
  still pass — this session made no changes to `cm_frechet_power_hessian_structured.jl`.
- Its known cost profile is unchanged from the prior port-prep session's own measurement: a
  genuinely new finite-point solve at real D=20/W=80,000/L=50 costs **~660-1000s** (n≈2382, 2,838,153
  dense Hessian entries, single-threaded KNITRO).
- One additional inefficiency was found (not fixed, disclosed only) during this session's Hessian-
  architecture audit: a per-iteration `Vector{Float64}(undef, nO)` allocation inside a doubly-nested
  `l×lp` loop in `cm_frechet_power_hessian_structured.jl`'s cross-block assembly (~2,500 small
  allocations per Hessian callback at L=50). Flagged as a disclosed follow-up for a future session
  that separately elects to invest in `:cdf_power`, not addressed here.
- The threaded/syrk Hessian-construction technique this session validated and applied to
  `:cdf_only` (`cm_frechet_hessian_threaded.jl`) was **not** ported to the `:cdf_power`
  combined-block kernel (`cm_frechet_power_hessian_structured.jl`) — a mechanical port along the
  same lines is plausible future work but out of this session's scope.
- The HVP/matrix-free backend, sparse-Hessian-registration test, and exact-Hessian-vs-HVP
  correctness gates described in the original (pre-addendum) task brief §6-§7 were **not**
  attempted for either feature set in this session — `:cdf_only`'s measured performance
  (see `THREADED_CDF_ONLY_HESSIAN_BENCHMARK_2026-07-24.md`) made this unnecessary for the actual
  production target, and the addendum explicitly deprioritizes further `:cdf_power` engineering.

## Recommendation

Do not let `:cdf_power`'s cost profile determine production readiness for common fixed-Fréchet CDF
marginals — that is `:cdf_only`'s job, and it is addressed as the primary deliverable of this
session (see `FIXED_FRECHET_CDF_ONLY_PRODUCTION_FEASIBILITY_VERDICT_2026-07-24.md`). If a future
task separately requests the combined CDF+POWER restriction as a production target, the direct,
lowest-risk starting point is: (1) port the same threaded/syrk bin-table technique to
`cm_frechet_power_hessian_structured.jl`, (2) fix the disclosed per-iteration `vraw` allocation,
(3) re-run the same KNITRO-thread sweep at n≈2382 (this session found no benefit at n=1382; that
does not necessarily generalize to a ~5× larger dense factorization).
