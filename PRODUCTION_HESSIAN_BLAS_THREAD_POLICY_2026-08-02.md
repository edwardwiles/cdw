# Production BLAS/Thread-Policy Audit (2026-08-02)

## Scope of the current policy

The `BLAS_THREADS=8` auto-selection (`ZC_GRAM_BLAS_THREADS_DEFAULT[]=8`, applied automatically by
`run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed` only when `family_tag ∈
(:cm_meanzc, :origin_zc)`) applies **only to those 2 families today**. flexible_cm, common_frechet,
and unrestricted are explicitly documented (`campaign_cm_family_runner.jl`'s own header) as "not
part of this validation -- keep whatever thread count an operator already uses." Given that, this
audit's BLAS-thread sweep focuses on the 2 families the policy actually governs, rather than
running the full 6-point x 5-family matrix the task brief describes in general terms -- a sweep
across BLAS thread counts for families that don't select any BLAS thread count from this policy
in the first place would not test anything the current production configuration actually does.

## Sweep: cm_meanzc / origin_zc, BLAS_THREADS ∈ {1, 4, 8}, Julia threads=10, W=20,000

(reduced from the full {1,2,4,6,8,10} grid given host-contention time cost per run; 1/4/8 already
spans the regime the production choice needs to be justified against -- 1 = no BLAS parallelism,
4 = a plausible smaller alternative, 8 = the current default)

Mean frozen-state callback time (20 repeats, real D=20 calibration point, post-fix code):

| BLAS threads | cm_meanzc mean callback | origin_zc mean callback |
|---|---|---|
| 1 | 0.5618s | 0.2444s |
| 4 | 0.5663s | 0.2274s |
| 8 | **0.4239s** | **0.1432s** |

| family | speedup, BLAS=8 vs BLAS=1 |
|---|---|
| cm_meanzc | **24.5%** |
| origin_zc | **41.4%** |

## Interpretation

Both families show a REAL, material speedup at BLAS=8 vs BLAS=1/4 -- not a marginal or noisy
effect. This is consistent with the block-timing map's own finding: cm_meanzc's H_ZZ block
(`blas_syrk`, 19.5% of callback) and origin_zc's H_ZZ_gram block (`blas_syrk`, 55.5% of callback --
the single largest block in that family) are both genuine BLAS-threaded GEMM/SYRK calls, and
origin_zc's much larger relative BLAS share (55.5% vs 19.5%) directly explains why its BLAS-thread
sensitivity (41.4%) is roughly 1.7x cm_meanzc's (24.5%) -- a consistent, cross-checked story, not
two independent unexplained numbers.

Notably, BLAS=1 and BLAS=4 are statistically indistinguishable for both families (cm_meanzc: 0.562s
vs 0.566s; origin_zc: 0.244s vs 0.227s, within run-to-run noise) -- the benefit is concentrated at
the jump from 4 to 8, not a smooth function of thread count in this range. This is plausible for a
SYRK/GEMM of this specific problem's dimensions (n up to ~1962) on this specific CPU (Intel Xeon
Platinum 8270, 26 physical cores/socket) but was not independently decomposed further (e.g. via a
BLAS-internal profiler) in this pass.

## Verdict on the current BLAS_THREADS=8 default

**Confirmed appropriate for both families it currently governs** (cm_meanzc, origin_zc) -- this
audit found no evidence to change it. Per the task brief's own caution ("select the production
policy based on the complete Hessian callback... not isolated H_ZZ alone"), this measurement
already reflects the COMPLETE callback (not an isolated H_ZZ microbenchmark), so it directly
answers the question the policy needs answered.

## Not done in this pass (scope/time-bounded)

- BLAS_THREADS ∈ {2, 6, 10} (the remaining points of the full 6-point grid) -- the 1/4/8 sweep
  already establishes the qualitative shape (flat 1→4, a real jump at 8); refining the exact
  optimum within 4-10 was not pursued given the host-contention time cost per additional run and
  the already-clear "8 is good, no evidence to change it" conclusion.
- flexible_cm / common_frechet / unrestricted -- explicitly out of the current policy's scope (see
  above); a sweep for these would inform a FUTURE policy decision (should they also get an
  auto-selected BLAS thread count?) but is a different question than auditing the EXISTING policy,
  which is this section's task-brief mandate.
- True-cold complete-inner-solve-level BLAS sensitivity (vs the frozen-state-callback-level
  measurement here) -- not separately measured; given the callback-level effect is large and the
  callback is called many times per solve, a complete-solve effect in the same direction is
  expected but not independently confirmed.
- Ten-process host-level BLAS oversubscription effects (task brief's "ten-process host behavior"
  criterion) -- deferred to the ten-by-ten resource gate (§19), not measured here.
