# Restricted Lookup-FG Production Port — Phase B1 — 2026-07-26

**State: MATCHED_AB_PASSED** (not merged to production/fullA-exact; on
`port/remediate-production-5x7-audit-2026-07-26`). **Default NOT flipped** — see verdict below.

## Scope

Per user decision during this remediation, Phase B1 covers **plain FLEXIBLE_CM only**.
`CMLookupState` (`cm_lookup_kernels.jl`) is CM-grid-specific (`x = [zeta; lambda_core; lambda_cm]`,
no room for a Fréchet level-anchor block or a mean/pair block) — extending it for
COMMON_FRECHET_CM and FLEXIBLE_CM_PLUS_ZC is new numerical-kernel development, tracked separately,
not part of this port.

## What was built

- `cm_lookup_production.jl` (new file): `_adapt_hess_cb_for_lookup` (thin userParams adapter —
  KNITRO's Hessian closures expect `userParams` to be the dense `obj`; the lookup FG callback
  needs `userParams` to be a `CMLookupState`, `st`; since `st.obj` already holds the dense object,
  the adapter just unwraps it — **zero changes to Hessian math**), plus
  `inner_loop_KNITRO_cmlookup_production`/`inner_loop_internal_cmlookup_production`, mirroring
  `inner_loop_KNITRO_archgeneric`/`inner_loop_internal_archgeneric`'s exact contract so
  `archC_base_state`/`archC_verified_state` can dispatch to either with no other code change.
- A new `inner_fg_backend::Symbol` field on `CMBinHessCtx`, threaded through `build_cm_bin_ctx` ->
  `build_cm_production_context` -> `run_cm_upper_checkpointed`, defaulting to
  `CM_INNER_FG_BACKEND_DEFAULT[] = :dense_reference` at every level (byte-identical to
  pre-existing production behavior unless explicitly overridden).
- Dispatch added to `archC_base_state`/`archC_verified_state`:
  `cctx.inner_fg_backend == :cm_lookup ? inner_loop_internal_cmlookup_production(...) :
  inner_loop_internal_archgeneric(...)`.
- Idempotent (`isdefined(Main, ...) || include(...)`) includes added to the top of
  `cm_production_bundle.jl` for `cm_lookup_kernels.jl`/`cm_lookup_live_knitro.jl`/
  `cm_lookup_production.jl`, following the SAME pattern `cm_hessian_architectures.jl` already
  uses for its own dependencies — avoids touching the 53 files that already `include`
  `cm_production_bundle.jl`.

## A real bug found and fixed during this work (in new code, not pre-existing production code)

The first draft hardcoded `CMLookupState(...; method = :interval, ...)`. Production's
`build_cm_production_context` calls `build_cm_augmented_obj` (cm_production_bundle.jl's own
header: "cumulative basis (build_cm_augmented_obj)") — i.e. production's CM columns are stored in
the **cumulative** basis, not the interval basis. This is exactly the "category error"
`c12i_validate_lookup_fg.jl`'s own comments warn against (comparing `:interval` lookup coefficients
against a cumulative-basis dense reference, or vice versa). Confirmed live: every real inner solve
under the first draft's `:interval` default reported KNITRO `nStatus=-400` (infeasible) at the
calibration point — not a KNITRO or production bug, a basis mismatch in this port's own first
draft, caught immediately by the D=4 correctness gate below. Fixed by defaulting to
`method = :suffix` (the cumulative-basis lookup kernel variant) instead.

## Correctness gates (real, both ALL PASS)

`test_phaseB1_cmlookup_production_correctness.jl` compares `:dense_reference` vs `:cm_lookup`
through the real production entry points (`archC_verified_state`), at the calibration point and a
perturbed point, checking `inner_status`, `zeta*`, `lambda*`, `m_star` (`obj.arg1`), `Delta_dual`,
and a downstream gradient (`cm_production_gradient`) fed each backend's own `base`.

- **D=4 square**, L in (10, 20, 50), contrasts in (:anchored, :orthonormal), both points: **ALL
  PASS**. All quantities agree to ~1e-14 to 1e-17 (machine precision).
- **D=20 real, W=80,000, L=50**, both contrasts, both points: **ALL PASS**. All quantities agree
  to ~1e-13 to 1e-18.

## Performance gate (real, D=20/W=80,000/L=50, calibration point, 5 warmed repetitions each)

```
dense_reference : median=0.9866s  median_alloc=66.4MB  Delta_dual=0.00866050
cm_lookup       : median=0.9003s  median_alloc=74.8MB  Delta_dual=0.00866050
speedup (dense/lookup) = 1.096x   allocation ratio (dense/lookup) = 0.888x
```

## Verdict: default NOT flipped

The task's own stated flip criteria (Phase B4) require ALL of: correctness passes; complete inner
solve improves or is noninferior within 5%; **allocation falls materially**; no stability
regression. Correctness passes cleanly. Wall-clock is modestly better (+9.6%). But **allocation
went UP, not down** (cm_lookup uses ~12.6% MORE memory per complete inner solve than the dense
baseline it was meant to replace) — this fails the stated bar, so **`CM_INNER_FG_BACKEND_DEFAULT`
remains `:dense_reference`**, unchanged from before this port. `:cm_lookup` is now a real,
validated, available alternative (`inner_fg_backend=:cm_lookup` kwarg, threaded all the way to
`run_cm_upper_checkpointed`), correctly classified as `AVAILABLE_BUT_NOT_DEFAULT`, not
`ACTIVE_OPTIMIZED`.

This is a genuinely useful, if less dramatic than hoped, result: the FG callback's dense-BLAS cost
is real (confirmed correct, confirmed present), but it is evidently a smaller share of the total
inner-solve wall-clock than the original audit's framing assumed — the full `archC_verified_state`
call also includes the Hessian callback (`build_bin_tables!`, H_EE/H_CC/H_EC assembly) and
KNITRO's own internal barrier/factorization work, which this port does not touch and which a prior
(pre-this-remediation) profile found to dominate Hessian-callback wall time. A single-point,
single-thread-count measurement is not the full Phase I profiling this task still owes; the
5×300s production profiles (Phase I) will attribute wall-clock across ALL stages, including this
one, with real call counts rather than a 5-repetition microbenchmark.

**Not chased further this session, flagged as a legitimate follow-up**: `CMLookupState`'s own
`build_weighted_histogram` step is `@threads`-parallelizable (`nthreads_use` parameter), but this
port hardcodes `nthreads_use=1` in `inner_loop_internal_cmlookup_production` rather than wiring it
to a resolved worker-count policy (matching e.g. `core_hessian_workers`'s own resolution). Given
this benchmark ran with `-t 8` available Julia threads, the lookup kernel's own histogram
construction left real, unused parallelism on the table — threading it properly could plausibly
close or reverse the current allocation/speed gap, but that is a distinct, scoped follow-up, not
established by evidence in this session.
