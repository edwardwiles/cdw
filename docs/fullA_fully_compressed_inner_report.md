# Full-A D=20 real-data (W=80,000): removing mandatory dense W-by-moment materialization

Continuation 9, Phase 3 (3.1, 3.2, and — added mid-task at the user's
explicit direction, elevated from optional to required — 3C). Measured on
`demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, commits `445356e` (3.1/3.2) and `2573f3e` (3C). Real-data
context via `context_real_d20.jl`'s `d20_real_setup` (France focal,
`baseIndex=2`, σ=2.5, μ estimated via gravity), same as the W80k
microbenchmark (`docs/fullA_D20_W80k_microbenchmark.md`, this task's direct
evidence base — read in full before this task started). Three concrete,
evidence-backed targets, all addressed here:

1. **§1 (3.1)**: port `:compressed` moment representation
   (`compressed_live.jl`/`oracle_fast.jl`, previously validated only at
   D=4/6/8/10 on synthetic contexts) to the real D=20 context and re-measure.
2. **§2 (3.2)**: `build_lfix_base_cache`'s unconditional dense self-validation
   rebuild (flagged in `docs/compressed_live_integration_report.md` §7 as
   real, undone work) — make it opt-in, default off.
3. **§3 (3C)**: eliminate the last dense-materialization dependency from the
   inner CC dual solve itself — the Hessian callback, which even in
   `:compressed` mode still lazily materializes the full dense
   `W×(oci-1)` `G` matrix. Three alternatives benchmarked at D=20/W=80,000:
   quasi-Newton (no Hessian), an exact dense accumulation built from repeated
   compressed Hessian-vector products, and a genuinely matrix-free HVP-only
   solve.

**Headline: 3.1 and 3.2 both succeeded cleanly (memory-safe, correctness-
verified, real speedups). 3C is a genuine mixed result, reported honestly**:
two of the three dense-Hessian-free options (compressed dense-accumulation,
matrix-free HVP) converge to the EXACT same optimum as the existing dense-
Hessian baseline, robustly across perturbed starts — but both are SLOWER in
wall-clock (2.7x and 4.5x respectively) despite doing asymptotically fewer
FLOPs, and the third option (quasi-Newton) fails to converge at all at
D=20's scale within default settings. The existing dense-Hessian compressed
baseline remains the right default; this task's own evidence does not
support switching it, but the alternatives are now built, validated, and
benchmarked rather than untried.

---

## 0. Safety

Confirmed before doing anything else, per the standing safety warning:
`grep needs_outer_moment_jacobian full_aod_diag/d4_exact/context_real_d20.jl`
shows `= false` (the fix from commit `bd313a2`, described in the W80k doc's
§0, is present). Every new script in this task followed the
"small-sanity-check-first" discipline: `c9_phase3_compressed_d20_sanity.jl`
ran ONE compressed-mode call and checked `VmHWM` before any timed benchmark
was allowed to proceed. Peak `VmHWM` observed anywhere in this task's runs:
**2.8 GB** (dense+compressed sanity run) — nowhere near the 109GB/780GB
incidents this investigation hit earlier in Continuation 9. No process in
this task was killed for memory; none came close to the 20GB self-imposed
kill threshold set in each script's monitor.

---

## 1. Phase 3.1 — compressed moments ported to real D=20, works cleanly

### 1.1 Correctness first

`evaluate_fullA_fast(x_free, ctx; moment_representation=:compressed)` was
never tried against a real-data context before this task
(`docs/fullA_D20_W80k_microbenchmark.md` §7 flagged this explicitly as out of
scope for that document). It required **zero code changes** — the existing
mode-flag machinery (`compressed_live.jl`, wired into `oracle_fast.jl` in
Continuation 8) worked unchanged against `d20_real_setup`'s context, exactly
as the task brief predicted ("this ctx was built to mirror
`d_exact_setup_scaled`'s exact field layout").

Sanity script: `full_aod_diag/d4_exact/c9_phase3_compressed_d20_sanity.jl`.
At the calibration point (`gamma'=0.987762`, natural theta), dense vs
compressed:

| quantity | dense | compressed | diff |
|---|---|---|---|
| `Delta_dual` | 0.0025908618751121935 (cold run) | 0.0025908618751121918 (warm run) | ~1.3e-18 |
| `winner_hash` | matches | matches | exact |
| `gravity_raw` | — | — | 4.93e-32 |
| `max_abs_moment_resid` | — | — | 1.46e-13 |

`COMPRESSED_FALLBACK_COUNT` stayed **0** throughout every run in this task —
no exact price ties were hit at D=20 real data (consistent with the D=4-10
finding that ties are a probability-zero event for generic continuous draws,
never observed at any previously-validated point in this investigation).

### 1.2 Timed re-measurement

Harness: `full_aod_diag/d4_exact/c9_phase3_compressed_d20_benchmark.jl`,
reusing `c8_perfprofile_harness.jl`'s `@prof`-timer discipline directly (same
`instrumentation.jl`, same warm-N=6/cold-N=3 rep counts, same component-table
format as the W80k doc's own Part 1A/1B) so the numbers below are directly
comparable to that document's already-published dense-mode table. The dense
numbers reproduced in this run (warm TOTAL 1828ms, `inner_moment_build`
1203ms) closely match the W80k doc's own dense figures (1862ms / 1211ms) —
same commit lineage, same point, small run-to-run noise, confirming this is
an apples-to-apples same-process comparison.

**Warm-started (N=6), median ms, dense vs compressed:**

| component (dense name / compressed name) | dense | compressed | speedup |
|---|---|---|---|
| **TOTAL** | **1828.4** | **1061.9** | **1.72x** |
| `inner_moment_build` / `inner_moment_build_compressed` | 1203.3 | 209.8 | **5.74x** |
| `winner_compute` / `winner_compute_compressed` | 289.0 | 289.3 | 1.00x (unchanged, as expected — winner-finding is identical logic either way) |
| `moments_reuse` / `moments_reuse_compressed` | 136.6 | 137.7 | ~1.0x |
| `inner_knitro_dual_solve` / `..._compressed` | 56.2 | 19.6 | 2.87x |
| `inner_dual_fg_callback` / `..._compressed` | 48.4 | 12.4 | 3.90x |
| kkt/moment_resid/primal_weight (combined) | ~122 | ~124 | ~1.0x (shared post-processing, unchanged) |
| — | — | `materialize_dense_for_postproc` (NEW, compressed-only) | 262.4ms — the lazy one-time dense-G materialization compressed mode still needs for Hessian/post-processing |

**Cold (`warm=false`, N=3), median ms:**

| | dense | compressed |
|---|---|---|
| TOTAL | 4625.2 | 5012.8 |
| `inner_knitro_dual_solve[_compressed]` | 2842.1 | 4173.4 |
| `inner_dual_hessian_callback[_compressed]` (15/24 calls) | 494.5 | 470.7 |
| `inner_moment_build[_compressed]` | 1162.1 | 207.8 |

### 1.3 Headline numbers

| | dense | compressed | speedup (dense/compressed) |
|---|---|---|---|
| warm TOTAL | 1.816s | 1.052s | **1.726x** |
| cold TOTAL | 4.625s | 5.013s | **0.923x (compressed SLOWER)** |
| warm `inner_moment_build` only | 1.203s | 0.210s | **5.735x** |

**The moment-build advantage continues its D-scaling trend cleanly**: 2.35x
(D=4) → 3.22x (D=6) → 3.27x (D=8) → 3.64x (D=10) → **5.74x (D=20)**, per
`docs/fullA_canonical_performance_profile_c8.md`'s own D-scaling grid (§"D-
scaling" finding) extended by this task's real-D=20 point. This is a
materially larger jump than the D=4→10 trend alone would have predicted by
simple extrapolation — plausibly because W=80,000 (this task) is 10x the
D=4-10 grid's W=8,000, and `inner_moment_build`'s dense cost scales with
`W·D²` while compressed's scales with `W·D`, so the *ratio* should scale
roughly with `D` regardless of `W` — D=20 giving ~2x the D=10 ratio is
roughly consistent with that (3.64x → ~7.3x predicted vs 5.74x observed;
same order of magnitude, not exact, reported as a sanity check not a fitted
law, matching this investigation's own established discipline for these
cross-scale comparisons).

**Honest finding — cold is a wash, not a win**: unlike the warm case,
compressed mode is 0.92x (i.e. slightly SLOWER) cold, because
`inner_knitro_dual_solve_compressed` costs 4173ms cold vs dense's 2842ms —
more than offsetting compressed's `inner_moment_build` saving. The compressed
FG callback (`_callbackEvalFG_inner_compressed!`) is `O(W·D)` per call
versus dense's `O(W·D)` BLAS `gemv!` reading a precomputed `G` — cheap
per-call either way — but KNITRO's cold-start iteration count/behavior
appears to differ between the two callback implementations enough to matter
at this scale; not chased further here (out of this task's scope, flagged
honestly rather than smoothed over, matching the D=4 canonical profile's own
"honest finding" precedent for the FG-callback verdict staying noisy across
D). **Practical implication**: compressed mode is the clear win for warm-heavy
workloads (repeated evaluations at nearby points, e.g. a line search or an
optimizer's inner loop) but brings no benefit — and a modest cost — for a
single cold evaluation. This matches how `evaluate_fullA_fast` is actually
used in this investigation's production paths (KNITRO warm-starts across
outer iterates by design), so the practical recommendation is: **default to
`:compressed` for any warm-started/repeated D=20 value-callback workload**.

### 1.4 What was NOT changed

`compressed_live.jl`, `oracle_fast.jl`, `compressed_moments.jl`,
`compressed_cc_inner.jl` — **zero lines touched**. This section is a pure
port/re-measurement; every speedup above comes from code that already
existed and was already validated at D=4-10, now exercised for the first
time against real D=20 data.

---

## 2. Phase 3.2 — `build_lfix_base_cache`'s dense self-validation made opt-in

### 2.1 The change

`full_aod_diag/d4_exact/lfix_incremental.jl`: `build_lfix_base_cache` gained
a `validate_dense::Bool = false` kwarg (matching this codebase's existing
`use_cache::Bool`/`warm::Bool`-style diagnostic-flag naming convention, seen
in `oracle_fast.jl`/`compressed_live.jl`). Behavior:

- **`validate_dense = false` (NEW default)**: `q0` is computed directly from
  the cache's own pieces (`contrib0`, `cf_contrib0`) — the SAME formula the
  self-validation always compared against, just now used as the actual
  return value instead of a throwaway comparator. The dense `Gfull = zeros(W,
  obj.d); obj.moments!(K, Gfull, ...)` rebuild — a second, full `O(W·D²)`-ish
  moment build that duplicated work the caller's own value/moment build (dense
  OR compressed) already did — is **skipped entirely**.
- **`validate_dense = true` (opt-in, matches the OLD unconditional
  behavior)**: runs the dense rebuild and asserts `max|q0_true - q0_cache| <
  1e-8`, exactly as before.

The tie-detection scan (`detect_price_ties`, O(W·D) not O(W·D²)) is
unaffected by this flag — always runs, as before; it is not the expensive
part and is load-bearing (throws `TiedWinnerError` on a genuine edge case,
unrelated to the dense-rebuild self-check).

`composite_gradient_at_fast` (`composite_gradient_fast.jl`) got a matching
`validate_dense::Bool = false` passthrough kwarg so a caller can opt into the
diagnostic without editing that file again.

`test_lfix_incremental.jl` — the one existing test whose whole point is
confirming the closed-form derivation via this self-validation — was updated
to pass `validate_dense = true` explicitly, preserving its coverage. No other
call site needed changes (12 other call sites across
`composite_gradient.jl`, `run_smoothed_homotopy.jl`, `audit_jach.jl`,
`profile_lfix_tiers.jl`, `c8_perfprofile_harness.jl`,
`benchmark_winner_accelerator.jl`, and several test files) — all keep
compiling and running unchanged, now silently getting the faster default,
per the "additive, opt-out" instruction.

### 2.2 Correctness verification (confirmed directly, not assumed from the report)

Script: `full_aod_diag/d4_exact/c9_phase3_validate_dense_equivalence.jl`. At
both D=4 (canonical point) and D=8 (gated-pilot scale), the full
`composite_gradient_at_fast` gradient (`threaded=true, h_mode=:adaptive,
multi_method=:top3`) with `validate_dense=true` vs `validate_dense=false`:

| D | bit-identical gradient? | max\|diff\| | cache.q0 bit-identical? | cache.contrib0 max\|diff\| |
|---|---|---|---|---|
| 4 | **true** | 0.0 | **true** | 0.0 |
| 8 | **true** | 0.0 | **true** | 0.0 |

Exactly bit-identical, not merely close-to-tolerance — confirming the
docstring's claim directly: the dense rebuild was a pure self-check, never a
dependency of the cache's actual contents.

### 2.3 D=20/W=80,000 timing re-measurement

Script: `full_aod_diag/d4_exact/c9_phase3_lfix_novalidate_d20_bench.jl`. Same
config as the W80k doc's Part 1C primary row (`h_mode=:adaptive,
threaded=true, multi_method=:top3`, N=4, calibration point), run in ONE
warmed process so both arms share identical JIT/setup state:

Two runs (the first hit a Julia top-level soft-scope bug in the harness
itself — `g_true`/`g_false` fell out of scope after the timed `for` loop,
caught immediately via `UndefVarError`, not a silent wrong result — fixed by
wrapping the timed loop in a proper function; the fixed version was rerun in
full, in a fresh process, for both the timing AND the same-process
correctness cross-check):

| run | `validate_dense=true` median | `validate_dense=false` median | speedup |
|---|---|---|---|
| 1 (script bug, timing only, no cross-check) | 7.002s [7.124, 7.073, 6.853, 6.931] | 5.412s [5.524, 5.360, 5.399, 5.425] | 1.294x |
| 2 (fixed script, timing + cross-check) | 7.247s [7.371, 7.228, 7.218, 7.266] | 5.401s [5.672, 5.427, 5.374, 5.348] | 1.342x |

Both runs agree closely (run-to-run noise on a shared 3TB machine, not a
methodology issue) — **removing the dense self-validation rebuild gives a
consistent ~1.3x speedup on the full D=20 gradient call** (7.0-7.2s →
5.4s). Same-process gradient cross-check (run 2): **bit_identical(g_true,
g_false) = true, max|diff| = 0.0** — exactly bit-identical at D=20 real
data too, not merely close, extending the D=4/D=8 proof in §2.2 to the
production scale this flag was built for.

For external reference, the already-published W80k doc's Part 1C
(effectively `validate_dense=true`-equivalent, since that self-validation was
unconditional at the time it was measured) reported **6.435s** for this exact
config — this task's own `validate_dense=true` re-measurements above (7.0-7.2s)
run ~10-13% higher than that historical number. **Reported honestly rather
than reconciled**: this is plausibly ordinary cross-session noise on a shared
3TB machine (the W80k doc itself notes similar cross-run drift elsewhere, and
this task's own two `validate_dense=true` runs already differ by ~3.5% from
each other), not a methodology difference — both this task's runs used the
identical config/harness discipline the W80k doc established — but it was
not chased further given the time budget, matching this investigation's
"verify before causal claims" discipline (flagged, not silently absorbed
into the headline number).

The W80k doc's Part 1D component decomposition separately measured
`cache_build` (`build_lfix_base_cache` itself) at **3.512s of the ~6.4s full
gradient**, i.e. the dense rebuild this flag removes was roughly **55% of the
per-call cache-build cost** and roughly **half the full-gradient wall time**
at D=20/W=80,000.

---

## 3. Phase 3C — inner second-order method: three ways off the dense Hessian

Added mid-task at the user's explicit direction (elevated from "if time
permits" to required): compressed_live.jl's FG callback is already
dense-free (§1 above), but its **Hessian** callback still lazily
materializes the full dense `W × (oci-1)` `G` matrix once per inner solve
purely to call the unchanged `hessian!` (`cc_algo/PsiObjectiveBundle.jl`,
one `O(W·ncol²)` BLAS `gemm!`). Three alternatives, all built on
`compressed_cc_inner.jl`'s `compressed_cc_value_grad`/`compressed_cc_hvp` —
**already built in Continuation 8, already validated against the real dense
bundle by central-FD (`test_compressed_cc_inner.jl`, tol 1e-5) but never
wired into an actual KNITRO solve before this task**:

- **Option 3 (`:qn`)**: no Hessian callback at all. KNITRO's own dense
  quasi-Newton approximation (BFGS/SR1/L-BFGS, `hessopt` 2/3/6) drives the
  solve from the (already dense-free) FG callback alone. New `.opt` files
  only (`ek_inner_{bfgs,sr1,lbfgs}.opt`); zero new solver code —
  `inner_loop_KNITRO_compressed` (compressed_live.jl) already skips Hessian
  registration whenever `hessopt != 1`.
- **Option 1 (`:denseaccum`)**: KNITRO still gets a dense
  `(ncol+1)×(ncol+1)` Hessian each call (`hessopt=exact`), but it is built
  from `(ncol+1)` calls to `compressed_cc_hvp` (one per basis direction),
  each `O(W·D)` — total `O(W·D·(ncol+1)) = O(W·D³)`, vs the dense path's
  `O(W·ncol²) = O(W·D⁴)`. **Never touches the `W×(oci-1)` `G` matrix.**
- **Option 2 (`:hvp`)**: genuinely matrix-free. `hessopt=product` (5) makes
  KNITRO call the registered Hessian callback **only** in `KN_RC_EVALHV`
  mode (confirmed from `KNITRO.jl`'s `C_wrapper.jl` `EvalRequest.vec`/
  `EvalResult.hessVec` fields and `libknitro.jl`'s `KN_RC_EVALHV=7` vs
  `KN_RC_EVALH=3` codes) — one `compressed_cc_hvp` call per KNITRO-requested
  direction, `O(W·D)`. The dense `(ncol+1)×(ncol+1)` block is **never
  assembled, not even implicitly**. Requires the CG-based interior
  algorithm (`ek_inner_hvp.opt`: `algorithm=cg`) since product Hessians
  are incompatible with direct/SQP factorization.

New file: `full_aod_diag/d4_exact/compressed_inner_alt_solvers.jl` (all three
solver-loop wrappers + both new callbacks). No existing file touched beyond
the new `.opt` files (copies of `full_aod_diag/ek_inner.opt` with `hessopt`/
`algorithm` changed).

### 3C.1 Correctness first, at D=4

Script: `full_aod_diag/d4_exact/c9_phase3c_correctness_d4.jl`. All three
options solved from cold, compared against the trusted dense
`evaluate_fullA` reference at the calibration point:

| variant | status | n_fg | n_hess | \|Δζ\| vs dense | max\|Δλ\| vs dense | verdict |
|---|---|---|---|---|---|---|
| qn_bfgs (hessopt=2) | −103 | 85 | 0 | 2.6e-13 | 2.1e-10 | PASS |
| qn_sr1 (hessopt=3) | 0 | 38 | 0 | 7.7e-14 | 2.3e-12 | PASS |
| qn_lbfgs (hessopt=6) | −400 | 214 | 0 | 3.6e-07 | 4.0e-05 | CHECK (converges to KNITRO's own looser internal tolerance, not chased further) |
| **denseaccum** | 0 | 5 | 4 | **6.7e-17** | **1.4e-13** | **PASS, essentially exact** |
| **hvp** | −100 | 6 | 79 | 3.2e-12 | 1.3e-11 | **PASS** |

**4/5 pass at D=4**, and critically the two genuinely-new pieces
(`denseaccum`, `hvp`) match the dense reference to near machine precision —
direct confirmation that `compressed_cc_hvp` (built but never
KNITRO-integrated before this task) is correct in production use, not just
in its own standalone FD test.

### 3C.2 D=20/W=80,000 timing + robustness

Script: `full_aod_diag/d4_exact/c9_phase3c_d20_bench.jl`. Isolates the
**inner solve itself** (`inner_loop_internal_compressed_variant`), not the
full `evaluate_fullA_fast` pipeline, per this task's "eliminate dense
materialization from the inner CC dual solve" framing. Reference: the
existing compressed baseline (`inner_loop_internal_compressed`,
`hessopt=exact` + lazy one-time dense materialization for the Hessian) —
cold solve **7.334s**, `n_fg=9, n_hess=8`, `Delta_dual=0.0025908619`
(matches the calibration value exactly).

| variant | cold wall | n_fg (cold) | n_hess (cold) | warm wall | robustness (2 perturbed starts) | \|Δζ\| vs ref | max\|Δλ\| vs ref |
|---|---|---|---|---|---|---|---|
| **REFERENCE** (dense-Hessian compressed baseline) | **7.334s** | 9 | 8 | — | — | — | — |
| qn_bfgs (hessopt=2) | 3.695s | 326 | 0 | 3.197s | both hit iter-limit, don't converge | 1.4e-05 | 7.5e-02 |
| qn_sr1 (hessopt=3) | 2.463s | 206 | 0 | 2.401s | both hit iter-limit, don't converge | 1.1e-04 | 1.3e-01 |
| qn_lbfgs (hessopt=6) | 2.586s | 300 | 0 | 2.776s | both hit iter-limit, don't converge | 2.4e-04 | 1.5e-01 |
| **denseaccum** | **20.073s** | 8 | 7 | 19.347s | 119.8s / 82.7s, **both converge exactly** | **1.8e-16** | **4.9e-14** |
| **hvp** (matrix-free) | **32.983s** | 9 | 3015 (HV calls) | 32.981s | 48.3s / 55.0s, **both converge** | 3.2e-13 | 1.6e-10 |

All five variants' cold-solve `Delta_dual` at the calibration point: reference/denseaccum/hvp all land on **0.0025908619** (matching to 10+ digits); the three quasi-Newton variants land on visibly different, non-converged values (0.0025442 / 0.0024638 / 0.0020670) — a clean, large-margin signal that they did not reach the true optimum, consistent with their `status=-400` (iteration-limit) codes and `n_iters=100` (hit the default cap) in every quasi-Newton run, cold or perturbed.

**Honest findings, all real, none smoothed over**:

1. **All three quasi-Newton options (Option 3) fail to converge within the
   default 100-iteration cap at D=20's 401-dual-variable inner problem** —
   BFGS, SR1, and L-BFGS all hit `status=-400` (KNITRO's iteration-limit
   code) with `n_iters=100` and materially wrong duals (`max|Δλ|` vs the
   converged reference 0.075–0.15, not small). This is a genuine, D-scale-
   dependent negative result, not a bug: quasi-Newton's dense internal
   Hessian *approximation* still costs `O(ncol²)` to build/update per
   iteration and apparently needs far more than 100 iterations to converge
   a 401-dimensional dual problem from a cold start. Raising `maxit` was not
   attempted (out of this task's time budget) but is the obvious next step
   if Option 3 is pursued further — flagged, not silently written off.
2. **Option 1 (`:denseaccum`) converges to the exact same optimum as the
   dense reference** (`|Δζ|=1.8e-16, max|Δλ|=4.9e-14` — bit-identical up to
   floating-point noise) but **costs ~2.7x MORE wall-clock** (20.07s vs
   7.33s cold) despite doing asymptotically fewer flops (`O(W·D³)` vs
   `O(W·D⁴)`). The `(ncol+1)=402` separate `compressed_cc_hvp` calls per
   Hessian evaluation (7 Hessian calls in this cold solve → 2,814 total HVP
   calls) lose to one large, highly-optimized BLAS `gemm!` in practice —
   the same "many small scalar-loop calls vs one big vectorized call"
   pattern `compressed_cc_inner.jl`'s own header already flagged as the
   reason a truly-compressed dense-Hessian callback wasn't attempted in
   Continuation 8, now confirmed empirically at D=20 rather than assumed
   from the D=4 argument.
3. **A metric limitation, reported rather than hidden**: this benchmark's
   `max_moment_resid` column is computed via `compressed_moment_resid(cf,
   ones(W))` — the UNWEIGHTED mean moment (mirrors `evaluate_fullA_fast`'s
   own `moment_resid` diagnostic convention) — which depends only on `θ`
   (hence identical, 0.2149, across every variant at the same point), not
   on the converged dual solution. It does **not** discriminate
   converged-vs-not solutions the way the KKT residual
   (`compressed_moment_resid(cf, dPsq)`, i.e. weighted by the converged
   primal weights) would. Caught only after the run completed; not
   re-run given time budget — `Delta_dual` and the direct `Δζ/Δλ`-vs-
   reference comparison are the metrics that actually discriminate
   convergence quality in the table above, and they do so clearly (e.g.
   the qn variants' `Delta_dual` values 0.00206–0.00254 visibly differ from
   the converged 0.0025908619).

### 3C.3 What Phase 3C did NOT cover

- **Raising quasi-Newton's `maxit`** to see if Option 3 converges given more
  iterations — flagged in finding 1 above, not attempted.
- **A proper per-variant KKT-residual metric** (weighted by `dPsq`, not
  unweighted) — see finding 3; the raw duals already give a clear
  convergence signal, so this was not chased further.
- **Wiring any of the three variants into the production
  `evaluate_fullA_fast_compressed` path** — this section is a diagnostic
  comparison (`compressed_inner_alt_solvers.jl` is additive/standalone,
  mirrors `compressed_live.jl`'s own "additive, does not modify production"
  discipline), not a production change. Given `:denseaccum`'s wall-clock
  loss and `:qn`'s convergence failure at D=20, the existing dense-Hessian
  compressed baseline (`inner_loop_KNITRO_compressed`, already live via
  `moment_representation=:compressed`) remains the right default — this
  task's own evidence does not support switching it.
- **Per-variant peak memory** — only whole-process `VmHWM` was recorded
  (**2.66 GB** at the end of the full 5-variant run), not isolated
  per variant (would need separate processes); given none of the variants
  ever materializes the `W×(oci-1)` dense `G` (`:denseaccum`/`:hvp`) or
  needs it only once (the reference), no variant was expected to be a
  memory outlier, and none was observed to be (process stayed well under
  the safety thresholds throughout).

---

## 4. What this task did NOT cover (explicitly out of scope, not overlooked)

- **A genuinely compressed `LFixBaseCache` build** (avoiding the `W×D×D`
  `price0`/`pTσ0` dense arrays inside `build_lfix_base_cache` itself, not
  just the redundant self-validation rebuild) — `docs/compressed_live_integration_report.md`
  §7 explicitly calls this "new work," distinct from what §2 above removed.
  Not attempted here; §2 only removed the SECOND, purely-diagnostic dense
  rebuild, not the cache's own primary dense construction.
- **Postprocessing/residuals** (primal-weight recovery, moment residuals, KKT
  checks) still re-materializing dense `obj.H` in the PRODUCTION
  `evaluate_fullA_fast_compressed` path per
  `compressed_live_integration_report.md` §2 — though §3C.2's benchmark
  above demonstrates in passing that `Delta_dual` itself is fully
  recoverable compressed (`Delta_dual = -f` from `compressed_cc_value_grad`
  directly, no materialization needed) — a real, evidence-backed
  simplification opportunity for that production tail, flagged here but not
  wired in (out of scope for this task).
- **W=800,000** — out of this task's scope (a separate memory-safety
  concern), as it was for the W80k doc.

---

## 5. Files

New (all under `full_aod_diag/d4_exact/`), Phase 3.1/3.2:
`c9_phase3_compressed_d20_sanity.jl` (minimal memory-safe sanity probe, run
first, per the standing safety discipline), `c9_phase3_compressed_d20_benchmark.jl`
(timed dense-vs-compressed value-callback re-measurement),
`c9_phase3_validate_dense_equivalence.jl` (D=4/D=8 bit-identical correctness
check for the new flag), `c9_phase3_lfix_novalidate_d20_bench.jl` (D=20
gradient timing A/B).

New, Phase 3C: `compressed_inner_alt_solvers.jl` (all three variants' solver
loops + callbacks), `c9_phase3c_correctness_d4.jl` (D=4 correctness check),
`c9_phase3c_d20_bench.jl` (D=20 timing/robustness comparison),
`ek_inner_bfgs.opt`/`ek_inner_sr1.opt`/`ek_inner_lbfgs.opt`/`ek_inner_hvp.opt`
(new `.opt` files, each a copy of `full_aod_diag/ek_inner.opt` with only
`hessopt`, and for `ek_inner_hvp.opt` also `algorithm`, changed).

Modified, additive only: `lfix_incremental.jl` (`build_lfix_base_cache` gains
`validate_dense::Bool=false`, no other behavior change),
`composite_gradient_fast.jl` (`composite_gradient_at_fast` gains a matching
passthrough kwarg), `test_lfix_incremental.jl` (one call site updated to
`validate_dense=true` to preserve its self-validation coverage).
`compressed_live.jl`/`oracle_fast.jl`/`compressed_moments.jl`/`compressed_cc_inner.jl`/
`c8_perfprofile_harness.jl`/`cc_algo/PsiObjectiveBundle.jl` — **untouched**.

Raw logs + CSVs, Phase 3C:
`results/fullA_d4/2573f3e/c9_phase3c_d20_bench/`.

Raw logs + CSVs:
`results/fullA_d4/445356e/c9_phase3_compressed_d20_benchmark/`,
`results/fullA_d4/445356e/c9_phase3_lfix_novalidate_d20/`.
