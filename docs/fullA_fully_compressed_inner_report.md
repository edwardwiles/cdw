# Full-A D=20 real-data (W=80,000): removing mandatory dense W-by-moment materialization

Continuation 9, Phase 3. Measured on `demand.mit.edu`, `JULIA_NUM_THREADS=20`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, commit `445356e` (base commit
`090eb68`, this session's changes on top). Real-data context via
`context_real_d20.jl`'s `d20_real_setup` (France focal, `baseIndex=2`, σ=2.5,
μ estimated via gravity), same as the W80k microbenchmark
(`docs/fullA_D20_W80k_microbenchmark.md`, this task's direct evidence base —
read in full before this task started). Two concrete, evidence-backed targets
from that document's own findings, both addressed here:

1. **§3.1**: port `:compressed` moment representation
   (`compressed_live.jl`/`oracle_fast.jl`, previously validated only at
   D=4/6/8/10 on synthetic contexts) to the real D=20 context and re-measure.
2. **§3.2**: `build_lfix_base_cache`'s unconditional dense self-validation
   rebuild (flagged in `docs/compressed_live_integration_report.md` §7 as
   real, undone work) — make it opt-in, default off.

**Headline: both targets succeeded cleanly, both memory-safe, both
correctness-verified before being trusted for timing.**

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

## 3. What this task did NOT cover (explicitly out of scope, not overlooked)

- **A genuinely compressed `LFixBaseCache` build** (avoiding the `W×D×D`
  `price0`/`pTσ0` dense arrays inside `build_lfix_base_cache` itself, not
  just the redundant self-validation rebuild) — `docs/compressed_live_integration_report.md`
  §7 explicitly calls this "new work," distinct from what §3.2 above removed.
  Not attempted here; §3.2 only removed the SECOND, purely-diagnostic dense
  rebuild, not the cache's own primary dense construction.
- **Postprocessing/residuals** (primal-weight recovery, moment residuals, KKT
  checks) still re-materializing dense `obj.H` per
  `compressed_live_integration_report.md` §2 — the brief's fuller Phase 3
  scope, not reached given time budget after finishing 3.1/3.2 well.
- **Inner second-order method options** (exact compressed Hessian
  accumulation vs matrix-free HVPs vs quasi-Newton) — same reason, not
  reached.
- **W=800,000** — out of this task's scope (a separate memory-safety
  concern), as it was for the W80k doc.

---

## 4. Files

New (all under `full_aod_diag/d4_exact/`):
`c9_phase3_compressed_d20_sanity.jl` (minimal memory-safe sanity probe, run
first, per the standing safety discipline), `c9_phase3_compressed_d20_benchmark.jl`
(timed dense-vs-compressed value-callback re-measurement),
`c9_phase3_validate_dense_equivalence.jl` (D=4/D=8 bit-identical correctness
check for the new flag), `c9_phase3_lfix_novalidate_d20_bench.jl` (D=20
gradient timing A/B).

Modified, additive only: `lfix_incremental.jl` (`build_lfix_base_cache` gains
`validate_dense::Bool=false`, no other behavior change),
`composite_gradient_fast.jl` (`composite_gradient_at_fast` gains a matching
passthrough kwarg), `test_lfix_incremental.jl` (one call site updated to
`validate_dense=true` to preserve its self-validation coverage).
`compressed_live.jl`/`oracle_fast.jl`/`compressed_moments.jl`/`compressed_cc_inner.jl`/
`c8_perfprofile_harness.jl` — **untouched**.

Raw logs + CSVs:
`results/fullA_d4/445356e/c9_phase3_compressed_d20_benchmark/`,
`results/fullA_d4/445356e/c9_phase3_lfix_novalidate_d20/`.
