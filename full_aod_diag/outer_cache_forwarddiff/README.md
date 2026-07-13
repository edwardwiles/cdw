# Outer-loop caching + ForwardDiff optimization + reverse-mode NaN bisection

Isolated, additive diagnostics directory. **No file under `cc_algo/`, `moments/`, or any other
production path was modified.** Continues `../ad_benchmark/` (which established the envelope-
scalar formula and validated it against production) and `../SESSION_SUMMARY_2026-07-12.md`
(which corrected the normalization/parameter-count assumptions this task was originally briefed
under). Config throughout: D=4, γ_d≡1-for-all-d + direct-γ' objective (this session's
best-conditioned variant).

## Normalization / parameter-count correction (carried into every file here)

The task that initiated this work assumed the OLD `A[1,d]=1` normalization and a `D·(D-1)`
free-bilateral-A-parameter count. The actual current configuration pins `γ_d≡1` for every
destination and lets the FULL `A_od` matrix float — **D² free bilateral parameters** (16 at D=4,
not 12; 400 at D=20, not 380), plus γ'_focal and μ, giving **18 free outer parameters at D=4**
(numerically confirmed in `audit_outer_callbacks.jl`'s own run: `free_outer_params=18`) — a
similar total count to the original brief's guess, but a different composition (D² A_od + 2
scalars, not "12 bilateral A + ~6 other"). See `call_graph_audit.md`'s opening section for detail.

## What this directory answers, and what it deliberately does NOT re-answer

`../ad_benchmark/` already answered (validated, not redone here): the envelope-theorem scalar
formula, ForwardDiff vs. production-Jacobian correctness (relerr ~1e-15), the structural
sparsity pattern, and the first pass of the Enzyme/Mooncake NaN investigation (ruling out
hard-max, the A_od reshape/vcat pattern, and — partially — the custom `gamma` rule). This
directory:

1. **Audits and fixes outer-loop callback duplication** (new — not covered by `ad_benchmark/`,
   whose diagnostics all used `eval_fcga=yes`, avoiding the very problem this audit found in
   PRODUCTION's actual `eval_fcga=no` default).
2. **Optimizes the direct-scalar-ForwardDiff configuration** (chunk size, `GradientConfig`
   caching, closure-vs-functor) — `ad_benchmark/` established that Method B is correct and
   faster than the dense Jacobian, but did not optimize Method B's OWN configuration.
3. **Pushes the Enzyme/Mooncake NaN bisection one level deeper** — rules out `sum(;dims=k)`, the
   two-way demeaning broadcast, and the dead `doubleDiff` preamble as the cause, narrowing (not
   yet solving) where the bug must be.

## 1. What exact scalar function is differentiated?

`../ad_benchmark/derivative_core.jl::envelope_scalar_div_ctx(θ, ctx)` — see `envelope_formula.md`
for the exact formula and what's held fixed. Unchanged from `ad_benchmark/`; not re-derived here.

## 2. What inner objects are held fixed by the envelope theorem?

`λ` (inner-problem multipliers) and `arg1` (per-draw dΨ weight), both taken from a REAL KNITRO
inner solve at the benchmark θ. See `envelope_formula.md`.

## 3. Did the old production code construct a full moment Jacobian unnecessarily?

Yes, established in `ad_benchmark/` (not redone here): production's default
(`moments_jacobian!=error` → `calculate_jac_θ_autodiff!`) builds the dense N×(d+2)×l Jacobian
via ForwardDiff for the divergence-budget constraint gradient, when a direct scalar gradient
(Method B) reproduces it exactly. This session's `audit_outer_callbacks.jl` run used production's
actual default (`moments_jacobian!=error`, i.e. the dense Jacobian) throughout, so all
inner-solve/gradient timings reported here reflect what production ACTUALLY pays today, not a
best-case.

## 4. Is the direct scalar ForwardDiff gradient correct?

Yes — re-confirmed here via a THIRD check beyond `ad_benchmark/`'s: every ForwardDiff
configuration variant benchmarked below (baseline closure, cached `GradientConfig`, functor, all
7 chunk sizes) is asserted (`@assert isapprox(...; rtol=1e-12)`) to match a common reference
gradient at all 4 points before its timing is recorded — see `forwarddiff_benchmark.csv`.

## 5. What chunk size and configuration are fastest?

`benchmark_forwarddiff.jl`, D=4 (l=23, N=8000), mean over the 4 frozen points
(`forwarddiff_benchmark.csv`):

```
config                              mean_time   mean_alloc
1_baseline_closure_autochunk        0.0799s     243.6 MB   <- current de-facto Method B (ad_benchmark's)
2_cached_config_closure_autochunk   0.0829s     111.7 MB
3_functor_cached_config_autochunk   0.0837s     111.7 MB
4_functor_chunk1                    0.3909s     231.5 MB
4_functor_chunk2                    0.2188s     170.7 MB
4_functor_chunk3                    0.1602s     147.1 MB
4_functor_chunk4                    0.1221s     135.3 MB
4_functor_chunk6                    0.1064s     123.5 MB
4_functor_chunk9                    0.1067s     130.1 MB
4_functor_chunk23(=l, full)         0.0822s     111.7 MB
```

Two findings, both checked against a common reference gradient (`@assert isapprox(...;
rtol=1e-12)` on every config, every point — no config's timing was recorded without first
confirming it reproduces the same answer):

- **ForwardDiff's automatic chunk selection is already optimal here** — `chunk1_autochunk`
  (0.0822s) is statistically indistinguishable from the best explicit chunk size (`chunk23`,
  the full l=23, 0.0822s), and manual explicit chunk sizes below l are strictly worse (chunk=1 is
  4.8x slower). **No manual chunk-size override is needed or beneficial for this problem size.**
- **Wall time is statistically tied across baseline/cached-config/functor** (0.080-0.084s, well
  within run-to-run noise for a single `@belapsed`-measured call) — the N=8000-draws evaluation
  inside `envelope_scalar_div_ctx` dominates wall time regardless of config. **Allocations are
  NOT tied**: cached `GradientConfig` + preallocated output roughly HALVES allocation (243.6MB →
  111.7MB, -54%) by avoiding re-materializing the dual-number seed/config machinery on every
  call. This matters cumulatively across a full outer solve's many repeated calls (GC pressure),
  even though it doesn't show up as a wall-time win in a single-call microbenchmark.
- Type stability: `@code_warntype` on `envelope_scalar_div_ctx` shows zero `Union{}`/`::Any`
  occurrences — fully concrete (`warntype_envelope.txt`).

**Recommendation**: cached `GradientConfig` (built once per outer solve, not per callback call)
+ preallocated gradient buffer + default (auto) chunk size — for the allocation reduction, not a
wall-time win that doesn't clearly exist at this problem size. A hand-picked chunk size is not
worth the added configuration complexity.

## 6. What share of a complete outer evaluation is spent on the gradient?

From `cache_benchmark.csv` (maxit=15 audit run, production's actual dense-Jacobian gradient
path): gradient time 1.10s vs. inner-solve time 0.18s in the uncached run (9 inner solves + 4
gradient computations) — **the dense-Jacobian gradient is ~86% of the combined inner-solve+
gradient cost at D=4**, confirming `ad_benchmark/`'s finding that the Jacobian, not the inner
solve, dominates once D grows (this was 6.33s vs 0.099s at D=10 in that session's numbers).
Caching does not change this ratio (it eliminates DUPLICATE inner solves, and — separately if
requested — duplicate gradient computations; it does not make one gradient computation itself
cheaper; that is `forwarddiff_benchmark.csv`'s question).

## 7. Which KNITRO callbacks are invoked at the same theta?

**Both**, under production's actual `eval_fcga=no` default: `callbackEval_and_ConsF_outer!` and
`callbackEval_and_ConsG_outer!` are called back-to-back at the exact same θ for every accepted
outer iterate (`callback_trace.csv`, rows 1-2, 3-4, 5-6, 8-9 — see `call_graph_audit.md` §6-7).

## 8. How many duplicate inner solves occurred before the cache improvement?

4 of 9 (44%) in the maxit=15 audit run; every accepted outer iterate pays for the inner CC solve
exactly twice (once from F, once from G). See `cache_benchmark.csv`.

## 9. How many inner solves and gradient calculations are avoided afterward?

Inner solves: 9 → 5 (exactly 1 per unique θ). Gradient calculations: 4 → 4 in this particular
short run (no repeat gradient REQUEST at an identical θ happened to occur in this trace — the
cache supports reusing a gradient too, via `grad_jac_set`, but this run didn't exercise that
path; see `cached_outer_loop.jl::ensure_grad!`).

## 10. Is there now approximately one inner solve per unique theta?

Yes — exactly 1.0 (5 inner solves / 5 unique θ), down from 1.8 (9/5) before caching.

## 11. Is the gradient computed lazily and at most once per theta?

Yes by construction (`ensure_grad!` checks `cache.grad_jac_set`, cleared only on a fresh inner
solve) — see `cache_design.md` §6C.

## 12. How effective are inner warm starts?

`warmstart_audit.jl`: warm-starting from the immediately-preceding (DIFFERENT) θ's solution cuts
mean inner KNITRO iterations by 30% (6.8→4.9) and wall time from 0.116s→0.025s per solve vs. cold
starts. But a duplicate solve AT THE IDENTICAL θ, even warm-started from its own just-found
solution, still costs a real re-solve — 0.019s and 1.6 KNITRO iterations on average, NOT zero.
**Warm start ≠ cache**: warm start makes a genuinely-new-θ solve cheaper; only an exact-point
cache makes a repeated-θ "solve" free. See `warmstart_benchmark.csv`.

## 13. What is the first operation causing reverse-mode NaNs?

Not fully identified, but substantially narrowed via 6 bisection steps (see
`reverse_nan_bisection.md`). RULED OUT, with real production data, exact match to ForwardDiff,
zero NaN under Enzyme (and Mooncake where tested): `sum(;dims=k)` reductions, the two-way
demeaning broadcast (`withinTransform`), the dead mutating `doubleDiff`/`meanτ` preamble, and —
critically — **the full real `make_gravity_grad` gravity-gradient object itself** (step 6, real
data, real θ). This last result directly CONTRADICTS the prior session's claim that "the gravity
moment alone reproduces the NaN pattern" and that hard-max is "unrelated" — that claim could not
be reproduced here and should be treated as unverified. The likely remaining site, by
elimination, is `hFunction!`/`hFunctionCounter!` (the trade-share moments, which DO use
hard-max/`MinInd!`) and/or their interaction with the `gamma()` normalization or the N=8000-draw
aggregation — not yet isolated.

Separately (step 5), found a genuinely different, real Enzyme hazard: `moments!.jl`'s
`UPow_scratch` reuse (gated on `eltype(γ)===Float64`, meant to distinguish "not being
differentiated" from "being ForwardDiff'd" but which is ALSO true under Enzyme, since Enzyme
doesn't change θ's element type) throws an explicit `EnzymeRuntimeActivityError` on a minimal
analogue, and Enzyme's own suggested workaround (`set_runtime_activity`) silently returns a WRONG
gradient (0.0 instead of 5.69) rather than fixing it. This branch is not exercised by the
diagnostics-only "ts" variant that produced the originally-observed NaN, so it's a separate,
additional landmine, not the explanation for what's already been seen.

## 14. Is there a compact Enzyme/Mooncake reproducer suitable for an upstream issue?

Not yet — this session's reproducers (`minimal_enzyme_nan.jl`, `minimal_mooncake_nan.jl`,
`minimal_enzyme_nan_step4.jl`) are compact and standalone but all show CORRECT behavior, so they
are negative results (useful for narrowing) rather than a failing reproducer to file upstream.
The prior session's finding that the digamma/gamma JIT-link failure matches a confirmed-open
upstream issue (Enzyme.jl ecosystem gap on Julia 1.12+) remains the one item with a citable
upstream match; the NaN-gradient issue itself still lacks a minimal failing case.

## 15. Should reverse mode be pursued now or deferred?

**Deferred** — unchanged conclusion from the prior session, reconfirmed by this session's own
numbers: production's dense-Jacobian gradient (not the inner solve) is the actual bottleneck
today (§6 above), and Method B (already correct, now further optimized — see §5) already fixes
that without needing reverse mode at all. See `reverse_nan_bisection.md`'s final section.

## 16. What is the measured end-to-end wall-time improvement?

Two data points, with an honesty caveat on the second:

- **maxit=15 audit** (`cache_benchmark.csv`, clean isolated process, JIT-warm-up-controlled
  per-path before timing): **1.29s → 0.95s, 1.36x**, driven entirely by the 4/9 duplicate inner
  solves avoided (44%). This is the reliable number.
- **maxit=25 integration test** (`outer_solver_comparison.csv`): counts reproduce the maxit=15
  finding exactly (9 callbacks, 5 unique θ, 9→5 inner solves, same κ/opt_err/status — the solver
  converges before hitting either iteration cap, so maxit=15 was already representative). BUT its
  own wall-time numbers (A=1.23s, B=2.19s, apparently a SLOWDOWN) are **not trustworthy**: this
  run was launched while `benchmark_forwarddiff.jl` was still running as a separate concurrent
  process on the same machine, competing for CPU — and unlike the maxit=15 audit, config B's
  single `time()` measurement landed during a period of contention. The COUNT-based evidence
  (inner solves avoided, cache hits, bit-identical κ/opt_err) is unaffected by this and remains
  fully valid; only the wall-clock number from this specific run should be discarded. **Use the
  maxit=15 number (1.36x) as the reliable timing figure**; a clean maxit=25 timing rerun (isolated
  process) is a cheap follow-up but was not repeated here given the count-based result was already
  unambiguous.

## 17. Did any output, derivative, or exact feasibility condition change?

**No.** `κ` matched to `0.912028119649117` (15 significant digits) between the cached and
uncached maxit=15 runs; `opt_err` matched to `0.005627205959304589` exactly. Caching changes only
which work is repeated, never the numerical result — see `cache_design.md`'s "Correctness
guarantee" section for why this is true by construction (the cache never computes anything new;
it only skips re-computing something already computed at an identical point).

## File map

- `cached_outer_loop.jl` — `OuterEvalCache`, instrumented+cached KNITRO callbacks,
  `outer_loop_instrumented` (parallel to `cc_algo/outer_loop_functions.jl::outer_loop`).
- `audit_outer_callbacks.jl` — the D=4, maxit=15, production-`eval_fcga=no` audit run (uncached
  vs cached). Outputs: `callback_trace.csv`, `cache_benchmark.csv`.
- `warmstart_audit.jl` — cold vs warm vs repeat-at-identical-θ benchmark. Output:
  `warmstart_benchmark.csv`.
- `benchmark_forwarddiff.jl` — ForwardDiff configuration/chunk-size sweep + type-stability check.
  Outputs: `forwarddiff_benchmark.csv`, `warntype_envelope.txt`.
- `outer_solver_comparison.jl` — maxit=25 integration test (config A: current: vs B: cached).
  Output: `outer_solver_comparison.csv`.
- `minimal_enzyme_nan.jl`, `minimal_mooncake_nan.jl`, `minimal_enzyme_nan_step4.jl` — standalone
  reverse-mode NaN bisection reproducers (steps 1-4, all negative/ruled-out so far).
- `call_graph_audit.md`, `cache_design.md`, `envelope_formula.md`, `reverse_nan_bisection.md` —
  write-ups.
- `ek_outer_loop_options_audit.opt`/`_audit25.opt`/`_warmup.opt` — diagnostics-only copies of
  PRODUCTION's actual `../../ek_outer_loop_options.opt` (`eval_fcga=no` preserved exactly; only
  `maxit`/`maxtime_real` bounded so audit runs finish in reasonable time).
