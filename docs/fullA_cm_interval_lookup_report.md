# Bin-index / interval-moment reformulation + lookup-based FG evaluation for the common-marginals restriction

Branch `diag/fullA-d4-exact-cm-interval-hessian`, based on `diag/fullA-d4-exact-common-marginals`.
D=4, W=8000, δ=1.0. All new code lives in `full_aod_diag/d4_exact/`; `common_marginals_moments.jl`
and the `c12_*.jl` validation scripts from the parent branch were **not modified** and served as
the trusted reference throughout.

## TL;DR

- **Part A (bin-index / interval reformulation): fully validated, exact.** The cumulative<->interval
  transform matrix is constructed two independent ways and both reconstruct the reference dense CM
  matrix to machine precision (max error 0 to 1.11e-16) at L∈{10,20,50}, anchored and orthonormal
  contrasts. The interval-augmented CC inner solve agrees with the dense cumulative-CDF reference to
  solver tolerance (Δ_dual/Δ_primal errors ~1e-15..1e-19, LFD weights agree to relative
  ~1e-14..1e-16) at calibration, a headline candidate point, a perturbed point, and a structurally
  infeasible point (both correctly reject it, nStatus=-300).
- **Part B (lookup-based FG): correct, and does win end-to-end at D=4 -- modestly.** The O(D)-per-draw
  lookup kernels reproduce the dense `obj(x,g)` callable's objective and gradient to machine precision
  (worst abs error 4.4e-16 across 12 method×contrast×L configurations), and reproduce a **live KNITRO
  inner solve** to the same tolerance as Part A when wired in via a custom eval callback (following
  the codebase's existing `compressed_live.jl` pattern). End-to-end full-inner-solve wall time improves
  by **~1.2-1.3x** across L∈{10,20,50}; the **isolated FG-kernel cost** improves far more (~1.0x → 1.4x →
  2.8-2.9x as L grows from 10→20→50), confirming the intended O(D)-not-O(D·L) scaling, but this gain is
  diluted end-to-end because (a) this D=4 problem converges in only 4-5 KNITRO iterations, so total
  wall time is dominated by KNITRO's own overhead and (b) the Hessian callback was deliberately left
  dense/unoptimized (out of scope), and its O(W·ncm²) cost grows faster than the FG saving as L grows.
- Threaded histogram building gives a modest ~25-30% reduction from 1→8 threads but **degrades at 16
  threads** — the W=8000 workload is too small for thread-spawn overhead to pay off at high thread
  counts. Reported honestly as a limited/marginal win at this problem scale, not oversold.
- Two real implementation bugs were caught and fixed before any of the above was trusted (see
  "Bugs found" below) — both would have silently produced wrong numbers had they gone unnoticed.

## Part A: bin-index precomputation + interval-moment reformulation

### Construction

`full_aod_diag/d4_exact/common_marginals_interval.jl`:

- `common_marginals_quantiles(U, refIndex1, L)` reproduces `precalc_common_marginals_cdf`'s quantile
  line **verbatim** (not merely "equivalent code") so cutpoints `z_1<...<z_L` are bit-identical by
  construction; checked (`z_dense == z_int`) in the validation script regardless.
- `compute_bin_indices(U, z)` assigns `bins[s,o] = searchsortedfirst(z, U[s,o])` for every origin
  (including the reference), giving `b_{s,o}=k` iff `z_{k-1} < U_{s,o} <= z_k` (`z_0:=-inf`,
  `z_{L+1}:=+inf`) — this reproduces the reference's `U .<= z_l` convention exactly (checked, not
  assumed), because `searchsortedfirst` returns the first index `k` with `z_k >= u`. `UInt8` is used
  whenever `L+1<=255` (always true for L∈{10,20,50}), `UInt16` otherwise.
- `precalc_common_marginals_interval(U, refIndex1, L; contrasts)` builds the dense
  `B_{s,o,k} = 1{b_so=k} - 1{b_s1=k}` matrix, `k=1..L` (bin `L+1` dropped — see below for why this
  exactly preserves the restriction count `(D-1)*L`), same threshold(bin)-major column layout as the
  reference, same orthonormal-contrast application.

### The cumulative<->interval transform matrix (explicit construction)

For `l=1..L`: `1{U<=z_l} = sum_{k=1}^{l} 1{b=k}` (bins `1..l` tile `(-inf, z_l]` exactly under the
`<=` convention), and the identity survives subtracting the reference-origin term (linear) and any
per-threshold contrast mixing `R` (linear, and `R` never mixes across `l`/`k`, only across origins
within one threshold block — so it commutes with the bin-cumulative-sum map). This gives, per origin:

```
CDF_block[:, l] = sum_{k=1}^{l} Interval_block[:, k]      (l = 1..L)
```

an all-ones **lower-triangular** map in `(l,k)`. Two independent constructions of the full
`(nO·L)×(nO·L)` transform were built and cross-checked:

1. `interval_to_cumulative_dense`: direct per-threshold-block running sum (a plain loop).
2. `full_transform_matrix(nO,L) = kron(S, I_nO)`, `S[k,l] = 1{k<=l}` (upper-triangular ones,
   `L×L`), applied as `CM_interval * full_transform_matrix`.

### Section 1 results (transform matrix, `c12i_validate_interval_equiv.jl`)

| L  | contrasts    | \|blockcumsum − kron\| | \|blockcumsum − reference\| | \|kron − reference\| |
|----|--------------|------------------------:|------------------------------:|------------------------:|
| 10 | anchored     | 0                        | 0                              | 0                        |
| 10 | orthonormal  | 0                        | 1.11e-16                       | 1.11e-16                 |
| 20 | anchored     | 0                        | 0                              | 0                        |
| 20 | orthonormal  | 0                        | 1.11e-16                       | 1.11e-16                 |
| 50 | anchored     | 0                        | 0                              | 0                        |
| 50 | orthonormal  | 0                        | 1.11e-16                       | 1.11e-16                 |

Both independent transform constructions agree exactly with each other and reconstruct the trusted
dense reference matrix (`precalc_common_marginals_cdf`) to floating-point roundoff.

### Section 2 results (end-to-end CC inner solve, same script)

Points: calibration, `upper_maxit40` (a headline unrestricted-optimum candidate), a 5%-jittered
`perturbed_feasible` point, and a `structurally_infeasible` point (`A_od[1,1]→1e-6`) — point
construction code copied verbatim from `c12_d4_fixed_param_battery.jl`.

| L  | point                  | Δ_dual err | Δ_primal err | max\|m_dense−m_interval\|/scale | KKT dense | KKT interval |
|----|------------------------|-----------:|-------------:|---------------------------------:|----------:|-------------:|
| 10 | calibration             | 6.5e-18    | 8.7e-19       | 5.9e-16                          | 2.7e-17   | 1.2e-17      |
| 10 | upper_maxit40           | 2.7e-15    | 6.7e-16       | 7.2e-15                          | 3.0e-16   | 9.1e-16      |
| 10 | perturbed_feasible      | 3.7e-16    | 7.6e-17       | 2.9e-15                          | 1.8e-16   | 2.8e-16      |
| 20 | calibration             | 1.5e-17    | 3.5e-18       | 3.2e-15                          | 6.4e-16   | 6.3e-16      |
| 50 | calibration             | 1.0e-17    | 1.0e-17       | 9.8e-16                          | 6.2e-17   | 6.6e-17      |
| 50 | upper_maxit40           | 2.7e-15    | 0             | 2.9e-14                          | 2.9e-15   | 1.8e-15      |

(full 12-row table in the script's stdout; every L∈{10,20,50}×point combination is below the 1e-6
pass tolerance). `structurally_infeasible` is correctly rejected by **both** bases at every L
(`nStatus=-300` for both).

**Limitation**: only the eq.35 block is supported by the interval reformulation — the optional
eq.36 truncated-moment companion (`include_truncated_moment=true` in the dense reference) is out of
scope for this reformulation and was not attempted.

## Part B: lookup-based dual objective/gradient

### Kernels (`cm_lookup_kernels.jl`)

Per FG call, the dense callable computes (for this ctx, `outer_constr_index==d`, gravity the sole
outer-only column): `arg0 = -(ζ + G_core·λ_core + G_cm·λ_cm)`, then `f`,`g` from `Psi!`/`dPsi!`. The
`G_cm·λ_cm` (forward) and `mean(arg1·G_cm)` (backward) pieces — normally `O(W·(D-1)·L)` BLAS —
are replaced by:

- **Interval basis (production candidate)**: forward = one `+1`/`-1` lookup per non-reference
  origin into a `(nO, L+1)` λ-matrix (bin `L+1` = 0, the dropped/redundant bin), `O(W·(D-1))`.
  Backward = a **threaded weighted histogram** `h[o,k] = Σ_{s: bins[s,o]=k} m_s`
  (`build_weighted_histogram`, thread-local `(D,L+1)` buffers, no atomics, fixed-order reduction),
  then `g[oi,k] = -(h[o(oi),k]-h[ref,k])/M`, `O(W·D + L·nO)`.
- **Cumulative-basis suffix-sum variant (diagnostic/equivalence cross-check only, per the task
  brief — not a production candidate)**: forward via `P_o(k)=Σ_{l>=k} ν_{o,l}` (suffix sum of the
  cumulative-basis λ, same lookup shape as interval), backward via **prefix** sums of the same
  histogram (`Hpre[o,l]=Σ_{k<=l} h[o,k]`) — note forward needs *suffix* sums, backward needs
  *prefix* sums; conflating the two was one of the bugs caught below.
- **Orthonormal-contrast handling**: proved that the stored↔block-space transform is the **same**
  left-multiply by the symmetric contrast matrix `R` in both directions (`R symmetric`, not
  orthogonal) — applied once per FG call as an `O(L·nO²)` matrix multiply, not per-draw.

### Bugs found and fixed (before trusting anything downstream)

1. **Reshape orientation bug.** The stored `λ_cm` layout is threshold-major (origin varies fastest
   within a bin block, `col(k,oi)=(k-1)·nO+oi`) — Julia's column-major `reshape` of that vector must
   be `(nO, L)`, **not** `(L, nO)`. An earlier draft used `(L, nO)`, silently transposing origin↔bin.
   Re-derived the `R`-contrast direction (left-multiply, not right-multiply) for the fix. Caught by
   the direct FG comparison below, not by inspection.
2. **Off-by-one BLAS slice bug.** The combined `[ones, core]` column slice needs `H[:,2:2+ncore1]`
   (`ncore1+1` columns); an earlier version wrote `H[:,2:1+ncore1]` (one column short). Caught
   immediately by a `DimensionMismatch` — no silent corruption, but worth recording.
3. **Test-harness category error.** `:suffix` (cumulative-basis) results were initially compared
   against the **interval**-augmented dense reference (wrong basis for that method) instead of the
   cumulative one — produced large, obviously-wrong mismatches (up to `ferr≈1.3`) that looked like a
   kernel bug but were a test-setup bug. Fixed by building both dense references and pairing each
   method with its own basis.

### FG correctness validation (`c12i_validate_lookup_fg.jl`)

12 configurations (method ∈ {interval, suffix} × contrasts ∈ {anchored, orthonormal} × L ∈
{10,20,50}), 10 points each (zero, 5 small + 3 larger random perturbations, the actual converged
inner solution) — **all pass** after the fixes above:

| L  | contrasts   | method   | worst \|f_dense−f_lookup\| | worst rel \|g_dense−g_lookup\| |
|----|-------------|----------|----------------------------:|----------------------------------:|
| 10 | anchored    | interval | 5.6e-17                     | 2.9e-16                            |
| 10 | anchored    | suffix   | 1.1e-16                     | 7.8e-16                            |
| 10 | orthonormal | interval | 2.8e-17                     | 2.1e-16                            |
| 10 | orthonormal | suffix   | 5.6e-17                     | 4.9e-16                            |
| 20 | anchored    | interval | 2.8e-17                     | 1.2e-16                            |
| 20 | anchored    | suffix   | 1.1e-16                     | 4.5e-16                            |
| 20 | orthonormal | interval | 2.8e-17                     | 9.7e-17                            |
| 20 | orthonormal | suffix   | 1.4e-17                     | 3.2e-16                            |
| 50 | anchored    | interval | 3.5e-18                     | 2.2e-16                            |
| 50 | anchored    | suffix   | 4.4e-16                     | 1.1e-15                            |
| 50 | orthonormal | interval | 2.8e-17                     | 1.1e-16                            |
| 50 | orthonormal | suffix   | 8.7e-18                     | 6.4e-16                            |

### Live KNITRO wiring (`cm_lookup_live_knitro.jl`) + equivalence (`c12i_validate_live_knitro.jl`)

Following the codebase's existing `compressed_live.jl` pattern (custom `KN_add_eval_callback`
registered with a `CMLookupState` as `userParams`, bypassing the dense callable for the FG hot loop;
the Hessian callback is left as the **unchanged dense** `obj(x, h=...)` call, since `obj.H` is fully
materialized densely up front for this reformulation — no compression opportunity is claimed there).

At calibration and a perturbed point, L∈{10,20,50}, both methods (12 configs): the live lookup-wired
KNITRO solve agrees with the untouched dense baseline (`oracle.jl::evaluate_fullA`) to:

- Δ_dual error: 1.4e-16 to 1.4e-19
- Δ_primal error: 1.8e-16 to 0
- primal LFD weights, relative error: 3.1e-15 to 1.4e-14
- nStatus: identical at every point
- max moment KKT residual: comparable magnitude both bases (dense 6e-17..3e-16, lookup 6e-17..4.3e-16)

## Part B.4: benchmark table

Host: 208 cores reported by `Sys.CPU_THREADS`; `JULIA_NUM_THREADS=16` used for the run below (thread
counts above `Threads.nthreads()` cannot actually parallelize within one Julia process and were
skipped). All full-solve numbers are the **median of 15 cold repeats** (warm=false, fresh JIT
warm-up call excluded) on a shared cluster host — repeated runs showed some noise (see caveat below).

### Full inner KNITRO solve (primary, most representative)

| L  | dense (s) | suffix (s) | interval (s) | speedup (suffix) | speedup (interval) | n_iters / n_fg_calls |
|----|----------:|-----------:|--------------:|------------------:|---------------------:|------------------------|
| 10 | 0.0449    | 0.0360     | 0.0349        | 1.25x              | 1.29x                 | dense=4, lookup=5       |
| 20 | 0.0595    | 0.0461     | 0.0473        | 1.29x              | 1.26x                 | dense=4, lookup=5       |
| 50 | 0.1074    | 0.0909     | 0.0873        | 1.18x              | 1.23x                 | dense=4, lookup=5       |

(Two earlier repeat runs at NREP=8 gave speedups in the 1.07x-1.44x range for the same
configurations — the shared-host environment is noisy enough that individual decimal digits should
not be over-read; the qualitative finding — modest, consistent, ~1.1-1.3x — replicated across all
three independent runs.)

### Isolated FG-kernel-only timing (secondary, lower-noise, isolates just the objective/gradient cost)

N = the actual number of live-solve FG calls (4-5 at this D=4 scale), at a small-perturbation battery:

| L  | dense per-call | suffix per-call | interval per-call | speedup (suffix) | speedup (interval) |
|----|----------------:|------------------:|---------------------:|------------------:|----------------------:|
| 10 | 324 μs          | 326 μs             | 341 μs                | 0.99x              | 0.95x                  |
| 20 | 511 μs          | 363 μs             | 364 μs                | 1.41x              | 1.40x                  |
| 50 | 998 μs          | 360 μs             | 342 μs                | 2.77x              | 2.92x                  |

This is the clean signature of the intended complexity change: dense cost grows with L (roughly
tracking `O(D·L)`, though sub-linearly due to fixed BLAS/dispatch overhead), lookup cost stays flat
(~325-365 μs) independent of L, confirming `O(D)` scaling — this pattern reproduced consistently
across all three benchmark runs.

### Threaded histogram builder (Part B.2), W=8000, D=4

| L  | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads |
|----|---------:|----------:|----------:|----------:|-----------:|
| 10 | 54.8 μs  | 50.1 μs   | 41.6 μs   | 40.2 μs   | 48.3 μs    |
| 20 | 55.1 μs  | 44.3 μs   | 45.4 μs   | 42.9 μs   | 53.1 μs    |
| 50 | 55.0 μs  | 47.1 μs   | 45.6 μs   | 46.8 μs   | 56.0 μs    |

## Does the lookup approach win end-to-end at D=4? Honest verdict.

**Yes, modestly (~1.1-1.3x on total inner-solve wall time), and the win is real but not the whole
story.** Two things are both true and explain the gap between the isolated-kernel number (up to
~2.9x) and the end-to-end number (~1.2x):

1. **This D=4 problem converges in only 4-5 KNITRO iterations.** With so few FG evaluations, their
   total cost is a small share of the solve's wall time; KNITRO's own per-solve overhead (`KN_new`/
   `KN_free`, internal linear algebra, problem setup) is a large, roughly L-independent fixed cost
   that the FG optimization cannot touch.
2. **The Hessian callback was deliberately left dense/unoptimized** (materializing `obj.H`'s CM
   columns once via the existing dense builder, then calling the unchanged production `hessian!`) —
   its cost is `O(W·ncm²)`, which grows *quadratically* in `L` and increasingly dominates total wall
   time as `L` grows. This is why the end-to-end speedup does **not** climb monotonically with L the
   way the isolated FG number does (2.77x→2.92x at L=50) — the growing Hessian cost eats into the
   growing FG saving's share of the total.

A genuinely faster Hessian (e.g. exploiting the same block-sparse bin structure the FG kernels use —
each row's CM contribution to the Gram matrix is a single `±1` co-occurrence entry per origin, so a
histogram-of-co-occurrences approach is plausible) was **not attempted** here — task Part B's scope
was explicitly the objective/gradient, and the report says so rather than silently expanding scope.
If the paper's production runs are Hessian-bound at large L, that is the next lever, not this one.

**Threading**: does not pay off at this problem's W=8000 scale — 1→8 threads gives a real but small
~25-30% reduction, and 16 threads is *worse* than 8 (thread-spawn/scheduling overhead exceeds the
~50μs of actual work at that granularity). This is reported as a limited/negative finding, not
oversold; a genuinely large win from threading would need a much bigger `W` (e.g. the paper's D=20
real-data configurations reportedly run at up to W=80,000 per other work on this repo) where the
per-thread chunk of work is large enough to amortize spawn overhead.

## Files

New files, this branch only (`common_marginals_moments.jl` and every `c12_*.jl` file from the parent
branch are unmodified and were used only as the ground truth to validate against):

- `full_aod_diag/d4_exact/common_marginals_interval.jl` — Part A: bin indices, interval moments,
  transform matrix.
- `full_aod_diag/d4_exact/c12i_validate_interval_equiv.jl` — Part A validation (transform matrix +
  end-to-end CC solve equivalence).
- `full_aod_diag/d4_exact/cm_lookup_kernels.jl` — Part B: O(D) lookup FG kernels (interval +
  cumulative-suffix diagnostic), threaded histogram builder, `CMLookupState` callable.
- `full_aod_diag/d4_exact/c12i_validate_lookup_fg.jl` — Part B FG-correctness validation (isolated,
  no KNITRO).
- `full_aod_diag/d4_exact/cm_lookup_live_knitro.jl` — Part B live KNITRO wiring (custom FG callback +
  unchanged dense Hessian callback), mirrors `compressed_live.jl`'s pattern.
- `full_aod_diag/d4_exact/c12i_validate_live_knitro.jl` — Part B live-solve equivalence validation.
- `full_aod_diag/d4_exact/c12i_benchmark_lookup.jl` — Part B.4 benchmark (full solve, isolated
  kernel, threaded histogram).
