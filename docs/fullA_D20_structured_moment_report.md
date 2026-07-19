# Full-A D=20 structured (rank-one + winner-scatter) moment construction (Continuation 10, Part 2)

Branch `c10-chunked-hessian`, worktree
`/bbkinghome/edav/gravity_robustness/trade_robustness_modular/.claude/worktrees/agent-a60f519ea80642e1b`.
Measured on `demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, KNITRO 14.2.0. Real-data context via
`context_real_d20.jl::d20_real_setup` (France focal, calibration point).

## 1. Identity check: the task's schematic does NOT literally hold — corrected and verified

This task's brief posited `G_{.,d,s} = v_{s,d} * (e_{w_sd} - lambda_hat_{.,d})`.
This is **not exactly what the code computes** — and `compressed_moments.jl`'s
own header (written in a prior phase of this investigation) already found
and documented the same correction:

> "the centering term is DRAW-INDEPENDENT (`-P_{od}*denom_d`), NOT scaled by
> the per-draw winner value `v_{sd}`"

Verified independently in this task (not just re-reading the prior note) by
comparing `structured_dense_factual`'s output directly against
`obj.moments!`'s trusted dense output at D=4
(`c10_structured_moment_verify.jl`): the corrected formula matches to
1.78e-15 (machine-precision level), confirming it, not the literal brief
schematic.

**Corrected identity, with the full post-processing (`SamplingWeights`,
`NormalizeMoments`/`gdiv`, `usePMM`) folded in** (derivation in
`structured_moment_build.jl`'s header comment):

```
G[s,j] = SW[s]*a[j]*1{o(j)=w_{s,d(j)}}*v[s,d(j)]   -   SW[s]*FixedCol[j]

a[j]        = nrm[j]*gdiv[j]
FixedCol[j] = a[j]*Pmat[o(j),d(j)]*denom[d(j)] + usePMM*nrm[j]*PMM[j]
```

`FixedCol` is `O(D^2)`, θ-dependent but **draw-independent**. The second term
(`-SW[s]*FixedCol[j]`) genuinely IS a rank-one outer product — of `SW`
(length W or chunk) and `FixedCol` (length `D^2`) — matching the task's
"rank-one BLAS-friendly component" framing, just with `SW` (not `v`) as the
per-draw vector and `FixedCol` (not `lambda_hat` alone) as the fixed vector.
The first term is the genuine per-draw winner-scatter (irregular, one nonzero
per `(s,d)` pair).

## 2. Implementation

`full_aod_diag/d4_exact/structured_moment_build.jl`:

- `structured_coeffs(cf)` — precomputes `a`/`FixedCol` (`O(D^2)`, tiny) from
  an already-built `CompressedFactual` (`compressed_moments.jl`'s
  `build_compressed_factual` — winner search, `O(W*D^2)` irreducible, and tie
  detection reused UNCHANGED, never re-implemented).
- `structured_fill_chunk!(Gc, cf, a, FixedCol, rows; use_ger=true)` — fills a
  `length(rows) x (oci-1)` buffer (may be a chunk view or the full `W`
  matrix) via (1) the rank-one fixed term, `BLAS.ger!` or an equivalent
  broadcast (`Gc[:,1:D^2] .= .-SWc .* FixedCol'`), both implemented and
  benchmarked; (2) a tight scalar scatter-add loop for the per-draw winner
  term (irregular/indexed, NOT forced into a BLAS call, per this task's
  explicit guidance); (3) the counterfactual price-index column, a direct
  `O(n)` fill (no search needed, already computed in `cf.cf_raw`).
- `structured_dense_factual(cf; use_ger=true)` — full-`W` convenience
  wrapper for direct comparison against `materialize_dense_factual`.
- `fill_K_directgp!` — trivial `O(W)` fill of the objective column `K`
  (`K[s] = θ_full[3+D]*SW[s]`), needed so structured/compressed-front-end
  inner solves can populate `obj.H` without a second dense `moments!` call.

**Nothing in `moments_gammanorm.jl`, `compressed_moments.jl`, or
`compressed_live.jl` was modified.** All new code is additive.

## 3. Correctness, including tie conventions (`c10_structured_moment_verify.jl`, D=4)

| comparison | max\|diff\| | bit_identical |
|---|---|---|
| structured(ger!) vs dense (`obj.moments!`) | 1.78e-15 | false (machine-precision level) |
| structured(ger!) vs `materialize_dense_factual` | 1.78e-15 | false (same level) |
| `materialize_dense_factual` vs dense (pre-existing reference gap) | 1.78e-15 | false |
| structured(ger!) vs structured(broadcast) | 0.0 | **true** |

**Tie handling**: `structured_fill_chunk!` is built directly ON TOP of `cf`
(`build_compressed_factual`'s output) — it NEVER re-implements winner search
or tie detection, so any exact price tie throws `TiedWinnerError` (reusing
`lfix_incremental.jl`'s type) BEFORE the structured fill ever runs, identical
to `materialize_dense_factual!`'s own exposure. This is a structural
guarantee (both consumers of the same `cf`), not merely "should agree" —
demonstrated empirically anyway: a synthetic exact tie was forced (two
origins' prices made bit-identical at one `(draw,destination)` pair by
solving for the exact `U` ratio that equalizes `constCons[o,d]/UPow[s,o]`)
and confirmed to raise `TiedWinnerError` correctly. Real D=20 data has
`n_tied=0` throughout (consistent with this investigation's established
"ties are a probability-zero event for generic continuous draws" finding).

## 4. W=80,000 benchmark

### 4.1 Isolated moment-construction timing (`c10_structured_moment_bench_d20.jl`, median of 5 reps)

| construction | median (s) | speedup vs dense |
|---|---|---|
| (a) dense (`obj.moments!`, production baseline) | 1.193 | 1.00x |
| (b1) structured+`ger!`, incl. own winner search | 0.290 | 4.12x |
| (b1) structured+`ger!`, fill only (`cf` reused) | 0.076 | 15.68x |
| (b2) structured+broadcast, fill only (`cf` reused) | **0.053** | **22.51x** |
| (c) compressed+`materialize_dense_factual!`, incl. own winner search | 0.469 | 2.54x |
| (c) compressed+`materialize_dense_factual!`, fill only (`cf` reused) | 0.259 | 4.61x |

**The structured construction is a clear win over BOTH alternatives on
isolated construction time** — not just over the dense baseline (expected,
matches the established compressed-vs-dense trend) but also over the
EXISTING compressed+materialize path: fill-only, structured is **3.4-4.9x
faster than `materialize_dense_factual!`** (0.053-0.076s vs 0.259s), despite
both starting from the identical `cf` (same winner search, same inputs).
`materialize_dense_factual!` fills all `D^2+1` columns with one nested loop
per `(draw, destination, origin)` triple, re-deriving the same `r`/scale
computation inline every cell; the structured construction instead does the
`O(D^2)`-cell fixed term ONCE per chunk as a bulk vectorized
op (`BLAS.ger!`/broadcast) and touches only the `O(W*D)` winner cells
individually — genuinely less redundant work, not just a different
constant factor.

**`ger!` vs broadcast**: broadcast is consistently ~30% faster
(0.053s vs 0.076-0.079s, reproduced across 3 independent runs) for this
specific rank-one pattern (`SW` outer `FixedCol`, only `D^2=400` columns
wide) — plausibly because Julia's fused broadcast avoids `BLAS.ger!`'s
call overhead for a matrix this narrow, or because the broadcast can fuse
directly into the destination array without the (small) BLAS dispatch cost.
Both are exact (bit-identical to each other, §3), so this is purely a speed
choice, not a correctness one.

### 4.2 Complete cold inner-solve, each construction as front end (SAME dense-Hessian KNITRO solve, unchanged)

**JIT-order caveat, exactly as Part 1's report flags**: the naive
first-pass numbers (dense run strictly first in-process) showed inflated
"2.0-2.6x" speedups. Per this investigation's standing "verify before causal
claims" discipline, all four builders were re-run a second time in the SAME
process, now that every one of them is already JIT-compiled — this is the
trustworthy number:

| construction | cold wall, JIT-warm (s) | speedup vs dense |
|---|---|---|
| dense | 3.978 | 1.00x |
| structured+`ger!` | 3.003 | **1.325x** |
| structured+broadcast | 3.006 | **1.324x** |
| compressed+`materialize_dense_factual!` | 3.234 | 1.230x |

**A genuine, real (not JIT-artifact) full-inner-solve speedup, ~1.32x for
structured vs ~1.23x for the existing compressed+materialize path** — smaller
than the isolated moment-build speedup (4-23x) because the KNITRO dual solve
itself (5 Hessian calls, 6 FG calls — identical iteration counts across all
four builders, confirming none of this changes the actual optimization path)
dominates total wall time; moment-build is roughly 30% of the dense variant's
total (1.19s of 3.98s), so even a large moment-build speedup translates to a
more modest whole-solve speedup — an expected, Amdahl's-law-consistent
relationship, not a discrepancy. `n_fg`/`n_hess`/`status` identical across
all four builders in every run; dual solution `max|dx|` at machine noise
(7.2e-16 to 2.6e-16).

## 5. Verdict

- **Identity verified directly against the actual dense code** (not just
  re-deriving), and found to require a correction from the task's literal
  schematic — matching a correction a prior session already made in
  `compressed_moments.jl`'s own header. The corrected identity is confirmed
  to machine precision.
- **Tie conventions match exactly, by construction** (both structured and
  the existing `materialize_dense_factual!` consume the identical `cf`
  object with its identical tie-detection), confirmed with a real forced-tie
  test.
- **Structured construction is a clear win**: 4-23x faster than dense and
  3.4-4.9x faster than the EXISTING compressed+materialize path on isolated
  construction time; a genuine ~1.32x full-cold-inner-solve speedup once
  JIT-order noise is controlled for (vs ~1.23x for the existing
  compressed+materialize path) — a real, if more modest, win at the
  whole-solve level.
- Broadcast beats `BLAS.ger!` by ~30% for this narrow (`D^2`-wide) rank-one
  fill; both are exact and either is a reasonable implementation choice.
- **RECOMMENDATION (not applied — production default untouched)**: the
  structured construction is a strong candidate to replace
  `materialize_dense_factual!` as the dense-materialization step inside
  `compressed_live.jl`'s Hessian-callback adapter (where that lazy
  materialization currently costs ~262ms per the Phase 3.1 report, out of
  which the structured approach's fill-only cost of 0.053-0.076s at
  W=80,000 would be directly competitive or better) — this was NOT wired
  into any production path in this task (out of scope, additive-only per
  this task's instructions), but is flagged as a concrete, evidence-backed
  next step given the clean win demonstrated here.

## 6. Files

New, all under `full_aod_diag/d4_exact/`: `structured_moment_build.jl` (the
implementation), `c10_structured_moment_verify.jl` (D=4 identity + tie-handling
correctness), `c10_structured_moment_bench_d20.jl` (D=20/W=80,000 three-way
comparison, isolated + full-inner-solve, with a JIT-order control pass). No
existing file modified.

Raw logs: `results/fullA_d4/1b7f06b/c10_structured_moment_bench_d20/harness_log.txt`
(final run, includes the Part 3 JIT-order control; an earlier same-config run
also landed at `results/fullA_d4/690b8f5/c10_structured_moment_bench_d20/`
before this task's own Part 1 commits advanced `HEAD` — superseded by the
`1b7f06b` run, kept only because this task's own commits changed the
output-path-by-commit convention mid-task, not because the earlier numbers
disagree, and D=4 correctness logs are inline in this document/§3).
