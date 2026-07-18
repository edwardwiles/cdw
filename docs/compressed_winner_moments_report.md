# Compressed winner-form factual moments — investigation report

Branch `diag/fullA-d4-exact` (worktree checkout at `bb74649`). All work is **additive** in new
files under `full_aod_diag/d4_exact/`; no production code, the dense constructor, the tie-breaking
rule, `lfix_incremental.jl`, or `composite_gradient_fast.jl` were modified.

**New files**
- `compressed_moments.jl` — compressed winner-form representation + fixed-dual scalar contraction + dense-materialization mode.
- `compressed_cc_inner.jl` — compressed CC-inner dual bundle (objective, gradient, exact Hessian-vector product).
- `test_compressed_moments.jl`, `test_compressed_cc_inner.jl` — strict equivalence tests.
- `benchmark_compressed.jl` — dense component profile + dense-vs-compressed (D,W) grid.

## Core verdict

**Yes — compression works, and it is equivalence-tested to machine precision.** The factual bilateral
moment matrix has exactly one nonzero winner entry per (draw, destination) plus a draw-independent
centering vector, so the whole fixed-dual pipeline (moment build, dual contraction, CC-inner
objective/gradient/HVP) can be evaluated in **O(W·D)** instead of the dense **O(W·D²)**, never
materializing the W×D² matrix. Every compressed quantity matches the trusted dense/production path to
~1e-13 or better across candidate points, random feasible points, all dual vectors tested, and D∈{4,6,8,10}.

## 1. Verified exact formula and signs (read from code, not assumed)

Active constructor for this investigation: `EK_moments_gammanorm_directgp!`
(`full_aod_diag/moments_gammanorm.jl`), autarky (`counterType==1`), `UoModel==1`, filling `G` via
`hFunction!` (`moments/hFunction.jl`), winner from `MinInd!` (`misc/smoothMinIndNew!.jl`).

For each draw `s`, destination `d`, and origin `o`, the **raw** bilateral moment (column
`d1 = d + (o-1)·D`) is:

```
r_{s,(o,d)} = pTσ_{s,o,d} · 1{o = winner_{s,d}}  −  P_{(o,d)} · denom_d
   winner_{s,d} = argmin_o  price_{s,o,d},   price = constCons_{o,d} / U_{s,o}^{−μ}
   pTσ_{s,o,d}  = constConsσ_{o,d} / Uσ_{s,o}^{−μ}            ← v_{s,d}, the "winning CES value"
   constCons_{o,d}  = wHat_o · AodPow_{o,d} · τ_{o,d}
   constConsσ_{o,d} = wHat_o^{1−σ} · (AodPow_{o,d}·τ_{o,d})^{1−σ}
   denom_d = γ_d^σ · wHat_d · L_d = wHat_d · L_d              (γ_d ≡ 1 normalization)
   P = observed bilateral shares λ̂  (ctx.γ.P)
```

Post-processing applied afterward, in order: (1) divide cols `1..D²+1` by `gammafac = Γ(μ(1−σ)+1)`;
(2) if `usePMM`: subtract `PMM_j`; (3) if `NormalizeMoments`: multiply by `1/σ_Moments_j`; (4) multiply
row `s` by `SamplingWeights_s`. So the fully-processed moment is
`G_{s,j} = SW_s · nrm_j · (r_{s,j}·gdiv_j − usePMM·PMM_j)` with `gdiv_j = 1/gammafac` for `j ≤ D²+1`.
This ctx's config: `usePMM=0, NormalizeMoments=0, σ_Moments≡1, SW≡1, gammafac=1.2254`, `oci=18`
(dual over cols 1..17 = 16 bilateral + 1 counterfactual price index; the gravity moment, col 18, is the
outer constraint and is **not** in the inner dual). The module implements the **general** form
(SW/nrm/PMM/gdiv) so it is provably correct beyond this specific config.

**Correction to the schematic in the task brief.** The brief proposed
`G_{·d,s} = v_{sd}(e_{w} − λ̂_{·d})`, i.e. a centering term scaled by the per-draw winner value `v_{sd}`.
The code's centering term `−P_{od}·denom_d` is **draw-independent** — it does *not* multiply `v_{sd}`.
Hence the exact fixed-dual contraction is

```
Σ_{o,d} β_{od} G_{s,(o,d)} = SW_s · [ Σ_d κ_{win_sd,d} · v_{s,d}  +  Σ_d C_d  −  usePMM·⟨β, nrm·PMM⟩ ]
    κ_{o,d} = β_{o,d}·nrm_{o,d}·gdiv_{o,d},   C_d = −denom_d · Σ_o κ_{o,d} · P_{o,d}
```

i.e. a per-draw sum over destinations (pick the winner's dual) plus **draw-independent constants** — even
cleaner than the brief's version. This matches, and re-derives independently, the `contrib0` collapse
already used and self-validated in `lfix_incremental.jl::build_lfix_base_cache`
(`contrib[s,d] = (SW/gammafac)(CONST_d + λ*_{d1(win)}·pTσ)`), which was the key prior art for this task.

## 2. Does the current dense constructor already exploit one-winner structure? — Partially

`hFunction!` **fuses** winner selection (`MinInd!`) into the same draw loop that accumulates `G`, and
under autarky its counterfactual branch is a single vectorized broadcast — so it is not naive. But it
still **fills the full W×D² matrix explicitly**: it computes the σ-value `pTσ` for **all D origins**
of each destination, writes **D columns per destination per draw** (losers written as
`−P_{od}·denom_d`), and allocates/zeroes the dense output. The compression removes exactly these:
evaluates the CES σ-value for the **winner only** (D× fewer), stores `winner[s,d]`+`v[s,d]` (O(W·D)),
and never allocates or writes the W×D² matrix.

**Dense component breakdown** (D=4, W=8000, median of 50 reps, 20 threads):

| phase | ms |
|---|---|
| score construction (`constCons`/`constConsσ`, O(D²)) | 0.0048 |
| `U^{−μ}` / `Uσ^{−μ}` power transform (O(W·D)) | **1.787** |
| hard winner search over D origins (O(W·D²)) | 0.147 |
| allocate + zero the dense W×d output matrix | 0.121 |

The `U^{−μ}` power transform dominates a single build (this is the same finding the earlier
`MuSigmaPowCache` audit exploited — it is `μ`-invariant and cacheable). The compressed path shares that
same power transform but then avoids the per-origin σ-value work, the dense write, and the allocation.

## 3. Equivalence tests (all PASS, machine precision)

`test_compressed_moments.jl` (candidate + random feasible points; infeasible-base points skipped exactly
as `test_lfix_incremental.jl` skips them):

| check | max abs error |
|---|---|
| dense materialization vs `obj.moments!` `G[:,1:oci−1]` | **1.8e-15** |
| fixed-dual contraction vs `G·β` (β ∈ {λ*, random normal, random positive, ones, unit columns}) | **1.7e-13** |
| `q = −ζ* − λ*'G` vs the `fixed_dual_L` path | **1.5e-13** |
| `L_fix` scalar vs trusted `fixed_dual_L` | **3.3e-15** |
| injected exact price tie | correctly throws `TiedWinnerError` (reused prior art) |

`test_compressed_cc_inner.jl` (compressed CC-inner bundle vs production `PsiObjectiveBundleImplicit` at
converged base duals):

| check | max abs error |
|---|---|
| dual objective `f` | **1.7e-15** |
| gradient wrt ζ | **1.8e-15** |
| gradient wrt λ (17 components) | **2.8e-15** |
| Hessian-vector product vs central-FD of the analytic gradient (3 random directions) | **4.8e-10** (FD-limited) |

The ~1e-13 (vs bit-exact 0) on the contraction is pure floating-point **re-association**: the compressed
form sums `Σ_d κ_win v + const` in a different order than the dense `Σ_j β_j G_{s,j}`. Winner-finding
itself is computed with the identical `constCons / U^{−μ}` division form as `MinInd!`, so the **winner
index and the exact-tie boundary match the dense path bit-for-bit** (important for tie detection).

## 4. Tie handling (reused prior art, built in from the start)

Per the handoff's explicit warning, the one-winner assumption breaks on exact price ties (1-in-32000 at
one historically-tested decoupled point). `build_compressed_factual` counts, per (draw,destination), how
many origins sit at the row-min using the same `price ≤ min` convention `MinInd!` uses, and throws the
existing `TiedWinnerError` (from `lfix_incremental.jl`) when any tie is found — the same
detection/fallback contract callers already handle. The injected-tie test confirms it fires
(`n_tied_pairs=1`, example `(7,1)`). Clean points never trigger it. A production caller should catch
`TiedWinnerError` and fall back to the dense rebuild for that one outer point (correctness over speed),
exactly as `composite_gradient_at_fast` already does.

## 5. Compressed CC-inner bundle (objective / gradient / exact HVP)

Confirmed from `PsiObjectiveBundleImplicit`'s body and `inner_loop_internal` (which sets
`H[:,1]=K`, `H[:,2]=1`, `H[:,3:2+d]=G`): the dual objective is `f = mean_s Ψ(q_s) + ζ`,
`q_s = −ζ − λ'G_{s,1:oci−1}`, with `g_ζ = 1 − mean_s Ψ'(q_s)` and `g_λ = −(1/M) Σ_s Ψ'(q_s) G_{s,·}`.

- **Objective / q**: `compressed_dual_contraction` gives `λ'G_s` in O(W·D); then Ψ elementwise.
- **Dual gradient** `g_λ`: the "transpose contraction" `Σ_s w_s G_{s,j}` reduces to **one O(W·D) pass**
  accumulating winner-origin bucket totals `B[o,d] = Σ_{s:win_sd=o} SW_s w_s v_{s,d}` and a global total
  `T = Σ_s SW_s w_s`, then an O(D²) per-column correction
  `nrm·gdiv·(B[o,d] − P_od·denom_d·T) − nrm·usePMM·PMM·T`. This is exactly the "winner-origin buckets +
  destination totals + target-share correction once per destination" the brief describes.
- **Exact HVP**: with `u_s = p_ζ + p_λ'G_s` (another forward contraction) and `r_s = Ψ''(q_s)·u_s`,
  `(Hp)_ζ = mean_s r_s` and `(Hp)_λ = (1/M)Σ_s r_s G_{s,·}` — the **same** transpose primitive with
  weights `r` instead of `Ψ'`. So exact Hessian-vector products are available directly from the
  compressed moments (verified to 5e-10 vs FD), with **no dense Hessian and no dense G**.

## 6. Dense vs compressed benchmark (warmed, median, 20 threads)

Times in ms; ratio = dense/compressed (>1 means compressed faster). "build" = full moment construction
(`obj.moments!` vs `build_compressed_factual`); "contract" = fixed-dual `λ'G_s` for all s; "transp" =
dual-gradient `G'w`; "CC v+g" = end-to-end objective+gradient from scratch (build + contraction + Ψ +
dual gradient).

| D | W | build dense→comp (×) | contract (×) | transp (×) | CC v+g (×) | build allocs (× fewer) |
|---|---|---|---|---|---|---|
| 4 | 8000 | 4.62 → 2.09 (**2.20×**) | 0.138 → 0.062 (2.2×) | 0.127 → 0.065 (1.9×) | 5.05 → 2.35 (**2.15×**) | 4123 → 1190 KB (3.5×) |
| 4 | 80000 | 49.1 → 21.9 (**2.25×**) | 1.176 → 0.464 (2.5×) | 0.138 → 0.453 (0.3×) | 53.9 → 23.3 (**2.31×**) | 40124 → 11878 KB (3.4×) |
| 6 | 8000 | 11.0 → 3.40 (**3.24×**) | 0.378 → 0.090 (4.2×) | 0.192 → 0.070 (2.7×) | 11.9 → 3.48 (**3.41×**) | 8154 → 1692 KB (4.8×) |
| 8 | 8000 | 18.4 → 4.41 (**4.17×**) | 1.038 → 0.096 (10.9×) | 0.149 → 0.094 (1.6×) | 21.0 → 4.79 (**4.37×**) | 13689 → 2194 KB (6.2×) |
| 10 | 8000 | 28.7 → 5.75 (**4.99×**) | 2.688 → 0.165 (16.3×) | 0.074 → 0.109 (0.7×) | 30.9 → 5.96 (**5.19×**) | 20745 → 2697 KB (7.7×) |

**Reading the table.** The build and the forward contraction show the predicted `O(W·D²)→O(W·D)`
scaling: the compressed advantage **grows with D** (build 2.2×→5.0×, contraction 2.2×→16.3×), while the
memory/allocation advantage grows 3.5×→7.7×. The end-to-end CC value+gradient is **2.15×–5.19× faster**.
The isolated **transpose** row is the one place compression does *not* win: `G'w` on an
already-materialized dense matrix is a multithreaded BLAS `gemv` (memory-bandwidth bound and extremely
fast), whereas the compressed transpose is a scalar bucket loop with scattered writes — competitive but
sometimes slower (0.3×–2.7×, noisy). The correct interpretation: **the compressed transpose's value is
that it never requires the dense matrix to exist in the first place** — its cost must be judged as part
of the end-to-end "CC v+g" column (which includes the build it eliminates), where compression wins
throughout.

Not benchmarked here (require the production KNITRO inner solver, which was intentionally **not**
rewritten until equivalence passed — it now does): complete warm/cold inner-solve time and end-to-end
short outer-run time. These are the natural next step once the compressed bundle is wired into the inner
loop (see recommendation).

## 7. Threading note

`build_compressed_factual`'s winner loop is currently single-threaded (the dense `obj.moments!` it is
compared against uses production's existing draw-chunk `Threads.@threads`, so the reported build ratios
are conservative — a threaded compressed build would widen the gap). Per the brief and
`docs/fullA_moment_construction_audit.md` §5, draw-level threading is the right axis and it must **not**
be nested inside coordinate-level `L_fix` parallelism (`composite_gradient_at_fast`'s
`Threads.@threads for k`). The compressed contraction/transpose primitives are used at the base-state
solve (outside any coordinate-parallel region), so no nesting risk in the current wiring; a threaded
compressed build should follow the same single-threaded-when-called-from-a-parallel-region discipline.

## 8. Recommendation

**Move it toward production, in this order:**

1. **Wire `compressed_cc_inner` into the fixed-dual `L_fix` / inner-dual evaluation** behind a mode flag,
   keeping the dense path as the default and the `materialize_dense_factual` mode for verification. Gate
   on `TiedWinnerError` → dense fallback (the contract already exists). This is where the 2.2×–5.2×
   end-to-end and 3.5×–7.7× allocation wins are realized on real evaluations, and it directly attacks the
   `inner_moment_build`-dominated cost identified in the earlier warmed profile.
2. **Benchmark the wired warm/cold inner-solve and a short outer run** to convert these component numbers
   into a realized per-iterate saving (the two rows this report could not measure without rewriting the
   solver).
3. **Thread the compressed build** over draw chunks (single-threaded when called from a coordinate-parallel
   region) for a further build-side gain, especially at D≥6 where more destinations give more independent
   work.

The compressed representation is the natural D-scaling lever for the full-A exact estimand: its advantage
is smallest exactly where the problem is cheapest (D=4) and grows with D, precisely the regime
(`gated D-scaling redo`) the continuation-6 handoff flags as the next scientific frontier.
