# Note: a likely source of L=10 Hessian-callback speedup, and why version B makes it easier

**Additive to the version-B (fixed cutoffs + free masses) work already in progress — this changes
no math and no interface.** It is about how the restriction Hessian is *assembled* after the hard
part is already done. Nothing here should be acted on before reading §4 (measure first).

---

## 1. The accounting that motivates this

At D=20, L=10, W=100,000 the restriction block has `n_rows = (L-1)·D + (L-1)²·C(D,2)
= 180 + 81·190 = 15,570`, and the inner dual vector (with the ~400 economic core duals) is
`n ≈ 15,970`. Per Hessian callback:

| stage | entries |
|---|---|
| unique values actually computed (deduped T1–T4 tables: `C(20,4)=4845` 4-way tables × `9⁴`) | **~32M** |
| structurally nonzero entries of `H_RR` | ~98M |
| entries the code currently **touches** | 242M raw fill + 242M centering + 128M packed write = **~612M** |
| entries handed to KNITRO | 128M (~1 GB) |

The expensive, clever part is already done and already parallel: `G` is never formed, the
co-occupancy counts collapse into joint histograms via the tensor structure (a pair indicator is
the product of two marginal indicators), and the dedup exploits the fact that each unique 4-way
histogram serves three different pair-pair groupings — `(o₁o₂|o₃o₄)`, `(o₁o₃|o₂o₄)`,
`(o₁o₄|o₂o₃)`.

**The gap is downstream of that.** ~32M computed values become ~98M distinct answers, delivered via
~612M serial element touches. That is roughly 6× of avoidable traversal, and — see §3 — essentially
all of it is single-threaded.

## 2. The single most promising item: the centering pass is a rank-2 update done elementwise

`center_and_scale_pairwise_quantile_hessian!` (`pairwise_quantile_hessian.jl`) ends with:

```julia
@inbounds for J in 1:nrow
    tJ = t[J]; rJ = r[J]
    for I in 1:nrow
        HfullR[I, J] = (HfullR[I, J] - tJ * r[I] - t[I] * rJ + S * t[I] * tJ) * invM
    end
end
```

That is exactly, in matrix form,

```
H  =  ( T  −  r·t'  −  t·r'  +  S·t·t' ) / W
           └──── rank 2 ────┘  └ rank 1 ┘
```

with `t` the centering vector (`pairwise_quantile_target_vector!` — under version B, the μ-derived
targets) and `r = Ind'h`. So `nrow² = 242M` scalar read-modify-writes are applying a **rank-2 plus
rank-1 correction**.

Two independent ways to fix it, and I'd do the first:

1. **Thread it.** Every `J` column is independent — no reduction, no race, no ordering concern.
   `Threads.@threads :static for J in 1:nrow` is a two-line change and matches the discipline
   already used in `build_pairwise_quantile_hessian_tables!`. This is the low-risk, obviously-correct
   version.
2. **Make it BLAS-2**: `BLAS.syr2!('U', -1.0, r, t, H)` + `BLAS.syr!('U', S, t, H)` + a scale.
   Note this repo pins `OPENBLAS_NUM_THREADS=1` in the environment (drivers opt in explicitly via
   `BLAS.set_num_threads`), so treat BLAS as a vectorization win, not automatically a threading one.
   It also naturally touches only one triangle — which is *sufficient*, because
   `fill_pairwise_quantile_hessian_raw!` mirrors both triangles and the packed write reads only
   `j >= i`. That is a further ~2× on this block, but confirm the lower triangle genuinely has no
   reader before relying on it.

## 3. The other two serial blocks

Verified by grep — the only `Threads.@threads` in the whole restriction are in
`build_pairwise_quantile_hessian_tables!` (the T1–T4 scatter) and
`build_pairwise_quantile_tables_threaded!`. Not threaded:

- `fill_pairwise_quantile_hessian_raw!` — the 242M-entry expansion of tables into the dense block.
  Parallel over output columns.
- the final packed write in `pairwisequantile_hess_cb_builder` (`pairwise_quantile_production.jl`) —
  128M entries with a per-entry branch selecting the source block (`hee_packed` / `HEQ` / `HRR`).
  The running counter `k` looks sequential but is closed-form per row
  (`k(i) = (i-1)·n − (i-1)(i-2)/2`), so this can be threaded over `i` and/or restructured into
  block copies that hoist the branch out of the inner loop.
- `pairwise_quantile_forward!` — `O(W·(D+npair))` per FG call and `L`-independent, so it matters
  less as `L` grows, but it is also serial.

Note the shape of the problem: **the block that was parallelized is the one whose cost is roughly
`L`-independent, and the blocks that scale as `n_rows² ∝ (L-1)⁴` are the serial ones.** That is why
this only became the bottleneck at L=10.

## 4. Measure before optimizing — the number that decides it

A sub-block profile is running: `profile_pairwise_quantile_d20.jl 100000 10`, log
`logs/pq_L10_blockprofile.log` (started 2026-08-10 under `screen -S pq_L10_blockprofile`; if it is
gone, just re-run it). It prints per-block seconds for one real Hessian callback.

**Compare the sum of those block timings against `total_solve_seconds / n_hess`.** If they are close,
assembly dominates and §2–§3 are worth doing. If the block sum is a small fraction, then KNITRO's own
work dominates — it factorizes a *dense* ~15,970² KKT system every interior-point iteration,
~1.4×10¹² flops, which no amount of faster filling touches. In that case the honest answer is that
L=10 is expensive for a structural reason and the lever is elsewhere entirely.

⚠️ Do **not** infer per-callback cost as `total / n_hess` the way an earlier note in this repo did
(including one of mine) — that silently assumes the callback is 100% of wall-clock, which is the very
thing being tested.

⚠️ Do **not** reach for `pairwise_quantile_hvp.jl`. It was measured at exactly this scale and
rejected: dense 9 Hessian calls / 343.72 s vs HVP 6084 Hessian-vector calls / 1512.30 s = **4.4×
slower**, both verifier-confirmed correct. See `PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_RESULTS_2026-08-09.md`
Part 1.

## 5. What version B changes: the pattern is now fixed forever, only the weights move

This is worth internalizing because it is *newly true* and it unlocks work that version A could not
have done.

Under version A the cutoffs were outer coordinates, so `bin[w,o]` was rebuilt at every outer point
(`refresh_pairwise_quantile_bins!`) and every downstream index pattern was, in principle, volatile.
**Under version B the cutoffs are fixed for the whole campaign, so `bin[w,o]` is a campaign-lifetime
constant.** The free parameters are the masses, which enter only the *centering vector* `t` — they
never move a draw between bins.

Concretely, for every Hessian callback:

- **the scatter pattern of T1–T4 is identical every time; only the weights `h_w = Ψ''(r_w)` change.**
  The table-cell index each draw contributes to, for each of the ~18,000 combos, is fixed. Today
  that index is recomputed from `bin[w,·]` on every callback.
- the **sparsity/structure pattern** of `H_RR` is fixed — so if a sparse or structured hand-off to
  KNITRO is ever pursued, the pattern is computed once, not per callback.
- `refresh_pairwise_quantile_bins!` runs **once per campaign**, not once per outer point.

What that permits, in rough order of effort:

1. **Precompute the per-draw → table-cell flat index once** (a `W × n_combos` worth of indices is too
   big to store naively at D=20 — 1.8B — so this wants care: store per-combo, or recompute the small
   integer arithmetic but from a precomputed per-draw bin row that is already contiguous). Measure
   before building; the current index computation may already be cheap relative to the scattered
   write.
2. **Reorder draws once for locality.** The scatter's cost is dominated by random writes into the
   tables. With a fixed pattern you can permute the draws once, at context-build time, to improve
   cache behaviour for the largest tables — a pure preprocessing step with no effect on results
   (the tables are order-independent sums; keep the existing fixed-order reduction so it stays
   bit-reproducible).
3. **Hoist anything else that depends only on bins, not on `h`.** Worth a grep for per-callback work
   that reads only `bin`/`op` and not `h`.

None of these change any result. All are only correct because the cutoffs are fixed — do not port
them back to a free-cutoff variant.

## 6. Suggested order

1. Read the profile (§4). If assembly is not dominant, stop and say so.
2. Thread the centering loop (§2 option 1). Smallest, safest, largest single item.
3. Thread `fill_pairwise_quantile_hessian_raw!` over columns.
4. Restructure the packed write (§3).
5. Only then consider the fixed-pattern preprocessing (§5), which is the most invasive and the least
   certain.

After each step, re-run the D=4 dense oracle and confirm the real-D20 verifier KKT residual is
unchanged (it was `9.5e-13` after the 2026-08-09 pass). These are pure performance changes; any
numerical movement means a bug, not a tolerance.
