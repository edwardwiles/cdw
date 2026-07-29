# Part C — H_CZ exact formula and redesign target (2026-07-29)

> Provenance note (Part C session, 2026-07-29): this file did not exist in the
> `partC-hcz-prep-2026-07-29` worktree at task start, despite being named as an already-present
> prerequisite doc. It was found (uncommitted) in the sibling Part B worktree
> (`diagnose-optimize-HZZ-BLAS-and-HCZ-prep-2026-07-29`) and copied here verbatim (read-only access
> to that worktree, nothing there was modified) so this branch is self-contained. Same for the two
> candidate implementation files it references
> (`hcz_drawchunk_candidate_2026-07-29.jl`, `hcz_sparse_spmm_candidate_2026-07-29.jl`), also copied
> verbatim from that same sibling worktree.

Full function-level trace already in `docs/PART_A_LABEL_MAPPING_2026-07-29.md` -- this doc states
the formula and redesign scope only.

## Formula

```
H_CZ = C' S Z,      Z = Phi - 1t'
```

Decomposed exactly as production computes it today (`bin_zc_cross_hessian_fill!` +
`bin_zc_cross_hessian_block!`, `winner_pair_cross_hessian.jl:605-652`):

1. **CZ raw bin-feature reduction** (draw-level, `O(W*D*n_z)`, THE dominant cost -- 0.996s/iter at
   t20, 57.4% of `cm_meanzc`'s whole Hessian-callback time):

   ```
   T[o,b,j] = Σ_{w : bin_o(w)=b} S_w * Z[w,j]      (Z already centered: Z = Phi - 1 t')
   ```
   `o` = CM origin (`D=20`), `b` = threshold bin (`L+1=51`), `j` = ZC-restriction feature
   (`n_z≈210` at K_mean=1/K_pair=1). Current code: `ZBinTab[x,j,b] += ZcS[w,j]` for every
   `(w,x,j)` triple, `ZcS` already `S`-weighted+centered by `refresh_zc_centered!` (shared with
   H_ZZ, NOT recomputed here).

2. **CZ cumulative transform** (small assembly, `O(D*n_z*L)`): `ZBinCScum[x,j,l] = Σ_{k≤l}
   ZBinTab[x,j,k]`.

3. **CZ centering correction**: already folded into step 1's input (`ZcS`) -- NOT a separate cost
   inside `CZ_prep`; it's counted in the `EM+MM` timer (`refresh_zc_centered!`, run once, shared).

4. **CZ contrast transform** (origin difference, per threshold block `l`, small,
   `O(n_z * n_origins)` per `l`): `H_CZ[j,o] = (1/M) * (ZBinCScum[o,j,l] - ZBinCScum[ref,j,l])`.

5. **CZ packing** (small, per `l`): optional `R`-congruence (`mul!` with the threshold-contrast
   matrix `R`) then write into the packed/`Hfull` output -- same per-`l` loop as H_EC's own asm,
   shares the loop, not separately timed.

Steps 2/4/5 are cheap and out of scope. **Step 1 is the sole redesign target.**

## Why step 1 is "flat" under current threading (0.88x at t20 -- see Part A)

Current threading (`bin_zc_cross_hessian_fill_threaded!`, `threaded_cross_hessian.jl:300-339`)
chunks the OUTER loop by **origin** (`x`): each of the `workers` threads owns a disjoint subset of
`D=20` origins but still loops over ALL `w in 1:W` for its subset, because `ZcS` (the shared
`W x n_z` array every worker reads) has no origin dimension at all -- only `Bidx[w,x]` does. Net
effect: the single largest array in this computation (`ZcS`, `W x n_z ≈ 100,000 x 210 ≈ 168MB` at
`W=100k`) gets streamed through cache **`workers`-fold redundantly**, once per worker, all
concurrently -- real memory-bandwidth contention, not merely a suspicion (this is the direct,
structural reason origin-ownership cannot avoid the redundant read: `ZcS[w,j]` does not depend on
which origin-chunk a worker owns, so every worker must read every row).

## Candidates (Part C benchmark plan)

- **Candidate 1 (reference, current)**: origin-owned, as above. Kept as correctness anchor and
  performance baseline.
- **Candidate 2 (draw-chunk thread-local)**: chunk the `W` draws across workers instead of origins.
  Each worker reads its OWN chunk of `ZcS` rows exactly once (no redundant reads across workers),
  maintains a persistent thread-local `D x L x n_z` accumulator table (`≈1.68MB` at
  `D=20,L=51,n_z=210`; `20` workers `≈34MB`, matching the task's own estimate), then a final
  cross-worker reduction (`workers x D x L x n_z` sum) -- asymptotically the same total FLOPs as
  Candidate 1, but total `ZcS` memory traffic drops from `workers*W*n_z` to `W*n_z` (one pass).
- **Candidate 3 (tiled draw-chunk)**: only pursued if Candidate 2's per-worker `D x L x n_z` local
  table itself shows poor cache behavior (unlikely at `D=20,L=51,n_z=210` -- fits in L2 -- but
  benchmarked, not assumed).
- **Candidate 4 (sparse one-hot SpMM)**: precompute `Bidx` as a `W x D` matrix of per-origin bin
  memberships into a compact sparse one-hot incidence (`B`, `(D*(L+1)) x W`, one nonzero per
  `(origin,draw)` pair) once per campaign (theta-independent), then `T = B' * ZcS` (or the
  transposed layout) via CSC-sparse-times-dense or a custom segmented reduction. Benchmarked
  against candidates 1-3, not assumed to win (task's own framing: "may include ... an already-
  approved threaded sparse-dense implementation").

Results: `docs/HCZ_PREP_CANDIDATE_BENCHMARK_2026-07-29.csv` (Part C benchmark run).
