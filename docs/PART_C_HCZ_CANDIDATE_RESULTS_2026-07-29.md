# Part C — H_CZ prep candidate bakeoff results (2026-07-29)

Full benchmark run: `full_aod_diag/d4_exact/hcz_candidate_bakeoff_2026-07-29.jl` (D=4 correctness-only
sweep across 4 configs, real D=20/W=100,000 correctness+timing sweep across 3 configs, 2
independent random-`S` draws per config). Raw output: `docs/HCZ_PREP_CANDIDATE_BENCHMARK_2026-07-29.csv`
(56 rows). Formula/redesign-target background: `docs/PART_C_HCZ_FORMULA_2026-07-29.md`.

## Result: ALL PASS (56/56 rows) — correctness, finiteness, determinism, non-aliasing

No candidate failed any gate at any config. Summary:

| candidate | vs reference | correctness (real D=20) | tol_kind |
|---|---|---|---|
| `origin_owned_threaded` (current production) | bit-identical | maxdiff = 0.0 | bit_identical |
| `draw_chunk_thread_local` (Candidate 2) | maxdiff = 5.28e-11 to 5.91e-11 | tol = 1e-9 * scale(~3.7e4) ≈ 3.7e-5 → **~5-6 orders of magnitude inside tolerance**, not a knife edge | relative_tol |
| `sparse_spmm` (Candidate 4) | bit-identical | maxdiff = 0.0 | relative_tol (trivially passes) |

`draw_chunk_thread_local`'s nonzero maxdiff is expected and disclosed in its own file header: floating-point
summation is not associative, and this candidate sums per-worker partial tables before a final reduction,
so it cannot be bit-identical to the strict `w=1:W` serial reference — only close to within float error. The
observed diffs (~1e-11) are consistent with that and nowhere near the 1e-9·scale tolerance.

## Timing (real D=20, W=100,000, D=20 origins, L=50, n_z=210, workers=20) — the only timed configs

| config | ref (s) | origin_owned_threaded | draw_chunk_thread_local | sparse_spmm |
|---|---|---|---|---|
| `d20_K1P1_exclude_row_anchored` (production config) | 0.9210 | 0.7741s (**1.19x**) | **0.0657s (14.02x)** | 1.0998s (0.84x, SLOWER) |
| `d20_K1P1_exclude_row_orthonormal` | 0.9284 | 0.7277s (1.28x) | **0.0753s (12.32x)** | 1.0135s (0.92x, SLOWER) |

(`d20_K1P1_all_legacy_orthonormal` — square-contrast config — was correctness-only per the script's
own `do_timing` flag, not timed.)

## Winner: Candidate 2, draw-chunk thread-local (`bin_zc_cross_hessian_fill_drawchunk!`)

**14.0x** speedup over the serial reference at the production config (anchored contrasts,
`exclude_row` destination sampling), **12.3x** at the orthonormal-contrast config — both far ahead
of current production's `origin_owned_threaded` (1.19x–1.28x only). This confirms the root cause
in `docs/PART_C_HCZ_FORMULA_2026-07-29.md`: origin-chunking forces every one of the 20 workers to
stream the entire `W x n_z` (`100,000 x 210 ≈ 168MB`) `ZcS` array once each — the redundant
memory-bandwidth pressure, not compute, was the real bottleneck. Draw-chunking gives each worker a
disjoint, single-pass slice of `ZcS`, and the payoff (14x from 20 workers, ~70% thread efficiency)
shows this was overwhelmingly a memory-traffic problem, not a cache-locality problem within a
worker's own `D x n_z x (L+1)` accumulator (`≈1.68MB`, fits comfortably in L2).

Candidate 4 (sparse one-hot SpMM) is bit-identical (a real linear operator, no reduction-order
change) but is **not** a performance win at this problem's shape — it is *slower* than the serial
reference (0.84x–0.92x) at `D=20, n_z=210`. The one-nonzero-per-row sparse GEMM does not amortize
its own indirection/overhead versus the dense reference at this size; it is dominated outright by
Candidate 2 and is not recommended.

## Candidate 3 (tiled draw-chunk): not needed, not written

Per `docs/PART_C_HCZ_FORMULA_2026-07-29.md`, Candidate 3 was explicitly conditional: "only pursued
if Candidate 2's per-worker `D x L x n_z` local table itself shows poor cache behavior." The real
benchmark shows the opposite of a cache problem — Candidate 2 already delivers 12-14x speedup with
20 workers, i.e. good scaling with no sign of memory-bound degradation inside the per-worker
accumulator. There is no genuine problem left for tiling to solve, so Candidate 3 was not
implemented. This is a benchmarked decision, not a default/assumption.

## Recommendation

Candidate 2 (`bin_zc_cross_hessian_fill_drawchunk!` / `BinZCrossDrawChunkScratch`) is the clear
winner and the recommended replacement for the H_CZ prep step's threading strategy, pending
downstream wiring into `bin_zc_cross_hessian_block!` / the production callback path (out of scope
for this benchmark-only session — no production files were modified). Use `HCZ_CANDIDATE_TOL =
1e-9` (relative, scaled by `max(1, max|ZBinCScum|)`) as the correctness gate for this candidate
going forward, matching what this bakeoff already validated at both D=4 and real D=20 scale.
