# ZC Gram (H_ZZ) BLAS experiment design (2026-07-28 addendum)

## Scope confirmation

Per the user's same-day addendum: dense BLAS remains explicitly **disallowed** for any block
touching the economic block `E` — `H_EE`, `H_EC`, `H_EZ` all keep the winner-aware segmented-
reduction kernels (`core_exact_hessian.jl::hessian_core_winner_pair!`, this task's own
`winner_pair_cross_hessian_fill_threaded!`, `winner_pair_cross_hessian_zc_block_threaded!`) —
**unchanged by this file**. `H_CZ` keeps the bin-structured segmented reduction
(`bin_zc_cross_hessian_fill_threaded!`) — also unchanged. This file is exclusively about `H_ZZ =
Z'SZ`, the mean/pair-ZC restriction SELF block, where dense BLAS on the raw feature matrix is both
permitted and requested.

## Representation

`Φ` (raw mean/pair feature matrix, `W x nx`) is immutable for the whole campaign
(`ZCRestrictionOperator.Zraw_all`/`Zpairraw_all`, theta-independent). `t` (current target vector)
changes once per inner solve. `S = diag(Ψ''(r))` changes every Hessian callback.

**Key structural change from the pre-addendum `zc_restriction_gram!`**: that function
rematerializes a CENTERED `Zc = Φ - 1t'` (plus its `S`-weighted copy `ZcS`) every Hessian callback
— even though `Φ` itself never changes. This addendum's algebraic identity avoids ever centering at
all:

```
Z'SZ = (Φ-1t')'S(Φ-1t') = Φ'SΦ - u t' - t u' + s0 t t',   u = Φ'S1,   s0 = 1'S1
```

`ZCRawWeightedWorkspace` (`zc_gram_blas_candidates.jl`) holds `Phi` (built once, campaign-lifetime,
concatenated from `Zraw_all`/`Zpairraw_all` in the SAME column order `refresh_zc_centered!` already
uses) and a single reusable row-weighted scratch `RW` — never both `sqrt(S).*Phi` and `S.*Phi`
retained simultaneously (per the addendum's explicit "do not retain both R and Y in production"
instruction; `RW` is refilled per-candidate during benchmarking, and only the CHOSEN production
candidate's fill discipline survives in the final default).

## Candidates implemented

- **`:blas_syrk`** (`zc_gram_blas_syrk!`): `RW = sqrt(S).*Phi`, `BLAS.syrk!('U','T',1.0,RW,0.0,raw)`
  — upper triangle only, half the FLOPs of a full GEMM.
- **`:blas_gemm`** (`zc_gram_blas_gemm!`): `RW = S.*Phi`, `mul!(raw, Phi', RW)` — full GEMM, both
  triangles computed (deliberately, to measure the wasted-lower-triangle cost against SYRK).
- **`:threaded_packed`** (`zc_gram_threaded_packed!`): non-BLAS, column-block ownership (see
  `THREADED_DIRECT_HZZ_RELEASE_2026-07-28.md`) — the addendum's own suggested simplification over a
  thread-local-accumulator design.
- **`:reference`**: the existing, unmodified, pre-addendum `zc_restriction_gram!` — kept as the
  correctness anchor and the current production default until this benchmark's own verdict says
  otherwise.

All three new candidates share the SAME `_zc_gram_apply_correction!` helper for the rank-2
target-correction (upper-triangle only, then mirrored once) — one implementation, not three, of the
one piece of math that's identical across candidates.

## H_CZ's own limited BLAS use (addendum §7)

`bin_zc_cross_hessian_fill!`/`_threaded!` (the draw-level bin reduction) is UNCHANGED — still a
segmented reduction, never a dense `C'SΦ`. The CM cumulative-basis / origin-contrast transforms
downstream of the raw bin-feature table (`winner_pair_cross_hessian_cm_block!`'s own `R`-congruence
step, `cctx.R` multiplication) already use `mul!`/BLAS for those SMALL transforms — this was
pre-existing production behavior (`block_ec = mul!(cctx.block_ec, Hraw_EC, cctx.R)`,
`cm_hessian_threaded.jl`), not something this task needed to add.

## Correctness gates

`test_threaded_cross_hessian_d4.jl` — `zc_gram_backend` ∈ {`:blas_syrk`, `:blas_gemm`,
`:threaded_packed`} compared against `:reference`, D=4, `K_mean∈{1,2}` × `K_pair∈{0,1}` (mean-only
and combined configs), calibration + perturbed points, both `cm_meanzc` and `origin_zc` (shared
dispatcher, one code path for both families). Tolerance `1e-9` (slightly looser than the pure-
threading kernels' `1e-12`, since this is a genuine algebraic re-derivation — Gram-plus-rank-2-
correction vs. direct centered-Gram — not merely a reordered sum of the identical formula).

## Performance gates

See `ZC_GRAM_BLAS_VS_THREADED_BENCHMARK_2026-07-28.csv`,
`ZC_GRAM_COMPLETE_SOLVE_AB_2026-07-28.csv`, `ZC_GRAM_MEMORY_COMPARISON_2026-07-28.csv`, and the
master report's `H_ZZ_BACKEND` verdict.
