# ZC feature / centering algebra — 2026-07-28

**Status note (read first):** the algebra below is *already implemented, shared, and gated*
on `production/fullA-exact` — this document is a from-first-principles derivation and
cross-check against the live code, not a new design. Where the task brief asked "confirm formulas
against dense reference at D=4 before performance work", that gate already exists and passes
(`test_flexible_cm_winner_bin_her_wiring_d4.jl`, `test_cm_meanzc_winner_bin_hez_wiring_d4.jl`,
`test_originzc_winner_bin_her_wiring_d4.jl`, `test_cm_meanzc_hcz_hzz_direct_d4.jl` — all present
and, per `git log`, all merged into the base commit this task starts from). See
`STRUCTURED_CROSS_HESSIAN_PROVENANCE_2026-07-28.md` for how this was discovered.

## Notation (matching the task brief and the code one-to-one)

- `E = Q - ν π'` — the winner-conditioned economic block. Row `w` (draw), column `j` (a
  bilateral (slot,origin) pair or the "cf"/gravity common-factor column): `E[w,j] = ν[w] *
  (y[w,slot(j)] * 1{winner(w,slot(j)) = origin(j)} - π[j])`. Code: `core_exact_hessian.jl`'s own
  `WinnerPairHessCtx` (`nu`, `y`, `winner`, `pi_vec` fields) is the live representation of exactly
  this decomposition; `winner_pair_cross_hessian.jl`'s header comment (lines 6–18) re-derives the
  single-`ν`-power vs. double-`ν`-power distinction used below directly from this struct's own
  internal consistency (not re-derived from a blank page).
- `C` — the CM bin/contrast operator (interval-bin membership, cumulative-prefix + origin-contrast
  transform applied downstream). Code: `cctx.Bidx` (raw bin membership) + `cctx.CScum`/`cctx.R`
  (cumulative + contrast transform).
- `Z = Φ - 1 t'` — the already-centered mean/pairwise-ZC restriction feature block. `Φ` = the raw,
  theta-INDEPENDENT feature matrix (`ZCRestrictionOperator.Zraw_all[k]`/`Zpairraw_all[k]`,
  immutable for the whole campaign). `t` = the current outer point's target vector
  (`ZCRestrictionWorkspace.targets_mean`/`targets_pair`, refreshed once per inner solve via
  `refresh_zc_targets!`, NOT per Hessian callback). Code: `zc_restriction_operator.jl`.
- `S = diag(Ψ''(r))` — the current dual point's Hessian weight vector, `obj.arg2` after
  `ddPsi!(obj.arg2, obj.arg0)`. Refreshed once per Hessian callback (dual-dynamic, see the
  lifecycle audit doc).
- `M` — draw count (`obj.M`, `= W`).

## H_EC = E'SC (flexible-CM / common-Frechet CM-grid block, and the true-economic sub-block of CM+ZC)

Dense reference (per-threshold-block `l`, per origin `o`, CM contrast against `refIndex1`):

```
H_EC[j, o, l] = (1/M) * Σ_w S[w] E[w,j] 1{bin(U[w,o]) <= l}   [difference against refIndex1's own column]
```

Exact decomposition used by `winner_pair_cross_hessian_fill!`/`_cm_block!` (no dense read of `E`):
substituting `E[w,j] = ν[w]*(y[w,slot(j)]*1{winner=origin(j)} - π[j])`,

```
Σ_w S[w] E[w,j] 1{bin<=l}
    = Σ_w (S[w]ν[w]) y[w,slot(j)] 1{winner(w,slot(j))=origin(j)} 1{bin(U[w,o])<=l}     ["QCScum" term]
      - π[j] * Σ_w (S[w]ν[w]) 1{bin(U[w,o])<=l}                                          ["NuCScum" term, rank-1 in j]
```

Both inner sums are **origin-indexed, cumulative-prefix-summable** exactly like the dense `CS_`
table this replaces — this is why the raw-table fill (`QTab`/`NuTab`, one `O(W*D)` pass for `NuTab`
and one `O(W*Ddest*D)` pass for `QTab`, since for FIXED `slot`, `j` only ranges over the
`origin`-indexed subset `{slot, slot+Ddest, slot+2Ddest, ...}`) is followed by an `O(D*L)` prefix
sum, then an `O(NCORE*nO)` per-`l` slice — same complexity class as the dense path, but the raw
fill never reads `E`/`obj.H`'s economic columns at all (§20–23 of `winner_pair_cross_hessian.jl`'s
own header).

The row-1 ("ones"/ζ-paired) column and the cf/gravity-common-factor column are NOT of the
`ν*(y*1{winner}-π)` shape (row 1 is a plain constant-1 moment; cf is a raw per-sample value, not
winner-selected) — both get their own dedicated single-`S`- or `Snu`-weighted accumulation
(`SOnlyTab`/`QCfTab`), documented and gated separately (see the file's own inline comments,
lines 116–122, 189–196).

## H_EZ = E'SZ (winner-aware economic × mean/pair-ZC cross, shared by CM+ZC and origin-ZC)

```
H_EZ = E'S(Φ - 1t') = E'SΦ - (E'S1)t' = E'SΦ - Esum*t'
```

`Z` (already centered) is passed in directly rather than threading `Φ`/`t`/`Esum` separately
through this function — algebraically identical (`E'S*(Φ-1t') = E'SΦ - (E'S1)t'`, the same
quantity), and BOTH production call sites (`_fill_cm_HEE!` for CM+ZC's `HEM`,
`archA_partitioned_hess_cb_builder` for origin-ZC's `HER`) already have `Z` sitting in a
`ZCCenteredScratch.Zc` local view for free every callback, built directly from `Zraw_all`/targets
(never from `obj.H`). Substituting `E`'s own decomposition exactly as for H_EC:

```
HEZ[j+1, x] = (1/M) * ( Σ_w (S[w]ν[w]) y[w,slot(j)] 1{winner=origin(j)} Z[w,x]   -- winner-conditioned scatter
                        - π[j] * Σ_w (S[w]ν[w]) Z[w,x] )                          -- rank-1 pi_vec correction, NuZ[x]
row 1:  HEZ[1,x] = (1/M) * Σ_w S[w] Z[w,x]                                        -- plain gemv, S-only
cf row: HEZ[jcf+1,x] = (1/M) * ( Σ_w (S[w]ν[w]) cf_raw_scaled[w] Z[w,x] - π[jcf]*NuZ[x] )
```

Unlike H_EC there is no threshold-binning (`Z`'s columns are continuous, not step functions of a
bin index), so this is filled in ONE pass, `O(W*Ddest*n_x)` for the winner-conditioned scatter
(loop order `slot -> x -> w`, keeping `Z[:,x]`/`winner[:,slot]` column-contiguous) plus `O(W*n_x)`
BLAS `gemv!`s for the non-winner-conditioned row-1/cf rows. Code: `winner_pair_cross_hessian_zc_block!`,
`winner_pair_cross_hessian_zc_prep!` (refreshes the shared `Snu[w]=S[w]ν[w]` once per callback).

## H_CZ = C'SZ (CM-grid × mean/pair-ZC cross, CM+ZC only)

No winner-selection at all here (`Z`'s columns are per-draw feature values, not a winner-argmin
outcome) — only a bin-membership test, so this is simpler than H_EC/H_EZ: a single per-(origin,
bin) accumulation of the already-centered, already-`S`-weighted `ZcS[w,j] = S[w]*(Φ[w,j]-t[j])`
(built once per callback by `refresh_zc_centered!`, shared verbatim with H_ZZ below — the SAME
scratch, not recomputed):

```
ZBinTab[x, j, b] = Σ_{w: bin(U[w,x])=b} ZcS[w,j]                     [O(W*D*n_z)]
ZBinCScum[x, j, l] = Σ_{b<=l} ZBinTab[x, j, b]                       [prefix sum, O(D*n_z*L)]
H_CZ[j, o, l] = (1/M) * (ZBinCScum[o,j,l] - ZBinCScum[refIndex1,j,l])  [per-l, per-origin slice]
```

Code: `bin_zc_cross_hessian_fill!`/`_block!`.

## H_ZZ = Z'SZ (mean/pair-ZC self block, shared by CM+ZC and origin-ZC)

Directly from the algebraic identity `Z'SZ = (Φ-1t')'S(Φ-1t')`, expanded and regrouped as the task
brief's own target-correction decomposition (`Φ'SΦ`, `Φ'S1`, `1'SΦ`, `1'S1`, plus rank-one `t`
corrections) — but the LIVE code does not materialize this expansion term-by-term; it instead
builds the centered `Zc[w,j] = Φ[w,j]-t[j]` and `ZcS[w,j] = S[w]Zc[w,j]` scratch matrices directly
(`refresh_zc_centered!`, `O(W*n_restriction)`, no `W x n_restriction` matrix EVER read from
`obj.H`) and takes a single dense BLAS Gram:

```
H_ZZ = (1/M) * Zc' * ZcS         [BLAS.gemm!('T','N', 1/M, Zc, ZcS, 0.0, HZZ)]
```

This is algebraically identical to the term-by-term expansion (`Zc = Φ - 1t'` is exact, not an
approximation of it) — the task brief's explicit alternative expansion (`Φ'SΦ, Φ'S1, ...` plus
rank-one updates) was considered as **Candidate for a possible future compressed/monomial
representation** (task Section 8) but is NOT how the current exact primitive is built; see the
Section 7/8 performance-gate results below for why re-deriving the expansion was not pursued (the
direct Gram is already the dominant-cost primitive worth threading, not FLOP-reducing further
until that's exhausted).

Code: `zc_restriction_operator.jl::zc_restriction_gram!`. Shared verbatim by CM+ZC's `HMM`
(`_fill_cm_HEE!`'s `ncore < NCORE` branch) and origin-ZC's `HRR`
(`archA_partitioned_hess_cb_builder`) — literally the same function call at both sites, confirmed
by direct code reading (not merely asserted), see `STRUCTURED_CROSS_HESSIAN_PROVENANCE_2026-07-28.md`.

## D=4 correctness (existing gates, re-confirmed for this task, not re-derived)

`test_winner_pair_cross_hessian_cm_d4.jl` / `_d20.jl` (H_EC), `test_flexible_cm_winner_bin_her_wiring_d4.jl`
/ `_d20.jl` (H_EC end-to-end wiring), `test_cm_meanzc_winner_bin_hez_wiring_d4.jl`/`_d20.jl` (H_EZ),
`test_originzc_winner_bin_her_wiring_d4.jl`/`_d20.jl` (H_EZ for origin-ZC), `test_cm_meanzc_hcz_hzz_direct_d4.jl`/`_d20.jl`
(H_CZ + H_ZZ), `test_winner_pair_cross_hessian_zc_d4.jl` (standalone H_EZ dense-reference gate) —
all present at the base commit. This task's own D=4 correctness gates (Section 11 below) extend
these with THREADED-vs-serial agreement, not a fresh dense-reference re-derivation (that
already-passing gate is not this task's open question).
