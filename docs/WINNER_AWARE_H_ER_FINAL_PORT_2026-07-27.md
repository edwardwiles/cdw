# Winner-Aware H_ER Cross-Hessian — Final Port Status — 2026-07-27

Task's Phase B ask, in full: implement `H_ER = Q'SR - π(ν'SR)` (the winner-aware economic ×
restriction cross Hessian) for every restriction family (flexible CM, common Fréchet, CM+ZC,
ZC-only), eliminate the corresponding dense economic-column reads, and add global no-dense-G
runtime counters.

## Honest status: 1 of 4 families, the primitive built and gated, NOT wired into production

| Family | Cross-block | Status |
|---|---|---|
| Flexible CM | `H_EC` (CM-grid) | **Built and gated** (`winner_pair_cross_hessian.jl`). D=4 (8 configs: anchored/orthonormal x L=10/20, calib+perturbed) and real D=20/W=80,000/L=50 (both contrasts) all PASS, agreement to machine precision. **NOT wired** into `hessian_cm_structured!` as an opt-in or default backend -- this commit is the validated primitive only. |
| Common Frechet | `H_EC` (CM-grid) + level-anchor column | NOT attempted. The CM-grid part should be a near-direct reuse of the flexible-CM primitive (same kernels); the level-anchor column needs its own winner-bin derivation (analogous to the level block added to `verify_inner_solution_operator_cm_frechet!` this session, see Phase A docs) -- not started. |
| CM+ZC | `H_EC` (CM-grid, shared) + mean/pair ZC cross block | NOT attempted. The CM-grid share should reuse the flexible-CM primitive directly; the mean/pair ZC cross block needs a parallel "winner-feature cross" derivation (task's own wording) -- not started. |
| ZC-only | `H_EZ` (mean/pair ZC only, no CM grid) | NOT attempted. |

Global no-dense-G runtime counters (`full_G_materializations`, `dense_economic_G_materializations`,
`generic_dense_FG_calls`, `dense_reference_verification_calls`, `dense_cross_hessian_calls`,
`operator_cross_hessian_calls`) -- **NOT added**. There is nothing new to count until a winner-bin
backend is actually wired in as a real alternative path (the dense path is still the only one that
runs in production for this block).

## What WAS built and validated this session

### The primitive: `winner_pair_cross_hessian.jl`

`winner_pair_cross_hessian_fill!` / `winner_pair_cross_hessian_cm_block!`, reusing the
already-validated `WinnerPairHessCtx` (`core_exact_hessian.jl`, the H_EE winner-pair backend)
rather than re-deriving `E`'s `Q - vpi'` decomposition from a blank page. The derivation is
cross-checked against that file's own internal consistency: its gradient-shaped accumulator
(`u[j] += Snu[w]*y[w,slot]`, single power of `nu`) is the linear-in-`E` contraction a cross term
against an EXTERNAL restriction column needs; its Hessian self-product accumulator (`Snu2`,
double power of `nu`) is specific to `E` contracted against itself and does NOT apply here.

Replaces the dense `build_bin_tables!`'s `O(W*D*NCORE)` `S`-table fill (which reads `obj.H`'s
dense economic `G` columns) with `O(W*D*Ddest)` winner-bin accumulation -- no dense `E` read at
all, a `D`-fold reduction (`NCORE = D*Ddest`).

### Two real bugs found and fixed during derivation (both caught by the gate, not shipped)

1. **Row-index off-by-one.** `cctx.NCORE = wctx.ncolI + 1`: `cm_hessian_architectures.jl`'s
   `E = @view H[:, 2:1+NCORE]` includes `H[:,2]`, an all-ones column
   (`compressed_live.jl: obj.H[:, 2] .= 1.0`) that is NOT one of the `nu`-weighted economic
   columns `WinnerPairHessCtx` tracks. Fix: shift economic rows by `+1`; give row 1 its own
   `S`-only (no `nu`, no `pi_vec` correction) accumulator, `SOnlyTab`/`SOnlyCScum`.
2. **Missing "cf" column.** When `cf.cf_col > 0` (a common-factor/gravity column present in the
   economic block), that column is NOT winner-conditioned like the regular `(slot, origin)`
   economic columns -- the main per-slot loop never touches it, silently leaving its row at zero.
   Fix: a dedicated `QCfTab`/`QCfCScum` accumulator over every sample (mirroring
   `core_exact_hessian.jl`'s own `uu += Snu[w]*crs[w]` gradient-shaped accumulator for this same
   column), overwriting that one row in the per-block fill.

Both bugs were large (max error 0.31 and 0.18 respectively, against a matrix scale of ~1) and
would have been immediately visible in any real downstream use -- they were caught by the gate
before any commit, not discovered later.

### Gates (all PASS)

- `test_winner_pair_cross_hessian_cm_d4.jl`: D=4, contrasts in {anchored, orthonormal}, L in
  {10, 20}, points in {calib, perturbed} -- 8 configurations, all bit-identical to
  `hessian_cm_structured!`'s own dense `H_EC` block to ~1e-16.
- `test_winner_pair_cross_hessian_cm_d20.jl`: real D=20/W=80,000/L=50, contrasts in {anchored,
  orthonormal} -- both PASS: anchored max|Delta|=1.81e-13 (scale~=18.8), orthonormal
  max|Delta|=1.07e-13 (scale~=18.0). The new fill step (`winner_pair_cross_hessian_fill!`)
  measured ~0.28s (anchored, cold) / 0.14s (orthonormal, warm) vs. the full dense
  `hessian_cm_structured!` (H_EE+H_EC+H_CC together) at ~4.0s / 3.2s -- not apples-to-apples (the
  dense timing includes H_EE/H_CC too), but a strong directional signal that the winner-bin
  approach is not just correctness-preserving but substantially cheaper for this block
  specifically.

## What "wiring it in" would require (not done, scoped for the next session)

1. Add `Hraw_EC`-backend selection to `hessian_cm_structured!` (mirroring `CMBinHessCtx`'s own
   `core_hessian_backend` kwarg pattern already used for `H_EE`) -- e.g.
   `cm_grid_hessian_backend in (:dense_reference, :winner_bin)`.
2. Replace the `build_bin_tables!`/`prefix_sum_tables!`'s `S`/`CScum` fill with
   `winner_pair_cross_hessian_fill!`'s `QCScum`/`NuCScum`/`SOnlyCScum`/`QCfCScum` ONLY when the
   winner-bin backend is selected -- `H_CC` (restriction x restriction) is untouched either way
   (this file does not touch that block; the task's own H_EE/H_EC/H_ER scope excludes H_RR).
3. Persist `WinnerPairHessCtx`/`WinnerBinCrossScratch` per-`cf`-identity (mirroring
   `cctx.core_ws`/`core_ws_for`'s own rebuild-on-identity-change discipline) rather than rebuilding
   every callback -- the D=4/D=20 gates above rebuild fresh every call, which is correct but not
   the production allocation profile.
4. A default-flip decision needs a broader gate (multiple `L`, multiple `delta`, both contrasts, a
   real short outer trajectory) -- same standard this project applied to the common-Frechet FG
   backend's own non-flip decision (`COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md`).
5. Only after flexible-CM's own wiring is complete and gated should the other 3 families be
   attempted -- CM+ZC and common-Frechet should both be able to reuse the CM-grid half of this
   primitive directly (same kernels, same `Bidx`/`origins`/`refIndex1`/`R` inputs); the level-anchor
   (common-Frechet) and mean/pair (ZC-only, CM+ZC) cross blocks are new derivations each, following
   the SAME reuse-`WinnerPairHessCtx`-don't-re-derive-`E` methodology this file establishes.

## Verdict

```text
HESSIAN_H_ER =
    flexible_cm: primitive_built_and_gated (D=4 8/8 PASS machine-precision, D=20/L=50 2/2 PASS
        machine-precision) NOT_WIRED_INTO_PRODUCTION_HESSIAN
    common_frechet: not_attempted
    cm_plus_zc: not_attempted
    zc_only: not_attempted

GLOBAL_NO_DENSE_G_COUNTERS = not_added (nothing new to count until a winner-bin backend is
    actually wired as a real alternative path)

NEXT_STANDALONE_TASK = wire winner_pair_cross_hessian_cm_block! into hessian_cm_structured! as an
    opt-in cm_grid_hessian_backend, gate broadly, THEN extend the same primitive to common-Frechet
    and CM+ZC's shared CM-grid block, THEN derive the mean/pair and level-anchor cross blocks
    (genuinely new math each, not reuse)
```
