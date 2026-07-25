# Structured Hessian for the Combined CDF+Power Fixed-Fréchet Block — 2026-07-24

**Status: implemented (`cm_frechet_power_hessian_structured.jl`) and validated against dense
Architecture A at D=4 to 2.3e-15 absolute error (relative 2.3e-15) — see
`test_logs/d4_gate_cdf_power_structured_hessian_2026-07-24.log`, gate P2, 13/13 PASS overall.**

This is the required new port-preparation deliverable identified by the task brief: the
pre-omit-ROW reconciliation archive (`experiment/fullA-fixed-frechet-basis-draft-reconciliation-2026-07-24`)
implemented eq. (38) (truncated-power) moment columns and validated Δ*/primal-weight nesting
through the **dense** (Architecture A) Hessian only; it explicitly disclosed leaving the fast
structured kernel for the power block unbuilt ("no fast structured Hessian exists yet for the
truncated-power feature block" — its own Part VIII, item 3). This document derives and records
that kernel.

## 1. Why a new kernel, not a reuse of the CDF-only one

The existing production Architecture-C kernel (`cm_hessian_architectures.jl`, reused unchanged by
this branch's `cm_frechet_hessian.jl` for the CDF-only case) builds weighted **bin-contingency
tables** `Ttab[x,y,k,k'] = Σ_s w_s·1{bin_x(s)=k}·1{bin_y(s)=k'}` and
`Stab[x,j,k] = Σ_s w_s·E[s,j]·1{bin_x(s)=k}`, then 2D/1D-prefix-sums them into cumulative tables
`CT`/`CScum`. This works because every CDF-family moment column is an **indicator** function
(`1{U_o ≤ z_l}`), and indicators are idempotent (`1{}·1{} = 1{}`), which the existing formulas
exploit directly (e.g. the marginal CDF count at origin `o`, threshold `l` is read straight off
the table's own diagonal, `CT[o,o,l,l]`).

The truncated-power family's moment columns are **not** indicators — they are
`U[s,o]^pw · 1{U_o(s) ≤ z_l}` (`pw = (1-σ)/θ*`), a continuous weight times an indicator. Squaring
this factor changes its value (`(U^pw)² ≠ U^pw`), so the CDF kernel's diagonal-reuse trick does
not carry over, and the cross-covariance between a CDF-family column and a POWER-family column
needs its own joint table (a CDF-side indicator times a POWER-side weighted indicator — asymmetric
in the two weight functions). This requires three new raw tables, built in one extra bin-indexed
sweep sharing the same bin index array `Bidx` (both families use the same threshold grid
`z = targets.thresholds`, so no new binning pass over `U` is needed, only new weighted
accumulation).

## 2. The three new tables

Let `φ_CDF(s,x) ≡ 1`, `φ_POW(s,x) = U[s,x]^pw`. For origins `x,y ∈ 1..D` and bins `k,k' ∈ 1..L+1`:

- `Ttab_pp[x,y,k,k'] = Σ_s w_s·φ_POW(s,x)·1{bin_x(s)=k}·φ_POW(s,y)·1{bin_y(s)=k'}` (POWER×POWER)
- `Ttab_cp[x,y,k,k'] = Σ_s w_s·1{bin_x(s)=k}·φ_POW(s,y)·1{bin_y(s)=k'}` (CDF-side x, POWER-side y)
- `Stab_p[x,j,k] = Σ_s w_s·φ_POW(s,x)·1{bin_x(s)=k}·Ê[s,j]` where `Ê = [E  1]` (core columns plus
  one extra constant-1 column, `j = NCORE+1`) — the extra column makes
  `CScum_p[x, NCORE+1, l] = Σ_s w_s·φ_POW(s,x)·1{U_x≤z_l}` double as the POWER-family's own
  **marginal** table (`MS_pow[x,l]`), by the same "treat a constant as a fictitious core column"
  trick, since POWER is not idempotent and cannot reuse a diagonal shortcut the way CDF does.

All three raw tables are built in a single extra `O(W·(D² + D·(NCORE+1)))` pass
(`build_frechet_power_bin_tables!`), then 2D/1D-prefix-summed exactly like the existing
`prefix_sum_tables!` (`_frechet_prefix_sum_2d!`/`_frechet_prefix_sum_1d!`, same algorithm, applied
to the new arrays). `CT_pc[x,y,l,l']` (POWER-side x, CDF-side y — needed for the "POWER-common ×
CDF-contrast" cross term) is obtained as `CT_cp[y,x,l',l]` (an index swap, no separate table).

## 3. Assembled sub-blocks

Every Hessian entry is `(1/M)·Σ_s w_s·g_a(s)·g_b(s)` for two centered moment columns `g_a,g_b`
(each either a **contrast** column, `g = φ_A(s,o)1{o≤l} − φ_A(s,ref)1{ref≤l}`, or a **common/pin**
column, `g = φ_A(s,ref)1{ref≤l} − t_A[l]`). Expanding the product of any two such columns (four
terms each) and summing gives every sub-block purely in terms of `CT_AB[x,y,l,l']` (`A,B` each
CDF or POWER), `CS_A[x,j,l]` / `MS_A[x,l]`, and the targets `p[l]`/`tpow[l]`. The six sub-blocks of
the combined `(NCORE + 2·(D·L)) × (NCORE + 2·(D·L))` Hessian:

| Block | Table(s) used | Formula source |
|---|---|---|
| CDF×CDF (contrast/common) | `CT`, `CScum` (existing) | unchanged from `cm_frechet_hessian.jl` |
| POWER×POWER (contrast/common) | `CT_pp`, `CScum_p` | structural analogue, `p→tpow`, `CT→CT_pp`, `CScum→CScum_p`; marginal via `CScum_p[·,NCORE+1,·]` in place of the CDF diagonal trick |
| CDF-contrast × POWER-contrast | `CT_cp` | `(CT_cp[o,p,l,l'] − CT_cp[o,ref,l,l'] − CT_cp[ref,p,l,l'] + CT_cp[ref,ref,l,l'])/M` |
| CDF-contrast × POWER-common | `CT_cp`, `CT` (for `MS_cdf` via diagonal) | `(CT_cp[o,ref,l,l'] − CT_cp[ref,ref,l,l'] − tpow[l']·(CT[o,o,l,l]−CT[ref,ref,l,l]))/M` |
| CDF-common × POWER-contrast | `CT_cp`, `CScum_p` | `(CT_cp[ref,p,l,l'] − CT_cp[ref,ref,l,l'] − p[l]·MS_pow[p,l'] + p[l]·MS_pow[ref,l'])/M` |
| CDF-common × POWER-common | `CT_cp`, `CT`, `CScum_p` | `(CT_cp[ref,ref,l,l'] − tpow[l']·CT[ref,ref,l,l] − p[l]·MS_pow[ref,l'] + p[l]·tpow[l']·M)/M` |

(`MS_pow[x,l] := CScum_p[x, NCORE+1, l]`.) All contrast-indexed blocks apply the same
`R`/orthonormal-contrast congruence transform as the CDF-only kernel, applied independently on
each origin-indexed axis (never mixing the two families' `R` applications — `R` only ever acts
within one family's own `nO`-length origin axis at a fixed threshold, exactly as in the existing
kernel).

## 4. Validation

`test_frechet_power_hessian_d4_gates.jl`, gate P2: dense (Architecture A, `obj_cm(x, h=...)`) vs.
structured (`hessian_cm_frechet_cdf_power_structured!`) at the D=4 calibrated point, `L=8`,
`ncore=18`, `ncm=64` (`2·D·L`). Result: **max abs difference 2.33e-15** (machine precision), and
an end-to-end KNITRO solve routed through the structured Hessian callback
(`archC_frechet_cdf_power_hess_cb_builder`) reproduces the dense solve's Δ* to **0.0** absolute
difference. Gate P1 (CDF-only, re-validated as part of this same run) matches to 4.9e-16. Gate P3
confirms the theoretically-required nesting
`Δ*_flexible(0.002967) ≤ Δ*_frechet,cdf(0.003566) ≤ Δ*_frechet,cdf+power(0.005816)`.

**Not yet run**: a bounded real-D=20/L=50 structured-vs-dense check (task brief §11's D=20
requirement) — the dense Architecture-A path is not memory-safe to run at full L=50/D=20 scale
(`ncm=2000`, a `2018×2018` dense Hessian per callback), so that gate is scoped to a **bounded
sub-problem** (a truncated `L` or a single feature family) rather than the full point; see the
port-readiness report for what was actually run at D=20.

## 5. What was deliberately NOT built in this pass

- A fast Architecture-B (bin-index-driven, no persistent dense CM matrix) **moment** construction
  for the combined CDF+POWER block — the Hessian (this document's subject, and the dominant
  per-callback cost at L=50) is fast/structured; the moment matrix itself is still built via the
  dense Architecture-A path (`build_cm_frechet_augmented_obj_basis`) for `:cdf_power`. At D=20/
  L=50 this is a `W×2000` (~1.28GB at W=80,000) matrix, built once per context (not per callback),
  which is plausibly fine but was not benchmarked against a hypothetical Architecture-B analogue
  in this pass — flagged as a disclosed follow-up, not silently skipped.
- An interval-basis (`:interval`) analogue of this combined structured kernel — the CDF-only
  interval kernel (`cm_frechet_bases_structured.jl`, ported unchanged from the archive) exists,
  but the combined CDF+POWER structured kernel built here is `:cumulative`-only
  (`build_cm_frechet_production_context` hard-errors otherwise for `:cdf_power`). See the basis
  decision in the port-readiness report.
