# Shared Economic FG Operator — Design — 2026-07-26

Branch `port/shared-inner-fg-operator-and-verification-2026-07-26`, base `production/fullA-exact@f1fa8e7`.

## 1. The core finding that shapes this whole design

Every restricted family's `moments!` closure (`wrap_moments_with_cm_archB`,
`wrap_moments_with_cm_frechet_archB`, `wrap_moments_with_cm_meanzc`, `wrap_moments_with_originzc`)
**already builds a `CompressedFactual` for the economic-core block on every outer point**, published
via a shared `core_cf_ref::Ref{Any}` — not for FG, but for the shared winner-pair Hessian backend
(`port/shared-winner-pair-core-hessian-production-2026-07-25`). This means the "shared economic
operator" the addendum asks for did not need to be built from scratch for the restricted families:
`compressed_dual_contraction!`/`compressed_transpose_contraction!` (Addendum Part A,
`compressed_moments.jl`/`compressed_cc_inner.jl`) already implement it, already validated to
machine precision, already the unrestricted family's own default FG path. The work was: (a) name it
as a shared interface (`economic_operator.jl`), and (b) point each restricted family's FG callback
at the SAME `cf` the Hessian already builds, instead of a dense `BLAS.gemv!` against `obj.H`.

## 2. A genuine scope boundary this finding also exposes

The production Hessian for every restricted family still needs *some* dense `H[:, 2:1+NCORE]`
(economic columns) — confirmed by reading both Hessian callbacks, not assumed:

- Flexible CM / common Fréchet: `hessian_cm_structured!` reads `E = @view H[:, 2:1+NCORE]` directly
  for `H_EC` (line `build_bin_tables!(cctx, E, w)`).
- Origin-ZC: `archA_partitioned_hess_cb_builder`'s `cf isa CompressedFactual` branch (the
  production default) uses the compressed winner-pair backend for `H_EE` only — but `H_ER`/`H_RR`
  still read `H_copy[:, 2:1+n] .= H[:, 2:1+n]` (full dense core+restriction columns) for the
  cross-term BLAS `gemm!`.

**This means dense `obj.H` economic columns continue to be built by `moments!` after this port** —
eliminating that would be Hessian cross-block rework, explicitly out of scope
("Do not expand this task into Hessian cross-block optimization"). What this port DOES eliminate is
the FG *callback's own* consumption of those dense columns — the same "producer still exists,
consumer stops using it" pattern already established for the CM block's `skip_cm_fill_ref`
(`RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md` §2). The no-dense-G counters
(`no_dense_g_counters.jl`) and their runtime proof doc are scoped accordingly — they prove the FG
*callback* path is dense-E-free, not that `obj.H`'s economic columns never exist in memory.

## 3. The shared operator interface

`economic_operator.jl`:

```julia
economic_forward!(out, lambda_E, cf::CompressedFactual, ws::EconomicFGWorkspace) -> out
economic_transpose!(grad_E, draw_weights, cf::CompressedFactual, ws::EconomicFGWorkspace) -> grad_E
```

Thin, explicitly-named wrappers around `compressed_dual_contraction!`/`compressed_transpose_contraction!`
— no new math, per the addendum's explicit "do not copy this algebra into family-specific files"
instruction: every family below calls these two functions directly. `economic_transpose!` returns
the raw `Σ_s w_s E_{s,j}` scatter (matching `compressed_transpose_contraction!`'s own contract) —
callers apply the family's own `-(1/M)` gradient scale themselves, so it composes with a
restriction operator's own transpose via plain accumulation into disjoint slices of the same
gradient buffer.

## 4. The ZC restriction operator (new)

`zc_restriction_operator.jl`: exact `R = Φ - 1t'` forward/transpose for the origin-specific
mean/pairwise-ZC restriction block (addendum §4's own formula, applied literally):

```
Rλ  = Φλ - 1(t'λ)         forward
R'v = Φ'v - t(1'v)        transpose
```

`Φ` = the immutable, theta-independent raw feature matrices (`Zraw_all[k]`/`Zpairraw_all[k]`,
built once per campaign). `t` = the per-outer-point target vector. Neither the centered `Φ-1t'`
matrix nor a temporary centered copy of `Φ` is ever constructed — and this is strictly cheaper than
the pre-existing status quo, which DID materialize a fresh centered `(W,D)`-or-`(W,npair)` block
(`dest = Z .- νtargets'`) on every outer point in both `wrap_moments_with_originzc` and
`wrap_moments_with_cm_meanzc`'s `moments!` closures.

Shared, unmodified, by both new operator families:

- **Origin-ZC**: targets from `OriginByPowerLayout` (`mean_targets`/`pair_targets`,
  `cm_originzc_target_layout.jl`) — one `nu_{o,k}` per origin.
- **CM+ZC**: targets from `SharedByPowerLayout(K_mean,K_pair)` — CM+ZC's own scalar-per-level
  `nu_k` (`mean_columns_direct!(dest,Z,ν::Float64)`) is exactly what `SharedByPowerLayout`'s
  `mean_targets`/`pair_targets` reduce to (broadcast `nu_k` to every origin/pair) — confirmed to
  machine precision by `test_meanzc_operator_correctness.jl`, not assumed from reading the two
  layouts' definitions.

## 5. Family composition (code column order, not the addendum's prose order)

| Family | Code column order | E | Restriction |
|---|---|---|---|
| Unrestricted | `[E]` | compressed (Addendum Part A) | — |
| Flexible CM | `[E\|C]` | dense (unchanged this branch) | CM bin-lookup (`cm_lookup_kernels.jl`, existing) |
| Common Fréchet | `[E\|C\|F]` | dense (unchanged this branch) | CM bin-lookup + level-anchor (`cm_frechet_lookup_kernels.jl`, existing) |
| CM+ZC | `[E\|Z\|C]` | **compressed (this branch, new)** | ZC operator (new) + CM bin-lookup (reused) |
| Origin-ZC | `[E\|Z]` | **compressed (this branch, new)** | ZC operator (new) |

Flexible CM's and common Fréchet's own economic-core block is **not** retrofitted to the compressed
operator by this branch — `CMLookupState`/`CMFrechetLookupState` are already the shipped-or-
validated default/near-default kernels for those two families, and retrofitting their E-block is
real, correctness-sensitive surgery on already-trusted code, explicitly the "harder ask" the
inherited handoff flagged as needing its own dedicated pass ("should not be rushed"). This branch
instead built the new operator on the two families that had **zero** prior FG alternative
(origin-ZC, CM+ZC) — both now genuinely compressed end-to-end for their E block — establishing the
pattern for a follow-on CM/Fréchet retrofit rather than attempting it in the same pass.

## 6. Two real bugs found and fixed while building the first operator (origin-ZC)

Both caught via this project's own "verify math independently at a fixed point before trusting a
downstream numerical discrepancy" discipline — a standalone `(x,g)` comparison against the dense
`obj(x,g)` callable, no KNITRO involved, run *because* the first live KNITRO solve failed in a
suspicious way, before assuming the kernel math was at fault:

1. **Missing `-1/M` gradient scale.** `economic_transpose!` intentionally returns the raw
   `Σ_s w_s E_{s,j}` scatter (docstring says so explicitly) — the restriction block's own transpose
   applies `-1/M` via the BLAS `alpha` argument, but the first draft of `OriginZCOperatorState`
   forgot the equivalent scale on the economic block. Off by a factor of `~M` (~8,000-80,000 at
   D=20) in the gradient's economic columns only. Caught by a fixed-point `(x,g)` unit check before
   any KNITRO run.
2. **A field-name collision.** `OriginZCCoreHessCtx.n_eta` (pre-existing field, = `n_mean+n_pair`,
   the RESTRICTION-COLUMN count the Hessian's `H_ER`/`H_RR` partition width needs) was wrongly
   reused to slice `νfull` out of `θ_ext` in the new FG dispatch. `θ_ext`'s trailing block actually
   has length `n_eta(layout)` (= `K_mean*D`, the eta/ν PARAMETER count — a same-named but
   different-valued quantity computed by a *function*, not this field). Silently corrupted `νfull`
   (grabbing part of `θ_econ` instead) whenever `n_mean+n_pair != n_eta(layout)`, i.e. whenever
   `K_pair>0`. Passed for `K_pair=0` by coincidence (`n_mean == n_eta(layout)` there); failed with
   KNITRO `nStatus=-400` for `K_pair=1` — caught by testing `K_pair>0` explicitly per the addendum's
   own "do not infer correctness from flexible CM"/test-every-config instruction, rather than
   assuming the `K_pair=0` case generalizes.

Neither bug recurred when building CM+ZC's own operator immediately after — both structural lessons
(apply the transpose scale; use the *function* `n_eta(layout)`, never a same-named field) 
transferred directly, and CM+ZC's D=4 gate passed 48/48 on the first run.

## 7. What this design explicitly does NOT claim

- Flexible CM's and common Fréchet's own economic-core block: still dense this branch (§5).
- A full-codebase audit of every dense-G consumer (task §13): not attempted — see
  `NO_DENSE_G_RUNTIME_PROOF_2026-07-26.md`'s own scope note.
- Operator-based verification (Job 2) removing `skip_cm_fill_ref`: see
  `OPERATOR_BASED_INNER_VERIFICATION_2026-07-26.md` for what was/wasn't done.
- Common-Fréchet's allocation regression root cause: see
  `COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md`.
