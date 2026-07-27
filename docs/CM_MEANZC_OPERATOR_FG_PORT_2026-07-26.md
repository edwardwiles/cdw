# CM+ZC Operator FG Port — Final Gate — 2026-07-26

## Summary

CM+ZC's `G=[E|Z|C]` (code column order; `[E|C|Z]` in the addendum's prose) inner FG had **no**
lookup/compressed alternative before this branch — 100% dense, per
`RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md` §5: "its entire inner FG callback is one
dense `BLAS.gemv!` against the full widened `obj.H`". This port adds `CMMeanZCOperatorState`
(`cm_meanzc_lookup_kernels.jl`) + KNITRO wiring (`cm_meanzc_lookup_production.jl`), composing
**three** already-validated pieces with no new algebra:

- **E**: the SAME `economic_forward!`/`economic_transpose!` origin-ZC now uses, against
  `cctx.core_cf_ref[]`.
- **Z**: origin-ZC's own `ZCRestrictionOperator`/`restriction_forward!`/`restriction_transpose!`,
  UNCHANGED, constructed with `SharedByPowerLayout(K_mean,K_pair)` instead of `OriginByPowerLayout`
  — CM+ZC's own scalar-per-level `ν_k` targets are exactly what `SharedByPowerLayout`'s
  `mean_targets`/`pair_targets` reduce to (broadcast `ν_k` to every origin/pair).
- **C**: `cm_lookup_kernels.jl`'s existing bin-lookup kernels (`apply_contrast!`/`suffix_sums!`/
  `build_weighted_histogram!`/`cumulative_backward_gradient!`), called verbatim — the SAME
  functions `CMLookupState` already uses for flexible CM's own CM-grid block.

Opt-in via `inner_fg_backend=:operator` on `CMBinHessCtx` (two new fields, `meanzc_zc_op`/
`meanzc_zc_layout`); default stays `:dense_reference`. Reuses the existing `cmlookup_st::Any` cache
field (already shared between plain-CM's `CMLookupState` and common-Fréchet's
`CMFrechetLookupState`, per that field's own established convention — "a `cctx` only ever belongs
to one family, no collision risk") for `CMMeanZCOperatorState` too.

## Bugs found

**None new.** Built immediately after origin-ZC's port; both structural lessons from origin-ZC's
two bugs transferred directly on the first attempt:

- The economic block's `-1/M` transpose scale is applied correctly from the start.
- The ν-slicing uses `n_eta(cctx.meanzc_zc_layout)` (the function), never a same-named-but-
  different-meaning field.

## D=4 correctness gate (`test_meanzc_operator_correctness.jl d4`)

`K_mean/K_pair ∈ {(1,0), (1,1), (2,2)}`, `L=10`, calibration + perturbed points:

```
48/48 PASS — ALL PASS on the first run
```

`zeta*`/`lambda*`/`m_star`/`Delta_dual`/downstream `(g,A_od,ν)` gradient agree to ~1e-12 to
~1e-18. Zero dense-economic-fallback calls throughout.

## D=20/W=80,000 correctness gate (`test_meanzc_operator_correctness.jl d20`)

Real omit-ROW, seed `20260719`, `L=50`, `K_mean=1/K_pair=1`, calibration + perturbed points:

```
16/16 PASS
```

Same agreement levels (~1e-9 to ~1e-16). Zero dense-economic-fallback calls.

## Operator-based verification

Not built for CM+ZC this session (origin-ZC's own `verify_inner_solution_operator_originzc!` was
the scoped proof of concept) — a CM+ZC analog would additionally need the CM-grid block's own
transpose contribution folded into the KKT residual (straightforward, reusing
`cumulative_backward_gradient!` the same way `restriction_transpose!` is reused, but not attempted
in the time available).

## Flip decision

**NOT flipped to default this session** — same reasoning as origin-ZC: correctness is fully gated
at both scales, but the performance A/B this session did not run is required before a default
change (task §16/§19).
