# Origin-ZC Operator FG Port — Final Gate — 2026-07-26

## Summary

Origin-ZC's `G=[E|Z]` inner FG had **no** lookup/compressed alternative before this branch — 100%
dense (`inner_loop_internal_archgeneric`). This port adds `OriginZCOperatorState`
(`cm_originzc_lookup_kernels.jl`) + KNITRO wiring (`cm_originzc_lookup_production.jl`), composing:

- **E**: `economic_forward!`/`economic_transpose!` (`economic_operator.jl`) against
  `octx.core_cf_ref[]` — the `CompressedFactual` `wrap_moments_with_originzc`'s `moments!` closure
  already builds for the shared winner-pair Hessian backend.
- **Z**: `restriction_forward!`/`restriction_transpose!` (`zc_restriction_operator.jl`, new) — exact
  `R=Φ-1t'` operator against the immutable raw feature matrices.

Opt-in via `fg_backend=:operator` on `OriginZCCoreHessCtx` (new fields); default stays
`:dense_reference`.

## Bugs found and fixed (both real, both caught before any gate reported false success)

1. **Missing `-1/M` gradient scale on the economic block** (`economic_transpose!` returns the raw
   scatter by design; the caller must apply the scale). Caught by a fixed-`(x,g)` unit check
   against the dense `obj(x,g)` callable, before any KNITRO run — gradient off by a factor of `~M`
   in the economic columns only.
2. **`OriginZCCoreHessCtx.n_eta` field-name collision** — a pre-existing field meaning
   "restriction-COLUMN count" (`n_mean+n_pair`, needed by the Hessian's `H_ER`/`H_RR` partition
   width) was wrongly reused to slice `νfull` out of `θ_ext`, which actually needs `n_eta(layout)`
   (the eta/ν PARAMETER count, `K_mean*D`) — a same-named, different-valued quantity computed by a
   function. Silently corrupted `νfull` whenever `K_pair>0` (`n_mean==n_eta(layout)` only when
   `K_pair=0`, masking the bug at that one config). Manifested as KNITRO `nStatus=-400` at
   `K_pair=1` — the isolated `(x,g)` unit check (100 random points, wide scale range) had already
   passed exhaustively at THIS SAME config, proving the kernel math itself was never wrong; the bug
   was purely in how `θ_ext` was sliced before the kernel ever saw it. Fixed by using
   `n_eta(octx.fg_layout)`.

## D=4 correctness gate (`test_originzc_operator_correctness.jl d4`)

`K_mean/K_pair ∈ {(1,0), (1,1), (2,2)}`, calibration + perturbed points:

```
48/48 PASS
```

`zeta*`/`lambda*`/`m_star`/`Delta_dual`/downstream `(g,A_od,eta)` gradient agree to ~1e-12 to
~1e-18 (machine precision, this codebase's own established tolerance convention given different
KNITRO iteration paths). Zero dense-economic-fallback calls throughout (operator took the
compressed-`cf` path every time).

## D=20/W=80,000 correctness gate (`test_originzc_operator_correctness.jl d20`)

Real omit-ROW, seed `20260719`, `K_mean=1/K_pair=1`, calibration + perturbed points:

```
16/16 PASS
```

Same agreement levels (~1e-9 to ~1e-16). Zero dense-economic-fallback calls.

## Operator-based verification proof of concept

`verify_inner_solution_operator_originzc!` (`operator_verification.jl`) independently recomputes
the KKT residual (stationarity check) from operator state only, agreeing with the pre-existing
dense verifier's `kkt_residual_blas` to ~1e-15 to ~1e-16 across all three D=4 configs (15/15 checks
pass, `test_operator_verification_originzc.jl`).

## Performance

Not separately benchmarked this session (time-bounded; see
`FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB_2026-07-26.md` for what was and wasn't measured). The
economic block reuses the SAME `compressed_dual_contraction!`/`compressed_transpose_contraction!`
already measured at 20,081x allocation reduction / 1.15x speedup for the unrestricted family
(Addendum Part A); the Z-block operator avoids a dense `(W,D)`-or-`(W,npair)` centered-matrix
materialization the dense-reference `moments!` path performs every outer point, so directional
allocation improvement is expected but not independently gated here.

## Flip decision

**NOT flipped to default this session.** `fg_backend=:operator` ships available, opt-in, D=4+D=20
correctness-validated. Promoting to default requires the performance A/B this session did not run
(task §16/§19's "complete inner solve faster or within 5%, allocation improves materially" bar) —
left for a follow-on gate.
