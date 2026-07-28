# Flexible-CM no-H bundle gate (2026-07-28)

## What was built

`full_aod_diag/d4_exact/operator_psi_bundle.jl`'s `OperatorPsiBundle` — restructured from the
inherited draft (which had a `K` field and was never actually `include`d) into the real,
load-bearing production type:

- **No** `H`, `H_copy`, `moments!`, `jac_h` fields (inherited draft already had this right).
- **No** `K` field — renamed `payoff` (the forbidden-field list names `K` itself, not just the
  legacy `[K|ones|G]` layout).
- **No** `ones` field, **no** `select_G_from_H`-reachable state (`select_G_from_H(::OperatorPsiBundle,
  ...)` throws; `obj.H`/`obj.moments!` throw via Julia's own default `getproperty` since the fields
  genuinely don't exist).
- `economic_state`/`restriction_state` fields added: **aliases**, not new storage —
  `economic_state` is literally the same `Ref{Any}` box as the owning `CMBinHessCtx.core_cf_ref`;
  `restriction_state` is the owning `CMBinHessCtx` itself. See `operator_psi_bundle.jl`'s header for
  why this, not a `dual_workspace`/`hessian_workspace` struct-of-structs, is the right shape given
  the shared Hessian/verification code's existing unconditional flat-field reads.

## Wiring (construction → priming → FG → Hessian → verification)

- `build_cm_production_context(...; moment_representation=:operator)` (new kwarg, production
  default) constructs `OperatorPsiBundle`; `:dense_reference` (explicit) constructs the unchanged
  `PsiObjectiveBundleImplicit`. Neither the `wrap_moments_with_cm_archB` closure nor `moments!` is
  even built in `:operator` mode.
- `inner_loop_internal_cmlookup_production` dispatches priming on `obj isa OperatorPsiBundle` →
  `prime_operator!(obj, θ, cctx.econ_ctx, cctx.core_cf_ref; restriction_state=cctx)` (new
  `CMBinHessCtx.econ_ctx` field carries the base `ctx` `prime_operator!` needs).
- FG: `CMLookupState`'s existing functor (`cm_lookup_kernels.jl`) already dispatches on
  `st.core_cf_ref[] isa CompressedFactual` for its economic block (`economic_forward!`/
  `economic_transpose!`, zero dense `obj.H` reads) vs. a dense `obj.H` BLAS fallback used only when
  `core_cf_ref[]` is not yet a `CompressedFactual`. Since `prime_operator!` unconditionally
  publishes a real `CompressedFactual` every priming call, the dense fallback branch is never
  reached for `OperatorPsiBundle` in practice — **zero changes needed** to this file.
- Hessian: `hessian_cm_structured!`/`_fill_cm_HEE!`/`build_bin_tables!` already dispatch via
  `_dense_H_or_nothing(obj)` (`nothing` for `OperatorPsiBundle`, `obj.H` for the dense type) —
  **zero changes needed**; only fix required was actually `include`-ing `operator_psi_bundle.jl`
  (was dead code before this session).
- Verification: `verify_inner_solution_operator_cm!` already takes `obj::Any` and only reads
  `obj.Psi!`/`obj.dPsi!` (both flat fields on `OperatorPsiBundle` too) — **zero changes needed**.

## Gate results (real KNITRO, D=4)

`test_operator_no_H_bundle_equivalence_flexcm.jl`, side-by-side `:operator` vs `:dense_reference`
contexts, identical draws/outer params/KNITRO options/threads:

```
Structural: fieldnames(OperatorPsiBundle) contains none of H/H_copy/G/K/ones/jac_h/moments! -- PASS
obj.H throws -- PASS.  select_G_from_H(obj, ...) throws -- PASS.

Fixed-point callbacks (x=0, 4 random duals, real solved dual): objective/gradient/packed-Hessian
  agree EXACTLY (0.0 abs diff) at every point.

Full inner solve from calibration: same accepted status (nStatus=0 both), delta*(zeta*) and dual
  vector (lambda*) agree EXACTLY (0.0 abs diff) -- better than the 1e-12/1e-10 target.
```

`test_operator_bundle_allocation_proof_flexcm.jl`: `bundle_type=OperatorPsiBundle`,
`has_H_field=has_H_copy_field=has_moments_field=has_K_field=has_ones_field=false`,
`inner_status_calib=0`.

## Scope

Flexible CM only. The other 4 families (unrestricted, common Fréchet, CM+ZC, ZC-only) were NOT
wired to `OperatorPsiBundle` this session — see `FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md` and
the master report for the concrete, evidence-based per-family assessment of what remains.

## Update: real D=20/W=100,000 confirmation

Re-run at real production scale (`d20_real_setup(W=100_000, destination_sample=:exclude_row)`,
L=50, threaded_bins=true, the actual production default): both dense-reference and operator
bundles reached matching nStatus=-103 (a feasible accepted status), with delta*/dual vector
agreeing exactly (0.0 abs diff) and FG/Hessian callbacks agreeing exactly at x=0 and 2 random
dual points. Operator-bundle solve was ~2.6x faster wall-clock (9.9s vs 25.4s) at this scale.
See `test_operator_no_H_bundle_equivalence_flexcm_d20.jl` and
`FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md` for the full cross-family table.

## Scope (superseded)

The "flexible CM only" scoping note below is superseded -- all five families now have this same
wiring, gated the same way, all passing. See `FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md`.
