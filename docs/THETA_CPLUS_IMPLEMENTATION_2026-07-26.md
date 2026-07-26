# Theta C+ fixed-dual evaluator — implementation, 2026-07-26

Source: `full_aod_diag/d4_exact/theta_cplus.jl` (new file). Wired into `c10_d20_production_driver_unified.jl`'s `cb_G!` flexible-mode block (`run_polish_checkpointed_unified`). Math: `docs/THETA_CPLUS_MATHEMATICAL_DERIVATION_2026-07-26.md`.

## What changed vs the brute-force path

The old `theta_fixed_dual_delta_pivot_A` (`flexible_theta_aspace_production.jl`, **left in place,
unmodified** — retained as the correctness reference for `test_theta_cplus_correctness.jl`, no
longer called by any production driver) routed every probe through the generic dense
`obj.moments!`/`CS.reconstruct_full`/`obj(x)` pipeline. The new path
(`theta_cplus_probe_value`/`theta_cplus_secant`) routes through
`build_compressed_factual!`/`compressed_cc_value_grad` — the same already-existing, already-
validated compressed winner-form machinery `screened_eval`'s `DualBank` scoring and the Hessian-
callback adapter already use elsewhere in this codebase, now applied to the theta-secant probes
too.

## `ThetaCPlusWorkspace`

```julia
mutable struct ThetaCPlusWorkspace
    cf_ws_plus::CompressedFactualWorkspace    # campaign-lifetime, W x Ddest winner/wval buffers
    cf_ws_minus::CompressedFactualWorkspace   # a SEPARATE buffer, not shared with cf_ws_plus,
                                               # so plus/minus winners can be diffed after both
                                               # probes complete (task §12's winner-change count)
    h_theta::Float64
    n_calls::Int
    n_winner_changes_total::Int
    generic_moments_calls::Int                # always 0 in production use of this path
    check_ties::Bool
end
```

Built once per outer campaign (`build_theta_cplus_workspace(D, Ddest, W)`, called in
`run_polish_checkpointed_unified`'s setup, only when `layout.trade_elasticity_mode==:flexible`).
Two `CompressedFactualWorkspace`s (not one) were chosen deliberately — reusing a single workspace
sequentially for plus-then-minus would overwrite the plus winner state before the winner-change
diagnostic could compare them; the doubled buffer cost is ~24MB one-time (campaign-lifetime, not
per-callback), a rounding error against the ~2.2GB *per callback* the old path allocated.

## What is NOT allocated per callback (task §6's explicit prohibition list, checked against the
actual implementation)

- `copy(w)`/`w_plus`/`w_minus`: **not present**. `theta_cplus_secant` reads `gp=w[2]`,
  `a_nonpivot=@view w[3:end]` (a view, zero-copy) and only the scalar `eta_theta=w[1]` is
  perturbed for each probe.
- Full `theta_full_plus`/`theta_full_minus`/`H_plus`/`H_minus`/perturbed moment matrices: **not
  present**. `decode_theta_probe` allocates only O(D·Ddest)-scale vectors (`z_nonpivot`,
  `Aod_levels`, `xf`, each length ≤382); `build_compressed_factual!` writes into the pre-allocated
  `cf_ws_plus`/`cf_ws_minus` buffers, never allocating a fresh `Matrix{Int}(undef,W,Ddest)` etc.
- New W-scale arrays per callback: **not present** beyond the above (`materialize_dense_factual!`-
  style dense reconstruction is never invoked).
- `build_pivot_elimination_cheap` call inside the theta path: **not present**. `decode_theta_probe`
  takes the already-built `pgc::PivotGravityElimCache` as an argument and calls
  `pivot_expand_cheap` (the O(1)-analytic-fit path) directly.

Residual small per-callback allocations (not W-scale, not targeted for removal): `z_nonpivot`
(~379 floats), `Aod_levels`/`xf` (~382 floats) per probe, and whatever
`canonical_price_precompute`'s own `constCons`/`constConsσ`/`logCC`/`wPow` arrays cost internally
(D×Ddest≈380-element, shared with every other caller of that function — `mulU`/`UPow`/`UσPow` are
already workspace-cached via `ctx.canonical_price_ws`, attached in this port's Phase 1
reconciliation, which `build_compressed_factual!` picks up automatically since it calls the same
function). These account for the workspace's own residual ~3.7MiB/probe measured in
`THETA_CPLUS_ALLOCATION_PROFILE_2026-07-26.md`, not chased further given the >99% reduction
already achieved.

## The eliminated third reconstruction (task §8)

`cb_G!`'s old flexible-mode block ended with a THIRD call —
`ctx.obj.moments!(@view(ctx.obj.H[:,1]), ..., θ_full_base, ...); ctx.obj.H[:,2].=1.0` — after the
two secant probes, at the theta_base point, writing into `ctx.obj.H`. Traced to its origin
(`PsiObjectiveBundleImplicitMethodBFullA.jl::inner_loop_internal`, the *canonical* place this
exact two-line pattern comes from): `obj.H` is a **mutable, campaign-lifetime scratch buffer**
that `inner_loop_internal` unconditionally repopulates from scratch at the **start of every inner
KNITRO solve**, and `hessian!` (`obj.H`'s only reader anywhere in this codebase) is invoked
**only from inside that inner-solve context** — never from the outer gradient callback. `cb_G!`'s
copy of the write therefore had no downstream consumer: nothing reads `obj.H` again before the
next real inner solve unconditionally overwrites it. **Removed entirely**, not replaced with a
faster equivalent — there was nothing to preserve. This is the single largest contributor to the
theta-block speedup (the old block's 3 reconstructions vs the new block's 2 probes — see the
allocation-profile doc for the per-piece breakdown).

## Addendum compliance (mid-task correctness requirement)

Every θ probe gets a full, exact `build_compressed_factual!` call — an unconditional re-scan of
all `D` origins for every `(draw, active-destination)` cell, at the probe's own theta. No
runner-up/top-3/largest-`U` shortcut anywhere in this file. No stability-radius-gated selective
rescan, no optional "analytic stable-winner" backend — the original task brief's §4/§9 stability-
certificate design was implemented in the math derivation doc as a documented-but-unused formula
only, per the user's explicit mid-task instruction to remove it as a code path. The speedup is
entirely attributable to *how* the exact rescan is computed (compressed winner-form vs dense
matrix materialization), never to computing less.
