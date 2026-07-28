# FLAGGED FINDING: gravity moment is zero only AT calibration, not identically zero — 2026-07-28

**Status: flagged for follow-up, not investigated to root cause.** Found incidentally during this
session's legacy-H storage cleanup while checking whether `fill_gravity_column_into!`'s "gravity
column" (one of the moment columns every restricted family's priming closure fills,
`wrap_moments_with_cm_archB` and its 3 siblings) was legacy/removable, per a direct question raised
mid-session. It is **not** part of the storage-cleanup work itself and was not touched — recorded
here purely because the numerical behavior did not match the expected identity and the user asked
for it to be written up explicitly.

## The claim tested

Given this codebase's A_od exact-elimination / pivot reparametrization (`gravity_elimination.jl`,
referenced throughout `CLAUDE.md`'s own standing operational notes), the expectation was that the
gravity moment condition should be satisfied **identically** by the reparametrization — i.e. the
per-draw gravity column (`Gcol` below) should be a vector of machine-precision zeros for *any* θ
in the free-parameter space that the pivot construction produces, not merely at the fitted
calibration point.

## What was measured

D=4 diagnostic setup (`d4_exact_setup(δ=1.0, find_smallest=true)`), `ctx.γ.indicators.gravMoment
== 1` (confirmed: this moment is active in this configuration, not disabled).

```julia
grav_raw = compressed_gravity_raw(θ_full, ctx)   # compressed_live.jl -- calls aod_pow_matrix(θ_full, ctx)
fill_gravity_column_into!(Gcol, grav_raw, ctx, d_idx)   # cm_hessian_architectures.jl -- per-draw broadcast
```

| Point | `grav_raw` (pre-broadcast scalar) |
|---|---|
| Exact calibration (`x_free_calib = ctx.θ0_up[ctx.free_idx]`, reconstructed via `CS.reconstruct_full`) | `5.1153295135411673e-17` — machine-precision zero |
| 5% random perturbation of `x_free_calib` (`x_free_calib .* (1.0 .+ 0.05 .* randn(...))`, seeded) | `0.0030620986076784545` — **not** machine precision; ~10 orders of magnitude larger than the calibration value |

`Gcol` itself (the per-draw broadcast, `γo.SamplingWeights[1:W] * nrm_g * (grav_raw - pmm_g)`) is
correspondingly a vector of machine-precision zeros at calibration and a vector of genuinely
nonzero values (~1e-3 scale, uniform sign/magnitude pattern from the `grav_raw` scalar times
per-draw sampling weights) away from it.

## Why this is flagged, not just noted

The reparametrization's whole *point*, per the user's own understanding of the design, is that the
gravity condition should hold as a structural identity of the pivot construction — true everywhere
in the reparametrized space the outer optimizer searches, not just at the eventual solution. What
was measured instead is consistent with the gravity condition being satisfied **only at the fitted
point** — i.e. behaving like an ordinary moment condition that the outer GMM/calibration procedure
drives to zero by choice of θ, not like an identity the reparametrization enforces independent of
θ. Those are two different things, and if the design intent was the latter, this is a real
discrepancy worth investigating — either in `aod_pow_matrix`/`compressed_gravity_raw`'s own
construction, or in how the pivot reparametrization's coverage of this specific moment was
intended to work.

## What was NOT done

- No investigation into *why* `grav_raw` is nonzero away from calibration (would require reading
  `aod_pow_matrix`, `newGravityMoment!`, and the pivot/elimination construction itself in detail —
  explicitly out of this session's scope, which was legacy dense-storage removal, not outer-loop/
  gravity-elimination correctness).
- No check of whether this is expected/intended behavior that the person who built the pivot
  construction would recognize as correct (e.g., if the reparametrization only guarantees the
  identity holds at the calibrated A_od level specifically, by construction, and genuinely varies
  away from it by design — plausible, but not confirmed either way here).
- No check of whether this has any live consequence today (e.g., whether the outer-loop A-gradient
  or bounds-screening code already accounts for a nonzero gravity residual away from calibration,
  or whether something implicitly assumes it is always zero and could be getting a wrong answer as
  a result).

## Reproduction

```julia
# From full_aod_diag/d4_exact/, with context.jl, gravity_elimination.jl, compressed_live.jl,
# cm_lookup_kernels.jl, cm_hessian_architectures.jl included:
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
grav_raw = compressed_gravity_raw(θ_full_calib, ctx)   # ~5e-17

x_free_pert = x_free_calib .* (1.0 .+ 0.05 .* randn(MersenneTwister(1), length(x_free_calib)))
θ_full_pert = CS.reconstruct_full(x_free_pert, ctx.m)
grav_raw_pert = compressed_gravity_raw(θ_full_pert, ctx)   # ~0.003
```
