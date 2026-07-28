# Gravity pivot vs. gravity moment — the actual algebra (2026-07-28)

This document traces, in full, the exact equations implemented by (1) the outer gravity equality
that `gravity_elimination.jl`'s pivot reparameterization eliminates, (2) the pivot reconstruction
itself, and (3) the inner-solve gravity moment (`compressed_gravity_raw`/`newGravityMoment!`/
`fill_gravity_column_into!`). No step here is inferred from names — every equation is read directly
from the implementing code, cited by file:line, and independently re-verified numerically (see
`GRAVITY_PIVOT_VALID_COORDINATE_TESTS_2026-07-28.csv` and the accompanying decision document).

## 1. The outer gravity equality (pre-elimination)

Active cells: all origins `o = 1..D` and all destinations `d = 1..Ddest`, where `Ddest = D` unless
`:exclude_row` mode drops ROW as a destination (`_ctx_ddest(ctx)`,
`full_aod_diag/d4_exact/gravity_elimination.jl:30`). No ROW-as-origin exclusion; ROW enters only as
a possible destination exclusion, per the existing `:exclude_row` production mode.

Let `z[o,d] := log(Aod_theta[o,d])` (log of the outer productivity/cost-shifter parameter — NOT
level `A_od`, and NOT the model's dense trade-flow matrix). Let `q̃[o,d]` be the two-way (origin +
destination fixed-effect) residualized log-tariff/cost regressor computed once from data
(`precompute_q_tilde`, feeding `ctx.q_tilde`), and `N_obs = D*Ddest`. Then, with `μ` the Fréchet
shape parameter (`ctx.fixed_vals[1]` in fixed-theta mode, the current base point's `μ` in
flexible-theta mode):

```
c[o,d]  = μ · q̃[o,d] / N_obs                          (gravity_linear_coeffs, :25-27)
g0      = g_gravity(z=0)                                (gravity_offset, :51 -- g_gravity AT Aod_theta≡1,
                                                          an arbitrary affine-map INTERCEPT, NOT calibration)
g_gravity(z) = Σ_{o,d} c[o,d]·z[o,d] + g0
```

`g_gravity` is verified (not assumed) to be **exactly affine in z** — `gravity_from_logz`
(`gravity_elimination.jl:38-48`) reconstructs the full nonlinear price/level system at `z` and
evaluates the true nonlinear gravity functional (`gravity_value`, see §3), and the closed-form
gradient `∂g_gravity/∂Aod_theta[o,d] = (q̃[o,d]/N_obs)·(μ/Aod_theta[o,d])` (chain rule through
`z=log(Aod_theta)`) gives the constant `∂g_gravity/∂z[o,d] = μ·q̃[o,d]/N_obs`, independent of
`Aod_theta`'s value — hence affine, not merely locally linearized.

**The outer gravity equality being eliminated is `g_gravity(z) = 0`.**

## 2. The pivot reconstruction

`build_pivot_elimination(ctx)` (`gravity_elimination.jl:71-78`) picks the pivot cell
`pivot = argmax|c|` (largest-magnitude gravity coefficient — "not near zero," so the solve in the
next step is never near-singular) and returns `PivotGravityElim(D, Ddest, pivot_lin, c, g0,
other_idx)`, `other_idx` = the remaining `D*Ddest - 1` linear indices.

`pivot_expand(z_free, pe)` (`gravity_elimination.jl:81-89`):
```
z[other_idx] = z_free                                       # every non-pivot cell: literal copy
z[pivot]     = ( -g0 - Σ_k c[other_idx[k]]·z_free[k] ) / c[pivot]   # pivot: solved algebraically
```
This makes `g_gravity(z) ≡ 0` **exactly**, for **any** choice of `z_free` — not just at calibration.
This is a genuine, general elimination (confirmed algebraically: substitute the pivot formula back
into `g_gravity(z) = Σc·z + g0` and every term cancels to `0`, independent of `z_free`'s value).
`pivot_reduce` is the exact inverse (drops the pivot coordinate; `:92`).

A cached, theta-invariant variant (`build_pivot_elimination_cheap`/`pivot_expand_cheap`,
`:153-224`) implements the identical map without re-deriving `pivot`/`other_idx`/`c` on every theta
probe (theta-invariance holds because `c(μ) = μ·q̃/N_obs` is exactly linear in `μ>0`, so `argmax|c|`
and the reconstruction slope `-c0[j]/c0[pivot]` never depend on `μ`; only the affine offset `g0(μ)`
does, fit from two probes). Mathematically identical output to `build_pivot_elimination`/
`pivot_expand` (not a separate algorithm).

## 3. The inner-solve gravity moment

`compressed_gravity_raw(θ_full, ctx)` (`full_aod_diag/d4_exact/compressed_live.jl:104-125`):
```
AodPow = aod_pow_matrix(θ_full, ctx)     # θ_full's A_od block (LEVELS, not log) -> AodPow, the
                                          # level->power transform every moment/screen consumes
                                          # (compressed_live.jl:85-91; same formula as
                                          # gravity_from_logz's own Aod_lvl/AodPow reconstruction,
                                          # gravity_elimination.jl:45-46 -- literally the same
                                          # expression, re-derived independently in each file)
grav_raw = newGravityMoment!(...)        # moments/newGravityMoment!.jl, UoModel==1 branch:
                                          #   Wτ  = within_transform_rect(τ)        (two-way FE-demean)
                                          #   WA  = within_transform_rect(AodPow)
                                          #   grav_raw = Σ_{o,d} Wτ[o,d]·WA[o,d]    (raw, unnormalized,
                                          #                                          unnegated)
```
`fill_gravity_column!`/`fill_gravity_column_into!` (`compressed_live.jl:138-147`,
`cm_hessian_architectures.jl:171-179`) then apply the SAME SamplingWeights/NormalizeMoments/PMM
post-processing every other inner moment column gets:
```
column[w] = SamplingWeights[w] · nrm_g · (grav_raw - pmm_g)
nrm_g = 1/σ_Moments[d_gravity]   (if NormalizeMoments==1 and this moment has variance)
pmm_g = PMM[d_gravity]           (if usePMM==1; ≈ 0 empirically for this moment, see below)
```
`d_gravity = obj.d` (the LAST inner moment index — gravity is the final column of the dense
`[K|ones|G]` layout, `obj.H[:, 2+d]`), so this is one ordinary equality moment inside the per-draw
inner (ζ,λ) divergence-minimization problem, dualized with its own λ multiplier exactly like every
other moment — **not** a constraint on the outer θ directly.

**Compare `gravity_value` (`full_aod_diag/gravity_tariff.jl:80-89`, used by `gravity_offset`/
`gravity_from_logz` to build `c`,`g0`)**:
```
gravity_value(τ, AodPow, q_tilde, N_obs) = -( Σ_{o,d} within(τ)[o,d]·within(AodPow)[o,d] ) / N_obs
```
**The core summand `Σ within(τ)·within(AodPow)` is the identical computation** in both places (same
active `D×Ddest` grid, same two-way FE-demeaning, same sign convention on the accumulation) — it is
literally the same orthogonality condition (`within-log-tariff ⟂ within-log-A_od-power`), evaluated
twice, independently, in two different files. The two differ only in **scale/normalization**:
`gravity_value` applies `-1/N_obs` (chosen so its own zero-set, used to build the pivot's affine
`c`,`g0`, is `sumGrav=0`); `compressed_gravity_raw` returns the raw, un-negated, un-scaled
`sumGrav`, which then receives a **third**, unrelated normalization (SamplingWeights × `1/σ_Moments`
× PMM-centering) before it becomes the actual KNITRO-facing moment column. All of these are affine
rescalings of the SAME underlying zero-set (`sumGrav=0` ⟺ `g_gravity(z)=0`, up to the empirically
near-zero `pmm_g` recentering — confirmed numerically, `PMM[d_gravity] ≈ 7.05e-17` at this
context's calibration, i.e. essentially the literal `0` target, not a nonzero empirical offset).

**Conclusion of the algebra alone: the inner gravity moment and the pivot's eliminated outer
equality are the SAME zero-set, evaluated through different (but exactly related, non-approximate)
normalizations of the identical `Σ within(τ)·within(AodPow)` orthogonality condition.** Whether this
makes the inner moment *redundant* depends entirely on whether a given caller's `θ_full` was
actually built via `pivot_expand` — see `GRAVITY_MOMENT_FINAL_DECISION_2026-07-28.md` for that
resolution, which is NOT visible from the algebra alone and required tracing the actual production
call graph (`CS.reconstruct_full` vs. `run_cm_upper_checkpointed`'s own `pivot_expand` wrapping).
