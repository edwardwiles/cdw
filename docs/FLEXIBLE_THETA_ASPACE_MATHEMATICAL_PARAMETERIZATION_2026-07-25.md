# Flexible-theta a-space mathematical parameterization — production port, 2026-07-25

Task §3 deliverable. Recovers the exact math from
`experiment/fullA-theta-aspace-reparam-2026-07-25` (commit `7e38940`,
`full_aod_diag/d4_exact/flexible_theta_aspace.jl`) and re-derives it against current production's
rectangular (post-omit-ROW, `D` origins × `D_dest` active destinations) layout. Cross-checked
against current production's own `gravity_elimination.jl::gravity_from_logz` — both give the
identical `AodPow` formula, confirming this is not a re-derivation invented for this port but the
same object current production already computes internally at every gravity-offset evaluation.

**Reminder (repo CLAUDE.md standing warning):** nothing in this document treats `A_od ≡ 1`
(`zfree=0`) as "the calibration point." `zfree=0` appears below only as the coordinate origin of an
affine reparametrization map (`gravity_offset`), the same legitimate usage current fixed-theta
production already makes of it in `gravity_elimination.jl`. The genuine calibration point is
`ctx.θ0_up`'s own `A_od` block, reconstructed and checked for equivalence in §7 below.

## 1. Objects

- **Trade elasticity**: `theta` (`> 2(σ-1)` for finite variance), `mu = 1/theta`.
- **z-space (current production's searched A coordinate)**: `z[o,d] = log(Aod_theta[o,d])`, where
  `Aod_theta` is the object `gravity_elimination.jl` calls `Aod_θ` — the free A-cell coordinate
  KNITRO's outer NLP has always searched over, at fixed theta.
- **Economic productivity level**: `A_od` (a.k.a. `Aod_lvl` in `gravity_elimination.jl`), the
  genuinely calibrated object living in `ctx.θ0_up`'s A-block (span ~11 orders of magnitude at real
  D=20 data — see CLAUDE.md).
- **AodPow (the new searched object, `a := log(AodPow)`)**: the *theta-independent-in-its-own-scale*
  transformed productivity object that actually enters `compressed_moments.jl::build_compressed_factual`'s
  price/winner formula. Defined (rectangularized, `o` = origin `1..D`, `d` = active-destination slot
  `1..Ddest`):

  ```
  X[o,d] = (wHat[o] * τ[o,d]) / (wHat[1,1] * τ[1,d])
  Y[o,d] = lambda[o,d] / lambda[1,d],   lambda = reshape(γ.P, (Ddest, D))'
  AodPow[o,d] = (Aod_theta[o,d] * Y[o,d])^(-mu) / X[o,d]
  ```

  `X`, `Y` are pure **data** (theta/gp/A-independent) — `cHat` cancels out of `AodPow` entirely
  (verified algebraically: `gravity_elimination.jl::gravity_from_logz` computes
  `Aod_lvl = Aod_θ .* cHat .* X.^(1/μ) .* Y`, `AodPow = (Aod_lvl./cHat).^(-μ) = (Aod_θ.*X.^(1/μ).*Y).^(-μ) =
  (Aod_θ.*Y).^(-μ) .* X.^(-1)`, exactly matching the formula above — `cHat` divides out before the
  `^(-μ)` power is taken). This is the SAME `X`/`Y`/`lambda` this port's `precompute_aspace_XY`
  computes, cross-checked against production's own `gravity_from_logz` rather than re-derived from
  the (pre-omit-ROW) experimental source alone.

## 2. The a↔z map (exact, pointwise affine at fixed theta)

```
a[o,d] = log(AodPow[o,d]) = -mu*(z[o,d] + logY[o,d]) - logX[o,d]
       = -(z[o,d] + logY[o,d])/theta - logX[o,d]                      (a_from_z)

z[o,d] = -theta*(a[o,d] + logX[o,d]) - logY[o,d]                       (z_from_a)
```

Both directions are exact algebraic inverses (no linearization) — `flexible_theta_aspace_production.jl`
implements `a_from_z`/`z_from_a` verbatim as broadcast one-liners; round-trip is verified to
`< 1e-10` (float roundoff only) at four theta values including `theta_min`, `theta_star`,
`theta_max` in `test_flexible_theta_aspace_d4.jl` gate 1, on BOTH the square and rectangular
(`D=4, D_dest=3`) samples (gate 7a).

`dz/da = -theta` — a single **scalar**, identical for every `(o,d)` cell, independent of `a` itself.
This scalar is what lets the existing z-space chain-rule gradient be reused verbatim (§5).

## 3. Why the old z-space parametrization entangled theta and A

Searching directly over `z = log(Aod_theta)` at fixed `a`-equivalent-intent means: at fixed `z`, a
theta move changes `AodPow = (exp(z)·Y)^(-mu)/X` through `exp(z)^(-mu) = exp(-mu·z)` — i.e. theta's
entire effect on the economically-relevant object is the single multiplicative factor
`exp(-mu·z)`, a **cell-magnitude-weighted rescaling of every A-cell simultaneously**, which is
exactly the kind of move the (gp, A) block can already produce on its own via a coordinated shift
of `z`. This makes theta's gradient direction a near-redundant linear functional of the same
information the outer (gp, A) block already spans — the outer solver has to spend search budget
"fighting" this redundancy any time it wants to move theta without accidentally re-triggering a
large, non-adaptive A-rescaling. Searching over `a = log(AodPow)` instead holds the
*economically-relevant, theta-normalized* productivity level fixed under a pure theta move (at
fixed `a`, `AodPow` does not move at all) — theta's only remaining effect is through the genuine
Frechet-dispersion channel `mu*log(U[s,o])` inside the winner/price comparison (see
`compressed_moments.jl`'s use of `Uσ.^(-mu)`), which is exactly the effect that should require an
independent search direction from A.

## 4. Outer coordinate vector and the pivot (gravity-consistent) cell

`w_ext_a = [eta_theta; gp; a_nonpivot]`, `eta_theta = log(theta)` (log-space box, avoids sign
issues and keeps KNITRO's box constraint linear in the searched coordinate), length
`D*Ddest + 1` (= 381 at real D=20 post-omit-ROW: `D=20`, `Ddest=19`, `D*Ddest-1=379` free A coords
+ `eta_theta` + `gp`).

The **pivot cell** (one A cell per context, chosen as `argmax|c0|` where `c0 = q_tilde./N_obs`,
pure data) stays **z-parametrized**: its value is whatever exactly nulls the gravity restriction
(§5), reconstructed via the theta-dependent affine-offset machinery
(`gravity_elimination.jl::pivot_expand_cheap`). Only the `D*Ddest-1` NON-pivot cells are ever
represented in `a`-space; the pivot's own `a`-value is a derived quantity, never searched
directly. This is deliberate, not an oversight — it means the gravity restriction is enforced
exactly (§5) at every outer point regardless of what a-space coordinates KNITRO proposes, with no
separate KNITRO equality constraint needed (matching current production's existing pivot-elimination
architecture for fixed theta).

`decode_and_expand_flexible_A(w_ext_a, ctx, xy)` (`flexible_theta_aspace_production.jl`):
1. `theta = exp(eta_theta)`, box-checked against `ctx.θ_lo[1]/θ_hi[1]` (reject via `reject_point`,
   not abort, on violation — task §8's "graceful reject, not KNITRO callback abort" contract).
2. `mu = 1/theta`.
3. Converts the `D*Ddest-1` non-pivot `a` coordinates to `z` pointwise at the current `theta`
   (§2's exact affine map).
4. Calls `pivot_expand_cheap(z_nonpivot, pgc, mu)` (§5) to reconstruct the FULL `D×Ddest` gravity-
   feasible `z` matrix (pivot cell included), exponentiates to `Aod_theta` levels.
5. Returns `xf = [mu; gp; Aod_levels...]` — exactly the shape current production's
   `screened_eval`/`evaluate_fullA_screened_ranged` already expect from a ctx built via
   `make_flexible_theta` (`flexible_theta.jl`; mu lives at `free_idx[1]`).

## 5. Gravity restriction in a-space — rectangularized, theta-invariance re-derived

See `docs/FLEXIBLE_THETA_RECTANGULAR_GRAVITY_AUDIT_2026-07-25.md` for the full re-derivation on
the ACTIVE (post-omit-ROW) sample and its D=4/D=20 reconstruction tests. Summary: the gravity
restriction is exactly affine in `z`, `g_gravity(z) = c(mu)'z + g0(mu)`, with
`c(mu) = mu .* c0` (`c0 = q_tilde./N_obs`, pure data) — a positive scalar multiple of a
theta-INDEPENDENT vector. Consequences, all re-verified on the rectangular sample (not assumed
from the square pre-omit-ROW claim):
- **Pivot index is theta-invariant**: `argmax|c(mu)| = argmax|c0|` for any `mu>0`.
- **Pivot reconstruction slopes are theta-invariant**: `-c(mu)[j]/c(mu)[pivot] = -c0[j]/c0[pivot]`
  (the `mu` cancels exactly).
- **Only the affine intercept `g0(mu)` is theta-dependent**, and is EXACTLY affine in `mu` at a
  fixed `(gp, γ)` base state: `g0(mu) = a_fit + b_fit*mu`, fit from two `gravity_offset` probes
  (any two distinct `mu` values give the identical fit, verified to machine precision in gate 7
  below).

This is what `gravity_elimination.jl::PivotGravityElimCache`/`build_pivot_elimination_cheap` (this
port's rectangularized generalization of the experimental square-only cache) exploits: the pivot
choice, `other_idx`, and slope vector are computed **once** per outer base point (data-only,
`O(D*Ddest)`), and every subsequent theta probe (e.g. both endpoints of the theta secant, §6) reuses
them via an `O(1)` affine evaluation (`pivot_expand_cheap`) — no re-derivation of the pivot choice
or a full `gravity_from_logz` reconstruction at every theta probe.

## 6. Theta derivative — fixed-dual secant, `a_nonpivot` held fixed

`theta_fixed_dual_delta_pivot_A(w_ext_a, inner_x_fixed, ctx, xy)`: decodes `w_ext_a` (§4),
recomputes `ctx.obj.H` (i.e. `K`, `G`, hence `z(theta)`, winners, every theta-dependent moment) at
the implied theta, **without re-solving the inner KNITRO dual problem**, then evaluates the SAME
fixed dual point `inner_x_fixed` (the base point's own optimized `(zeta, lambda)`) against this new
`H` — mirroring exactly how `inner_loop_internal` itself calls `obj.moments!`, minus the solve step.

Central-difference secant: `grad_eta_theta = (D(w+h*e1) - D(w-h*e1)) / (2h)`, holding
`a_nonpivot` fixed across the probe (the entire point of a-space for this derivative — see §3: a
theta perturbation at fixed `a_nonpivot` lets `z_nonpivot` shift exactly enough to keep `AodPow`
economically fixed, isolating the genuine Frechet-dispersion channel instead of re-triggering the
dominant `mu*z` rescaling the OLD z-space secant conflated it with).

D=4 and D=20 accuracy validation: see
`docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md`.

## 7. Gp and A-block gradient — reuse via `freeze_theta_ctx`

The existing production C+ gradient kernel (`composite_gradient_at_Cplus`) is NOT theta-aware and
must not be fed a flexible ctx directly (its internal `CS.reconstruct_full(x_free, ctx.m)` calls
hard-assume the fixed-mode `[gp; Aod(D*Ddest)]` free-vector shape). `freeze_theta_ctx(ctx, mu_frozen)`
(`flexible_theta.jl`) rebuilds a `CS.FreeParamMap` shaped exactly like the ORIGINAL fixed-theta ctx
(mu fixed at `mu_frozen`, `[gp; Aod(D*Ddest)]` free) — everything else (γ, U, obj, τ, q_tilde, ...)
carried through unchanged, no re-solve, no pivot/nullspace re-derivation.

At the current outer point: `composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_frozen, ...)`
gives `[d(Delta)/d(gp); d(Delta)/d(z_nonpivot)]` — the EXACT partial derivative holding theta fixed
at `mu_frozen`, using the SAME validated kernel current fixed-mode production uses, zero new
gradient code. The A-block entries are then rescaled by the exact scalar chain-rule factor from §2:

```
d(Delta)/d(a_nonpivot) = d(Delta)/d(z_nonpivot) * dz/da = gfull_reduced[2:end] .* (-theta)
```

`d(Delta)/d(gp)` (`gfull_reduced[1]`) is untouched — `a` does not touch `gp`.

## 8. Calibration-point equivalence assertion (task §3's closing requirement)

`test_flexible_theta_aspace_d4.jl` gates 2/3/7b construct the SAME economic point
(`theta_star = 1/ctx.μHat`, `gp0`, `logA_full0` — all read from `ctx_fixed.θ0_up`'s own genuine
calibration, via `CS.pack_free`, NOT `zfree=0`) two ways — once via the z-space reduce
(`reduce_to_w_ext`) and once via the a-space reduce (`reduce_to_w_ext_A`) — decode both, and assert:
`d_z.theta ≈ d_a.theta` (rtol 1e-13), `d_z.gp == d_a.gp` (bit-identical, no transform), and
`d_z.xf ≈ d_a.xf` (rtol 1e-8, the reconstructed economic `A_od`/moments/gravity residual/Delta all
agree). Both the square and rectangular (D=4, D_dest=3) samples pass this (rel differences at or
near machine precision — see §10 results).

## 9. Fixed-theta embedding

`make_flexible_theta(ctx; theta_lo=theta_star, theta_hi=theta_star)` (a degenerate box) reproduces
the exact fixed-mode calibration point bit-for-bit in `eta_theta` (a single feasible value,
`log(theta_star)`), and `decode_theta_full`/`decode_and_expand_flexible_A` at that point reduce to
the identity map on `mu`. Production's OWN fixed-mode driver (`run_profile_checkpointed`/
`run_polish_checkpointed`) is entirely untouched by this port (see §12 fixed-mode regression) — the
embedding above is a mathematical/testing property of the flexible machinery, not a live code path
fixed-mode production runs through.

## 10. D=4 gate results (this port, current production interfaces)

`test_flexible_theta_aspace_d4.jl`, both the square (`D=4, D_dest=4`, all_legacy-shaped via
`context_scaled.jl`'s `row_idx=nothing`) and rectangular (`D=4, D_dest=3`, last-position omission,
`row_idx=4`) samples:

- Gate 1 (a↔z round-trip): exact to `< 1e-10` at 4 theta values, both samples.
- Gate 2 (z-space vs a-space `xf` equivalence at theta-star): `max|xf_z - xf_a|` at machine
  precision, both samples.
- Gate 3 (cold-verify, gravity satisfied, Delta_dual agreement): `gravity_value` `< 1e-8` in
  magnitude, `Delta_dual` agreement to rtol 1e-8, both samples. Rectangular sample measured:
  `Delta_dual_z=0.0009662387552909866`, `Delta_dual_a=0.0009662387552909862`,
  `gravity_a=-1.43e-18`.
- Gate 4b (decisive, noise-free chain-rule check — direct a-space FD vs direct z-space FD along
  the scaled direction, no `composite_gradient` dependency): rectangular sample measured
  `rel_err = 2.22e-12` — effectively machine precision, the single strongest correctness signal in
  this battery.
- Gate 5 (theta secant vs fully-resolved finite difference): monotone error shrinkage confirmed as
  `h` decreases (both samples).
- Gate 7 (full battery re-run on the genuinely rectangular D=4/D_dest=3 sample, ADDED by this
  port — not present in the pre-omit-ROW experimental source): all sub-gates pass, including a
  checkpoint-field round-trip check (`reduce_to_w_ext_A` inverts `decode_and_expand_flexible_A`
  to machine precision).

Full raw log excerpt (real, verified PASS/FAIL lines extracted from the live run):
`docs/key_results/flexible_theta_aspace_d4_gate_log_2026-07-25.txt` (committed to this repo).
