# Powered profiled-relative A coordinates — derivation (task §4)

All formulas below are re-derivations from direct reads of the live source, not re-statements from
memory: FULL's powered-A machinery (`flexible_theta_aspace_production.jl::APivotXY/precompute_aspace_XY`,
`cm_aspace_coordinate.jl::CMAPivotXY/precompute_cm_aspace_xy/cm_z_from_a/cm_a_from_z`,
`outer_coordinate_layout.jl::decode_outer_unified/reduce_to_w_unified/gradient_transform_unified`)
and REDUCED's native retained-relative machinery (`relative_a_coordinate_2026-07-31.jl::AnchorSpec/
decode_relative_A/encode_relative_A`, `gravity_pivot_on_retained_2026-07-31.jl::
PivotGravityElimOnRetained/pivot_expand_on_retained/pivot_reduce_on_retained`).

Notation: `D` = origins, `Ddest` = destinations, `z[o,d] = log(Aod_theta[o,d])`, where
`Aod_theta` is the codebase's own internal theta-space A parameter — the raw value KNITRO's free
variables represent, stored directly in `ctx.θ0_up`'s A_od block (confirmed by direct read of
`gravity_elimination.jl::gravity_from_logz`, which recovers the true level `Aod_lvl` from `Aod_θ =
exp(z)` via `Aod_lvl = Aod_θ · cHat · X^(1/μ) · Y`, i.e. `Aod_theta` is NOT simply `A_od^θ` in
levels — it is whatever gauge-transformed object the model's own reparametrization already stores;
this doc does not need or assume a closed form for that relationship, only that `z = log(Aod_theta)`
is the coordinate both FULL's gravity pivot and REDUCED's anchor/gravity machinery already operate
in, which REDUCED's own `relative_a_coordinate_2026-07-31.jl` header states explicitly: "the SAME
coordinate `gravity_elimination.jl`'s pivot already operates in"). `i = o + (d-1)·D` is the shared
column-major linear index convention both sides use.

## 1. FULL powered-A coordinates (existing, unchanged)

FULL's gravity pivot (`gravity_elimination.jl`, reused unmodified by both FULL and REDUCED)
selects one linear index as pivot and calls the remaining `D·Ddest − 1` indices `other_idx`
("z_nonpivot", ordered by `other_idx`). `xy = precompute_aspace_XY(ctx)` (equivalently
`CMAPivotXY` for the fixed-theta CM drivers) gives ctx-only, θ/A-independent constants
`logX[o,d]`, `logY[o,d]` (from the model's own gravity constants `wHat`, `τ`, `λ = reshape(P,·)`,
each divided by its own origin-1/destination-1 reference row — see `precompute_cm_aspace_xy`).

For every retained index `i` (destination `d`), with `θ` the model's trade elasticity (fixed for
the CM-family drivers; searched jointly as `η_θ = log θ` for the flexible-θ unrestricted driver):

```
decode (a -> z):   z[i] = -θ·(a[i] + logX[i]) - logY[i]
encode (z -> a):   a[i] = -(z[i] + logY[i])/θ - logX[i]
```

Pointwise affine in `a` (and in `z`) at fixed `θ`, with constant slope `dz/da = -θ` for every
coordinate. Gradient chain rule (`gradient_transform_unified`, confirmed by direct read):
`d(Δ)/da = d(Δ)/dz · (-θ)` — a scalar rescale of the existing z-space analytic gradient, no new
gradient computation.

## 2. REDUCED native retained relative-log-A coordinates (existing, unchanged)

REDUCED's reduction has TWO stages, both affine and both already implemented:

**Stage A — anchor-relative reduction** (`relative_a_coordinate_2026-07-31.jl`): pick one anchor
origin `j_d` per destination (`AnchorSpec.anchor_origin`), fix `gauge[d] = z_calib[j_d, d]` at a
GENUINE calibration point (never `zfree=0` — see this repo's own standing CLAUDE.md warning on
that exact mistake). For every retained (non-anchor) linear index `i` at destination `d`:

```
encode (z -> r):   r[i] = z[i] - gauge[d]      (anchor cells dropped entirely)
decode (r -> z):   z[i] = r[i] + gauge[d]       (anchor cells forced to z[i]=gauge[d])
```

`n_retained = D·Ddest − Ddest` coordinates survive this stage.

**Stage B — gravity pivot among retained cells** (`gravity_pivot_on_retained_2026-07-31.jl`):
gravity is exactly affine in `r` (composition of two affine maps — gravity is affine in `z`,
`z` is affine in `r`), so the SAME pivot-elimination algorithm `gravity_elimination.jl` already
uses applies again, restricted to retained positions only:

```
r[pivot_pos]        = (-offset_r0 - Σ_k cr[other_pos[k]]·r_free[k]) / cr[pivot_pos]
r[other_pos[k]]      = r_free[k]                          (decode: pivot_expand_on_retained)
r_free[k] = r[other_pos[k]]                                (encode: pivot_reduce_on_retained)
```

`n_retained − 1 = D·Ddest − Ddest − 1` coordinates survive Stage B — this is `r_free`, REDUCED's
actual outer A-block coordinate today (`:profiled_pivot_anchor_relative`, the only mode currently
implemented). Composing both stages: `r_free -> z` is affine (linear map + constant offset), with
a computable constant Jacobian (see §5).

## 3. Proposed powered retained-relative coordinates (`:profiled_powered_relative_A`)

Apply FULL's own per-cell affine transform (§1) to the SAME retained linear indices REDUCED's
Stage B already selects as free (`i = ridx[other_pos[k]]`, destination `d(i)`), using the SAME
`logX[i]`, `logY[i]` constants (`precompute_cm_aspace_xy(ctx)` — ctx-only, so REDUCED's `ctx`
gives byte-identical `X,Y` to FULL's, since both are built from `d20_real_setup_design`'s shared
`γ` object):

```
encode (r_free -> a_free):  a_free[k] = -(r_free[k] + gauge[d(i)] + logY[i]) / θ - logX[i]
decode (a_free -> r_free):  r_free[k] = -θ·(a_free[k] + logX[i]) - gauge[d(i)] - logY[i]
```

(substituting `z[i] = r_free[k] + gauge[d(i)]` into FULL's `a[i] = -(z[i]+logY[i])/θ - logX[i]`
from §1, and its inverse).

This is a pure per-coordinate affine reparametrization of `r_free`: scale `-1/θ` (decode direction
`-θ`), constant per-coordinate shift `-(gauge[d(i)] + logY[i])/θ - logX[i]`. It commutes cleanly
with Stage B's own already-affine pivot elimination (composition of affine maps is affine), so the
full chain `w_free (KNITRO outer vector) -> a_free -> r_free -> r -> z -> Aod levels` remains
affine end to end, exactly as FULL's own `w -> a_nonpivot -> z_nonpivot -> Aod levels` chain is.

### 3a. Anchor recovery

Unchanged from native REDUCED — anchor cells are still fixed at `z[j_d,d] = gauge[d]` (§2, Stage
A decode), entirely upstream of and independent from the powered transform, which only touches
retained cells. No new anchor-recovery formula needed.

### 3b. Gravity-pivot recovery

Unchanged in STRUCTURE from native REDUCED (§2, Stage B decode: `r[pivot_pos]` solved from the
same affine gravity constraint `offset_r0 + cr'r = 0`) — only the INPUT to that step differs: under
`:profiled_powered_relative_A` the KNITRO outer vector holds `a_free`, so decode must first apply
§3's `a_free -> r_free` map, THEN run `pivot_expand_on_retained(r_free, pe)` unchanged.

### 3c. Full-A reconstruction

Unchanged — once `z` (D×Ddest) is recovered via Stage A's `decode_relative_A`, the existing
Topic-2 level-A reconstruction (`Aod_theta -> Aod_lvl`, independent of A-coordinate choice per
`relative_a_coordinate_2026-07-31.jl`'s own header) applies exactly as it already does for native
REDUCED.

### 3d. Jacobian

Let `S = diag(-θ)` (scalar, same for every coordinate — NOT per-coordinate-varying, since θ is
fixed for every REDUCED family; only the flexible-θ unrestricted FULL driver would need a
per-call-varying θ, which REDUCED does not support and this task does not add). Decode Jacobian
`∂z/∂a_free` (composing §3 encode^{-1} with Stage B's decode, both affine):

```
∂r_free/∂a_free = -θ · I_{n_retained-1}                         (§3 decode, diagonal, constant)
∂r/∂r_free       = pivot expansion matrix (§2 Stage B, EXISTING, unchanged: identity on
                    other_pos rows, cr[other_pos]/(-cr[pivot_pos]) on the pivot_pos row)
∂z/∂r            = identity on retained rows, zero on anchor rows                (§2 Stage A)
```

So `∂z/∂a_free = (∂z/∂r)·(∂r/∂r_free)·(-θ·I)` — i.e. EXACTLY REDUCED's existing native
`∂z/∂r_free` Jacobian (already computed/used by the current analytic outer gradient, unchanged),
rescaled by the same constant `-θ` FULL's own `gradient_transform_unified` uses. This is the key
practical payoff: **no new Jacobian code is needed** — `d(Δ)/da_free = d(Δ)/dr_free · (-θ)`, a
scalar rescale of the analytic gradient REDUCED already computes in native coordinates, mirroring
`gradient_transform_unified`'s `g[2:end] .*= (-θ)` line verbatim.

### 3e. Inverse Jacobian

`∂a_free/∂z` restricted to the retained/free directions is `(-1/θ)` times the (pseudo-)inverse of
the existing native Stage-A/B encode map — not needed for the gradient chain rule above (only the
forward decode Jacobian is), but relevant for encoding an arbitrary full-A point (e.g. a warm
start or a cross-formulation comparison point) into `a_free`: apply REDUCED's existing
`encode_relative_A` + `pivot_reduce_on_retained` (unchanged) to get `r_free`, then §3 encode.

### 3f. Bounds and initialization

`a_free` bounds: since `a_free = f(r_free)` is a strictly monotonic (slope `-1/θ`, θ>0) affine
map per coordinate, an interval bound on `r_free` maps to an interval bound on `a_free` by
applying `f` to both endpoints (order flips because the slope is negative) — exactly how FULL's
own `_run_full_unrestricted`/CM drivers already convert `z_halfwidth`-style bounds under
`:powered_aspace`. Initialization: `a_free0 = f(r_free0)` where `r_free0` is REDUCED's existing
native calibration encode (`encode_relative_A` + `pivot_reduce_on_retained` at `ctx.θ0_up`) — same
"encode the genuine calibration point" discipline REDUCED's own `w0` construction already follows,
not an invented start.

## 4. Bijection determination

**Yes — `:profiled_powered_relative_A` is a pure bijection on the existing REDUCED outer space**,
by construction: it is a composition of (a) REDUCED's own existing Stage A/B maps, both already
proven bijective (affine, invertible — `decode_relative_A`/`encode_relative_A` and
`pivot_expand_on_retained`/`pivot_reduce_on_retained` are stated inverses of each other in their
own docstrings), with (b) the §3 per-coordinate affine map, invertible for any `θ ≠ 0` (θ is a
trade elasticity, always strictly positive in this codebase — `ctx.fixed_vals[1]`/`ctx.θ_lo[1]`
are bounded away from 0 by construction elsewhere). Composition of bijective affine maps is a
bijective affine map. No obstruction was found; §5 (implementation) below adds this mode.

## 5. Implementation

Added `full_aod_diag/d4_exact/profiled_powered_relative_a_2026-08-04.jl` (additive only):
`encode_powered_relative_A`/`decode_powered_relative_A` (§3), `powered_relative_gradient_rescale`
(§3d, the `-θ` scalar rescale), reusing `precompute_cm_aspace_xy`/`cm_fixed_theta` from
`cm_aspace_coordinate.jl` (already REDUCED-ctx-compatible, per that file's own "ctx-only" framing)
and `AnchorSpec`/`PivotGravityElimOnRetained` from the two existing REDUCED coordinate files
unchanged. `:profiled_powered_relative_A` is registered as an allowed `coordinate_modes` value but
is NOT made the default for any REDUCED family (native `:profiled_pivot_anchor_relative` remains
default) — per task §4.2's explicit "do not make the new mode the default before testing."

Round-trip/consistency gates (encode/decode identity, same full log A, same full A, same gravity
residual, same Δ* at a fixed decoded state, same decoded-state directional derivative after chain
rule, checkpoint/manifest mode compatibility, cache invalidation on mode change) are implemented in
`test_powered_relative_a_roundtrip_2026-08-04.jl` — see the CURRENT_STATE doc's companion results
entry for pass/fail status once run.
