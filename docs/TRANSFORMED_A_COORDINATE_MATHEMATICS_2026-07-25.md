# Transformed-A coordinate mathematics — production port, 2026-07-25

Task Part I deliverable for `port/transformed-A-and-flexible-theta-production-2026-07-25`. This
port did not re-derive the math from scratch: it rebases and hardens
`port/flexible-theta-aspace-production-2026-07-25` (tag
`flexible-theta-aspace-port-ready-2026-07-25`), whose own
`docs/FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md` (carried forward
unchanged in this branch, reproduced in full below with one added section) already recovered and
verified this math against current production's own `gravity_elimination.jl::gravity_from_logz`,
not re-derived independently. That doc is the primary source; this file adds only what the current
task brief asked for beyond it (§2's price-level/CES-score distinction) and records what changed in
Phase 1 reconciliation.

**Reminder (repo CLAUDE.md standing warning), restated here because it is the single most
recurring mistake in this codebase's history:** nothing below treats `A_od ≡ 1` (`zfree=0`) as "the
calibration point." It is the pivot reparametrization's coordinate origin, nothing more. The real
calibrated `A_od` is `ctx.θ0_up`'s own A-block, spanning ~11 orders of magnitude at real D=20
data — every equivalence claim in this doc and its gates is checked against that object, never
against `zfree=0`.

## 0. Price-level vs CES-score coordinate (task brief §2, newly answered here)

The task brief asks this port to explicitly distinguish:
- `a^price_od = log(c_od)` — the bilateral price/cost shifter's own log-level, and
- `a^CES_od = (1-σ)·log(c_od)` — the object actually raised into the CES aggregator.

Checked directly against `compressed_moments.jl` (unchanged by this port): `AodPow[o,d]` (§1
below) enters the winner/price comparison as

```
constConsσ[o,d] = wHat[o]^(1-σ) * (AodPow[o,d] * τ[o,d])^(1-σ)          # compressed_moments.jl:208
```

i.e. the `(1-σ)` power is applied to `AodPow·τ` **downstream**, at the point `constConsσ` is
formed for the CES/winner comparison — it is not baked into `AodPow` itself. `a := log(AodPow)`,
the coordinate this port actually searches (§2 below), is therefore the **price-level** coordinate
`a^price` (up to the `τ` factor, which is fixed data, not part of the search), not the CES-powered
`a^CES`. This is the economically correct choice for an outer search coordinate: `a^CES` would
absorb `σ` into the coordinate's own scale for no benefit, and would need a compensating `1/(1-σ)`
factor threaded through every chain-rule step in §5/§7 of the parameterization doc below. This port
adds no `(1-σ)` factor anywhere in `z_from_a`/`a_from_z`/`gradient_transform_unified` — confirmed
by direct inspection of `flexible_theta_aspace_production.jl` and `outer_coordinate_layout.jl`, and
implicitly re-confirmed by every D=4/D=20 exact-equivalence gate in §10 of the doc below (those
gates would not pass at machine precision if a spurious `(1-σ)` factor were present, since `σ` does
not cancel out of any of the checked quantities on its own).

## 1–10. Recovered definitions, exact a↔z map, and gate results

Reproduced verbatim from `docs/FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md`
(this branch, unmodified by Phase 1) — see that file for the full text: objects (`theta`, `mu`,
z-space `z=log(Aod_theta)`, economic `A_od`, the theta-independent `AodPow`/`a=log(AodPow)`
coordinate and its exact `X`/`Y`/`lambda` construction); the exact affine `a↔z` map and its
gradient chain rule `d(Delta)/d(a_nonpivot) = d(Delta)/d(z_nonpivot)·(-theta)`; why z-space
entangles theta and A while a-space decouples them; the outer vector layout and pivot-cell
handling; the rectangularized gravity-restriction re-derivation (theta-invariant pivot/slope,
affine-in-mu intercept only); the fixed-dual theta secant; the C+ gradient reuse via
`freeze_theta_ctx`; the calibration-point equivalence assertion (checked against `ctx.θ0_up`, not
`zfree=0`); and the D=4 gate battery (21/27 gates cited there, all PASS, decisive chain-rule
`rel_err` at machine precision).

## 11. What Phase 1 reconciliation changed (and did not change)

Rebasing this branch onto the current canonical `cdw/production/fullA-exact` tip (`39b89c5`,
32 commits ahead of this branch's fork point) applied with **zero git conflicts** — the 32
commits never touch `gravity_elimination.jl`, `outer_coordinate_layout.jl`,
`flexible_theta_aspace_production.jl`, or `flexible_theta.jl` (confirmed identical byte-for-byte
pre/post-rebase). **None of the math in this doc or its source changed.** What Phase 1 did change,
purely in the driver-integration layer:
- `run_polish_checkpointed_unified` now attaches the same three campaign-lifetime workspaces
  (`CompressedFactualWorkspace`, canonical-price-precompute, hard-score-B cache) and the same
  `blas_threads`/`pin_outer_algorithm` opt-in kwargs that landed in
  `run_polish_checkpointed`/`run_profile_checkpointed` in those 32 commits — previously the
  unified driver silently lacked all of this hardening (a fork-staleness risk, not a math bug).
- `production_backend_manifest.jl`'s `resolve_unrestricted_manifest` gained
  `trade_elasticity_mode`/`A_coordinate_mode`/`A_coordinate_mapping_version`/`gp_coordinate_mode`/
  `theta_coordinate`/`theta_bounds`/`theta_derivative_backend`/`theta_aware_dual_bank`/
  `outer_dimension` fields, all defaulted so the two pre-existing call sites are byte-identical.
- `DualBank`'s warm-start nearest-neighbor selection was **theta-blind** on the source branch (task
  §12's explicit, previously-unmet "fix this" requirement) — fixed via
  `outer_coordinate_layout.jl::dual_bank_zfree`, which prepends `eta_theta=log(theta)` to the key
  vector in flexible mode only. See that function's docstring for the normalization rationale
  (task §12's "document the normalization" requirement). Fixed mode is unaffected (identity).

Smoke-validated end-to-end post-reconciliation (60s truncated runs, all three arms, real D=20/
W=80,000/`:exclude_row`): `fixed_legacyz` cold-verified exactly; `fixed_aspace` reproduced
`kappa=0.0649042169269014` at n_eval=7, bit-identical to the source branch's own
`FLEXIBLE_THETA_THREE_ARM_FOLLOWUP_2026-07-25.md` delta=1 result — confirming the reconciliation
is a pure hardening/plumbing change with zero effect on the underlying numerics, exactly as
intended (`blas_threads=nothing`/`pin_outer_algorithm=false` in the smoke test, i.e. the new kwargs
were exercised in their zero-behavior-change default state). See
`docs/OUTER_COORDINATE_PUBLIC_DRIVER_ASSERTIONS_2026-07-25.md` for the full post-rebase gate
report including the `flexible_aspace` arm.
