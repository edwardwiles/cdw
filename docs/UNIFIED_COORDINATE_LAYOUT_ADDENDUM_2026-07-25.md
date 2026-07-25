# Unified outer-coordinate-layout addendum — 2026-07-25

Documents the scope addendum that arrived mid-session, after the original flexible-theta a-space
port (docs `FLEXIBLE_THETA_ASPACE_*_2026-07-25.md`) was already gated and its §15 matched
comparison complete. The addendum does two things the original brief explicitly did NOT ask for:

1. Promotes the theta-decoupled `a`-space coordinate (`AodPow`, `a := log(AodPow)`) to a
   **fixed-theta production candidate** too, not just a flexible-theta one — superseding the
   original port's own §12 statement ("Do not switch fixed production to a-space in this task").
2. Requires a **single shared outer-coordinate-layout architecture** across every
   (trade_elasticity_mode, A_coordinate_mode, gp_coordinate_mode) combination, rather than
   separate near-duplicate fixed/flexible drivers — superseding the original port's own §4
   decision to keep a fully separate flexible-only driver file.

## Architecture

`full_aod_diag/d4_exact/outer_coordinate_layout.jl` — `OuterCoordinateLayout(trade_elasticity_mode,
A_coordinate_mode, gp_coordinate_mode)`, three independent axes:

- `trade_elasticity_mode`: `:fixed` | `:flexible`
- `A_coordinate_mode`: `:legacy_z` (current production's `z=log(Aod_theta)`) | `:powered_aspace`
  (`a=log(AodPow)`, theta-decoupled)
- `gp_coordinate_mode`: `:raw` | `:scaled_log` (`u_g = s_g*log(gp/gp_star)`)

Constraint (enforced in `make_layout`): `:flexible` REQUIRES `:powered_aspace` — the
theta-decoupling a-space provides has no z-space analogue (old z-space flexible theta stays D=4
comparison scaffolding only, never a production combination, exactly as in the original port).

`decode_outer_unified`/`reduce_to_w_unified`/`gradient_transform_unified`/`layout_fingerprint`
are the four shared primitives every mode combination goes through — reusing, VERBATIM, the
already-validated theta-invariant pivot cache (`gravity_elimination.jl::PivotGravityElimCache`)
and a↔z conversion (`flexible_theta_aspace_production.jl::z_from_a`/`a_from_z`/
`precompute_aspace_XY`) the ORIGINAL port already built and gated — no new core math, only a new
dispatch layer around it.

`full_aod_diag/d4_exact/c10_d20_production_driver_unified.jl` —
`run_polish_checkpointed_unified`, ONE driver function for every combination, replacing the
original port's separate `run_polish_checkpointed_flexible_theta_A`. Reuses the existing
production `screened_eval`/`DualBank`/`SafeExactCache`/`composite_gradient_at_Cplus` exactly as
both prior drivers did — only decode/gradient-rescale/checkpoint-field construction differ by
layout, everything else (screens, cache, incumbent ledger, cold-retry policy) is identical code
across every mode.

`D20CheckpointUnified` — one checkpoint schema for every combination (addendum §4): always stores
the canonical z-space representation (`zfree`/`logA_full`, universal/theta-consistent) PLUS the
layout identity (`trade_elasticity_mode`/`A_coordinate_mode`/`gp_coordinate_mode`/`amap_version`)
and the outer-searched-native coordinate (`A_native`/`gp_native`) so a resume reconstructs the
identical search point without re-derivation, and a schema/layout mismatch on resume is rejected
loudly (see `load_checkpoint_unified`).

## Gate results

### D=4 (fixed-theta unified layout)

First run: 11/15 pass, 4 error — all 4 errors traced to ONE root cause, a missing
`pairwise`/`witness` field on the D=4 test ctx (`context_scaled.jl`'s `d_exact_setup_scaled`
doesn't populate them, and `screened_eval` unconditionally passes them through to
`evaluate_fullA_screened_ranged`) — the EXACT same gap already discovered and fixed earlier this
session in `test_flexible_theta_aspace_d4.jl`, simply omitted by mistake when writing the new
unified-layout test file. Not a production bug, not a flaw in the new math: the 4 tests that
exercise the genuinely NEW code with no ctx dependency (`z<->a` round trip, `gradient_transform_
unified` vs manual `-theta` rescale, `gp_coordinate_mode=:scaled_log` encode/decode, `reduce_to_w_
unified`/`decode_outer_unified` round trip) all passed cleanly on the FIRST run.

Fixed and rerun: **[RESULT PENDING — filled once the rerun completes]**

### D=20 (fixed-theta unified layout, real post-omit-ROW data)

**[PENDING]**

## Matched comparisons (addendum §6/§7)

**[PENDING]**

## Verdict

**[PENDING — integrated with the original port's own verdict in the final production-port doc.]**
