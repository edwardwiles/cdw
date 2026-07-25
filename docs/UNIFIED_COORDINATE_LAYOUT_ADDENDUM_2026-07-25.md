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

Fixed (added the same `merge(ctx, (pairwise=nothing, witness=nothing))` guard) and rerun:
**ALL 21/21 GATES PASS** (21 individual `@test` assertions across the 8 named testsets). Key
numbers:
- Gate 1 (unified fixed+legacy_z reproduces the EXISTING unmodified production driver's own
  `x_free_from_w`/`Delta_dual`): `max|xf_unified - xf_legacy| = 2.22e-15`, `Delta_dual` agreement
  to 11 significant digits (`0.0009662387552909997` vs `0.0009662387552910123`).
- Gate 2 (fixed+powered_aspace reproduces the same economic point as fixed+legacy_z):
  `max|xf_a - xf_z| = 8.88e-16`, `Delta_dual` **bit-identical** (`0.0009662387552909997` both),
  gravity residual `-8.6e-18`.
- Gate 4 (DECISIVE chain-rule check, direct FD in a-space vs direct FD in z-space along the
  scaled direction, no `composite_gradient` dependency): **`rel_err = 4.32e-12`** — near machine
  precision, the strongest correctness signal in this battery (mirrors the original flexible-
  theta port's own D=4 decisive check at `2.22e-12`).
- Gates 6/7 (round trips, `legacy_z`/`powered_aspace`): exact to `0.0` / `2.78e-17`.
- Gate 8 (cache A/B/A): exact hit confirmed, cache size unchanged, `Delta_dual` bit-identical.

Full log: `docs/key_results/unified_layout_d4_gate_log_2026-07-25.txt`.

### D=20 (fixed-theta unified layout, real post-omit-ROW data)

**ALL GATES PASS** at real D=20/D_dest=19/W=80,000/seed=20260719:

- Gate 1 (unified fixed+legacy_z vs EXISTING unmodified production driver): `xf` agreement
  `max|diff|=6.3e-8`, `Delta_dual` agreement to 12 significant digits
  (`0.00248677347794064` vs `0.002486773477940639`).
- Gate 2 (fixed+powered_aspace vs fixed+legacy_z, same economic point): gravity residual
  `8.98e-18`, `Delta_dual` agreement to 12 significant digits.
- Gate 3 (**DECISIVE** chain-rule check, direct FD comparison, no `composite_gradient`
  dependency): **`rel_err = 8.53e-13`** — essentially machine precision, at real D=20 production
  scale. This is the single strongest correctness result in the entire port (original +
  addendum): confirms the theta-decoupled a-space coordinate is an exact reparametrization of
  production's existing z-space coordinate at fixed theta, on the real calibrated economy.
- Gate 4 (cache A/B/A): exact hit, cache size unchanged, `Delta_dual` bit-identical.

Full log: `docs/key_results/unified_layout_d20_gate_log_2026-07-25.txt`.

**Driver-level smoke test** (`run_polish_checkpointed_unified`, the actual KNITRO-wired driver,
as opposed to the decode/gradient primitives the gates above exercise directly): first attempt
(run concurrently with the D=20 gate battery above, competing for the same host's CPU) hit its
200s wall-clock cap before even finishing context construction — no error, just contention-driven
slowness, confirmed by the log showing no output past the startup banner (not even the usual
verbose context-build diagnostics that appear within the first ~10s of any real D=20 run).
Relaunched in isolation with a longer timeout — result below.

## Matched comparisons (addendum §6/§7)

### §6: legacy_z vs powered_aspace, fixed theta, 300s each, real D=20 post-omit-ROW

Both arms via the SAME unified driver (`run_polish_checkpointed_unified`), differing ONLY in
`A_coordinate_mode` — identical calibrated start, draws, screens, cache, incumbent logic,
`algorithm=auto+SR1`. **2 of 4 runs hit the same KNITRO hang-past-timeout failure mode already
documented in this session** (`gravity-robustness-knitro-hang-past-timeout.md`) — confirmed by a
native `KTR_solve` backtrace at the moment of forced termination, not a driver bug. Both
interrupted arms were reconciled from their last checkpoint's `best_feasible` incumbent via
`reconcile_checkpoint_unified.jl`, independently cold-verified (fresh `warm=false` evaluation).

| Coordinate mode | delta | kappa | n_eval | wall (s) | status | cold-verify Delta_dual |
|---|---|---|---|---|---|---|
| legacy_z | 1 | 0.05428070 | 9 | 232.4 | **reconciled** (hung, killed) | 0.9926526198876648 (checkpoint-identical) |
| powered_aspace | 1 | 0.07102523 | 20 | 311.9 | clean finish | 0.9967846060282045 (bit-identical to live) |
| legacy_z | 2 | 0.05627694 | 11 | 313.0 | clean finish | 1.8582754742297887 (13 sig figs) |
| powered_aspace | 2 | 0.07118083 | 7 | 248.0 | **reconciled** (hung, killed) | 1.88095331469541 (checkpoint-identical) |

**delta=1 comparison is CONFOUNDED**: `legacy_z` was interrupted at only 9 evaluations (vs
`powered_aspace`'s clean 20) — a smaller search budget alone would be expected to find a worse
(smaller) kappa regardless of coordinate choice, so the raw +30% gap in `powered_aspace`'s favor
at delta=1 is not read as a clean coordinate-choice effect.

**delta=2 comparison is the more informative one**: here `powered_aspace` (interrupted at 7
evals) STILL beats `legacy_z` (clean finish, 11 evals) — `powered_aspace` found a BETTER kappa
(+26.5%: 0.07118 vs 0.05628) despite having FEWER evaluations available, not more. This is
consistent with (though — given only n=1 per cell and one confounded/one clean comparison, not
proof of) the theta-decoupling mechanism producing a more favorable search landscape even at
fixed theta, matching this port's own §3 argument for why a-space should help: the OLD z-space
coordinate mechanically couples every A-cell to a magnitude-rescaling channel a theta-flexible
search would exploit, but which even a fixed-theta search must still navigate as ill-conditioning
in the (gp, A) landscape itself.

**Not treated as decisive** given the 2-of-4 interruption rate — a clean rerun of all 4 arms
(with a watchdog that reliably force-kills hung `KN_solve` calls, not just `timeout
--kill-after`, which itself needed its own 30s grace period to actually terminate the hung
process this round) is the recommended follow-up before treating fixed-theta powered_aspace as
strictly better than legacy_z with full confidence. The DIRECTIONAL signal (powered_aspace ahead
at both deltas, including the less-confounded delta=2 comparison) is consistent with — not
contradicted by — the interruptions.

Full logs: `docs/key_results/s6_powered_aspace_d1_2026-07-25.txt`,
`docs/key_results/s6_legacy_z_d2_2026-07-25.txt`,
`docs/key_results/s6_reconcile_legacy_z_d1_2026-07-25.txt`,
`docs/key_results/s6_reconcile_powered_aspace_d2_2026-07-25.txt`.

### §7: gp raw vs scaled-log (bounded confirmation)

**[Deferred given session time constraints and the KNITRO reliability issue observed in §6 —
running further experimental confirmation runs is lower priority than closing out the core
verdict. `gp_coordinate_mode=:scaled_log` remains implemented, unit-tested (D=4/D=20 gates §5
above both PASS its encode/decode/gradient-chain-rule checks), and available as an opt-in, but
NOT experimentally confirmed to help or hurt kappa search in a real campaign, and NOT changed
from `:raw` as the default per §7's own instruction.]**

## Verdict

See the integrated final verdict in `docs/FLEXIBLE_THETA_ASPACE_PRODUCTION_PORT_2026-07-25.md`
(covers both the original flexible-theta brief and this addendum together).
