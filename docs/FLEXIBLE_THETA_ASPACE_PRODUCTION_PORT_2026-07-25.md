# Flexible-theta a-space production port — 2026-07-25

Primary deliverable for the task "productionize the flexible-theta a-space reparametrization onto
`production/fullA-exact`." Covers task §1-§17. This document is the top-level narrative; detailed
math/audit/validation content lives in the companion docs listed in §17 below.

**STATUS PLACEHOLDER — this document is updated in-place as later sections' live runs complete.
See the final verdict line at the bottom for the authoritative outcome.**

## §1. Starting point

Forked from `production/fullA-exact@c55e81e` (bare repo `trade_robustness_modular`), branch
`port/flexible-theta-aspace-production-2026-07-25`, worktree
`/bbkinghome/edav/gravity_robustness/worktrees/port-flexible-theta-aspace-production-2026-07-25`
(pre-created; not created by this session). Confirmed present at the fork point (via direct file
inspection, not assumed from a stale handoff): omit-ROW destination default
(`destination_sample=:exclude_row`, `context_real_d20.jl`/`c10_d20_production_driver.jl`);
`:all_legacy` replication mode (same files); canonical top-1 winner engine
(`infeasibility_screen.jl::screen_hard_winners`, `fast_range_screen.jl::screen_hard_winners_ranged`);
C+ (`:cplus`) top-3 outer-gradient backend as the resolved default
(`resolve_price_cache_backend`); current unrestricted compressed core
(`compressed_moments.jl`/`compressed_cc_inner.jl`); threshold-10 typed certificates
(`cc_algo/threshold_early_abort.jl`); current screen stack (pairwise/hard-winner/envelope/
winning-range/safety-net, `fast_range_screen.jl`); current incumbent ledger
(`incumbent_logic.jl`); current checkpoint schema (`D20CheckpointV4`, `CHECKPOINT_SCHEMA=4`);
current supervisor/process-group scripts (untouched by this port).

The experimental source (`experiment/fullA-theta-aspace-reparam-2026-07-25`, commits `7e38940`/
`8844433`/`c2247e4`, checked out in a separate worktree) predates the omit-ROW-destination
production release entirely — its own `flexible_theta_aspace.jl`/D=4 test/D=20 driver assume a
square `D x D` layout throughout. **No file was copied wholesale from that branch.** Every math
object and driver structure was re-derived/re-implemented against the CURRENT rectangular
`D x D_dest` production interfaces — see §3/§6 below and the companion docs for exactly what
changed and why.

## §2. Production scope delivered

- Unrestricted full-A model; theta flexible; post-omit-ROW rectangular layout (`D=20, D_dest=19`
  at real data — genuinely exercised, not merely square-D4-with-a-comment).
- Both upper (`find_smallest=true`) and lower (`find_smallest=false`) bound directions supported
  (`run_polish_checkpointed_flexible_theta_A`'s `find_smallest_in` parameter, mirroring the fixed
  driver exactly).
- Fixed theta remains the default; flexible theta is opt-in via `make_flexible_theta(ctx; ...)` —
  a fixed-mode ctx never calls this function, so fixed-mode production code paths are literally
  unreached by any new file this port adds.
- Configuration surface implemented exactly as specified:
  ```julia
  ctx.trade_elasticity_mode   # :fixed (default, every existing production ctx) | :flexible
  ctx.A_coordinate_mode       # :theta_decoupled_aspace (this port's flexible mode)
  ```
- Outer vector in flexible mode: `w_ext_a = [eta_theta; gp; a_nonpivot]` — one eta-theta
  coordinate, one gp coordinate, `D*D_dest-1` gravity-consistent free powered-A coordinates
  (379 at real D=20 post-omit-ROW; NOT hardcoded — every dimension in the new code is derived from
  `ctx.D`/`ctx.D_dest`, verified by the genuinely-rectangular D=4/D_dest=3 test case, §6/§13).
- Not enabled in combination with CM/ZC/fixed-Frechet in this task (out of scope per the brief);
  the layout-metadata design (`ctx.free_idx`/`ctx.fixed_idx`/`ctx.m`, `make_flexible_theta`'s
  additive surgery on them) does not hard-code "unrestricted" anywhere structural, so a future
  CM/ZC combination is a smaller incremental change than a rewrite — not attempted or claimed
  ready here.

## §3-§9. Math, rectangular gravity, screens, gradients, theta derivative

See the companion docs:
- `docs/FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md` (§3)
- `docs/FLEXIBLE_THETA_RECTANGULAR_GRAVITY_AUDIT_2026-07-25.md` (§6)
- `docs/FLEXIBLE_THETA_SCREENS_AUDIT_2026-07-25.md` (§7)
- `docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md` (§8/§9, D=20 numbers)

Architecture summary (§4): the fixed and flexible drivers share EVERY core piece per task §4's
requirement — data/draw context (`d20_real_setup_design`), active destination layout
(`cc_algo/active_layout.jl`), winner/moment operators (`compressed_moments.jl` etc.), screens
(`fast_range_screen.jl`/`infeasibility_screen.jl`), inner solve (`CS.inner_loop_internal`), C+
A-gradient machinery (`composite_gradient_at_Cplus`, reused via `freeze_theta_ctx` at the current
base-point mu, §7 of the math doc), incumbent ledger (`incumbent_logic.jl`), checkpointing
(`D20CheckpointV4`'s complete field set, extended not replaced, §11), and bound-direction
semantics (`find_smallest`). The flexible-only additions are exactly the three pieces task §4
calls out: an extra eta-theta coordinate, theta bounds, and theta-aware gravity
reconstruction/derivative/cache-fingerprint (§8/§9/§10) — implemented as new files
(`flexible_theta.jl`, `flexible_theta_aspace_production.jl`,
`c10_d20_production_driver_flexible_theta_A.jl`) rather than branching inside the existing driver,
following this codebase's own established "near-duplicate rather than deep-parameterize"
convention (already used throughout the CM/origin-ZC family) — judged lower-risk for a first
production port than threading a new coordinate axis through the heavily-optimized existing
`run_polish_checkpointed` in place.

Layout metadata (task §4's "explicit coordinate-layout metadata, not positional assumptions"):
`ctx.free_idx`/`ctx.fixed_idx`/`ctx.m::CS.FreeParamMap` are the SAME metadata object current fixed
production already uses to avoid positional assumptions — `make_flexible_theta`/`freeze_theta_ctx`
manipulate this object directly rather than introducing a parallel metadata scheme; every function
that consumes the outer vector (`decode_and_expand_flexible_A`, `screened_eval_flexible_A`,
`composite_gradient_at_Cplus` via the frozen ctx) obtains its shape from `ctx.D`/`ctx.D_dest`, never
a hardcoded `399`/`400`/`D^2`.

Raw draws (task §5): `ctx.U` is built once by `d20_real_setup_design` and never touched by any
theta-changing code — `decode_and_expand_flexible_A` reconstructs `Aod_theta` (hence, downstream,
`z_o(omega)`-style transformed draws inside `compressed_moments.jl`) fresh from the CURRENT theta
at every call, but the underlying `U` array itself is passed through `merge(ctx, ...)` unchanged
through the entire flexible-mode ctx lifecycle. Theta-update cost is profiled and reported
separately in the D=20 derivative validation doc (theta-secant wall time vs total gradient
callback wall time).

## §10-§11. Cache, checkpoint

See `docs/FLEXIBLE_THETA_CACHE_CHECKPOINT_AUDIT_2026-07-25.md`.

## §12. Fixed-mode regression

**Only ONE existing production file was modified**: `full_aod_diag/d4_exact/gravity_elimination.jl`
— every change is an ADDITIVE optional keyword (`μ::Union{Nothing,Float64} = nothing`/
`ctx.fixed_vals[1]`-default) on existing functions, plus new functions/structs appended at the end
of the file; no existing function body's default-argument behavior changed. Every other file this
port touches is BRAND NEW (never included by any fixed-mode production entry point — `c10_d20_
production_driver.jl`/`unrestricted_stage_runner.jl`/CM family files do not `include()` any of
`flexible_theta.jl`/`flexible_theta_aspace_production.jl`/
`c10_d20_production_driver_flexible_theta_A.jl`).

Regression evidence:
- `test_gravity_elimination.jl` (existing, unmodified production test, direct coverage of the one
  file this port edits): **ALL GRAVITY ELIMINATION TESTS PASSED** (affine structure, pivot
  coefficient, nullspace rank/orthonormality, gravity-feasibility at random points, pivot/unpivot
  round-trip, transformed gradient of gravity == 0) — re-run unmodified after this port's edit,
  byte-identical pass record to the pre-port baseline.
- `test_exclude_row_gateA_layout.jl` (existing, unmodified — CM-family omit-ROW layout/dimension
  checks): **28 PASS / 0 FAIL** (every check line reports PASS, no FAIL line present).
- `test_composite_gradient.jl` (existing, unmodified — C+/buffered gradient backend family,
  unaffected by this port's changes but re-run to confirm): **PASS** — `gamma component matches
  Delta_FD to <0.1% relative at all points: true`, `A-block cosine > 0.9 at all points: true`
  (both headline and stalled-candidate reference points). This run is also the direct evidence
  behind the Gate 4 methodology correction in the D=20 derivative validation doc: it independently
  reproduces, on this SAME unmodified production kernel, the same random-direction sign-disagreement
  pattern (3/6, 4/6) that this port's own Gate 4 first-attempt flagged as alarming in isolation —
  confirming that pattern is a pre-existing characteristic of the kernel, not something this port
  introduced.

All three regression logs: `docs/key_results/flexible_theta_aspace_fixed_mode_regression_log_2026-07-25.txt`
(committed to this repo, real extracted PASS/FAIL lines from the live re-runs).

**Not run in this session** (acknowledged limitation): the FULL breadth task §12 lists (flexible
CM fixed theta, CM+ZC fixed theta, origin-ZC fixed theta, all four destination-sample/replication
combinations, checkpoint/resume, both directions) would require running each family's own
multi-hundred-line gate suite — out of this session's time budget beyond the two additional
targeted regression runs above, given that the actual code-change surface (one additively-edited
file) makes broad CM/ZC-family regression risk structurally near-zero (those families never
include or call anything this port added or touched). If the team wants the full CM/ZC/origin-ZC
regression matrix run explicitly before merge, that is a bounded, mechanical follow-up (run each
family's existing `test_*` file unmodified), not a design gap.

## §13-§14. D=4 and D=20 correctness gates

D=4 (`test_flexible_theta_aspace_d4.jl`, run against BOTH a square D=4 and a genuinely rectangular
D=4/D_dest=3 last-position-omission sample): **32/32 PASS.** Decisive chain-rule check
(rectangular sample): `rel_err = 2.22e-12`. Full log excerpt:
`docs/key_results/flexible_theta_aspace_d4_gate_log_2026-07-25.txt` (committed to this repo).

D=20 real post-omit-ROW (`test_flexible_theta_aspace_d20_gates.jl`, D=20/D_dest=19/W=80,000/seed
20260719/20 threads/BLAS threads=1): see
`docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md` for the full gate-by-gate record.

## §15. Matched practical-value comparison

See `docs/FLEXIBLE_THETA_POST_OMIT_ROW_MATCHED_COMPARISON_2026-07-25.md`.

## §16. Production merge rule and verdict

[FILLED IN AFTER ALL GATES COMPLETE — see the closing section of this document.]

## §17. Deliverables manifest

- `docs/FLEXIBLE_THETA_ASPACE_PRODUCTION_PORT_2026-07-25.md` (this file)
- `docs/FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md`
- `docs/FLEXIBLE_THETA_RECTANGULAR_GRAVITY_AUDIT_2026-07-25.md`
- `docs/FLEXIBLE_THETA_SCREENS_AUDIT_2026-07-25.md`
- `docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md`
- `docs/FLEXIBLE_THETA_CACHE_CHECKPOINT_AUDIT_2026-07-25.md`
- `docs/FLEXIBLE_THETA_POST_OMIT_ROW_MATCHED_COMPARISON_2026-07-25.md`
- Source: `full_aod_diag/d4_exact/{gravity_elimination.jl (edited), flexible_theta.jl,
  flexible_theta_aspace_production.jl, c10_d20_production_driver_flexible_theta_A.jl,
  matched_comparison_fixed_vs_flexible_A.jl}`
- Tests: `full_aod_diag/d4_exact/{test_flexible_theta_aspace_d4.jl,
  test_flexible_theta_aspace_d20_gates.jl, test_flexible_theta_aspace_cache_checkpoint.jl}`
- Raw logs / cold-verification records / timing CSVs: pushed to Dropbox under
  `key_results/` (this repo's own docs are not the place for multi-MB raw KNITRO logs).
- Git provenance: `provenance.txt` in the Dropbox package (branch/HEAD/log/`git diff --stat`).
- SHA256 manifest: `MANIFEST.sha256` in the Dropbox package.

---

## FINAL VERDICT

[PLACEHOLDER — updated once §14/§15/§16 complete. Do not treat any verdict text elsewhere in this
document as final until this section is filled in.]
