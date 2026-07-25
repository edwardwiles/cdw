# Common outer-coordinate architecture — production port, 2026-07-25

Task Part II deliverable. This port inherits its architecture from
`port/flexible-theta-aspace-production-2026-07-25`'s own addendum
(`docs/UNIFIED_COORDINATE_LAYOUT_ADDENDUM_2026-07-25.md`, carried forward unmodified — see that
file for the full original writeup, gate results, and the fixed-mode `legacy_z` vs
`powered_aspace` §6 matched comparison). It already satisfies the task brief's Part II
requirements: one `OuterCoordinateLayout` abstraction (`trade_elasticity_mode ∈ {:fixed,
:flexible}` × `A_coordinate_mode ∈ {:legacy_z, :powered_aspace}` × `gp_coordinate_mode ∈ {:raw,
:scaled_log}`, the last left at `:raw`-default-only per this task's own scope instruction), one
shared driver (`run_polish_checkpointed_unified`), one checkpoint schema
(`D20CheckpointUnified`), and a single confirmed-380/381 outer dimension at real D=20. This file
records what Phase 1 reconciliation of *this* port added on top, and the smoke-test evidence that
the shared architecture survived rebasing onto the current canonical production tip intact.

## What changed in Phase 1 (this port only)

The rebase itself (`c55e81e..cdw/production/fullA-exact@39b89c5`, 32 commits, 14 cherry-picked
commits onto the new base) applied with zero conflicts — none of the 32 intervening commits touch
any file this architecture depends on. What Phase 1 added, all inside
`run_polish_checkpointed_unified`/`production_backend_manifest.jl`/`outer_coordinate_layout.jl`
(no inner Hessian/moment/winner/CM/ZC code touched, per the brief's explicit non-overlap
instruction with the concurrent winner-pair Hessian work):

1. **Workspace/BLAS-thread parity with the current fixed-mode driver.** The unified driver was
   forked from `run_polish_checkpointed` *before* the 32 commits' allocation/Hessian hardening
   landed (`CompressedFactualWorkspace`/canonical-price-precompute/hard-score-B campaign-lifetime
   workspaces, `blas_threads`/`pin_outer_algorithm` opt-in kwargs) — a silent staleness risk, not
   a conflict a `git rebase` would have surfaced. Phase 1 added the identical attach calls and
   kwargs to `run_polish_checkpointed_unified`, in the identical place in the ctx-build sequence
   (attached to `ctx_base` before `build_unified_ctx` wraps it for flexible mode, so
   `make_flexible_theta`'s `merge(ctx, (...))` inherits them for free).
2. **Backend-manifest coverage.** `production_backend_manifest.jl::resolve_unrestricted_manifest`
   gained the task's requested fields (`trade_elasticity_mode`, `A_coordinate_mode`,
   `A_coordinate_mapping_version`, `gp_coordinate_mode`, `theta_coordinate`, `theta_bounds`,
   `theta_derivative_backend`, `theta_aware_dual_bank`, `outer_dimension`), all defaulted to the
   pre-port fixed/legacy-z values so `run_profile_checkpointed`/`run_polish_checkpointed`'s
   existing manifest calls are byte-identical. `run_polish_checkpointed_unified` is now the 7th
   manifest-producing public call site (previously outside the 32-commit "all 6 public entry
   points" inventory entirely).
3. **Theta-aware `DualBank`** (task §12's explicit, previously-unmet "fix this"): see
   `outer_coordinate_layout.jl::dual_bank_zfree` and
   `docs/TRANSFORMED_A_COORDINATE_MATHEMATICS_2026-07-25.md` §11.

## Outer vector shape (unchanged, restated for §21's assertion requirement)

```
w = [eta_theta?; gp_coord; A_coord(D*Ddest-1)]
```

`eta_theta` present iff `trade_elasticity_mode==:flexible`. At real D=20 post-omit-ROW
(`D=20, D_dest=19`): **380** free coordinates fixed-mode, **381** flexible-mode — both re-confirmed
live via the `[backend-manifest] outer_dimension=...` startup print in every Phase 1 smoke run
(§ below), not just the D=4/D=20 unit gates.

## Post-reconciliation smoke evidence (D=20/W=80,000/`:exclude_row`, 60s truncated budgets)

All three production-relevant arms run through the SAME reconciled `run_polish_checkpointed_unified`,
differing only in `layout`:

| Arm | outer_dimension | kappa (truncated, not a matched-budget result) | cold-verify | notes |
|---|---|---|---|---|
| `fixed_legacyz` | 380 | 0.05294 (n_eval=8) | Delta_dual bit-identical live vs cold | baseline, unchanged path |
| `fixed_aspace` | 380 | 0.06490 (n_eval=7) | agrees to 1e-9 | **bit-identical to the source branch's own 300s-budget delta=1 result (0.0649042169269014)** — confirms the reconciliation is pure plumbing, zero effect on numerics |
| `flexible_aspace` | 381 | 0.04905 (n_eval=4) | agrees to 1e-9 | theta varied 8.76→8.46 across evals; theta-aware `DualBank`, manifest `theta_bounds`/`theta_derivative_backend` fields all populated correctly; no crash in `cb_G!`'s theta-secant path |

(`fixed_aspace`'s exact match to the pre-rebase 300s result, despite this being a 60s truncated
run, holds because both runs share the same seed/draws/screens/cache and KNITRO's search
trajectory through `n_eval=7` is deterministic given identical inputs — this is a strong
zero-regression signal, not a coincidence.)

Full logs (not yet trimmed into `docs/key_results/`, raw scratch output — trimmed copies to be
added to the deliverable package): `smoke_fixed_legacyz.log`, `smoke_fixed_aspace.log`,
`smoke_flexible_aspace.log` (session scratchpad; see provenance in the final deliverable package).

## Deferred/out-of-scope for this port (unchanged from the source branch, restated)

- `gp_coordinate_mode=:scaled_log` remains implemented and unit-gated but not experimentally
  confirmed or defaulted — explicitly out of this task's scope (brief: "leave scaled-log g_p
  outside the default scope").
- CM/ZC/fixed-Frechet restriction-family combinations: not implemented for either release in this
  port. Release A (fixed transformed-A) is architecturally family-agnostic (it only changes the
  outer coordinate, not the restriction basis) but has not been *tested* against those families
  yet — see Phase 2 for what was actually run. Release B stays unrestricted-only per the brief's
  explicit initial-scope allowance.
- The standalone (pre-addendum) `c10_d20_production_driver_flexible_theta_A.jl` and its
  `D20CheckpointFlexA` schema remain in the tree, superseded but not deleted (still referenced by
  older test/reconcile scripts from before the addendum). Not wired into anything new; retained
  only as inert scaffolding. Deleting it was judged out of scope for this port (no production code
  path reaches it) rather than a required cleanup.
