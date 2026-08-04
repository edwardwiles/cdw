# Section 6 — powered profiled-relative coordinates, production wiring

## What was wired (task §6.1)

Added `full_aod_diag/d4_exact/profiled_coordinate_mode_dispatch_2026-08-04.jl`: the ONE boundary
layer between KNITRO's own outer vector (in whichever A-coordinate mode a run selects) and
REDUCED's native r_free space that `evaluate_profiled_point`/`shared_family_outer_gradient`/every
existing `decode_outer_profiled` call site (~15 sites) already operate in UNCHANGED. Wired through:

- **`run_profiled_upper_constrained`** (`profiled_production_outer_constrained_2026-08-02.jl`):
  new required-by-CLI, defaulted-for-other-callers `a_coordinate_mode` kwarg. Initial point and
  box bounds are built in native units then transformed via `encode_w_native_to_mode`/`mode_bounds`;
  the KNITRO callback (`solve_at`/`cb_F!`/`cb_G!`) decodes KNITRO's raw iterate back to native
  before calling `evaluate_fn`/`shared_family_outer_gradient` (unchanged), and the returned
  native-space analytic gradient is rescaled back to mode units via
  `rescale_full_gradient_for_mode` before being handed to KNITRO.
- **Checkpoint namespace** (`ProfiledProductionConfig`/`build_profiled_production_config`,
  `profiled_ab_comparability_and_plumbing_2026-08-01.jl`): `a_coordinate_mode` folded into the
  namespace string exactly like `economic_parameterization`/`family_kind`/layout digest already
  are — a checkpoint written under one mode is refused on resume under a different mode
  automatically (`assert_checkpoint_compatible`'s existing mismatch check), no bespoke new
  validation needed. Confirmed live (task §6.2 gate).
- **`bin/run_profiled_model.jl`**: new required `--a-coordinate-mode` CLI flag (only required for
  `--formulation reduced`), validated against the two known modes, threaded through to
  `run_profiled_upper_constrained` and recorded in `run_manifest.json`'s `A_coordinate_mode` field
  (previously hardcoded to native regardless of what actually ran).
- **`run_outer_flexcm_reduced_constrained_2026-08-02.jl`/`run_outer_originzc_reduced_constrained_2026-08-02.jl`**:
  include-list updated for the new dependency files (default kwarg preserves their existing
  native-only behavior unchanged).

**Family scope**: fixed-theta only, by the derivation's own explicit boundary (the `-theta`
gradient rescale requires a CONSTANT theta; `:unrestricted`'s theta is jointly searched, not
fixed). `validate_mode_family_compatibility` hard-errors on `:profiled_powered_relative_A` +
`:unrestricted` rather than silently falling back to native — confirmed live (gate check 8).

`run_profiled_upper_constrained` is used by **flexible_cm and common_frechet** (both share the
identical `shared_family_outer_gradient`/`run_profiled_upper_constrained` code path, dispatch is
on `a_coordinate_mode`, not `family_kind` — no family-specific copy). **origin_zc/cm_meanzc**
dispatch through the SEPARATE `run_profiled_upper_constrained_free_nu` driver
(`profiled_zc_free_nu_production_driver_2026-08-04.jl`, extended `[gp;A_free;eta_nu]` outer
vector), which was **NOT** wired for powered mode this session — real, precisely-scoped remaining
work, not a silent gap (see `FamilyRegistry.jl`'s own updated notes on those two rows).

## Gates run (task §6.2)

Real D4 KNITRO context, flexible_CM family (`test_powered_coordinate_production_gate_2026-08-04.jl`):

1. encode/decode round trip: max abs err `2.2e-16` (machine precision)
2. identical reconstructed full log-A / gravity residual at the shared calibration start, native
   vs powered basis: `2.2e-16` / `5.2e-19`
3. identical fixed-state Delta-star at that point (native start vs decoded-from-powered start),
   same inner_status: diff `3.5e-18`
4. native-gradient vs powered-gradient chain rule (`-theta` scalar rescale): exact, `0.0`
5. central FIXED-DUAL finite difference in powered units, using the SAME adaptively-selected
   bandwidth the production analytic gradient uses (`profiled_select_bandwidth`) rather than a
   naive fixed step: max_rel_err `4.85e-15` (native-space diagnostic: exact `0.0`)
6. checkpoint mode-mismatch refusal: real `run_profiled_upper_constrained` write-then-resume,
   correctly refused with `"assert_checkpoint_compatible: checkpoint namespace mismatch"`
7. short constrained KNITRO run completes cleanly under powered mode through the real production
   callback path (not just the standalone dispatch functions): 211 evals / 20 grads, feasible
8. `:unrestricted` + powered mode is a hard error (`validate_mode_family_compatibility`), not a
   silent fallback

**`POWERED_COORDINATE_PRODUCTION_GATE (D4, flexible_cm): PASS`** — all 8 checks pass.

## Two real bugs found and fixed while building this gate (both in the test, not the wiring)

1. **Wrong family context.** A first attempt reused `UnrestrictedFamilyCtx` as a convenient
   stand-in to test the shared engine plumbing. `validate_mode_family_compatibility` correctly
   blocked it, and a separate check (central FD) failed with 1560% relative error — both were the
   SAME root cause: `:unrestricted`'s own theta is not `cm_fixed_theta(ctx)` at all (a materially
   different, jointly-searched value), so testing powered mode's fixed-theta-only math against the
   wrong family was internally inconsistent from the start. Fixed by rebuilding the gate against
   the real `flexible_cm` family construction (`build_cm_augmented_obj_archB`/`build_cm_bin_ctx`/
   `build_flexcm_family_ctx`), matching production's own pattern.
2. **FD bandwidth mismatch** (this repo's own standing pitfall,
   `feedback-fd-bandwidth-mismatch-looks-like-a-bug`). After fixing (1), central FD STILL showed
   ~1560% relative error, identically in both native and powered space — a diagnostic (direct
   native-space FD vs the analytic gradient, no powered coordinates involved) reproduced the exact
   same number, proving the bug was never in the coordinate transform. Root cause: a naive fixed
   `h=1e-5` FD step can straddle a winner-switch kink in `Delta_dual` that the production
   analytic gradient's own adaptive bandwidth (`profiled_select_bandwidth`) is specifically chosen
   to avoid. Fixed by using that SAME adaptively-selected native-space `h` for both the native and
   (unit-converted via `h_a = h_native/theta`) powered FD comparisons — result dropped from a
   1560% relative error to `4.85e-15`.

## Not done this session (real remaining work, not silently skipped)

- **origin_zc/cm_meanzc powered-mode wiring**: `run_profiled_upper_constrained_free_nu` was not
  touched. FamilyRegistry's own notes on those two rows state this precisely.
- **D20/W=20,000 and D20/W=100,000 production-context gates**: only D4 was run this session (task
  §6.2's own gate list requires all three scales). Real remaining work.
- **common_frechet's own dedicated gate**: shares the identical code path as flexible_cm (confirmed
  by construction — dispatch is on `a_coordinate_mode`, not `family_kind`), but was not
  independently executed against the production KNITRO callback this session.
- **Section 10 (coordinate-mode tournament)**: depends on this section; not started (see MASTER.md).

## Verdict

```
POWERED_PROFILED_MODE =
    unrestricted:   not_applicable_fixed_theta_only (hard error by design, not a gap)
    flexible_CM:    pass (D4 production-context gate; D20/W20k, D20/W100k not yet run)
    common_frechet: pass_by_shared_code_path (identical dispatch as flexible_cm; not independently
                     executed against the production callback this session)
    origin_ZC:      fail_not_wired (run_profiled_upper_constrained_free_nu untouched)
    CM_plus_ZC:     fail_not_wired (same driver as origin_ZC)
```
