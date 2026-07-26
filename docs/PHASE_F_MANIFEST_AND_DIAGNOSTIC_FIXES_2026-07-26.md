# Phase F — Manifest and Diagnostic Fixes (items 1–4) — 2026-07-26

**State: MATCHED_AB_PASSED for items 1, 2, 4; comment-only (no test needed) for item 3.** Not
merged to `production/fullA-exact`; on `port/remediate-production-5x7-audit-2026-07-26`. Item 5
(transformed-A default promotion for the four restricted families) is **deliberately deferred**,
see "Not done" below.

## Item 1 — common-Fréchet manifest resolver (commit `3c8942d`)

Common-Fréchet was the only restricted family with no structured, JSON-able manifest resolver —
only the print-only `print_frechet_startup_manifest` (`cm_frechet_level.jl:322`), which cannot be
serialized or compared programmatically the way `resolve_flexible_cm_manifest`/
`resolve_origin_zc_manifest` already can be.

Added `resolve_common_frechet_manifest(; cctx, blas_threads)` to `production_backend_manifest.jl`,
mirroring `resolve_flexible_cm_manifest`'s fields/style (common-Fréchet shares the same
`CMBinHessCtx`/`build_cm_bin_ctx` machinery for its economic core — confirmed in the baseline
audit's Area 5/6 findings), adding Fréchet-specific fields: `frechet_feature_set`,
`frechet_basis`, `frechet_grid_size`, `frechet_level_count`, `level_hessian_backend` (reports
whether the threaded level-block Hessian variant is active).

**Gate** (`test_phaseF_frechet_manifest.jl`, D=4, real cctx): builds a real common-Fréchet
production context, resolves the manifest, checks `family`, `frechet_grid_size`,
`frechet_level_count`, `checkpoint_schema`, `level_hessian_backend`, `threaded_bins`, and
confirms JSON serialization. **ALL PASS** (7/7).

## Item 2 — CM+ZC congruence-label bug (commit `3c8942d`)

`restriction_hessian_backend` was keyed on `is_meanzc` alone
(`is_meanzc ? :cm_bin_prefix_plus_congruence : :cm_bin_prefix`), but R-congruence is actually
gated on `cctx.R !== nothing` (`contrasts == :orthonormal`), not on whether the meanzc extension
is active. `run_cm_upper_checkpointed`'s own default is `contrasts=:anchored` for both the plain
and meanzc branches — so this label previously claimed congruence was active for CM+ZC under
*ordinary default settings* when it was not (baseline static audit finding B5). Reporting-only fix
— the Hessian computation itself was already correct either way, no numerical change.

**Gate** (`test_phaseF_manifest_label_fix.jl`, D=4, real cctx under both `:anchored` and
`:orthonormal` contrasts, plus CM+ZC under its own production default): confirms the label now
tracks the real `cctx.R` state — `:cm_bin_prefix` for flexible-CM under `:anchored`,
`:cm_bin_prefix_plus_congruence` under `:orthonormal`, and (the actual bug) `:cm_bin_prefix` — not
`_plus_congruence` — for CM+ZC under its own default `:anchored` config. **ALL PASS** (6/6).

## Item 3 — stale origin-ZC docstring (commit `1fd8674`)

`run_originzc_upper_checkpointed`'s docstring claimed this family "always uses Architecture A" for
its Hessian — stale relative to the `shared-winner-pair-core-hessian-production-2026-07-25` port:
H_EE dispatches to the same shared `exact_winner_pair_parallel` backend every other family uses by
default; only H_ER/H_RR remain dense (small, fixed-size restriction dimension, no CM grid to
structure), not H_EE. Comment-only change, zero behavior/numerical impact — no test needed.

## Item 4 — dead `price_cache_backend` kwarg was a latent crash bug (commit `c30e252`)

`c10_d20_production_driver_unified.jl`'s `price_cache_backend` kwarg accepted any `Symbol` but
only ever had a live effect on whether `lfix_c_ws` gets built. `cb_G!` unconditionally calls
`composite_gradient_at_Cplus(..., lfix_c_ws, ...)`, which requires
`ws::LFixFactorizedWorkspace` — a **concrete, non-nullable** type. Passing anything other than
`:cplus`/`nothing` set `lfix_c_ws = nothing` and would have crashed with a confusing `MethodError`
deep inside `cb_G!` mid-solve, not a clear error at call time. This is worse than a merely "dead"
kwarg — it's a latent crash bug. Unlike the older `c10_d20_production_driver.jl`, this unified
driver never ported the `:pooled`/`:aplus`/`:kbplus` alternative gradient backends; `:cplus` is
the only gradient kernel this driver's own header describes ("Reuses ... the C+ gradient kernel
... exactly as both prior drivers already did").

**Fix**: fail fast right after `resolved_backend` is computed — any value other than `:cplus`
raises an immediate, actionable error instead of silently propagating to a downstream crash.

**Gate** (`test_phaseF_price_cache_backend_validation.jl`, real D=20/W=80,000): calls
`run_polish_checkpointed_unified` with `price_cache_backend=:bogus`, confirms it raises the new
clear message ("price_cache_backend=:bogus is not supported ... pass :cplus or nothing") and
explicitly confirms it is **not** a `MethodError` (i.e. never reached `cb_G!`). **ALL PASS** (3/3).
A `global`-scoping bug in the test's own `try`/`catch` block (the same soft-scope trap class as
Phase A's own fix, this time in new test code) was found and fixed along the way.

## Item 5 — NOT DONE, deferred

Promoting `A_coordinate_mode=:powered_aspace` ("transformed_a") to the default for the four
restricted families (currently `:legacy_z`) was explicitly sized but **not implemented** this
session. Phase A's own equivalence-gate pattern (`test_unrestricted_legacy_vs_unified_equivalence.jl`
— compare `:legacy_z` decode/eval/gradient against the old convention at a real D=20 calibration
point) is directly reusable here, since the restricted families' own `A_coordinate_mode` kwarg was
already ported (opt-in) in the prior session (`f1fa8e7`, base of this remediation branch). The
remaining work is: (a) an equivalence gate proving `:legacy_z` and `:powered_aspace` decode to the
same economic point for each of the four families, (b) flipping each family's own kwarg default,
(c) a live CLI smoke test per family. Deferred pending explicit user direction, since — like
Phase A's own default change — it touches checkpoint-schema compatibility for any in-flight
campaign under the current default.
