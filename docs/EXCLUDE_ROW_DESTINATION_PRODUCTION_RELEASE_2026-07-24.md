# exclude-ROW-destination production release — 2026-07-24

Canonical record of the selective port of the validated omit-ROW/C+ rectangular implementation
onto the current `production/fullA-exact` tip. `destination_sample=:exclude_row` is now the
documented production default for CM / CM+mean-ZC / origin-specific-ZC. The unrestricted family
is `:all_legacy`-only for this release (see "Known scoped gap" below). Everything in
`docs/PRODUCTION_STATE_2026-07-24_SCREENS_THRESHOLD10.md` and the earlier `CM_PRODUCTION_STATE_*`
docs about screens/threshold-10/checkpoint mechanics/gradient backend is unaffected and still
accurate — this release is additive on top of that state.

## Release identity

- Release branch: `release/fullA-exclude-row-production-2026-07-24`, fast-forwarded into
  `production/fullA-exact` on both `cdw` (canonical remote) and local.
- Final commit: `fcd8e9e` (was `e162f64477212efde5eeca1a58a6efbf3a0610b1` before this release; local
  and `cdw` tips agreed exactly before the merge).
- Release tag: `exclude-row-destination-production-ready-2026-07-24` (annotated), pushed to `cdw`.
- Provenance: selectively ported from the uncommitted working-tree state of
  `release/fullA-omit-row-restore-screens-2026-07-23` (base `fd21f9f2`, committed head `6f7693b`),
  preserved verbatim as `archive/fullA-omit-row-validated-source-2026-07-24` (6 commits, not merged
  — kept only as an immutable, cherry-pickable record). That prior work's own validation is
  recorded in memory `cplus-exclude-row-production-2026-07-24` (real D=20/W=80,000 sustained
  KNITRO campaigns, `:cplus`-vs-`:reference` cross-checks) and pushed to
  `dropbox:Gravity robustness/Analysis/Server Output/cplus_exclude_row_production_2026-07-24/`.
- Julia: 1.12.6 (juliaup). KNITRO: 13.0.1. Country data: `real_data/noah_D20/*.csv`, D=20, ROW is
  country index 20 (last), unchanged from prior releases.

## Selective-port map

Of the 30 files the archived omit-ROW/C+ work touched, 25 were byte-identical between the fork
point (`fd21f9f2`) and the current production tip — the archive's own diff applied to those
directly, with zero reconciliation needed:

| Category | Files |
|---|---|
| Rectangular C+ backend | `winner_certificate.jl`, `lfix_factorized.jl`, `lfix_factorized_workspace.jl`, `lfix_cm_cplus.jl`, `lfix_incremental.jl`, `lfix_base_workspace.jl`, `composite_gradient.jl`, `composite_gradient_fast.jl`, `oracle.jl`, `oracle_fast.jl`, `winners.jl`, `winners_v2.jl` |
| Rectangular gravity/moments | `gravity_tariff.jl`, `moments_gammanorm.jl`, `moments/hFunction.jl`, `moments/moments!.jl`, `moments/newGravityMoment!.jl`, `misc/doubleDiff.jl`, `prepare_cc/buildObjectsForMoments.jl`, `prepare_cc/master_prepare_cc.jl`, `prestep/master_prestep.jl` |
| Context/screens | `draw_design.jl`, `gravity_elimination.jl`, `infeasibility_screen.jl`, `context_scaled.jl` |
| Test-harness fix | `test_cm_checkpoint_backend_provenance.jl` |
| New validation scripts | `lfix_cplus_exclude_row_validation.jl`, `checkpoint_write_exclude_row_cplus.jl`, `checkpoint_resume_exclude_row_cplus.jl` |

5 files had genuinely diverged (production had independently added the screens-threshold10
release on top of the same fork point) and required hand reconciliation, not a direct apply:

| File | Production-side addition (kept) | Archive-side addition (ported) | How reconciled |
|---|---|---|---|
| `context_real_d20.jl` | `threshold_state` wiring (identical hunk in both — same acae5c7 lineage) | `destination_sample`/`row_idx`/`D_dest` | Diffed `prod_tip → archive` directly (the threshold_state hunk was a no-op since already identical); ported only the destination_sample-specific hunks |
| `cm_production_stage_runner.jl` | Screened preflight check (`cm_production_value_verified_screened`, bug-fix #2 from the screens-threshold10 merge) | `CM_DESTINATION_SAMPLE` env var + threading | Applied archive diff, then manually restored the screened-preflight line the archive's pre-bugfix copy would have reverted |
| `originzc_production_stage_runner.jl` | (no conflict) | `CM_DESTINATION_SAMPLE` env var + threading | Applied directly |
| `cm_checkpoint.jl` | Entire screen-counter/threshold-banner infrastructure (`with_screen_counters`, `print_screen_startup_banner`, `print_screen_summary`, `counters=` on every `*_screened` call) — archive predates this entirely | `CMCheckpointV6` schema, `destination_sample` kwarg/resume-refusal | Hand-ported only the destination_sample-specific additions; zero lines of the screen infrastructure touched |
| `cm_originzc_checkpoint.jl` | Same screen infrastructure | `CMCheckpointV7` schema (extends V5, skips V6 — already claimed by the CM-family bump), `destination_sample` kwarg/resume-refusal | Same hand-port discipline |

## New work beyond the archived port (found live, not in the original validated source)

The archived work only covered the CM/CM+ZC/origin-ZC families ("Part A", the unrestricted
driver, was explicitly deferred in that work's own report). Bringing the unrestricted family up
to the same bar surfaced real bugs the archived validation never exercised:

1. **`c10_d20_production_driver.jl` D² sizing** (`run_profile_checkpointed`/
   `run_polish_checkpointed`): `D2`/`n` were recomputed as `D^2`/`D^2-1`, a genuine
   `D_origin==D_destination` hot-path assumption. Fixed by deriving them from `zfree_start`'s own
   length instead (correct under either regime, matches whatever `free_idx` already computed).
2. **`fast_range_screen.jl` — `precompute_envelope`**: the entire envelope-bound derivation
   (`K2`/`M`/`b`/`Pmat` matrices, `reshape(γo.P,(D,D))`, `Aod_offset+D^2` assertions) is
   square-D-only by construction. Guarded with the same `EnvelopeUnsupportedContext` exception the
   function's other unsupported-config checks already use (`usePMM==1`, μ/σ not fixed) —
   `build_ranged_screen_context` already catches this gracefully (`rsc.envelope=nothing`, screen
   disabled, every downstream call site already gated behind `rsc.envelope !== nothing`). This
   screen's own prior finding (`fullA-fast-range-screen-production-integration`, 2026-07-20):
   **zero organic hit rate** in real D=20 search, verdicted "near-free exact insurance," not a
   measured speedup — disabling it under `:exclude_row` is not a meaningful production
   regression.
3. **`fast_range_screen.jl` — witness loop** (`evaluate_fullA_screened_ranged`, `use_witness`
   defaults to true in production since `ctx.witness !== nothing`): `for d in 1:ctx.D, o in
   1:ctx.D` indexed the `D x D_dest` `Pmat` out of bounds. Fixed using the canonical
   `active_od_cells(ctx)` accessor (`cc_algo/active_layout.jl`).
4. **`compressed_moments.jl`'s `CompressedFactual`**: discovered to be square-D-only *throughout*
   (`Pmat`/`winner`/`wval` all `D x D`, linear-index convention `j=d+(o-1)*D`) — not a guardable
   screen but the unrestricted driver's actual real evaluation machinery
   (`moment_representation=:compressed`, the production default every `cb_F!`/`cb_G!` routes
   through). Rectangularizing it is real, unvalidated engineering out of this release's
   selective-port scope. **Decision (discussed directly with the user): the unrestricted family
   ships `:all_legacy`-only for this release** — loud, explicit guards (not a silent fallback) in
   `run_profile_checkpointed`, `run_polish_checkpointed`, `build_fullA_context`, `reuse_matches`,
   and `unrestricted_stage_runner.jl`. Documented, scoped follow-up, not silently broken.
5. **`compressed_live.jl` — real regression, destination_sample-independent**: `newGravityMoment!`
   gained a `Ddest` parameter as part of the port; this call site (unreachable from any CM/origin-ZC
   entry point, only reachable from the unrestricted driver's real evaluation path) was never
   updated and threw `MethodError` on every real call, **including under `:all_legacy`** — this
   broke legacy reproducibility, not just the new rectangular path. Fixed by passing `ctx.D_dest`
   (a no-op under `:all_legacy` where `D_dest==D`). Other stale-signature callers
   (`autarky_cf.jl`, `autarky_cf_v2.jl`, `smoothed_consistent.jl`, `moments_fast.jl`) were audited
   and confirmed unreachable from any of the four production entry points — standalone D=4
   diagnostic/benchmark scripts, left as-is, flagged here rather than silently ignored.
6. **`unrestricted_stage_runner.jl`** (new file — the unrestricted family had no standalone
   process-launchable CLI before this release, unlike the CM/origin-ZC stage runners, so it could
   not be exercised through the real process-group supervisor at all). Also gained its own trivial
   field-name bug in its final summary print (`res.best.gp`/`.Delta` — the actual returned
   NamedTuple has `.Delta_dual`/`.n_eval`), found and fixed live.
7. **`c10_d20_production_driver.jl` `D20Checkpoint` → `D20CheckpointV4`**: the unrestricted
   family's checkpoint schema had no `destination_sample` field at all. Bumped following the same
   new-type-name Serialization-safe pattern as the CM-family bumps below.

## Active-layout accessors now load-bearing

`cc_algo/active_layout.jl` (`active_origins`/`active_destinations`/`active_od_cells`) existed in
production as a permanent no-op (added by the screens-threshold10 release specifically so this
work could plug in later without touching call sites). `context_real_d20.jl` now populates
`ctx.active_origins = 1:D` / `ctx.active_destinations = 1:D_dest` under `destination_sample`, so
`cm_screen_bridge.jl`'s witness loop (already wired to these accessors) and
`fast_range_screen.jl`'s witness loop (newly wired, this release) now actually exclude the
omitted destination under `:exclude_row` instead of iterating the full square grid.

## Production toggle and defaults

| Entry point | Default | `:all_legacy` |
|---|---|---|
| CM (flexible) | `:exclude_row` | supported, explicit opt-out |
| CM + common mean/ZC | `:exclude_row` | supported, explicit opt-out |
| Origin-specific ZC | `:exclude_row` | supported, explicit opt-out |
| Unrestricted | `:all_legacy` (opposite of every other entry point — see scoped gap above) | the *only* supported value; `:exclude_row` is a hard, immediate error |

Every startup banner now prints (via `print_active_layout_banner`, `cc_algo/active_layout.jl`):
```
[active-layout] mode=<mode> destination_sample=<sym> origins=<n> destinations=<n> active_A_cells=<n> free_reduced_A_dim=<n> gravity_sample_version=<v> theta_calibration_version=<v>
```
`GRAVITY_SAMPLE_VERSION`/`THETA_CALIBRATION_VERSION` (`context_real_d20.jl`) are code-provenance
constants (currently `2` for both, bumped from the implicit pre-release `1`), distinct from the
runtime `destination_sample` choice.

## Economic specification (`destination_sample=:exclude_row`)

- Origins: all 20 countries, including ROW.
- Destinations: the 19 named countries; ROW (index 20, always the last index by construction) is
  excluded as a destination — no destination-ROW variables, winner/probe cells, trade-share
  moments, γ-normalization, or tariffs. `A_{o,ROW}` is never searched, reconstructed, inverted, or
  reported.
- Focal country (`baseIndex`/`bi`, France=2 in production) rejected outright if it ever coincided
  with the omitted ROW destination (`context_real_d20.jl`'s new guard) — GT is undefined for a
  focal country that isn't itself a valid destination in the resolved sample. Confirmed live:
  production's real `baseIndex=2` and `row_idx=20` do not collide.
- Gravity sample recomputed from raw observations on the rectangular sample (all origins x named
  destinations only) — origin/named-destination fixed effects, residualized log trade/tariff,
  θ_star, gravity moment scaling all re-derived, not obtained by deleting a column from the
  legacy full-sample residualized objects. Same frozen underlying uniform draws; θ-dependent
  productivity draws regenerated for the new θ_star.
- Real observed effect: `numMoments` 382 (`:exclude_row`) vs 402 (`:all_legacy`) at the same real
  D=20/W=80,000 point — a genuinely different (smaller) rectangular sample, not a truncated copy
  of the square one.

## Rectangular C+ backend

Same `D_origin`/`D_destination` split pattern as `moments-vs-aod-linear-index-convention`
throughout: `winner_certificate.jl`'s `constCons_matrix`/`WinnerRefCache`, `lfix_factorized.jl`/
`lfix_factorized_workspace.jl`'s `LFixFactorizedWorkspace`/`LFixBaseCacheC`, and
`lfix_cm_cplus.jl`'s `composite_gradient_at_Cplus_from_cache` (the one entry point both CM and
origin-ZC `:cplus` funnel through). `cm_gradient_backend=:cplus` remains the production default
under `:exclude_row`; the old hard-error guard forbidding that combination is gone (real,
validated support, not a guard removal without proof).

## Checkpoint schema

| Family | Old schema | New schema | New fields |
|---|---|---|---|
| CM (flexible/CM+ZC) | `CMCheckpointV4` (schema 4) | `CMCheckpointV6` (schema 6, **skips 5** — already taken by `CMCheckpointV5`, confirmed by direct grep) | `destination_sample`, `row_idx`, `D_dest` |
| Origin-specific ZC | `CMCheckpointV5` (schema 5) | `CMCheckpointV7` (schema 7, extends V5, skips 6 — claimed by the CM bump above) | `destination_sample`, `row_idx`, `D_dest` |
| Unrestricted | `D20Checkpoint` (schema 3) | `D20CheckpointV4` (schema 4) | `destination_sample`, `row_idx`, `D_dest` |

All three: full upgrade-from-old-schema fallback chains, mismatch-refusal hard-errors on resume
(destination_sample changes `n_free`, D² vs D·D_dest — same discipline as the pre-existing
draw-design/K_mean/backend mismatch guards). **`D20CheckpointV4` has no migration path from an
older schema** (matches that file's own pre-existing "start a fresh run" policy — it never had
one, even before this release).

## Gate results (real D=20/W=80,000 unless noted)

**Gate A** (construction-only, small W): 27/27 checks pass. Confirms the task's exact target
numbers: `D_origin=20`, `D_dest=19`, raw active A cells=380, free A coordinates=379 (after the one
gravity-restriction pivot reduction); default (no-arg) resolves to `:exclude_row`; explicit
`:all_legacy` reproduces square (D=20, 400 cells, 399 free) behavior; checkpoint
schema-mismatch/destination_sample-mismatch refusal; `focal_country==ROW` guard.

**Gate B** (real D=20/W=80,000):
- Base dual solves: 4/4 `VerifiedSolved` (CM/origin-ZC × exclude_row/all_legacy), KKT residuals
  ~1e-14.
- Exact-tier (`lfix_incremental_at_Cplus!` vs `fixed_dual_L`): 4/4 pass, max diff 1.8e-15 to
  1.95e-14.
- Full production gradient, `:cplus` vs `:reference`: 4/4 pass, cosine=1.000000000000, max diff
  ~2-4e-15, **speedup 6.84x (CM excl) / 7.15x (CM legacy) / 7.70x (originZC excl) / 7.76x
  (originZC legacy)**.
- Threaded vs serial: exact (0.0 diff).
- Finite-difference sign-agreement: 4/4 pass.
- Genuine K_mean=1/K_pair=1 (not previously tested — prior origin-specific campaigns used
  K_pair=0): CM+mean-ZC and origin-ZC both `VerifiedSolved`, finite residuals, threshold/screens
  active, C+-vs-Reference agreement (max diff 2.2-2.5e-15), **speedup 4.13x (CM+ZC) / 5.82x
  (originZC)**.

**Gate C** (real production-supervisor smokes, `setsid`/`pgid` process-group launch via
`scripts/exclude_row_gateC_smoke_test.sh`, sourcing the real `scripts/cm_production_supervisor.sh`
mechanics): 4/4 families (unrestricted via the new CLI, flexible CM, CM+mean-ZC K=1/K_pair=1,
origin-ZC K=1/K_pair=1) — all 7 steps pass each: real checkpoint written, deliberate
SIGTERM→SIGKILL-if-needed, zero stray processes confirmed, resume through the same supervisor
mechanics, `STAGE_DONE` sentinel, `[active-layout]`/`destinations=19`/screen-infrastructure
banners confirmed in the real log, resume REFUSED under a mismatched `destination_sample`.
Explicit `:exclude_row` proven for `cm`; the omitted/default argument proven for `cmzc`,
`originzc` (both correctly resolve to `:exclude_row`), and `unrestricted` (correctly resolves to
`:all_legacy`).

**Gate D**: `checkpoint_write_exclude_row_cplus.jl` → `checkpoint_resume_exclude_row_cplus.jl`
(genuinely separate processes): resumed eval counters continued (not reset), re-verified Delta at
the checkpoint's own recorded incumbent **bit-identical, diff=0.00e+00**, 4/4 checks pass.
`checkpoint_legacy_toggle_test.jl` (7/7 pass): resume of an `:exclude_row` checkpoint under
`:all_legacy` refused; a short real `:all_legacy` KNITRO smoke ran and wrote a correct legacy
checkpoint (`destination_sample=:all_legacy`, `row_idx=nothing`, `D_dest==D`); resume of that
legacy checkpoint under `:exclude_row` refused.

## Performance (old = `:all_legacy`, new = `:exclude_row`, same real D=20/W=80,000 point)

| Metric | Old (`:all_legacy`) | New (`:exclude_row`) |
|---|---|---|
| Gravity observations (`numMoments`) | 402 | 382 |
| Raw active A cells / free A coords | 400 / 399 | 380 / 379 |
| CM inner dual dim (`d_new`, L=10) | 592 | 572 |
| originZC inner dual dim (`d_new`, K_mean=1) | 422 | 402 |
| CM full-gradient wall, `:reference` / `:cplus` | 46.83s / 6.55s | 57.81s / 8.45s |
| originZC full-gradient wall, `:reference` / `:cplus` | 41.73s / 5.38s | 38.32s / 4.98s |
| `:cplus` speedup vs `:reference` | 7.15x (CM) / 7.76x (originZC) | 6.84x (CM) / 7.70x (originZC) |

The rectangular sample is strictly smaller (fewer destinations, fewer moments, smaller inner dual
dimension) but per-callback wall time is comparable to the square case at this D=20 scale — the
`:cplus` speedup over `:reference` is materially preserved in both regimes (~6.8-7.8x throughout).
These are short-smoke/validation-point timings, not a claim about final bound convergence speed.

## Known scoped gap

Unrestricted family: `destination_sample=:exclude_row` is unsupported and hard-errors
immediately (confirmed live: rejects before any KNITRO work starts, not a silent fallback).
Root cause: `compressed_moments.jl`'s `CompressedFactual` (the real evaluation machinery behind
`moment_representation=:compressed`, the production default) is square-D-only throughout — this
is the unrestricted driver's own "Part A," never touched by the validated omit-ROW work at any
point. Rectangularizing it is real, unvalidated engineering (re-deriving `Pmat`/`winner`/`wval`
sizing and the `j=d+(o-1)*D` linear-index convention for the rectangular case) — a legitimate
follow-up task, out of this release's selective-port scope. `:all_legacy` for the unrestricted
family is fully validated (Gate C smoke passes end-to-end, including the screens/threshold-10
stack and the envelope pre-winner screen, which is *not* disabled under `:all_legacy`).

## Launch commands

```bash
# CM (flexible), destination_sample=:exclude_row is the default -- omit CM_DESTINATION_SAMPLE
julia --project=. -t 20 full_aod_diag/d4_exact/cm_production_stage_runner.jl <ckpt_dir> <delta> <budget_s> calibration ""

# CM + common mean/ZC, K_mean=1 K_pair=1, exclude_row default
CM_EXTENSION=cm_plus_equal_means_zero_covariance MEANZC_K_MEAN=1 MEANZC_K_PAIR=1 \
  julia --project=. -t 20 full_aod_diag/d4_exact/cm_production_stage_runner.jl <ckpt_dir> <delta> <budget_s> calibration ""

# origin-specific ZC, K_mean=1 K_pair=1, exclude_row default
DISTRIBUTION_RESTRICTION=origin_specific_moments_zero_covariance ORIGIN_K_MEAN=1 ORIGIN_K_PAIR=1 \
  julia --project=. -t 20 full_aod_diag/d4_exact/originzc_production_stage_runner.jl <ckpt_dir> <delta> <budget_s> calibration ""

# unrestricted -- :all_legacy only (default; DESTINATION_SAMPLE=exclude_row hard-errors)
julia --project=. full_aod_diag/d4_exact/unrestricted_stage_runner.jl <ckpt_dir> <delta> <budget_s> calibration ""

# explicit legacy reproduction (any CM-family entry point)
CM_DESTINATION_SAMPLE=all_legacy julia --project=. -t 20 full_aod_diag/d4_exact/cm_production_stage_runner.jl ...

# real production-supervisor smoke (any family)
bash scripts/exclude_row_gateC_smoke_test.sh <unrestricted|cm|cmzc|originzc> <stage_dir> <budget_s> [destination_sample]
```

## Finite decision

**READY_AND_MERGED.** Old validated source preserved (`archive/fullA-omit-row-validated-source-
2026-07-24`, not merged). Selective port based on current production
(`e162f64`→`fcd8e9e`). Current screens/threshold-10 code fully preserved (zero lines touched in
the conflicted files' screen infrastructure). `:exclude_row` is the tested default for
CM/CM+ZC/origin-ZC. `:all_legacy` remains explicit and functional for those families, and is the
sole supported mode (loudly guarded, not silently broken) for the unrestricted family. Genuine
CM+ZC/origin-ZC K_pair=1 passes at D=20. All four supervisor smokes cold-verify with zero orphan
processes. Save/resume bit-identical; legacy-toggle mismatch checks pass both directions.
