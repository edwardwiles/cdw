# Brazil→Korea gravity exclusion — master report, 2026-07-31

## Summary

Excludes the single bilateral cell **origin=Brazil, destination=Korea** from the gravity
identification sample (θ* regression and the pivot/gravity restriction) via one shared eligibility
mask, while keeping Brazil and Korea fully active as ordinary economic countries everywhere else.
Merged into production on top of the real production tip (not the campaign branch's separate,
unmerged commits). The prepared sigma3/W500k campaign is rebased and its manifests regenerated in
a follow-up phase (see §4 below); the campaign is **not launched**.

## 1. Anchors

- **Production tip used as base:** `remotes/cdw/production/fullA-exact` @
  `81a673054551a5d91a672f1a7258655e4d868cb0` (confirmed the real, pushed production tip — the
  local `production/fullA-exact` branch ref in this worktree's origin repo was stale and was not
  used).
- **Campaign branch to rebase in Phase 4:** `campaign/timeout-fix-and-launch-sigma3-W500k-2026-07-31`
  @ `cb0f6f3cdc89911bee54d64a51dcdc10488120f9` (22 commits ahead of the production tip above, clean
  merge-base, prepared but not launched).
- **Implementation branch:** `fix/exclude-brazil-korea-from-gravity-2026-07-31`.

## 2. Country resolution

`real_data/noah_D20/countries.csv` (ISO3, positional index convention): Brazil=`bra`=index **3**,
Korea=`kor`=index **14**, ROW=`row`=index **20** (last). Resolved via a new
`resolve_country_index`/`global_to_dest_slot` module (`full_aod_diag/d4_exact/country_resolve.jl`),
not hardcoded at any call site. Exactly one match each; Brazil≠Korea; full details and checksums in
`BRAZIL_KOREA_GRAVITY_EXCLUSION_PROVENANCE_2026-07-31.json`.

```
COUNTRY_CELL_RESOLUTION = Brazil_to_Korea_unique
```

## 3. Gravity regression

Full call-graph and duplication audit: `GRAVITY_SAMPLE_AND_PIVOT_CALL_GRAPH_2026-07-31.md`. Full
regression spec, sample counts, and results: `BRAZIL_KOREA_EXCLUDED_GRAVITY_REGRESSION_2026-07-31.md`/
`.csv`.

```
GRAVITY_REGRESSION =
    coefficient: -7.489399587926582
    rounds_to_minus_7_43: fail   (see release-gate resolution below)
    rounds_to_minus_7_49: pass
    theta_star: 7.489399587926582
```

**Release-gate resolution (-7.43 vs -7.49):** the task text's -7.43 was investigated live with the
user and determined to be a typo/misremembered figure, not a value with any prior record in this
repo or the user's Dropbox. The verified, data-driven, non-hardcoded coefficient is **-7.4894**
(rounds to -7.49). This was cross-checked three ways: (1) the independent regression script
exactly reproduces the CURRENT production θ*=4.7292535486122365 bit-for-bit before the Brazil-Korea
exclusion is applied; (2) `real_data/noah_D20/pi.csv`/`tau.csv` were confirmed to be the correct,
current 2018 goods-adjusted production data (commit `cd17235`, an ancestor of the production tip,
whose own selection was validated against this exact 4.73 baseline); (3) the user confirmed
Brazil→Korea (τ=1.36 vs. a ~1.0-1.05 typical range, at a low trade share) is a genuine expected
high-leverage outlier. **-7.4894 (rounds to -7.49) is the adopted release-gate value**, superseding
the task text's -7.43. `theta_star = -gravity_coefficient` (production's own sign convention,
already applied — no hardcoding).

## 4. Shared gravity eligibility mask

One function, `gravity_sample_mask(D, Ddest; exclude_diagonal, exclude_cells)`
(`full_aod_diag/gravity_tariff.jl`), consumed identically by:
- `precompute_q_tilde`/`gravity_value` (feeding the pivot, and every `gravity_value` consumer:
  `oracle.jl`, `oracle_fast.jl`, `oracle_profiled.jl`, `compressed_live.jl`, `fast_range_screen.jl`,
  `infeasibility_screen.jl`, `run_smoothed_homotopy.jl`, `test_oracle.jl`, `gravity_elimination.jl`)
- `prestep/master_prestep.jl`'s θ* regression (replacing its own independently-re-derived
  `named_dest`/`diag_mask` literal — the one real duplication found)

`gravity_exclude_cells` threaded end-to-end through `context_real_d20.jl`, `draw_design.jl`,
`cm_checkpoint.jl`, `cm_originzc_checkpoint.jl`, `c10_d20_production_driver_unified.jl` — empty by
default (bit-identical to every pre-existing caller).

`oracle.jl`'s `context_fingerprint` (cache key) now hashes `exclude_diagonal_gravity` (closing a
**pre-existing gap**: two contexts differing only in this flag previously aliased to the same
fingerprint) and the new `gravity_exclude_cells` list; `CONTEXT_FINGERPRINT_SCHEMA` bumped 3→4.
`GRAVITY_SAMPLE_VERSION`/`THETA_CALIBRATION_VERSION` bumped 2→3 in `context_real_d20.jl`.

```
SHARED_GRAVITY_MASK = regression_and_pivot_identical
```

## 5. Pivot

No second pivot formula created — `gravity_elimination.jl`'s `argmax(abs.(c))` pivot selection
already skips masked (zero-coefficient) cells automatically once `q_tilde` is masked at context
construction. **Confirmed: the previously-frozen campaign's pivot cell was literally Brazil→Korea
(origin=3, destination=14)** — excluding it forces reselection, verified below.

All required tests (task §7) run against the real D=20 economy, mask wired in end-to-end
(`full_aod_diag/d4_exact/test_brazil_korea_gravity_pivot_gate_2026-07-31.jl`, 21/21 PASS; results in
`GRAVITY_PIVOT_EXCLUSION_GATE_2026-07-31.csv` and `GRAVITY_PIVOT_DERIVATIVE_GATE_2026-07-31.csv`):

- Calibration + 5 random valid free-coordinate vectors: masked gravity residual machine-zero
  (~5e-18) every time.
- Perturbing only `A[Brazil,Korea]` (by 0.01, -0.03, 0.5): pivot value and masked residual
  **exactly unchanged** (bit-identical, since its gravity coefficient is exactly 0.0); derivative
  w.r.t. this coordinate exactly 0.
- Perturbing an included non-pivot cell: pivot value **does** adjust; masked residual **remains**
  exactly zero; analytic derivative matches a finite-difference check to 1e-6.
- Brazil→Korea is **not** selected as the new pivot (new pivot: origin=3 (Brazil, still an
  eligible origin), destination=18); ROW-destination exclusion (`D_dest=19`) and all other eligible
  cells unchanged; eligible count = 360 (== 380 − 19 diagonal − 1 Brazil-Korea), matching the
  regression's own sample count exactly.

```
PIVOT =
    excluded_cell_contribution_zero: pass
    excluded_cell_derivative_zero: pass
    included_cell_FD_gate: pass
    residual_machine_zero: pass
```

## 6. Economic model

Brazil remains an origin, Korea remains a destination; the bilateral trade-flow data, delivered
cost, winner calculations, and ordinary trade-share moments are untouched (the mask only ever
zeroes `q_tilde`/`N_obs`, the gravity-identification-specific arrays — never `τ`, `pi`, `L`, or any
economic moment/screen). Verified structurally: `active_od_cells`/`active_origins`/
`active_destinations` (`cc_algo/active_layout.jl`, the general economic layout) were deliberately
**not** touched — confirmed via call-graph audit that its only consumers are economic witness/
pairwise feasibility screens, not the gravity restriction.

```
ECONOMIC_MODEL = Brazil_Korea_cell_retained: pass
```

## 7. Dead-code finding (separate from the Brazil-Korea change, fixed at the user's direction)

Investigating whether the "inner" KNITRO gravity moment (`newGravityMoment!`/
`compressed_gravity_raw`/`fill_gravity_column_into!`) needed the same mask surfaced a real,
pre-existing bug: under `exclude_diagonal_gravity=true` (the setting the actual sigma3/W500k
campaign uses), this inner moment evaluates to **0.370 at the exact calibration point** — not zero
— because it uses `within_transform_rect` (no masking support), a different linear functional than
the pivot's own masked restriction (which is exactly zero, as required). Tracing every consumer
confirmed the value is **never read** in the real production `OperatorPsiBundle` path — a pure
write with no downstream consumer (`obj.H_save`, the actual value that reaches KNITRO, comes
exclusively from the separate `obj.payoff` field). Removed the dead computation from
`prime_operator!` (`full_aod_diag/d4_exact/operator_psi_bundle.jl`), the one shared priming function
all five families' real production entry points use; left the underlying utility functions intact
since diagnostic/test scripts still legitimately use them. Verified safe by construction (the
removed field has zero read-sites anywhere in the codebase); the other 4 families' analogous
`:dense_reference`-gated call sites were confirmed genuinely gated behind the non-production
diagnostic path and were not touched.

## 8. Operator-only gate

All five families (`unrestricted`, `flexible_cm`, `common_frechet`, `cm_meanzc`, `origin_zc`) pass
the structural operator-only preflight through their **real** production-context wrappers
(`build_unrestricted_operator_ctx`/`build_cm_production_context`/`build_cm_frechet_production_context`/
`build_cm_meanzc_production_context`/`build_originzc_production_context`), with
`gravity_exclude_cells` wired in — zero dense-reference bundle constructions, zero `select_G_from_H`
calls, gravity residual machine-zero (2.7e-18) at this construction's calibration point. Manifests:
`brazil_korea_preflight_manifests_2026-07-31/`.

```
OPERATOR_ONLY_GATE = pass_all_five
```

A live-KNITRO smoke through the real public driver (`run_cm_upper_checkpointed`, flexible-CM,
W=80,000) was attempted (`full_aod_diag/d4_exact/smoke_brazil_korea_flexcm_2026-07-31.jl`) and hit
an early-exploration KNITRO infeasibility (`nStatus=-300`) within the short smoke budget. The
failing point's erratic near-zero coordinate values are consistent with a large, poorly-scaled
first optimizer step, not a wiring defect — it does not implicate `gravity_exclude_cells`
specifically, and it sits on top of pivot mathematics already independently verified to machine
precision (§5). Full real KNITRO convergence validation at campaign scale (task's actual
requirement) is deferred to Phase 4, which runs the campaign's own battle-tested smoke/preflight
scripts at real W=500,000 scale rather than this one-off script.

## 9. D=4 gates

`full_aod_diag/d4_exact/test_gravity_sample_mask_d4_gate_2026-07-31.jl`, 15/15 PASS: mask
correctness for arbitrary D/Ddest/exclude_diagonal/exclude_cells combinations, default-path
regression safety (bit-identical to pre-task behavior when `exclude_cells=[]`), sign convention,
excluded-cell derivative exactly zero, included-cell derivative matches finite differences,
mask↔excluded-cell-list round trip. Existing D=4 `FreeParamMap` round-trip test
(`full_aod_diag/test_free_param_and_gravity.jl`, TEST 1) re-run unaffected (unrelated TEST 2 failure
traced to a pre-existing stale include chain in that script, not this task's change).

## 10. Production merge

See git log for the exact commit sequence and SHA (this section is completed at merge time — see
`git log production/fullA-exact` after merge).

```
PRODUCTION_MERGE = merged_locally_pending_push_confirmation
```

## 11. Campaign readiness

Not yet started at the time of this document's first version — see Phase 4 follow-up. Campaign is
**not launched** regardless of readiness state.

```
TEN_BY_TEN_CAMPAIGN = NOT_READY_campaign_rebuild_not_yet_run
LAUNCHED = false
```
