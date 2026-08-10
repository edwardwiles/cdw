# CM+ZC-CROSS, and both CROSS families made campaign-selectable — 2026-08-09

**Branch:** `feature/ozc-cross-2026-08-09`, worktree `/bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09`,
based on `origin/production/fullA-exact`@`4df5254`. **Everything is uncommitted working-tree state.
Nothing pushed to any remote.**

**Task:** (1) build `CM+ZC-CROSS`, the common-marginals analog of the already-built `OZC-CROSS`
family; (2) make BOTH cross families first-class, selectable options in the existing production
campaign infrastructure, usable *instead of* the two current ZC families with no ad-hoc scripts.

Both parts are complete and gated. Section 7 records what is genuinely *not* settled.

---

## 1. What CM+ZC-CROSS is

For every unordered origin pair `(o,p)`, `o<p`, and every **ordered** level pair
`(k1,k2) ∈ {1..K_pair}²`:

```
E_F[ z_o(ω)^k1 · z_p(ω)^k2 ] = ν_k1 · ν_k2
```

`K_pair²` restrictions per origin pair instead of the diagonal family's `K_pair`. Under common
marginals every origin shares one marginal and therefore one `ν_k` per level, so the target is
origin-free — simpler than OZC-CROSS's `ν_{o,k1}·ν_{p,k2}`.

**No new outer parameters.** `n_eta = K_mean` exactly as before, so every `eta_nu`-shaped caller,
checkpoint field, campaign `w0` and seed-generator coordinate keeps its length. That is what makes a
diagonal-vs-cross comparison at matched `K` interpretable: the two families differ *only* in the pair
block.

### New files (all additive)

| File | Contents |
|---|---|
| `cm_meanzc_cross_target_layout.jl` | `SharedByPowerCrossLayout` + its `pair_targets` method |
| `cm_meanzc_cross_moments.jl` | `n_meanzc_cross_moments`, `build_cm_meanzc_cross_augmented_obj`, and the nu-gradient trio |
| `cm_meanzc_cross_production.jl` | context builder, `meanzc_cross_fixed_contribution`, `build_lfix_base_cache_cm_meanzc_cross`, `cm_meanzc_cross_production_gradient` |
| `cm_meanzc_cross_cplus.jl` | Backend C+ trio (`cm_gradient_backend=:cplus` is the driver's DEFAULT) |

`build_raw_cross_pair_matrix_levels` / `cross_pair_level_index` are **reused verbatim** from
`cm_originzc_cross_moments.jl` — those build origin-indexed raw feature columns and are indifferent to
whether ν is shared or origin-specific.

### Reused completely unchanged (confirmed by reading, then by the gates)

`ZCRestrictionOperator` (keyed off `length(Zpairraw_all)`, so `n_pair(op)` becomes `K_pair²·npair`
automatically), `refresh_zc_targets!` (calls `pair_targets`, so it dispatches to the new layout by
itself), `CMMeanZCOperatorState`, `zc_restriction_gram!`, `bin_zc_cross_hessian_fill!`,
`winner_pair_cross_hessian_zc_block!`, `cm_fixed_contribution_meanzc_layout`,
`composite_gradient_at_Cplus_from_cache`, and the whole CM-grid block. Column layout is
`[econ | mean | pair | CM-grid | gravity]` and every CM offset derives from `aug.n_pair`, so the
widened pair block auto-shifts the CM columns correctly.

---

## 2. The one genuinely new piece of mathematics — and why it needed an FD check

The base CM+ZC family writes **one independent gradient component per level**
(`d_delta_dual_d_nu_vec`, `cm_meanzc_moments.jl:589-608`, using `d_pair_dnu = -2ν`), because its outer
Jacobian is block-diagonal in `k`: level `k`'s moment columns depend on `ν_k` only.

**That is false for the cross grid.** Block `klin = (k1,k2)` has target `ν_{k1}·ν_{k2}`, so it
contributes to **two different slots**:

```
∂Δ/∂ν_{k1} += −ν_{k2}·Σ_j λ*_pair,klin,j
∂Δ/∂ν_{k2} += −ν_{k1}·Σ_j λ*_pair,klin,j
```

When `k1 == k2` both land in the same slot and sum to exactly `−2·ν_k·Σ_j λ*` — i.e. the formula
reduces to the base family's term-for-term. That reduction is checked with **zero tolerance** (check
A2 below: `max abs diff = 0.0`).

Using `Σ_j λ*` (a plain sum) rather than a per-pair `dot` is correct **here and only here**, because
the shared-ν target is origin-free; OZC-CROSS, whose targets are origin-specific, necessarily keeps a
per-pair loop.

### Variant D under a shared layout — the piece the handover said not to assume

Under `SharedByPowerCrossLayout`, `target_index(layout,o,k) = k`, so `aml.dense_omit_idx == aml.kstar`.
The focal `ν_{k*}` therefore appears in cross-pair blocks at **both** `(k*,k2)` for every `k2` **and**
`(k1,k*)` for every `k1` — `2·K_pair − 1` blocks, versus the diagonal family's single `(k*,k*)`. So
`d_delta_d_nu_star` collects strictly more terms. The accumulate-into-both-slots loop handles it, but
it was FD-checked explicitly rather than by inspection (check C).

Also load-bearing, and inherited from the base family's own bug-fix comment: under a shared layout
`n_mean` must be `mean_offset_from_aml(aml)[end]` (a **row** count, `K_mean·D − 1`), **not**
`aml.n_eta_active` (`K_mean − 1`). Those coincide for origin-specific layouts and differ by orders of
magnitude here. Asserted directly in the gate.

---

## 3. Verification — what was actually run, and what it showed

All at D4 unless stated. `julia --project=. -t 4`.

**Every D4 gate below was re-run once more at the very end, against the final code state** (after all
late edits, including the `companion_implied_nu_originzc` signature fix of section 6b):
**119 PASS, 0 FAIL** across all four. Transcript: `key_results/14_final_consolidated_gate_rerun.txt`.

### `smoke_cmzc_cross_d4_2026-08-09.jl` — **ALL PASS** (K = 1/1, 2/2, 3/3)

Real KNITRO inner solves converge; every new cross-power restriction is genuinely satisfied at the
recovered primal weights (not merely "didn't crash"): residuals **1e-12 … 1e-16** across all 9 cross
blocks at K=3/3. Dimension/dispatch assertions confirm `n_eta` unchanged, `n_pair = K_pair²·npair`,
and that `build_cm_meanzc_bin_ctx` actually picked up the cross layout rather than silently
constructing `SharedByPowerLayout`.

At K=1/1 base and cross give **bit-identical** `Δ* = 4.883042358024e-03` — the degenerate-case
sanity check.

### `verify_cmzc_cross_gradient_d4_2026-08-09.jl` — **ALL PASS**

| Check | Result |
|---|---|
| (B) fixed-contribution fold vs independent operator-FG recompute of `q0` | max abs diff **1.7e-15 / 4.0e-14 / 4.5e-11** (K=1/1, 2/2, 3/3) |
| (A) analytic η-gradient vs reoptimized central FD, h=1e-6 | rel **3.3e-6 / 6.1e-6 / 5.9e-6** |
| (A2) cross formula vs base diagonal formula at K_pair=1 | **0.0** (exact) |
| (C) Variant D active η-gradient vs FD | rel **2.0e-5 / 2.0e-6** (K=2/2, 3/3) |
| (C) `d_delta_d_nu_star` (the cross-coupling term) vs FD | rel **3.3e-8 / 5.1e-6** |

Check (B) runs **before** the FD probes on purpose: the FD helper re-solves at many perturbed η
points and leaves `cctx.cmlookup_st`'s cached state at the last probe, so running (B) afterwards would
compare a calibration-point `λ*` against stale operator state. Same trap, same ordering fix, as the
OZC-CROSS gate.

### `verify_cmzc_cross_cplus_ab_d4_2026-08-09.jl` — **ALL PASS**

C+ backend vs the reference gradient at the same point, **rel 6.3e-16 … 4.6e-19**, for K = 1/1, 2/2,
3/3, with and without Variant D. Both backends are passed the same `h_mode` **and the same shared
`bandwidth_cache` Dict** — a mismatch there fakes a ~1e-3 gap that reads as a real bug.

### `test_cmzc_cross_wiring_2026-08-09.jl` — **29/29 PASS, 0 FAIL** (solver-free)

Config resolution, layout dispatch, checkpoint schema-11 round-trip, schema-10 auto-upgrade, backend
manifest family symbols, and the reproducibility digest (section 6).

### `test_cross_seed_family_dispatch_2026-08-09.jl` — **ALL PASS** (real D20 context)

The seed-campaign counterpart of the wiring gate, covering what a solver-free test cannot: that
`build_family` / `derived_focal_nu` / `evaluate_family` actually dispatch `:origin_zc_cross` and
`:cm_zc_cross` to the right layout and context builder, and that a real verified evaluation comes
back through the same screened evaluators the diagonal kinds use. D20/W=100k/K=2/2:

| kind | Δ\* | verified | `inner_status` | class | wall |
|---|---|---|---|---|---|
| `:origin_zc_cross` | 2.0895397955e-03 | true | 0 | `VerifiedSolved` | 126.5 s |
| `:cm_zc_cross` | 2.2123187224e-03 | true | 0 | `VerifiedSolved` | 109.8 s |

This gate is what caught the one real bug in the integration (section 6b).

### Real D20 runs through the actual production entry points

- `production_smoke_cmzc_cross_2026-08-09.jl` — `run_cm_upper_checkpointed`, W=100,000, L=50, K=2/2,
  Variant D (`meanzc_profiled_level=2`), `:powered_aspace`, `:exclude_row`, σ=3.0, Sobol, on the
  driver's **own default** `cm_gradient_backend=:cplus`:

  ```
  eval 1 t= 101.0s gp=0.9840278852 Delta=0.0022463109 feasible=true  verified=true
  eval 2 t= 356.1s gp=0.9821309083 Delta=2.9809220655 feasible=false verified=true
  eval 3 t= 621.9s gp=0.9838997768 Delta=1.0524803994 feasible=false verified=true
  RESULT: wall=1304.7s knitro_status=-401 n_eval=6 n_grad=3 kappa=0.039991299891983134
    best feasible: gp=0.9731581988  Delta=0.6327085626  at eval 5 (t=853.3s)
    kappa = 3.999130%
    checkpoint: schema=11  meanzc_target_layout=shared_by_power_cross  cm_extension=cm_plus_moments
                K_mean=2 K_pair=2 n_eta_stored=1  reason=stage_complete
    checkpoint layout tag round-trips as CM+ZC-CROSS: true
    checkpoint eta length matches Variant D active count (1): true
    backend manifest family = cm_meanzc_cross
  ```

  `-401` is budget termination — this is a **wiring smoke, not converged science**. Every outer
  evaluation verified; the schema-11 layout tag and the Variant D active-η count both round-tripped
  through a real checkpoint write.
- `campaign_cm_family_runner.jl` run directly for `cm_meanzc_cross` and `origin_zc_cross` (and the two
  base families as controls) against a real 5-start manifest generated by
  `make_cross_campaign_test_manifest_2026-08-09.jl`. See section 6a — this is what surfaced the
  pre-existing `UndefKeywordError` breakage.

  All four cells completed, zero exceptions:

  | family | wall | status | n_eval | verified | best Δ | layout selected |
  |---|---|---|---|---|---|---|
  | `cm_meanzc_cross` | 515.7 s | −401 | 1 | 1/1 | 2.246311e-03 | `shared_by_power_cross` |
  | `origin_zc_cross` | 422.0 s | −401 | 1 | 1/1 | 2.089531e-03 | `origin_by_power_cross` |
  | `cm_meanzc` (base) | 504.8 s | −401 | 1 | 1/1 | 1.197542e-03 | `shared_by_power` |
  | `origin_zc` (base) | 382.2 s | −401 | 1 | 1/1 | 1.088536e-03 | `origin_by_power` |

  cross/base Δ\* ratios at K=2/2: **CM+ZC 1.876×, origin-ZC 1.920×**. The CM+ZC figure independently
  matches `bench_cmzc_cross_ncore_ext`'s 1.88×, measured through a different entry point with a
  different ν seeding.

  The `cm_meanzc_cross` cell's log:

  ```
  >> K resolution: MEANZC_K=2 ORIGINZC_K=2  (from manifest config)
  >> required scientific params: include_truncated_moment=true (runner constant)  inner_lower_limit=-10.0 (runner constant)
  >> CROSS family: 4 ordered (k1,k2) restrictions per origin pair (vs 2 diagonal); n_eta UNCHANGED
  [cm_meanzc_cross_upper_d1.0_s1] cm_extension=cm_plus_moments K_mean=2 K_pair=2 meanzc_target_layout=shared_by_power_cross
  [cm_meanzc_cross_upper_d1.0_s1] DONE  wall=515.7s  knitro_status=-401  n_eval=1  verified=1/1  best_Delta=2.246311e-03
  ```

  **Independent cross-check:** that cell's `best_Delta = 2.246311e-03` matches the *direct driver*
  smoke's eval-1 `Delta = 0.002246310887256548` **exactly**. Two different entry points
  (campaign runner → driver, vs. the standalone smoke → driver) reach the same number at the same
  point, so the campaign layer is not perturbing the science.

  Both runs' persisted `*_backend_manifest.json` record `"family": "cm_meanzc_cross"` — the
  provenance separation of section 6e, verified on real artifacts rather than asserted.

- The campaign shell layer was exercised against the real `run_campaign_wave.sh` /
  `run_full_campaign_supervisor.sh` with a stub `julia` and an isolated `HOME` (so the scripts' own
  `export PATH="$HOME/.juliaup/bin:$PATH"` could not shadow the stub). Default wave = the historical
  five byte-identically; cross wave substitutes correctly; `unrestricted` still routes to its own
  separate runner; a bad family name exits 2 **before** launching any chain. Full transcript in
  `key_results/08_campaign_shell_dispatch.txt`.

---

## 4. An apparent bug that was not one — recorded so it is not re-investigated

Check (A) initially ran at **h = 1e-4** and showed a **3–5 % relative gap** at every K. That looks like
a wrong analytic formula.

It is not. Check (A2) already showed the cross formula reproduces the base family's own
`d_delta_dual_d_eta_nu_vec` **bitwise** at K_pair=1 — so if the restructure were the cause, (A2) could
not pass. Following the repo's own control-first rule, `control_base_cmzc_etagrad_fd_d4_2026-08-09.jl`
ran the identical comparison against the **unmodified base diagonal family** and swept `h`:

| h | base rel gap | cross rel gap |
|---|---|---|
| 1e-3 | 8.2e-1 | 8.2e-1 |
| 1e-4 | 3.3e-2 | 3.3e-2 |
| 1e-5 | 3.3e-4 | 3.3e-4 |
| 1e-6 | 3.3e-6 | 3.3e-6 |
| 1e-7 | 3.4e-8 | 3.4e-8 |

Textbook **O(h²) central-difference truncation error**, with an identical signature in both families.
A genuine formula error would have been flat in `h`. `|∂Δ/∂η_ν|` runs 1e1–1e3 against `Δ ~ 1e-2` here,
which is why h=1e-4 is far too coarse for this family even though it is fine for OZC-CROSS.

The gate now uses **h = 1e-6** with a `rel < 1e-4` tolerance (~30× the observed residual, so still a
real gate). **Do not raise h back to 1e-4.** `h = 1e-3` is unusable at K=3/3: the perturbed ν leaves
the feasible set and the inner solve returns `nStatus = -300`, a genuine infeasibility certificate —
also reproduced in the base-family control, so it is a property of the problem, not of this family.

---

## 5. A real pre-existing bug found and fixed in shared code

`build_cm_meanzc_bin_ctx` (`cm_meanzc_production.jl`) does a runtime `include("zc_restriction_operator_ragged.jl")`
**inside its own function body** and then immediately calls the just-defined 4-argument
`ZCRestrictionOperator(..., aml)` constructor **in the same frame**. Julia's world-age rules pin an
executing frame to the world it was entered at, so that call fails:

```
MethodError: no method matching ZCRestrictionOperator(::Vector{Matrix{Float64}},
             ::Vector{Matrix{Float64}}, ::Int64, ::ActiveMeanLayout{...})
```

Confirmed live on the first cold-process call with an aml-active `aug`.

This is **not** specific to CM+ZC-CROSS: it bites the **base diagonal CM+ZC family identically**
(`meanzc_profiled_level` + `:operator`, i.e. the real production Variant D config). Production scripts
survive it only by happening to list `zc_restriction_operator_ragged.jl` in their own top-level
includes. It is the same bug class already found and fixed in `build_originzc_core_hess_ctx`, whose
own fix comment noted "the same pattern exists elsewhere" — this is the elsewhere.

Fixed the same way, at both constructor sites: `Base.invokelatest`. Zero behavior change, negligible
cost (once per context build).

---

## 6. Campaign integration

The handover's warning about **two distinct things called "five family"** is real and both were changed.

### 6a. Execution campaign

**A second pre-existing breakage, found only by actually running the campaign runner end to end.**
Every family arm of `campaign_cm_family_runner.jl` — not just the new ones — has been unable to launch:

```
cm_meanzc / flexible_cm / common_frechet  ->  UndefKeywordError: `include_truncated_moment`
origin_zc                                 ->  UndefKeywordError: `inner_lower_limit`
```

**Confirmed pre-existing by running the unmodified BASE families through the same runner first**
(the repo's own control-first rule) — they fail identically. Those two parameters were correctly made
required-with-no-default by the 2026-08-05 truncated-power and 2026-08-06 lower-limit hardening
passes; this runner was never updated to supply them. That is exactly the intended consequence
CLAUDE.md describes for the ~200 old diagnostic scripts, except this one is a **production campaign
entry point**, so it had to be fixed rather than left throwing.

Fixed by supplying both **explicitly** — from the manifest's own `config` block when the manifest
records them (the manifest is the run's scientific provenance), otherwise from named, documented
runner constants carrying the current production values (`include_truncated_moment = true`,
`inner_lower_limit = -10.0`). Every *other* scientific parameter these drivers take already defaults
to the production value (σ=3.0, Brazil–Korea gravity exclusion, `exclude_diagonal_gravity=true`,
`destination_sample=:exclude_row`, `A_coordinate_mode=:powered_aspace`) and was left alone.

**Deliberately not changed:** `meanzc_profiled_level` / `originzc_profiled_level` (Variant D) stay at
the drivers' `nothing` default unless a manifest asks for them. Turning Variant D on is a real
scientific change to the *base* arms' campaign behavior and is not this task's to make — but note it
**is** the intended production spec at K≥2, so a campaign wanting it must set `profiled_level` in its
manifest config block. Flagging rather than silently deciding.

Other changes:

- `campaign_cm_family_runner.jl`: whitelist now accepts `cm_meanzc_cross` / `origin_zc_cross`; each
  selects itself purely through `meanzc_target_layout` / `power_target_layout` — same
  `cm_extension`/`distribution_restriction`, same `K`, same `w0`/`eta_nu` length as its diagonal
  counterpart.
- **K resolution hardened (fixes a pre-existing latent hazard).** The manifest's
  `shared_extra_coordinates` ν vectors are sized at the K the *manifest* was built at, while the
  runner hardcoded its own `MEANZC_K = ORIGINZC_K = 1`. Those only ever agreed by convention — a
  manifest built at K=2 fed to this runner produced a `w0` whose `eta_nu` block was the wrong length.
  The manifest is now authoritative when it records K (the historical constants remain as fallbacks),
  and ν length is **asserted** against `n_eta` before any solve.
- A K_pair=1 cross run is warned about loudly: the `K_pair²` grid then has exactly one combo `(1,1)`,
  making the arm mathematically identical to the diagonal family it is meant to be compared against.
- `run_campaign_wave.sh` / `run_full_campaign_supervisor.sh`: family list is now the env variable
  `CAMPAIGN_FAMILIES`, defaulting **byte-identically** to the historical five. The supervisor validates
  the list up front (fail fast) and exports it so the two scripts cannot disagree. To run a cross wave:

  ```bash
  CAMPAIGN_FAMILIES="flexible_cm common_frechet cm_meanzc_cross origin_zc_cross unrestricted" \
    ./run_full_campaign_supervisor.sh ...
  ```

### 6b. Reproducible multistart-seed campaign

- New `kind` symbols `:origin_zc_cross` / `:cm_zc_cross`, constructors
  `origin_zc_cross_family_spec` / `cm_zc_cross_family_spec` (both **refuse `K_pair < 2`**), and a new
  preset `production_five_family_cross_seed_specs` that swaps only the two ZC arms. The three non-ZC
  arms deliberately keep their diagonal-wave ids, so a cross wave and a diagonal wave should reproduce
  identical seeds for them — a free cross-check.
- `build_family`, `derived_focal_nu`, `evaluate_family` dispatch the new kinds. The screened
  evaluators were already layout-agnostic.
- **One real bug in this integration, caught by the real-context dispatch gate** (and impossible for
  the solver-free wiring gate to catch, since it never calls the function):
  `companion_implied_nu_originzc`'s signature was typed `::OriginByPowerLayout`, so an
  `OriginByPowerCrossLayout` hit a `MethodError` inside `evaluate_family`. Its **body** was already
  layout-agnostic — it reads only `K_mean` / `n_eta` / `target_index`, all identical between the two
  layouts, and it solves the truly *unrestricted* companion, which has no ZC structure at all — so
  the fix was to widen the signature to a `Union`, not to change any math.

### 6c. The reproducibility-digest collision — fixed and verified

`family_spec_descriptor` encodes `(id, kind, K_mean, K_pair, L, contrasts, include_truncated_moment,
meanzc_basis, probs)`. A diagonal and a cross spec agree on **every one of those** — the cross
extension adds no scientific parameter. So a boolean/enum layout flag bolted onto the existing kinds
would have left two genuinely different economic problems sharing a digest.

Fixed **two ways**:

1. **Distinct `kind` symbols** — makes the descriptor differ structurally, and changes **no existing
   digest** (unlike adding a field to the descriptor, which would have invalidated every archived
   manifest's digest for no benefit).
2. **`assert_distinct_family_descriptors`**, called unconditionally from `compute_manifest_digest` —
   hard-fails if any two specs in a requested list produce the same descriptor, whatever mechanism a
   future spec field uses.

Verified in the wiring gate, at **identical `id`, `K_mean`, `K_pair`, `L`, `contrasts`, `basis`**:

```
diagonal digest: e5a6d3ff5f6ba6addfecfa4193b0540ea35071010aa64385b2f2de8abb6a01bd
cross    digest: 5dd9cea6d2c148d09a0bac91000ad0aed0a0672719bc7e3fdc16a00ad153e1e7
```

### 6d. Checkpoint schema 11

CM+ZC persisted **no layout field at all** (unlike origin-ZC). The layout is not recoverable from any
other stored field, so `CMCheckpointV11` adds `meanzc_target_layout`, and the driver **hard-refuses** a
resume mismatch on it — the same no-escape-hatch discipline already applied to `cm_extension` /
`meanzc_K_mean` / `destination_sample`.

Schema-10 files **are** auto-upgraded, filling `:shared_by_power` — correct by provenance, not a guess:
every schema-10 file predates the cross layout's existence. Same argument as
`upgrade_schema8_to_v9`'s `:legacy_z`. The schema-9-and-older hard refusal is untouched.

Both directions of the guard are tested on a **real** checkpoint written by the D20 production smoke:

- *positive* — resuming the cross checkpoint as cross works and carries state forward:

  ```
  [cmzc_cross_K2_W100000] RESUMING from ...cmzc_cross_K2_W100000_latest.jls (n_eval=6 n_grad=3 wall_elapsed=1244.9s)
  RESULT ... resume=true: wall=648.8s knitro_status=-401 n_eval=8 n_grad=4 kappa=0.039991299891983134
    best feasible: gp=0.9731581988  Delta=0.6327085626  found at eval 5 (t=853.3s)
    checkpoint layout tag round-trips as CM+ZC-CROSS: true
    checkpoint eta length matches Variant D active count (1): true
  ```

  `n_eval` 6 → 8, `n_grad` 3 → 4, and the incumbent found at eval 5 of the *fresh* run survives the
  resume intact;
- *negative* — `test_cmzc_cross_resume_guard_2026-08-09.jl`, **ALL PASS**:

  ```
  run_cm_upper_checkpointed(guardtest): meanzc_target_layout MISMATCH on resume -- checkpoint was
  written with meanzc_target_layout=:shared_by_power_cross, this call requests :shared_by_power --
  refusing to resume under a different pairwise-ZC restriction ...
  ```

  This is the case that actually matters: before schema 11 there was *nothing in the file* to
  distinguish the two, and every other resume guard would have passed.

`MEANZC_MOMENT_LAYOUT_VERSION` deliberately **not** bumped: it tracks the diagonal family's column
order, which the cross family does not touch. Bumping it would invalidate in-flight diagonal resumes
for nothing.

### 6e. Cache-key and provenance separation

At identical `(K_mean, K_pair)` the two families give genuinely different `Δ*`, so an exact-cache
collision would be a **silent wrong answer**, not a perf nit:

- driver `family_tag` is `:cm_meanzc_cross` (mirroring OZC-CROSS's `:origin_zc_cross`), and
  `family_tag`/`family_tag_pre` were collapsed to **one** source of truth so a future third arm cannot
  be added to only one of them;
- `resolve_flexible_cm_manifest` / `resolve_origin_zc_manifest` report the cross family symbols, so the
  persisted backend manifest cannot mislabel a cross run in its own provenance record.

---

## 7. Cost, and what remains open

### Measured cost (`bench_cmzc_cross_ncore_ext_2026-08-09.jl`, real D20/W=100,000/L=50, `-t 8`)

| | NCORE_ext | packed H | Ews | 1st solve | steady-state solve | Δ* | status |
|---|---|---|---|---|---|---|---|
| base K=2/2 | 802 | 55.7 MB | 612 MB | 53.5 s | 1.89 s | 1.186e-3 | 0 |
| **cross K=2/2** | **1182** | **72.5 MB** | **902 MB** | **187.4 s** | **54.5 s** | **2.230e-3** | **−100** |
| ratio | 1.47× | 1.30× | 1.47× | 3.5× | **28.9×** | 1.88× | |
| base K=3/3 | 1012 | 64.7 MB | 772 MB | 94.0 s | 2.92 s | 4.531e-3 | 0 |
| **cross K=3/3** | **2152** | **125.3 MB** | **1642 MB** | **432.0 s** | **293.4 s** | **3.937e-2** | **−100** |
| ratio | 2.13× | 1.94× | 2.13× | 4.6× | **100.4×** | 8.69× | |

(`ncm = 1900` at L=50 two-family; peak RSS across a context build 1.7–2.6 GB in every cell.)

**Memory is a non-issue** — the feasibility risk the handover flagged as "the single biggest" did not
materialize. Even at K=3/3 the packed Hessian is 125 MB and total RSS growth ~2.6 GB, on a 3 TB
machine. The packed Hessian grows only 1.94× despite a 9× pair block, because `ncm = 1900` dominates
the `(NCORE_ext + ncm)²` dimension — the CM-grid block, which OZC-CROSS does not carry, is what
cushions it.

**Wall-clock is the real cost.** ~293 s per inner solve at K=3/3, W=100k — comparable in absolute
terms to OZC-CROSS's 200–280 s, but ~100× the diagonal CM+ZC family's steady-state 2.9 s. The
steady-state ratio is far worse than the first-solve ratio (100× vs 4.6×) because the *base* family
warm-starts almost for free from its own converged iterate while the cross family does not converge
at all (`-100`) and re-does most of the work. **A K=3/3 cross campaign cell should be budgeted at
~5 min per outer evaluation**, i.e. an 1800 s budget buys roughly 5–6 evals — a wiring smoke, not
converged science.

### Still open (inherited, not introduced)

- **The cross inner solve stops at `inner_status = -100`** while the base family reaches `0` at the
  same point, at both K=2/2 and K=3/3. Identical to OZC-CROSS's own open item, which persisted at
  every W tested up to 500k. Not investigated here, and it is the direct cause of the 29×/100×
  steady-state timing gap above.

  **A lead, not a finding:** `test_cross_seed_family_dispatch_2026-08-09.jl` got `inner_status = 0`
  (`VerifiedSolved`) for **both** cross families at D20/W=100k/K=2/2 —
  OZC-CROSS Δ\*=2.0895e-03 in 126.5 s, CM+ZC-CROSS Δ\*=2.2123e-03 in 109.8 s — whereas the
  benchmark at the same scale returned `-100`. Two things differ between those runs at once (the
  seed generator uses the **companion-LFD-implied** ν policy and has Variant D active; the benchmark
  used theoretical-Γ ν with no `aml`), so this does **not** isolate a cause. It does suggest the ν
  policy and/or Variant D is worth trying first if someone picks up the `-100` item — a controlled
  one-factor-at-a-time comparison would settle it cheaply.
- `Δ*` cross/base = **1.88× at K=2/2 and 8.69× at K=3/3**. The K=3/3 figure closely echoes
  OZC-CROSS's own 7.03× at the same K and W, and the prior diagnostic campaign established that
  *that* number is a finite-sample effect decaying as ~W^-1.5 with the ratio plateauing near 4.7.
  The parallel is suggestive but **not established for this family** — no W-scaling was run for
  CM+ZC-CROSS. Do not assume the OZC-CROSS W-scaling result transfers.

---

## 8. Files changed

**New production code (4):** `cm_meanzc_cross_target_layout.jl`, `cm_meanzc_cross_moments.jl`,
`cm_meanzc_cross_production.jl`, `cm_meanzc_cross_cplus.jl`.

**New gates / tools (10)**, all `_2026-08-09.jl`: `smoke_cmzc_cross_d4`,
`verify_cmzc_cross_gradient_d4`, `verify_cmzc_cross_cplus_ab_d4`, `test_cmzc_cross_wiring`,
`control_base_cmzc_etagrad_fd_d4`, `bench_cmzc_cross_ncore_ext`, `production_smoke_cmzc_cross`,
`make_cross_campaign_test_manifest`, `test_cross_seed_family_dispatch`,
`test_cmzc_cross_resume_guard`. Plus this document.

**Modified (9):** `cm_meanzc_production.jl` (layout pickup + world-age fix), `cm_meanzc_config.jl`
(`meanzc_target_layout`, `meanzc_make_layout`), `cm_checkpoint.jl` (schema 11 + driver wiring),
`cm_originzc_checkpoint.jl` (manifest family symbol), `production_backend_manifest.jl` (both family
symbols), `campaign_cm_family_runner.jl` (cross families + K resolution + the required-kwarg fix),
`run_campaign_wave.sh`, `run_full_campaign_supervisor.sh`, `multistart_seed_generator.jl`.

## 9. How to actually launch a CROSS campaign

Execution campaign, full supervisor (both waves):

```bash
export CAMPAIGN_FAMILIES="flexible_cm common_frechet cm_meanzc_cross origin_zc_cross unrestricted"
./full_aod_diag/d4_exact/run_full_campaign_supervisor.sh <manifest.json> <outroot> <maxtime_real> <threads> <hard_cap_s>
```

One family, one cell (what the gates above used):

```bash
julia --project=. -t 10 full_aod_diag/d4_exact/campaign_cm_family_runner.jl \
    cm_meanzc_cross upper <manifest.json> <outroot> <maxtime_real> <deltas_csv> <starts_csv>
```

The manifest must record `meanzc_K_mean` / `originzc_K_mean` **≥ 2** in its `config` block and carry
matching-length ν vectors in `shared_extra_coordinates` (the runner asserts this). At `K_pair = 1`
the cross grid degenerates to the diagonal family and the runner warns. To generate a wiring-test
manifest without running a full starts search:

```bash
julia --project=. -t 4 full_aod_diag/d4_exact/make_cross_campaign_test_manifest_2026-08-09.jl 2 100000 out.json
```

Reproducible seed campaign: use `production_five_family_cross_seed_specs(ctx)` in place of
`production_five_family_seed_specs(ctx)`. Its digest will differ from the diagonal wave's — by
construction, and asserted.

To enable Variant D (the focal `k*=σ−1` mean-row omission, the intended production spec at K≥2), set
`"profiled_level": 2` in the manifest's `config` block. It is **off** by default in the campaign
runner — see section 6a.

## 10. Reproducing the gates

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
cd /bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09

julia --project=. -t 4 full_aod_diag/d4_exact/smoke_cmzc_cross_d4_2026-08-09.jl
julia --project=. -t 4 full_aod_diag/d4_exact/verify_cmzc_cross_gradient_d4_2026-08-09.jl
julia --project=. -t 4 full_aod_diag/d4_exact/verify_cmzc_cross_cplus_ab_d4_2026-08-09.jl
julia --project=. -t 2 full_aod_diag/d4_exact/test_cmzc_cross_wiring_2026-08-09.jl
julia --project=. -t 8 full_aod_diag/d4_exact/bench_cmzc_cross_ncore_ext_2026-08-09.jl d4
```
