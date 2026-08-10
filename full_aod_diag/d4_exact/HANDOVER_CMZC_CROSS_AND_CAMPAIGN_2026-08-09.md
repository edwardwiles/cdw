# Handover: CM+ZC-CROSS, and making both CROSS families campaign-selectable

Paste everything below the line as your task prompt in a fresh Claude Code session.

---

## Your goal

Two things, in order:

1. **Build `CM+ZC-CROSS`** — the common-marginals analog of the already-built `OZC-CROSS` family.
2. **Make both CROSS families first-class options in the existing production campaign
   infrastructure**, selectable *instead of* the two current ZC families, using all the usual
   production machinery (checkpointed driver, multistart seeds, campaign runner, reproducibility
   digest).

End state: a real production campaign run can be launched with OZC-CROSS and CM+ZC-CROSS in place of
`origin_zc` / `cm_meanzc`, with no ad-hoc scripts.

## Read these first, in this order

1. `/bbkinghome/edav/gravity_robustness/CLAUDE.md` — project-wide rules. The ones that will actually
   bite you here: **no defaults on any scientific parameter** (sigma, W, K_mean, K_pair, draw_design,
   draw_seed, gravity exclusions, destination_sample — plain kwargs with no `= value`, so an omission
   is an `UndefKeywordError`); **never add a `:dense_reference` fallback**; **check on background jobs
   within 30-60 s** via `ps` + log tail; **push a package to Dropbox at the end of the session**
   (`rclone copy <dir> "dropbox:Gravity robustness/Analysis/Server Output/<new-subfolder>"`).
2. Auto-memory, especially `ozc-cross-kpair2-grid-build-2026-08-09` (the authoritative history of this
   whole line of work — read it in full) and `feedback-draw-seed-inert-under-pseudorandom`.
3. The OZC-CROSS implementation, all short and heavily commented — this is the pattern you are
   mirroring: `cm_originzc_cross_moments.jl`, `cm_originzc_cross_target_layout.jl`,
   `cm_originzc_cross_production.jl`, `cm_originzc_cross_cplus.jl`.

## Where things stand

Worktree `/bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09`, branch
`feature/ozc-cross-2026-08-09`, based on `origin/production/fullA-exact`@`4df5254`.
**Everything is uncommitted working-tree state.** Nothing has been pushed to any remote.

**OZC-CROSS is DONE and production-wired**: restriction family, Variant D (focal `k*=sigma-1`
mean-row omission), outer-gradient envelope extension, fixed-contribution fold, Backend C+ gradient,
and dispatch inside the real driver `run_originzc_upper_checkpointed` (`cm_originzc_checkpoint.jl`).

Verified (do not redo):
- D4 residuals ~1e-12..1e-15; fixed-contribution fold vs independent operator recompute at machine
  precision.
- Dense-G (literal matrix-multiplication Hessian) vs operator path agree to ~10 significant figures,
  at D4 and at D20/W=100k.
- C+ vs reference gradient: rel 1e-16..1e-18, K=1/1,2/2,3/3, with and without Variant D
  (`verify_cross_cplus_ab_d4_2026-08-09.jl`).
- Real production outer-loop runs through the checkpointed driver on its DEFAULT `:cplus` backend:
  K=2/2 W=100k -> 18 evals, kappa 3.2369%; resumed from checkpoint -> 24 evals, kappa 3.2577%
  (incumbent improved, layout tag and Variant D eta length both round-tripped);
  K=3/3 W=100k -> 5 evals, kappa 2.4253%. All budget-terminated (`-401`), i.e. wiring smoke tests,
  not converged science.

**Delta* context you need so you don't re-investigate it.** An extensive campaign established that
the cross family's Delta* at the calibration point (0.0667 at W=100k, K=3/3) is *correct arithmetic
and finite-sample*: it falls as W^-1.5 (0.0667 @100k -> 0.0210 @200k -> 0.0059 @500k, vs base 0.0095
-> 0.0044 -> 0.0013), moment discrepancies are unbiased and exactly Monte-Carlo-sized, and Sobol QMC
lowers it ~2x. The cross/base ratio settles near ~4.7. Separately, the cross block is genuinely
ill-conditioned (cond of centered data 1.15e6 vs base 2.9e4) with **approximate, not exact,**
redundancy — no restriction is duplicated. **Open item: the cross inner solve stops at
`inner_status=-100` at every W tested including 500k, while base reaches `0`.** Full write-up in
`dropbox:.../ozc_cross_EXPORT_2026-08-09/01_FINDINGS.md`.

## Part 1 — CM+ZC-CROSS

### The key structural fact

CM+ZC uses **`SharedByPowerLayout`** (one `nu_k` shared across origins), **hardcoded, not
configurable** — construction sites `cm_meanzc_production.jl:109` (`meanzc_zc_layout`), `:117`
(`hzz_zc_layout`), `cm_checkpoint.jl:1174` (aml base), `multistart_seed_generator.jl:416`. So:

- the cross target is simply `E[z_o^k1 z_p^k2] = nu_k1 * nu_k2` (simpler than origin-ZC's),
- `n_eta` stays `K_mean` (NOT `K_mean*D`),
- `target_index(layout,o,k) = k`, so `aml.dense_omit_idx == aml.kstar`.

Unlike origin-ZC there is **no `power_target_layout` config knob to extend** — you must create one.

### What to build

Near-verbatim mirrors of the OZC-CROSS work (6-7 items):
- new `SharedByPowerCrossLayout` + its `pair_targets` method — mirror
  `cm_originzc_cross_target_layout.jl:25,54`
- `build_cm_meanzc_augmented_obj` cross variant — `cm_meanzc_moments.jl:470`, pair sizing
  `:506,519,520`; **reuse `build_raw_cross_pair_matrix_levels` (`cm_originzc_cross_moments.jl:77`)
  verbatim**
- `build_lfix_base_cache_cm_meanzc` — `cm_meanzc_production.jl:451`
- `cm_meanzc_production_gradient` — `cm_meanzc_production.jl:488`
- the C+ trio — `cm_meanzc_cplus.jl:48,66,93`, mirroring `cm_originzc_cross_cplus.jl`

**Three pieces are genuinely NEW work — do not blind-copy:**

1. **The nu-gradient restructure.** `d_delta_dual_d_nu_vec` / `d_delta_dual_d_eta_nu_vec`
   (`cm_meanzc_moments.jl:589,610`) currently write **one component per level**
   (`out[k] = mean_m*total`, `:606`, using `d_pair_dnu = -2nu` at `:255`). The cross version must
   **accumulate** `-nu_k2` into slot `k1` and `-nu_k1` into slot `k2` across `klin` — i.e. adopt the
   shape of `d_delta_dual_d_eta_origin_cross_vec` (`cm_originzc_cross_production.jl:139`).
   Same restructure for `d_delta_dual_d_eta_active_and_nustar_shared` (`:635`), plus the ragged
   `pair_start0` (`:644`).
2. **Plumbing a layout choice through a family that has never had one** —
   `cm_meanzc_production.jl:109,117,169`, `cm_checkpoint.jl:1174`, `cm_meanzc_config.jl:46`
   (`MEANZC_NAMED_ARMS`) and `:58-77` (`meanzc_resolve_K`). Design call the code does not settle:
   new arm symbol vs. a new `meanzc_target_layout` field. **Note CM+ZC has no persisted layout field
   at all** (origin-ZC does, `cm_originzc_checkpoint.jl:126`), so you need a checkpoint schema bump
   or an encoding in `cm_extension` — see `cm_checkpoint.jl:291-294`, resume guard `:1078`,
   layout-version const `:366`.
3. **The `nu_star` cross-coupling under Variant D.** Under the shared layout the focal `nu_star`
   appears in cross-pair rows at **both** `(kstar,k2)` and `(k1,kstar)` for all `k`, so
   `d_delta_d_nu_star` collects strictly more terms than in the diagonal family. The
   accumulate-both-slots pattern at `cm_originzc_cross_production.jl:229-231` handles this correctly,
   but **FD-check it explicitly** — do not assume. Also read `cm_meanzc_moments.jl:510-518`
   carefully: for the shared layout, `n_mean` must be `mean_offset_from_aml(aml)[end]` (a ROW count),
   NOT `aml.n_eta_active` — they are completely different scales here (`n_eta = K_mean` vs
   `K_mean*D`).

**Reusable unchanged (verified by reading, do not touch):** `ZCRestrictionOperator`
(`zc_restriction_operator.jl:100,106,167,196,225` — keyed off `length(Zpairraw_all)`),
`refresh_zc_targets!` (`:158-168`, calls `pair_targets` so it dispatches to your new layout
automatically), `CMMeanZCOperatorState` (`cm_meanzc_lookup_kernels.jl:102,127,210`),
`cm_fixed_contribution_meanzc_layout` (`cm_meanzc_production.jl:436-438`, reads `aug.n_pair`),
`bin_zc_cross_hessian_fill!` (`winner_pair_cross_hessian.jl:773`),
`zc_gram_blas_candidates.jl:69,101`.

`meanzc_fixed_contribution` (`cm_meanzc_production.jl:386`, pair loop `:413-417`) needs the one
substitution `nuvec[k]^2*sum(lambda)` -> `dot(pair_targets(layout,nu,klin,D),lambda)`, exactly as
`cm_originzc_cross_production.jl:287` does.

### CM-block coupling — mostly a non-issue, with one real cost risk

Column layout is `[econ | mean | pair | CM-grid | gravity]` (`cm_meanzc_moments.jl:36-43`). The CM
columns are nu-independent and every CM offset derives from `aug.n_pair` / `n_pair(op)`, so a
`K_pair^2` pair block **auto-shifts them correctly**. The pair block is mathematically and index-wise
independent of the CM grid.

**But** `NCORE_ext = ncore_econ + n_mean + n_pair` (`cm_meanzc_production.jl:84`) feeds a dense
`(NCORE_ext + ncm)^2` packed Hessian buffer (`:131`). At D=20/K=3 the pair block is 9x bigger than
the diagonal family's. **No benchmark exists at that scale — measure memory and per-solve wall-clock
at D4 and at a small D20 W before committing to a full campaign.** This is the single biggest
feasibility risk in Part 1.

## Part 2 — Campaign integration

There are **two distinct things called "five family"** — do not conflate them:

1. **Execution campaign**: `campaign_cm_family_runner.jl` (whitelist `:60`, `<family>` CLI arg `:52`,
   family->driver map `:188-204`, K constants `:70-71`) + `campaign_unrestricted_runner.jl`, launched
   by `run_campaign_wave.sh:60-64` and `run_full_campaign_supervisor.sh:21`
   (`FAMILIES=(flexible_cm common_frechet cm_meanzc origin_zc unrestricted)`).
   `run_family_chain.sh` and `campaign_cell_io.jl` are family-string generic — no change needed.
2. **Reproducible multistart-seed campaign**: `multistart_seed_generator.jl`,
   `production_five_family_seed_specs` at `:180-188` — `U_MEAN3`, `COMMON_FRECHET`, `CM_MEAN3`,
   `ORIGIN_ZC_K3` (3/3), `CMZC_K3` (3/3, L=50). Selection via the `FamilySeedSpec` struct (`:119`),
   `kind` in `{:origin_zc,:cm_zc,:common_frechet}`.

**The two current ZC versions to offer alternatives to**: `ORIGIN_ZC_K3` (`:185`) and `CMZC_K3`
(`:186`); in the execution campaign, `origin_zc` and `cm_meanzc`.

### Concrete change list

- `campaign_cm_family_runner.jl:60` (whitelist), `:70-71`, `:86`, `:188-204` (family->driver/`extra`).
  `:181-182` (nu lift) needs no change — `n_eta` is unchanged.
- `multistart_seed_generator.jl:119-129,147-160,180-188,228,408-434,499-507,551-572`.
- `production_backend_manifest.jl:115,243` (family symbols).
- `run_campaign_wave.sh:60-64`, `run_full_campaign_supervisor.sh:21`.
- For CM+ZC-CROSS only: `cm_checkpoint.jl:1046,1174,1281-1286,1565,1568` and `family_tag` at
  `:1254,1305`. **`family_tag` MUST get a distinct symbol** — mirror
  `cm_originzc_checkpoint.jl:896`, where the OZC-CROSS cache key uses `:origin_zc_cross`. At
  identical `(K_mean,K_pair)` the diagonal and cross families produce genuinely different Delta*
  (0.0095 vs 0.0667), so an exact-cache key collision would be a **silent wrong answer**, not a perf
  nit.
- OZC-CROSS's own driver side is already done (`cm_originzc_checkpoint.jl:781-784,938-955`,
  `cm_originzc_config.jl:100,147`) — only the campaign/seed-generator layers need it.

### Reproducibility digest — a real collision risk to fix

`scientific_manifest/ScientificManifest.jl` and `configs/*.toml` are **not on this branch** (explicit
at `multistart_seed_generator.jl:238-244`). The stand-in is `compute_manifest_digest` (`:246`) over
`family_spec_descriptor` (`:228`), which encodes `kind`/`K_mean`/`K_pair`.

**Two specs differing only by diagonal-vs-cross layout would currently produce the SAME descriptor
unless `kind` differs.** Fix this deliberately — either give the CROSS families their own `kind`
symbols, or add the layout to `family_spec_descriptor`. Do not leave it implicit; verify by computing
digests for a diagonal and a cross spec and asserting they differ.

## Traps this line of work has already hit — do not repeat

1. **`nu0` seeding.** Use the theoretical population mean `Gamma(1 - mu*k)` (the mean of
   `z^k = U^(-mu*k)`). Do **not** copy `smoke_delta1_originzc.jl`, which seeds `mean(U[:,o]^k)` —
   wrong twice over (wrong transform: the feature is `z=U^(-mu)` not `U`; and a sample average of the
   very draws the restriction is imposed on, instead of the population mean).
2. **`draw_seed` is INERT under `draw_design=:pseudorandom`** — an inner `Random.seed!(seedU=888)`
   overrides it. Seed-sensitivity tests silently compare identical draws. Use `:sobol_randomized`,
   `:halton_scrambled`, or `:precomputed`. If a "different seeds" comparison returns identical
   numbers to many digits, that is the cause.
3. **`W = 8000` does not work at D20** — the calibration point is degenerate at that scale
   (pre-existing, memory `d20-realdata-w-sensitivity`). Use W >= 80,000. Iterate at D4 instead.
4. **`cm_gradient_backend = :cplus` is the production driver's DEFAULT.** If you add a family without
   a C+ gradient, implement it — do not make it error, or the family is unusable in its own default
   config. (That mistake was made and corrected here.)
5. **When A/B-ing two gradient backends, pass BOTH the same `h_mode` and the SAME shared
   `bandwidth_cache` Dict** — mismatched FD bandwidth produces a false ~1e-3 gap that looks like a
   real bug (memory `feedback-fd-bandwidth-mismatch-looks-like-a-bug`).
6. **Control against the unmodified base family before bug-hunting your own code.** This resolved
   four separate apparent "bugs" instantly across these sessions (memory
   `feedback-control-against-base-family-before-bug-hunting`).
7. **Julia world-age**: a runtime `include()` inside a function body, followed by calling the
   just-defined method in the same frame, throws `MethodError: ... too new`. Fixed in
   `cm_hessian_architectures.jl` with `Base.invokelatest`; the same pattern exists elsewhere.
8. **Julia top-level `for`-loop scoping**: rebinding an outer-scope variable from inside a top-level
   `for` in a script needs an explicit `global`. (Mutation via `push!`/`setindex!` does not.)
9. **Checkpoint loader**: `run_originzc_upper_checkpointed` writes `OriginZCCheckpointV10` — load
   with `load_cm_checkpoint_v10`, not `load_cm_checkpoint_v5`. Field is `checkpoint_reason`, not
   `stop_reason`.
10. **Timing budget at D20/W=100k, K=3/3**: base inner solve ~25-45 s, OZC-CROSS ~200-280 s. A 1800 s
    outer budget bought only 5 evals. Budget accordingly; CM+ZC-CROSS will be slower still (CM grid
    block + 9x pair block).

## Environment

```bash
export PATH="$HOME/.juliaup/bin:$PATH"     # NOT /opt/shared_sw — that Julia is broken here
export OPENBLAS_NUM_THREADS=1              # hard rule under Julia threading
cd /bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09
julia --project=. -t 8 full_aod_diag/d4_exact/<script>.jl
```

Standard D20 config used by every result quoted above:
```julia
d20_real_setup_design(W = W, delta = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    sigmaHat = 3.0, inner_lower_limit = -10.0)
# K_mean = K_pair = 3;  Variant D: originzc_profiled_level = 2  (= sigma-1)
```

## Suggested order of work

1. D4 first, always — it is seconds per solve and catches essentially every wiring bug.
2. `SharedByPowerCrossLayout` + `pair_targets`, then the augmented-obj builder; smoke-test that an
   inner solve converges and that recovered residuals against the new cross targets are ~0.
3. The nu-gradient restructure, FD-checked at D4 (the genuinely new math — do not skip the FD).
4. Variant D for the shared layout, FD-checking `d_delta_d_nu_star` specifically.
5. C+ gradient + an A/B gate against the reference gradient (mirror
   `verify_cross_cplus_ab_d4_2026-08-09.jl` exactly, including the shared `bandwidth_cache`).
6. Measure the `NCORE_ext` memory/time cost before going to D20.
7. Driver + config + checkpoint-schema plumbing; then a real checkpointed outer-loop smoke **and a
   resume test** (mirror `production_smoke_ozc_cross_2026-08-09.jl` and
   `production_resume_ozc_cross_2026-08-09.jl`).
8. Campaign + seed-generator integration; verify the manifest digests differ between diagonal and
   cross specs.
9. Update memory `ozc-cross-kpair2-grid-build-2026-08-09` (append, dated section) and push a Dropbox
   package to a NEW subfolder.

Existing reference packages on Dropbox: `ozc_cross_session_2026-08-09`,
`ozc_cross_variant_d_2026-08-09`, `ozc_cross_delta_diagnostics_2026-08-09`,
`ozc_cross_EXPORT_2026-08-09` (the fullest — code + findings + all logs).
