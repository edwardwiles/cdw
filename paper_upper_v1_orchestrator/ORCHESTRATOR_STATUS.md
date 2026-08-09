# paper_upper_v1 orchestrator -- build status (2026-08-08 session)

## Done and validated

1. **`multistart_seed_generator.jl` extension** -- two new `FamilySeedSpec` kinds (`:unrestricted`,
   `:cm_only`) + `paper_five_family_seed_specs(ctx)`. Full test suite passes
   (`full_aod_diag/d4_exact/test_multistart_seed_generator.jl`), including real KNITRO evaluation
   at both new kinds.
2. **`objective_mode=:min_delta_fixed_gp`** added to `run_cm_upper_checkpointed` and
   `run_originzc_upper_checkpointed` (Stage B / Stage R0's fixed-gp Δ*-restoration NLP, mirroring
   `run_profile_checkpointed`'s existing pattern for Unrestricted). Confirmed via a REAL KNITRO
   smoke test at W=20,000/production draw_seed=20260719
   (`full_aod_diag/d4_exact/smoke_objective_mode_min_delta_fixed_gp.jl`):
   - CM-only: Δ* went from a companion start to 0.0048, verified feasible.
   - Origin-ZC K=1/K_pair=1: Δ* went from 0.83 to 0.0049, verified feasible.
   - The pre-existing `objective_mode=:min_gp` path is unchanged (confirmed real gp movement).
3. **Protocol manifest** `protocols/paper_upper_v1.toml` -- all 5 family definitions, deltas,
   algorithms, budgets, concurrency (2 starts x 5 families = 10 simultaneous Phase-I processes),
   projection graph, checkpoint policy.
4. **`family_start_chain.jl`** -- the Phase-I cell runner. One process per (family, start) pair;
   runs the full delta=0.1->0.5->1->2 chain; each delta cell runs Stage R0 (if needed) -> A -> B ->
   C per the protocol's exact budgets; writes an immutable, protocol_sha-checked
   `resume_bundle.jls` per delta (never overwrites a completed one); dispatches to the right real
   production driver per family (`run_cm_upper_checkpointed` / `run_originzc_upper_checkpointed` /
   `run_polish_checkpointed_unified`+`run_profile_checkpointed` for Unrestricted, mirroring the
   real, already-validated `stage_driver.jl` campaign template for that family's different call
   convention).
5. **`launch_wave.sh`** -- launches exactly 10 simultaneous `family_start_chain.jl` processes (2
   starts x 5 families) on 10 disjoint 10-core `taskset` ranges, `JULIA_NUM_THREADS=10`,
   `OPENBLAS_NUM_THREADS=1`; blocks until the whole wave completes before returning (no
   overlapping waves).
6. **`scan_seed_scales.jl`** -- empirical (A_scale, gp_scale) grid scan via `evaluate_attempt`
   directly, required before real seed generation (the generator's own header: these are not
   tuned defaults).
7. **`generate_seeds.jl`** -- real 10-seed generation driver, manifest-configured.
8. **`run_paper_upper_bounds.jl`** -- top-level command: seeds (if not already generated) -> Phase
   I waves in sequence. Idempotent (checks `seeds/seeds/manifest.jls` / each cell's own
   `resume_bundle.jls` before doing anything).

## Explicitly NOT done yet (do not treat as implemented)

- **Phase II (cross-family projection/inheritance)** and **Phase III (final polish)** -- not
  started. These only run after ALL Phase-I waves complete (up to ~60h pessimistic ceiling per
  the manifest), so they are not on the critical path for tonight's launch, but they are real,
  substantial remaining work (candidate-pool construction, projection-graph traversal, the
  deterministic winner-selection logic, nesting/monotonicity checks).
- **`--continue-cell` / `--continue-status` convergence-extension launcher** -- stubbed to a hard
  error in `run_paper_upper_bounds.jl`. Do not hand-roll a substitute that touches
  `phase1_discovery/` directly.
- **Dashboards / `PAPER_UPPER_V1_*.csv` / LaTeX table generation** -- not started.
- **`freeze_protocol_source.sh`** (records real julia/KNITRO/BLAS versions + `protocol_sha` into
  the manifest) -- not written; `[source].protocol_sha` in the manifest is still the literal
  placeholder `"PENDING_COMMIT"`, and `run_paper_upper_bounds.jl` hard-refuses to launch while
  that's true. Commit the two production-code extensions to the protocol branch and fill this in
  before the REAL launch (the toy dry run uses its own separate `toy-dryrun-local` stub value).
- **Full per-stage metadata recording** (protocol section 9's ~25-field list) -- `family_start_chain.jl`
  records a genuinely-populated SUBSET (status, wall, budget, best gp/Delta/vector, knitro_status,
  n_eval, n_grad) rather than fabricating fields the current driver return values don't expose
  (e.g. screen-rejection counts split by certificate kind). The full driver return value
  (`.trace`, `.screen_summary`) is available to any later pass that wants to mine it, just not
  copied field-by-field into the resume bundle yet.
- **Origin-ZC / CM+ZC seed loading for K_mean >= sigma-1** (the focal row-omission case) --
  `family_start_chain.jl`'s `load_seed_w0()` has an explicit `error(...)` placeholder for this
  path (real protocol families ORIGIN_ZC/CM_PLUS_ZC both use K_mean=3 >= kstar=2, so this path
  WILL be hit for real Phase-I launches) -- needs `scatter_nu_eff` + the family's own
  `ActiveMeanLayout` wired in before real K=3 seeds can be loaded. This is a concrete, scoped gap,
  not a design question -- the pieces it needs (`scatter_nu_eff`, `ActiveMeanLayout`,
  `originzc_profiled_nu_value`/`meanzc_profiled_nu_value`) all already exist and are already used
  correctly inside `multistart_seed_generator.jl`'s own `evaluate_family`; this loader just needs
  to call them the same way.

## Recommended next steps (in order)

1. Fix `load_seed_w0()`'s K_mean>=kstar gap (small, scoped -- reuse the exact pattern from
   `multistart_seed_generator.jl`'s `dense_nu_for_solve`).
2. Run the toy dry run (`protocols/paper_upper_v1_toy_dryrun.toml`) end to end: seed gen -> one
   `family_start_chain.jl` call per toy family -> confirm resume bundles + idempotent re-run.
3. Commit the protocol-branch extensions, run `freeze_protocol_source.sh` (write this script:
   record `git rev-parse HEAD`, `julia --version`, KNITRO version, data checksum) and fill in
   `protocols/paper_upper_v1.toml`'s `[source].protocol_sha`.
4. Run `scan_seed_scales.jl` at real W=100,000, freeze `[seeds].A_scale`/`gp_scale`.
5. Run `generate_seeds.jl` for real (10 starts, W=100,000) -- real KNITRO compute, budget hours,
   run in background.
6. Launch `run_paper_upper_bounds.jl` for Phase I once seeds are confirmed accepted.
7. Build Phase II/III while Phase I runs (it will not finish quickly).
