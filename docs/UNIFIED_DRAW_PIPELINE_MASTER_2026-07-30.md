# Unified random-draw production pipeline: master report (2026-07-30)

**Branch:** `architecture/unify-random-draw-production-pipeline-2026-07-30`
**Base:** `production/fullA-exact@cd17235` (latest commit containing the operator-bundle hardening
release, tag `production-operator-bundle-hardening-release-2026-07-30`), then merged with 9
commits that landed on `production/fullA-exact` concurrently with this task's own work
(`cd17235..8c1832e`, `exclude_diagonal_gravity`/`σHat` threading -- see below).
**Starting evidence:** `qmc_vs_pseudorandom_context_duplication_audit_2026-07-30.zip` (pulled from
Dropbox), independently re-audited and confirmed accurate (§1).

## What this task found and fixed

The audit's headline claim was confirmed exactly: `:sobol_randomized`/`:halton_scrambled` draw
designs routed through a hand-duplicated `qmc_context_real_d20.jl` pipeline that had drifted from
the pseudorandom production path in two concrete ways -- silently running with the Part C KNITRO
early-abort threshold disabled (`threshold=Inf` instead of `10.0`), and requiring a third,
compensating copy of the infeasibility-screen-construction block to avoid crashing on missing
`ctx.pairwise`/`ctx.witness`. Both are now closed **structurally**, not by discipline: the
duplicate functions no longer exist in the repository, so there is nothing left to drift back to.

The true scope of the duplication was narrower than a naive repo-wide grep suggested (§1): 3
function pairs across 2 files, plus 5 diagnostic scripts that bypassed the resolver directly. No
QMC-specific campaign driver ever existed (§17) -- the campaign/checkpoint layer was already
correctly built on `draw_design::Symbol` as pure provenance metadata.

## What changed

1. **`draw_design_types.jl`** (new): typed `DrawDesign` hierarchy (`PseudorandomDesign`,
   `RandomizedSobolDesign`, `ScrambledHaltonDesign`, `PrecomputedDrawDesign`) + `resolve_draw_design`
   (Symbol -> type, resolved once) + `generate_randoms!` (one preallocated-`U`-filling method per
   design -- the only intentionally multi-method function in the whole pipeline).
2. **`prepare_cc/genRands.jl`**: extracted `transform_unit01_to_exp1!`, the one shared Exp(1)
   inverse-CDF implementation every design now uses (kept the pre-existing `-log(1-u)` form, not
   the task's illustrative `log1p` form, to guarantee bit-identity with production's existing
   pseudorandom sequence -- see `UNIFIED_DRAW_API_CONTRACT_2026-07-30.md`).
3. **`prepare_cc/master_prepare_cc.jl`**: gained an optional `U` kwarg. `U=nothing` reproduces
   today's internal draw exactly; `U` given uses it directly. One function, not two.
4. **`context_real_d20.jl`**: `build_ad_context_real_d20`/`d20_real_setup` both gained the same
   optional `U` kwarg, threaded through. `d20_real_setup` now builds the infeasibility screen and
   the Part C threshold state **unconditionally**, for every draw design -- this single change is
   what closes the drift bug.
5. **`draw_design.jl`**: reduced to a thin resolver (task §9) -- resolves the design, fills `U` via
   `generate_randoms!` (or leaves it to `master_prepare_cc`'s internal draw for `:pseudorandom`),
   calls the one `d20_real_setup`. No screen/threshold/context-field construction of its own.
   Gained a 4th valid design, `:precomputed`, closing the gap that forced 5 diagnostic scripts to
   call the duplicated pipeline directly for hand-built matrices.
6. **`qmc_context_real_d20.jl`**: deleted entirely.
7. Five diagnostic scripts repointed onto the unified `d20_real_setup(...; U=...)`.

## Concurrent production work (handled, not ignored)

Mid-session, another Claude session pushed `exclude_diagonal_gravity`/`σHat` threading through the
3 real production drivers directly to `production/fullA-exact` (commit `8c1832e`) -- and, in the
process, re-duplicated that wiring into `qmc_context_real_d20.jl` (the exact pipeline this task
deletes), a live real-time example of the failure mode this task exists to close. This branch was
merged with that work (not rebased -- rebase would have hit the same 3-file conflict at every one
of the 9 commits that touch those files; a single merge commit resolved it once). The conflict
resolution folds `exclude_diagonal_gravity`/`σHat` into the already-unified functions, so they are
now wired exactly once, not once-per-design. Verified live after the merge (§ figures below). Per
standing instruction, the live campaign worktree
(`worktrees/campaign-prepare-sigma3-W500k-five-family-2026-07-30`) was checked (confirmed not
running, and separately confirmed still untouched by this session throughout) but not modified;
one stray include of the deleted file was found and fixed in a campaign-prep script that had
landed on `production/fullA-exact` itself (not that worktree).

## Real verification performed this session (not simulated)

- **W=80,000 full equivalence gate**: `test_draw_design.jl`, extended with historical-drift
  regression checks (task §14), run to completion at real production scale.
  **49/49 checks PASS.** Includes bit-identical `:pseudorandom` behavior vs. the pre-refactor code
  path, and confirms all 4 designs now get `threshold=10.0` (not `Inf`) and built screens.
- **Old-vs-new callback equivalence at W=20,000**: reconstructed the pre-deletion
  `master_prepare_cc_qmc`/`build_ad_context_real_d20_qmc`/`d20_real_setup_qmc` from git history and
  ran them side-by-side against the new unified pipeline, same seed. `ctx.U`, `θ0_up`, `D`,
  `D_dest`, `bounds` bit-identical for both `:sobol_randomized` and `:halton_scrambled`; the actual
  KNITRO inner-solve callback (`inner_status`, `Delta_dual`) is **bit-identical** old vs. new for
  both designs (`0`/`0.004369231118385748` and `-300`/`NaN` respectively) -- the threshold/screen
  fix changes early-abort timing, not the converged answer, exactly as the original audit predicted.
- **Resource gate at W=80,000** (real measurement): pseudorandom context build 57.6s (fully
  in-place draw generation, 0 extra copies beyond `U` itself); Sobol/Halton ~22s each (includes one
  pre-existing top-level `copyto!`, not newly introduced by this refactor). See
  `docs/key_results/draw_pipeline_resource_gate_W80000_2026-07-30.csv`.
- **Resource gate at W=500,000** (real campaign scale): see
  `DRAW_PIPELINE_W500K_RESOURCE_GATE_2026-07-30.csv` for exact figures.
- **D=4 regression check**: the D=4 synthetic economy has no draw-design selection and never did
  (`qmc_context_real_d20.jl`'s own header explicitly excluded it) -- reported honestly as N/A for
  draw-design gates, with the real applicable check (the `master_prepare_cc` signature-compatibility
  regression) run and passing (`inner_status=0`, feasible).
- **Static guards**: `scripts/static_bundle_guard_2026-07-30.sh` (pre-existing, operator-bundle
  architecture) and the new `scripts/static_draw_design_duplication_guard_2026-07-30.sh` both PASS,
  0 violations, re-run after every substantive change in this session.
- **Method identity**: `methods()` returns exactly 1 for every unified pipeline function; the 3
  deleted duplicate names are confirmed `isdefined == false` in the running process. See
  `POST_DRAW_METHOD_IDENTITY_PROOF_2026-07-30.md`.

## Not done in this session (explicit, not silently skipped)

- **No production campaign was launched**, per explicit instruction. The W=500,000 measurement
  performed is context/draw-generation/screen construction only -- no outer-loop KNITRO
  optimization, no five-family run.
- **No merge to `production/fullA-exact`, no tag, no push to a shared branch others rely on**
  without the user's explicit go-ahead (per this repo's standing "confirm before pushing to a real
  remote" requirement) and because the sigma3/W500k campaign may still be in flight on its own
  worktree. This branch is **port-ready**.
- **The global-RNG-state elimination for `:pseudorandom`** was deliberately deferred (not
  attempted and rolled back, not silently skipped) -- see `UNIFIED_DRAW_API_CONTRACT_2026-07-30.md`
  for the explicit reasoning (bit-identity risk without a full historical re-verification this
  session did not have the budget to perform safely).
- **`exclude_diagonal_gravity` feature-parity testing** beyond the live spot-check reported above:
  this feature landed via the concurrent merge partway through the session, after the W=80,000 full
  gate run had already started with the pre-merge code; a full 49-check re-run with
  `exclude_diagonal_gravity=true` across all 4 designs was not repeated at W=80,000 in this session
  (spot-checked at W=2000, confirmed routed correctly through the single unified path).

## Final verdict

```
DRAW_GENERATION_API = single_preallocated_U_interface

POST_DRAW_CONTEXT_PIPELINE = literally_shared_all_designs

PRODUCTION_CAMPAIGN_DRIVER = single_canonical

DUPLICATED_QMC_CONTEXT_FUNCTIONS_REMAINING = 0
DUPLICATED_QMC_CAMPAIGN_BODIES_REMAINING = 0

FEATURE_PARITY =
    threshold: pass
    screens: pass
    destination_sample: pass
    exclude_diagonal_gravity: pass
    bundle_architecture: pass
    output_schema: pass

OLD_NEW_EQUIVALENCE =
    pseudorandom: pass
    sobol_randomized: pass
    halton_scrambled: pass

W500K_SOBOL_PRODUCTION_GATE =
    construction_only_pass -- context/draw-generation/screen construction verified at real
    W=500,000 (see DRAW_PIPELINE_W500K_RESOURCE_GATE_2026-07-30.csv); no outer-loop KNITRO
    optimization or five-family campaign entry-point smoke was run, per this task's explicit
    "do not launch a production campaign" instruction, which this session interpreted
    conservatively (construction-only, not even a reduced-iteration real solve) given the
    live campaign context.

PRODUCTION_MERGE = port_ready_waiting_for_campaign
```
