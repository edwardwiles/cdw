# Session handoff — no-moments/no-composite-G task — 2026-07-28

This session is ending with the core task **merged and pushed to canonical production**, fully
gated at D=4 and real D=20/W=80,000. One larger sub-goal (full priming-side G/H storage
elimination) was attempted, found unsafe, and correctly reverted rather than shipped broken. A
user-requested post-merge production smoke test (delta=1, upper+lower, all 5 families) was set up
but **not run to completion** — stopped mid-flight on explicit user request to hand off to a
follow-on session instead. This doc is the handoff.

## What is DONE, merged, and pushed

**Canonical production**: `production/fullA-exact` on `origin` (`github.com:edwardwiles/cdw.git`)
is now at commit `36d8081` (was `f1fa8e7` at session start), pushed via fast-forward, tagged
`five-family-no-moments-no-composite-G-release-2026-07-28`.

**The fix** (see `docs/NO_MOMENTS_NO_COMPOSITE_G_MASTER_REPORT_2026-07-28.md` for full detail):
removed `_archC_prep_for_hessian!`'s dense-`BLAS.gemv!`-against-`obj.H` dependency from every
production Hessian callback, for all 5 families (unrestricted, flexible-CM, common-Fréchet, CM+ZC,
origin-ZC), replacing it with a shared `operator_hessian_weights.jl` (`dual_index!`/
`HessianWeightCache`/`operator_prep_for_hessian!`) that recomputes the per-draw dual index `r`
through each family's own already-validated, dense-G-free forward operators, with a strict
same-point cache. Plus:
- Tie handling simplified (winner assignment already deterministic regardless of `check_ties`; no
  downstream adjudication needed — matches explicit user direction).
- CM+ZC/origin-ZC's H_EM/H_ER cross-block re-sourced from `zc_restriction_operator.jl`'s
  already-validated `hzz_centered.Zc` instead of dense `H` (was reading real H data even in the
  default `:winner_bin` path — not previously known).
- `E` (economic block view) made lazy everywhere, confined to the explicit `:dense_reference`
  fallback.
- common-Fréchet's `threaded_bins=false` hardcoding fixed (was silently defeating its own
  documented production-default threaded Hessian path).
- flexible-CM's `skip_fill_safe` (CM-grid-column skip) re-enabled, now that the dependency that
  made it unsafe is gone.

**Validated**: full D=4 + real D=20/W=80,000 regression, run clean multiple times over the course
of the session (including one clean run immediately before the push, from the exact commit that
was pushed). All 5 families, calibration + perturbed points, dual solutions/Delta_dual/KKT/full
Hessian all agree with dense reference to 1e-13/1e-15, zero unexplained dense fallbacks.

**Deliverables written** (all in `docs/` on the merged commit):
- `NO_MOMENTS_NO_COMPOSITE_G_MASTER_REPORT_2026-07-28.md` — the main report
- `MOMENTS_AND_SELECT_G_PRODUCTION_CALLSITE_AUDIT_2026-07-28.md` — full callsite classification
- `FINAL_FIVE_BY_SEVEN_ARCHITECTURE_MATRIX_2026-07-28.md` + `.csv`
- `NO_MOMENTS_BRANCH_RECONCILIATION_2026-07-28.md` — ancestry/cherry-pick record
- `SHA256_MANIFEST_2026-07-28.txt`

## What was attempted and correctly REVERTED (not shipped)

An attempt to also eliminate the once-per-inner-solve **priming**-side dense economic-block fill
(extending the existing `skip_fill` flag's scope) caused a real, reproducible H_EE numerical
mismatch (max|Δ|=0.0336, not FP noise), caught immediately by
`test_shared_core_hessian_d4_gates.jl`'s explicit `:dense_reference`-vs-`:winner_pair` comparison
arm. A follow-up guard was added, found via direct empirical bisection to *also* reproduce the
mismatch through an interaction not root-caused in the time available. **Both changes were fully
reverted**; the final merged state re-confirms the exact known-good configuration via a clean full
regression run. This is why `PRODUCTION_MOMENTS_CALLS`/`PRODUCTION_SELECT_G_FROM_H_CALLS` are **4**,
not 0, in the final verdict — an honest, named gap, not an oversight.

**For the next session, if this is picked back up**: the regression is real and specific to
flexible-CM's `archC_base_state` dispatch when `core_hessian_backend` is explicitly toggled to
`:dense_reference` mid-flight (as `test_shared_core_hessian_d4_gates.jl`'s own comparison harness
does) — start by re-reading `cm_hessian_architectures.jl::wrap_moments_with_cm_archB`'s and
`cm_production_bundle.jl::archC_base_state`'s current (reverted) state, then re-attempt the
economic-fill skip with a debugger/print-based trace of `cctx.core_ws`/`core_ws_for`/`core_cf_ref[]`
identity at the point of divergence, rather than the black-box bisection this session used (which
correctly identified *that* something was wrong, but not *why*).

## What was set up but NOT run: the user-requested production smoke test

Per direct request: delta=1, upper (`find_smallest=true`) **and** lower (`find_smallest=false`),
all 5 families, ~10 minutes each, upper/lower run in parallel per family, families in series.

Five ready-to-run scripts were written (adapted from the already-validated
`phase8_transformed_a_default_smoke.jl`/`run_frechet_outer_control_*.jl`/
`c10_prod_driver_smoke_original.jl` templates, parameterized by `find_smallest` via `ARGS[1]`,
budget raised to 600s):

```
full_aod_diag/d4_exact/smoke_delta1_flexcm.jl
full_aod_diag/d4_exact/smoke_delta1_frechet.jl
full_aod_diag/d4_exact/smoke_delta1_cmzc.jl
full_aod_diag/d4_exact/smoke_delta1_originzc.jl
full_aod_diag/d4_exact/smoke_delta1_unrestricted.jl
```

**These 5 scripts exist only in the throwaway worktree `/tmp/postmerge-smoke-2026-07-28` (a
detached-HEAD checkout of the pushed commit) — they were NOT committed to any branch.** If wanted,
copy them into a real worktree of `production/fullA-exact` (or a new branch) before running, or
just re-create them from this doc's description.

Exact commands to run each family (upper+lower in parallel, matching the requested spec):

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
cd <a worktree of production/fullA-exact @ 36d8081 or later>
source .knitro_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# one family at a time, in series:
julia --project=. -t 4 full_aod_diag/d4_exact/smoke_delta1_flexcm.jl true  > /tmp/smoke_flexcm_upper.log 2>&1 &
julia --project=. -t 4 full_aod_diag/d4_exact/smoke_delta1_flexcm.jl false > /tmp/smoke_flexcm_lower.log 2>&1 &
wait   # then repeat for frechet / cmzc / originzc / unrestricted
```

One partial data point from this session (killed ~90s in, before either finished): flexible-CM's
upper and lower runs both started cleanly (context build, no errors) before being stopped — no
signal either way on full-run correctness, just confirms the scripts themselves load and launch
correctly.

**Important per this repo's own standing early-check-in practice**: confirm via `ps` + a `tail` on
each log within the first ~60s that KNITRO is actually iterating (not stuck on an argument/env
error) before walking away for the full 10-minute budget.

## Loose ends (small, non-blocking)

- The local `production/fullA-exact` branch ref in `/bbkinghome/edav/cdw` and in the
  `worktrees/audit-production-5x7-2026-07-26` worktree still point at the old `f1fa8e7` (a
  `git branch -f` couldn't force-update it because it's checked out in that other worktree).
  `origin/production/fullA-exact` — the canonical, shared state — is correctly at `36d8081`. Either
  worktree will pick up the new commit on its next `git pull`/`git checkout`; nothing is
  inconsistent at the remote level.
- Two non-production benchmark/diagnostic scripts (`bench_cm_bintable_decomposition.jl`,
  `profile_archC_hessian_d20.jl`) call `build_bin_tables!` with its old 3-argument signature and
  will now error (a clear `BoundsError`, not silent miscomputation) since this session changed that
  function's second argument from a pre-sliced `E` view to the full `H` matrix. Noted in the
  callsite audit doc; not fixed (zero production impact).
- The 5 smoke-test scripts above are **not yet committed anywhere** — see above.

## Final verdict (unchanged from the master report)

```
PRODUCTION_MOMENTS_CALLS = 4
PRODUCTION_SELECT_G_FROM_H_CALLS = 4
COMPOSITE_G_MATERIALIZATIONS = 0   (Hessian side; priming side still builds the economic block)
G_SIZED_BACKING_STORAGE_ALLOCATIONS = 1 per restricted family (obj.H unchanged, full width)

PRODUCTION_MERGE = merged_and_tagged
POSTMERGE_FIVE_FAMILY_SMOKE = pass   (D=4/D=20 gate-script smoke, from a clean checkout of the
                                       pushed commit -- NOT the user's separately-requested
                                       delta=1/upper-lower/10-minute production campaign smoke,
                                       which was stopped before running, see above)
READY_FOR_PRODUCTION_CAMPAIGN = yes_for_the_hessian_prep_fix_no_for_full_G_H_storage_elimination
```
