# Repository branch policy

This repository carries two protected, high-level modeling lineages. They must never be
merged into one another, and code must never move between them by guesswork.

## The two production trunks

### `production/fullA-exact`
All bilateral A_od objects are outer variables (the full D²-1-dimensional free-parameter
block, via the exact gravity-pivot/elimination map). Production entry point:
`full_aod_diag/d4_exact/c10_d20_production_driver.jl` (`run_profile_checkpointed`,
`run_polish_checkpointed`). Default price-cache/gradient backend is C+ (`:cplus`).

### `production/sequential-linearized`
Only the France/focal-country A_od terms are outer variables (D+1 or D+2 free parameters).
All other bilateral A_od terms are recovered by inverting factual trade shares under the
least-favorable distribution (LFD) / sequential linearized map (`recover_lfd`). Production
entry point: `sequential_gravity/run_profiled_production.jl`.

## Rules

1. **Never merge the two production branches into each other**, including "just to keep
   current." They model different economic objects and have different outer-loop
   dimensionality; a merge would silently conflate them.
2. **Shared fixes are reviewed and cherry-picked separately**, with lineage-specific tests,
   never assumed to transfer automatically in either direction.
3. **New branches use lineage-specific prefixes**: `feature/fullA-*`, `diag/fullA-*` for the
   full-A lineage; `feature/sequential-*`, `diag/sequential-*` for the sequential lineage.
   A branch name containing "sequential" is not sufficient evidence of which lineage it
   belongs to on its own — verify from the outer-variable construction in the code.
4. **Every handoff document must report**: exact commit, branch, worktree, dirty state, and
   whether the code described is source-only, merged, production-active, default-on, and
   tested.
5. **Generated results, checkpoints, and logs belong outside source control** unless
   deliberately curated as a small, reviewed artifact. Do not let large generated data
   bloat this repository.

See `repo_cleanup_2026-07-22/` (external to this repo, on the host) for the full
2026-07-22 consolidation census, branch classification, and preservation manifest.
