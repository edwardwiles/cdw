# Source snapshot — profiled restricted-family inner end-to-end (2026-08-01)

## Branch / worktree identity

- New branch: `architecture/profiled-restricted-inner-endtoend-2026-08-01`
- New worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-restricted-inner-endtoend-2026-08-01`
- Created via `git worktree add <path> -b architecture/profiled-restricted-inner-endtoend-2026-08-01 f1f969b`
  from the `cdw` repo (`/bbkinghome/edav/cdw/.git`), confirmed at creation time by `git rev-parse HEAD`
  → `f1f969b5fe26661b25d7fd8abd7210b662222611`.
- Source worktree (`architecture-profiled-economic-block-remaining-crossblocks-2026-08-01`) was **not
  modified**: `git status --porcelain=v1 -uall` there is empty (clean, no untracked files) both before
  and after this session's worktree-add.

## Source SHA and provenance

- Recorded task-prompt HEAD: `f1f969b`
- Confirmed identical to `HEAD` of `architecture/profiled-economic-block-remaining-crossblocks-2026-08-01`
  in the source worktree at inspection time.
- `git rev-parse --git-common-dir` from the source worktree → `/bbkinghome/edav/cdw/.git` — this is the
  canonical `cdw` repo (per standing repo memory: use `cdw`, not `habibiscoding/Trade-Model-Robustness`).

## git status / untracked files (source worktree, at inspection time)

```text
$ git status --porcelain=v1 -uall
<empty>
```

Zero untracked files, zero modified/staged files. The handoff zip
`profiled_remaining_crossblocks_hef_hez_bugfix_2026-08-01.zip` was retrieved from
`dropbox:Gravity robustness/Analysis/Server Output/` (per this repo's standing note that task-referenced
zips live there, not on the local filesystem) and cross-checked as the doc/results package pushed off
this same branch — not a separate code source.

## Commits `6f6a878..f1f969b` (the range named in the task prompt)

Only **one** commit exists in that range (`6f6a878` is the immediate parent of `f1f969b`):

```text
f1f969b Fix H_EF/H_EZ pi_vec-vs-Lam_homog formula bug (same pattern as H_EC's fix)
```

Full stat:

```text
 docs/PROFILED_REMAINING_CROSSBLOCKS_MASTER_2026-08-01.md      | 262 +++++++++++++++++++++
 full_aod_diag/d4_exact/test_profiled_france_row_d4_2026-08-01.jl      |  47 ++-
 full_aod_diag/d4_exact/test_profiled_hef_correction_d4_2026-08-01.jl  |   9 +-
 full_aod_diag/d4_exact/test_profiled_hez_correction_d4_2026-08-01.jl  |   5 +-
 full_aod_diag/d4_exact/threaded_cross_hessian.jl                      |  20 +-
 full_aod_diag/d4_exact/winner_pair_cross_hessian.jl                   |  76 ++++--
 6 files changed, 387 insertions(+), 32 deletions(-)
```

Full unified diff saved to scratchpad (`source_tracked_patch_6f6a878_f1f969b.diff`, 667 lines) for
reference; not copied into this branch's history since it is fully reconstructible via
`git diff 6f6a878..f1f969b` in `cdw`.

The broader ancestry (`6f6a878` itself, and everything back through the fork point of this whole
sub-lineage) is documented in the **source branch's own** master doc, carried into this worktree
unmodified at `PROFILED_REMAINING_CROSSBLOCKS_MASTER_2026-08-01.md`. Key excerpt of that lineage
(`git log --oneline` on `f1f969b`, most-recent-first): `f1f969b → 6f6a878 → 69913ff → 8f65010 →
364054d → 50d181b → 79cc0bd → 1afe1be → 56ba93e → 81b6225 → 25b20f0 → 774a223 → 6741ec1 → b2b418 →
10b5940 → c49f843 → a06fcc4 → f439109 → 1aeec43 → f483cae → …`. `6f6a878`'s own commit message
root-causes the `nStatus=-400` failure that motivated this entire sub-lineage: `winner_pair_cross_hessian_cm_block!`'s
`use_profiled_correction=true` path used the structured-formulation `pi_vec[j]` instead of the correct
homogeneous-formulation `Lam_homog[j] = kappa0[j]*Pmat[o,slot]`, plus a missing `denom_cf_scaled` term
in the France row.

## Current state per the source branch's own accounting (not re-derived here — see Phase 1 reproduction below for independent confirmation)

Per `PROFILED_REMAINING_CROSSBLOCKS_MASTER_2026-08-01.md`'s own final verdict block:

- `H_EC`: corrected + independently verified (fixed by `6f6a878`, unchanged since).
- `H_EF` (`winner_pair_cross_hessian_colsum!`/`_esum!`): `Lam_homog` fix + France-constant fix, both
  brute-force verified. **ForwardDiff-of-real-objective: not run. D4 KNITRO: not run** (no reduced-layout
  orchestrator exists yet for common Fréchet).
- `H_EZ` (`winner_pair_cross_hessian_zc_block!`/`_threaded!`): same status as H_EF, serial+threaded both
  fixed+brute-force-verified, serial/threaded bit-identical. **ForwardDiff: not run. D4 KNITRO (ZC-only,
  CM+ZC): not run** (no reduced-layout gather branch exists for either).
- Flexible CM: the **only** family with an existing reduced-layout Hessian orchestrator
  (`hessian_cm_structured!`'s profiled branch) and a reported real D4 KNITRO solve (4-6 iterations,
  `nStatus=0`), built by a prior session, unchanged by `6f6a878`/`f1f969b`.
- `PRODUCTION_FG`: fails for common Fréchet / ZC-only / CM+ZC (`:cm_lookup` operator path not wired for
  any of them); flexible CM itself is only wired through `:dense_reference`, not `:cm_lookup` either.
- `test_threaded_cross_hessian_d4.jl`'s common-Fréchet section: `FieldError: type OperatorPsiBundle has
  no field moments!` — confirmed by the source branch as **pre-existing at the fork point** (reproduced
  against the unmodified `architecture/profiled-economic-block-all-families-complete-2026-08-01`
  worktree), not introduced by any commit in this lineage. This session's job (per the task prompt) is
  to diagnose it as an obsolete test harness and migrate it to the operator-only architecture, not to
  restore `moments!`/`H`/`G`.

## Scope note (added mid-session, from the task owner)

A separate Claude session now owns the profiled restricted-family **outer-gradient / outer A/B layer**
(A/gp outer-gradient wrappers, `ProfiledLFixCache` refactoring, incremental winner-update gradient code,
gravity-pivot gradient chain rule, outer A/B harnesses, outer-search performance comparisons). This
session's scope is strictly the **inner pipeline**: independent D4 truth gates, genuine reduced family
layouts, real operator FG, real Hessian callback orchestration, restriction-preparation freshness, D4
KNITRO + recover-then-resolve equivalence, full-path no-overhead, small-D20 inner correctness/solve
gates. This session additionally owes that other workstream a stable typed accessor surface
(`profiled_economic_layout`, `economic_dual_range`, `restriction_dual_ranges`, `profiled_anchor_spec`,
`profiled_outer_coordinate_layout` or equivalents) over the reduced-layout objects built in Phases 6-7,
documented with its contract, live checksums, and exact integration commit in the final handoff.

## Next step

Reproduce every currently-passing gate listed in the task prompt's §1 before making any edits. That run
is in progress in the background (`run_repro_gates.sh` / `key_results_repro_2026-08-01/`); results will
be appended to this document once complete.
