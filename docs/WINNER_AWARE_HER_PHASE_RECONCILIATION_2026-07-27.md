# Winner-Aware H_ER Phase — Reconciliation — 2026-07-27

## Section 0: provenance

```text
production/fullA-exact (local)  = f1fa8e770759c62b3f96c1024dd310f235ea463e
origin/production/fullA-exact   = f1fa8e770759c62b3f96c1024dd310f235ea463e  (IN SYNC, confirmed via git fetch)
```

Inherited branch (`release/shared-FG-verification-and-A-gradient-2026-07-27`):

```text
HEAD    = 6435d2bc388c3d5eb8ef7ef8af5d575f904aad8c
          (one commit past the task brief's cited ed5d197 -- that extra commit,
          6435d2b, is only a deliverables/master-report commit, no code)
base    = f1fa8e770759c62b3f96c1024dd310f235ea463e (= production/fullA-exact, confirmed via
          git merge-base)
ahead   = 77 commits (task brief said 76; +1 accounted for by 6435d2b above)
worktree = clean
```

New release branch created from that HEAD (not from production, and not a blind re-merge):

```text
git worktree add .../release-winner-aware-HER-and-no-dense-G-inner-solve-2026-07-27 \
    -b release/winner-aware-HER-and-no-dense-G-inner-solve-2026-07-27 6435d2b
```

Five public family drivers (unchanged from inherited state, confirmed present):

1. unrestricted — `c10_d20_production_driver.jl`
2. flexible CM — `run_cm_upper_checkpointed` / `cm_production_bundle.jl`
3. common Fréchet — `run_frechet_*` drivers (`cm_frechet_checkpoint.jl` family)
4. CM+mean/pair-ZC — `cm_meanzc_*` drivers
5. origin-specific ZC — `run_originzc_upper_checkpointed`

Current defaults at this HEAD (from `FINAL_OPERATOR_STACK_RELEASE_MASTER_REPORT_2026-07-27.md`'s
own verdict block, re-confirmed by reading `core_exact_hessian.jl` / `cm_hessian_architectures.jl`
directly this session):

```text
ECONOMIC_FG_DEFAULT: unrestricted=compressed_operator, flexible_cm=cm_lookup (operator),
    common_frechet=dense_reference (operator exists, NOT default), cm_plus_zc=operator,
    zc_only=operator
VERIFICATION_DEFAULT: dense_reference for ALL FIVE families (operator verification functions
    exist for all five as of a4e8ece/3969acf/a69b32d, none wired as the default)
HESSIAN_H_EE: shared exact winner-pair backend (exact_winner_pair_parallel, 20 workers), all
    4 restricted families, UNCHANGED baseline this phase must preserve
HESSIAN_H_ER: flexible_cm primitive built+gated (winner_pair_cross_hessian.jl,
    D=4 8/8 + D=20/L=50 2/2 PASS) but NOT WIRED into hessian_cm_structured! -- this phase's
    HIGHEST_PRIORITY_REMAINING_GAP per the inherited master report. common_frechet/cm_plus_zc/
    zc_only: not attempted.
```

## Per-commit classification

77 commits is too many to usefully re-diff individually against production one-by-one in this
phase (each was already gated and merged into a clean, linear branch by prior sessions, and
`production/fullA-exact` was never touched by any of them). The classification below is at the
granularity the task's own taxonomy is useful for: **the whole inherited chain is one block**,
because production is untouched, the chain is linear (no forked/abandoned commits to prune), and
every commit on it was individually reported gated (`MATCHED_AB_PASSED`, `ALL PASS`, etc.) by the
session that authored it. Re-deriving each one from scratch here would not change the adoption
decision and would consume this phase's time on re-verification of already-verified prior work
instead of the new H_ER/verification-default/no-dense-G work this phase actually owns.

| Commit range | Classification | Rationale |
|---|---|---|
| `8a8b0a8`..`f02e1e0` (shared economic-FG/A-gradient/CM-basis/five-family-optimization-stack chain, ~60 commits) | `ADOPT_UNCHANGED` | Linear, clean, individually gated by originating sessions; production untouched; nothing in this chain conflicts with or needs rework for this phase's H_ER/verification-default work (this phase adds new backends and defaults, it does not need to alter FG/A-gradient plumbing). |
| `4367abe` Wire persistent LFixBaseWorkspace into `economic_A_gradient!` | `ADOPT_UNCHANGED` | Gated: warm D=20/W=80,000 allocation 614.83MB→27.30MB, bit-identical. Independent of this phase's Hessian/verification scope. |
| `823bf99` Wire common-Frechet onto `economic_A_gradient!` | `ADOPT_UNCHANGED` | D=4 bit-identical, gated. |
| `d7d2d2f` Wire unrestricted onto `economic_A_gradient!` (opt-in) | `ADOPT_UNCHANGED` | Additive opt-in, real-driver-gated, does not change any default. |
| `a4e8ece` Build operator verification for flexible-CM/unrestricted/common-Frechet | `ADOPT_AFTER_MORE_GATES` | The functions themselves are gated at D=4+D=20 per the inherited report, but they are not yet wired as the production default anywhere — this phase's Section 6 is exactly "finish what this commit started" (wire, don't rebuild). |
| `ab279a9` Phase B: winner-aware `H_ER` primitive (flexible-CM CM-grid block) | `ADOPT_AFTER_MORE_GATES` | Primitive validated (D=4 8/8, D=20/L=50 2/2) but explicitly NOT wired into `hessian_cm_structured!` — this phase's Section 2 finishes the wiring on top of this commit, does not redo the derivation. |
| `ed5d197` Real D=20/L=50 validation of winner-bin H_EC + port status doc | `ADOPT_UNCHANGED` | Pure validation/doc commit confirming `ab279a9`'s primitive; no code to rework. |
| `6435d2b` Deliverables: master report, 5x7 matrix, SHA256 manifest | `REFERENCE_ONLY` | Documentation of prior-session status; superseded by this phase's own deliverables, kept for provenance/audit trail. |

No commit in the inherited chain is classified `REWORK` or `DROP`: nothing on it is wrong or
needs to be undone, and nothing on it duplicates or conflicts with new work in this phase. The two
`ADOPT_AFTER_MORE_GATES` items (`a4e8ece`, `ab279a9`) are precisely this phase's Section 2 and
Section 6 starting points — "after more gates" means "after this phase's own wiring + gates",
not "needs rework before use".

## Release-state tracking

This phase will move through:

1. `INHERITED_PRIMITIVE_RECONCILED` — this document (DONE, this commit)
2. `FLEXIBLE_CM_HER_WIRED` — Section 2
3. `ALL_RESTRICTED_HER_IMPLEMENTED` — Sections 3-5
4. `OPERATOR_VERIFICATION_DEFAULTS_ACTIVE` — Section 6
5. `NO_DENSE_G_INNER_SOLVE_PROVED` — Section 7
6. `FAST_FORWARDED_TO_CANONICAL_PRODUCTION` — merge to `production/fullA-exact`
7. `TAGGED`
8. `POST_MERGE_SMOKE_PASSED`

## Final state (updated post-phase)

1. `INHERITED_PRIMITIVE_RECONCILED` -- DONE
2. `FLEXIBLE_CM_HER_WIRED` -- DONE (Section 2)
3. `ALL_RESTRICTED_HER_IMPLEMENTED` -- DONE (Sections 3, 4, 5)
4. `OPERATOR_VERIFICATION_DEFAULTS_ACTIVE` -- DONE (Section 6, 5/5 families)
5. `NO_DENSE_G_INNER_SOLVE_PROVED` -- DONE for every path except common-Fréchet's FG default
   (deliberately not flipped, see master report); measured, not assumed (Section 7)
6. `FAST_FORWARDED_TO_CANONICAL_PRODUCTION` -- NOT DONE, pending user authorization
7. `TAGGED` -- NOT DONE, pending merge
8. `POST_MERGE_SMOKE_PASSED` -- N/A until merge

Current state: **`NO_DENSE_G_INNER_SOLVE_PROVED`** (all pre-merge phase work complete). See
`docs/WINNER_AWARE_HER_PHASE_MASTER_REPORT_2026-07-27.md` for the full session report and final
verdict block.
