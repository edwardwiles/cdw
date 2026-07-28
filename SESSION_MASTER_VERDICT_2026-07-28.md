# Session master verdict — parallel legacy-CC-H removal + Hessian upper-only cleanup — 2026-07-28

Branch: `cleanup/remove-legacy-CC-H-G-storage-2026-07-28`. Base:
`campaign/five-family-bounds-2026-07-28@93f26df7`. Two independent tasks worked on this one
isolated branch, each committed separately per instructions:

1. **Legacy CC `H=[K|ones|G]` storage removal** — audited and designed this session;
   **not implemented** (honest gap, see below).
2. **Hessian upper-only cleanup** (mirror/symmetrize redundancy) — audited **and implemented**
   for 3 of 5 families this session, committed as `c350280`.

## Campaign isolation

No file in `worktrees/campaign-five-family-bounds-2026-07-28` was read, written, or executed by
this session. See `PARALLEL_LEGACY_CC_H_REMOVAL_PROVENANCE_2026-07-28.md` for the full isolation
record (base-SHA verification, `/proc/*/cwd` scan, worktree topology).

## Task 1: legacy CC H removal — final verdict

```
OPERATOR_BUNDLE_TYPE =
    unrestricted:PsiObjectiveBundleImplicit (unchanged)
    flexible_cm:PsiObjectiveBundleImplicit (unchanged)
    common_frechet:PsiObjectiveBundleImplicit (unchanged)
    cm_plus_zc:PsiObjectiveBundleImplicit (unchanged)
    zc_only:PsiObjectiveBundleImplicit (unchanged)

LEGACY_CC_H_MATRIX_ALLOCATIONS = 5 live + 2 throwaway (flexible-CM, common-Fréchet; new finding)
G_SIZED_BACKING_STORAGE_ALLOCATIONS = same
CC_PAYOFF_K_VECTOR_ALLOCATIONS = 0   (K is a required scalar-fill moment, already at target --
                                       see LEGACY_CC_H_G_CONSTRUCTOR_AND_CALLSITE_AUDIT §3)
ONES_VECTOR_ALLOCATIONS = 1 per PsiObjectiveBundleImplicit construction (struct default), not 0

PRODUCTION_MOMENTS_CALLS = 4   (unchanged from inherited state)
PRODUCTION_SELECT_G_FROM_H_CALLS = 4   (unchanged, paired 1:1)
COMPOSITE_G_MATERIALIZATIONS = 0 on the Hessian-callback side (inherited, re-verified);
                                >0 on the priming side (the named blocker)

OPERATOR_BUNDLES_WITH_H_FIELD = all 5 (unrestricted, flexible_cm, common_frechet, cm_plus_zc, zc_only)
OPERATOR_BUNDLES_WITH_K_FIELD = all 5
OPERATOR_BUNDLES_WITH_MOMENTS_FIELD = all 5

FIVE_FAMILY_ARCHITECTURE = incomplete_economic_moment_construction_flexcm_frechet_cmzc_originzc

PRODUCTION_MERGE = not_ready   (this task's own storage-removal work was not implemented, only
                                 audited/designed; nothing to merge for this specific goal beyond
                                 documentation)

HIGHEST_PRIORITY_REMAINING_GAP = empirically root-cause the H_EE priming-fill regression a prior
    session hit and reverted (see OPERATOR_DENSE_BUNDLE_TYPE_SEPARATION_2026-07-28.md §1 for this
    session's sharper, not-yet-confirmed lead), OR implement the designed genuinely-separate-types
    architecture directly (bypassing the shared-closure/runtime-flag pattern both prior attempts
    used, which is the likely root cause of the regression class itself) rather than attempting a
    third variant of the same flag-based approach.
```

Deliverables produced for this task: `PARALLEL_LEGACY_CC_H_REMOVAL_PROVENANCE_2026-07-28.md`,
`LEGACY_CC_H_G_CONSTRUCTOR_AND_CALLSITE_AUDIT_2026-07-28.md`,
`OPERATOR_DENSE_BUNDLE_TYPE_SEPARATION_2026-07-28.md`,
`FIVE_FAMILY_NO_LEGACY_H_STORAGE_GATE_2026-07-28.md`,
`FINAL_FIVE_BY_SEVEN_NO_LEGACY_STORAGE_MATRIX_2026-07-28.md` + `.csv`, this document. **Not
produced** (no corresponding measurement was performed, so no doc was fabricated):
`FIVE_FAMILY_OPERATOR_DUAL_INDEX_HESSIAN_WEIGHT_GATE` and
`FIVE_FAMILY_OPERATOR_VERIFICATION_NO_LEGACY_STORAGE_GATE` as standalone docs — both gates were
re-exercised indirectly via `test_shared_core_hessian_d4_gates.jl` (40/40 PASS, unchanged from
before this session's Hessian-upper-only edits) as part of verifying the addendum task didn't
regress anything, confirming the *inherited* operator dual-index/Hessian-weight/verification
machinery from the prior merged session is still intact — but no NEW dual-index/verification work
was done this session, so no new gate report was written. `OPERATOR_BUNDLE_MEMORY_BEFORE_AFTER`
was not produced — no code change was made to legacy storage this session, so there is no "after"
state to measure; fabricating numbers was avoided.

## Task 2: Hessian upper-only cleanup — final verdict

```
PRODUCTION_HESSIAN_ASSEMBLY = remaining_symmetrization_H_EE_H_CC_and_all_common_frechet_blocks
PRODUCTION_HESSIAN_LOWER_ENTRIES_WRITTEN = >0 (origin-ZC's one dead site fixed and now 0 there;
                                                CM/CM+ZC/common-Fréchet's mirror WRITES
                                                intentionally retained -- see release doc rationale;
                                                only the unnecessary READS/averaging were removed)
PRODUCTION_HESSIAN_SYMMETRIZATION_PASSES = >0 (H_EC's averaging removed for flexible-CM/CM+ZC;
                                                H_EE/H_CC/all 4 common-Fréchet sites retained)
PRODUCTION_HESSIAN_LOWER_TRIANGLE_READS = >0 (same scope as above)
PACKED_UPPER_SENTINEL_TEST = pass_flexible_cm_and_cmzc_H_EC_region_scoped
                              (5/5 checks pass, real KNITRO D=4, including a negative control
                               proving the sentinel isn't vacuous -- not pass_all_families:
                               common-Fréchet not attempted, H_EE/H_CC genuinely still need the
                               lower triangle by design)

PRODUCTION_MERGE = port_ready_waiting_for_campaign
```

Deliverables: `PRODUCTION_HESSIAN_UPPER_ONLY_AUDIT_2026-07-28.md`,
`PRODUCTION_HESSIAN_UPPER_ONLY_RELEASE_2026-07-28.md`,
`full_aod_diag/d4_exact/test_hessian_upper_only_sentinel_d4.jl` (new sentinel test), code changes
in `cm_hessian_architectures.jl`/`cm_hessian_threaded.jl`, all committed at `c350280`.

## Verification summary (both tasks combined)

All real KNITRO, D=4, `d4_exact_setup(δ=1.0, find_smallest=true)`:

| Test | Result | When run |
|---|---|---|
| `test_shared_core_hessian_d4_gates.jl` | 40 PASS / 0 FAIL | baseline (pre-edit) |
| `test_shared_core_hessian_d4_gates.jl` | 40 PASS / 0 FAIL | after origin-ZC dead-mirror deletion |
| `test_shared_core_hessian_d4_gates.jl` | 40 PASS / 0 FAIL | after CM/CM+ZC pack-loop optimization |
| `test_shared_core_hessian_d4_gates.jl` | 40 PASS / 0 FAIL | after extracting shared `pack_upper_cm_hessian!` (final state) |
| `test_frechet_hessian_structured_vs_dense_d4.jl` | 20 PASS / 0 FAIL | confirming no collateral effect on untouched common-Fréchet |
| `test_hessian_upper_only_sentinel_d4.jl` (new) | 5 PASS / 0 FAIL | final state |

**Real D=20/W=100,000 gates were not run this session** for either task — an honest scope
limitation given the session's split across two substantial tasks plus the deliberate decision not
to attempt a live, unvalidated fix to the harder legacy-H regression. The D=4 gates above are the
same ones the prior merged session used to validate its own, larger Hessian-callback-side change,
so this is consistent verification depth with that precedent, but real D=20 confirmation remains
an open item before any merge to `production/fullA-exact`.

## Merge policy

Per task instructions and this repo's standing merge-confirmation requirement: **not merged, not
pushed to any remote.** The campaign worktree
(`worktrees/campaign-five-family-bounds-2026-07-28`, branch `campaign/five-family-bounds-2026-07-28`)
had no live process at any point this session but its own handoff doc describes itself as an
in-progress, not-yet-concluded piece of work (5 written-but-unrun smoke scripts) — this is
therefore treated as **not confirmed finished**, so per the task's explicit merge policy:

```
PRODUCTION_MERGE (overall) = port_ready_waiting_for_campaign
```

The cleanup branch (`cleanup/remove-legacy-CC-H-G-storage-2026-07-28`, current HEAD after this
session's two commits) is left in place, not pushed to `origin`/`cdw` without explicit
authorization (this project's own standing rule — see memory
`feedback-confirm-before-pushing-to-real-remote-2026-07-25`).
