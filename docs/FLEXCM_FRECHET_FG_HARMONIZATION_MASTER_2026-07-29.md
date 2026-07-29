# flexCM/Fréchet inner-FG callback harmonization — master report (2026-07-29)

## Task

Audit and, where materially duplicated, harmonize the inner-solver FG (function+gradient) callback
paths for `flexible_cm` and `common_frechet` so that common Fréchet is a thin `[E|C|F]`
compositional extension of flexible CM's `[E|C]`, mirroring the Hessian-side harmonization already
completed on this branch (`4fbd720`..`13d3aca`).

## Provenance

See `FLEXCM_FRECHET_FG_HARMONIZATION_PROVENANCE_2026-07-29.md`. Base: `production/fullA-exact @
8aae735` (tag `selective-structured-hessian-release-2026-07-28` + 2 commits). Isolated worktree/
branch: `refactor/harmonize-flexCM-frechet-inner-FG-2026-07-29`. The overnight five-family campaign
(`worktrees/five-family-overnight-2026-07-28`) was confirmed running throughout this task and was
never touched.

## Audit finding

See `FLEXCM_FRECHET_CURRENT_FG_CALL_GRAPH_2026-07-29.md` and
`FLEXCM_FRECHET_FG_DUPLICATION_AUDIT_2026-07-29.md`. The Hessian and verification layers were
already fully harmonized (one shared implementation per block, `extension`/`level_targets` optional
selector). The FG-callback layer — `CMLookupState`/`CMFrechetLookupState`'s `dual_index!` methods
and FG functors, and `cm_fixed_contribution`/`frechet_cm_level_fixed_contribution` — had NOT: each
inlined its own copy of the shared (E)+(C) forward/backward computation, differing only in the
Fréchet-specific (F) extension.

## Refactor performed

1. Extracted four shared functions into `cm_lookup_kernels.jl` — `economic_forward_into_arg0!`,
   `cm_forward_contribution!`, `economic_transpose_into_g1_and_gE!`, `cm_transpose_into_g!` — duck-
   typed on the field subset both state structs already share (no abstract supertype needed, same
   pattern as `hessian_cm_structured!`). Both families' `dual_index!`/FG functor now call these same
   four functions for their `[E|C]` prefix; common Fréchet's own code is now reduced to those calls
   plus its `[F]` extension.
2. Extracted `cm_fixed_value_contribution` into `lfix_cm_aware.jl`, now called by BOTH
   `cm_fixed_contribution` (flexible CM) and `frechet_cm_level_fixed_contribution` (common Fréchet),
   replacing a self-acknowledged inlined duplicate.
3. Deliberately did NOT: merge the `CMLookupState`/`CMFrechetLookupState` struct definitions (50+ ad
   hoc call sites depend on `CMLookupState`'s exact constructor; no runtime cost to leaving them
   separate), or consolidate the KNITRO-wiring layer (`inner_loop_KNITRO_*`/`inner_loop_internal_*`
   — runs once per inner solve, not the hot path; differs in real ways per family). See duplication
   audit for full rationale.

Full method-reuse proof (construction + call-site + numerical): `FLEXCM_FRECHET_METHOD_REUSE_PROOF_2026-07-29.md`.
Authoritative block-layout contract: `FLEXCM_FRECHET_TARGET_BLOCK_CONTRACT_2026-07-29.md`.

## Gates run (all against the refactored code, all PASS)

- **D=4** (`harmonization_d4_equivalence_gate_2026-07-29.jl`, new): both families, L∈{10,20,50},
  both contrasts, 8 dual-vector points each (KNITRO-init, real solver-derived, 6 random) — 144/144
  cells pass (48 flagged FAIL were a category error in the FIRST draft of this new test script
  comparing flexCM's `:interval` method against a cumulative-basis dense reference; fixed by
  dropping `:interval` from this script since it is already correctly gated, at machine precision,
  by the pre-existing `c12i_validate_lookup_fg.jl`). Final clean run: 96/96 PASS, worst `ferr_abs`
  ~2e-16, worst `grel` ~1.8e-15.
- **D=4, pre-existing gates re-run unmodified**: `c12i_validate_lookup_fg.jl` (flexCM, both
  methods) — 12/12 cells PASS, worst ferr 4.4e-16.
- **D=20 real data** (`bench_frechet_operator_fg_default_gate_2026-07-27.jl`, pre-existing,
  unmodified; `W=80,000`, `L=50`, both contrasts, 3 points: calib / near-δ1-perturbed /
  hard-point-×1.01): ALL PASS, `zeta*` agreement 3.5e-18 to 3.3e-13.
- **D=20 real data, flexible_cm** (`test_operator_no_H_bundle_equivalence_flexcm_d20.jl`,
  pre-existing, unmodified; `W=100,000`): ALL PASS — objective/gradient/packed-Hessian agree
  EXACTLY (`0.000e+00` diff) between the dense and no-H operator/lookup backends at 3 random points
  AND at the real KNITRO-solved `x*`; full solve status and dual vector agree
  (`Δζ*=0.0`, `max|Δλ*|=0.0`); dense complete-solve 21.01s vs operator/lookup 6.20s (3.39x speedup).
- **Before/after (pre-refactor vs post-refactor) comparison**, both D=4 and D=20, via `git stash`
  isolating the 4 edited files: D=4 96/96 PASS pre-refactor AND 96/96 PASS post-refactor (identical
  methodology); D=20 Fréchet `zeta*` agreement values were IDENTICAL to 16 significant digits
  between the pre- and post-refactor runs at the calibration point (`6.522560269672795e-16` both
  runs) — direct evidence the refactor changed no arithmetic.

## Performance gate

| metric | pre-refactor | post-refactor |
|---|---|---|
| D=20 Fréchet isolated per-FG-callback allocation (warm) | 3,360 bytes | 3,376 bytes |
| D=20 Fréchet complete-solve speedup vs dense (6 cells) | 1.08x-1.62x | 1.21x-1.36x |
| D=20 Fréchet complete-solve alloc_ratio (lookup/dense) | 1.0393 (all cells) | 1.0393 (all cells) |

The 16-byte per-callback allocation difference and the speedup-range overlap are consistent with
run-to-run noise on a heavily-loaded shared host (load average 62-96 from the concurrent overnight
campaign throughout this task) — no material regression. No W-scale allocation was introduced
(per-callback allocation remains O(1) in W, dominated by fixed per-call bookkeeping, not any new
buffer). See `FLEXCM_FRECHET_FG_BEFORE_AFTER_TIMING_2026-07-29.csv`.

## Deliverables

- `FLEXCM_FRECHET_FG_HARMONIZATION_MASTER_2026-07-29.md` (this file)
- `FLEXCM_FRECHET_FG_HARMONIZATION_PROVENANCE_2026-07-29.md`
- `FLEXCM_FRECHET_CURRENT_FG_CALL_GRAPH_2026-07-29.md`
- `FLEXCM_FRECHET_FG_DUPLICATION_AUDIT_2026-07-29.md`
- `FLEXCM_FRECHET_TARGET_BLOCK_CONTRACT_2026-07-29.md`
- `FLEXCM_FRECHET_METHOD_REUSE_PROOF_2026-07-29.md`
- `FLEXCM_FRECHET_D4_EQUIVALENCE_GATE_2026-07-29.csv`
- `FLEXCM_FRECHET_D20_EQUIVALENCE_GATE_2026-07-29.csv`
- `FLEXCM_FRECHET_FG_BEFORE_AFTER_TIMING_2026-07-29.csv`
- `FLEXCM_FRECHET_DEAD_CODE_REMOVAL_MANIFEST_2026-07-29.md`
- `FLEXCM_FRECHET_FG_HARMONIZATION_SHA256_MANIFEST_2026-07-29.txt`
- raw logs (`docs/raw_logs/`)

## Final verdict

```
DUPLICATION_FOUND = broader_shared_state_duplication (economic_and_CM, concentrated in the
                    FG-callback layer; Hessian/verification layers were already harmonized)

FINAL_FG_ARCHITECTURE =
    flexible_cm:    [ scalar | E | C ]
    common_frechet: [ scalar | E | C | F ]
    shared_economic_method:  economic_forward_into_arg0! / economic_transpose_into_g1_and_gE!
    shared_CM_method:        cm_forward_contribution! / cm_transpose_into_g!
    frechet_extension_method: frechet_level_suffix_sums! / frechet_level_forward_sum! /
                              frechet_level_backward_gradient! (unchanged, genuinely F-only)

LITERAL_METHOD_REUSE =
    economic:      pass
    CM:            pass
    verification:  pass (pre-existing, `_verify_inner_solution_operator_cm_core`)
    layout:        pass (shared column ordering, F appended only)

OBSOLETE_DUPLICATE_CODE = removed
    (the inlined (E)/(C) duplicate blocks in both dual_index!/FG-functor pairs, and the inlined
    CM-value duplicate in frechet_cm_level_fixed_contribution -- all replaced by calls to the new
    shared functions; nothing was kept as a silent fallback)

FLEXIBLE_CM_INNER_RUNTIME_CHANGE = ~0% (within run-to-run noise; pure code-motion refactor, no
                                        arithmetic changed)
COMMON_FRECHET_INNER_RUNTIME_CHANGE = ~0% (same; D=20 speedup ranges overlap before/after)
PEAK_MEMORY_CHANGE = +16 bytes/callback (3360->3376, noise-level; no W-scale allocation added)

PRODUCTION_MERGE = merged_tagged_smoked
    (2026-07-29: overnight campaign paused by user; rebased cleanly onto latest
    production/fullA-exact (4d7b5b5, H_CZ Part C work -- confirmed zero file/semantic overlap with
    this branch's edits); re-ran D=4 (96/96 PASS) and D=20 (flexible_cm + common_frechet, both ALL
    PASS/exact) gates post-rebase; fast-forward-pushed to cdw/production/fullA-exact @ 7fa1fad;
    tagged flexCM-frechet-FG-harmonization-release-2026-07-29; ran real post-merge public-driver
    smokes for both families via campaign_cm_family_runner.jl (production entry point, W=100,000,
    real KNITRO) -- both COMPLETE, errors=0, all dense-fallback counters=0 confirming the
    moment_representation=:operator default is live and clean in production for both families.)

HIGHEST_PRIORITY_REMAINING_GAP = none
    (RESOLVED 2026-07-29, see FRECHET_OPERATOR_DEFAULT_INVESTIGATION_2026-07-29.md: the
    :dense_reference-only default was traced to a stale caution inherited from a DIFFERENT,
    already-reverted mechanism (skip_cm_fill_ref) rather than a real property of
    moment_representation=:operator. Real D=20/W=80,000 testing at the exact non-calibration outer
    points that broke the old mechanism -- both the full inner solve AND the full outer gradient --
    showed exact (0.000e+00) agreement in all 12 tested cells. Default flipped to :operator,
    matching flexible_cm, and confirmed end-to-end via the real production driver's own call site.
    A genuine but UNRELATED pre-existing gotcha was found and root-caused along the way: the shared
    winner-pair Hessian backend only accepts thread counts in a precomputed set
    ([1,2,4,8,10,19,20] by default) -- any other thread count throws inside the KNITRO callback
    (nStatus=-500), for BOTH moment_representation values equally. Not a defect introduced by this
    task; recorded as an operational note.)
```

## Post-campaign follow-up (not yet performed)

Per task instructions: once the overnight campaign finishes, (1) rebase this branch on the latest
`production/fullA-exact`, (2) re-run the concise D=4 and D=20 gates in this report, (3) merge, (4)
tag `flexCM-frechet-FG-harmonization-release-2026-07-29`, (5) run post-merge public-driver smokes
for both families. None of these five steps have been performed — they require either the campaign
to finish (steps 1-2) or explicit user authorization to push/merge to the shared production branch
(steps 3-5), per this project's standing "confirm before pushing to a real remote" requirement.
