# Shared FG and Verification Release — Summary — 2026-07-27

Overview doc tying together the release's FG-operator and verification state across all five
families. Per-topic detail lives in the sibling deliverable docs listed at the bottom.

## What changed this session vs. what was inherited

Inherited (from the three reconciled branches, unchanged in substance, adopted verbatim):
shared `economic_forward!`/`economic_transpose!` consumed by all 4 restricted families' FG;
flexible-CM/CM+ZC/ZC-only default FG flipped to their own matrix-free backend
(`cm_lookup`/`operator`/`operator`); operator-based verification proof-of-concept for origin-ZC
and CM+ZC (not wired as production default for either).

New this session: (1) a real, previously-undisclosed correctness bug in the common-Frechet lookup
FG backend found and fixed (`COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md`) — the backend is now
provably correct at D=20, not merely fast; (2) the shared outer A-gradient wired onto 2 more
families (CM+ZC, flexible-CM), for 3 of 5 total (`FIVE_FAMILY_SHARED_A_GRADIENT_GATE_2026-07-27.md`);
(3) the persistent L-fix base cache generalized from square-only to rectangular
(`PERSISTENT_LFIX_BASE_CACHE_RELEASE_2026-07-27.md`); (4) a rectangular square-layout crash fixed
across the buffered/pooled gradient backends (`SHARED_A_GRADIENT_RECTANGULAR_FIX_AND_RELEASE_2026-07-27.md`).

## What did NOT change this session (honest gaps)

- **Operator verification** (task §3): still exactly 2 of 5 families (origin-ZC, CM+ZC), both
  inherited proof-of-concept, neither wired as the production `verification_backend` default for
  ANY family. Flexible-CM/common-Frechet/unrestricted operator verification was NOT attempted this
  session — this is the single largest unclosed gap from the original task relative to its own
  ambitions. See `FIVE_FAMILY_OPERATOR_VERIFICATION_RELEASE_2026-07-27.md`.
- **`skip_cm_fill_ref` removal** (task §3's explicit ask): NOT done — removing it project-wide
  requires operator verification for flexible-CM first (per the inherited branches' own stated
  dependency), which wasn't reached. The ONE `skip_cm_fill_ref` site that WAS touched this session
  (common-Frechet's `archC_frechet_base_state`) was removed for a correctness reason (the Hessian
  bug), not as part of a general architectural removal — see the Frechet fix doc.
- **Common-Frechet and unrestricted A-gradient wiring**: not attempted (2 of 5 families remain on
  their legacy gradient path).
- **Persistent cache NOT wired into `economic_A_gradient!`**: the underlying mechanism is now
  rectangular-capable and gated, but `shared_a_gradient.jl` still calls the allocating
  `build_lfix_base_cache` — the ~615 MB/gradient warm allocation this task's §5 asks to reduce by
  ≥80% is UNCHANGED this session (614.83 MB warm, same as the inherited figure). See
  `PERSISTENT_LFIX_BASE_CACHE_RELEASE_2026-07-27.md` for the precise remaining wiring step.
- **Threading gate** (task §6, serial/4/8/10/20 workers): not run as a systematic sweep this
  session. The D=20 regression gates this session DID run both serial and one threaded
  configuration (matching `Threads.nthreads()` on this machine) as part of their own bit-identical
  checks, but that is not the 5-point worker-count sweep the task asks for.

## Final verdict (this release's own scope)

```text
ECONOMIC_FG_DEFAULT =
    unrestricted:compressed_operator
    flexible_cm:shared_economic_operator (via cm_lookup, default)
    common_frechet:dense_reference (shared_economic_operator available via cm_frechet_lookup --
        now correct AND gated this session, still not default, see the Frechet fix doc)
    cm_plus_zc:shared_economic_operator (via operator backend, default)
    zc_only:shared_economic_operator (via operator backend, default)

RESTRICTION_FG_DEFAULT =
    unrestricted:not_applicable
    flexible_cm:cm_lookup
    common_frechet:dense_reference (cm_frechet_lookup available, correct, not default)
    cm_plus_zc:operator
    zc_only:operator

VERIFICATION_DEFAULT =
    unrestricted:dense_reference (no operator verification built, any session)
    flexible_cm:dense_reference (no operator verification built, any session)
    common_frechet:dense_reference (no operator verification built, any session)
    cm_plus_zc:dense_reference (operator verification exists as inherited proof-of-concept,
        NOT wired as production default)
    zc_only:dense_reference (operator verification exists as inherited proof-of-concept,
        NOT wired as production default)

A_GRADIENT_DEFAULT =
    unrestricted:composite_gradient_at_fast_buffered (legacy; NOT wired to economic_A_gradient!
        this task -- would require touching the large, actively-used production driver file)
    flexible_cm:shared_inplace_pooled (economic_A_gradient!, WIRED this session)
    common_frechet:legacy_unbuffered (composite_gradient_at_fast; NOT wired)
    cm_plus_zc:shared_inplace_pooled (economic_A_gradient!, WIRED this session)
    zc_only:shared_inplace_pooled (economic_A_gradient!, inherited, pre-existing)

A_GRADIENT_WARM_D20_ALLOCATION = 614.83 MB (UNCHANGED this session -- see
    PERSISTENT_LFIX_BASE_CACHE_RELEASE_2026-07-27.md for why, and the precise remaining step)
A_GRADIENT_WORKERS = not systematically swept this session (serial + one threaded config only,
    as part of existing bit-identical regression gates)

FG_AND_VERIFICATION_DENSE_G =
    present_flexible_cm,common_frechet,cm_plus_zc (Hessian H_EC/H_ER cross-block dense-column
    reads remain -- inherited, confirmed unchanged this session, see
    REMAINING_DENSE_G_CONSUMERS_HANDOFF_2026-07-27.md)

CM_BASIS_DEFAULT = cumulative
ORIGIN_CONTRAST_DEFAULT = anchored
CM_FEATURE_IMMUTABILITY = pass

PRODUCTION_MERGE = port_ready_not_merged
    (every commit on this branch is individually gated with real command output; nothing pushed
    to origin or merged into production/fullA-exact -- per this project's own standing rule,
    that requires explicit user authorization not granted this session)

NEXT_STANDALONE_TASK = winner_aware_H_ER_cross_hessian
```

## Deliverable index

- `THREE_BRANCH_RECONCILIATION_PLAN_2026-07-27.md`
- `COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md`
- `FIVE_FAMILY_OPERATOR_VERIFICATION_RELEASE_2026-07-27.md`
- `SHARED_A_GRADIENT_RECTANGULAR_FIX_AND_RELEASE_2026-07-27.md`
- `PERSISTENT_LFIX_BASE_CACHE_RELEASE_2026-07-27.md`
- `FIVE_FAMILY_SHARED_A_GRADIENT_GATE_2026-07-27.md`
- `CM_BASIS_AND_CONTRAST_DEFAULT_RECONCILIATION_2026-07-27.md`
- `REMAINING_DENSE_G_CONSUMERS_HANDOFF_2026-07-27.md`
- `SHARED_ECONOMIC_MOMENT_STATE_BUILDER_2026-07-27.md` / `ALLOCATING_COMPRESSED_FACTUAL_CALLSITE_AUDIT_2026-07-27.md` / `FIVE_FAMILY_INPLACE_COMPRESSED_FACTUAL_GATE_2026-07-27.md` (moment-construction addendum, dispatched to and completed by a background sub-agent this session)
