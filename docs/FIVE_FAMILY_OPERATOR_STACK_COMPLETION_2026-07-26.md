# Five-Family Operator Stack Completion — Master Report — 2026-07-26/27

Continuation of `port/shared-inner-fg-operator-and-verification-2026-07-26`, on branch
`port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26`. See
`docs/PHASE0_PROVENANCE_AND_COMMIT_CLASSIFICATION_2026-07-26.md` for provenance/branch discipline.

## What this session did (real, gated, all with actual test runs — not read-only inspection)

### Phase A — finish all five FG operators (COMPLETE for FG; verification 3/5)

1. **Flexible CM** (`cm_lookup_kernels.jl`/`cm_lookup_production.jl`): retrofitted `CMLookupState`'s
   economic (core) block from a dense `obj.H` `BLAS.gemv!` to the shared `economic_forward!`/
   `economic_transpose!`, via the existing `core_cf_ref` box `wrap_moments_with_cm_archB` already
   publishes for the winner-pair Hessian. Falls back safely (byte-identical) when no compressed
   factual is available. Verified D=4 + real D=20/W=80,000, ALL PASS, diffs ≤2.2e-11.
2. **Common Fréchet** (`cm_frechet_lookup_kernels.jl`/`cm_frechet_lookup_production.jl`): fixed the
   inherited **14,066,064 bytes/callback allocation regression** by parameterizing
   `CMFrechetLookupState{O}` (`obj::O` instead of `obj::Any`) — an **8,140x reduction to 1,728
   bytes/callback**, confirmed at BOTH D=4 and real D=20/W=80,000 (not a scale-dependent artifact).
   Also retrofitted the same struct's economic block to the shared operator (same pattern as
   flexible CM). Verified D=4 + real D=20/W=80,000, ALL PASS, diffs ≤3.7e-11.
3. **CM+ZC and Origin-ZC** (inherited opt-in operators from the prior session): ran the
   complete-inner-solve performance A/B the prior session left as the single largest open gap
   (`FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB_2026-07-26.md`'s own "NOT MEASURED" list). Real
   D=4 + D=20/W=80,000 results: correctness ALL PASS, speedup 1.001x–1.084x (within the 5%
   criterion), allocation at parity (~1.0002x — not materially reduced, for the same
   "shared post-solve overhead dominates" reason this codebase's OWN `CM_INNER_FG_BACKEND_DEFAULT`
   flip precedent already documents). **Flipped both families' production default to `:operator`**
   (new `ORIGINZC_FG_BACKEND_DEFAULT`/`CM_MEANZC_INNER_FG_BACKEND_DEFAULT` Refs,
   `core_exact_hessian.jl`), eliminating the last dense `obj.H` read from their ordinary FG path.

### Phase A — operator verification (extended 1→2 of 5 families)

Extended the origin-ZC-only `verify_inner_solution_operator_originzc!` proof-of-concept
(inherited) to CM+ZC (`verify_inner_solution_operator_cmmeanzc!`, `operator_verification.jl`),
reusing CM+ZC's own free-function kernels against fresh, independent scratch. D=4 gate, 3 configs,
ALL PASS — operator-recomputed KKT residual agrees with the dense verifier's own
`max_abs_moment_kkt_resid` to ~1e-15, both effectively zero (stationarity independently confirmed).
Flexible-CM and common-Fréchet's own operator verification were NOT attempted this session (see
`HIGHEST_PRIORITY_REMAINING_GAP` below).

### Phase B — no-dense-G proof (scoped)

Resolved the prior session's open `select_G_from_H` question (confirmed a genuine zero-byte
`@view`, not a dense materialization — see `docs/NO_DENSE_G_GLOBAL_RUNTIME_PROOF_2026-07-26.md`
for the full writeup and the real culprit this misdirected investigation toward, the outer
A-gradient's `composite_gradient_at_fast`). Real runtime-counter evidence (not just code reading)
that flexible_cm/cm_plus_zc/zc_only run their ordinary FG path dense-G-free AT THEIR PRODUCTION
DEFAULT (not merely opt-in). Confirmed by grep that Hessian H_EC/H_ER cross-block dense-column
reads remain (a deliberate, documented Phase A scope boundary) — task §10's winner-aware
cross-Hessian rework was NOT attempted this session. The full ~500-file repository audit task §9
literally asks for was also NOT attempted (scope reasons, consistent with the prior session's own
triage).

### Dispatched, in-progress background work (see their own deliverable docs when complete)

Two large follow-on investigations this session judged too large to fold into the main thread were
dispatched as separate background agents, each in its own isolated git worktree/branch:

- **Shared outer A-gradient allocation rework** (worktree `shared-a-gradient-2026-07-27`, branch
  `feature/shared-outer-a-gradient-2026-07-27`) — directly following on from the hot-path
  allocation audit's finding that the real ~13GB/complete-solve figure comes from
  `composite_gradient_at_fast`'s outer-coordinate finite-difference loop, not this task's own
  FG/Hessian/verification scope. Builds one shared `economic_A_gradient!` entry point for all five
  families atop the existing (but only-unrestricted-wired) pooled/buffered gradient kernels.
- **CM basis diagnosis** (worktree `cm-basis-diagnosis-2026-07-27`, branch
  `diag/cm-basis-interval-orthonormal-2026-07-27`) — Phase C's interval-vs-cumulative CM basis and
  anchored-vs-orthonormal origin-contrast investigation, explicitly kept in scope per the user's
  own instruction not to drop it.

Both were still running at the time this document was written; their own final-verdict documents
(see task list) supersede any Phase C/outer-gradient claims made in earlier planning within this
session's own conversation history.

## Also completed this session (parallel background agent, already finished)

**Hot-path array allocation audit** (worktree `audit-hot-path-array-allocation-2026-07-27`, commit
`e5e5c8f7a14cef3e4ef54c82453a9ea31f9ba997`, already pushed to Dropbox): resolved the
`select_G_from_H` question above, and identified that the real large-allocation site in this
codebase is the outer A-gradient, not the inner FG/verification path this task's main scope
covers — directly motivating the dispatched follow-on above.

## Final verdict (this session's own scope — Phase A/B; see separate docs for Phase C / A-gradient)

```text
ECONOMIC_FG =
    unrestricted:compressed_operator/default (unchanged, Addendum Part A)
    flexible_cm:shared_economic_operator/default (NEW this session)
    common_frechet:shared_economic_operator/available_not_default (CM_FRECHET_INNER_FG_BACKEND_DEFAULT stays :dense_reference pending a clean D=20 perf confirmation -- D=4 shows 1.57x-1.87x speedup, D=20 correctness ALL PASS, D=20 timing run itself crashed/inconclusive this session, see below)
    cm_plus_zc:shared_economic_operator/default (flipped this session)
    zc_only:shared_economic_operator/default (flipped this session)

RESTRICTION_FG =
    unrestricted:not_applicable
    flexible_cm:cm_lookup/default (unchanged, inherited)
    common_frechet:cm_frechet_lookup/available_not_default (unchanged default; economic block now retrofitted)
    cm_plus_zc:operator/default (unchanged, inherited default, now perf-gated this session)
    zc_only:operator/default (unchanged, inherited default, now perf-gated this session)

VERIFICATION =
    unrestricted:dense_reference (not attempted this session)
    flexible_cm:dense_reference (not attempted this session)
    common_frechet:dense_reference (not attempted this session)
    cm_plus_zc:operator/available_not_default (NEW this session, D=4 gated ALL PASS)
    zc_only:operator/available_not_default (inherited from prior session)

HESSIAN_EE = shared winner-pair backend, all 5 families (unchanged, pre-existing)
HESSIAN_ER = family-specific, still dense economic-column-dependent for flexible_cm/common_frechet/cm_plus_zc (confirmed by grep this session, not eliminated); origin-ZC's own exact ZC cross-block unchanged (pre-existing, out of this session's scope)
HESSIAN_RR = family-specific, unchanged (pre-existing)

FULL_G_MATERIALIZATION =
    zero_in_ordinary_FG_callback_for_4_of_5_families (unrestricted, flexible_cm, cm_plus_zc, zc_only all confirmed by real runtime counters; common_frechet has the operator available but not yet default) |
    present_in_Hessian_cross_blocks_by_design (H_EC/H_ER dense-column reads remain, deliberate Phase A scope boundary, task Section 10 not attempted)

CM_FEATURE_IMMUTABILITY = see cm-basis-diagnosis-2026-07-27 background agent's own final doc
CM_BASIS_DEFAULT = see cm-basis-diagnosis-2026-07-27 background agent's own final doc
ORIGIN_CONTRAST_DEFAULT = see cm-basis-diagnosis-2026-07-27 background agent's own final doc
INTERVAL_HESSIAN = see cm-basis-diagnosis-2026-07-27 background agent's own final doc

PRODUCTION_MERGE = port_ready_not_merged
    (real, gated, committed work on this branch; not pushed to origin or merged to
    production/fullA-exact this session -- per feedback-confirm-before-pushing-to-real-remote-2026-07-25,
    push/merge requires explicit user authorization, not granted this session)

HIGHEST_PRIORITY_REMAINING_GAP =
    common-Frechet's D=20 performance confirmation (the D=20 timing bench crashed with no
    stack trace mid-run this session, cause not diagnosed -- D=4 evidence is strongly positive
    (1.57x-1.87x speedup) and D=20 correctness is solid, but the task's own decision rule requires
    a real D=20 timing result before flipping the default, which this session does not have clean
    evidence for)
```
