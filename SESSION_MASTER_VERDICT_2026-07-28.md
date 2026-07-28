# Session master verdict — parallel legacy-CC-H removal + Hessian upper-only cleanup — 2026-07-28

Branch: `cleanup/remove-legacy-CC-H-G-storage-2026-07-28`. Base:
`campaign/five-family-bounds-2026-07-28@93f26df7`. **Superseded an earlier draft of this document**
that stopped at "audited and designed, not implemented" — real, root-caused, validated fixes were
made after that draft was written, on direct user push-back against stopping at an intermediate
step. This is the accurate final state.

## Campaign isolation

No file in `worktrees/campaign-five-family-bounds-2026-07-28` was read, written, or executed by
this session. The campaign resumed with live processes partway through this session (confirmed via
`/proc/*/cwd` scan) — the isolation held throughout. See
`PARALLEL_LEGACY_CC_H_REMOVAL_PROVENANCE_2026-07-28.md` for the full isolation record.

## Commits on this branch (chronological)

```
c350280  Hessian upper-only cleanup: remove no-op symmetrization for mirrored blocks
e292403  Legacy CC-H removal: audit, root-cause analysis, and target-architecture design (no code change)
2ffd4f2  Flexible-CM: eliminate priming-side dense economic-block fill (real fix, root-caused)
83f28c8  CM+ZC and origin-ZC: add and validate the same economic-block skip mechanism
eca6945  Add shared OperatorPsiBundle type (design + Hessian-side wiring, not yet integrated)
```

## Task 1: Hessian upper-only cleanup (`c350280`)

```
PRODUCTION_HESSIAN_ASSEMBLY = remaining_symmetrization_H_EE_H_CC_and_all_common_frechet_blocks
PRODUCTION_HESSIAN_LOWER_ENTRIES_WRITTEN = >0 (origin-ZC's one dead site fixed and now 0 there;
                                                CM/CM+ZC/common-Fréchet's mirror WRITES
                                                intentionally retained -- cctx.Hfull is read
                                                directly, both triangles, by real diagnostic
                                                scripts; only the unnecessary READS/averaging in
                                                the packing step were removed)
PRODUCTION_HESSIAN_SYMMETRIZATION_PASSES = >0 (H_EC's averaging removed for flexible-CM/CM+ZC;
                                                H_EE/H_CC/all 4 common-Fréchet sites retained)
PACKED_UPPER_SENTINEL_TEST = pass_flexible_cm_and_cmzc_H_EC_region_scoped (5/5, real KNITRO D=4,
                              with a negative control proving it isn't vacuous)
PRODUCTION_MERGE = port_ready_waiting_for_campaign
```

See `PRODUCTION_HESSIAN_UPPER_ONLY_AUDIT_2026-07-28.md` / `_RELEASE_2026-07-28.md` for detail. Not
revisited after the initial pass — common-Fréchet's 4 analogous sites remain the clear next target,
untouched.

## Task 2: legacy CC-H storage removal — real fixes, not just design

### The root cause, found and fixed (not just hypothesized)

The prior two sessions' attempts to skip the priming-side dense economic-block fill for flexible-CM
were reverted after what looked like a real numerical regression (`max|Δ|=0.0336`, then an outright
`nStatus=-400`). This session found the actual mechanism empirically (instrumented live runs, not
static reading) and it was **not** a production correctness bug:

1. `archC_base_state`'s `skip_fill_safe` gate checked `cctx.cm_cross_hessian_backend` (irrelevant)
   instead of `cctx.core_hessian_backend` (the flag that determines whether a context's Hessian
   path will read dense H) — a context deliberately built to want `:dense_reference` ground truth
   got the skip applied to it anyway.
2. Independently, `test_shared_core_hessian_d4_gates.jl`'s own `full_hessian`/`full_hessian_mz`
   test helpers called the legacy dense-only `_archC_prep_for_hessian!` directly for **both**
   comparison arms, bypassing the real production dispatcher (`_prep_dual_index_for_archC!`) — a
   test-methodology bug layered on top of (1).

Both fixed. `core_hessian_backend=:dense_reference` is never set outside this repo's own comparison
harnesses, so fix (1) is a no-op for every real production run.

### What's actually true now, per family

```
flexible_cm:  economic-block priming fill genuinely skippable in production default config.
              G[:,1:pregrav] copy also skipped (was previously unconditional, would have
              propagated undef-backed stale scratch memory once the fill itself was skipped).
              VALIDATED: test_shared_core_hessian_d4_gates.jl, 40/40 PASS, real KNITRO D=4,
              real-solved-point arms included. Committed 2ffd4f2.

cm_plus_zc:   same fix, extended -- this family had NO skip mechanism at all before this session
              (wrap_moments_with_cm_meanzc always filled unconditionally). Added skip_fill kwarg,
              second closure sharing core_cf_ref, threaded through cctx.moments_skip! (a
              pre-existing but never-populated struct field) and inner_loop_internal_meanzc_operator.
              VALIDATED: same test, 40/40 PASS. Cross-checked against test_cm_originzc_pure_moments.jl
              (unaffected, all gates pass). Committed 83f28c8.

zc_only:      same fix. OriginZCCoreHessCtx gained a NEW moments_skip! field (single construction
              site, low blast radius). VALIDATED: same test, 40/40 PASS. Committed 83f28c8.

common_frechet: NOT changed. Investigated at length this session (see below) -- inconclusive.
              Left exactly as the campaign merged it (skip_fill_safe_frechet hardcoded false).
```

### Common-Fréchet investigation (inconclusive, left unchanged — full honesty on this)

Common-Fréchet's own code history documents two prior real (not test-artifact) `nStatus=-400`
failures when this same class of skip was attempted, the second one explicitly re-tested *after*
the operator dual-index infrastructure this session's flexible-CM fix also depends on was already
in place. Given flexible-CM's "real regression" turned out to be a red herring, this session
re-tested common-Fréchet's skip on the current code:

- D=4 (`test_frechet_hessian_structured_vs_dense_d4.jl`, skip re-enabled diagnostically): 20/20
  PASS. **Known insufficient** — the code's own comment explicitly says a passing D=4 gate is not
  sufficient evidence (it also passed D=4 historically, then failed at real D=20).
- First real-D20 attempt (`test_cm_frechet_threaded_hessian_gates.jl`): passed, but **the test
  itself never exercises the skip mechanism at all** — it calls `inner_loop_internal_archgeneric`,
  which always uses the unconditional-fill closure. This was caught as a real analysis error mid-session,
  not left uncorrected.
- Second real-D20 attempt (custom script, `archC_frechet_base_state` at 1%/3%/5%/10% multiplicative
  perturbations of the calibrated θ): all 4 failed with `nStatus=-300`.
- **Control run** (identical script, skip left at its original `false`): **the exact same
  `nStatus=-300` failure at all 4 perturbation levels.** This proves the failure is an artifact of
  the perturbation script itself (a naive multiplicative perturbation on a parameter vector
  spanning ~1 to ~5e7 in magnitude produces a nonsensical point), not evidence about the skip
  mechanism either way.

Net result: **no valid D=20 evidence was obtained, for or against.** The diagnostic edit to
`cm_frechet_cplus.jl` was reverted (`git stash`, not committed) rather than left half-tested.
Common-Fréchet's skip remains `false`, unchanged from the campaign's own merged state. A real D=20
answer requires a properly-scaled perturbation (or a genuinely feasible non-calibration point from
elsewhere in the codebase) — left as the named next step, not attempted further given the time
already spent on this specific sub-investigation this session.

### `OperatorPsiBundle` — genuinely separate type, designed and partially wired, NOT integrated

Per direct user request to pursue the actual field/allocation removal, not stop at the
skip-mechanism fix: `full_aod_diag/d4_exact/operator_psi_bundle.jl` defines a new struct with no
`H` field and no H-sized preallocation at all, **shared across every restricted family** (not
per-family — an earlier draft was mistakenly named `OperatorCMBundle` before a side-by-side check
of all four families' priming closures showed the logic is identical, differing only in the
θ-slicing each family's own entry point already does before calling in). `prime_operator!` is the
one shared priming function; `_dense_H_or_nothing(obj)` lets the shared Hessian-callback code
(`hessian_cm_structured!`/`_v2!`, `_fill_cm_HEE!`, `build_bin_tables!`) work for both this type and
the unchanged `PsiObjectiveBundleImplicit` without an unconditional `@unpack H`.

**Not done**: construction (`build_cm_production_context` still builds `PsiObjectiveBundleImplicit`
unconditionally) and priming-call-site wiring (`inner_loop_internal_cmlookup_production` still uses
the `obj.H`-based dispatch, not `prime_operator!`) for any family. The type exists, is gated not to
regress the existing path (40/40 PASS unchanged), and is a concrete, close-to-actionable next step
— but it is not yet reachable from any production code path. Committed `eca6945` as exactly that:
a real, honest, partial step, not a claimed completion.

### Incidental finding, flagged separately

`FLAG_GRAVITY_MOMENT_NONZERO_AWAY_FROM_CALIBRATION_2026-07-28.md` — the gravity moment column is
machine-precision zero exactly at calibration, genuinely nonzero (not noise) at even small
perturbations away from it. Not part of the storage-cleanup work; recorded because it surfaced
while checking whether the gravity column was legacy/removable, and the user asked for it written
up explicitly. Not investigated to root cause (outer-loop/pivot-construction question, out of this
session's scope).

## Verdict block

```
LEGACY_CC_H_MATRIX_ALLOCATIONS   = 5 live + 2 throwaway (unchanged this session -- obj.H's field
                                    and default allocation are untouched; only what gets WRITTEN
                                    into it, for 3 of 4 restricted families, is now conditional)
G_SIZED_BACKING_STORAGE_ALLOCATIONS = same (allocation size unchanged; OperatorPsiBundle would
                                    eliminate this but is not yet wired into construction)
CC_PAYOFF_K_VECTOR_ALLOCATIONS   = 0 (K is a required O(W) scalar fill, already at target)
ONES_VECTOR_ALLOCATIONS          = 1 per PsiObjectiveBundleImplicit construction (struct default),
                                    not 0 -- unchanged

PRODUCTION_MOMENTS_CALLS         = 4 (call COUNT unchanged -- moments!/its skip-variant is still
                                    called once per inner solve for all 4 restricted families; what
                                    changed is that 3 of the 4 calls now do genuinely less work)
PRODUCTION_SELECT_G_FROM_H_CALLS = 4, unchanged
COMPOSITE_G_MATERIALIZATIONS     = 0 on the Hessian-callback side (inherited); the priming-side
                                    economic-block materialization is now SKIPPED (not just
                                    read-but-unused) for flexible-CM/CM+ZC/origin-ZC in production
                                    default config -- common-Fréchet still materializes it
                                    unconditionally

OPERATOR_BUNDLES_WITH_H_FIELD    = unrestricted, flexible_cm, common_frechet, cm_plus_zc, zc_only
                                    (all 5 -- OperatorPsiBundle exists as a type with no H field,
                                    but is not yet the type any family actually constructs)
OPERATOR_BUNDLES_WITH_K_FIELD    = same 5 (unchanged)
OPERATOR_BUNDLES_WITH_MOMENTS_FIELD = same 5 (unchanged)

FIVE_FAMILY_ARCHITECTURE = incomplete_operator_bundle_type_not_wired_common_frechet_skip_unresolved

PRODUCTION_MERGE = port_ready_waiting_for_campaign
    -- real, validated, D=4-gated fixes for 3 of 4 restricted families' priming-side waste
    -- common-Fréchet's own skip status genuinely unresolved (not merely undone), left safe
    -- OperatorPsiBundle designed and Hessian-side-ready, not yet load-bearing anywhere

HIGHEST_PRIORITY_REMAINING_GAP = wire OperatorPsiBundle's construction + priming call site into
    flexible-CM's build_cm_production_context/inner_loop_internal_cmlookup_production (the
    best-understood, most-validated family) and gate at D=4 -- this is the actual remaining path
    to LEGACY_CC_H_MATRIX_ALLOCATIONS=0/G_SIZED_BACKING_STORAGE_ALLOCATIONS=0, not a redesign.
    Second-priority: get a valid (properly-scaled) real-D20 answer for common-Fréchet's skip.
```

## Deliverables (this session, both tasks)

`PARALLEL_LEGACY_CC_H_REMOVAL_PROVENANCE_2026-07-28.md`,
`LEGACY_CC_H_G_CONSTRUCTOR_AND_CALLSITE_AUDIT_2026-07-28.md`,
`OPERATOR_DENSE_BUNDLE_TYPE_SEPARATION_2026-07-28.md` (superseded in part by the real fix found
after it was written — kept as the historical root-cause writeup, see its own text),
`FIVE_FAMILY_NO_LEGACY_H_STORAGE_GATE_2026-07-28.md`,
`FINAL_FIVE_BY_SEVEN_NO_LEGACY_STORAGE_MATRIX_2026-07-28.md` + `.csv` (both now stale relative to
this document — see note in each), `PRODUCTION_HESSIAN_UPPER_ONLY_AUDIT_2026-07-28.md` + `_RELEASE`,
`FLAG_GRAVITY_MOMENT_NONZERO_AWAY_FROM_CALIBRATION_2026-07-28.md`, this document, `SHA256_MANIFEST_2026-07-28.txt`
(stale — regenerate before merge if needed).

## Merge policy

Campaign confirmed **active** partway through this session (live processes observed under
`worktrees/campaign-five-family-bounds-2026-07-28`) — not merged, not pushed to any remote.

```
PRODUCTION_MERGE (overall) = port_ready_waiting_for_campaign
```
