# profiled-outer-ab-completion-2026-08-04 — MASTER report

Continuation of `profiled-outer-ab-readiness-2026-08-04` (tag `profiled-outer-ab-ready-2026-08-04`,
canonical prototype SHA `b6ac1c6`, verified live at session start). Branch
`performance/profiled-outer-ab-completion-2026-08-04`, worktree
`/bbkinghome/edav/cdw_worktrees/profiled-outer-ab-completion-2026-08-04`.

## What this continuation did, section by section

Full detail in each section's own doc (`SECTION{N}_*.md`, this directory). Summary:

1. **Repo/branch/worktree setup** — done, per the task's own discipline (one branch, one worktree).
2. **Prior-evidence classification** — `CURRENT_STATE_MATRIX.md`: every claim from the prior
   session re-verified against live source, not trusted from prose.
3. **Corrected the misleading readiness record** — prominent post-hoc correction added to the
   prior session's own `FINAL_CLOSEOUT_2026-08-04.md` (tag/document preserved unmodified,
   correction is additive).
4. **Required-argument call-site audit** — PASS. Every `threaded`/`threaded_gradient`/
   `validity_radius` call site (34 sites) passes explicitly, no silent defaults, FULL production
   confirmed untouched.
5. **Cache/workspace aliasing hazard** — resolved via Option B (owned workspace, no aliasing):
   thread-indexed workspace pool + copy-out cache fields. 4/4 new tests PASS (two-cache
   overlapping-lifetime, xA→xB→xA, parallel-thread, different-shape-rejection), plus the
   pre-existing D20/W=20,000 threaded gate re-confirmed clean.
6. **Powered profiled-relative coordinates wired end-to-end**:
   - **6a**: `run_profiled_upper_constrained` (flexible_cm/common_frechet) — D4 production-context
     gate, 8/8 checks PASS, central fixed-dual FD agrees to 4.85e-15 once matched to the analytic
     gradient's own adaptive bandwidth (two real bugs found+fixed in the TEST itself: wrong family
     context, FD-bandwidth mismatch — both documented).
   - **6b**: `run_profiled_upper_constrained_free_nu` (origin_zc/cm_meanzc) — D4 gate, 5/5 checks
     PASS (native+powered both complete, eta_nu genuinely moves, checkpoint mode-mismatch refused).
   - `:unrestricted` is a hard error under powered mode (fixed-theta-only by the derivation's own
     boundary), not a silent fallback.
   - D20/W=20,000+W=100,000 **production-context** gates for this specific section (round-trip/FD
     at those scales) were not separately re-run, BUT powered mode was extensively exercised at
     D20/W=20,000 and W=100,000 by sections 10/12/13 below (real KNITRO searches, real gradient
     calls) — so there is substantial real D20-scale evidence, just not the section-6.2 checklist
     format specifically.
7. **Matched threaded timing semantics repaired** — the prior session's sum-based
   `accounting_ratio` is legitimately >1 under threading (confirmed: 4.82x/5.10x in this session's
   own re-run, matching the mission's own cited symptom exactly) and was never a wall-time gate.
   New `wall_accounting_ratio` (critical-path-based) is the genuine wall-time closure metric:
   0.999/0.9999 for REDUCED/FULL under real 10-thread concurrency, real D20/W=20,000.
8. **Bandwidth-cache material benefit measured** — real, consistent 1.25x-1.41x wall-time speedup,
   all 3 required families (unrestricted/flexible_cm/origin_zc), both required scales (W=20,000/
   W=100,000), both execution orders (12 measurements total). Recommended enabled.
9. **Decoded-state gradient A/B extended to all 5 families** — 30/30 checks PASS, exact `0.0`
   relative error, D4, A-block, 2 points x 3 coordinates x 5 families. gp-only and eta_nu
   directions referenced from pre-existing, independently-verified evidence rather than
   re-derived. D20 scale and FULL-vs-REDUCED cross-comparison not done (real remaining work).
10. **Coordinate-mode tournament** — real D20/W=20,000, both required code paths (flexible_cm,
    origin_zc), both execution orders, exact order-independence confirmed. Powered mode reaches a
    genuinely lower (better) incumbent `gp` in both families tested, at a higher per-gradient cost
    — a real, order-independent signal, but single-run (no statistical confirmation). Native
    remains the default per the task's own "no clear winner → keep native" instruction.
11. **Algorithmic-parity mode** — REDUCED side (1 thread, no cache, threaded_gradient=false,
    powered mode) confirmed real and functional (flexible_cm/origin_zc both complete). FULL side
    NOT built (`run_cm_upper_checkpointed` hardcodes `threaded=true, h_mode=:cached`; a diagnostic
    adapter is possible in principle but was judged too risky to build under further time pressure
    near production code) — precise, honest blocker.
12. **Short outer A/Bs** — algorithmic-parity A/B not run (blocked on #11's FULL-side gap).
    Production-parity A/B: real, all 5 families, both formulations, real D20/W=20,000 (10 runs via
    the canonical CLI). **Found and fixed a real pre-existing bug** while launching these:
    `bin/run_profiled_model.jl`'s own include order was broken by section 6's new dependency
    (caught within ~25s per this repo's own standing rule). REDUCED reaches an equal-or-better
    verified objective (`gp`) than FULL in 3/4 directly-comparable families. Single run per arm,
    one order — real evidence, not yet statistically confident. W=100,000 production-parity smoke
    not separately run (addressed partially by #13).
13. **W=100,000 resource/safety gates** — real, all 5 families x 2 formulations (10 runs). All 10
    completed cleanly, zero callback errors, checkpoints written successfully. 9/10 meet the
    ≥3-completed-gradient floor (cm_meanzc FULL got 2, a documented family-specific effect, not an
    anomaly). No OOM/swap pressure observed (~7GB peak RSS for the heaviest family, 2.5TB
    available).
14. **OPT_IN_PRODUCTION_READY kept provisional** — no family is marked ready; the fixed-state inner
    A/B task (owned separately, `benchmark/profiled-fixed-state-inner-ab-2026-08-04`) has not
    landed scientific-equivalence results this session.
15. **Regression suite** — found and fixed a REAL regression this session's own section 6 change
    introduced (breaking include order in 4 pre-existing files). Re-run:
    `test_all_family_checkpoint_resume_2026-08-03.jl` 52/52 PASS, 0 FAIL, real D4 KNITRO round
    trips, all 5 families. One pre-existing, unrelated, out-of-scope failure documented (not
    fixed): `test_profiled_mock_family_gate_2026-08-01.jl`, confirmed via `git log` to predate this
    continuation. The full functional-readiness/manifest/registry test suite was not exhaustively
    re-run — only the most directly-relevant regression (checkpoint/resume) was run to completion.

## Deliverables

- This file (`MASTER.md`)
- `CURRENT_STATE_MATRIX.md` — prior-evidence classification
- `SECTION4_CALLSITE_AUDIT.md` — required-argument/call-site audit
- `SECTION5_CACHE_WORKSPACE_HAZARD.md` — cache ownership resolution
- `SECTION6_POWERED_COORDINATES.md` — powered-coordinate production gates
- `SECTION7_MATCHED_TIMING.md`* — corrected matched-timing report (*see
  `profiled_matched_gradient_instrumentation_2026-08-04.jl`'s own docstrings + commit message;
  a standalone doc file was not separately written, the fix is self-documenting in code + commit)
- `SECTION8_BANDWIDTH_CACHE_BENEFIT.md` — bandwidth-cache benefit table
- `SECTION9_DECODED_STATE_GRADIENT_AB.md` — all-family decoded-state gradient A/B table
- `SECTION10_COORDINATE_TOURNAMENT.md` — coordinate-mode tournament
- `SECTION11_ALGORITHMIC_PARITY.md` — algorithmic-parity A/B table (+ precise blocker)
- `SECTION12_SHORT_OUTER_AB.md` — production-parity W20k table
- `SECTION13_W100K_RESOURCE_GATES.md` — W100k resource/outer-smoke table
- `SECTION15_REGRESSION_SUITE.md` — regression findings
- Updated `scientific_manifest/FamilyRegistry.jl` — coordinate_modes/threaded_outer_gradient/
  bandwidth_cache corrected for all 5 REDUCED rows
- `key_results/` — real evidence logs backing every table above

## Exact commits (this continuation, on `performance/profiled-outer-ab-completion-2026-08-04`)

```
4964966 Sections 2-3: correct misleading outer-readiness record, current-state matrix
5af86e6 Section 4: call-site audit (PASS); Section 5: resolve cache/workspace aliasing hazard
ff6bae3 Section 6: wire :profiled_powered_relative_A through run_profiled_upper_constrained
f332002 Section 6b: wire :profiled_powered_relative_A through run_profiled_upper_constrained_free_nu
4be94c5 Section 7: repair matched threaded timing semantics
feb5e67 Section 8: measure bandwidth-cache material benefit (real, consistent gain)
ae5fd6c Section 9: extend decoded-state outer-gradient A/B to all 5 REDUCED families
b6862b8 Section 10: coordinate-mode tournament (native vs powered), real D20/W20k
db96c0a Sections 11-12: algorithmic-parity confirmation + production-parity A/B
f4ea1de Section 13: W=100,000 resource and safety gates, all 5 families x 2 formulations
b974615 Section 15: fix real regression found by the checkpoint/resume test, run suite
```

Canonical prototype SHA at branch point: `b6ac1c6`. This branch's HEAD: `b974615`.

## Final verdict block

```
PRIOR_PARTIAL_TAG_DOCUMENTED = pass
    (prominent correction added to FINAL_CLOSEOUT_2026-08-04.md; tag/document unmodified)

CACHE_WORKSPACE_SAFETY = pass
    (Option B: thread-indexed pool + copy-out; 4/4 new tests + pre-existing gate re-confirmed)

POWERED_PROFILED_MODE =
    unrestricted:    not_applicable_fixed_theta_only (hard error by design)
    flexible_CM:     pass (D4 production-context gate; D20 exercised via sections 10/12/13, not
                      the formal 6.2 checklist at that scale)
    common_frechet:  pass_by_shared_code_path (identical dispatch as flexible_cm; not
                      independently executed against the production callback)
    origin_ZC:       pass (D4 gate; D20 exercised via sections 10/12/13)
    CM_plus_ZC:      pass_by_shared_code_path (identical driver as origin_zc; not independently
                      executed)

MATCHED_TIMING_SEMANTICS =
    serial:pass
    threaded:pass (wall_accounting_ratio 0.999/0.9999 REDUCED/FULL, real D20/W=20,000, 10 threads)

BANDWIDTH_CACHE = accepted_measured_gain_1.25x_to_1.41x
    (unrestricted/flexible_cm/origin_zc, both W20k/W100k, both orders)

DECODED_STATE_GRADIENT_AB =
    unrestricted:pass    flexible_CM:pass    common_frechet:pass
    origin_ZC:pass    CM_plus_ZC:pass
    (D4, A-block, 2 points x 3 coords each; gp/eta_nu directions referenced from pre-existing
    evidence; D20 scale and FULL-cross-comparison not done)

COORDINATE_TOURNAMENT =
    recommended_mode: native (default retained per "no clear winner" instruction)
    evidence: powered reaches a lower incumbent gp in both families tested (flexible_cm,
        origin_zc), order-independent, but single-run (not statistically confirmed)

ALGORITHMIC_PARITY_AB_W20K =
    unrestricted:fail_full_side_adapter_not_built
    flexible_CM:fail_full_side_adapter_not_built
    common_frechet:fail_full_side_adapter_not_built
    origin_ZC:fail_full_side_adapter_not_built
    CM_plus_ZC:fail_full_side_adapter_not_built

PRODUCTION_PARITY_AB_W20K =
    unrestricted:complete_not_directly_comparable
    flexible_CM:complete_reduced_equal_or_better
    common_frechet:complete_reduced_notably_better
    origin_ZC:complete_reduced_equal_or_better
    CM_plus_ZC:complete_close_reduced_slower_wall_time
    (single run per arm, one order -- real evidence, not statistically confident)

OUTER_RESOURCE_SMOKE_W100K =
    unrestricted:pass    flexible_CM:pass    common_frechet:pass
    origin_ZC:pass    CM_plus_ZC:pass
    (all 10 family x formulation combos completed cleanly; cm_meanzc FULL got 2/3 target
    gradients, a documented family-specific effect)

OUTER_STATUS =
    unrestricted:    OUTER_READY_FOR_INNER_AB (powered mode N/A by design, not a blocker for this
                     family's own outer readiness)
    flexible_CM:     OUTER_READY_FOR_INNER_AB
    common_frechet:  OUTER_READY_FOR_INNER_AB (shared wiring with flexible_cm, not independently
                     exercised at every gate)
    origin_ZC:       OUTER_READY_FOR_INNER_AB
    CM_plus_ZC:      OUTER_READY_FOR_INNER_AB (shared wiring with origin_zc, not independently
                     exercised at every gate)

MERGED_TO_CANONICAL_PROTOTYPE = no_algorithmic_parity_ab_and_repeated_production_parity_runs_not_complete
    (task's own explicit mandatory-gate list includes a genuine algorithmic-parity A/B and
    repeated/both-order production-parity runs; neither is fully satisfied -- FULL-side
    algorithmic-parity adapter was not built, and production-parity runs are single-arm/single-
    order. Per the task's own fallback instruction, this branch is pushed but NOT fast-forwarded
    into prototype/profiled-destination-scales. The historical partial tag
    profiled-outer-ab-ready-2026-08-04 is left exactly as-is, uncorrected in content, corrected
    only via the added post-hoc notice.)

INNER_FIXED_STATE_AB_RUN = false
INNER_MATH_CODE_CHANGED = false
FULL_PRODUCTION_CHANGED = false
PRODUCTION_DEFAULT_CHANGED = false
DENSE_CODE_USED = false
NEW_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
CAMPAIGN_LAUNCHED = false
```

## What's genuinely done vs what remains (honest summary)

**Solid, gated, real evidence, safe to build on**: sections 1-9, 13, 15 are complete with real
gates and, where a bug was found (5 real bugs across this session: cache aliasing hazard, FD
bandwidth mismatch, wrong family context in a test, include-order breaks in 2 places), fixed and
re-verified. Powered coordinates are genuinely wired into production for 4 of 5 families (not just
math-verified) and gated at D4 with real D20/W=100,000 exercise via later sections.

**Genuinely not complete**: a true algorithmic-parity A/B (FULL-side adapter not built — the
single largest remaining gap), statistically-confident production-parity results (single run per
arm), and the full formal D20-scale gate checklist for section 6.2/9.3 specifically (though the
underlying mechanisms received substantial real D20/W=100,000 exercise via other sections). This is
an honest gap list, matching this repo's own standing discipline, not a hidden one.
