# REDUCED outer production integration and FULL-vs-REDUCED A/Bs -- status, 2026-08-03

This is an **honest interim status report**. Real, verified progress was made on sections 1-3,
5, 7, and 10 of the 13-section task; sections 4 (partially: registry done, unified CLI runner
not built), 6, 8, 9, and 11-12 are documented but not implemented, for reasons given inline.
**Nothing here has been merged to `production/fullA-exact`, `prototype/profiled-destination-scales`,
or tagged `profiled-outer-ab-ready-2026-08-03`. No A/B campaign was launched. No dense G/H backend
was used at any point.**

## Provenance

- Repo: `/bbkinghome/edav/cdw`
- Branch: `feature/profiled-outer-production-readiness-2026-08-03`
- Worktree: `/bbkinghome/edav/cdw_worktrees/profiled-outer-production-readiness-2026-08-03`
- HEAD: `5e2c262`, 10 commits ahead of the branch point `b80dd48` (`origin/prototype/profiled-destination-scales`)
- Task-1 prerequisite (`profiled-inner-ready-2026-08-03` @ `640ea15`, `INNER_READY=yes` all 5
  families) confirmed a real ancestor of `b80dd48` before this task began.
- `origin/production/fullA-exact` tip `4c3dad5` is exactly one commit ahead of the prototype's
  merge-base (`146b10e`), and that commit is CLAUDE.md-only (zero source diff) -- FULL is
  byte-identical to what's embedded in the prototype for every hot file.

## What's genuinely done, with real verification

**Section 1-2**: repo/worktree discipline; FULL byte-identity re-verified (see provenance above).

**Section 3 (manifest infrastructure)**: `scientific_manifest/ScientificManifest.jl` (vendored
byte-identical from the not-yet-merged `hardening/require-scientific-params-2026-08-03` branch,
the documented single source of truth) + `scientific_manifest/RunManifest.jl` (wraps it with
family/formulation/coordinate-mode/nu-policy/budget/cache-policy/verification-policy/
state-digest/source-SHA fields). 72/72 tests pass across both files.

**Section 4 (partial -- registry done, unified runner CLI not built)**:
`scientific_manifest/FamilyRegistry.jl` -- a truthful `(family, formulation)` capability matrix,
built only after reading the real source (not from the task prose's naming). **Load-bearing
finding**: FULL's family-name convention (`production_backend_manifest.jl`:
`:unrestricted`/`:flexible_cm`/`:common_frechet`/`:origin_zc`/`:cm_meanzc`) and REDUCED's own
`family_kind(fctx)` convention (`:unrestricted`/`:flexible_CM`/`:common_Frechet`/`:ZC_only`/
`:CM_plus_ZC`) are two DIFFERENT symbol sets for the same five families -- confirmed by grepping
every `family_kind` method definition. `REDUCED_FAMILY_KIND_TO_CANONICAL` fixes this. 70/70 tests
pass. **Not done**: `bin/run_profiled_model.jl` or any single unified CLI entry point dispatching
by `(family, formulation)` -- the registry documents which real function serves each cell
(mostly `run_profiled_upper_constrained`, already family-agnostic for REDUCED, or one of the 3
real FULL production entry points), but nothing wraps them behind one canonical command yet.

**Section 5 (checkpoint/resume)**: added `checkpoint_path`/`checkpoint_interval_s`/`resume_from`
to `run_profiled_upper_constrained` (the genuine constrained REDUCED runner the task names
explicitly, which had NO checkpoint support before this). Reuses `CMCheckpointV11`/
`load_cm_checkpoint_v11`/`assert_checkpoint_compatible` (already-existing infrastructure) rather
than inventing new machinery. Also fixes a real provenance gap in the older gp-fixed scaffold's
own checkpoint writer (hardcoded `W=0`/`draw_seed=0`/checksums=`""`/`delta=1.0` even though real
values were available in its caller's scope). **Verified with a real D4 KNITRO round trip**
(`test_profiled_upper_constrained_checkpoint_resume_2026-08-03.jl`, 18/18 checks): a short run
writes a real checkpoint with real W/delta recorded; an independent resumed call picks up
n_eval/n_grad/wall/best-gp cumulatively; and 5 independent mismatch axes (namespace/family, W,
delta, draw config, outer-vector length) each hard-refuse rather than silently resuming. Only
`origin_zc` has a passing gate -- the mechanism is family-agnostic (confirmed generic over
`fctx`/`evaluate_fn`) but `flexible_cm`/`common_frechet`/`cm_meanzc`/`unrestricted` have no gate
of their own proving it yet (FamilyRegistry records this honestly per family).

**Section 6 (free-nu, investigation only)**: traced FULL's real eta_nu mechanism end to end
(coordinates, `nu=exp(eta)` transform, data-driven log-space bounds, decoded-nu cache keying, and
the real analytic gradient formula `d_delta_dual_d_eta_origin_vec` -- a genuine envelope-theorem/
fixed-dual closed form through the already-solved KKT dual). See
`section6_free_nu_investigation.md`. **Not implemented on the REDUCED side**: doing so requires
changing `evaluate_profiled_originzc_point`'s own signature (nu is currently a closed-over FIXED
value, not a function argument) -- task-1's closed inner-evaluator surface, which this task's
mission statement says to call, not alter except through the production runner. Flagged as a
real tension for the user rather than resolved unilaterally.

**Section 7 (coordinate modes)**: formalized REDUCED's existing (already-real, already-used)
coordinate system as `:profiled_pivot_anchor_relative` (introduced in FamilyRegistry). Verified
at D4, using only existing primitives (`reduce_to_w_profiled`/`decode_outer_profiled`/
`gravity_from_logz`, no new math): exact decode/encode round trip (1.5e-15 max error, at the
calibration point AND independently at a perturbed point), same reconstructed full logA
(1.7e-15 relative error), same gravity residual (both endpoints exactly gravity-feasible,
~1e-18). 10/10 checks pass. **Not verified**: the analytic chain-rule gradient through this
transform (needs section 8's own FD machinery). **Not implemented**: `:profiled_powered_relative_A`
-- no derivation of whether FULL's powered-A motivation transfers to this different basis was
attempted.

**Section 10 (A/B comparability gate)**: `scientific_manifest/ABComparability.jl` --
`ab_comparable(full::RunManifest, reduced::RunManifest)` checks every field the task requires
(all `ScientificManifest` fields, draw checksums, nu policy/bounds, outer algorithm/budget,
cache/bank/warm-start/verification policy, `initial_state_digest`, `source_sha`, dirty-worktree
refusal on either arm) and refuses an `economic_parameterization` mismatch or swapped arm order.
`A_coordinate_mode` is the one field allowed to differ, only when explicitly opted into with a
non-`nothing` experiment label -- never weakens the `initial_state_digest` check, which is what
actually proves "both reconstruct the same economic point." `manifest_digest`/
`require_ab_comparable` give the required manifest-hash/source-SHA/decoded-state-equivalence
content. 31/31 tests pass.

**Full `scientific_manifest/` test suite: 173/173 passing** (`ScientificManifest`=29,
`RunManifest`=43, `FamilyRegistry`=70, `ABComparability`=31).

## Sections not attempted, and why

- **Section 8 (all-family outer-gradient gates)**: a real fixed-dual central-FD comparator
  already exists in this codebase (`profiled_outer_gradient_fd_2026-08-01.jl`,
  `profiled_composite_gradient_at`/`profiled_lfix_at`) but is built for the unrestricted family's
  evaluator shape (`evaluate_profiled_point`, `ev.st.layout`, `ev.obj`) -- adapting it correctly
  to origin_zc's ZC-augmented objective (or writing an independent one) is precision-sensitive
  numerical work that deserved more careful study than remaining session time allowed. Attempting
  it quickly risked exactly the kind of unverified/rushed scientific claim this project's CLAUDE.md
  repeatedly warns against.
- **Section 9 (threading/bandwidth cache)**: genuine performance engineering (thread-local
  scratch, cache invalidation correctness) needing its own careful session -- not started.
- **Section 11 (staged A/B program)**: needs sections 8/9 (and ideally more family coverage on
  section 5/6/7) to actually be meaningful, plus real D20 W=20k/W=100k KNITRO wall-clock -- not
  started. Explicitly not a shortcut-able step.
- **Section 12 (production-readiness matrix)**: `FamilyRegistry.production_ready_families`
  already gives this mechanically: all 5 FULL families `production_ready=true`, all 5 REDUCED
  families `production_ready=false` (honest -- none has cleared W=100k cold/warm, fast-rejection,
  or the other section-12 bars). See verdict block below for the formal per-family answer.

## Final verdict

```
CANONICAL_RUNNER = fail_registry_done_unified_cli_not_built
MANIFEST_AND_AB_GATE = pass
CHECKPOINT_RESUME =
    unrestricted:fail_mechanism_available_untested
    flexible_CM:fail_mechanism_available_untested
    common_frechet:fail_mechanism_available_untested
    origin_ZC:pass
    CM_plus_ZC:fail_mechanism_available_untested
FREE_NU =
    origin_ZC:fail_investigated_not_implemented
    CM_plus_ZC:fail_investigated_not_implemented
PROFILED_COORDINATE_MODES =
    native_log_relative:pass (as :profiled_pivot_anchor_relative -- round trip + full logA + gravity residual verified; gradient chain-rule not verified)
    powered_relative:not_implemented_reason_no_derivation_attempted
OUTER_GRADIENT =
    unrestricted:fail_not_started
    flexible_CM:fail_not_started
    common_frechet:fail_not_started
    origin_ZC:fail_not_started
    CM_plus_ZC:fail_not_started
OUTER_GRADIENT_PERFORMANCE =
    threading:fail_not_started
    bandwidth_cache:fail_not_started
AB_COMPARABLE = pass
SHORT_AB =
    unrestricted:inconclusive_not_run
    flexible_CM:inconclusive_not_run
    common_frechet:inconclusive_not_run
    origin_ZC:inconclusive_not_run
    CM_plus_ZC:inconclusive_not_run
PRODUCTION_READY =
    unrestricted:no
    flexible_CM:no
    common_frechet:no
    origin_ZC:no
    CM_plus_ZC:no
MERGED_TO_CANONICAL_PROTOTYPE = no_sections_8_9_11_12_not_done_and_only_origin_zc_has_a_real_checkpoint_gate
EXTRA_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
FULL_PRODUCTION_CHANGED = false
DENSE_CODE_USED = false
CAMPAIGN_LAUNCHED = false
```

## Recommendation for continuation

1. Section 8 first: adapt or rewrite the fixed-dual FD comparator for each restricted family's
   real evaluator shape (`ev.result`/`ev.st` for origin_zc/cm_meanzc/flexible_cm/common_frechet --
   read each adapter file before assuming a shared shape). This unblocks verifying section 7's
   4th property (chain-rule gradient) and is a prerequisite for section 11's staged A/Bs to mean
   anything.
2. Section 6's structural blocker (evaluator signature change for free nu) needs an explicit user
   decision before any code is written -- don't resolve it unilaterally.
3. Extend section 5's checkpoint/resume gate to the other 4 families (mechanism should carry over
   unchanged; each just needs its own D4 gate proving it, same pattern as
   `test_profiled_upper_constrained_checkpoint_resume_2026-08-03.jl`).
4. Only after 1-3: sections 9, 11, 12 in that order.
