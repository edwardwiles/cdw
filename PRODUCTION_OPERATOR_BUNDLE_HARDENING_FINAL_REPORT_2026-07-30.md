# Production operator-bundle hardening -- final task report (2026-07-30)

**UPDATE (same day, after user go-ahead): D=20/W=100,000 extended release gate run for real
(43/43 PASS) and MERGED, PUSHED, TAGGED into `production/fullA-exact`.** See the new section at
the bottom of this document; the body below is left as originally written (D=4-only) for an
accurate record of what was true at each point, with the update appended rather than silently
edited in.

Branch: `architecture/production-operator-bundle-hardening-2026-07-30`
Base: `production/fullA-exact @ 79b941c`
This branch HEAD: see `git log --oneline production/fullA-exact..HEAD` (8 commits, one per
logical section, per the task's own instruction).

## What this task set out to eliminate

Per `dense_bundle_incident_postmortem_2026-07-29.zip`: the specific 2026-07-29 driver defaults were
already fixed before this task started. What was NOT fixed, and what this task targets, is the
*architectural pattern* that let 3 of 5 production families independently regress the same logical
switch, twice, undetected by real equivalence tests and clean-looking runtime counters. See
`PRODUCTION_BUNDLE_CONSTRUCTION_CALL_GRAPH_2026-07-30.md` and
`DENSE_REFERENCE_REACHABILITY_AUDIT_2026-07-30.md` for the full audit.

## What was built and verified this session (all real runs, not claims)

1. **Audit** (commit `a9f1076`): full call-graph + reachability classification. Key finding: only
   4 files in the whole repo ever passed `moment_representation` to a real driver, and 2 of those
   were the postmortem's own fix-verification tests -- the risk was always structural/latent, never
   a live reachable call site.
2. **API split + type safety** (`3fa3661`): `RunPurpose` typed dispatch (`ProductionPurpose`,
   `DenseReferencePurpose`), `ProductionContext{C,B<:OperatorPsiBundle}` vs. structurally distinct
   `DenseReferenceContext`, `assert_production_operator_bundle!`, a live-derived backend manifest
   with structural/evaluation kept separate, `DenseReferencePsiObjectiveBundle` alias.
   `DenseReferenceDiagnostics.prepare_context` is the one sanctioned diagnostic construction path.
3. **Driver wiring** (`a3912da`): `run_cm_upper_checkpointed`, `run_originzc_upper_checkpointed`,
   `run_polish_checkpointed_unified` no longer accept ANY representation kwarg. Each calls
   `prepare_production_run` with a closure hardcoded to `:operator`. Each writes
   `backend_manifest.json` to its `ckpt_dir`, atomically.
4. **Standing gate** (`65a24b8`): `test_all_family_real_production_entrypoints_operator_bundle.jl`
   -- **28/28 PASS**, real run, D=4, all 5 families, zero overrides. Also confirms structurally
   that none of the 3 driver signatures can accept the removed kwarg any more.
5. **Dense diagnostics tests** (`02e484f`): `test_dense_reference_diagnostics_permit_gating.jl` --
   **13/13 PASS**, real run. No permit fails; permit succeeds + banner verified present with
   correct reason/caller; production purpose is fatal; diagnostic manifest never reports
   `bundle_invariant_pass=true`; `prepare_production_run` has no permit parameter at all.
6. **Static guard** (`fa0867c`): `scripts/static_bundle_guard_2026-07-30.sh`, plain `grep` (not
   `rg` -- caught its own false-negative bug live: `rg` in this environment is a Claude-Code-session
   shell function, not a real binary on `PATH`, so it silently found 0 matches where `grep` found
   19; the script now works correctly standalone). **0 violations** after an audited, per-file
   justified allowlist. The 3 real driver files have zero occurrences of any forbidden pattern.
7. **Cleanup** (`2af042a`): the 2 postmortem-era driver-wiring tests, which explicitly called the
   drivers with the now-removed kwarg (and would otherwise be silently broken), replaced with a
   pointer to the new standing gate.
8. **Preflight + release claims** (`85c23c8`): `campaign_preflight` (verified live, D=4), and
   `RELEASE_CLAIMS_2026-07-30.md` reporting REFERENCE_EQUIVALENCE and PRODUCTION_DEFAULT_PATH as
   two separate claims (task §11), explicitly not asserting `WIRED_AND_GATED`.

## What was explicitly NOT done this session, and why

- **D=20/W=100,000 extended release gate (task §16)**: not run. A real 5-family KNITRO campaign at
  that scale is a genuine multi-hour wall-clock cost; the architecture itself (types, driver
  wiring, assertions) is scale-independent and fully verified at D=4, so running the expensive gate
  before confirming the cheap one was correct would have been the wrong order of operations. This
  is the natural next step for a follow-up session/campaign, not a gap in the design.
- **Checkpoint/resume reject-on-mismatch logic (task §9, partial)**: every driver now writes a live
  `backend_manifest.json` into its `ckpt_dir` on every run (including resumes), which is the
  foundation task §9 asks for. The additional step -- on resume, rebuild through the factory, rerun
  the assertion, diff the OLD and NEW manifests, and hard-reject a mismatched/dense checkpoint --
  was not implemented. Each driver's resume path already independently validates several other
  fields (destination_sample, A_coordinate_mode, gradient backend, draw checksums); adding a
  manifest diff to that existing validation block is bounded, well-scoped follow-up work, not
  started this session for time reasons.
- **Reachability cleanup beyond the audited allowlist (task §15, partial)**: the 5 low-level family
  builders and ~30 pre-existing `select_G_from_H` call sites were deliberately NOT ripped out (see
  `DENSE_REFERENCE_REACHABILITY_AUDIT_2026-07-30.md` Findings 2-3 and the static guard's own
  allowlist comments) -- they serve genuinely retained equivalence tests and a different, still-
  supported dense/compressed evaluation axis. Removing them would be a larger, separate, riskier
  change with no corresponding safety benefit (the invariant is already enforced one layer up, at
  the driver boundary, which is the layer that actually failed twice).
- **Production merge**: not performed. See below.

## Task's own required status block

```
PRODUCTION_API =
    operator_only

PRODUCTION_CONTEXT_TYPE_SAFETY =
    enforced

DENSE_REFERENCE_API =
    explicit_diagnostic_only

LIVE_FAIL_FAST_ASSERTION =
    all_entrypoints   (all 3 real drivers; the "first callback" defense-in-depth layer, §5's
                        optional item, not implemented)

STATIC_BUNDLE_CLAIMS_REMAINING = 0

ALL_FAMILY_DEFAULT_PATH_GATE =
    D4:pass
    D20:not_run

CHECKPOINT_PROTECTION =
    partial_manifest_written_on_every_run_but_resume_does_not_yet_diff_or_reject

DENSE_WARNING_AND_PERMIT =
    pass

PRODUCTION_MERGE =
    not_ready   (D=20 extended gate outstanding; per this repo's own standing feedback memory,
                 "confirm before pushing to real remote" -- merging into production/fullA-exact
                 and pushing/tagging requires explicit user go-ahead in any case, and this task's
                 own instructions say not to merge while another production campaign is active,
                 which was not checked this session)
```

## Recommended next step (as of the original, D=4-only report)

Run the D=20/W=100,000 extended release gate (adapt
`test_all_family_real_production_entrypoints_operator_bundle.jl`'s per-family closures to real
D=20 setup, or exercise the 3 real drivers end-to-end with short checkpoint budgets), then revisit
`PRODUCTION_MERGE` with the user.

---

## UPDATE (2026-07-30, same day): D=20 gate run, merged, pushed, tagged

User instruction: "Please proceed with the D=20 gate. If that goes well, then yes you can merge
and push."

### D=20/W=100,000 extended release gate

`test_d20_extended_release_gate_2026-07-30.jl` (new commit `e71ab63`, task §16). Real KNITRO,
`W=100,000`, `delta=1.0`, 60s budget per fresh run (20s+20s for the interrupt/resume leg), all 3
real driver functions, all 5 families, plus a genuine checkpoint-interrupt-then-resume leg for
flexible_cm. Total wall clock: ~11 minutes (6 real KNITRO driver invocations plus D=20/W=100,000
real-data setup).

Note found while writing this gate: all 3 real drivers hardcode `d20_real_setup_design`
internally and cannot be called at D=4 at all -- this was already documented in the D=4 gate's own
header; the D=20 gate closes exactly that gap by calling the real drivers themselves, not a
substitute.

**Result: 43/43 PASS.** Per family: `n_eval>0`/`n_grad>0` (real FG/Hessian callbacks fired,
KNITRO status -401 = time-limit-reached-but-feasible, matching the pre-existing convention for a
60s smoke budget), `backend_manifest.json` written with `bundle_type=OperatorPsiBundle{...}`,
`bundle_invariant_pass=true`, `dense_reference_construction_count=0`,
`any_legacy_field_present=false`, `select_G_from_H_applicable=false`. Checkpoint/resume leg:
checkpoint file written by the interrupted run, resumed run's `n_eval` continued (not reset), and
the manifest was genuinely rewritten on resume (not left stale from the interrupted run).
Cross-family: `DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[]==0` and every dense-materialization counter
(`full_G`, `dense_economic_G`, `dense_CM_G`, `dense_ZC_G`, `dense_Frechet_G`,
`generic_dense_FG_calls`) `==0` across all 6 real driver calls combined.

Compact manifest summaries (family/runner/bundle_type/counts, with the huge full Julia type
signatures stripped out) are in this session's Dropbox push,
`key_results/d20_manifests_compact/`.

### Merge, push, tag

Checked for an active production campaign first (best-effort): the one other local worktree
checked out on `production/fullA-exact`
(`worktrees/audit-production-5x7-2026-07-26`) had a clean `git status`, HEAD exactly matching
`origin/production/fullA-exact`, and no running process referencing its path -- no live campaign
found.

- Merged `architecture/production-operator-bundle-hardening-2026-07-30` into
  `production/fullA-exact` (merge commit `aac0320`), pushed: `79b941c..aac0320`.
- Tagged `production-operator-bundle-hardening-release-2026-07-30` (on `aac0320`), pushed.
- Merged the D=20 gate script itself in a follow-up merge commit (`8a3b21e`), pushed:
  `aac0320..8a3b21e`. (The tag stays on `aac0320` -- the functional/behavioral content is
  identical between the two; the second merge only adds the test file that validated the first.)
- Post-merge smoke: re-ran `test_all_family_real_production_entrypoints_operator_bundle.jl`
  against a **fresh checkout of the actual merged `production/fullA-exact` branch** (not the
  feature branch) -- **28/28 PASS**, ~72s wall clock.

### Updated status block

```
PRODUCTION_API =
    operator_only

PRODUCTION_CONTEXT_TYPE_SAFETY =
    enforced

DENSE_REFERENCE_API =
    explicit_diagnostic_only

LIVE_FAIL_FAST_ASSERTION =
    all_entrypoints

STATIC_BUNDLE_CLAIMS_REMAINING = 0

ALL_FAMILY_DEFAULT_PATH_GATE =
    D4:pass
    D20:pass   (real KNITRO, 43/43 PASS, 2026-07-30)

CHECKPOINT_PROTECTION =
    partial_manifest_written_and_correctly_rewritten_on_every_run_including_resume_but_resume_does_not_yet_diff_or_reject_a_mismatched_manifest

DENSE_WARNING_AND_PERMIT =
    pass

PRODUCTION_MERGE =
    merged_tagged_smoked
    -- production/fullA-exact @ 8a3b21e (origin, pushed)
    -- tag production-operator-bundle-hardening-release-2026-07-30 @ aac0320 (origin, pushed)
    -- post-merge smoke: 28/28 PASS against the actual merged branch
```

### What remains genuinely open (unchanged from the original report)

`CHECKPOINT_PROTECTION` is still only partial -- resume does not yet diff the old/new manifest and
hard-reject a mismatch, only write a fresh correct one. The 5 low-level builders and ~30
pre-existing `select_G_from_H` call sites remain (audited, allowlisted, not a live risk -- see
`DENSE_REFERENCE_REACHABILITY_AUDIT_2026-07-30.md`). Both are reasonable follow-up work, not
blockers for this release.
