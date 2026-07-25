# Final allocation/Hessian production release — 2026-07-25

**SHA note**: this document and `POST_MERGE_PRODUCTION_SMOKE_2026-07-25.md` were drafted against
release tip `e061134`; one purely-additive commit (this deliverable-doc set itself) landed after,
making `6f1d2c537d285d78cc0500e1c3ce0bfb2fbfa004` the actual final `production/fullA-exact` HEAD
(verified ancestor of both the local and `origin` branch, both re-fast-forwarded and re-tagged to
this final commit — see `PROVENANCE.txt` in the pushed deliverable package for the authoritative
final SHAs/tag targets). No code or test content differs between `e061134` and `6f1d2c5` — only
these docs were added.

Master summary for the selective productionization of
`port/production-allocation-and-hessian-optimizations-2026-07-25` (base `b7435ee`, feature HEAD
`588adf3`) and `audit-unrestricted-allocation-gap-2026-07-25` (snapshot `2a555fb`) onto canonical
`production/fullA-exact`.

**Release states used throughout, per the task's own definitions** — `IMPLEMENTED_ON_FEATURE_BRANCH`
→ `WIRED_IN_PUBLIC_PRODUCTION_DRIVER` → `VALIDATED_THROUGH_PUBLIC_ENTRY_POINT` →
`FAST_FORWARDED_TO_CANONICAL_PRODUCTION` → `TAGGED` → `POST_MERGE_SMOKE_PASSED`. "Merged" is used
below **only** for commits verified as ancestors of `production/fullA-exact` with a production
tag — never loosely for "wired on a branch."

## What happened

1. Confirmed `production/fullA-exact` was still at `b7435ee` (unmoved since the feature branch
   forked) — verified fresh both at session start and immediately before the final fast-forward.
2. Created `port/final-allocation-hessian-production-release-2026-07-25` off `b7435ee` and
   cherry-picked 15 of the source branch's 19 commits, **in original chronological order**,
   explicitly excluding `ad6f701` ("Force explicit outer KNITRO algorithm... guard against auto")
   — the task's own §1.3 lists a forced outer-algorithm change as out of scope. That functionality
   was rebuilt as an opt-in `pin_outer_algorithm::Bool=false` kwarg instead (default `false` =
   zero behavior change vs. today's production; `true` used only by this session's own matched
   benchmark harnesses for controlled A/B comparisons).
3. Built the central, machine-readable production backend manifest (task §2) and wired it into
   all three public checkpointed drivers, right after their existing startup banners.
4. Wrote public entry-point backend assertion tests that call the real drivers (not helpers) —
   **11/11 passing** against real licensed KNITRO.
5. Ran the required benchmark matrix at real D=20/W=80,000 scale, bounded per this session's own
   disclosed scope reductions (never silently narrowed):
   - Unrestricted BLAS sweep at P0/P1/P2 (15 real runs) — fills the source branch's own
     disclosed P0-only gap.
   - CM threaded-Architecture-C sweep at P0/P1/P2 (9 real runs, plus 2 point-harvest runs) —
     confirms the already-committed `threaded_bins=true` default beyond the single hard point
     the source branch measured.
   - Origin-ZC bounded BLAS benchmark (11 real runs) — confirms Architecture A remains
     appropriate; no Hessian-backend change made.
   - Matched 300-second before/after, both required families, against a genuinely unmodified
     `b7435ee` checkout in an isolated worktree.
6. Filled the source branch's disclosed correctness-gate gaps: D=4 rectangular with a non-last
   omitted destination (39/39 PASS), D=20 rectangular screen behavior (14/14 PASS), direction-bounds
   smoke (26/26 PASS), CM+meanZC/origin-ZC K=1 real-point gate (15/15 PASS). **0 failures across
   every gate run this session.**
7. Fast-forwarded `production/fullA-exact` to this branch's tip, verified ancestry, created and
   pushed 4 production tags, pushed the branch to `origin` (fast-forward, no force) — **with
   explicit user confirmation before the push**, per this session's own default caution around
   actions visible to others.
8. Ran post-merge smokes on all three public drivers against the now-canonical commit.

## Verdicts

```text
UNRESTRICTED_PREALLOCATION = MERGED
CM_COMPRESSED_CORE         = MERGED
CM_THREADED_HESSIAN        = MERGED
UNRESTRICTED_BLAS_DEFAULT  = unchanged (opt-in kwarg only; BLAS=8 validated at P0/P1, NOT hard-defaulted -- P2 evidence non-monotone, see UNRESTRICTED_BLAS_DEFAULT_SELECTION_2026-07-25.md)
CM_THREAD_DEFAULTS         = threaded_bins=true (confirmed P0/P1/P2, no regression); blas_threads left ambient/opt-in
ORIGIN_ZC_HESSIAN          = retained_architecture_a
HVP                        = not_triggered_for_implementation (trigger condition measured 69.09% on the source branch; not re-measured this session; HVP itself explicitly out of scope for this release per task §10)
POST_MERGE_SMOKE           = pass
```

All four items above marked `MERGED` are verified via, at the time of writing:

```bash
git merge-base --is-ancestor <sha> production/fullA-exact        # local: confirmed
git merge-base --is-ancestor <sha> remotes/origin/production/fullA-exact  # post-push: confirmed
git tag --contains <sha>   # unrestricted-preallocation-production-ready-2026-07-25,
                            # cm-compressed-core-production-ready-2026-07-25,
                            # threaded-exact-hessian-production-ready-2026-07-25,
                            # allocation-hessian-production-release-2026-07-25
```

for `<sha> = e061134533464e4921397168b1630cc8032870ac` (the release branch tip / new
`production/fullA-exact` HEAD).

## Component reports

- `PRODUCTION_BACKEND_MANIFEST_2026-07-25.json` — resolved manifest, all 4 families.
- `PUBLIC_ENTRY_POINT_BACKEND_ASSERTIONS_2026-07-25.md` — 11/11 pass detail.
- `UNRESTRICTED_PREALLOCATION_RELEASE_2026-07-25.md`
- `CM_COMPRESSED_CORE_RELEASE_2026-07-25.md`
- `CM_THREADED_ARCHC_RELEASE_2026-07-25.md`
- `UNRESTRICTED_BLAS_DEFAULT_SELECTION_2026-07-25.md`
- `ORIGIN_ZC_HESSIAN_DIAGNOSIS_2026-07-25.md`
- `MATCHED_300S_BEFORE_AFTER_2026-07-25.md`
- `POST_MERGE_PRODUCTION_SMOKE_2026-07-25.md`
- `key_results/` — CSVs and trimmed summary logs backing every headline number above.
- `PROVENANCE.txt`, `MANIFEST_SHA256.txt` (this package).

## What is honestly NOT in this release

- **HVP**: not implemented, not re-triggered this session (source branch measured 69.09% Hessian
  share on a hard point after threading; not reproduced here, given HVP is explicitly out of
  scope for this release regardless per task §10).
- **`omit_row_context_reuse`**: identified as the single highest-leverage remaining unrestricted
  allocation fix (~80% of a call's allocation is one-time context setup, not hot-path) but not
  implemented — flagged as `future_task` per task §9's own instruction not to delay this release
  for it.
- **A `pin_outer_algorithm=false` "after" arm for CM was added on review** (a `BENCH_PIN_OUTER_ALGORITHM`
  toggle on the CM harness, plus a second 300s run) after the first pass's confounded result was
  questioned — see `MATCHED_300S_BEFORE_AFTER_2026-07-25.md`'s "Second CM pass" for the clean
  result (allocation −49.8%, n_eval +45.5%, essentially identical best answer). **The equivalent
  unrestricted arm was not run** — unrestricted's own 300s table still uses `pin_outer_algorithm=true`
  on "after" only, carrying the same theoretical (unverified) confound; flagged explicitly in that
  doc's reconciliation section rather than left as a silent asymmetry.
- **Repeated trials**: every benchmark in this release is a single real run per configuration
  (real, reproducible via recorded seeds/fixtures), not averaged over repeats.
- **A genuine forced exact-tie exercise**: the `TiedWinnerError` fallback mechanism's type/catch
  behavior was confirmed reachable by source inspection but not exercised via an actual
  constructed tied point this session.
