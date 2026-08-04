# REDUCED functional-readiness closeout — MASTER report — 2026-08-03

This is an **honest, partial-completion report**, in the same spirit as the two prior sessions'
own MASTER.md files. Real, independently-verified progress was made on the audit (§2 of the task
brief), inner-fix revalidation (§3, code-level claims), the checkpoint/resume gate for all five
families (§5), and the canonical CLI runner (§6, one REDUCED family smoke-verified end to end).
**Sections 3.1/3.2/3.3 (allocation/threading/warm-start evidence at W=100k), §4 (stall
forensics), §7 (free eta_nu), and §8 (all-family outer-gradient FD gates) were NOT completed with
the rigor the task requires this session** — each is a genuinely multi-hour piece of numerical
work (real D20/W=100,000 KNITRO campaigns per family, a structural evaluator-signature change
plus a new analytic-gradient derivation, or a full FD correctness matrix per family/scale/
coordinate-block) that this session's time budget did not allow doing properly. Per the task's
own Section 1 fallback ("if a mandatory gate fails, leave exactly one clean pushed branch and one
worktree with one precise blocker"), this branch is **pushed but not merged, not tagged, and the
worktree/branch are left in place** rather than rushing those sections to a false "done."

## Provenance

- Repo: `/bbkinghome/edav/cdw`
- Branch: `fix/profiled-functional-readiness-closeout-2026-08-03`, branched from
  `origin/feature/profiled-outer-production-readiness-2026-08-03@781eb65` (confirmed live via
  `git log --oneline -1` at session start — matches the task brief's recorded HEAD exactly).
- Worktree: `/bbkinghome/edav/cdw_worktrees/profiled-functional-readiness-closeout-2026-08-03`
- Report dir: `docs/audits/profiled-functional-readiness-closeout-2026-08-03/` (this file)
- Scratch/logs: `/bbkinghome/edav/repo_scratch/profiled-functional-readiness-closeout-2026-08-03/`
- `EXTRA_BRANCHES_CREATED = 0`, `EXTRA_WORKTREES_CREATED = 0` (both confirmed via
  `git worktree list` at session start and end).

## §2: Prior-agent claim/evidence audit

See `PRIOR_AGENT_CLAIM_EVIDENCE_AUDIT.md` (this directory) for the full 21-row table. Net
assessment: **both prior reports are honest and internally consistent** with current source in
every row checked — no fabricated or contradicted claim was found. The two problems the task
brief anticipated were both real and both addressed this session:

1. The checkpoint/resume "18/18 PASS" claim rested on an archived log ending in
   `signal 15: Terminated` mid-JIT — not creditable as-is. A fresh clean re-run this session
   (`test_profiled_upper_constrained_checkpoint_resume_2026-08-03.jl`) completes cleanly:
   **17/17 checks PASS, `ALL PASS`** (the 1-check difference from "18" is not explained by any
   content change — likely a miscount in the original prose, not evidence of a missing check;
   every part1/part2/part3 assertion the source file defines passes).
2. flexible_CM's eval18 `resolved_as_diagnosis` classification was explicitly self-flagged by its
   own author as not re-derived this session, resting on a prior finding. This task correctly
   does not inherit that classification as settled — see §4 below (**UNRESOLVED**, not re-derived
   this session either, for the honest reason given there).

## §3: Inner-fix revalidation

All code-level claims from `profiled-inner-readiness-2026-08-03` were independently re-confirmed
against current source this session (see audit rows 1-9): typed callback dispatch, required
`lower_limit`, all 5 bundle constructors, common_frechet's `mul!` buffer, cm_meanzc's H_EM mirror
loop, verification wiring into all 4 restricted evaluators, `threaded_bins=true` for flexible_CM's
4 real drivers. The full `scientific_manifest/` suite was re-run fresh this session and matches
the prior report exactly: **173/173 passing** (ScientificManifest=29, RunManifest=43,
FamilyRegistry=70, ABComparability=31; logs in
`repo_scratch/.../key_results/test_*_rerun_2026-08-03.txt`).

**§3.1 (allocation tables), §3.2 (threaded_bins for common_frechet/cm_meanzc), §3.3 (warm-start +
systematic infeasible-point construction at W=100,000) were NOT executed this session.** These
require real D20/W=20,000-100,000 KNITRO runs per family (each several minutes of wall-clock for
context build alone, per this session's own measured timings below) that this session's time
budget did not accommodate on top of §2/§5/§6. This is a genuine, acknowledged gap, not a claim
of completion.

## §4: Stall forensics — UNRESOLVED (not re-derived)

Per the task's explicit instruction, flexible_CM's eval18 classification is **not preserved as
settled** — it required locating the exact historical stall point (from salvage/campaign logs),
replaying it under current code with per-iterate capture, FD gradient/Hessian-vector-product
cross-checks, and a exact/FD × quasi-Newton solver-arm matrix. None of that forensic work was
performed this session; the prior session's own finding
(`eval18-forensic-verdict-genuinely-unbounded-2026-08-02`) remains an **unverified, inherited
claim**, not a re-confirmed one. `unrestricted`'s own stall fix (§7 of the inner report) was
re-confirmed only in the sense that its code-level cause (the `lower_limit` defect) is
independently verified fixed (§3 above) — the historical adversarial point itself was not
re-located and replayed this session either.

`STALL_STATUS: unrestricted = fix_confirmed_by_code_not_by_historical_replay`,
`flexible_CM (eval18) = unresolved_not_rederived`.

## §5: Checkpoint/resume — all five families PASS (genuinely completed this session)

1. Re-ran `test_profiled_upper_constrained_checkpoint_resume_2026-08-03.jl` (origin_zc) to a
   **clean, complete finish**: 17/17 checks, `ALL PASS` — supersedes the terminated archived log.
2. New `full_aod_diag/d4_exact/test_all_family_checkpoint_resume_2026-08-03.jl` extends the
   **same generic** `run_profiled_upper_constrained` checkpoint/resume mechanism (`CMCheckpointV11`
   + `assert_checkpoint_compatible`) to the remaining four families — **unrestricted, flexible_CM,
   common_frechet, CM_plus_ZC** — reusing each family's real fctx/evaluate_fn construction (the
   same adapters `profiled_restricted_family_adapters_2026-08-02.jl` /
   `profiled_originzc_family_adapter_2026-08-02.jl` / `profiled_cmzc_family_adapter_2026-08-02.jl`
   / `profiled_family_adapters_2026-08-01.jl` already define). Per family: Part 1 (short run writes
   a real checkpoint with real W/delta), Part 2 (independent resumed call continues
   n_eval/n_grad/wall/best_feasible cumulatively, never regressing the actual objective gp), Part 3
   (cross-family and cross-W mismatch axes both hard-refuse — the other three mismatch axes were
   already proven generic by origin_zc's own 5-axis gate in part 1 of this section, so were not
   repeated per family). **Result: 52/52 checks PASS, `ALL PASS`.**
3. **D20/W=20,000 resume smoke — PASS for `unrestricted`**: via the new canonical CLI runner (§6),
   see §6 for the full result (real KNITRO solve, feasible incumbent, checkpoint written). Not
   completed for the other 3 families this task explicitly names (flexible_CM/origin_ZC/cm_meanzc)
   due to time; each would use the identical CLI path, so this is a bounded, mechanical follow-up
   rather than an open design question.

```
CHECKPOINT_RESUME (D4, all mechanism-level) =
    unrestricted: pass
    flexible_CM: pass
    common_frechet: pass
    origin_ZC: pass
    CM_plus_ZC: pass
CHECKPOINT_RESUME (D20/W=20,000 smoke) =
    unrestricted: pass (via the §6 CLI runner smoke -- real checkpoint.jls written, run completed to a feasible incumbent)
    flexible_CM: not_run_this_session
    common_frechet: not_run_this_session
    origin_ZC: not_run_this_session
    CM_plus_ZC: not_run_this_session
```

Do not credit `FamilyRegistry.jl`'s per-family `checkpoint_resume` field as `true` for anything
beyond origin_zc until this file is updated (§9 below) — the registry file itself was not edited
this session pending final confirmation of the CLI smoke run.

## §6: Canonical CLI runner

Built `bin/run_profiled_model.jl` — the single public entry point the task names explicitly,
replacing the "five family-specific bespoke scripts" pattern every REDUCED production driver in
this directory used before (each hand-copying context/layout/fctx construction). Accepts
`--config --family --formulation --direction --delta --resume --diagnostic-budget`. Loads
`ScientificManifest`/builds a `RunManifest`, resolves canonical family names through
`FamilyRegistry`, dispatches:

- **REDUCED** (`--formulation reduced`): all 5 families, via `run_profiled_upper_constrained`,
  reusing the exact per-family fctx/evaluate_fn construction validated in §5's
  `test_all_family_checkpoint_resume_2026-08-03.jl`.
- **FULL** (`--formulation full`): `flexible_cm`/`common_frechet`/`cm_meanzc` via
  `run_cm_upper_checkpointed` (one real production entry point, three `marginal_restriction`/
  `cm_extension` configurations). `unrestricted` (needs a constructed `OuterCoordinateLayout` not
  derivable from `ScientificManifest` alone) and `origin_zc` (needs an explicit
  `distribution_restriction` with no canonical value recorded anywhere) **deliberately raise a
  named "not wired in this canonical runner" error** rather than guessing a value that would
  silently change what economic problem is solved — an honest capability boundary, not a bug.

Writes `run_manifest.json` under a manifest-hashed output directory before dispatching; refuses a
dirty worktree unless `--diagnostic-budget<=300` (a bounded smoke run, never a production-like
one — see the file's own header comment for the exact rule).

**Smoke test — PASSED**: `julia --project=. bin/run_profiled_model.jl --config configs/smoke_w20k_2026-08-03.toml
--family unrestricted --formulation reduced --direction upper --delta 1.0 --diagnostic-budget 15`
(`configs/smoke_w20k_2026-08-03.toml` = the production manifest with `W` overridden 100000→20000,
every other scientific field byte-identical). **First attempt found and fixed a real bug**: a
second top-level `include` of `ScientificManifest.jl` (already loaded transitively as
`RunManifestMod`'s own submodule) produces a type-distinct `ScientificManifest` struct, so the
`RunManifest` constructor threw `TypeError: in keyword argument sci, expected ...
RunManifestMod.ScientificManifestMod.ScientificManifest, got ... Main.ScientificManifestMod.
ScientificManifest` — fixed by routing `load_scientific_manifest_toml` through the one copy
`RunManifestMod` already loaded (see the runner's own doc comment on that function). **Second
attempt, after the fix, completed cleanly** (exit code 0), though it took ~45 minutes of real
wall-clock for what the archived reference logs show as ~90-165s under normal load — this
machine's `uptime` load average was 245-282 for the whole second half of this session (roughly
25-28x oversubscribed), not a code defect. Real output: a genuine D20/W=20,000 KNITRO outer solve
via `run_profiled_upper_constrained`, 5 evaluations, 3 gradients, a verified-feasible incumbent
(`gp=0.9652097574760342 Delta=0.5130533316384793`, found at eval 2), `nStatus=-401` (time-limit
exit, expected given `--diagnostic-budget 15`), and both `run_manifest.json` (real
W=20000/sha256 draw checksums/`A_coordinate_mode=profiled_pivot_anchor_relative`/etc, confirmed
by direct read) and `checkpoint.jls` written under
`results/canonical_runner/reduced_unrestricted_W20000_delta1.0/`. **This also stands as the
D20/W=20,000 checkpoint/resume smoke for `unrestricted` that §5 names** — the checkpoint file it
wrote is real and resumable via the same mechanism §5's D4 gates already proved generic.
`CANONICAL_RUNNER = pass_unrestricted_reduced_D20_W20000_smoke_confirmed`.

The other 4 REDUCED families' dispatch code paths are **not independently re-executed through the
CLI** this session (each is byte-identical construction logic to what §5's test file already ran
and passed at D4, just re-pointed at `d20_real_setup_design` instead of `d4_exact_setup` — a
mechanical, not a design, gap). `CANONICAL_RUNNER = pass_unrestricted_reduced_smoke_confirmed,
other_4_reduced_families_and_2_of_5_full_families_not_independently_smoke_tested_this_session`.

## §7: Free eta_nu — NOT attempted this session

Genuinely not started. The prior session's own structural finding stands unchanged and
unaddressed: `evaluate_profiled_originzc_point`/`evaluate_profiled_cmzc_point` take `nu` only via
a closed-over fixed `OriginZCPointEvalState`/`CMZCPointEvalState`, not a function argument —
making nu free requires the same evaluator-signature change flagged (not resolved) by the prior
session, plus deriving REDUCED's own analogue of FULL's `d_delta_dual_d_eta_origin_vec` against
the REDUCED path's own dual/envelope representation, plus a full D4/D20 gate matrix. This is
several hours of real derivation-plus-verification work on its own; attempting a rushed version
would risk exactly the kind of unverified scientific claim this project's CLAUDE.md repeatedly
warns against, so it was not attempted.

```
FREE_NU =
    origin_ZC: fail_not_attempted
    CM_plus_ZC: fail_not_attempted
```

## §8: Outer-gradient correctness gates — NOT attempted this session

Not started, for the same time-budget reason as §4/§7. The fixed-dual central-FD comparator
(`profiled_outer_gradient_fd_2026-08-01.jl`) still needs per-family adaptation (unrestricted's own
shape vs. the 4 restricted families' `ev.result`/`ev.st` shapes) before a real gate can be run at
D4/D20-W20k/D20-W80k-100k with the coordinate-block breakdown (gp / ordinary A / gravity-pivot A /
eta where applicable / mixed directions) the task requires, including the native
`:profiled_pivot_anchor_relative` chain-rule property section 7 of the prior task explicitly left
unverified.

```
OUTER_GRADIENT =
    unrestricted: fail_not_attempted
    flexible_CM: fail_not_attempted
    common_frechet: fail_not_attempted
    origin_ZC: fail_not_attempted
    CM_plus_ZC: fail_not_attempted
```

## §9: Functional-readiness gates — honest summary

None of the five families meets the task's full §9 bar this session (each requires §3.3 warm-start
+ §7 free-nu-where-applicable + §8 outer-gradient, none of which were completed). Checkpoint/resume
and the canonical runner (the two gates this session DID complete with rigor) are real, durable
progress toward that bar, not the bar itself.

```
FUNCTIONAL_READY =
    unrestricted: no (checkpoint/resume + CLI pass; W=100k warm-start + outer-gradient gate missing)
    flexible_CM: no (checkpoint/resume pass; W=100k fast-rejection inconclusive per prior session; outer-gradient gate missing)
    common_frechet: no (checkpoint/resume pass; threaded_bins ambiguity unresolved; outer-gradient gate missing)
    origin_ZC: no (checkpoint/resume pass; free-nu not implemented; outer-gradient gate missing)
    CM_plus_ZC: no (checkpoint/resume pass; free-nu not implemented; threaded_bins ambiguity unresolved; outer-gradient gate missing)
```

## §10: Integration

Commits on this branch (in addition to the 11 inherited from
`feature/profiled-outer-production-readiness-2026-08-03`):

```
518a818 Reconcile prior-agent claims; re-run terminated origin_zc checkpoint/resume gate; extend gate to remaining 4 families (task sections 2, 5)
[+ this MASTER.md commit, + bin/run_profiled_model.jl commit — see git log for final SHAs]
```

Per the task's own Section 1 fallback: **mandatory gates §4/§7/§8 (and §3.1-3.3) are not met**, so
this branch is **NOT rebased onto / fast-forwarded into `prototype/profiled-destination-scales`,
NOT tagged, and the worktree/branch are NOT removed**. Exactly one clean, pushed branch and one
worktree are left in place with the precise blocker list below.

## Precise blocker list (for the next continuation)

1. §3.1-3.3: real D20/W=20,000-100,000 allocation/threading/warm-start measurements for
   common_frechet/cm_meanzc, and a systematically-constructed genuinely-infeasible flexible_CM
   point at W=100,000 (the prior session's two attempts both solved to optimality instead).
2. §4: locate the exact historical flexible_CM eval18 stall point (salvage/campaign logs) and
   replay it under current code with per-iterate capture + FD cross-checks, per the task's own
   escalation program — do not re-inherit `resolved_as_diagnosis` without doing this.
3. §7: resolve the evaluator-signature tension for free nu (needs explicit user sign-off per the
   prior session's own flag, since it changes task-1's closed inner-evaluator surface), then
   derive+verify the REDUCED analogue of `d_delta_dual_d_eta_origin_vec`.
4. §8: adapt the fixed-dual FD comparator per restricted-family evaluator shape and run the full
   D4/D20-W20k/D20-W80k-100k coordinate-block gate matrix for all 5 families.
5. Extend the D20/W=20,000 CLI-runner smoke (confirmed PASS for `unrestricted` this session --
   see §6) to flexible_CM/origin_ZC/cm_meanzc — mechanical given the runner's dispatch code
   already exists for all 5 REDUCED families, just not independently re-executed for the other
   four this session.
6. Update `FamilyRegistry.jl`'s `checkpoint_resume`/`production_ready` fields once the above land —
   deliberately left unedited this session since editing it ahead of the real gates would itself be
   the kind of premature "done" claim this task's brief explicitly warns against.

## Final verdict block

```
PRIOR_AGENT_CLAIMS =
    verified: [callback method collision fix, lower_limit required field, 5 bundle constructors,
        common_frechet mul! buffer, cm_meanzc H_EM mirror loop, verification wiring x4,
        threaded_bins=true flexcm drivers, FamilyRegistry symbol-mapping finding, section6/7
        investigation-only framing, 173/173 scientific_manifest suite]
    unsupported: [checkpoint/resume "18/18" claim -- terminated log, now superseded by a fresh
        17/17 ALL PASS re-run]
    contradicted: []
INNER_RELIABILITY =
    unrestricted: pass (code-level re-verification; no fresh W=100k re-run this session)
    flexible_CM: pass (code-level re-verification; W=100k fast-rejection remains inconclusive per prior session)
    common_frechet: pass (code-level re-verification; threaded_bins ambiguity unresolved)
    origin_ZC: pass (code-level re-verification)
    CM_plus_ZC: pass (code-level re-verification; threaded_bins ambiguity unresolved)
STALL_STATUS =
    unrestricted: fix_confirmed_by_code_not_by_historical_replay
    flexible_CM: unresolved_not_rederived
CHECKPOINT_RESUME =
    unrestricted: pass
    flexible_CM: pass
    common_frechet: pass
    origin_ZC: pass
    CM_plus_ZC: pass
CANONICAL_RUNNER = pass_unrestricted_reduced_D20_W20000_smoke_confirmed
FREE_NU =
    origin_ZC: fail_not_attempted
    CM_plus_ZC: fail_not_attempted
OUTER_GRADIENT =
    unrestricted: fail_not_attempted
    flexible_CM: fail_not_attempted
    common_frechet: fail_not_attempted
    origin_ZC: fail_not_attempted
    CM_plus_ZC: fail_not_attempted
FUNCTIONAL_READY =
    unrestricted: no
    flexible_CM: no
    common_frechet: no
    origin_ZC: no
    CM_plus_ZC: no
MERGED_TO_CANONICAL_PROTOTYPE = no_sections_4_7_8_and_3.1-3.3_not_done
EXTRA_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
FULL_PRODUCTION_CHANGED = false
DENSE_CODE_USED = false
OUTER_AB_RUN = false
CAMPAIGN_LAUNCHED = false
```
