# Inherited Remediation Commit Review — 2026-07-26

## Release-state ladder used below

`INHERITED_WORK_AUDITED` → `IMPLEMENTED_ON_FEATURE_BRANCH` → `VALIDATED_ALL_FAMILIES` →
`MATCHED_AB_PASSED` → `FAST_FORWARDED_TO_CANONICAL_PRODUCTION` → `TAGGED` →
`POST_MERGE_SMOKE_PASSED` → `FINAL_PROFILE_COMPLETED`. "Merged" is reserved for state 5+.

## 0. Canonical state check

- `production/fullA-exact` HEAD = `f1fa8e770759c62b3f96c1024dd310f235ea463e`, identical to
  `origin/production/fullA-exact` (same SHA, same commit date `2026-07-26 13:06:52 -0400`).
  **Production has not advanced since the remediation branch's stated base** — no rebase is
  mechanically required, only re-gating.
- Working tree at `/bbkinghome/edav/cdw` (the shared clone) is currently dirty on an unrelated
  branch (`port/fixed-frechet-cdf-production-2026-07-25`, in-progress Fréchet-CDF port work, not
  touched by this session). New work was done in a fresh `git worktree` at
  `gravity_robustness/worktrees/finish-five-family-optimization-stack-2026-07-26` on new branch
  `port/finish-five-family-optimization-stack-2026-07-26`, based directly on
  `production/fullA-exact`.
- Public drivers present at base: `c10_d20_production_driver.jl` (legacy unrestricted, both
  profile and unified-polish entry points), `c10_d20_production_driver_unified.jl`,
  `c10_d20_production_driver_flexible_theta_A.jl`, `cm_checkpoint.jl` (`run_cm_upper_checkpointed`,
  flexible-CM + common-Fréchet share this entry via basis selection), `cm_originzc_checkpoint.jl`
  (`run_originzc_upper_checkpointed`), `unrestricted_stage_runner.jl`.
- Active backend manifest at base: `production_backend_manifest.jl` resolves per-family backend
  labels; no Fréchet-specific structured resolver yet (fixed by inherited Phase F item 1).

The inherited branch (`port/remediate-production-5x7-audit-2026-07-26`, HEAD `be434bc`, 12 commits
on top of the same `f1fa8e7` base) was reviewed commit-by-commit below. **Its own session report is
internally consistent with the actual diffs** (spot-checked commit messages/diffstats against
`PRODUCTION_5X7_REMEDIATION_SESSION_MASTER_REPORT_2026-07-26.md`) — but its self-reported gate
scope was **taken at face value for the classification below only after independently checking each
commit's actual default values in the diff**, not assumed from the prose. One material discrepancy
was found this way (item 5 below).

## Commit-by-commit classification

| # | Commit | Item | Classification | Why |
|---|---|---|---|---|
| 1 | `2c4dceb` | Freeze static-audit baseline doc | **ADOPT_UNCHANGED** | Docs only, zero code diff. |
| 2 | `38e4c46` | Phase A: unrestricted → unified direct-bound driver | **ADOPT_AFTER_MORE_GATES** | Report's own gate covered decode/eval/gradient equivalence + one CLI smoke + migration-refusal smoke — narrower than task §1.1's full list (direct upper/lower semantics, transformed-A default, legacy-z explicit mode, checkpoint/resume, startup manifest, resolved outer algorithm all independently exercised). Re-gated below (§1.1). |
| 3 | `55c294a` | Phase B1: CM lookup-FG kernel, opt-in | **ADOPT_AFTER_REBASE** | Default kwarg confirmed `inner_fg_backend=:dense_reference` (byte-identical unless overridden) — genuinely opt-in, low blast radius. D=4 + real D=20/W=80,000 gates in the diff are real. Adopt as-is; production-flip decision stays `AVAILABLE_BUT_NOT_DEFAULT` per report's own honestly-reported allocation regression (+12.6%), which fails task §5.6's flip bar. |
| 4 | `993362d` | Phase C: exact-point cache, all 4 families | **ADOPT_AFTER_MORE_GATES** | Report's own text says its gate was **D=4 only** ("Gate (D=4, real `archC_verified_state` call path)"). Task §1.2 explicitly requires the D=20/W=80,000 public-driver 5-point sequence (A, grad-at-A, repeat-A, B, back-to-A) with `same_point_inner_resolves=0` for **all four** restricted families. Not yet done for common-Fréchet, CM+ZC, or origin-ZC at D=20. Re-gated below. |
| 5 | `c13f129` | Phase D: restricted dual bank, distance-only | **REWORK** (default flip reverted; implementation kept) | **Finding**: the commit sets `use_dual_bank::Bool = true` as the *default* on both real public entry points (`cm_checkpoint.jl:614` → `run_cm_upper_checkpointed`, `cm_originzc_checkpoint.jl:459` → `run_originzc_upper_checkpointed`), not opt-in — despite the report's own text explicitly disclosing this bank is distance-only (no KKT/residual proxy) and "validated only on a tiny D=4 two-point sequence." This is exactly the situation task §2 requires `KEEP_OPT_IN` for. The `RestrictedDualBank` implementation itself (`cm_dual_bank_production.jl`) is sound and worth keeping, but the default must revert to `false` until real-trajectory evidence exists (§2 below). Classified REWORK, not DROP, because only the default value needs to change — the code path is otherwise adoptable. |
| 6 | `1a58aa9` | Phase E pt.1: CompressedFactualWorkspace, all 4 families | **ADOPT_AFTER_MORE_GATES** | Report's gate: "D=4, byte-exact agreement" only. Task §1.3 requires D=20 full-family tests for all four restricted families **including a real origin-ZC context** — report explicitly discloses origin-ZC was *not* reached ("An equivalent origin-ZC test was attempted... set aside given time budget"). This is the single largest verification gap inherited from the prior session. Re-gated below, with origin-ZC closed. |
| 7 | `879c8bb` | Bugfix: `CMBinHessCtx` missing field in CM+ZC constructor | **ADOPT_UNCHANGED** | Pure correctness fix for a real crash (CM+ZC construction raised `MethodError` since commit 3). No behavior change beyond making CM+ZC constructible again. Independently re-verified below that CM+ZC now constructs. |
| 8 | `41e577c` | Phase E pt.2: dense-vs-bin-fill benchmark + gap tests | **ADOPT_UNCHANGED** | Analysis/tests only, D=20/W=80,000, no production default touched; conclusion (keep CM+ZC dense columns) is evidence-based per the task's own "don't change representation without benchmark evidence" rule. |
| 9 | `3c8942d` | Phase F: Fréchet manifest resolver + CM+ZC congruence-label fix | **ADOPT_UNCHANGED** | Reporting-only; D=4 gated both contrasts. |
| 10 | `1fd8674` | Phase F: origin-ZC docstring fix | **ADOPT_UNCHANGED** | Comment-only. |
| 11 | `c30e252` | Phase F: fail-fast `price_cache_backend` validation | **ADOPT_UNCHANGED** | Turns a latent deep `MethodError` into a clear call-time error; D=20 gated. |
| 12 | `be434bc` | Phase F consolidated write-up | **ADOPT_UNCHANGED** | Docs only. |

## Untracked scratch/debug files in the inherited worktree

`full_aod_diag/d4_exact/logs/`, `test_phaseC_load_smoke.jl`, `test_phaseD_load_smoke.jl`,
`test_phaseE_frechet_originzc_end_to_end.jl` — confirmed superseded iteration artifacts per the
inherited session's own report. Not copied onto this branch; the working origin-ZC end-to-end gate
this session needs is written fresh (§1.3) rather than resurrecting the abandoned draft.

## Net effect of this review

- 8 of 12 commits adopt cleanly (`ADOPT_UNCHANGED` / `ADOPT_AFTER_REBASE`).
- 3 commits need additional D=20/all-family gates this session was not yet given credit for
  (Phase C exact cache, Phase E workspace, Phase A driver repoint — re-gated in §1 below).
- 1 commit (Phase D dual bank) had a **default-safety regression relative to the task's own stated
  policy** — caught by diffing actual kwarg defaults rather than trusting the commit's prose
  summary, consistent with this project's standing "verify hot-path claims, don't trust grep /
  don't trust the report" pattern. Reworked below before adoption.
