# Phase 0 — Provenance and Branch Discipline (finish-operator-stack task, 2026-07-26 continuation)

## Provenance

- Canonical remote: `cdw` = `git@github.com:edwardwiles/cdw.git` (this is "the cdw repo" per
  `reference-cdw-repo-going-forward` memory; the `habibiscoding/Trade-Model-Robustness` origin seen
  on some sibling checkouts is legacy and not used here).
- `git fetch origin production/fullA-exact --tags` run this session: **no change**.
  `production/fullA-exact` HEAD = `f1fa8e770759c62b3f96c1024dd310f235ea463e`
  ("Port transformed-A coordinate to all four restricted-family drivers (opt-in)").
  `origin/production/fullA-exact` is identical (0 commits ahead/behind).
- **Production has NOT advanced beyond the inherited base** `f1fa8e7` since the prior session's
  `shared-inner-fg-operator-port-2026-07-26` memory was written. No rebase is required — this
  port branches directly off the existing worktree HEAD.
- Tag reachable from `f1fa8e7`/current production HEAD:
  `common-frechet-cdf-cm-plus-level-production-ready-2026-07-26`.
- Working tree (`worktrees/shared-inner-fg-operator-2026-07-26`, itself a `git worktree` of the
  canonical checkout at `/bbkinghome/edav/cdw`): clean, no untracked/modified files, at branch
  creation time.
- New branch created: `port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26`,
  from `port/shared-inner-fg-operator-and-verification-2026-07-26@e48a71d` (36 commits ahead of
  `f1fa8e7`).

## Current five public family drivers (unchanged this session, confirmed present)

- Unrestricted: `c10_d20_production_driver.jl`, `c10_d20_production_driver_unified.jl`,
  `c10_d20_production_driver_flexible_theta_A.jl`.
- Flexible CM / Common Fréchet / CM+ZC / Origin-ZC: share `cm_outer_driver.jl` plus family-specific
  production-context builders (`cm_production_bundle.jl`, `cm_meanzc_production.jl`,
  `cm_originzc_production.jl`, `cm_frechet_*`).
- Backend manifest: `production_backend_manifest.jl` (central, machine-readable resolver — NOT a
  duplicate of the two startup banners; already reports per-family `core_hessian_backend`,
  `cross_hessian_backend`, `restriction_hessian_backend`, `checkpoint_schema`, `screen_stack`).

## Release-state as of this branch's creation

`INHERITED_WORK_AUDITED` (this document). Prior sessions already reached
`PHASE_A_FG_AND_VERIFICATION_COMPLETE` for exactly 2 of 5 families (origin-ZC, CM+ZC) at the
correctness level only — performance gating (a required part of Phase A per this task's own §6/§16)
was explicitly not run (see `docs/FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB_2026-07-26.md`, added at
`e48a71d`, honest "NOT measured" section). Flexible-CM and common-Fréchet's own economic-block
retrofits (task §3/§4) were NOT started. This continuation begins from that honest state.

## Per-commit classification (36 commits, `f1fa8e7..e48a71d`)

**Commits `8a8b0a8..29b1c79` (30 commits, pre-existing `finish-five-family-optimization-stack`
chain)**: classification already performed and recorded verbatim in this branch's own commit
`fea8e9a` (message reproduced in full below for a single source of truth) — **ADOPT (29/30)**,
**ADOPT_AFTER_MORE_GATES (1/30: `06d4bd6`, A_coordinate_mode default promotion, orthogonal to this
task's scope, left as-is)**. No re-litigation performed; re-quoting here would only drift from the
original. See `git log -1 fea8e9a` on this branch.

**Commits added by the prior `shared-inner-fg-operator` session (6 commits, `b399de4..e48a71d`)**,
classified fresh under this task's taxonomy:

| Commit | Summary | Classification | Rationale |
|---|---|---|---|
| `b399de4` | Fix include-order gap (`compressed_cc_inner.jl` missing from `cm_hessian_architectures.jl` guard) | **ADOPT_AFTER_REBASE** | Pure bugfix, no rebase needed (base unchanged), keeps inherited branch's own gates honest. |
| `74c1450` | Shared `economic_forward!`/`economic_transpose!` + origin-ZC `[E\|Z]` operator FG, opt-in | **ADOPT_AFTER_PERFORMANCE_GATE** | Correctness gated D=4+D=20 ALL PASS; task §6 explicitly requires a complete-inner-solve timing/allocation A/B vs dense before the default may change. This is Phase A item 6's outstanding gate — addressed below (§ Perf A/B). |
| `47daf6f` | CM+ZC `[E\|C\|Z]` operator FG, opt-in, + D=20 perturbation-point fix for origin-ZC gate | **ADOPT_AFTER_PERFORMANCE_GATE** | Same rationale as above, task §5. |
| `ae2a790` | Real D=20/W=80,000 correctness gates for both new operators + `no_dense_g_counters.jl` | **ADOPT_AFTER_REBASE** | Correctness evidence only, no default change; counters are additive instrumentation, safe. |
| `3969acf` | `verify_inner_solution_operator_originzc!` — operator verification PoC, origin-ZC only | **REFERENCE_ONLY (until extended)** | Task §8 requires extending this pattern to all 5 families before it is a production verifier; kept as the reference implementation this session's extension work builds from. |
| `e48a71d` | Final deliverable docs (honest gap reporting) | **REFERENCE_ONLY** | Docs only, no code; superseded incrementally as each gap closes this session. |

No `DROP` or `REWORK` commits in either chain — every commit is either adopted infrastructure or a
correctly-scoped, honestly-labeled opt-in candidate awaiting the specific gate this task asks for.

## `fea8e9a` classification (reproduced verbatim for single-source-of-truth)

> Branch setup: adopt finish-five-family-optimization-stack chain, save addendum verbatim
>
> Base: production/fullA-exact@f1fa8e7 (common-Frechet Phase 0 + transformed-A merge). Fast-forwarded
> onto port/finish-five-family-optimization-stack-2026-07-26@29b1c79 (28 commits, all linear on top
> of f1fa8e7, no divergence) after inspecting every commit message and reading the two detailed
> Phase 5.5/5.2/AddendumA writeups in full.
>
> Classification (per addendum §0):
> - ADOPT (25/28): 8a8b0a8, cb4b6b2 (audit docs, no code); 01559e3 (Phase A unrestricted stage
>   runner repoint, MATCHED_AB_PASSED); dd85847 (Phase B1 wire lookup FG, default unchanged,
>   MATCHED_AB_PASSED); 01ebea0 (Phase C exact-point cache, MATCHED_AB_PASSED); ac2ede8 (Phase D
>   dual-bank warm starts, MATCHED_AB_PASSED); 053bb27 (Phase E1 CompressedFactualWorkspace wiring,
>   MATCHED_AB_PASSED); a34198b (bugfix: CMBinHessCtx missing inner_fg_backend field, CM+ZC was
>   broken since Phase B1); 1ef4578 (Phase E2, MATCHED_AB_PASSED); fce210e/47fe5e5/88c7658/3d5ba5a
>   (Phase F manifest/docstring/fail-fast fixes, MATCHED_AB_PASSED); 1900635 (self-correcting REWORK
>   already folded into this linear history); abbecb8/ca5f176/0a85db4 (D=20 gates, counters, honest
>   gap docs); 2fa61b7/223fd02/dad6b9c/b20902a/08550a8/c3c0073 (origin-ZC dual-bank data, final
>   5x7 matrix, task-prompt archival); 217e91b (flexible-CM cm_lookup allocation fix + default flip,
>   D=4+D=20 ALL PASS, alloc parity, 1.1-1.6x faster at every thread count); c5f29eb (doc update);
>   b31e88c (common-Frechet CM+level lookup operator, D=4+D=20 ALL PASS, 1.11-1.19x faster, alloc
>   +21% -- stays opt-in, root-cause deferred to this branch's own §6); fbb7d79 (unrestricted
>   allocation-free compressed FG, 20,081x allocation reduction, bit-identical, existing integration
>   suite ALL PASS); 29b1c79 (handoff note, now superseded by this branch's own docs).
> - ADOPT_AFTER_MORE_GATES (1/28): 06d4bd6 (promotes A_coordinate_mode default :legacy_z ->
>   :powered_aspace for all 4 restricted families). This has its own real gate evidence (6/6 a<->z
>   round-trip + gradient-rescale checks, real D=20 default-kwargs smoke ALL PASS for 3/4 families)
>   but is ORTHOGONAL to this addendum's FG-operator/verification scope and is not independently
>   re-validated by this session. Left as-is (not reverted) since reverting would itself be an
>   unreviewed behavior change; flagged here for the user's own awareness that it diverges from
>   memory `common-frechet-phase0-merge-and-transformed-a-port-2026-07-26`'s statement that this
>   promotion was "a separate follow-up decision" not yet made -- this branch inherits it as already
>   decided by a sibling same-day session, not something this session is deciding or re-litigating.
>
> No DROP/REWORK commits -- the entire chain was linear, individually gated, and directly relevant
> groundwork for this addendum's Job 1/Job 2 asks (lookup kernels for CM/Frechet, exact cache,
> dual-bank, workspace wiring are all prerequisites the shared-operator and operator-verification
> work below builds on).
>
> Also saves the addendum task prompt verbatim as docs/ADDENDUM_SHARED_ECONOMIC_FG_2026-07-26.md
> per the prior handoff's explicit request (previously only in conversation history).
