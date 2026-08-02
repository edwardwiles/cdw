# Profiled restricted-family production outer bridge — source snapshot (2026-08-01)

## Branch / base

- New branch: `architecture/profiled-restricted-production-outer-bridge-2026-08-01`
- Base: `architecture/profiled-restricted-outer-gradient-2026-08-01@2b95408`
  (`2b95408cc3d470e0d0513f751097f820a2eb4c85`, "Add SHA256 manifest for the outer-gradient branch
  deliverables") — exact commit named in the task spec, verified via `git rev-parse`.
- Repo: `github.com/edwardwiles/cdw` (`origin`), worktree checked out at
  `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-restricted-production-outer-bridge-2026-08-01`.
- The outer-gradient branch is **local-only**: `git fetch origin
  architecture/profiled-restricted-outer-gradient-2026-08-01` fails with "couldn't find remote ref" —
  matches the sibling memory note (`profiled-restricted-outer-gradient-layer-2026-08-01`: "committed
  locally... not merged, not pushed to origin"). This new bridge branch is likewise local-only unless
  explicitly pushed.

## Worktree status at branch creation

`git status` on the new worktree immediately after `git worktree add`: clean (`nothing to commit,
working tree clean`), confirming the base commit was reproduced exactly, byte for byte, before any new
work started.

## Scaffold file checksums (SHA256, at base commit, before any edits by this task)

See `repro_logs_2026-08-01/scaffold_sha256.txt`:

```
a71d803c1ac468084d5e1208773b714c80d8a4b4ce16b0b1d502614be0d08092  profiled_outer_gradient_layout_contract_2026-08-01.jl
5a84df8d0a311f14195608b2ad266fbc7575af16f0e87abee824586377941fbe  profiled_shared_economic_gradient_engine_2026-08-01.jl
9a714d88405d5c7cb9a272b49c67ea1ebab852c94b007d3a1d4edf2b1d913136  profiled_family_adapters_2026-08-01.jl
ff3c1633a0a8d721ac11d7b89a2af182f913544ff1e9b2460a001df34d6bc755  profiled_lfix_incremental_2026-08-01.jl
6937e6c6eff026ba460f35259e70fb81522d8e5800d2d341905a5010297a6fd8  PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl
648a783c1e01d3d2d83b904e7b682ba361ed3a8959d61717470e1de9bb52d785  profiled_restricted_full_rebuild_gradient_reference_2026-08-01.jl
91ae5b9ab4aa8a0dacae80ecd5bc13593bc06326f70edf1710a97fd7435cf2c1  profiled_outer_evaluator_2026-08-01.jl
2423facfdfacb285a5e29ae5d3e3d924d4a446c70289d42cfb3c657014389146  profiled_operator_bundle_2026-08-01.jl
```

## Relationship to the live inner branch

`architecture/profiled-restricted-inner-endtoend-2026-08-01`, tip `bee0303` ("Document CM+ZC
widened-core finding; Phase 8 deferred for this family by explicit user decision"). Merge-base with
this bridge branch is `f439109` — the two branches diverged at the outer-gradient branch's own base
point and have run as genuinely parallel workstreams since, exactly as the task describes.

**Important, time-sensitive observation**: the inner worktree
(`/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-restricted-inner-endtoend-2026-08-01`)
has **uncommitted, untracked** files as of this snapshot (`git status` there shows untracked
`full_aod_diag/d4_exact/profiled_restricted_accessors_2026-08-01.jl`,
`test_profiled_restricted_accessors_2026-08-01.jl`, `key_results_repro_2026-08-01/`,
`run_repro_gates.sh`, most recently modified within the last few minutes of this snapshot — i.e. that
session is actively working right now). `profiled_restricted_accessors_2026-08-01.jl` implements
almost exactly this task's five-accessor surface (`profiled_economic_layout`, `economic_dual_range`,
`restriction_dual_ranges`, `profiled_anchor_spec`, `profiled_outer_coordinate_layout`) against real
`CMBinHessCtx`/`OriginZCCoreHessCtx` reduced contexts, dispatched per family, with an explicit
"CM+ZC widened-core: not supported" guard.

This is **read for situational awareness only** — it is uncommitted, unreviewed, and mid-edit in
someone else's live worktree, so this task does **not** depend on it, per the task's own explicit
fallback instruction ("if a missing inner API blocks live integration, implement the adapter against
the existing production restriction operators and provide a narrow typed hook for the inner branch to
expose later"). It is flagged here so that whoever next rebases this bridge branch onto the inner
branch's eventual committed tip knows to check whether that file (or its committed descendant) landed,
and can compare its accessor names/semantics against this branch's narrow hook (§8 below) at that time.
Nothing in this branch reads, imports, or numerically depends on that file.

## Relationship to the ZC Hessian production tag

**No formal tag exists yet.** Per project memory
(`zc-hessian-backend-closeout-2026-08-01`/`genuine-cold-zc-hessian-k3-optimization-2026-08-01`), the
validated ZC Hessian work (H_ZZ=blas_syrk, H_CZ=draw_chunk_reordered, H_EZ=drawmajor_v2) is committed
at `1807ef5` on branch `optimize/genuine-cold-zc-hessian-k3-closeout-2026-08-01` — **committed but not
merged**, and not tagged. `1807ef5` is not an ancestor of this bridge branch's base and is not on
`origin`.

**Plan for eventual rebase** (task §16, not executed in this task — no rebase performed, per the "do
not overwrite the outer-gradient scaffold with ZC kernel files" instruction): once (a) the inner branch
lands a reviewed/committed accessor surface and (b) the ZC Hessian work is tagged/merged to a stable
point, rebase this bridge branch onto the inner branch's tip first (replacing the mock/current-full-
context adapters built in this task with real reduced-context constructors), re-run the q-decomposition
and combined-gradient gates, and only then rebase the integrated result onto the ZC production point —
preserving `H_ZZ`/`H_CZ`/`H_EZ` exactly and implementing profiled `H_EZ` only as a thin correction
adapter, per the task's explicit sequencing. This bridge branch touches no Hessian kernel file, so
that rebase should be a pure fast-forward/merge of unrelated files in the common case.

## Reproduction of the completed scaffold (task §2), re-run live on this new branch/worktree

All four gates re-run from a clean worktree before any new code was written, per the task's explicit
"before editing, reproduce" instruction:

| Gate | Result | Log |
|---|---|---|
| Interface contract tests (8/8) | PASS | `repro_logs_2026-08-01/interface.log` |
| D4 unrestricted regression (shared engine vs. pre-refactor) | PASS, `max_abs_err=0.0`, bit-identical | `repro_logs_2026-08-01/unrestricted_regression.log` |
| D20 `:exclude_row` W=20,000 unrestricted regression | PASS, `max_abs_err=0.0`, bit-identical, real KNITRO (`inner_status=0`) | `repro_logs_2026-08-01/unrestricted_regression_D20.log` |
| Mock restricted-family gate (4 families × 2 checks) | PASS, `max_rel_err` 3.3e-15–5.5e-15 (A-block), machine precision | `repro_logs_2026-08-01/mock_family_gate.log` |
| A/B harness smoke (unrestricted arm; restricted arms throw loudly) | PASS | `repro_logs_2026-08-01/ab_harness_smoke.log` |

No regression. Safe to build on top of this base without re-deriving anything the scaffold already
established, per the task's §2 instruction.

## Environment

- Julia 1.12.6 via `~/.juliaup/bin` (not `/opt/shared_sw` — see project memory
  `julia-toolchain-use-juliaup-not-shared-sw`).
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` exported for every run (project hard-cap policy).
- All commands: `julia --project=<worktree-root> full_aod_diag/d4_exact/<file>.jl`, run from the
  worktree root.
