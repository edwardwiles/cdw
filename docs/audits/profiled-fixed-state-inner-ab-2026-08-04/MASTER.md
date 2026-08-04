# Fixed-state FULL-vs-REDUCED inner A/B -- status, 2026-08-04

**Stopped after step 2 per the task's own instruction: "If the outer task has not completed,
stop after step 2. Do not wait by launching background work."** Steps 3-12 (decoded-state
equivalence, sentinel panel, cold/warm protocols, Mode A/B execution, telemetry, results) were
NOT started. No KNITRO solve was run. No point bank was built.

## Provenance

- Repo: `/bbkinghome/edav/cdw`
- Branch: `benchmark/profiled-fixed-state-inner-ab-2026-08-04`, created from
  `origin/prototype/profiled-destination-scales` at `395dec3e1e68844128cc98c16be17e91bc9b6603`
- Worktree: `/bbkinghome/edav/cdw_worktrees/profiled-fixed-state-inner-ab-2026-08-04`
- Verified live remote state (step 1) exactly matched the task's expected values:
  - `origin/prototype/profiled-destination-scales` tip: `395dec3e1e68844128cc98c16be17e91bc9b6603`
  - `git tag --list 'profiled-functional-ready-*'` -> `profiled-functional-ready-2026-08-04`,
    resolving to the same SHA
  - Working tree at that commit was clean before branching.

## Why step 2 stopped here: live evidence the outer task has NOT completed

The task names a concurrent "second Claude" owning src outer-gradient implementation,
coordinate-mode implementation, bandwidth-cache implementation, canonical outer runner
internals, and FamilyRegistry capability definitions, and says not to update
`prototype/profiled-destination-scales` while that Claude is still working.

Two independent pieces of evidence, both collected live during this task's own step 1/2:

1. `git worktree list` showed `/bbkinghome/edav/cdw_worktrees/profiled-outer-ab-readiness-2026-08-04`
   on branch `performance/profiled-outer-ab-readiness-2026-08-04`, working tree clean, at
   `395dec3` -- the exact same tip as canonical prototype. This is a freshly-provisioned worktree
   for outer/AB-readiness work, not an old artifact.
2. Re-checking that same worktree only a few minutes later (while writing this file) showed it
   had already advanced to a new commit, `cac0cb4` (`2026-08-04 10:12:12 -0400`, "Section 2:
   correct stale documentation/capability contradictions") -- i.e. a session was actively
   committing to it in real time, concurrently with this task's own step 1/2 work.

A third, corroborating (but stale) data point: the pre-existing `feature/profiled-outer-production-readiness-2026-08-03`
worktree's own `MASTER.md` states outright "sections 8, 9, and 11-12 are documented but not
implemented" and "Nothing here has been merged... No A/B campaign was launched" (dated
2026-08-03 21:19). Its early sections (1-3, 5, 7, 10 -- manifest infra, checkpoint/resume,
coordinate-mode formalization, the ABComparability gate) DID subsequently land on canonical
`prototype/profiled-destination-scales` (confirmed via `git merge-base --is-ancestor`: both
`5e2c262` and its own HEAD `781eb65` are ancestors of `395dec3`), and prototype continued forward
from there through the separate "profiled-functional-readiness" Phase 1/2 continuation
(eval18 resolution, free eta_nu, W100k warm-start/fast-reject for all 5 families) up to the
`profiled-functional-ready-2026-08-04` tag this task started from. But that same outer-readiness
line of work evidently did NOT stop at `781eb65`/`395dec3` -- it is continuing right now in the
`performance/profiled-outer-ab-readiness-2026-08-04` worktree, which by name is doing exactly the
outer/AB-readiness work this task's own step 13 says to wait for.

**Conclusion: the outer task has not completed.** Per the task's explicit contingency, this
session stopped after step 2 rather than building the full point bank / running any solves that
could be invalidated by, or interfere with, that concurrent work.

## What step 2 actually produced

`tools/benchmarks/fixed_state_inner_ab/frozen_manifest.jl` -- read-only with respect to
`scientific_manifest/{ScientificManifest,RunManifest,ABComparability}.jl` (only `include`d, never
edited; those files, plus `FamilyRegistry.jl`, `coordinate_mode` and outer-gradient/
bandwidth-cache code, are the concurrent Claude's territory and were not touched). Defines:

- `mode_a_scientific_manifest()` -- Mode A (diagnostic parity): `W=20_000`, `julia_threads=1`,
  `blas_threads=1`; every other field taken verbatim from the real current production manifest
  `configs/fullA_production_2026-08-03.toml` (dataset/checksum, country order, France focal,
  `sigma=3.0`, gravity mask including the Brazil-Korea exclusion, `destination_sample=:exclude_row`,
  `sobol_randomized` draw design/seed, `L=50`/`K_mean=1`/`K_pair=1`, inner/outer KNITRO option
  checksums).
- `mode_b_scientific_manifest()` -- Mode B (production parity): identical to Mode A except
  `W=100_000`, `julia_threads=10`, `blas_threads=8` (the toml's own real values, unchanged).
- `FIXED_STATE_LOWER_LIMIT = -50.0` -- the documented deliberate KNITRO inner-objective floor
  (see this repo's own top-level `CLAUDE.md`), frozen so both arms share it.

Verified by direct execution (`julia`, `OPENBLAS_NUM_THREADS=1`, via `juliaup` per this repo's
own toolchain guidance) that both manifests construct without error and have the expected
field values; confirmed `dataset_checksum` is identical between the two modes.

The task's `ab_comparable`/`ABComparabilityResult` mechanism already in
`scientific_manifest/ABComparability.jl` (added by the concurrent task, already merged onto
canonical prototype at commit `5e2c262`) already implements exactly what step 2 asked for if the
gate were to reject a pair "solely because the formulations necessarily have different inner
dimensions or coordinate names": `allow_coordinate_mode_diff=true` + a required
`coordinate_mode_experiment_label`, with every other field (including
`initial_state_digest`/decoded-state equivalence) still enforced unconditionally. **No new
comparability rule was added** -- the existing one already satisfies the requirement without
weakening the general gate, so adding a second one would have been needless duplication.

## Explicitly NOT done (open points for whoever resumes this task)

- No `RunManifest` instances were constructed (they need real per-point `nu_policy`/
  `draw_checksum_*`/`outer_algorithm`/`initial_state_digest`/`source_sha` values, which only
  exist once a real point bank and solver run exist -- step 3 onward).
- No sentinel point bank (`FIXED_STATE_POINT_BANK.jld2`) was built.
- No decoded-state equivalence check was run.
- No cold/warm solves, no Mode A or Mode B execution, no telemetry.
- Verification tolerances (task section 2's "verification tolerances" field) were deliberately
  left unfrozen here: the codebase has no single canonical tolerance constant (checked --
  tolerances are scattered per-family `verify_fn` kwargs across `full_aod_diag/d4_exact/*.jl`).
  Whoever resumes step 2 in full should source these from `FamilyRegistry.jl`'s
  `inner_verifier` field per family, not invent a number.
- Mode A's BLAS thread policy for CPU-set pinning ("fixed disjoint CPU set", task section 7,
  Mode B) has no ScientificManifest field and was not addressed -- it is a runtime/OS-level
  concern for the harness (`tools/benchmarks/fixed_state_inner_ab/run.jl`, not yet written),
  not a manifest field.

## Resuming this task

Re-run this task's own step 1 checks against `performance/profiled-outer-ab-readiness-2026-08-04`
(or whatever branch canonical prototype has advanced to by then). Only proceed past step 2 once
that concurrent work has genuinely landed on `prototype/profiled-destination-scales` with no
worktree still actively committing to it.
