# Fixed-state FULL-vs-REDUCED inner A/B -- status, 2026-08-04

**UPDATE (same day, later): unblocked and resumed.** Originally stopped after step 2 per the
task's own instruction ("if the outer task has not completed, stop after step 2"). The outer
task (`performance/profiled-outer-ab-readiness-2026-08-04`, later continued on
`performance/profiled-outer-ab-completion-2026-08-04`) subsequently finished its own scope and
explicitly declared, for all 5 families, `OUTER_STATUS = OUTER_READY_FOR_INNER_AB` (see that
branch's own `docs/audits/profiled-outer-ab-completion-2026-08-04/MASTER.md`). This branch was
rebased onto that completion branch's tip (`c59add8`) and force-pushed; see "Rebase" section
below. Step 3 (decoded-state equivalence) is now genuinely started, with real passing evidence
at D20/W=20,000 and D20/W=100,000 -- see its own section below. Steps 4-12 remain open.

## Rebase (2026-08-04, after unblock)

Rebased `benchmark/profiled-fixed-state-inner-ab-2026-08-04` (previously based on
`origin/prototype/profiled-destination-scales`@`395dec3`) onto
`origin/performance/profiled-outer-ab-completion-2026-08-04`@`c59add8` -- the fullest available
state (coordinate-mode tournament, bandwidth cache, threaded REDUCED gradient, corrected
`FamilyRegistry.jl` all wired in), NOT the partially-landed canonical prototype tip (`36aacb2`,
which only cherry-picked 2 small verifier-fix commits from the completion branch and does not yet
include its bulk). This was a deliberate user decision (asked directly, given the fork between
"build on the unmerged-but-more-complete completion branch" vs. "wait for an actual
fast-forward into canonical prototype") -- not a default I chose unilaterally. `frozen_manifest.jl`
re-verified working (both mode_a/mode_b construct correctly) on the new base.

## Original stop-after-step-2 record (preserved below for provenance)

Steps 3-12 (decoded-state equivalence, sentinel panel, cold/warm protocols, Mode A/B execution,
telemetry, results) were NOT started as of the original stop. No KNITRO solve was run. No point
bank was built.

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

## Step 3: decoded-state equivalence -- real evidence, real D20, both scales

`tools/benchmarks/fixed_state_inner_ab/decoded_state_equivalence.jl`. Modeled on the pre-existing
D4-only gate (`full_aod_diag/d4_exact/test_profiled_coordinate_mode_roundtrip_2026-08-03.jl`) --
same primitives (`reduce_to_w_profiled`/`decode_outer_profiled`/`gravity_from_logz`, all
`include`d unmodified), same properties, but run against the REAL production context
(`d20_real_setup_design`, every argument taken from `frozen_manifest.jl`'s frozen
`ScientificManifest` -- no function default relied on) at real D20 scale, which the pre-existing
gate never did.

**What it checks**: takes the canonical decoded FULL calibration state (`ctx.θ0_up`'s own full
`gp0`/`z_calib` -- NOT the `A_od≡1`/`zfree=0` reparameterization-offset trap this repo's own
CLAUDE.md warns about repeatedly; this is the real calibrated block spanning several orders of
magnitude), encodes it into REDUCED's native `:profiled_pivot_anchor_relative` coordinates
(`reduce_to_w_profiled`), recovers the full FULL A matrix back out (`decode_outer_profiled`), and
compares the recovered object against the original -- not two raw vectors in different coordinate
systems, the same reconstructed economic object both ways, per the task's own explicit
instruction.

**Real results, D20/W=20,000 (Mode A scale) and D20/W=100,000 (Mode B scale), both `ALL PASS`**:

```
gp: decoded == original (exact)
max|logA_decoded - logA_original| = 4.441e-15 (relative 1.332e-15)
max relative |A_decoded - A_original| = 4.437e-15   (real A spans ~5.9 orders of magnitude here)
gravity residual: original=-1.833e-19  decoded=8.553e-20   (both endpoints exactly feasible)
decoded_state_digest = 9f4ab049ee88c6185a2d79d2480d8c0851fb207e82af34b67230dedce1b70aa5
  (IDENTICAL between W=20,000 and W=100,000 -- confirms the calibration point, θ0_up's own
  gp/A_od block, is genuinely W-independent, as expected: gravity/theta estimation runs on real
  trade data, not the Monte Carlo draws, which are the only thing that varies with W)
```

Full logs: `key_results/dse_mode_a.log`, `key_results/dse_mode_b.log` (this doc's own
`repo_scratch` mirror; not committed to git, per this task's storage-location rules).

**One deliberate correction made mid-run**: the script's first draft additionally re-derived a
second `digest_economic_state` from the *decoded* state and required bit-exact equality against
the original's digest -- this FAILED, but was a flaw in the check, not a real problem:
`digest_economic_state` is an exact byte-level digest by its own docstring ("two states that
print identically digest identically, full stop"), and the round trip legitimately introduces
~1e-15 floating-point noise (log/exp/encode/decode) even though every real numerical check above
passes at machine precision. Fixed by computing exactly ONE digest, from the canonical original
state, that both arms' `RunManifest.initial_state_digest` should cite going forward -- which is
what the task's own section 3 actually asks for ("record a decoded-state digest shared by both
arms"), not a second independently-rederived digest checked for bit-exact equality.

**What this does NOT yet cover** (real remaining work, not silently skipped):
- Family-specific restriction targets (only the common `gp`/`A_od` calibration state was
  checked here -- restriction moments are family-specific and require each family's own
  context/layout, step 4's point-bank work).
- `nu` for the ZC families (`origin_zc`/`cm_meanzc`) -- this script passes `nu=Float64[]`
  throughout since the calibration point has no restriction active; extending to a real ZC point
  with genuinely nontrivial free `nu` (task section 4's explicit requirement for ZC sentinel
  points) is open.
- Points other than the single calibration point P0 -- P1/P2/P3 (ordinary feasible, difficult
  feasible, infeasible/unbounded) require the sentinel point bank (step 4), not yet built.
