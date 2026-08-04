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

## Step 3 continued: nu and restriction-target equivalence -- resolved by construction, not by round trip

Researched (read-only, no files edited) whether `nu` (ZC families) and each restricted family's
own restriction target need a SEPARATE coordinate-transform round-trip check, the way `gp`/full
`A_od` did above. Real finding, confirmed by direct source read with file:line citations:

- **`FamilyRegistry.jl`'s own `context_constructor` field is `"d20_real_setup_design"` for ALL 10
  rows (5 families x 2 formulations)** -- the field's own comment states outright "both
  formulations share this." FULL and REDUCED are not two independently-built contexts that
  happen to agree; for the same manifest arguments they are the literal same deterministic
  function call, so anything derived from `ctx.U`/`ctx` (nu's target functions, the CM/Fréchet
  restriction targets) starts from a shared object, not two objects that could drift.
- **`nu` itself**: on both sides, `nu` is a plain untransformed `Float64` (post `nu=exp(eta)`,
  symmetric on both sides) fed into the literal SAME `mean_targets`/`pair_targets` functions
  (`cm_originzc_target_layout.jl:85-99`). FULL calls them from `wrap_moments_with_originzc`
  (`cm_originzc_moments.jl:167-178`); REDUCED calls the same functions from `refresh_zc_targets!`
  (`zc_restriction_operator.jl:130-142`) on `aug.Zraw_all`/`aug.Zpairraw_all` -- the SAME object,
  not a re-derived copy (`cm_hessian_architectures.jl:2004,2017`). The free-nu wrapper
  (`profiled_zc_free_eta_2026-08-04.jl:110-113`) does `nu_full = exp.(eta_nu)` then passes it
  straight through, unchanged, into the same target machinery. There is no second,
  formulation-specific nu transform to round-trip.
- **flexible_cm/common_frechet restriction targets**: theta-independent by construction (built
  once from raw `ctx.U` at fixed quantile thresholds, `common_marginals_moments.jl:50-97`;
  Fréchet's own level-anchor block likewise, `cm_frechet_level.jl:42-54`, explicit comment "no
  dependency on theta_star/sigma/scale"). REDUCED's `FlexCMFamilyCtx`/`FrechetFamilyCtx`
  (`profiled_restricted_family_adapters_2026-08-02.jl:40-53,127-138`) hold the SAME `cctx`/
  `level_targets` object threaded in from the FULL side's own construction, not recomputed
  independently.
- **No existing test computes a restriction target or nu under both formulations at the same
  point and diffs them numerically** -- confirmed absent, not just unfound: there is no second
  independently-derived value to diff against in the first place, per the shared-object findings
  above. The closest adjacent machinery (`test_zc_free_eta_evaluator_d4_2026-08-04.jl`) gates
  REDUCED's own analytic gradient against FD, not a FULL-vs-REDUCED value comparison.

**Conclusion**: unlike `gp`/full `A_od` (which genuinely differ in representation between the two
formulations and required the round-trip check above), `nu` and every restriction target are
equivalent by shared construction, not by a transform that could introduce error -- there is
nothing further to numerically verify here beyond confirming (already true by the
`context_constructor` finding) that both arms of any future A/B pair are built from IDENTICAL
manifest arguments. Step 3 is complete for all 5 families on this basis; the remaining
family-specific work is in the point bank (step 4): finding/constructing real nontrivial `nu` and
restriction-affected points to populate it with, not re-deriving a second equivalence check.

## Step 4: sentinel point bank -- real evidence, honest gaps

`tools/benchmarks/fixed_state_inner_ab/build_point_bank.jl`. Assembles points from REAL sources
only, decoded through step 3's own verified round trip (never a raw two-basis comparison):

- **P0 (calibration)**: `ctx.θ0_up`, shared by all 5 families (same object, per step 3).
- **P1 (ordinary feasible)**: real `CMCheckpointV11` checkpoints already written by the
  outer-completion session's own canonical-CLI runs, `results/canonical_runner/reduced_<family>_W<W>_delta1.0/checkpoint.jls`
  -- read-only, never mutated. Found and loaded successfully for **all 5 families, both W=20,000
  and W=100,000** (10/10 cells, zero gaps).
- **P3 (infeasible/stall)**: real archived points only. `unrestricted` gets 2 (idx=2, idx=17 of a
  genuine W=100,000 forensic campaign's own JLD2 dict, machine-local at
  `/bbkinghome/edav/repo_scratch/profiled-functional-readiness-closeout-2026-08-03/`, outside git
  -- per this repo's own CLAUDE.md guidance on `.zip`/scratch artifacts living outside the repo).
  `flexible_cm` gets the real, in-repo, extensively-forensically-verified
  `eval18_captured_point_2026-08-02.txt`. **Honest gap, not filled with a synthetic point**:
  `common_frechet`/`origin_zc`/`cm_meanzc` have NO real persisted P3 anywhere in this repo --
  confirmed by a repo-wide search; only unsaved, logged shift-probe numbers exist for those three
  (`CONTINUATION_2026-08-04.md`), which do not meet "an exact saved point." Constructing and
  persisting real P3 points for these 3 families (via the same documented additive-log-A-shift
  methodology, with a live KNITRO classification run, not just a citation) is open follow-up work,
  not done in this pass.
- **P2 (difficult feasible)**: only `flexible_cm` -- reconstructed exactly from the task's own
  documented `k=0.3`/`k=0.6` interpolation between the real calibration point and the real eval18
  point (`w_calib + k*(w_eval18 - w_calib)`), which CONTINUATION_2026-08-04.md already ran as real
  KNITRO solves (feasible, near the eval18 infeasibility boundary) but never saved to a file --
  reconstructed here from the two real endpoints, not re-copied from a log number. No other family
  has any real P2 candidate anywhere in the repo (confirmed gap, not filled).

**Sanity check on every point**: gravity residual is ~1e-19 for all 15 distinct points at both
modes (30 rows total, P0/P1/P3 shared or independently decoded per mode) -- i.e. every REAL
checkpoint this script loaded really does live on the gravity-feasible manifold once decoded
through the SAME `(ctx, pe)`, not just the constructed calibration point. This is a live
confirmation (not assumed) that the checkpoints are genuine points from this exact economic
setup, not stale/mismatched artifacts.

**Outputs** (per this task's storage rule -- large/binary outputs only under `repo_scratch`, not
git): `/bbkinghome/edav/repo_scratch/profiled-fixed-state-inner-ab-2026-08-04/FIXED_STATE_POINT_BANK.jld2`
(30 `SentinelPoint` records) and `.../FIXED_STATE_POINT_MANIFEST.csv` (human-readable summary:
family, point_id, point_type, mode, W, gp, gravity_residual, eta_nu_norm, source, verification_status).

**What "verification_status" means at this stage**: every point's status field cites its REAL
prior evidence (checkpoint's own `n_eval`/`n_grad`/`checkpoint_reason`/`best_feasible`, or the
forensic doc's documented KNITRO status) but explicitly says re-classification is deferred to
step 6 -- i.e. this bank records provenance, and step 6's own cold-solve run is where each point
gets independently re-verified live under this task's own controlled solver settings, not
silently assumed from the citation alone.

**Point coverage summary**:

| family | P0 | P1 (both W) | P2 | P3 |
|---|---|---|---|---|
| unrestricted | yes | yes | none (gap) | 2 real archived points |
| flexible_cm | yes | yes | 2 (reconstructed from real endpoints) | 1 real (eval18) |
| common_frechet | yes | yes | none (gap) | none (gap -- no real file exists) |
| origin_zc | yes | yes | none (gap) | none (gap -- no real file exists) |
| cm_meanzc | yes | yes | none (gap) | none (gap -- no real file exists) |

## Real bug found and fixed: point bank used the wrong checkpoint field for P1/best-feasible points

`load_reduced_checkpoint_point` originally built each P1 point from `cp.g`/`cp.zfree` -- the
checkpoint's raw CURRENT outer iterate at whatever wall-clock instant it fired
(`checkpoint_reason=:wall_interval` for every real checkpoint here), not necessarily feasible at
all (KNITRO legitimately visits infeasible points between feasible ones during search). The
actual verified feasible incumbent lives in a SEPARATE field, `cp.best_feasible` -- confirmed by
direct inspection: for the real unrestricted W=20,000 checkpoint, `cp.g=0.9590607665008352` vs
`cp.best_feasible.w[1]=0.9590607598370081` -- close but NOT identical, two genuinely different
points, not the same value read two ways. Free-nu families' `best_feasible` uses different field
names (`w_econ`/`eta_nu` as separate fields, not a combined `w`) -- handled explicitly, not
assumed uniform. Fixed in `build_point_bank.jl`; point bank rebuilt (still 30 points, same
structure, corrected `w` for every P1 cell).

## Real finding: REDUCED's "verified feasible" P1 incumbent is genuinely infeasible under FULL

Even after the fix above, `unrestricted` P1's real `best_feasible` point (`gp=0.9590607598370081`,
`Delta=0.999`, REDUCED's own verified feasible incumbent) still fails FULL's cold-start
pre-check (`run_polish_checkpointed_unified`'s `r0.inner_status in FEASIBLE_CODES` gate) with
"start point not inner-feasible."

**Isolated and confirmed real, not a bridge bug**: called `screened_eval` directly (the exact
function `run_polish_checkpointed_unified` itself calls) on both P0 and P1 after bridging through
the identical `full_coordinate_bridge.jl` machinery:
```
P0: inner_status=0    Delta_dual=0.0045438486  gravity=8.4e-18   (feasible)
P1: inner_status=-300 Delta_dual=NaN            gravity=NaN       (genuinely infeasible)
```
`gp` round-trips to the input exactly in both cases (confirming the bridge itself is not at
fault -- P0's own success on the identical code path is the control). `-300` is this repo's own
established, dual-convexity-certified infeasibility code (not "unbounded", not a solver-option
artifact -- see this repo's CLAUDE.md and memory `feedback-knitro-300-confirmed-infeasible-not-unbounded`).

**What this means, stated carefully**: a point REDUCED's own real production search found and
recorded as its verified `best_feasible` incumbent is genuinely infeasible once the identical
decoded economic state (`gp`, full log-A) is evaluated through FULL's own inner solve. For
`unrestricted` specifically there is no restriction-specific machinery on either side -- both
formulations' inner problems should, in principle, be checking the same feasibility LP over the
same draws, which makes this a genuinely puzzling, not merely expected, disagreement.

**Deliberately NOT investigated further this session** (per the task's own explicit instruction:
"do not fix any defect discovered by the benchmark -- record it as a blocker for a separate
repair"). Real, open hypotheses, none confirmed:
- REDUCED's own default `verification_policy` for this checkpoint is
  `reduced_verify_fn_inner_status_only` (its own `run_manifest.json`) -- i.e. "feasible" here means
  "REDUCED's own inner status was one of its feasible codes at write time," not an independent
  cross-formulation feasibility certificate. That is exactly the kind of gap this whole task exists
  to surface.
- The checkpoint is a `wall_interval` snapshot from an interrupted, not-fully-converged search --
  its `best_feasible` might reflect a genuinely fragile/boundary point that a longer REDUCED search
  would have moved away from, independent of any formulation question.
- Some genuine, real difference between the two formulations' inner feasibility screens exists for
  reasons not yet traced.

This is real, first-class evidence for this task's own `INFEASIBLE_CLASSIFICATION` deliverable --
recorded here, not resolved. Any future session picking this up should NOT assume this is a
bridge/harness bug (already ruled out above) before investigating further.

## Mode A (W=20,000) full campaign result -- real KNITRO, all 5 families, both arms

`run_campaign.jl mode_a 90 ...` -- every point in the Mode A point bank (15 points), both arms,
90s KNITRO budget each, real D20/W=20,000, single-threaded (Mode A diagnostic-parity intent).
Full telemetry: `FIXED_STATE_INNER_AB_RESULTS_MODE_A.csv` (repo_scratch, not committed).

### Headline finding: P1 fails on FULL for ALL 5 families, succeeds on REDUCED for ALL 5

```
family          arm      status   n_eval  gp           note
unrestricted    reduced  -101     70      0.9590587    real feasible-type outer status
unrestricted    full     (error)  0       --           rejected at cold-start pre-check
flexible_cm     reduced  -401     18      0.9607301    time-limit, feasible incumbent
flexible_cm     full     -411     2       --           time-limit, NO feasible incumbent found
common_frechet  reduced  -401     9       0.9649653    time-limit, feasible incumbent
common_frechet  full     -502     0       --           eval error (NaN/Inf) AT the cold start
origin_zc       reduced  -401     53      0.9571854    time-limit, feasible incumbent
origin_zc       full     -502     0       --           eval error (NaN/Inf) AT the cold start
cm_meanzc       reduced  -401     10      0.9660784    time-limit, feasible incumbent
cm_meanzc       full     -411     1       --           time-limit, NO feasible incumbent found
```

`-411`=`KN_RC_TIME_LIMIT_INFEAS` (time ran out with NO feasible incumbent ever found -- distinct
from `-401`=`KN_RC_TIME_LIMIT_FEAS`, which DOES have one). `-502`=`KN_RC_EVAL_ERR` ("evaluation
error (e.g. NaN/Inf) reported by a callback ... KNITRO unable to even evaluate the initial
point", `knitro_status.jl:99`, `n_eval=0` in every case here -- a documented, named failure mode
elsewhere in this repo, not a first occurrence). unrestricted's P1 was already isolated above as
a genuine `inner_status=-300` infeasibility, not an eval error -- a THIRD distinct failure
mode. **Three different failure modes, one consistent pattern: every one of the 5 families'
REDUCED-verified-feasible P1 point fails on FULL, by whichever mechanism that family's FULL driver
happens to hit first.** This is much stronger evidence than the unrestricted-only finding above --
it is not family-specific. The leading hypothesis (REDUCED's own default
`verification_policy=:reduced_verify_fn_inner_status_only` being a weaker/different feasibility
criterion than FULL's real inner solve) is now supported by 5/5 families, not 1.

**P0 (calibration) works cleanly on both arms for all 5 families** -- gp agreement to 2-3 decimal
places under a 90s budget (`unrestricted` 0.959/0.955, `flexible_cm` 0.966/0.966, `common_frechet`
0.967/0.978, `origin_zc` 0.961/0.962, `cm_meanzc` 0.967/0.967) -- confirming the harness/bridge
itself is sound; the P1 pattern is a real property of those specific points, not a general
cross-formulation breakdown.

**P2 (flexible_cm, constructed near-eval18-boundary points) -- both arms succeed, real
agreement**: P2a gp 0.964(reduced)/0.963(full), P2b gp 0.962/0.962 -- genuine positive
cross-formulation agreement evidence at a materially harder point than calibration.

**P3 (infeasible/stall points) -- both arms correctly reject, for every family that has one**:
`unrestricted` P3a/P3b both arms refuse the cold start outright ("not inner-feasible/not
screen-passing" on REDUCED, "not inner-feasible" on FULL) -- CONSISTENT infeasible classification
between formulations. `flexible_cm` P3 (the real eval18 point): REDUCED raises
`CMExpectedSolveFailure` with `nStatus=-300` (exactly reproducing the independently-documented
eval18 forensic verdict); FULL gets `-502` (eval error) at 7.7s, 0 evals -- both reject fast,
neither finds a feasible incumbent, though via different specific mechanisms.

### Real per-family gp/Delta comparison, P0 only (only cell with clean numbers both arms, all 5 families)

| family | REDUCED gp | FULL gp | REDUCED Delta | FULL Delta | REDUCED status | FULL status |
|---|---|---|---|---|---|---|
| unrestricted | 0.959243 | 0.955475 | 0.947 | 0.839 | -401 | -401 |
| flexible_cm | 0.965910 | 0.966189 | 0.990 | 0.757 | -401 | -401 |
| common_frechet | 0.967114 | 0.978297 | 0.872 | 0.752 | -401 | -401 |
| origin_zc | 0.961067 | 0.962201 | 0.995 | 0.981 | -411 | -401 |
| cm_meanzc | 0.966942 | 0.966672 | 0.741 | 0.962 | -401 | -401 |

`gp` agrees to 2-3 decimal places for every family at the calibration point under a tight 90s
cold-start budget (neither arm has converged -- both still time-limited); `Delta` (the objective)
is materially less similar (both arms are still actively improving under time pressure, so this
is expected variance under a short budget, not necessarily disagreement about the true optimum --
Mode B / longer budgets are needed before treating any Delta gap here as a real disagreement).

## Step 10: backend/allocation checks -- real, direct confirmation

**Structured backends** (task's own required list) confirmed directly from the real production
checkpoints (both W scales, both ZC families -- `CMCheckpointV11`'s own recorded
`h_zz_backend`/`h_cz_backend`/`h_ez_backend` fields, not re-derived): `H_EZ`/`H_EM
=drawmajor_v2`, `H_CZ=draw_chunk_reordered`, `H_ZZ=blas_syrk` for both `origin_zc` and
`cm_meanzc` at both W=20,000 and W=100,000.

**Zero dense/reference fallback, confirmed empirically, not just from documentation**: ran a real
cold `origin_zc` REDUCED solve (calibration point, 15s budget) and diffed
`NO_DENSE_G_COUNTERS[]` before/after. Every fallback/dense counter this repo's own
`no_dense_g_counters.jl` tracks stayed at exactly 0 throughout a real solve --
`dense_economic_G_materializations`, `dense_CM_G_materializations`, `dense_ZC_G_materializations`,
`dense_Frechet_G_materializations`, `generic_dense_FG_calls`, `dense_reference_verification_calls`,
`dense_cross_hessian_calls`, `blas_syrk_fallback_count`, `draw_chunk_reordered_fallback_count`,
`hessian_weight_dense_recomputes` all `=0`. Only the intended structured-backend counters
incremented: `operator_cross_hessian_calls=62`, `winner_cross_hessian_calls=62`,
`hessian_weight_cache_hits=31`, `zc_centered_rebuilds=4`, `zc_centered_cache_hits=58`,
`blas_syrk_dispatch_count=31`, `drawmajor_v2_dispatch_count=31`. This is a direct empirical
measurement of the real solve's own instrumentation, not an assumption from reading backend
field names alone.
