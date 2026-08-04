# REDUCED outer production integration and FULL-vs-REDUCED A/Bs -- interim status, 2026-08-03

This is an **interim, honest status report**, not a completed closeout. The full task
(profiled-outer-production-readiness-2026-08-03, "Claude Code task 2 of 2") spans 13 sections;
sections 1-3 are genuinely complete and verified below. Sections 4-13 have not been started, and
this document exists specifically to record the real starting point for whoever continues --
including what already exists in the repo that a continuation should reuse, not rebuild.

**Nothing here has been merged to `production/fullA-exact`, `prototype/profiled-destination-scales`,
or tagged `profiled-outer-ab-ready-2026-08-03`. No A/B campaign was launched. No dense G/H backend
was used.**

## Provenance

- Repo: `/bbkinghome/edav/cdw` (canonical, per CLAUDE.md 2026-08-03 update)
- Branch: `feature/profiled-outer-production-readiness-2026-08-03`
- Worktree: `/bbkinghome/edav/cdw_worktrees/profiled-outer-production-readiness-2026-08-03`
- HEAD: `427d502`
- Branched from `origin/prototype/profiled-destination-scales` @ `b80dd48`
- Task-1 prerequisite tag: `profiled-inner-ready-2026-08-03` @ `640ea15`, confirmed a real
  ancestor of `b80dd48` (`git merge-base --is-ancestor` = true); `docs/audits/profiled-inner-readiness-2026-08-03/MASTER.md`
  confirmed `INNER_READY=yes` for all five families before this task began.
- Current `origin/production/fullA-exact` tip: `4c3dad5` -- exactly one commit ahead of the
  prototype's merge-base (`146b10e`), and that one commit (`4c3dad5`) is a CLAUDE.md-only
  documentation addition with **zero source diff**. The FULL implementation embedded in the
  prototype is therefore confirmed byte-identical to current FULL production for every hot file;
  no re-merge of FULL into the prototype was needed (task section 2).

## What's genuinely done (sections 1-3)

**Section 1 (repo discipline):** exactly one branch, one worktree, both at the required exact
paths; nothing else created.

**Section 2 (architecture re-verification):** see provenance above. Recorded, not assumed.

**Section 3 (canonical manifest infrastructure):** two commits, both real and tested.

1. `fec40e5` -- vendored `scientific_manifest/ScientificManifest.jl` (+ its test file +
   `configs/fullA_production_2026-08-03.toml`) byte-identical from
   `origin/hardening/require-scientific-params-2026-08-03` (`330cb23..b646eca`), which is **not
   merged into `production/fullA-exact` or the prototype** but is the documented single source of
   truth for scientific parameters. Its own test suite passes unmodified here: **29/29**.
2. `427d502` -- new `scientific_manifest/RunManifest.jl`, wrapping `ScientificManifest` with the
   run/formulation-specific fields task section 3 additionally requires: `family`,
   `economic_parameterization` (formulation), `A_coordinate_mode`, `nu_policy`/`nu_bounds`,
   draw checksums, outer algorithm/budget, cache/bank/warm-start policy identifiers, an
   `initial_state_digest` (via `digest_economic_state`), and `source_sha`/`source_dirty` (via
   `current_source_sha`/`source_is_dirty`/`refuse_if_dirty`, which shell out to real `git`
   commands against the actual repo, not a stub). TOML + JSON (`run_manifest.json`) round-trip,
   `validate_manifest` internal-consistency checks, and the git-provenance helpers are all
   exercised against the real filesystem and the real repo. **42/42 tests pass.**

Family and formulation symbols used (`VALID_FAMILIES`, `VALID_ECONOMIC_PARAMETERIZATIONS`) are
the **real symbols already used in the codebase** (`production_backend_manifest.jl`,
`profiled_ab_comparability_and_plumbing_2026-08-01.jl`): `:unrestricted`, `:flexible_cm`,
`:common_frechet`, `:origin_zc`, `:cm_meanzc`; `:full_gamma_normalized` (FULL) /
`:profiled_destination_scales` (REDUCED). Task section 3's prose labels ("origin_ZC", "CM_plus_ZC",
"flexible_CM") map onto `:origin_zc`, `:cm_meanzc`, `:flexible_cm` respectively -- **do not
invent new symbols for these**; the code's own names are the source of truth.

**Explicitly not yet done, even though the manifest type exists:** no runner calls
`write_run_manifest_json` or `refuse_if_dirty` yet. `A_coordinate_mode` is stored but not
validated against an enumeration -- see section 7 note below for why.

## What a continuation needs to know before touching section 4 (canonical runner + family registry)

This is the load-bearing finding of this session: **section 4's raw material already exists in
the repo, scattered across per-family/per-formulation scripts, and section 4 is fundamentally an
integration task, not a from-scratch build.** Do not re-derive any of the following; read and
reconcile it.

Confirmed-present infrastructure (file paths under `full_aod_diag/d4_exact/`, all as of `b80dd48`):

- `production_backend_manifest.jl` already resolves per-family backend choices
  (`resolve_unrestricted_manifest(...)` and siblings) from values the calling driver has in
  scope -- this is a **partial** family registry already, for FULL. It does not cover REDUCED and
  does not cover the "context constructor / outer evaluator / inner verifier / outer gradient /
  screens / cache-bank / checkpoint-resume / free-nu / coordinate modes / production readiness"
  capability matrix section 4 asks for, but its resolver pattern is the right thing to extend, not
  replace.
- `cm_checkpoint.jl` defines `CMCheckpointV11`, which **already has** `A_coordinate_mode`,
  `economic_parameterization`, `eta_nu` (a checkpoint slot for free-nu, unused as of this
  session -- see section 6 note below), `outer_layout_digest`, `inner_layout_digest`,
  `h_zz_backend`/`h_cz_backend`/`h_ez_backend`, `recovery_convention`, and
  `checkpoint_namespace` (collision-prevention key). This is almost certainly the "versioned
  successor" task section 5 says is acceptable in place of porting V11 verbatim -- confirm by
  reading `save_cm_checkpoint`/`load_cm_checkpoint` and `assert_checkpoint_compatible`
  (referenced in the struct's own docstring, in `profiled_ab_comparability_and_plumbing_2026-08-01.jl`)
  before writing any new checkpoint code.
- REDUCED-side constrained-search scripts already exist and were not read in depth this session:
  `run_outer_flexcm_reduced_constrained_2026-08-02.jl`, `run_outer_originzc_reduced_constrained_2026-08-02.jl`,
  `profiled_production_outer_constrained_2026-08-02.jl`. FULL-side counterparts:
  `run_outer_flexcm_full_2026-08-02.jl`, `run_outer_originzc_full_2026-08-02.jl`. None of these
  declare `A_coordinate_mode` at all (grepped, zero hits) -- consistent with
  [[full-vs-reduced-forensic-audit-2026-08-03]]'s finding that REDUCED's A-coordinate system has
  no formalized mode symbol yet. **This is why `RunManifest.A_coordinate_mode` is deliberately
  left unvalidated against an enumeration in this session's commit** -- asserting a REDUCED mode
  name now, before section 7's derivation work, would misrepresent what's actually wired and is
  exactly the "looks canonical but isn't" failure mode the task explicitly warns against (section
  4's "Do not leave the gp-fixed run_profiled_production_outer scaffold looking canonical").
- `run_profiled_production_outer` itself (referenced by name in task section 4) was not located
  by an exact-name grep in this session -- confirm whether it still exists under that name, was
  renamed, or was one of the files above, before assuming section 4's "rename or make
  private/test-only" instruction applies to a specific still-live file.
- Free nu: confirmed (again) that **no `CMZCFreeNuAdapter` or free-nu variant exists anywhere in
  the tree** -- matches [[full-vs-reduced-forensic-audit-2026-08-03]] finding #7 exactly, nothing
  changed since. `CMCheckpointV11.eta_nu` is a real field but is not populated by any live free-nu
  search path. Task section 6's instruction to inspect (not apply)
  `diagnostic-profiled-outer-fastfail-nu-ab-2026-08-02` for the D4-complete salvaged
  implementation still stands as the right starting point.
- Threading/caching: confirmed (again, not re-derived) that REDUCED has no threaded
  outer-gradient loop and no bandwidth cache, per the same forensic audit -- section 9's premise
  is still accurate.

## Sections not started

Sections 4 through 13 (canonical runner, checkpoint/resume porting, free-nu for origin_zc and
cm_meanzc, REDUCED coordinate-mode formalization, all-family outer-gradient gates, threading,
bandwidth caching, the A/B comparability gate, the staged A/B program, the production-readiness
matrix, and final integration) have **not been started**. Each historically took a full dedicated
session elsewhere in this project's history for a single family or a single concern (see this
repo's own memory: e.g. the free-nu/ZC-Hessian and reduced-operator work spanned multiple full
sessions each) -- attempting all of them in one pass here would have meant either rushing past
real KNITRO verification or fabricating results, both of which this project's CLAUDE.md
explicitly and repeatedly warns against. Stopping here, with two small, fully real, fully tested
commits and a precise map of what continuation work needs to read first, was the judgment call
made instead.

## Final verdict

```
CANONICAL_RUNNER = fail_not_started
MANIFEST_AND_AB_GATE = fail_manifest_type_done_ab_gate_not_started
CHECKPOINT_RESUME =
    unrestricted:fail_not_started
    flexible_CM:fail_not_started
    common_frechet:fail_not_started
    origin_ZC:fail_not_started
    CM_plus_ZC:fail_not_started
FREE_NU =
    origin_ZC:fail_not_started
    CM_plus_ZC:fail_not_started
PROFILED_COORDINATE_MODES =
    native_log_relative:fail_not_started (exists implicitly in REDUCED code, has no formal mode symbol yet)
    powered_relative:not_implemented_reason_not_started
OUTER_GRADIENT =
    unrestricted:fail_not_started
    flexible_CM:fail_not_started
    common_frechet:fail_not_started
    origin_ZC:fail_not_started
    CM_plus_ZC:fail_not_started
OUTER_GRADIENT_PERFORMANCE =
    threading:fail_not_started
    bandwidth_cache:fail_not_started
AB_COMPARABLE = fail_not_started
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
MERGED_TO_CANONICAL_PROTOTYPE = no_sections_4_through_13_not_started
EXTRA_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
FULL_PRODUCTION_CHANGED = false
DENSE_CODE_USED = false
CAMPAIGN_LAUNCHED = false
```

## Recommendation for continuation

Start the next session by reading, in order: `production_backend_manifest.jl`,
`cm_checkpoint.jl`'s `CMCheckpointV11` + its save/load/`assert_checkpoint_compatible` functions,
then the five REDUCED/FULL constrained-runner scripts named above. Only after that reading should
section 4's family registry and canonical runner be written -- writing them first risks exactly
the misrepresented-capability failure mode the task warns about twice.
