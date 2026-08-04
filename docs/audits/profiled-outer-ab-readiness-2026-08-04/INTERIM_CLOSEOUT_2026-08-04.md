# profiled-outer-ab-readiness-2026-08-04 — interim closeout

Session started from canonical `prototype/profiled-destination-scales` HEAD
`395dec3e1e68844128cc98c16be17e91bc9b6603` (tag `profiled-functional-ready-2026-08-04`), verified
live against `origin` at session start. Branch `performance/profiled-outer-ab-readiness-2026-08-04`
and worktree `/bbkinghome/edav/cdw_worktrees/profiled-outer-ab-readiness-2026-08-04` created per
task §1 (exactly one of each, confirmed via `git worktree list`/`git branch -a`).

## What was completed and verified this session (sections 1-4)

**Section 2 (documentation/capability cleanup).** Corrected two real, confirmed-live
contradictions: `CONTINUATION_2026-08-04.md`'s closing paragraph claimed the merge/tag/push/
cleanup sequence was still "awaiting confirmation," but `git merge-base --is-ancestor` confirms
that document's own start commit is a straight-line ancestor of the current canonical HEAD via a
clean 20-commit chain that carries the functional-ready tag — the sequence already happened.
`FamilyRegistry.jl`'s module docstring claimed `free_nu_supported` was "deliberately still false"
for origin_zc/cm_meanzc REDUCED rows, but the actual capability-row booleans (and their own notes)
already recorded `true`. Both corrected in place (commit `cac0cb4`), with the historical prose
preserved via a labeled correction rather than deleted.

**Section 3 (canonical ZC free-nu parity) — genuinely closed for both origin_zc and cm_meanzc.**
Tracing `bin/run_profiled_model.jl`'s dispatch end to end (task §3.1) surfaced a real bug the task
brief's own framing understated: it isn't just FULL's origin-ZC path that was fixed-nu — REDUCED's
own canonical dispatch was too, calling the 3-arg fixed-nu evaluator even though the registry
already (correctly) recorded `free_nu_supported=true` on the strength of a free-nu driver that
existed in the tree but wasn't actually wired into the CLI. Fixed both sides (commit `b40de6f`):
REDUCED origin_zc/cm_meanzc now dispatch through `run_profiled_upper_constrained_free_nu`; FULL's
origin_zc wrapper now uses `run_originzc_upper_checkpointed`'s own genuine data-driven
`originzc_default_nu_bounds` default instead of a pinned near-1.0 box copied from a diagnostic
script. **Verified live**, not just by code inspection: real D20/W=20,000 CLI smokes for origin_zc
(both formulations) and cm_meanzc (REDUCED; FULL cm_meanzc was already correct and untouched).
Both origin_zc arms wrote `nu_policy="free"` with an **identical** `nu_bounds` summary
(`[-15.222177262685872, 3.9134314695358654]`) in their `run_manifest.json`, and `eta_nu` visibly
moved across real outer KNITRO evaluations in every log rather than staying at its start value.
`ABComparability.jl` was independently confirmed to already hard-fail on a `nu_policy`/`nu_bounds`
mismatch between arms (pre-existing, unmodified) — task §3.3's mandatory hard-fail requirement was
already met by existing infrastructure.

```
ZC_FREE_NU_PARITY =
    origin_ZC:    pass (real D20/W=20,000 CLI smoke, both formulations, matched nu_policy+nu_bounds)
    CM_plus_ZC:   pass (real D20/W=20,000 CLI smoke, REDUCED; FULL side already correct, unchanged)
```

**Section 4 (powered profiled-relative A coordinates) — derived, implemented, not yet fully
production-gated.** Full derivation in `POWERED_PROFILED_COORDINATE_DERIVATION_2026-08-04.md`
(this directory): `:profiled_powered_relative_A` is a per-coordinate affine reparametrization of
REDUCED's existing native `r_free` coordinate, built from the SAME `(logX,logY,theta)` constants
FULL's own `:powered_aspace` mode already uses, and is proven a bijection (composition of
invertible affine maps, theta always >0). The practical payoff worked out in the derivation: the
outer-gradient chain rule reduces to a scalar `-theta` rescale of REDUCED's existing analytic
gradient — no new gradient code needed, mirroring FULL's own
`gradient_transform_unified`. Implemented additively in
`full_aod_diag/d4_exact/profiled_powered_relative_a_2026-08-04.jl` (commit `6af41f1`); native mode
remains every family's default. Verified via a synthetic round-trip + finite-difference gradient
chain-rule check (machine-precision round-trips, ~5.8e-10 FD-vs-analytic agreement) — this
confirms the *math* is correct but is **not** the full production-context D4/D20 gate list task §4
specifies (same full log A, same full A, same gravity residual, same Delta-star at a fixed decoded
state, checkpoint/manifest compatibility, cache invalidation on mode change) — those require a
real `ctx`/`pe` built from `d20_real_setup_design`, not yet run.

```
POWERED_PROFILED_MODE = pass_math_derivation_and_synthetic_check;
    production_context_D4_D20_gates_not_yet_run
```

## What was NOT started this session (sections 5-13) — the actual reason this is an interim, not final, closeout

Sections 5 through 13 are genuinely multi-hour-to-multi-day pieces of work each — matched
FULL/REDUCED timer instrumentation, porting FULL's coordinate-parallel threading structure to
REDUCED with its own race/generation-ID gates, a REDUCED bandwidth-search cache with its own
invalidation-correctness gates, decoded-state gradient A/Bs across 4+ representative points per
family, and short outer-search A/Bs run in both execution orders with ≥5 completed gradients per
arm — none of which were attempted this session. This mirrors the scale every comparable prior
session in this repo's history needed a full dedicated session for (see e.g.
`profiled-outer-production-readiness-2026-08-03-interim`, `restricted-dual-bank-outer-ab-2026-08-01`
in this repo's memory record) — attempting to compress all of sections 3-13 into one session would
have meant either not verifying section 3/4 live (the actual bug-fix work, now confirmed real and
correct) or fabricating completion of sections 5-13 without genuine KNITRO evidence. Neither is
acceptable; sections 5-13 are left honestly open for the next continuation.

```
GRADIENT_TIMER_PARITY = not_started
REDUCED_GRADIENT_THREADING =
    unrestricted:not_started  flexible_CM:not_started  common_frechet:not_started
    origin_ZC:not_started  CM_plus_ZC:not_started
REDUCED_BANDWIDTH_CACHE = not_started
DECODED_STATE_GRADIENT_AB =
    unrestricted:not_started  flexible_CM:not_started  common_frechet:not_started
    origin_ZC:not_started  CM_plus_ZC:not_started
SHORT_OUTER_AB_ALGORITHMIC_PARITY =
    unrestricted:not_started  flexible_CM:not_started  common_frechet:not_started
    origin_ZC:not_started  CM_plus_ZC:not_started
SHORT_OUTER_AB_PRODUCTION_PARITY =
    unrestricted:not_started  flexible_CM:not_started  common_frechet:not_started
    origin_ZC:not_started  CM_plus_ZC:not_started
OPT_IN_PRODUCTION_READY =
    unrestricted:no  flexible_CM:no  common_frechet:no  origin_ZC:no  CM_plus_ZC:no
    (blocked on sections 5-10 above for every family — none of the mandatory gates in task §13
    beyond fixed-state scientific equivalence + functional-ready tag have been attempted)
```

## Relationship to the parallel fixed-state inner A/B task

Per task §12, this session did not rerun or duplicate `benchmark/profiled-fixed-state-inner-ab-2026-08-04`
(a separate Claude's ownership). That branch did not exist at session start; it appeared on
`origin` partway through this session (`9aa3b40`, from the same canonical `395dec3` start point) —
its own record states it deliberately stopped after step 2/13 ("manifest freeze only") specifically
because it observed this worktree actively committing in real time and its own task brief's
contingency says to stop rather than race a concurrent outer-readiness session. It has produced no
scientific-equivalence results yet (only a frozen `ScientificManifest` + a `MASTER.md` skeleton) —
nothing for this task to consume or be consistent with yet. No outer conclusion in this doc depends
on it.

## Final verdict block

```
ZC_FREE_NU_PARITY =
    origin_ZC: pass
    CM_plus_ZC: pass

POWERED_PROFILED_MODE = not_implemented_production_context_gates_not_yet_run
    (math derivation + implementation done and synthetically verified; the task's own required
    D4/D20 production-context gate list — same full A, same gravity residual, same Delta-star,
    checkpoint/manifest compatibility, cache invalidation — was not run this session)

GRADIENT_TIMER_PARITY = fail_not_started

REDUCED_GRADIENT_THREADING =
    unrestricted:fail  flexible_CM:fail  common_frechet:fail  origin_ZC:fail  CM_plus_ZC:fail
    (not started this session)

REDUCED_BANDWIDTH_CACHE = rejected_not_started_this_session

DECODED_STATE_GRADIENT_AB =
    unrestricted:fail  flexible_CM:fail  common_frechet:fail  origin_ZC:fail  CM_plus_ZC:fail

SHORT_OUTER_AB_ALGORITHMIC_PARITY =
    unrestricted:fail  flexible_CM:fail  common_frechet:fail  origin_ZC:fail  CM_plus_ZC:fail

SHORT_OUTER_AB_PRODUCTION_PARITY =
    unrestricted:fail  flexible_CM:fail  common_frechet:fail  origin_ZC:fail  CM_plus_ZC:fail

OPT_IN_PRODUCTION_READY =
    unrestricted:no  flexible_CM:no  common_frechet:no  origin_ZC:no  CM_plus_ZC:no

MERGED_TO_CANONICAL_PROTOTYPE = no_sections_5-13_mandatory_gates_not_attempted_this_session

INNER_MATH_CODE_CHANGED = false
FULL_PRODUCTION_CHANGED = false
PRODUCTION_DEFAULT_CHANGED = false
NEW_BRANCHES_CREATED = 1   (performance/profiled-outer-ab-readiness-2026-08-04, per task §1)
NEW_WORKTREES_CREATED = 1  (/bbkinghome/edav/cdw_worktrees/profiled-outer-ab-readiness-2026-08-04, per task §1)
DENSE_CODE_USED = false
CAMPAIGN_LAUNCHED = false
```

Branch is pushed nowhere yet (local only, per this repo's own standing "confirm before pushing to
real remote" practice) — commits: `cac0cb4` (docs/registry cleanup), `b40de6f` (ZC free-nu
parity fix), `6af41f1` (powered-coordinate derivation+implementation), `1b0c8cb` (status doc
update). Worktree is clean (all generated `results/canonical_runner/*` smoke-test byproducts
reverted, not committed, per this repo's established convention for those files).
