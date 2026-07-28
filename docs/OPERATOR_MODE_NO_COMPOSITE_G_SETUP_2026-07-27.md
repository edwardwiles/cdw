# Explicit `moment_representation` dispatch, no composite-G setup where safe — 2026-07-27 (Task C)

## What this task found, before changing anything

The generic once-per-inner-solve setup call the task described —
`obj.moments!(@view(H[:,1]), select_G_from_H(obj,H), θ, obj.U, obj)`, filling the entire
composite `H`/`G` matrix once per inner solve, unconditionally — is real and lives in exactly the
places the task's search hint pointed at: `cc_algo/inner_loop_functions.jl`'s six
`inner_loop_internal(obj::PsiObjectiveBundle*, θ)` methods (the original, now largely superseded
generic dispatch), and its `d4_exact/`-local descendants: `inner_loop_internal_archgeneric`
(`cm_hessian_architectures.jl`), `inner_loop_internal_cmlookup_production`
(`cm_lookup_production.jl`), `inner_loop_internal_cmfrechetlookup_production`
(`cm_frechet_lookup_production.jl`), and `inner_loop_internal_profiled` (`oracle_fast.jl`, the
unrestricted family's own bespoke dispatcher). **Crucially, the per-Newton-iterate FG callback was
already correctly forked between dense and operator before this task** — `inner_loop_KNITRO_archgeneric`
always registers the dense functor (`callbackEvalFG_inner!`), while every operator-FG family
(flexible CM, common-Fréchet) dispatches through an entirely SEPARATE top-level function that
registers a DIFFERENT, operator-based callback (`_callbackEvalFG_inner_cmlookup!`/
`_callbackEvalFG_inner_cmfrechetlookup!`). This task's real target is narrower than "the FG
callback still reads dense G" (that was already false) — it is specifically the ONE-TIME setup
call at the top of each of those functions, which fills the family-specific "restriction" columns
(CM bins, Fréchet level anchor) even for families whose operator FG callback never reads them.

**A real, already-documented prior bug directly bears on this task.** Common-Fréchet's own
`skip_cm_fill_ref` mechanism (Phase 5.2, 2026-07-26) tried exactly this — skip the restriction-
column fill under `:cm_frechet_lookup` — and it produced a real, reproduced (4/4) `nStatus=-400`
infeasible termination away from the calibration point, because that family's Hessian callback
(`archC_frechet_hess_cb_builder`) reads the dense CM/level columns **regardless of FG backend**.
The fix (already on this branch, `archC_frechet_base_state`'s own long comment) was to REMOVE the
skip for common-Fréchet entirely, not narrow it. This is decisive evidence that "safe to skip" is
a per-family, per-Hessian-backend property, not a global one — implementing Task C naively (one
global switch that skips the fill for every family) would silently reintroduce that exact bug.

## What was implemented

**New selector**, `full_aod_diag/d4_exact/no_dense_g_counters.jl`:

```julia
const MOMENT_REPRESENTATION = Ref{Symbol}(:operator)
```

`:operator` (default) permits skipping the family-specific restriction-column dense fill wherever
it is SAFE (nothing downstream in the same inner solve, Hessian included, still reads those
columns); `:dense_reference` always fills everything, exactly as the pre-existing code did —
retained as an explicit diagnostic backend, not deleted. Full per-family safety reasoning is in
the selector's own docstring.

**Wired, genuinely new-work (flexible CM):** `cm_production_bundle.jl::archC_base_state`'s
pre-existing `skip_cm_fill_ref` toggle (previously gated only on
`cctx.inner_fg_backend==:cm_lookup`) is now additionally gated on `MOMENT_REPRESENTATION[]==:operator`
**and** `cctx.cm_cross_hessian_backend==:winner_bin` — the skip is only engaged when the Hessian's
own H_EC cross-block is ALSO operator-based (confirmed live: `dense_cross_hessian_calls=0`,
`winner_cross_hessian_calls>0`, `docs/GLOBAL_NO_DENSE_G_INNER_SOLVE_PROOF_2026-07-27.md` C.2). This
is a **separate boolean from the FG-dispatch decision** (`use_lookup`, unchanged) — an earlier
draft of this change conflated the two (making `MOMENT_REPRESENTATION[]=:dense_reference` also
switch which FG callback got registered, not just whether the fill is skipped); caught and fixed
before running any gate, since it would have silently changed which family's FG backend was used
whenever the selector was toggled for diagnostics.

**Deliberately NOT wired (common-Fréchet):** `archC_frechet_base_state`/
`wrap_moments_with_cm_frechet_archB` are untouched behaviorally — the dense CM/level columns are
always filled, exactly as the post-Phase-5.2 bugfix state already has it. `MOMENT_REPRESENTATION[]`
has no effect on this family; this is intentional, not an oversight, and is asserted by the gate
below (section 3).

**Deliberately not touched (CM+ZC / origin-ZC):** out of this task's scope by the assignment brief
(a separate agent owns their Hessian internals: `cm_meanzc_production.jl`'s `_fill_cm_HEE!`,
`H_CZ`, `H_ZZ`). `production_backend_manifest.jl` already records `cross_hessian_backend=:dense_exact`/
`restriction_hessian_backend=:dense_exact` for origin-ZC's H_ER/H_RR, and CM+ZC's own H_EC is
structurally excluded from `:winner_bin` whenever `ncore_core<NCORE` — both real, pre-existing
reasons a naive skip would be unsafe there too, left for that agent to assess.

**Not applicable (unrestricted):** its compressed-only FG path (`inner_loop_internal_profiled`,
`oracle_fast.jl`) never goes through `select_G_from_H`/`inner_loop_internal_archgeneric` at all —
predates this counter set entirely. Not touched.

**Counter wiring** (existing counter names, no new names invented, per the task's own instruction):
- `record_generic_dense_fg!()` now fires unconditionally at the top of `inner_loop_internal_archgeneric`
  (`cm_hessian_architectures.jl`) — this function is ALWAYS the dense-FG generic dispatcher (see
  above), so firing here is unconditionally correct and closes the gap
  `docs/GLOBAL_NO_DENSE_G_INNER_SOLVE_PROOF_2026-07-27.md` B.2 explicitly flagged
  ("NOT wired at the one site that would make it meaningful ... `inner_loop_internal_archgeneric`").
- `record_dense_cm_g!()` now fires inside `wrap_moments_with_cm_archB`'s `fill_cm_columns_from_bins!`
  branch (`cm_hessian_architectures.jl`) — exactly when the dense CM-column fill actually executes.
  Was previously defined but dead (B.2).
- `record_dense_frechet_g!()` now fires inside `wrap_moments_with_cm_frechet_archB`'s fill branch
  (`cm_frechet_level.jl`) — fires on every common-Fréchet inner solve today, by design (see above),
  recorded honestly rather than left dead.
- `full_G_materializations` was deliberately left unwired at these sites, matching the audit's own
  explicit, reasoned decision (B.2 last row: wiring it here would make it always-nonzero for every
  family, conflating a real once-per-inner-solve K-computation with the counter's intended meaning
  of a per-*callback* violation) — not re-litigated, per the task's own instruction not to invent
  new counter semantics.

**Structural "cannot silently select the dense FG callback" guarantee** (task's second requirement):
verified by direct code read, not re-architected, since it already holds. `inner_loop_KNITRO_archgeneric`
unconditionally registers the dense functor; every operator-FG family's own top-level dispatcher
(`inner_loop_internal_cmlookup_production`/`inner_loop_internal_cmfrechetlookup_production`) is a
**separate function** that registers a **different** callback and never calls
`inner_loop_internal_archgeneric`. There is no shared runtime branch where a flag could steer the
SAME code path to the wrong callback — the choice is made once, by which top-level function a
family's own base-state builder calls (itself driven by the family's already-existing, already-
gated FG-backend default). No new guard was added at this generic dispatcher itself, since a hard
error there would break its many legitimate `:dense_reference` diagnostic callers.

**Naming note, disclosed to avoid future confusion:** `oracle_fast.jl::evaluate_fullA_fast` already
has an UNRELATED, pre-existing local kwarg also named `moment_representation` (`:dense`|`:compressed`,
unrestricted-family-specific: whether the ECONOMIC core itself is built via a dense per-cell loop
or the compressed/winner-form shortcut). This task's new selector is a distinct, module-level global
`MOMENT_REPRESENTATION::Ref{Symbol}` (all-caps, matching `CM_INNER_FG_BACKEND_DEFAULT`-style naming
convention) governing a different axis (family-specific *restriction*-column fill skipping across
all families' generic setup, not just unrestricted's own economic-core construction). No Julia name
collision (different case, different scope), but flagged here explicitly since the task brief asked
for exactly this name and the collision risk is real for a future reader grepping "moment_representation".

## Gate (this session, real run)

New script `full_aod_diag/d4_exact/test_moment_representation_default_2026-07-27.jl`, D=4
(`d4_exact_setup`), covering flexible CM, common-Fréchet, and unrestricted (3 of the 5 families —
exceeds the task's "at least 2" requirement).

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --project=. -t 4 full_aod_diag/d4_exact/test_moment_representation_default_2026-07-27.jl
```

### Results — 15/15 PASS

```
MOMENT_REPRESENTATION[] initial default: operator
  PASS: MOMENT_REPRESENTATION[] defaults to :operator
  PASS: CM_INNER_FG_BACKEND_DEFAULT[] is :cm_lookup (precondition for skip to engage)
  PASS: CM_CROSS_HESSIAN_BACKEND_DEFAULT[] is :winner_bin (precondition for skip to engage)

=== 1. Flexible CM, MOMENT_REPRESENTATION[]=:operator (default) ===
  generic_dense_FG_calls=0 dense_CM_G_materializations=0 full_G_materializations=0 nStatus=-103
  PASS: flexible CM :operator -- generic_dense_FG_calls==0
  PASS: flexible CM :operator -- dense_CM_G_materializations==0
  PASS: flexible CM :operator -- full_G_materializations==0
  PASS: flexible CM :operator -- inner solve feasible

=== 2. Flexible CM, MOMENT_REPRESENTATION[] forced to :dense_reference (selector control-check) ===
  generic_dense_FG_calls=0 dense_CM_G_materializations=1 nStatus=0
  PASS: flexible CM :dense_reference -- dense_CM_G_materializations>0 (selector genuinely controls behavior)
  PASS: flexible CM :dense_reference -- inner solve still feasible
  PASS: flexible CM -- zeta* unaffected by MOMENT_REPRESENTATION (same answer both ways)

=== 3. Common-Fréchet, MOMENT_REPRESENTATION[]=:operator (deliberately NOT wired -- must be unaffected) ===
  dense_Frechet_G_materializations=1 nStatus=0
  PASS: common-Fréchet :operator -- dense_Frechet_G_materializations>0 (deliberately still fills, safety-preserved)
  PASS: common-Fréchet :operator -- inner solve feasible

=== 4. Unrestricted, MOMENT_REPRESENTATION[]=:operator (structural, not new work) ===
  generic_dense_FG_calls=0 full_G_materializations=0 status=0
  PASS: unrestricted :operator -- generic_dense_FG_calls==0 (own bespoke dispatcher, untouched)
  PASS: unrestricted :operator -- full_G_materializations==0
  PASS: unrestricted -- inner solve feasible

TOTAL: 15 passed, 0 failed
```

Notes on the results: (1) section 1 vs section 2 converge to the same ζ* (|Δ|<1e-9) but with
different KNITRO termination codes (`-103` vs `0`) — both are in the feasible-code set
(`(0,-100,-101,-103)`); this reflects a genuinely different floating-point path (fewer BLAS ops
under the skip), not a correctness gap. (2) Section 3's `dense_Frechet_G_materializations=1`
firing is the EXPECTED, correct outcome given the deliberate non-wiring, not a bug — the check
asserts `>0`, not `==0`, precisely to confirm this family's safety-preserving behavior is intact.

## Scope note

Not attempted, and not claimed: extending the safe skip to CM+ZC or origin-ZC (out of this task's
scope), or removing `oracle_fast.jl`'s bespoke unrestricted dispatcher (never needed one). Real
D=20 confirmation of this specific change was not run separately — Task A's and Task B's own real
D=20 gates already exercise `inner_loop_internal_archgeneric` and the flexible-CM/common-Fréchet
production paths this change touches at real D=20/W=80,000 scale (via `bench_frechet_operator_fg_default_gate_2026-07-27.jl`
and the pre-existing `no_dense_g_full_family_audit_d20_2026-07-27.jl` pattern), and none of those
runs regressed; a dedicated D=20 counter-instrumented rerun of this exact new script would be a
reasonable but non-blocking follow-up.
