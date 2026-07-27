# Global "no dense G in inner-solve hot path" — counter completeness + measured proof — 2026-07-27 (task §7, Parts B+C)

Agent: `agent/dense-g-audit-2026-07-27`, branched from
`release/winner-aware-HER-and-no-dense-G-inner-solve-2026-07-27` @ `9e42d92`.

Companion doc: `docs/REMAINING_DENSE_MOMENT_CONSUMERS_2026-07-27.md` (Part A, the classified
site-by-site audit). This doc covers Part B (counter completeness) and Part C (the real
instrumented measurement), following the honesty/detail level of
`docs/FLEXIBLE_CM_WINNER_BIN_HER_RELEASE_2026-07-27.md`: every number below is from an actual run
on this branch, not asserted from static analysis.

## Part B: counter completeness

### B.1 — struct check

Compared `NoDenseGCounters` (`no_dense_g_counters.jl`, as consolidated by this session's base
commit `1aeac63`) against the task brief's required 13-counter list:

```
full_G_materializations, dense_economic_G_materializations, dense_CM_G_materializations,
dense_Frechet_G_materializations, dense_ZC_G_materializations, generic_dense_FG_calls,
dense_reference_verification_calls, dense_cross_hessian_calls, operator_cross_hessian_calls,
operator_economic_FG_calls, operator_restriction_FG_calls, operator_verification_calls,
winner_cross_hessian_calls
```

**Result: all 13 already exist in the struct.** `1aeac63` (immediately preceding this session)
already added the ones the task brief names that weren't there before (`dense_Frechet_G_materializations`,
`operator_economic_FG_calls`, `operator_restriction_FG_calls`). No struct edit was needed or made
this session.

### B.2 — which counters are actually wired to a real call site (not just defined)

Grepped every `record_*!()` call across the full `full_aod_diag/d4_exact/` tree (excluding the
recorder definitions themselves) to find every place a counter is actually incremented in
production code:

| Counter | Wired? | Call site(s) |
|---|---|---|
| `operator_forward_calls`/`operator_FG_calls` | **YES** | `economic_operator.jl:62` (`economic_forward!`) |
| `operator_transpose_calls` | **YES** | `economic_operator.jl:81` (`economic_transpose!`) |
| `dense_economic_G_materializations` | **YES** | `cm_lookup_kernels.jl:410`, `cm_meanzc_lookup_kernels.jl:121`, `cm_frechet_lookup_kernels.jl:215`, `cm_originzc_lookup_kernels.jl:104` — each family's own operator-FG dense-fallback branch |
| `generic_dense_FG_calls` | **PARTIAL** | `cm_meanzc_lookup_production.jl:119`, `cm_originzc_lookup_production.jl:140` only. **NOT** wired at the one site that would make it meaningful for flexible-CM/common-Fréchet: `inner_loop_internal_archgeneric` (`cm_hessian_architectures.jl:866`) — see below. |
| `operator_verification_calls` | **YES, but test-only** | `operator_verification.jl` (5 sites, one per family's `verify_inner_solution_operator_*!`). All 5 functions are called **only** from their own `test_operator_verification_*.jl` files (grepped every non-test consumer repo-wide; found none) — so in a real production run today this counter is always 0, correctly reflecting that the operator-verification path is validated-but-not-deployed, not that verification never runs (it does — via the dense path, see `dense_reference_verification_calls` below). |
| `dense_reference_verification_calls` | **NO — dead** | Defined (`record_dense_reference_verification!`), never called anywhere. The natural site is inside `archC_verified_state`/`archC_frechet_verified_state`/`archC_meanzc_verified_state`/`archOZ_verified_state` (all four, right at/after their `obj(inner_x, constr=...)` recompute) — these are the functions `cm_checkpoint.jl`'s real production `cb_F!` (line 1033, confirmed by direct read) calls on **every outer KNITRO iterate**, for every restricted family, today. |
| `dense_cross_hessian_calls`/`operator_cross_hessian_calls`/`winner_cross_hessian_calls` | **YES, but flexible-CM-H_EC-scoped only** | `cm_hessian_architectures.jl:693,698`, `cm_hessian_threaded.jl:207,212` — these three counters were added specifically for the 2026-07-27 winner-bin H_EC phase and are wired **only** in those two files. They do not (and were never claimed to) track origin-ZC's separate H_ER/H_RR dense block or common-Fréchet's cross-Hessian — see Part A row 9. |
| `dense_CM_G_materializations` | **NO — dead** | Defined (`record_dense_cm_g!`), never called. Natural site: `cm_hessian_architectures.jl`'s `wrap_moments_with_cm_archB` closure, right where `fill_cm_columns_from_bins!` actually executes (i.e. inside the `if skip_cm_fill_ref === nothing || !skip_cm_fill_ref[]` branch, `cm_hessian_architectures.jl:259-262`). |
| `dense_ZC_G_materializations` | **NO — dead** | Defined (`record_dense_zc_g!`), never called. Natural site: `cm_originzc_moments.jl`'s `wrap_moments_with_originzc` closure (`mean_columns_direct!`/`pair_columns!` calls, ~lines 142/149) and the analogous CM+meanZC closure in `cm_meanzc_moments.jl` — both currently unconditional (no skip gate exists for this block at all). |
| `dense_Frechet_G_materializations` | **NO — dead** | Defined (`record_dense_frechet_g!`), never called. Natural site: `cm_frechet_level.jl`'s `wrap_moments_with_cm_frechet_archB` closure, where `fill_frechet_level_columns_from_bins!` executes (`cm_frechet_level.jl:251-253`). |
| `operator_economic_FG_calls` | **NO — dead** | Never incremented (grepped directly, only appears in its own field definition/report line). See §B.3 for why this is intentionally left unwired rather than double-counted with `operator_forward_calls`/`operator_transpose_calls`. |
| `operator_restriction_FG_calls` | **NO — dead** | Never incremented. Natural site: `zc_restriction_operator.jl`'s `restriction_forward!`/`restriction_transpose!` (lines 104-146) — the ZC block's own shared operator, currently uninstrumented even though `economic_operator.jl`'s analogous `economic_forward!`/`economic_transpose!` already are. |
| `full_G_materializations` | **NO — correctly unwired, not an oversight** | See companion doc §2: the only per-outer-point-not-per-callback site that "builds the complete matrix in one shot" (`obj.moments!` at the top of `inner_loop_internal_*`) is deliberately NOT this counter's target, because wiring it there would make the counter nonzero on every single inner solve for every family unconditionally, which is a real but misleading signal for a counter meant to catch a per-FG-callback violation. No code path materializes a full dense G matrix *inside* an FG/Hessian callback in one shot; `0` (via "no call site" rather than "call site fired 0 times") is the historically accurate state. |

### B.3 — why counter edits were NOT made this session

Per this task's own explicit scoping ("Your primary scope is `no_dense_g_counters.jl`
(instrumentation you own) and read-only classification of the rest of the repo... if you do want
to add a counter call inside a file you suspect one of them is also actively editing... just
document the exact call site... don't risk a conflicting edit"): **every one of the 5 dead
counters' natural call sites lives in a file plausibly being actively edited by one of the three
parallel agents** (Hessian backends for common-Fréchet: `cm_frechet_hessian.jl`,
`cm_frechet_level.jl`, `cm_hessian_architectures.jl`; CM+ZC/ZC-only: `cm_meanzc_production.jl`,
`cm_originzc_moments.jl`, `zc_restriction_operator.jl`; verification defaults:
`operator_verification.jl`, and by direct implication every `archC*_verified_state`/`archOZ_verified_state`
function in `cm_production_bundle.jl`/`cm_frechet_cplus.jl`/`cm_meanzc_production.jl`/
`cm_originzc_production.jl`). Rather than risk a conflicting edit on any of them, this session made
**zero** changes to any file outside its own two new driver scripts
(`no_dense_g_full_family_audit_2026-07-27.jl`, `no_dense_g_full_family_audit_d20_2026-07-27.jl`)
and this doc pair. The exact recommended call sites are listed in the table above and in the
companion doc's classification table; whichever session next touches those files can wire them
directly from this table.

`operator_economic_FG_calls`/`operator_restriction_FG_calls` specifically: the task brief suggests
these might belong in `economic_operator.jl`'s `economic_forward!`/`economic_transpose!` and the
family-specific restriction dispatch points. Judgment call, documented rather than acted on: those
two functions already have `operator_forward_calls`/`operator_transpose_calls` (fired every call,
confirmed live below), so a coarser `operator_economic_FG_calls` there would either duplicate that
signal exactly (if incremented once per `economic_forward!`+`economic_transpose!` pair) or need a
new, different aggregation rule (e.g. "once per family per inner solve") that isn't specified
anywhere else in this codebase's counter discipline — recommend whoever owns `economic_operator.jl`
next decide the intended granularity rather than this audit guessing. `operator_restriction_FG_calls`
has a cleaner, unambiguous natural site (`zc_restriction_operator.jl`'s `restriction_forward!`/
`restriction_transpose!`, mirroring `economic_forward!`/`economic_transpose!` exactly) — recommended
as-is in the table above.

## Part C: measured invariant, family by family

**Method**: `julia --project=. -t 4 <driver>.jl`, `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
Each family: `reset_no_dense_g_counters!()`, one real inner solve at its CURRENT production-default
backend selection (no override — whatever `CM_INNER_FG_BACKEND_DEFAULT[]`/
`CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]`/`ORIGINZC_FG_BACKEND_DEFAULT[]`/
`CM_MEANZC_INNER_FG_BACKEND_DEFAULT[]`/`CM_CROSS_HESSIAN_BACKEND_DEFAULT[]` resolve to on this
branch right now), then `no_dense_g_report()`, full field dump. Driver scripts and their attribution
are documented in their own headers — both extend the pre-existing `smoke_no_dense_g_five_families.jl`
(commit `ccc2bbb`, already on this branch's history before this audit session) to print every
counter field instead of a subset, and add an explicit PASS/FAIL line against the task's own
5-counter required-zero invariant (`full_G_materializations`, `dense_economic_G_materializations`,
`generic_dense_FG_calls`, `dense_reference_verification_calls`, `dense_cross_hessian_calls`).

### C.1 — D=4 (`no_dense_g_full_family_audit_2026-07-27.jl`, raw log in `docs/key_results_2026-07-27/partC_d4_full_family_run.log`)

| Family | FG default | `full_G_mat` | `dense_econ_G` | `dense_CM_G` | `dense_ZC_G` | `dense_Frechet_G` | `generic_dense_FG` | `operator_FG` (fwd/T) | `operator_econ_FG` | `operator_restr_FG` | `operator_verif` | `dense_ref_verif` | `dense_cross_hess` | `operator_cross_hess` | `winner_cross_hess` | **5-counter invariant** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Unrestricted | N/A (single compressed path, no branch) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **Not measured by these counters** (architecturally has no dense/operator branch for them to distinguish — see below) |
| Flexible CM | `:cm_lookup` | 0 | 0 | 0 | 0 | 0 | 0 | 5 (5/5) | 0 | 0 | 0 | 0 | 0 | 4 | 4 | **HOLDS** (genuinely — FG callback + H_EC cross-Hessian both confirmed operator-based) |
| Common-Fréchet | `:dense_reference` | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **Reports HOLDS, but this is a measurement gap, not compliance** — see C.3 |
| CM+ZC (CM+meanZC) | `:operator` | 0 | 0 | 0 | 0 | 0 | 0 | 5 (5/5) | 0 | 0 | 0 | 0 | **4** | 0 | 0 | **VIOLATED** — `dense_cross_hessian_calls=4` |
| Origin-ZC / ZC-only | `:operator` | 0 | 0 | 0 | 0 | 0 | 0 | 5 (5/5) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **Reports HOLDS, but does not cover this family's own dense H_ER/H_RR block** — see C.4 |

(One inner solve per family = 5 KNITRO FG callback iterations at D=4 for the three operator
families, matching each other exactly since the shared `economic_operator.jl` drives all three;
common-Fréchet's dense path shows 0 recorded FG calls of any kind because nothing instruments it,
not because it made 0 calls — see C.3.)

### C.2 — flexible CM: genuinely holds

`operator_cross_hessian_calls=4=winner_cross_hessian_calls`, `dense_cross_hessian_calls=0`: the
`:winner_bin` H_EC backend (this session's own base commits `f4c149e`/`1aeac63`) fired on every one
of the 4 Hessian callbacks this inner solve triggered, with zero dense fallback — live confirmation,
at D=4, that the just-landed flip is exercised by a real inner solve, not just its own dedicated
gate script. `operator_FG_calls=5` (5 forward + 5 transpose) confirms the `:cm_lookup` FG default is
genuinely the operator path for the whole inner solve, no dense fallback triggered.

### C.3 — common-Fréchet: the reported "HOLDS" is not evidence of compliance

All-zero across every counter for common-Fréchet is the **expected result of an instrumentation
gap, not a demonstration that the dense_reference default avoids dense G**. Structurally (companion
doc, Part A finding #2 and row 1): `CM_FRECHET_INNER_FG_BACKEND_DEFAULT[] === :dense_reference`
dispatches through `inner_loop_internal_archgeneric` (`cm_hessian_architectures.jl:866`), whose
registered per-Newton-iterate FG callback is `cc_algo/PsiObjectiveBundle.jl`'s own callable method —
a real `BLAS.gemv!` against `H[:, 2:1+outer_constr_index]` every callback, confirmed by reading that
file directly. Neither `inner_loop_internal_archgeneric` nor `PsiObjectiveBundle`'s callable method
calls any `no_dense_g_counters.jl` recorder, so this family's actual dense FG usage is invisible to
every counter measured here. **Common-Fréchet does not currently satisfy the "no dense G in FG
callback" invariant** — it is the one family among the four restricted families whose FG default has
not been flipped off `:dense_reference` (the flip criteria and gate results are explicitly
documented as pending in `core_exact_hessian.jl:167-179`'s own docstring). This is expected per the
task background (parallel Hessian-backend agent's active track covers common-Fréchet) but should
not be read from this report's raw PASS line alone — hence this explicit callout.

### C.4 — origin-ZC: reported "HOLDS" is real for what it measures, silent on H_ER/H_RR

Origin-ZC's FG callback (`operator_FG_calls=5`, all forward/transpose) and core H_EE Hessian both
confirmed operator/winner-pair-based — genuinely no dense `obj.H` read in the ordinary per-callback
hot loop for those two blocks. But `dense_cross_hessian_calls`/`winner_cross_hessian_calls` are
wired ONLY inside `cm_hessian_architectures.jl`/`cm_hessian_threaded.jl` (flexible-CM's own H_EC
phase, §B.2 above) — they structurally cannot see origin-ZC's separate, deliberately-dense H_ER/H_RR
block (`production_backend_manifest.jl:224-225`: `cross_hessian_backend = :dense_exact`,
`restriction_hessian_backend = :dense_exact`, confirmed via the codebase's own self-reporting
manifest). That block is real, current, dense, and out of both this session's and this counter set's
scope — belongs to the parallel CM+ZC/ZC-only Hessian-backend agent's active track per this task's
background section.

### C.5 — CM+meanZC: violation is real, expected, and documented

`dense_cross_hessian_calls=4` is a genuine, correctly-measured violation of the 5-counter invariant
for this family — not a gap. CM+meanZC's `ncore_core < NCORE` (mean/pair columns widen the "core"
block) structurally excludes the `:winner_bin` scope guard (`_cm_cross_hessian_wants_winner_bin`,
`cm_hessian_architectures.jl:634-644`: requires `cctx.ncore_core == cctx.NCORE`), so every one of
this inner solve's 4 Hessian callbacks fell back to `:dense_reference` for H_EC, each one correctly
recorded. `FLEXIBLE_CM_WINNER_BIN_HER_RELEASE_2026-07-27.md` (this session's own prior deliverable)
already discloses this as out of Section 2's scope ("CM+meanZC's own `build_cm_meanzc_bin_ctx`
passes a locally-hardcoded `:dense_reference` default, unaffected by this flip"); this measurement
is the live confirmation that the disclosed gap is real and currently active in production, not
theoretical.

### C.6 — unrestricted: architecturally not covered by this counter set

Confirmed via the pre-existing smoke test's own comment (accurate, re-verified by reading
`compressed_cc_inner.jl`/`oracle_fast.jl`'s default `inner_loop_internal_profiled`): the unrestricted
family has had a single compressed-only FG path since "Addendum Part A," predating
`no_dense_g_counters.jl` entirely, with no dense/operator fork for the instrumentation to
distinguish. Its own post-solve KKT recompute (`oracle_fast.jl:296`, `evaluate_fullA_fast`) is the
same `select_G_from_H`-view-then-BLAS pattern as row 6/7 of the companion doc (once per outer point,
not per FG callback) — structurally identical in kind to the other four families' verification
dense-read, just not routed through this particular counter struct.

### C.7 — real D=20/W=80,000

<!-- D20_RESULTS_PLACEHOLDER -->

## Summary: does the invariant hold?

| Family | Holds? |
|---|---|
| Unrestricted | Not applicable — single compressed path, no dense/operator fork exists to violate |
| Flexible CM | **YES**, genuinely, measured (FG + H_EC cross-Hessian both operator, 0 dense) |
| Common-Fréchet | **NO** — production default is `:dense_reference`; the report's own all-zero counters reflect an instrumentation gap (`inner_loop_internal_archgeneric`/`PsiObjectiveBundle` callable uninstrumented), not compliance |
| CM+ZC (CM+meanZC) | **NO** — FG callback is operator (genuinely clean), but H_EC cross-Hessian is always dense by structural exclusion (`ncore_core<NCORE`), measured live (`dense_cross_hessian_calls=4`) |
| Origin-ZC / ZC-only | **PARTIALLY** — FG callback + H_EE core Hessian genuinely operator-based; H_ER/H_RR cross/restriction Hessian block is deliberately dense by design, not covered by any of the 5 required-zero counters |
