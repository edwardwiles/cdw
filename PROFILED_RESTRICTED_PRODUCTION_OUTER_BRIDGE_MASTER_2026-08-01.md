# Profiled restricted-family production outer bridge — master report (2026-08-01)

**STATUS: FINAL for this session.** Sections 1-4, 12-15 complete and gated directly. Sections 5-6
(real restriction operators, q-decomposition gate) complete and gated for all 4 restricted
families. Sections 8-9 (real five-accessor family adapter, combined outer layout) are **honestly
BLOCKED** for all 4 families on a genuine, well-documented structural gap (see §8-9 below) — not
attempted, not faked. Section 10-11 located a real restriction-parameter gradient for 2 of 4
families (ZC-only, CM+ZC) and confirmed the other 2 have none to wrap.

## Mission recap

Continue the completed outer-gradient scaffold
(`architecture/profiled-restricted-outer-gradient-2026-08-01@2b95408`) by building the real
restricted-family outer-gradient adapters and production integration scaffold: real
`restriction_contrib0`, real family adapters, combined A/gp + restriction-parameter gradient, live
gates, a stable cross-process layout digest, an evaluation-aware layout validator, and the actual
production outer runner (not just the fixed-gp A/B harness). Branch:
`architecture/profiled-restricted-production-outer-bridge-2026-08-01`.

## 1. Base, branch, and source provenance

See `PROFILED_RESTRICTED_PRODUCTION_OUTER_BRIDGE_SOURCE_2026-08-01.md` for the full writeup: base
SHA (`2b95408`), worktree status, scaffold file checksums, relationship to the live inner branch
(`architecture/profiled-restricted-inner-endtoend-2026-08-01@bee0303` — flagged an uncommitted
candidate accessor file found there, `profiled_restricted_accessors_2026-08-01.jl`, for
situational awareness only; not depended on anywhere in this branch), and relationship to the ZC
Hessian production work (committed@`1807ef5` on `optimize/genuine-cold-zc-hessian-k3-closeout-2026-08-01`,
not merged, not tagged — no formal production tag exists to rebase onto yet).

## 2. Scaffold reproduction (task §2)

All four pre-existing gates re-run live on the new branch/worktree before any new code was
written: interface contract (8/8 PASS), D4 and real D20/`:exclude_row`/W=20,000 unrestricted
regression (both bit-identical, `max_abs_err=0.0`), mock restricted-family gate (4 families,
machine precision), A/B harness smoke. No regression.

## 3. Stable layout digest (task §3)

`full_aod_diag/d4_exact/profiled_stable_layout_digest_2026-08-01.jl`: `stable_layout_digest(fctx)`
is a canonical, fixed-order SHA256 (stdlib `SHA`, not `Base.hash`) over family/D/Ddest/
destination_ids/anchor map/retained economic-row map/France-ratio index/gravity pivot
position+map/economic+restriction dual ranges/restriction-outer-parameter names/normalization
convention. `runtime_structural_check(fctx)` is an explicit-name wrapper around the existing
`validate_family_layout_contract` for the "cheap in-process validation" concept the task names
separately. **Fresh-process gate**: 3 genuinely separate `julia` subprocesses (new PID/hash-seed
each), byte-identical digests for an unrestricted and a mock-restricted family — 5/5 PASS
(`PROFILED_STABLE_LAYOUT_DIGEST_FRESH_PROCESS_GATE_2026-08-01.csv`).

## 4. Evaluation-aware layout validator (task §4)

`full_aod_diag/d4_exact/profiled_evaluation_aware_validator_2026-08-01.jl`:
`validate_family_layout_against_evaluation(fctx, ev)` closes the gap where a structurally
well-formed adapter's ranges could still not match what actually got solved. Checks
economic/restriction ranges are in-bounds of `ev.result.beta`, partition it exactly (no
gap/overlap/unexplained trailing entry), zeta convention, family-kind and stable-digest agreement
against `ev`'s own manifest fields WHEN `ev` exposes them (a narrow typed hook — no live restricted
evaluator stamps these fields yet, so the checks are real, callable code, on by explicit request
via `require_manifest_digest`, never silently skipped for every caller). 9/9 gate PASS
(`PROFILED_EVALUATION_AWARE_VALIDATOR_GATE_2026-08-01.csv`): 2 positive + 7 negative
throw-proving cases against a real D4 KNITRO solve.

## 5-6. Real restriction-contribution operators + q-decomposition gate

`full_aod_diag/d4_exact/profiled_restriction_contrib0_operators_2026-08-01.jl`: in-place,
persistent-workspace `restriction_contrib0_flexcm!`/`restriction_contrib0_frechet!`/
`restriction_contrib0_originzc!`/`restriction_contrib0_cmzc!`, each built by calling the EXACT
existing production forward kernels unchanged — `cm_forward_contribution!` (`cm_lookup_kernels.jl`),
`restriction_forward!` (`zc_restriction_operator.jl`) — never a hand-rolled restriction formula. No
dense G, no winner recomputation.

**q-decomposition gate** (`PROFILED_RESTRICTED_Q_DECOMPOSITION_GATE_2026-08-01.csv`, real D4
calibration, real production FG as ground truth): **all 4 families PASS at machine precision** on
the restriction-block-only comparison (`-zeta - econ_dense - restriction_contrib0` vs. the real
family's own solved `q`): flexible CM `max_abs_q_err=0.0`; common Fréchet `2.2e-16`; ZC-only
`5.6e-17`; CM+ZC `5.6e-17`. The CM+ZC case required discovering and using the CORRECT
`[economic(cf.oci-1) | Z_mean | Z_pair | CM_grid]` dual split (taken directly from the existing
production verifier `verify_inner_solution_operator_cmmeanzc!`), not the naive `cctx.NCORE`-based
split, which conflates the economic and Z blocks — confirmed live via a `BoundsError` before the
fix, exactly the "widened core" issue this task's own spec anticipated for CM+ZC.

## 7. Allocation

Persistent per-family workspace structs (`FlexCMRestrictionWorkspace`, `FrechetRestrictionWorkspace`,
etc.) hold every scratch buffer (`nO×L` matrices, length-`W` vectors) — no fresh `W`-vector
allocated per `restriction_contrib0_*!` call; `dest` is filled in place by the caller-owned buffer.

## 8-9. Real family adapters + combined outer layout — HONESTLY BLOCKED (all 4 families)

**Root cause (confirmed empirically, not assumed):** the five-accessor contract's
`profiled_economic_layout(fctx)` requires a *reduced* (anchor-cell-excluded)
`ProfiledEconomicMomentLayout` — 13 economic moments at D4. Every restricted family's REAL
production inner-solve context in this worktree (`build_cm_production_context`, the origin-ZC/
CM+ZC equivalents) is *dense* — `cctx.NCORE - 1 == cf.oci - 1` (17 at D4), not
`total_reduced_economic_moments`. There is no reduced/pivoted inner-solve construction for any
restricted family in this worktree to build a real adapter against. A closed-form gauge-shift
transform from the dense to the reduced coordinate basis was attempted and empirically failed
(residual ~0.81 at D4 — not remotely machine precision, so correctly abandoned rather than shipped).

One read (situational awareness only, per this branch's own stated policy — not copied from, not
depended on) of the sibling `architecture/profiled-restricted-inner-endtoend-2026-08-01` worktree's
own **uncommitted** `profiled_restricted_accessors_2026-08-01.jl` confirmed that the real
reduced-context machinery needs new fields (`ncore_core`, `profiled_layout`, `econ_ctx`) on
`CMBinHessCtx`/`OriginZCCoreHessCtx` that do not exist in this worktree and live in files on this
task's own forbidden list (Hessian-architecture / restriction-moment-definition files) — i.e.
building it here would require either duplicating that other session's in-flight work or editing
forbidden files, both explicitly disallowed by this task's own file-ownership boundary. This is
squarely the sibling inner workstream's job, not this bridge's.

Consequently `combined_outer_coordinate_names`/`economic_outer_range`/`restriction_outer_ranges`
(§9) were not built for the 4 restricted families — they would need to be built against a real
`fctx`, which does not exist. The unrestricted family's own combined layout continues to work
unchanged (it was already real, from the base scaffold).

## 10-11. Restriction-parameter gradients — located, not fabricated

`PROFILED_RESTRICTION_PARAMETER_GRADIENT_LIVE_REGRESSION_2026-08-01.csv`:

- **flexible CM, common Fréchet**: BLOCKED for a *different*, more fundamental reason than §8's —
  grepping the real production drivers (`cm_outer_driver.jl`, `run_frechet_outer_control_*.jl`)
  shows the outer coordinate vector for both families is `w=[gp_focal; zfree]`, IDENTICAL to the
  unrestricted family's own — CM marginal targets / Fréchet level targets are fixed campaign-level
  configuration baked in at context-build time, never a live KNITRO decision variable, never
  differentiated. **No restriction-parameter gradient exists anywhere in this worktree to wrap**
  for either family (confirmed by grep, not assumed absent). A `combined_profiled_family_outer_gradient`
  for these two families would be identical to `shared_family_outer_gradient` — there is no
  restriction term to append, so building the wrapper is not meaningful until/unless a real
  restriction outer parameter is introduced at the production level (out of this task's scope).
- **ZC-only, CM+ZC**: a REAL, existing, UNCHANGED production sensitivity gradient DOES exist —
  `d_delta_dual_d_nu_vec`/`d_delta_dual_d_eta_nu_vec` (`cm_meanzc_moments.jl:514,535`), confirmed
  present, confirmed unchanged, confirmed callable against the real `aug`/`zc_op` objects this
  session's own q-decomposition gate already builds. **Wrapping it into
  `combined_profiled_family_outer_gradient` is blocked ONLY on §8** (needs a real family ctx
  satisfying the five-accessor contract) — for CM+ZC, additionally on the widened-core split (§8-9
  above) — both explicitly resolvable, mechanically, once the reduced-context machinery lands.
  This is recorded as `PARTIAL_REAL` in the CSV, not `BLOCKED`, to distinguish "the gradient exists
  and is real, only the wrap is pending" from flexible-CM/Fréchet's "nothing exists to wrap."

## 12. Bandwidth gate artifact correction (task §12)

`test_profiled_incremental_vs_fullrebuild_2026-08-01.jl`'s own committed prose (and
`PROFILED_UNRESTRICTED_OUTER_AB_MASTER_2026-08-01.md`) claimed the incremental-vs-full-rebuild
comparison hit "machine precision (cos_sim=1.0000000000, max rel err ~6e-6)". Re-running the
unedited script reproduces `cos_sim≈0.9999`, `max_rel_err≈2.0-3.65` at D4 instead — root cause,
confirmed directly: the full-rebuild comparator used a single fixed `h=0.01` while the incremental
method uses an adaptive, per-coordinate `h`. **Fix**: extended `profiled_composite_gradient_at` to
accept `h::Union{Float64,AbstractVector{Float64}}` (purely additive, scalar default byte-identical
to the old code path); the test script now reports two SEPARATE, correctly-labeled comparisons
every run — `mismatched_bandwidth` (the historical ~0.9999 comparison, an approximation/robustness
check, never claimed to be machine precision) and `same_bandwidth` (the genuine
formula-equivalence claim, A-block(2:end), excluding gp's own already-documented small
analytic-vs-FD gap). Re-verified at D4: same-bandwidth `max_rel_err=1.5e-15`/`1.8e-15`,
`cos_sim=1.0000000000` — genuine machine precision. Stale prose corrected in
`PROFILED_UNRESTRICTED_OUTER_AB_MASTER_2026-08-01.md` with an explicit, dated correction block
(checked `PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md` too — it does not contain the stale
claim, so it was not edited). D20/W=80,000 re-verification completed
(`repro_logs_2026-08-01/bandwidth_correction_D20.log`,
`PROFILED_INCREMENTAL_VS_FULLREBUILD_2026-08-01_D20_W80000.csv`): confirms the D4 finding exactly.
`mismatched_bandwidth` reproduces `cos_sim≈0.9999`, `max_rel_err≈27-2846` (worse than D4, expected
— more coordinates means more chances for a near-zero true-gradient entry to inflate a relative
error) — an approximation artifact, not a formula bug. `same_bandwidth` A-block hits
`max_rel_err=8.65e-13`/`1.31e-11`, `cos_sim=1.0000000000` — genuine machine precision at real D20
scale, not just D4.

## 13-15. Production outer runner, A/B comparability hardening, production plumbing (task §13-15)

`profiled_production_outer_runner_2026-08-01.jl`: `run_profiled_production_outer(mode, ...)`, two
explicit modes, never a silent default. `:fixed_gp_parameterization_ab` is a thin, byte-identical
pass-through to the existing `run_profiled_family_outer_search`. `:production_bound_search` is a
mechanical extension of the same callback shape with `gp` as a genuine free KNITRO variable and
the FULL `shared_family_outer_gradient` output used (that function already promises
`grad[1]=dK*/dgp` — no new gradient formula). Both modes record an `OuterRunManifest`.

`profiled_ab_comparability_and_plumbing_2026-08-01.jl`: `assert_ab_comparable` throws unless
dual-bank policy, `obj.x` reuse policy, exact-cache policy, screen set, restriction backend,
solver options, `hessopt_tag`, family, and `economic_parameterization` all agree between two arms
— and, critically, throws rather than silently passing if either side has an unrecorded
`:not_yet_wired` placeholder (a shared placeholder is not evidence of comparability).
`build_profiled_production_config`/`assert_checkpoint_compatible`: immutable config keyed to
`economic_parameterization ∈ {:full_gamma_normalized, :profiled_destination_scales}` +
`stable_layout_digest`; `checkpoint_namespace` is derived from `(parameterization, family, digest)`
so a full-formulation checkpoint cannot be loaded into a profiled context or vice versa.

**Honest scope note**: this scaffold does NOT wire the actual production KNITRO option stack
(screens, exact cache, dual-bank policy, checkpointing/continuation) — those are substantial,
separately-owned production subsystems, and no live restricted-family evaluator exists yet to
exercise them meaningfully through this bridge. Every `OuterRunManifest.production_subsystems`
field is honestly tagged `:not_yet_wired`/`:inherited_from_ab_harness` where a real value is not
yet determinable — never a fabricated concrete value.

**Gate**: 13/13 PASS (`PROFILED_PRODUCTION_RUNNER_AB_HARDENING_PLUMBING_GATE_2026-08-01.csv`) —
real D4 KNITRO smoke of both runner modes (unrestricted family, the only real evaluator available),
6 `assert_ab_comparable` positive/negative cases, 4 config/namespace/checkpoint-compatibility
cases. No production campaign launched (`maxit_override=4` on both smoke runs).

## 16. Integration with the inner branch and ZC production tag

Not performed (task's own explicit sequencing: only once the inner branch lands a reviewed/
committed accessor surface and the ZC Hessian work is tagged/merged). See §1/source-snapshot doc
for the current relationship to both.

## 17. Deliverables checklist

| Deliverable | Status | Location |
|---|---|---|
| Master report | this file | `PROFILED_RESTRICTED_PRODUCTION_OUTER_BRIDGE_MASTER_2026-08-01.md` |
| Source/provenance snapshot | done | `PROFILED_RESTRICTED_PRODUCTION_OUTER_BRIDGE_SOURCE_2026-08-01.md` |
| Stable layout digest + fresh-process gate | done, 5/5 PASS | `profiled_stable_layout_digest_2026-08-01.jl`, `PROFILED_STABLE_LAYOUT_DIGEST_FRESH_PROCESS_GATE_2026-08-01.csv` |
| Evaluation-aware layout validator | done, 9/9 PASS | `profiled_evaluation_aware_validator_2026-08-01.jl`, `PROFILED_EVALUATION_AWARE_VALIDATOR_GATE_2026-08-01.csv` |
| Real restriction_contrib0! operators (4 families) | done, real production kernels reused | `profiled_restriction_contrib0_operators_2026-08-01.jl` |
| q-decomposition gate | done, 4/4 PASS at machine precision | `PROFILED_RESTRICTED_Q_DECOMPOSITION_GATE_2026-08-01.csv`, `profiled_restricted_q_decomposition_gate_2026-08-01.jl` |
| Combined outer-coordinate manifests | BLOCKED (4 families) — see §8-9 | n/a |
| Restriction-parameter gradient regression | done: 2 located real, 2 confirmed nonexistent | `PROFILED_RESTRICTION_PARAMETER_GRADIENT_LIVE_REGRESSION_2026-08-01.csv` |
| Real D4/D20 combined-gradient gates | BLOCKED — depends on §8 | n/a |
| Corrected bandwidth-comparison artifacts | done, D4+D20 | `test_profiled_incremental_vs_fullrebuild_2026-08-01.jl`, `PROFILED_INCREMENTAL_VS_FULLREBUILD_2026-08-01_{D4,D20_W80000}.csv` |
| Production outer-runner scaffold | done, 2 modes, 13/13 gate PASS | `profiled_production_outer_runner_2026-08-01.jl` |
| A/B comparability manifest/assertions | done | `profiled_ab_comparability_and_plumbing_2026-08-01.jl` |
| Checkpoint/result schema manifest | done | same file, `ProfiledProductionConfig` |
| Exact branches/commits/ancestry/worktrees | done | §1, `PROFILED_RESTRICTED_PRODUCTION_OUTER_BRIDGE_SOURCE_2026-08-01.md` |
| Raw logs | done | `repro_logs_2026-08-01/` |
| Clean status | done | `git status` clean at every commit boundary |
| SHA256 manifest | done | `PROFILED_RESTRICTED_PRODUCTION_OUTER_BRIDGE_SHA256_MANIFEST_2026-08-01.txt` |

## Verdict block

```text
SHARED_ECONOMIC_ENGINE =
    unrestricted_regression_pass
    (D4 and real D20/W=20000, bit-identical, max_abs_err=0.0 -- reproduced live this session)

STABLE_LAYOUT_DIGEST =
    fresh_process_pass
    (SHA256, 3 genuinely separate julia subprocesses, byte-identical digests, 5/5 gate PASS)

REAL_RESTRICTION_CONTRIB =
    flexible_CM:pass
    common_Frechet:pass
    ZC_only:pass
    CM_plus_ZC:pass
    (all 4: restriction_contrib0! built by reusing existing production forward kernels
     unchanged -- cm_forward_contribution!, restriction_forward! -- no dense G, no hand-rolled
     formula, persistent workspace)

Q_DECOMPOSITION =
    pass_all_live_families
    (restriction-block-only comparison against real D4 production FG: flexible_CM max_abs_err=0.0,
     common_Frechet=2.2e-16, ZC_only=5.6e-17, CM_plus_ZC=5.6e-17 -- all machine precision.
     full_profiled_reduced_economic_dual comparison is fail_all_4_families -- BLOCKED, see below,
     not a numerical failure but "the reduced context this comparison needs does not exist yet")

COMBINED_OUTER_GRADIENT =
    flexible_CM:blocked
    common_Frechet:blocked
    ZC_only:blocked
    CM_plus_ZC:blocked
    (all 4 blocked on the SAME root cause: no reduced/pivoted inner-solve context exists for any
     restricted family in this worktree -- profiled_economic_layout requires a REDUCED economic
     layout, every real production restricted-family context here is DENSE. A closed-form
     dense-to-reduced transform was attempted and empirically failed (residual ~0.81, not machine
     precision) -- correctly abandoned, not shipped. This is the sibling inner-endtoend
     workstream's own deliverable, per this task's explicit sequencing (task §16); not attempted
     here beyond one read-only situational-awareness look at its uncommitted work-in-progress)

RESTRICTION_PARAMETER_GRADIENTS =
    missing_flexible_CM
    missing_common_Frechet
    (confirmed by grep of every real production outer driver: neither family has a live KNITRO
     restriction outer parameter at all -- CM marginal targets / Frechet level targets are fixed
     campaign config, not decision variables, so there is nothing to differentiate)
    located_but_wrap_blocked_ZC_only
    located_but_wrap_blocked_CM_plus_ZC
    (a real, unchanged production gradient exists -- d_delta_dual_d_nu_vec/
     d_delta_dual_d_eta_nu_vec, cm_meanzc_moments.jl:514,535 -- wrapping it is blocked only on
     COMBINED_OUTER_GRADIENT's same root cause, mechanical once that unblocks)

PRODUCTION_RUNNER =
    ready
    (two explicit modes, :fixed_gp_parameterization_ab byte-identical pass-through to the
     existing AB harness, :production_bound_search a mechanical gp-free extension of the same
     gradient call; 13/13 gate PASS on real D4 KNITRO smoke; production subsystems -- dual-bank/
     exact-cache/screens/checkpointing -- honestly tagged not_yet_wired, not fabricated)

AB_COMPARABILITY_ASSERTIONS =
    pass
    (assert_ab_comparable: 6/6 positive+negative cases pass, including refusing to treat two
     shared :not_yet_wired placeholder fields as evidence of comparability)

DENSE_ECONOMIC_G_MATERIALIZATIONS =
    0_in_production
    (grep-confirmed over every new file this session touched, including the restriction_contrib0!
     operators -- all reuse existing operator-only kernels)

BANDWIDTH_GATE_ARTIFACT =
    corrected
    (D4 and D20/W=80000 same-bandwidth A-block: max_rel_err 1.5e-15/1.8e-15 (D4),
     8.65e-13/1.31e-11 (D20) -- genuine machine precision; mismatched-bandwidth comparison
     correctly relabeled as an approximation check, not a formula-equivalence claim; stale
     "1.0000000000" prose in PROFILED_UNRESTRICTED_OUTER_AB_MASTER_2026-08-01.md corrected with an
     explicit, dated note)

PRODUCTION_DEFAULT_CHANGED = false
CAMPAIGN_LAUNCHED = false
```
