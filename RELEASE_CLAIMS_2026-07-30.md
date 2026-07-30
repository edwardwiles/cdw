# Release claims (task §11): REFERENCE_EQUIVALENCE vs PRODUCTION_DEFAULT_PATH

These are two different claims. The postmortem's own root-cause finding #2 is that a prior release
document (`FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md`) conflated them for `unrestricted`,
`cm_meanzc`, and `origin_zc`. This document reports them separately, on purpose, and neither implies
the other. `WIRED_AND_GATED` (or this task's equivalent, `PRODUCTION_MERGE=merged_tagged_smoked`)
requires both to independently pass.

## REFERENCE_EQUIVALENCE

**Claim**: `OperatorPsiBundle` and the dense reference (`PsiObjectiveBundleImplicit`, aliased
`DenseReferencePsiObjectiveBundle` in new diagnostic code) agree to machine precision when a
low-level family builder is called directly with each value explicitly.

**Evidence** (pre-existing, unchanged by this task, retained per task §11):

| Family | D=4 test | D=20 test |
|---|---|---|
| flexible_cm | `test_operator_no_H_bundle_equivalence_flexcm.jl` | `..._flexcm_d20.jl` |
| common_frechet | `test_operator_no_H_bundle_equivalence_frechet.jl` | `..._frechet_d20.jl` |
| cm_meanzc | `test_operator_no_H_bundle_equivalence_cmzc.jl` | `..._cmzc_d20.jl` |
| origin_zc | `test_operator_no_H_bundle_equivalence_originzc.jl` | `..._originzc_d20.jl` |
| unrestricted | `test_operator_no_H_bundle_equivalence_unrestricted.jl` | `..._unrestricted_d20.jl` |

**Status this task**: not re-run (out of scope -- these tests were not touched by this task's
edits, and the builders they call directly are unchanged). Their result says nothing about which
type a real, unmodified production call reaches -- see PRODUCTION_DEFAULT_PATH below.

## PRODUCTION_DEFAULT_PATH

**Claim**: an unmodified public production entry point -- called with zero overrides, normal
production settings -- constructs `OperatorPsiBundle`, and only `OperatorPsiBundle`, with no
representation choice reachable at all.

**Evidence** (new this task):

- `test_all_family_real_production_entrypoints_operator_bundle.jl`: D=4, all 5 families, real run
  **28/28 PASS** (2026-07-30). Exercises the exact `prepare_production_run` closures now embedded
  in the 3 real driver functions (`run_cm_upper_checkpointed`, `run_originzc_upper_checkpointed`,
  `run_polish_checkpointed_unified`) -- not a reimplementation. Confirms structurally that none of
  the 3 driver signatures accept `moment_representation` any more (so there is no override to omit
  -- the prior form of this claim, "no override was passed", is now "no override is possible").
- `test_dense_reference_diagnostics_permit_gating.jl`: **13/13 PASS**. Confirms the converse: a
  dense bundle is reachable ONLY through the explicit, permitted, banner-emitting diagnostic path,
  and is fatal under a declared production purpose.
- `scripts/static_bundle_guard_2026-07-30.sh`: **0 violations** (after an audited, justified
  allowlist -- see the script's own comments and `DENSE_REFERENCE_REACHABILITY_AUDIT_2026-07-30.md`
  Finding 2/3). The 3 real driver files have zero occurrences of any forbidden pattern, allowlisted
  or not -- not "clean because excluded", clean because absent.

**Status this task**: D=4 gate real-run and PASSING, as of commit history on this branch
(`architecture/production-operator-bundle-hardening-2026-07-30`). The D=20/W=100,000 extended
release gate (task §16) has **not** been run this session -- see the final task report for why
(wall-clock cost of a real D=20/W=100,000 KNITRO campaign across 5 families was judged out of
proportion to run speculatively before the D=4 architecture itself was confirmed correct; now that
it is, the D=20 gate is the natural next step for a follow-up session).

## Why neither implies the other (concretely, using this task's own audit)

`build_cm_meanzc_production_context` and `build_originzc_production_context` have supported
`moment_representation=:operator` correctly, and passed their own equivalence gates, since well
before the 2026-07-29 postmortem's second incident. The incident existed anyway, for weeks, because
the REAL driver (`run_cm_upper_checkpointed`) never threaded the kwarg through to reach that
already-correct code -- REFERENCE_EQUIVALENCE was true the entire time PRODUCTION_DEFAULT_PATH was
false. This task's structural fix (removing the kwarg entirely, one shared factory) makes that
specific gap impossible to reopen the same way again, but it does not, and should not, retroactively
validate REFERENCE_EQUIVALENCE claims it never touched -- hence reporting the two separately here,
not as a single combined status line.

## WIRED_AND_GATED gate (task §11's own rule)

```
REFERENCE_EQUIVALENCE:    PASS (pre-existing, not re-verified this session -- see above)
PRODUCTION_DEFAULT_PATH:  PASS (D=4, this session, 28/28 + 13/13 + 0 static violations)
D20_EXTENDED_GATE:        NOT RUN this session
=> WIRED_AND_GATED: NOT CLAIMED (D=20 extended gate outstanding; see PRODUCTION_MERGE status
   in the final task report -- port_ready_waiting_for_campaign, not merged)
```
