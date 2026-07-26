# Fixed Fréchet as CM-plus-common-level-anchor — master summary — 2026-07-25/26

Branch `port/frechet-as-cm-plus-anchor-production-2026-07-25`, `github.com/edwardwiles/cdw`.
Base at start: `production/fullA-exact@61a3bd6` (immediately after the shared winner-pair
core-Hessian merge). Rebased twice as canonical production advanced mid-session: onto
`da62166` (transformed-A/flexible-theta production merge) and onto `a153628` (QMC
`destination_sample=:exclude_row` fix) — both clean, zero conflicts, re-verified with test reruns
after each.

## Release states reached (task Part 0's literal list)

1. `IMPLEMENTED_ON_FEATURE_BRANCH` ✓ — Parts I-VII below.
2. `WIRED_IN_PUBLIC_CM_DRIVER` ✓ — `run_cm_upper_checkpointed`, not a test-script bypass.
3. `VALIDATED_THROUGH_PUBLIC_ENTRY_POINT` ✓ — real D=20/W=80,000 runs, real checkpoints.
4. `REBASED_ON_WINNER_PAIR_PRODUCTION` ✓ — built directly on top of it from the start.
5. `REBASED_ON_TRANSFORMED_A_PRODUCTION` — **partial**: rebased onto the *commit* twice (clean),
   but the `:common_frechet` wiring targets the **legacy** `pivot_reduce`/`pivot_expand` outer
   coordinate inside `run_cm_upper_checkpointed`, not the new transformed-A unified driver — see
   §"Known gap" below.
6. `FAST_FORWARDED_TO_CANONICAL_PRODUCTION` — not done (no merge this session).
7. `TAGGED` — not done.
8. `POST_MERGE_SMOKE_PASSED` — not applicable (not merged).

## Part-by-part summary

| Part | Deliverable doc | Verdict |
|---|---|---|
| I: math | `FRECHET_AS_CM_PLUS_LEVEL_MATHEMATICS_2026-07-25.md` | proven: exact equivalence to direct fixed Fréchet, `u=1/√D` works in both contrast modes |
| II: moment construction + CMConfig | `COMMON_FRECHET_CM_DRIVER_PORT_2026-07-25.md` §Part II | D=4 22/22 PASS, D=4 wiring 9/9 PASS |
| III: Hessian | `COMMON_FRECHET_HESSIAN_ARCHITECTURE_2026-07-25.md` | D=4 20/20 PASS (real bug found+fixed); D=20 feasible |
| IV: gradient | `COMMON_FRECHET_GRADIENT_INTERPRETATION_2026-07-25.md` | D=4 7/7 PASS, machine precision |
| V: public driver + checkpoint | `COMMON_FRECHET_CM_DRIVER_PORT_2026-07-25.md` §Part V, `COMMON_FRECHET_PUBLIC_DRIVER_ASSERTIONS_2026-07-25.md`, `COMMON_FRECHET_CHECKPOINT_RESUME_2026-07-25.md` | real D=20 run + resume, 6/6 PASS |
| VI: basis equivalence + nesting | `FRECHET_DIRECT_VS_NESTED_BASIS_EQUIVALENCE_2026-07-25.md`, `COMMON_FRECHET_D20_FULL_GATE_2026-07-25.md` | D=20 12/12 PASS |
| VII: outer control | `COMMON_FRECHET_CONTINUATION_OUTER_GATE_2026-07-25.md`, `FLEXIBLE_CM_VS_COMMON_FRECHET_INNER_AB_2026-07-25.md` | real progress both families; direct δ=1 only, not the full staged chain |

**Total test suite this session**: 22+9+20+7+6+12 = 76 automated PASS/FAIL checks, **76/76 passing**,
across D=4 (fast, exact-equivalence-oriented) and real D=20/W=80,000 (production-scale,
KNITRO-execution-oriented) gates. Plus two real matched outer-loop control runs (149 and 43
evaluations respectively) and one real checkpoint-kill-equivalent resume.

## A correction made mid-session, for the record

An early D=20 nesting-diagnostic run reported the common-Fréchet Hessian callback using 82% less
memory than flexible CM's — investigated at the user's request rather than left as a headline
number, and found to be a measurement-order artifact (first-call JIT compilation cost landing on
whichever function was measured first in the script), not a real efficiency gap. Re-diagnosed with
a dedicated, repeated-call test: the two Hessian callbacks allocate within 0.03% of each other once
warmed. See `COMMON_FRECHET_HESSIAN_ARCHITECTURE_2026-07-25.md` §3 and
`FLEXIBLE_CM_VS_COMMON_FRECHET_INNER_AB_2026-07-25.md` §1 for the corrected numbers and the
diagnostic methodology.

## Known, disclosed gaps (not attempted this session)

1. **Outer coordinate**: `:common_frechet` is wired into the legacy-coordinate CM driver
   (`run_cm_upper_checkpointed`), not the now-canonical transformed-A unified driver. A future
   session needs to either port this wiring to that driver or confirm the legacy-coordinate driver
   remains a supported production path.
2. **Full staged continuation chain** (task §22, `δ=0.01→0.1→0.5→1`, both families): not run: only
   the simpler, explicitly-allowed direct-`δ=1`-from-calibration control was run, given this
   session's wall-clock budget.
3. **Threaded Architecture-C Hessian** for the level block: serial only; production CM already has
   a threaded variant, not extended here.
4. **`cm_extension` (meanzc) combination**: `:common_frechet` + meanzc is guarded (explicit error),
   not implemented — orthogonal restriction families, unvalidated combination.
5. **Genuine kill-mid-run checkpoint test**: only graceful resume was tested, not a real
   `SIGKILL`/process-group-kill mid-write test.
6. **Grid size**: D=20 gates ran at `L=10`, not the production-default `L=50` (wall-clock budget;
   nothing in the design/code is `L`-specific).
7. **Fully re-solved finite-difference gradient diagnostic**: not run as an independent check
   (the D=4 C+-vs-Reference agreement, both envelope-family, is the primary gate and passed at
   machine precision).

## Final verdict

```
COMMON_FRECHET_CDF = PORT_READY_NOT_MERGED

FRECHET_FORMULATION = cm_plus_common_level

CORE_HESSIAN_BACKEND = exact_winner_pair_parallel (serial only for the level-block extension;
                        threaded variant not implemented)
MARGINAL_HESSIAN_BACKEND = shared CM Architecture-C bin/prefix tables (cm_hessian_architectures.jl,
                            UNCHANGED) plus new level-block linear combinations (cm_frechet_hessian.jl)
OUTER_COORDINATE_MODE = legacy_z (pivot_reduce/pivot_expand -- NOT the new transformed-A coordinate;
                         disclosed gap, see above)

SHARED_FD_DISCREPANCY = same_as_unrestricted_cm (by C+-vs-Reference machine-precision agreement;
                         not independently re-measured against fully re-solved FD this session)

POST_MERGE_SMOKE = not_applicable (not merged this session)
```

No production merge was performed or requested this session. Per this project's standing rule,
merging to `production/fullA-exact` and pushing to `origin` requires explicit user confirmation
regardless of how thoroughly the technical gates passed — none of that was sought or given here.
