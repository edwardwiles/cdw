# Common Fréchet / flexible-CM harmonization: operator equivalence gate (2026-07-28)

Final, comprehensive re-run of the operator-vs-dense equivalence gates on the harmonized branch
tip (after all 7 harmonization commits), for all 5 families -- not just the 3 the refactor
directly touched. Same gate scripts already validated in the Phase 1 no-H release
(`TRUE_OPERATOR_NO_H_PREMERGE_GATE_2026-07-28.md`); rerun fresh here to confirm the harmonization
introduced zero regressions anywhere.

## D=4 (synthetic, real KNITRO)

| Family | Checks | Failures | Verdict |
|---|---|---|---|
| unrestricted | 31 | 0 | ALL OPERATOR-VS-DENSE UNRESTRICTED EQUIVALENCE GATES PASSED |
| flexible CM (serial + threaded) | 37 | 0 | ALL OPERATOR-VS-DENSE FLEXIBLE-CM EQUIVALENCE GATES PASSED |
| common Fréchet | 31 | 0 | ALL OPERATOR-VS-DENSE COMMON-FRECHET EQUIVALENCE GATES PASSED |
| CM+ZC | 31 | 0 | ALL OPERATOR-VS-DENSE CM+ZC EQUIVALENCE GATES PASSED |
| origin-ZC | 31 | 0 | ALL OPERATOR-VS-DENSE ORIGIN-ZC EQUIVALENCE GATES PASSED |

Every family: `x=0`/random-probe/real-solved-x* objective+gradient+packed-Hessian agreement at
`0.000e+00`, full inner solve same accepted status, `Δζ*=0.0`, `max|Δλ*|=0.0`.

## Real D=20/W=100,000 (production Sobol data, destination_sample=:exclude_row, L=50)

| Family | Checks | Failures | Verdict |
|---|---|---|---|
| flexible CM | 23 | 0 | ALL OPERATOR-VS-DENSE FLEXIBLE-CM D=20/W=100000 EQUIVALENCE GATES PASSED |
| common Fréchet | 23 | 0 | ALL OPERATOR-VS-DENSE COMMON-FRECHET D=20/W=100000 EQUIVALENCE GATES PASSED |
| CM+ZC | 23 | 0 | ALL OPERATOR-VS-DENSE CM+ZC D=20/W=100000 EQUIVALENCE GATES PASSED |

(unrestricted and origin-ZC were not re-run at real D=20 here -- the harmonization task never
touched their code paths at all, and Phase 1's own real-D20 gates already validated them fresh on
this exact commit lineage; re-running was judged unnecessary re-verification of unchanged code.)

## Dedicated verification-specific gates (separate from the family equivalence-gate scripts above)

| Gate | Result |
|---|---|
| `test_operator_verification_cm.jl` (L=10, L=50) | ALL PASS, dense/operator KKT residuals bit-identical |
| `test_operator_verification_cm_frechet.jl` (L=10, L=50) | ALL PASS, dense/operator KKT residuals bit-identical |
| `test_cm_frechet_threaded_hessian_gates.jl` (serial vs threaded, real point) | ALL PASS -- v2-serial bit-exact (0.0) vs serial, v2-threaded ~1.4e-14 vs serial (matches this repo's own documented ~1e-14 threaded-vs-serial tolerance) |

## Pre-refactor vs post-refactor vs dense-reference

The task's own equivalence-gate scripts already compare **operator vs dense-reference** at every
commit (both before and after each harmonization step -- see the per-commit messages in this
branch's git log for the specific gate rerun after each of the 7 steps). Because every
intermediate step was gated before the next was attempted, the chain of "step N passes operator-
vs-dense at 0.0" for N=1..7 IS the pre-refactor-vs-post-refactor proof: each step's output is
bit-identical to the previous step's output (both being bit-identical to dense-reference), so by
transitivity the final harmonized state is bit-identical to the original pre-harmonization state
for every quantity these gates check.

## Verdict

```
D4_ALL_FIVE_FAMILIES = pass (0 failures each)
D20_W100000_THREE_TOUCHED_FAMILIES = pass (0 failures each)
DEDICATED_VERIFICATION_GATES = pass
THREADED_VS_SERIAL_GATE = pass
NO_MATERIAL_SLOWDOWN = confirmed for flexible-CM's real D=20 full inner solve (operator ~1.6-2.4s,
    consistent with Phase 1's own pre-harmonization timings; no regression observed)
OPERATOR_EQUIVALENCE_GATE = pass
```
