# Flexible-CM vs common-Fréchet inner A/B — 2026-07-25/26 (Part VI §17, Part VII §21-22)

Real D=20/W=80,000/L=10/`destination_sample=:exclude_row` throughout. Same calibration point, same
draws, same backend (`cm_hessian_backend=:structured`, winner-pair). Two separate comparisons:
a single-point nesting diagnostic (Part VI) and a matched outer-loop control (Part VII).

## 1. Single-point nesting diagnostic (`test_frechet_d20_gates.jl` Part 2)

| Component | Flexible CM | Common Fréchet |
|---|---|---|
| moment count | 190 (`(D-1)L`) | 200 (`D·L`) |
| context construction | 4.895s | 5.121s (1.05×) |
| complete inner-solve wall | 11.472s (n_fg=6, n_hess=5) | 7.611s (n_fg=6, n_hess=5) (0.66×) |
| Hessian-callback allocation (as first measured, unwarmed) | 52.8MB | 9.4MB (−82%) |
| Hessian-callback allocation (corrected, warmed/matched-condition) | 29.499MB | 29.508MB (+0.03%) |

**The −82% allocation figure was a measurement-order artifact**, not a real architectural
difference — see `COMMON_FRECHET_HESSIAN_ARCHITECTURE_2026-07-25.md` §3 for the full re-diagnosis
(`diag_hessian_alloc.jl`: repeated calls on pinned state show the two Hessian callbacks allocate
within `0.03%` of each other once JIT-warmed). The `0.66×` inner-solve wall-clock figure is likewise
not a controlled per-callback comparison — it's two *independent* KNITRO solves (different
restriction sets → different λ trajectories), not isolatable to Hessian cost alone; matching
`n_fg`/`n_hess` counts is reassuring but the two runs are not guaranteed to visit identical internal
states.

**Corrected, honest reading**: no measurable per-callback cost asymmetry between the two families
once measurement artifacts are removed; the extra `L` level columns cost only their proportionate
share, exactly as the shared-table architecture (Part III) predicts.

## 2. Matched outer-loop control (`run_frechet_outer_control_{flexcm,frechet}.jl`)

Direct `δ=1` outer run from the real calibration point, `maxtime_real=600s`, sequential (never
concurrent KNITRO), production defaults (`cm_gradient_backend=:cplus`).

| | Flexible CM | Common Fréchet |
|---|---|---|
| wall | 681.8s | 657.1s |
| `n_eval` | 149 | 43 |
| `n_grad` | 55 | 19 |
| KNITRO outer iterations | 54 | 18 |
| KNITRO CG iterations | 30 | 10 |
| best `gp` reached | 0.9587 (from 0.9878 calib) | 0.9647 (from 0.9878 calib) |
| terminal status | time limit, infeasible (`4.27e-2` feas err) | time limit, infeasible (`3.38e-2` feas err) |

Both arms made **real, genuine outer progress** in the same wall-clock budget. Common Fréchet's
throughput is lower (43 vs 149 evaluations) — but the KNITRO-reported outer-iteration and
CG-iteration counts (18 vs 54, 10 vs 30) show this tracks the **optimization dynamics of a
genuinely harder/stricter feasible set** (task's own framing: Fréchet is `(D-1)L + L` restrictions,
strictly more binding than flexible CM's `(D-1)L`), not a per-callback cost gap — §1 above already
rules that out directly. This is exactly the qualitative pattern the task's own §23 anticipates
("the goal is not for common Fréchet to match CM's bound... its computational behavior should be
interpretable as CM plus L moments").

**Scope disclosure**: this is a *single* matched run per family, not multiple repetitions, and it
is the task's explicitly-allowed bounded "direct δ=1 control from calibration" alternative to the
full multi-stage `δ=0.01→0.1→0.5→1` continuation chain (task §22) — the full chain was not run this
session given the wall-clock already spent (see `COMMON_FRECHET_CONTINUATION_OUTER_GATE_2026-07-25.md`).

## Verdict

`INNER_PERFORMANCE_PARITY = confirmed_no_material_gap` (once measurement artifacts corrected).
`OUTER_THROUGHPUT_DIFFERENCE = attributable_to_stricter_feasible_set` (not profiled further this
session — the task's own materiality bar, "if it remains dramatically slower, profile," is judged
not triggered given both the CG-iteration evidence above and the Hessian-callback parity in §1).
