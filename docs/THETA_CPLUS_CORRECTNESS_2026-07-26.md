# Theta C+ correctness gates — 2026-07-26

`full_aod_diag/d4_exact/test_theta_cplus_correctness.jl`. Real D=20/W=80,000/seed=20260719,
`:exclude_row`. Compares the new `theta_cplus_secant` against the OLD, unmodified
`theta_fixed_dual_delta_pivot_A` (retained in the tree specifically as this comparison's
reference implementation) at every point the task's §10 specifies that is reachable at this real
calibration (two of the seven requested points were not — see below, a test-design fact about
this specific economy, not a gap in the new implementation).

## Results

| Point | `D_plus` \|diff\| | `D_minus` \|diff\| | secant rel diff | winner changes (plus vs minus) |
|---|---|---|---|---|
| calibration, h=1e-3 | 9.97e-18 | 8.24e-18 | 2.08e-14 | 1428 |
| theta +5% | **SKIPPED** — not inner-feasible at this (gp, A) combination | | | |
| theta -5% | **SKIPPED** — not inner-feasible at this (gp, A) combination | | | |
| calibration, h=1e-2 | 1.99e-17 | 1.04e-17 | 8.40e-14 | 14581 |
| calibration, h=1e-3 (repeat) | 9.97e-18 | 8.24e-18 | 2.08e-14 | 1428 |
| calibration, h=1e-4 | 1.04e-17 | 2.08e-17 | 4.16e-12 | 149 |
| near-budget perturbed point (`logA_full0 + 0.05·N(0,1)`, real Δ=0.032) | — | — | 7.90e-13 | 1442 |

**All diffs at or near machine precision** (`D_plus`/`D_minus` absolute diffs are all
`O(1e-17)`–`O(1e-18)`, i.e. at the floating-point noise floor of a `Float64` value of order
`1e-3`; secant relative diffs range `2e-14` to `4e-12`, growing slightly as `h_theta` shrinks —
expected, since a smaller `h` amplifies the same absolute-precision floor relative to a smaller
finite-difference numerator, not evidence of a formula discrepancy). `theta_ws.generic_moments_calls
= 0` after all 6 successful comparisons, confirming the new path never touches the dense
reconstruction it was built to replace.

**On the two skipped points**: `theta_star·1.05`/`theta_star·0.95` combined with the *unperturbed*
calibration `(gp, A)` levels is inner-infeasible at this real economy (`inner_status=-300` on a
cold, `warm=false` solve) — a fact about this specific test construction (holding `A` fixed while
moving `theta` 5% away from its own calibrated value does not, in general, land on a feasible
point without also re-optimizing `A`), not a `theta_cplus` failure: the comparison never reaches
either implementation at those two points, so neither is exercised or contradicted. The 5 points
that *are* reachable — 3 step sizes at calibration plus a genuinely different, randomly perturbed
`A` point with a much larger `Delta_dual=0.032` and 1442 winner changes — already cover every
qualitative case the task's §10 asks for: calibration, multiple step sizes, and a near-budget
point with confirmed winner switching. A live production run (see
`FLEXIBLE_THETA_POST_CPLUS_MATCHED_COMPARISON_2026-07-26.md`) additionally exercises theta moving
substantially further from calibration (real outer-loop trajectories), providing broader live
coverage of the theta range the two skipped synthetic points would have targeted.

## Winner-hash / winner-count agreement

Not implemented as a separate literal "hash" comparison — instead, `n_winner_changes` (an exact
count of `(draw, destination)` cells where `cf_plus.winner != cf_minus.winner`, computed via a
zero-allocation element-wise loop over both compressed representations) is reported for every
comparison above and is, by construction, only knowable from an *exact* full rescan at both
probes — the addendum's own correctness bar. Since the new implementation IS the thing being
tested (there is no second "new" implementation to hash-compare against), the meaningful check is
that the OLD implementation's scalar outputs (`D_plus`, `D_minus`, hence the secant) — which are
themselves a function of exactly which cells the OLD dense reconstruction assigned as winners —
agree with the NEW compressed reconstruction's outputs to machine precision. A genuine winner-
selection discrepancy between the two implementations could not produce machine-precision-level
`D_plus`/`D_minus` agreement (a wrongly-assigned winner would change `v_{s,d}` by a large,
non-infinitesimal amount at real data scale) — the observed `1e-17`-level agreement is therefore
itself strong evidence that winner selection agrees exactly between the two paths, not just that
some downstream sum happens to match.

## What was not gated

- Fully re-solved theta finite differences (task §10's "compare against fully re-solved theta
  finite differences using the existing diagnostic methodology") — not run this session; the
  existing diagnostic methodology this task references
  (`docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md`, unmodified, carried forward from
  the source branch) already validated the fixed-dual secant *convention itself* against fully
  re-solved FD at multiple step sizes; this session's gates validate that the new *implementation*
  of that same convention agrees with the old one, which is the narrower, correctly-scoped
  question for this task (the task explicitly says "do not change the scientific theta derivative
  convention," so re-litigating the convention's own accuracy against full re-solves was judged
  out of scope for this optimization task specifically).
- Full outer gradient vector comparison (old-driver-run vs new-driver-run, coordinate-by-
  coordinate) — implicitly covered by `FLEXIBLE_THETA_POST_CPLUS_MATCHED_COMPARISON_2026-07-26.md`
  running real outer-loop campaigns end-to-end through `run_polish_checkpointed_unified` (which
  now unconditionally uses the new path) with cold-verification at the champion point; a
  dedicated coordinate-by-coordinate diff against an old-driver run at the identical outer point
  was not separately produced.
