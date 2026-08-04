# Pilot algorithm tournament verdict — 2026-08-03

Per task Section 8: 3 representative cells, each run under EXPLORE (Direct+SR1) and POLISH
(SQP for CM-family/origin-ZC, Direct+BFGS for unrestricted — see
`ALGORITHM_INVENTORY_2026-08-03.md` for why SQP isn't wired for unrestricted), as two **separate**
standalone runs from the identical seed (not chained), 1500s budget each, real KNITRO, real
production manifest. Raw data: `pilot_output/pilot_summary.csv` in repo_scratch; full logs
alongside it.

## Results

| Cell | Family | Old campaign (7200s, algorithm=auto) | EXPLORE Direct+SR1 (1500s) | POLISH (1500s) | Winner |
|---|---|---|---|---|---|
| A | flexible_cm upper δ=2 | GT=0.0601 (6 verified, `-401`) | **GT=0.07298**, Δ*=1.265, 5 evals, `-401` | GT=0.07243, Δ*=1.000, 3 evals, `-401` | **EXPLORE** |
| B | origin_zc upper δ=2 | GT=0.0669 (10 verified, `-101` step-tol) | GT=0.07616, Δ*=1.000, 7 evals, `-401` | **GT=0.07668**, Δ*=1.147, 6 evals, `-401` | **POLISH** |
| C | unrestricted lower δ=1 (seed@δ=0.5) | δ=1/δ=2 both stuck at seed's own point (documented non-monotonicity) | GT=0.00332, Δ*=0.385 (=seed, zero improvement, 476 evals) | GT=0.00332, Δ*=0.385 (=seed, zero improvement, 555 evals) | **tie — both flat** |

## Interpretation

**Cell A (flexible_cm, ~380-dim outer space):** EXPLORE beats both POLISH and the old campaign by
a wide margin, in 1/5th the wall-clock. Direct+SR1 alone, in 1500s, already exceeds the δ=1
inherited incumbent (0.0724) and gets partway toward δ=2's loose budget (Δ*=1.265 of an allowed
2.0) — confirmed genuinely under-budget, not stalled: only 5 evals were needed, meaning there is
real headroom for a longer explore budget to push further. POLISH (SQP, `maxit=15`) underperforms
here specifically because each SQP/active-set iteration's LP subproblem is too expensive at this
dimension to make meaningful progress within 1500s (only 3 evals fit).

**Cell B (origin_zc, ~400-dim outer space including nu):** Opposite ranking. EXPLORE made
essentially zero net progress beyond the seed in 1500s (7 evals, extremely expensive per-eval —
origin_zc's per-iteration cost is structurally higher than flexcm's, consistent with the larger
outer dimension and its own origin-specific moment machinery). POLISH (SQP) genuinely pushed past
the δ=1 boundary toward the δ=2 target (Δ*=1.147) and posted the best GT of any arm for this cell.
The tight `maxit=15` cap that hurt cell A actually helps here — a small number of well-targeted SQP
steps found real improvement faster than Direct+SR1's broader but slower search.

**Cell C (unrestricted, lower direction):** Both algorithms converge to bit-identical results
(GT=0.0033238282701397726, Δ*=0.38519170195952107 — matching the seed's own values to full
precision) despite ~500+ evals each and gp briefly reaching better values (up to 0.99973) during
the search. **This reproduces, under two different algorithms, the exact non-monotonicity already
documented in `CONTINUATION_PRIORITY_SUMMARY_2026-08-03.md`** ("δ=1.0 and δ=2.0 landed on the
literal same point... the clearest, most reproducible non-monotonicity signature in the whole
dataset"). Confirms that finding is a genuine feature of this (family, direction) cell's outer
landscape — some verified-feasibility barrier the outer search cannot cross from this seed within
budget, not an algorithm artifact or a one-off convergence fluke. The never-regress rule
(`apply_never_regress`) handles this correctly by construction: export the δ=0.5 seed unchanged as
the δ=1 result, `result_source=inherited_incumbent`. No further pilot time spent chasing this
further; a longer single-algorithm budget could be tried later on this specific cell as an
adaptive waypoint (task Section 5's 0.75 intermediate delta) if it becomes a priority, but is not
worth blocking the broader campaign for.

## Verdict: per-family algorithm choice for the staged campaign (Section 9)

```
PILOT_ALGORITHMS =
    unrestricted:      explore=Direct+SR1, polish=Direct+BFGS (both tried, flat on the one lower
                        cell tested; use EXPLORE first for upper cells per cell-A-like reasoning
                        pending direct confirmation, chain into POLISH as run_target_cell! already does)
    flexible_cm:        explore=Direct+SR1 (primary), polish=SQP maxit15 (secondary/tightening only)
    common_frechet:     same driver as flexible_cm (run_cm_upper_checkpointed) -- same choice
    cm_meanzc:          same driver as flexible_cm -- same choice
    origin_zc:          explore=Direct+SR1 (still needed to move off a stale/cold seed if one is
                        ever used), polish=SQP maxit15 as PRIMARY driver of real improvement
```

Both EXPLORE and POLISH remain wired into every `run_target_cell!` call (per its own chained
explore→polish design, unlike this pilot's deliberately-separated arms) — the family-specific
finding here is about which arm is expected to do the *heavy lifting* for that family, not that
either arm should be skipped. The never-regress rule makes running both arms safe even when one
underperforms (worst case for a cell: the better of the two arms' own results, never a regression
below the inherited incumbent).

```
PILOT_ALGORITHMS =
    exploration: EXPLORE_DIRECT_SR1 (outer_direct_hessopt=:sr1)
    polish:      POLISH_SQP for {flexible_cm, common_frechet, cm_meanzc, origin_zc},
                 POLISH_DIRECT_BFGS for {unrestricted}
```
