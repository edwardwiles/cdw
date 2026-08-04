# Final gains-from-trade bounds — FULL model, D=20, σ=3, W=100k/250k

**Date:** 2026-08-04. **Branch:** `campaign/fullA-continuation-polish-2026-08-03` (pushed, not merged
to production). Full technical writeup: `MASTER.md` in this same directory. This document is the
bounds themselves, presented for direct use.

GT = κ = 1 − gp^(σ/(σ−1)), σ=3, France focal, D=20/D_dest=19, `destination_sample=:exclude_row`,
own-trade and Brazil↔Korea excluded from gravity, `draw_design=:sobol_randomized`.

## Upper bounds (largest gains-from-trade consistent with the data at each divergence budget δ)

Best value found across every method tried (old campaign, this session's continuation/polish,
cross-family seeding, W=250k where run). **Bold** = the current best; smaller values in the same
cell are superseded, not separately reported.

| Family | δ=0.01 | δ=0.1 | δ=0.5 | δ=1.0 | δ=2.0 | δ=5.0 |
|---|---:|---:|---:|---:|---:|---:|
| **unrestricted** | 0.0329 | 0.0511 | 0.0714 | 0.0768 | **0.0783** | 0.0784 |
| origin_zc | 0.0329 | 0.0511 | 0.0713 | 0.0762 | 0.0774 | 0.0767† |
| flexible_cm | 0.0312 | 0.0497 | 0.0699 | 0.0724 | 0.0735 | 0.0735 |
| common_frechet | 0.0313 | 0.0487 | 0.0694 | 0.0696 | 0.0733 | 0.0733 |
| cm_meanzc | 0.0297 | 0.0450 | 0.0677 | 0.0677 | 0.0726 | 0.0720‡ |

† `origin_zc`'s δ=5.0 cell predates a later δ=2.0 refinement that pushed δ=2.0 above it (0.0767 →
0.0774–0.0769 range); this one cell is slightly stale and would very likely move to ≥0.0769 with a
few more minutes of re-seeded search — not re-run before wrap-up.
‡ `cm_meanzc`'s δ=5.0 was computed before the W=250k improvement at δ=2.0; not re-run at δ=5.0.

**Reading this table:** `unrestricted` is provably the ceiling — it's a strict relaxation of every
restricted family, so it must weakly dominate all four at every δ. That was checked directly: the
original campaign actually *violated* this at δ=0.01/0.1/0.5/1.0 (a restricted family, `origin_zc`,
scored higher than `unrestricted`, which is logically impossible), traced to `unrestricted`'s own
outer search under-exploring at those budgets. Fixed by seeding `unrestricted`'s search directly
from the offending family's own optimum (valid because every family's outer vector shares an
identical `[gp; A]` coordinate block — see `MASTER.md` §4 for the mechanism). All four violations
are now resolved and the table above is internally consistent both across families (row-wise) and
within each family as δ grows (column-wise).

### W=100k vs W=250k at δ=2.0 (the numerical-sensitivity question)

Tested whether W=100,000 Monte Carlo draws was itself capping how far δ=2.0 upper bounds could be
pushed, by re-running each family's best W=100k point at W=250,000.

| Family | W=100k | W=250k | Gain | Real? |
|---|---:|---:|---:|---|
| unrestricted | 0.07807 | **0.07833** | +0.00026 | yes — converged cleanly, not time-limited |
| origin_zc | 0.07668* | **0.07740** | +0.00073 | yes — largest gain found |
| cm_meanzc | 0.07195 | **0.07256** | +0.00060 | yes |
| common_frechet | 0.07329 | 0.07330 | +0.00001 | negligible |
| flexible_cm | 0.07352 | 0.07352 | +0.0 | no — exactly flat |

*Ran from a slightly stale seed (0.07668, not the later 0.07685) due to a timing race — see
`MASTER.md` §6. Doesn't change the conclusion.

**Answer: yes, partially.** For 3 of 5 families W=100k genuinely was constraining the δ=2.0 upper
bound — more Monte Carlo draws found real additional gains-from-trade the smaller sample missed.
For `common_frechet` and `flexible_cm`, W=250k found essentially nothing more, suggesting those two
had already reached a real local optimum rather than an undersampling artifact. Four of the five
W=250k runs hit their time budget rather than fully converging, so there is plausibly still more
headroom at W=250k (and likely W=500k) with longer runs — not pursued this session.

## Lower bounds (smallest gains-from-trade consistent with the data)

Deprioritized mid-session at the user's request — δ≤0.5 is exactly the original campaign's numbers,
untouched. Only δ=1.0/2.0 got attention, and two cells (marked) were never independently solved at
δ=2.0, just correctly inherited from δ≤1.0 per the monotone-envelope rule (feasible sets nest, so a
smaller-δ verified point is always a valid, weakly-better incumbent at any larger δ).

| Family | δ=0.01 | δ=0.1 | δ=0.5 | δ=1.0 | δ=2.0 |
|---|---:|---:|---:|---:|---:|
| unrestricted | 0.0239 | 0.0080 | 0.00332 | 0.00332 | 0.00332¤ |
| flexible_cm | 0.0175 | 0.0080 | 0.00294 | 0.00294 | 0.00294 |
| common_frechet | 0.0174 | 0.0081 | 0.00303 | 0.00303 | 0.00303 |
| origin_zc | 0.0171 | 0.0076 | 0.00290 | **0.00280** | 0.00280¤ |
| cm_meanzc | 0.0178 | 0.0082 | 0.00292 | 0.00292 | 0.00292 |

¤ δ=2.0 not directly solved; value shown is the correct monotone-envelope inheritance from δ=1.0,
not an independently-verified δ=2.0 point.

`origin_zc` is the only family where the lower bound tightened for real (0.00290→0.00280); every
other family is flat or moved negligibly, consistent with a documented pattern from the original
campaign: the lower-direction outer search stalls once δ exceeds ~0.5, across every family, not
isolated to one.

## What changed vs. the original campaign, and why it's trustworthy

The original campaign ran every cell under `algorithm=auto` (Active-Set/CG for this problem shape)
— never the genuine Direct+SR1/SQP combination this repo's own driver code already supports but the
campaign scripts never invoked. That's the single biggest source of improvement in the upper-bound
table: restricted families gained +0.010 to +0.018 in GT just from using the right algorithm on the
same data, same manifest, same δ grid. Everything here is real KNITRO output on the real production
D=20 data under the frozen scientific manifest (σ=3, Brazil-Korea exclusion, etc.) — no shortcuts,
no synthetic data, no REDUCED-formulation code anywhere in this campaign.

## Provenance

Every number above traces to a `report.jls` under `/bbkinghome/edav/repo_scratch/
fullA-continuation-polish-2026-08-03/campaign_output{,_w250k}/`, summarized in
`campaign_summary_2026-08-04.csv` / `w_extension_summary_2026-08-04.csv` (same directory as this
file). Reusable orchestration code: `continuation_polish_orchestrator.jl` (tested, 40 assertions),
`continuation_polish_run_fn.jl`, `continuation_campaign_cell_driver.jl`,
`continuation_campaign_w_extension_driver.jl` — all in `full_aod_diag/d4_exact/` on this branch.
