# Warm-start reliability: open issue, kept out of scope for this integration

**Status: note only. No code change in `integration/fullA-fast-range-screen`.**

This integration branch adds exact fast infeasibility screens
(`full_aod_diag/d4_exact/fast_range_screen.jl`). It deliberately does **not**
touch `DUAL_WARM_MODE`, `screened_eval`'s `warm=` policy, or any other part of
the CC inner-solve warm-start machinery in `c10_d20_production_driver.jl` /
`compressed_live.jl`. This note records why that policy is a real,
unresolved reliability question — surfaced during the prior experimental
review (`diag/fullA-d20-range-screen-review`, Section 8) — without
conflating it with this branch's screening work, so a benchmark of the
screens is not confounded by a simultaneous warm-start policy change.

## The observed pattern (real D=20/W=80,000, 3 independent point pairs)

Attributed to the prior review branch's `warm_start_reliability_sample_d20.jl`
(commit `a3b1c11`), extending an earlier single-pair finding from
`diag/fullA-d20-fast-infeasibility`. Not re-run on this integration branch;
reported here as the open question this branch's own benchmark methodology
must be aware of (see the production integration doc's benchmark section for
how solves are timed to avoid this confound).

| scenario | wall time | outcome |
|---|---|---|
| warm-start from a **compatible** prior point (e.g. calibration -> A) | 0.70s | large win vs. cold |
| cold solve at A | 9.37s | baseline |
| warm-start from a **different converged dual state** (A warm-from-B) | 41.76s | **slower than cold** |
| warm-start B from A | 38.31s | slower than cold (29.88s cold at B) |

Pattern: warm-starting from a *compatible* prior point is a large win;
warm-starting from a *genuinely different* converged dual state is
consistently and substantially **slower than a cold start** — not a wash,
a real regression, reproduced across 3 independent pairs.

## Why this matters for screening work specifically

`c10_d20_production_driver.jl`'s `cb_F!` callback (the same callback this
integration branch's screens sit in front of) already tries `warm=true`
first for every outer trial point, falling back to a cold retry only on
rejection (and, per this integration branch's ported overnight baseline --
see the "Port overnight Continuation-11..." commit -- `skip_cold_retry=true`
is now the production default, since the cold retry rescued 0/30 genuine
warm failures in that investigation). If an outer search revisits a region
whose warm-start anchor is *incompatible* with the current trial point's
dual state, every such point pays the ~4-5x warm-start penalty shown above
regardless of whether it is screened first — the screens in this branch
reduce the number of KNITRO calls made at all (for screen-certified-infeasible
points), but do not change the per-call warm-start cost distribution for
points that pass every screen and reach the real solve.

## Proposed separate experiment (not started, no code here)

1. Characterize what makes two dual states "compatible" vs "incompatible"
   for warm-starting (distance in reduced A_od coordinates? gravity-tangent
   direction? delta value?) using a larger sample than the 3 pairs above.
2. If a cheap-to-compute compatibility predictor exists, consider gating
   `warm=true` on it (skip straight to cold when predicted incompatible) --
   this would need its own validation that it never produces a WORSE outcome
   than always-warm or always-cold, and its own benchmark isolated from any
   screening change, per this note's own framing.
3. Re-run this integration branch's warmed benchmark (Section 7 of the
   production integration doc) AFTER any warm-start policy change lands, to
   re-confirm the screens' overhead numbers still hold under the new policy
   -- they should (the screens run before the warm/cold decision, in every
   design considered here), but should be re-checked, not assumed.

No code in this branch depends on or changes the outcome of this future
experiment.
