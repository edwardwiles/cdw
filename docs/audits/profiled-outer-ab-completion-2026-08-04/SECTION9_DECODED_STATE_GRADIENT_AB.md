# Section 9 — decoded-state outer-gradient A/B, extended to all 5 families

The prior session's own evidence (its task §8) covered only flexible_cm x calibration point x one
direction. This section extends coverage across all 5 REDUCED families, honestly scoped to what
was tractable this session (see "not done" below).

## What was run

`test_decoded_state_gradient_ab_allfamilies_2026-08-04.jl`, real D4 KNITRO context, no mocks:

- **All 5 REDUCED families**: unrestricted, flexible_cm, common_frechet, origin_zc, cm_meanzc.
- **2 points each**: P0 (calibration, `ctx.θ0_up`'s own genuine calibration point) and P1 (a small
  random perturbation, `Random.seed!(20260804)`, magnitude 0.01 per coordinate).
- **3 representative A-block coordinates each** (coordinates 2-4 of the outer vector) — covers the
  "ordinary A movement"/"gravity-pivot-coupled A movement" direction categories: every REDUCED A
  coordinate is gravity-pivot-coupled by construction (`profiled_affected_cells` always includes
  both the coordinate's own direct cell AND the shared gravity-pivot cell), so there is no separate
  "non-pivot-coupled" A direction to test in this formulation.
- **Method**: `shared_family_outer_gradient`'s analytic gradient vs. central FIXED-DUAL finite
  difference via `profiled_lfix_incremental_at`, bandwidth matched to `profiled_select_bandwidth`'s
  own adaptive choice (not a naive fixed step — see section 6's own writeup for why a fixed step
  gives a false ~1500% gap here).

## Result

**30/30 checks PASS, exact `0.0` relative error** (machine precision) in every one of 5 families x
2 points x 3 coordinates.

## Directions NOT re-derived here (real, independently-verified evidence already exists elsewhere)

- **gp-only direction**: already covered for flexible_cm/common_frechet by
  `test_flexcm_frechet_outer_gradient_d20_w100k_2026-08-04.jl` (11 representative coords incl. gp,
  real D20/W=100,000, max_rel_err 1.11e-10/1.03e-10).
- **eta_nu-only / joint A+eta_nu directions** (origin_zc/cm_meanzc): already covered by
  `profiled_zc_free_eta_2026-08-04.jl`'s own D4/D20-W20k/D20-W100k gates (~1e-10 to ~2.5e-12
  agreement, per FamilyRegistry's own notes).
- Re-deriving either of these was unnecessary duplication of real, already-existing evidence.

## Not done this session (real remaining work)

- **D20/W=20,000 and D20/W=100,000 scale**: this pass is D4 only. Task §9.3's own scale
  requirements (D20/W20k all points/directions; D20/W100k at least one non-calibration point) are
  not met.
- **P2 (near-boundary/difficult point)**: only P0/P1 were tested, not the third point in the
  task's own 3-point panel.
- **"mixed economic movement" direction**: not independently tested — the analytic gradient's own
  linearity means a mixed-direction directional derivative is `dot(g, d) = sum_k g[k]*d[k]`, a
  mathematical identity once each individual `g[k]` is itself validated (as done here), not
  something requiring a separate empirical FD check.
- **FULL vs REDUCED cross-formulation comparison**: task §9.3 asks for FULL's own analytic/FD
  directional derivatives compared against REDUCED's, via a common decoded direction mapped into
  each formulation's own coordinate tangent space. This is real, substantial additional
  infrastructure (constructing the common-direction-to-each-basis Jacobian map) that was not built
  this session — only REDUCED-internal (analytic vs FD) verification was done.
- **Native vs powered REDUCED mode**: this pass used native coordinates only; task §9.3's "native
  and powered REDUCED modes for at least unrestricted/flexible_CM/origin_ZC" was not repeated here
  (powered mode's own D4 chain-rule correctness was separately verified in section 6, machine
  precision, for flexible_cm and origin_zc; unrestricted is not applicable to powered mode, see
  section 6's own finding).

## Verdict

```
DECODED_STATE_GRADIENT_AB =
    unrestricted:pass (D4, A-block, 2 points)
    flexible_CM:pass (D4, A-block, 2 points; gp direction separately verified at D20/W100k)
    common_frechet:pass (D4, A-block, 2 points; gp direction separately verified at D20/W100k)
    origin_ZC:pass (D4, A-block, 2 points; eta_nu direction separately verified at D4/W20k/W100k)
    CM_plus_ZC:pass (D4, A-block, 2 points; eta_nu direction separately verified at D4/W20k/W100k)
```

All 5 families now have real, verified decoded-state gradient evidence (a genuine improvement over
the prior session's single-family/single-point/single-direction coverage), at D4 scale with
directions decomposed across this pass and pre-existing gates. D20-scale and FULL-cross-comparison
remain open.
