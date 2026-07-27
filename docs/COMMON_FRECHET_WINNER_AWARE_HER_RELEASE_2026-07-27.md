# Common-Fréchet Winner-Bin H_ER Release — 2026-07-27 (Section 3)

## What changed

`hessian_cm_frechet_structured!` (serial, `cm_frechet_hessian.jl`) and
`hessian_cm_frechet_structured_v2!` (threaded, `cm_frechet_hessian_threaded.jl`, production default
via `archC_frechet_hess_cb_builder`/`_v2`) now support `cctx.cm_cross_hessian_backend = :winner_bin`
for **both** new blocks this Hessian callback owns beyond flexible-CM's own H_EE/H_EC/H_CC:

- **H_EC (CM-grid cross block, Part A)**: mathematically IDENTICAL to flexible-CM's own H_EC (same
  `CS_[o,j,l]-CS_[refIndex1,j,l]` formula). Wired the SAME way `hessian_cm_structured!` was in
  Section 2: reuses `_cm_cross_hessian_wants_winner_bin`/`_ensure_cm_cross_scratch!`
  (`cm_hessian_architectures.jl`, not redefined) and the already-validated
  `winner_pair_cross_hessian_fill!`/`winner_pair_cross_hessian_cm_block!` primitive.
- **H_E,level (level-anchor cross block, Part B — genuinely new math)**: the level restriction sums
  over ALL D origins (weight `1/sqrt(D)`) rather than differencing against `refIndex1`, so it needs a
  **sum-over-origins** read of the same cumulative tables, not the difference
  `winner_pair_cross_hessian_cm_block!` already provides. Two new primitives added to
  `winner_pair_cross_hessian.jl`:
  - `winner_pair_cross_hessian_colsum!`: `colsum[j] = sum_{o=1}^D CS_[o,j,l]`, `O(D)` per row.
  - `winner_pair_cross_hessian_esum!`: `Esum[j] = sum_s w[s]*E[s,j]` (UN-binned), needed for the
    level feature's nonzero-target correction term. Required extending `WinnerBinCrossScratch` with
    a new `EsumEcon::Vector{Float64}` field (length `ncolI`), filled inside
    `winner_pair_cross_hessian_fill!`'s existing per-slot loop (one extra `O(W*Ddest)` accumulation,
    negligible next to that loop's own `O(W*Ddest*D)`).

`H_CC`, `H_CM,level`, and `H_level,level` are **unchanged** — confirmed by re-reading
`cm_frechet_hessian.jl`'s own derivation comment that all three read only `CT`/`T1`/`Wtot`, built
from `Bidx`/`w` alone, never from `E`. No winner-bin variant was needed or added for them.

`CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT` (new `Ref`, `core_exact_hessian.jl`) flipped
`:dense_reference` -> `:winner_bin` after both gates below passed. **Deliberately a separate `Ref`**
from flexible-CM's own `CM_CROSS_HESSIAN_BACKEND_DEFAULT` (already `:winner_bin` from the prior
session) so that session's flip did not silently also change common-Fréchet's behavior before this
family's own gates ran — mirrors CM+meanZC's existing precedent of a locally-hardcoded, independent
default.

## Bug found and fixed during this phase (Part B, `winner_pair_cross_hessian_esum!`)

The "cf"/common-factor economic column (`wctx.has_cf`, index `wctx.ncolI`) is not a `(slot,origin)`
pair, so — exactly like `QTab[jcf,:,:]` in the binned case (already handled by
`winner_pair_cross_hessian_cm_block!`'s own dedicated `QCfCScum` override) — `EsumEcon[jcf]` is left
at zero by the slot loop. The first version of `winner_pair_cross_hessian_esum!` used it
unconditionally, silently dropping the entire cf-column contribution from `Esum[jcf+1]`.

Caught live by the D=4 wiring gate's first run: 66/114 checks FAILED, isolated cleanly to the
`H_E,level` slice (H_EC/H_CC matched dense to ~1e-15 throughout, correctly ruling out Part A). One
config (`anchored L=20 calib`) happened to PASS to 1e-15 by coincidence — at that specific point
`core_cf_ref[]` was a tied-winner/dense-fallback `Symbol`, not a `CompressedFactual`, so
`_cm_cross_hessian_wants_winner_bin` returned `false` and BOTH backends silently ran the identical
dense path there — not a real winner-bin exercise, flagged here so it isn't mistaken for
independent confirmation.

Fixed by computing `ecf = sum_w S[w]*nu[w]*cf_raw_scaled[w]` in the same `O(W)` pass already
computing `t0` (no second traversal) and overwriting `Esum[jcf+1]` with it, mirroring
`winner_pair_cross_hessian_cm_block!`'s own cf-column override pattern exactly.

## A second bug found and fixed — this one in the gate harness, not production code

The first real D=20 run FAILED at the `hard_point_x1.01` config (both contrasts): complete-Hessian
`max|Δ|~5990` at Hessian scale `~1590`. A targeted diagnostic
(`diag_frechet_hardpoint_2026-07-27.jl`, kept in the repo) isolated this by block:

```
max|Delta H_EE|        = 1590.36   <- the SHARED winner-pair core block, untouched by this phase
max|Delta H_EC|        = 9.59
max|Delta H_CC|        = 1.08
max|Delta H_E,level|   = 1.69
max|Delta H_CM,level|  = 0.0092
max|Delta H_level,level| = 0.53
```

`H_EE` alone accounted for the entire discrepancy — and `H_EE` is never touched by this phase's own
code at all, proving the "failure" was not a Part A/B bug. Root cause: unlike `calib`/
`near_delta1_perturbed` (which only perturb the *dual* point — `obj.H` stays fixed at the
calibration `theta_full` the whole time, this codebase's own established "same theta, different
Hessian-weight vector" gate convention), `hard_point_x1.01` requires a genuinely new `theta_full`.
The first attempt solved that new point through `pcx_dense` only, leaving `pcx_wbin`'s `obj.H` stale
at the OLD calibration theta — so the two sides were comparing Hessians built from two entirely
different moment matrices, not exercising the same backend question at all. `cf_w` (the stale side's
own `core_cf_ref[]`) confirmed this directly: `isa CompressedFactual = false` at the point of
comparison.

Fixed in `test_frechet_winner_bin_her_wiring_d20.jl`: the hard point is now solved through **both**
backends (mirroring how the calibration `base_dense`/`base_wbin` pair is already built), with
explicit status/dual-point-match checks for that solve, before the Hessian comparison runs.

## Gates (both ALL PASS after the two fixes above)

**D=4** (`test_frechet_winner_bin_her_wiring_d4.jl`, modeled on
`test_flexible_cm_winner_bin_her_wiring_d4.jl`): 2 contrasts (anchored, orthonormal) x 3 `L` values
(10, 20, 50) x 3 points (calibration, 2 independent perturbed points) x 2 architectures (serial
`hessian_cm_frechet_structured!`, threaded `hessian_cm_frechet_structured_v2!`) = 36 complete
packed-Hessian comparisons **plus** 36 **isolated H_E,level slice** comparisons (read directly off
the persistent `cctx.Hfull` before the next call overwrites it, so a level-block-only bug localizes
cleanly from the CM-grid reuse), plus per-`(contrasts,L)` inner-solve status/dual-point match and
persistent-workspace identity checks. **ALL PASS, 114/114 checks**, `max|ΔH|` in
`[3.55e-15, 1.22e-14]` against a Hessian scale of ~1.0-1.02.

**Real D=20/W=80,000/L=50** (`test_frechet_winner_bin_her_wiring_d20.jl`,
`destination_sample=:exclude_row`): 2 contrasts x 3 points (calibration, a near-delta=1 perturbed
point, and the hard point `x_free0.*1.01` — the exact perturbation that exposed the
`skip_cm_fill_ref` crash in `docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md`, reused per this
family's own established convention) x 2 architectures = 12 complete packed-Hessian comparisons plus
12 isolated H_E,level slice comparisons, plus inner-solve/workspace checks (now including the hard
point's own status/dual-point match, added by the harness-bug fix above). **ALL PASS, 44/44
checks**, `max|ΔH|` in `[7.17e-12, 1.10e-10]` against a Hessian scale of `[1590, 5198]`. Complete
inner-solve status matches at every point (`nStatus=0` throughout) and dual point agrees to
`<1.5e-12`, including at the hard point.

## Runtime counters (task Section 7, partial)

D=20 gate run: `dense_cross_hessian_calls=54`, `winner_cross_hessian_calls=58`
(`=operator_cross_hessian_calls`), `dense_Frechet_G_materializations=54` (the counter this phase
added specifically for the `H_E,level` dense fallback path). These totals include the internal
Hessian-callback invocations inside every `archC_frechet_base_state`'s own KNITRO solve, not just
the 12+12 explicit comparison calls, so they exceed the 24-call comparison-loop count.

## Timing (informational)

Real D=20, `threaded_v2` warm: ~0.35-0.57s (`winner_bin`) vs ~0.7-3.1s (`dense_reference`) per
Hessian call, roughly the same 2-7x range flexible-CM's own Section 2 release measured. Warm
`hessian_cm_frechet_structured_v2!` (`:winner_bin`) allocates a stable ~9.4-9.9 MB/call at real D=20,
with zero persistent-workspace resizes across repeated calls (`cross_scratch` `objectid` stable).

## Not done in this section

- CM+ZC, ZC-only H_ER (separate tracks per the master task).
- Rectangular (`D != Ddest`) and non-last-omitted-destination D=4 configurations were not separately
  exercised, same disclosed gap as flexible-CM's own Section 2 release (no rectangular-layout D=4
  context builder exists in this repository).
- The D=20 gate's "hard point" only re-solves through both backends for the Hessian comparison
  itself — it does not additionally sweep multiple hard points or multiple `L` values at D=20 (time
  budget); the D=4 gate's own 3-`L`-value sweep is the breadth check for `L`-dependence.
