# Flexible-theta screens audit — production port, 2026-07-25 (task §7)

## Screens preserved, active, unchanged under flexible theta

- **Pairwise/support certificate** (`infeasibility_screen.jl::pairwise_certificate`,
  `precompute_pairwise_M`): draw-only (depends on `ctx.U`/`ctx.D`/`ctx.D_dest`), never touches
  theta/mu. Valid unchanged in flexible mode.
- **Canonical hard-winner/range screen** (`infeasibility_screen.jl::screen_hard_winners`,
  `fast_range_screen.jl::screen_hard_winners_ranged`): re-evaluated per-point from the CURRENT
  `θ_full` (via `compute_a_od`'s `μ = θ_full[1]`) exactly as fixed-mode production already does —
  it was never precomputed assuming a fixed mu, so it needs no change at all. Confirmed by reading
  (agent-A reconnaissance + direct read): `infeasibility_screen.jl`'s `compute_a_od` takes `μ`
  freshly from the full theta vector on every call.
- **Threshold-10 inner lower-bound certificate** (`cc_algo/threshold_early_abort.jl`,
  `ThresholdAbortState`/`resolve_threshold_for_delta`): operates on the INNER KNITRO dual solve's
  own running lower bound, independent of how the OUTER theta/A coordinates are parametrized.
  Valid unchanged.
- **Winning-range / general moment-range safety net**
  (`fast_range_screen.jl::range_screen_standalone`, `evaluate_fullA_screened_ranged`'s safety-net
  branch): re-derived per-point from the current `CompressedFactual`, not precomputed against a
  fixed mu. Valid unchanged.

All four of the above are exercised, unmodified, through the SAME `screened_eval` (
`c10_d20_production_driver.jl`) both fixed- and flexible-mode drivers call — this port adds no new
screen code and does not bypass any of them.

## Screen disabled in flexible mode: the pre-winner envelope screen

`fast_range_screen.jl::precompute_envelope` hard-errors (`EnvelopeUnsupportedContext`) whenever
`ctx.θ_lo[1] != ctx.θ_hi[1]` (mu not fixed to a point) or `ctx.θ_lo[2] != ctx.θ_hi[2]` (sigma not
fixed). `make_flexible_theta` sets `θ_lo[1] = log(theta_lo)`, `θ_hi[1] = log(theta_hi)` — distinct
whenever `theta_lo != theta_hi` — so ANY flexible-theta ctx trips this guard; `build_ranged_
screen_context` catches it once at context-build time and returns `rsc.envelope === nothing`. This
port's D=20 driver logs this explicitly at startup
(`envelope_screen_supported=false (reason: ... -- expected in flexible mode, ...)`) rather than
silently omitting the screen.

**Why the guard exists (the factorization assumption that breaks)**: the envelope screen's
correctness rests on precomputing, ONCE per ctx, a data-only per-column upper bound
`M[o,d] = max_s(K2[o,d]/UsigmaPow[s,o])` where `K2[o,d]` bakes in the FIXED exponent
`mu*(sigma-1)` on `Aod_theta[o,d]`, and `UsigmaPow = γ.Uσ.^(-mu)` is a FIXED matrix computed once
at that one mu. The screen's nonnegativity/monotonicity argument (`K2>=0`, `b>=0`,
`exponent=mu*(sigma-1)>0`) is verified pointwise at that one fixed mu, not proven to hold
uniformly across an entire flexible-theta BOX. Under flexible theta, reusing a single
precomputed `M`/`UsigmaPow` across every candidate mu in the box would either (a) require
recomputing them at every outer point (defeating the screen's whole "precompute once, O(D*Ddest)
per point thereafter" design), or (b) risk a FALSE-POSITIVE infeasibility certificate if the
monotonicity argument does not hold uniformly across the box (not verified either way in this
port — see "bounded audit" below).

## Bounded mathematical audit (task §7's explicit requirement): can it be safely re-derived?

Assessed, not attempted, within this port's time budget. Two paths exist:

1. **Per-point re-derivation** (drop the "precompute once" design, recompute `K2(mu)`/`UsigmaPow(mu)`
   fresh at every outer point from the CURRENT mu): mechanically straightforward (the formulas in
   §1 above are already mu-parametrized in `precompute_envelope`'s own body — `μ = ctx.θ_lo[1]` would
   become an explicit function argument), but destroys the screen's performance rationale (it exists
   specifically to be cheap versus the alternative of a full winner/moment reconstruction) and its
   nonnegativity argument (`K2>=0`) would need re-verification at every candidate mu rather than
   once — effectively turning a screen into a partial re-solve.
2. **Box-uniform bound** (derive a single certificate valid for ALL mu in `[theta_lo, theta_hi]`
   simultaneously, e.g. by bounding `K2(mu)`/`UsigmaPow(mu)` over the box rather than evaluating at
   a point): mathematically the "correct" fix, but requires establishing monotonicity of
   `K2(mu)`/`UsigmaPow(mu)` in mu over the box (not obviously monotone — `K2` involves
   `wPow^(1-σ)*τ^(1-σ)*C1^(1-σ)` and `C1` itself depends on mu through a `^(1/μ)` exponent inside
   `B`), which is genuinely new derivation work, not a small extension of the fixed-mu proof.

**Decision**: option 1 forfeits the screen's purpose; option 2 is open-ended derivation work beyond
this port's bounded scope (task §7 explicitly allows leaving it disabled with a precise reason
rather than requiring a re-derivation). **Kept explicitly disabled in flexible mode**, with the
precise startup reason logged (see above) — not silently dropped, not silently kept active with an
unverified assumption.

## Why this is not a blocker

Per task §7's own framing ("a disabled optional envelope screen is not itself a blocker if the
remaining exact stack is active"): the envelope screen is a PERFORMANCE optimization (an early,
cheap rejection of provably-infeasible candidates before the more expensive winner-scan/safety-net
screens run) — its historical organic hit rate in production was already measured as **zero** in
real D=20 search sessions (per `diag/fullA-d20-range-screen-review`'s own finding, referenced in
agent-A's reconnaissance and this repo's docs). Disabling it does not weaken CORRECTNESS (no exact
certificate is lost — the pairwise/hard-winner/winning-range/safety-net screens remain fully
active and exact), only removes one layer of a defense-in-depth performance stack whose measured
marginal contribution was already near zero even in fixed-theta production.

## Regression check

`test_gravity_elimination.jl` (existing, unmodified production regression test) re-run against this
port's additive `gravity_elimination.jl` changes (new optional `μ` keyword, default preserves
byte-identical behavior) — see `docs/FLEXIBLE_THETA_ASPACE_PRODUCTION_PORT_2026-07-25.md`'s fixed-
mode regression section for the pass/fail record.
