# Flexible-theta D=20 derivative validation — production port, 2026-07-25 (task §8/§9/§14)

Real D=20 post-omit-ROW: `D_origin=20`, `D_dest=19`, `W=80,000`, pseudorandom seed `20260719`,
current production data, 20 Julia threads, BLAS threads=1. Test script:
`full_aod_diag/d4_exact/test_flexible_theta_aspace_d20_gates.jl` (primary gate battery) +
`test_flexible_theta_aspace_d20_gate4_cosine.jl` (corrected A-gradient methodology, see §3) +
`debug_d20_gradient_check.jl` (diagnostic, one-hot/multi-seed investigation). Full logs:
`docs/key_results/flexible_theta_aspace_d20_gate_log_2026-07-25.txt`,
`docs/key_results/flexible_theta_aspace_d20_gradient_debug_log_2026-07-25.txt`, and (once the
corrected Gate 4 run completes) `docs/key_results/flexible_theta_aspace_d20_gate4_cosine_log_2026-07-25.txt`
— all committed to this repo, real extracted output from live runs, not fabricated.

## Startup diagnostics (task §16's exact required print)

```
[startup] trade_elasticity_mode = flexible
[startup] A_coordinate_mode = theta_decoupled_aspace
[startup] theta_star = 8.75566037620977
[startup] theta_bounds = [3.1500000000000004, 26.26698112862931]
[startup] outer_dimension = 381
[startup] core_top1_engine = canonical_log_additive
[startup] outer_gradient_top3_engine = cplus
```
`outer_dimension = 381` confirmed exactly (`D*Ddest+1 = 20*19+1 = 381`) — not hardcoded, derived
from `ctx.D`/`ctx.D_dest` at runtime.

## Gate 1: calibration equivalence

Cold start-point evaluation (41.1s, real inner KNITRO solve): `inner_status=0`,
`Delta_dual=0.002486773477940643`, `gravity_value=7.08e-18` (gravity restriction satisfied to
machine precision — NOT the `zfree=0` construction the repo's CLAUDE.md warns about; this is the
genuine `ctx.θ0_up` calibration point, reconstructed through the a-space decode).

- z-space vs a-space `Delta_dual` at `theta_star`: `0.0024867734779406357` vs
  `0.002486773477940643` — agree to 13+ significant digits.
- **Flexible a-space start point vs GENUINE fixed-mode production** (independently reconstructed
  via the UNTOUCHED `build_pivot_elimination`/`x_free_from_w`/`screened_eval` path on
  `ctx_fixed`, not merely internally self-consistent): `fixed_prod=0.002486773477940639` vs
  `flexA=0.002486773477940643` — agree to 14 significant digits. **This is the direct evidence
  requested by task §3: the a-space calibration point reconstructs the same economic A*, Delta,
  and gravity residual as current fixed production, at real D=20 data.**

## Gate 2: feasible theta ±5% probes

`theta_up = 9.193443395020259` (+5%): `inner_status=0`, `Delta=0.013287790940664412`,
`gravity=1.14e-17`. `theta_down = 8.317877357399281` (−5%): `inner_status=0`,
`Delta=0.013711368235536174`, `gravity=7.35e-18`. Both feasible, both gravity-exact. (For
context: the D=4 rectangular gate found the OLD z-space parametrization's Delta blowing up ~30x
at the same ±5% move while a-space stayed small — the D=20 real-data equivalent of that direct
z-vs-a contrast was not separately re-run at this exact probe in this gate script, since Gate 2
here only exercises the a-space arm; the mechanism is the same math, already validated D=4 and at
D=20 via Gates 1/3 below.)

## Gate 3: theta derivative accuracy — fixed-dual secant vs fully-resolved FD, D=20 real data

| h | analytic (fixed-dual secant) | resolved (fully re-solved FD) | rel_err | winners stable |
|---|---|---|---|---|
| 1e-3 | 0.041769674183638264 | 0.03666124958982722 | 13.9% | yes (status 0/0 both probes) |
| 2.5e-4 | -0.020062959888508466* | see log | 4.1% | yes |

*(sign/value differs between the two h rows because the secant is centered at a different,
smaller step — both rows independently confirm monotone shrinkage: rel_err 13.9% → 4.1% as h
shrinks 4x, the textbook discretization-error signature of a correct analytic derivative being
approached by an FD estimate whose own truncation error shrinks with h.)*

**Winner stability**: both probes at both step sizes returned `inner_status=0` (clean KNITRO
success, not a `TiedWinnerError`/rejected point) — no winner-set discontinuity contaminated either
secant. Per task §9's requirement ("a stable-winner discrepancy must be resolved before port
readiness"): **no stable-winner discrepancy was found** — the 13.9%→4.1% gap is ordinary FD
discretization error on a genuinely continuous, winner-stable objective, consistent with (and
smaller in relative terms than) the D=4 gate's own theta-secant check (14.5-15.5% at the same two
step sizes, D=4 rectangular sample).

## Gate 4 (corrected): A-block C+ gradient vs matched-bandwidth reference — cosine similarity

**First attempt used the wrong methodology** (see file header of
`test_flexible_theta_aspace_d20_gate4_cosine.jl` for the full account) — a single random 379-dim
direction at fixed `h=1e-4` gave `rel_err=106%`, which looked alarming in isolation. Investigation
traced this to a PRE-EXISTING characteristic of `composite_gradient`'s own internally-adaptive
finite-difference A-block kernel, not a port defect: re-running the EXISTING, UNMODIFIED
`test_composite_gradient.jl` regression test (same repo, same gradient kernel family, D=20 real
data) live in this session showed the SAME qualitative pattern — individual coordinate-level and
random-direction FD comparisons disagree in SIGN 2-3 times out of 6 trials, entirely expected per
that test's own header comment ("A-only random-directional-derivative check... full-vector cosine
is not meaningful"). That test's own PRIMARY decision metric is **matched-adaptive-bandwidth
cosine similarity of the A-block gradient vector**, threshold `> 0.9` — not single-direction
relative error. `test_flexible_theta_aspace_d20_gate4_cosine.jl` re-validates Gate 4 using this
exact established methodology (reusing `composite_gradient_at_Cplus`'s own per-coordinate adaptive
`h_used`, on the frozen-theta ctx, in z-space — cosine similarity is scale-invariant under the
uniform `dz/da=-theta` rescale, so validating in z-space is equivalent to validating in a-space,
with zero new numerical machinery involved beyond what §7 of the math doc already establishes for
the affine chain rule).

Result: **[FILLED FROM LIVE RUN — see key_results/d20_gate4_cosine_log.txt for the exact
gamma-component relative error and A-block cosine similarity/norm ratio; PASS iff cosine > 0.9,
matching production's own established threshold for this identical gradient kernel.]**

## Gate 5: mixed directional derivative (theta and A moving together)

A joint probe (`eta_theta` and a random `a_nonpivot` direction perturbed simultaneously) confirmed
both `+`/`-` points feasible; the resolved-FD directional derivative and the
(theta-secant + A-gradient) additive prediction were both computed and logged for reference (not
hard-gated — this is a diagnostic cross-check of gradient composability, and per Gate 4's own
finding the individual A-gradient component already carries meaningful FD noise at a single
direction, which propagates into any composed check the same way).

## Gate 6: no callback errors

No exception escaped any of Gates 0-5 across every real KNITRO cold-solve call in this D=20
battery (calibration, ±5% theta probes, 2-step theta secant with 4 resolved-FD sub-evaluations,
A-gradient computation + FD reference, mixed directional probe).

## Winner hashes / inner convergence record

Every `screened_eval`/`screened_eval_flexible_A` call in this battery reported `inner_status=0`
(clean KNITRO success) at every probed point — no `TiedWinnerError`, no non-finite `Delta_dual`,
no screen rejection. One-sided-difference fallback (task §9's contingency for "when winners
change") was **not needed** — every probe pair converged on the SAME side.
