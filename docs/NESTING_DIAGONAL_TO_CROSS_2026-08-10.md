# What do the off-diagonal (cross-power) ZC restrictions cost? — evaluating the diagonal family's δ=1 best point under OZC-CROSS

2026-08-10, user request, alongside the 8-hour cross upper-bound runs.
Worktree `/bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09`, branch `feature/ozc-cross-2026-08-09`,
commit `f897485`. Script: `full_aod_diag/d4_exact/eval_originzc_handoff_point_2026-08-10.jl`.

## The question

Take the best δ=1.0 point the `paper_upper_v1` campaign found for the **diagonal** ORIGIN_ZC family
(gp = 0.9574025551574022, Δ\* = 0.9999667279174992, κ = 6.321080%, 6 completed multistarts, status
`TIME_LIMIT_STALLED`), and evaluate it under **OZC-CROSS**, which adds the `K_pair²` off-diagonal
cross-power restrictions. The families are nested, so Δ\* can only rise. By how much?

## 1. The transfer is exact — there is nothing to reinitialize

The handoff README warned that "any new off-diagonal etas need their own initialization". **They do
not exist.** From `cm_originzc_cross_target_layout.jl:38-39`, `OriginByPowerCrossLayout` has
`n_eta = K_mean*D` and a byte-identical `target_index` — the cross extension adds **no new outer
parameters at all**; its pair targets are *products* `ν[o,k1]·ν[p,k2]` of the same νs. Asserted live
before spending any solve:

| layout | `n_eta` | active | omitted dense idx |
|---|---|---|---|
| `OriginByPowerLayout(20,3,3)` | 60 | 59 | 22 |
| `OriginByPowerCrossLayout(20,3,3)` | 60 | 59 | 22 |

So the 439-vector `w = [gp(1); a_nonpivot(379); eta_nu(59)]` transfers verbatim. `gp` round-trips
exactly and the ν range reproduces the README's `[1.0760748687756763, 1.5988108982487386]` to the
last digit.

**Nothing was decoded by hand.** The raw `w0` was handed to `run_originzc_upper_checkpointed`, which
applies its own `xf_from_w_econ` and `scatter_nu_eff(aml, exp.(w[381:439]), originzc_profiled_nu_value(...))`.
Hand-rolling that decode is precisely the class of error this repo has committed before (memory
`feedback-campaign-seed-w0-encoding-not-raw-theta-free`).

## 2. Control first

Per CLAUDE.md, the diagonal family was run at the same point through the same path before any
conclusion was drawn from the cross number:

```
handoff states : Delta* = 0.9999667279174992
reproduced     : Delta* = 0.9999667279167646     agreement 7.3e-13, feasible=true verified=true
```

The transfer is verified.

## 3. Result: at the δ=1 optimum, the cross family is *infeasible*, not merely worse

```
inner solve nStatus = -300   (a confirmed-infeasibility certificate in this codebase --
                              memory feedback-knitro-300-confirmed-infeasible-not-unbounded --
                              NOT the unboundedness/lower_limit case)
KNITRO: "Could not evaluate objective or constraints at the initial point", tried perturbed
        points, EXIT: Evaluation error.  knitro_status = -502, n_eval = 0.
```

There is **no finite Δ\*** at this point for the cross family. Two independent mechanisms agree: the
pairwise certificate screen proved 11 of 13 candidate points infeasible *without any inner solve*
(`pairwise_hits=11 inner_solves_avoided=11`), and the 2 that passed the screen returned −300.

## 4. The quantitative gap, measured where both are finite

Δ\* is a property of θ and the restriction family, so it can be compared at any point. At the
calibration point both families are finite:

| | Δ\* at gp = 0.9840278851786317 |
|---|---|
| diagonal `:origin_by_power` | 0.003931288835716324 |
| cross `:origin_by_power_cross` | 0.034413319325412530 |
| **ratio** | **8.75×** |

Scaling the handoff point's 0.99997 by 8.75 predicts ≈ 8.75 — about **9× over the δ=1 budget**,
consistent with hard infeasibility rather than a near-miss. Independently corroborated by
`cm_originzc_checkpoint.jl`'s own source comment recording 0.0095 vs 0.0667 (7.0×) at a K=3/3 point
under a different configuration.

**The cross restrictions cost roughly an order of magnitude in Δ\*.**

## 5. What this does NOT establish: the κ cost

It is tempting to compare the cross family's κ to the diagonal's 6.321% and conclude the off-diagonal
restrictions cost more than half the identified-set width. **That comparison is invalid**, and was
corrected here after being stated once: it sets a *single start* against a *6-start multistart*.

Like-for-like, both single-start from the calibration point:

| family | budget | gp | Δ | κ |
|---|---|---|---|---|
| diagonal | 900 s, 61 evals | 0.9802020552 | 0.9999150863 | **2.954944%** |
| cross (live 8 h run, eval 40) | 2.5 h so far | 0.9838820121 | 0.9426767701 | **2.4079%** (unconverged) |
| diagonal, 6× multistart | full campaign | 0.9574025552 | 0.9999667279 | 6.321080% |

The decisive fact is the third row against the first: **multistart more than doubled the diagonal
family's own single-start value.** A single start from the calibration point is therefore not a family
frontier, and **the cross family's δ=1 κ remains unestablished.** Settling it needs a cross multistart
wave comparable to the diagonal's — a separate, much larger task.

## 6. Incidental findings

* **The handoff point is not optimal, as its README said.** The diagonal family restarted *from* it
  improved on it within 2447 s: gp = 0.9572894588, Δ = 0.9979430657 at eval 28, **κ = 6.337679%** (vs
  6.321080%). Worth re-extracting once S6–S9 finish.
* That run also probed Δ = 1.005–1.076 repeatedly and rejected nearly everything, independently
  corroborating that the handoff point sits essentially exactly on the δ=1 constraint.
* **The README's remedy is not available in this codebase.** `objective_mode = :min_delta_fixed_gp` /
  `gp_fixed` does not exist in this worktree's `run_originzc_upper_checkpointed` — it is a
  `paper_upper_v1` orchestrator primitive. Restore-then-push here would be new implementation work,
  not a configuration change. Given the gap is ~9× rather than marginal, restoration at fixed gp would
  be expected to fail — which by the README's own logic is the informative outcome (back gp off,
  do not loosen δ).
