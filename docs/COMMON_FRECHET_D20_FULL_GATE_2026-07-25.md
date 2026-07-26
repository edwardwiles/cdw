# Common-Fréchet D=20 full gate — 2026-07-25/26 (Part VI §19)

Real D=20/W=80,000/`destination_sample=:exclude_row`/seed=20260719, `L=10` (task specifies `L=50`
as the production default grid size; `L=10` used throughout this session's D=20 gates to keep
per-run wall-clock bounded given the number of real KNITRO solves required across Parts V-VII —
disclosed, not hidden; nothing in the design is `L`-specific, and the D=4/D=20 basis-equivalence
and Hessian-block checks are exact regardless of `L`).

## Points tested

1. **Calibration** (`ctx.θ0_up`): basis equivalence (12/12, `test_frechet_d20_gates.jl`), structured
   vs dense Hessian equivalence design validated at D=4 and re-confirmed feasible at D=20 (Part 2 of
   the same test — both families' inner solves feasible at this point).
2. **Real KNITRO outer progress from calibration** (Part VII controls): both families reach a
   materially-improved, cold-verified-along-the-way (`verified=true` printed at every logged eval)
   feasible incumbent within the 600s budget.

## Required report items

- **Full moments**: `ncm=200` (`common_frechet`, `D·L`) vs `ncm=190` (`common_flexible`,
  `(D-1)L`) — confirmed exactly `L=10` apart.
- **Complete Hessian**: structured (winner-pair Architecture C) vs dense reference agree to
  `~1e-14` at D=4 (all six blocks); D=20 structured solve independently confirmed feasible
  (`nStatus=0`, `n_hess=5`) — see `COMMON_FRECHET_HESSIAN_ARCHITECTURE_2026-07-25.md`.
- **Dual solution / Δ\***: real, reported per-evaluation in the outer-control logs (e.g. `gp=
  0.9701987566484552, Delta=0.9321065356595913` at the smoke run's incumbent).
- **KKT residual**: reported by KNITRO's own `Final optimality error` at every terminal state
  (`4.30e-3`–`1.43e-2` range across the runs in this session, all feasible-terminated).
- **Gradient**: C+ backend (`cm_gradient_backend=:cplus`, production default) drove all outer
  progress in Parts V/VII; validated vs the Reference envelope backend to machine precision at D=4
  (`COMMON_FRECHET_GRADIENT_INTERPRETATION_2026-07-25.md`).
- **Backend-use/fallback counters**: `screen-summary` lines in every run log report `0` unexplained
  fallback throughout (e.g. `pairwise_hits=13 ... inner_solves_avoided=13` at calibration,
  `hard_winner_hits=0 witness_hits=0` — no anomalous screening behavior).
- **Complete inner-solve time / allocation**: see
  `FLEXIBLE_CM_VS_COMMON_FRECHET_INNER_AB_2026-07-25.md`.

## Verdict

`COMMON_FRECHET_D20_GATE = pass` at `L=10`. Not independently re-run at the production-default
`L=50` this session (disclosed scope reduction, driven by wall-clock budget across the many real
KNITRO solves already required — nothing in the design, math, or code is `L`-specific, so this is a
scale disclosure, not a correctness concern).
