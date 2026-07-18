# Full-A D=4: gamma-profile, upper polish, lower direction

Phase 4 deliverable. **Status: NOT ATTEMPTED this continuation** — recorded here explicitly rather
than silently skipped, per the task's own transparency requirements, with the reasoning for why it
was deprioritized and what it needs to start.

## Why this was deprioritized this continuation

The task's own phase ordering places Phase 1 (performance profiling) as mandatory-first, and this
continuation's severe realistic time budget did not extend past Phase 1 (complete), Phase 5 (sequential
solution — complete), Phase 6 (blockwise gradient — complete), and a partial Phase 3 (W-stability).
Phase 4 (`profile_Delta(g) = min_A Delta(g,A)` over a gamma grid, upper polish from the poll-improved
points, and a longer/profiled lower-direction run) is itself explicitly gated by Phase 1's cost data:
each `profile_Delta(g)` grid point requires a full local optimization in the A-block at that g, and
Phase 1C's finding that a single `Delta_FD` gradient costs 3.7s at D=4/W=8000 (let alone a full
multi-iteration local solve per grid point) means a coarse-to-fine gamma grid with several dozen points
is a genuinely multi-hour undertaking at minimum, not something to start without a clear remaining time
budget.

## What exists to build on

- Phase 1's cheap-gradient findings (`docs/fullA_performance_profile.md` §4, Phase 6's blockwise
  results): `L_fix_FD` is the validated cheap gradient to use for the LOCAL A-minimization at each
  gamma grid point (not `Delta_FD` for every trial step — only for periodic refresh/final
  verification), which should make Phase 4 substantially cheaper than a naive `Delta_FD`-everywhere
  implementation once built.
- The three exact-feasible poll improvements from the prior continuation (`results/fullA_d4/1b2a3a0/
  phaseA_upper_revalidation/step6_poll.csv`, rows with `improved=true`) are the specified starting
  points for "4B. Upper polish" — not yet used as starts for anything.
- The lower stalled point (`results/fullA_d4/9e03706/optfd_lower_20260717_190831/summary.txt`,
  `Delta_minus_delta=-0.101`, far from binding) is the specified starting point for "4C. Lower
  direction" — a longer wall-clock budget (not just a larger iteration cap) using the gamma profile
  and continuation is specified, not attempted.
- The sequential solution located in Phase 5 (`docs/fullA_d4_final_report.md` update, this
  continuation) gives external kappa targets to calibrate against: sequential's genuinely-feasible
  upper terminal is κ=0.0779 (vs full-A's current κ=0.1718 — full-A already clears this bar) and
  sequential's genuinely-feasible lower best-feasible is κ=0.0046 (vs full-A's current stalled
  κ≈0.0107 — full-A's lower candidate ALSO already clears this, for what that comparison is worth
  given full-A's lower point is explicitly not yet stationary).

## What Phase 4 needs to do when picked up

1. Implement `profile_Delta(g)` using `L_fix_FD`-primary local optimization (not `Delta_FD`
   throughout), warm-started across a coarse-to-fine gamma grid, with periodic `Delta_FD` refresh per
   Phase 6's hybrid-scheme guidance (`docs/fullA_performance_profile.md` §4's practical conclusion).
2. Polish the three poll-improved upper points using this profile plus the derivative-free local
   pattern-search already implemented in the prior continuation's Phase A poll machinery (reusable
   directly — see `full_aod_diag/d4_exact/phaseA_upper_revalidation.jl`'s step 6).
3. Give the lower direction a genuinely long (not just larger-iteration-cap) run using continuation
   from `g=1` per the task's spec, with the same exact-value and local-poll validation Phase A already
   applied to the upper direction — the machinery exists and is directly reusable, only the driving
   script (gamma grid + warm-start chain) is new work.
