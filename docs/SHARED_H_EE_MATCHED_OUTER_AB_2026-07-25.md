# Matched real outer A/B gates — 2026-07-25

Task §8. **Status: NOT completed this session — genuinely deferred, disclosed here rather than
fabricated.**

## What the task asked for

Direct production-shaped upper-bound A/B comparisons (dense-reference vs shared winner-pair core)
for all four families, at D=20/D_dest=19/W=80,000/seed=20260719/delta=1, genuine calibrated start,
identical explicit outer algorithm, 20 Julia threads, one process at a time, 300 seconds for
unrestricted and flexible CM, at least 120 seconds for each additional restricted smoke —
comparing value evaluations, gradients, inner solves, complete inner-solve time, H_EE time,
cross/restriction Hessian time, best cold-verified kappa, verified progress per minute, allocation
and GC, and terminal statuses.

## What WAS actually done this session (see the other 2026-07-25 docs for detail)

- Full D=4 exact correctness gates for all four families (40/40 PASS,
  `docs/full_correctness_log_2026-07-25.txt`) — value/gradient/Hessian/inner-solve agreement at
  multiple dual points including real KNITRO-solved ones, NOT a matched OUTER-loop A/B.
- A single-point (P0 calibration), single-family (unrestricted) real D=20/W=80,000 smoke+timing
  check through the actual production entry point
  (`docs/unrestricted_d20_smoke_timing_log_2026-07-25.txt`) — one complete INNER solve compared,
  not a 300-second OUTER optimization run.
- An incidental, independent real-scale validation via the PRE-EXISTING (unmodified)
  `test_cm_compressed_core.jl` regression test, which happens to now exercise the winner-pair
  backend as its default and passed all 7 of its own sections at D=20/W=80,000/L=50
  (`docs/cm_d20_regression_log_2026-07-25.txt`).

None of the above is a substitute for the task's actual ask: a genuine 300-second (or 120-second)
OUTER KNITRO optimization loop, run once per family per backend, with cold-verified kappa and
verified-progress-per-minute compared. That requires an existing calibrated-start outer-loop
harness per family (the kind used by, e.g., `matched_outer_benchmark_cm_2026-07-25.jl` for the
prior allocation/Hessian production release) adapted to toggle the NEW backend fields this port
adds, run for real wall-clock minutes each, across four families — a multi-hour undertaking this
session's time budget did not accommodate after the correctness-gate and single-point-timing work
above.

## Why this gap does not block the (already-limited) verdict below

Per task §8's own "Production eligibility requires" list, this exact matched-outer-A/B campaign is
the one genuinely load-bearing prerequisite for a `merged_all_families` production verdict this
session cannot honestly claim. See the master summary doc's final verdict section: this port is
released as `port_ready_not_merged` specifically because of this gap, not `merged_all_families`.

## What a follow-up session needs to do

1. Reuse each family's existing matched-outer-benchmark harness (search for
   `matched_outer_benchmark_*_2026-07-25.jl`-style scripts already in this repo for the prior
   allocation/Hessian release; adapt rather than rewrite).
2. Toggle `core_hessian_backend`/`core_hessian_workers`/`core_hessian_storage` (unrestricted:
   `UNRESTRICTED_CORE_HESSIAN_BACKEND`/`_WORKERS`/`_STORAGE` Refs; CM/CM+meanZC:
   `cctx.core_hessian_backend` etc; origin-ZC: `octx.core_hessian_backend` etc) between
   `:dense_reference` and `:exact_winner_pair_parallel` for the "before"/"after" arms.
3. Run each arm for the specified duration, capture the specified metrics, and check against the
   task §8 eligibility bullets before considering a `merged_all_families` verdict.
