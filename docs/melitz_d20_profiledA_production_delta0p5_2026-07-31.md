# Melitz D20 profiled-A production campaign, delta=0.5 (2026-07-31)

**STATUS: campaign launched, running unattended in the background as six process-isolated
chains. This document is being written progressively -- the sections below that require the
chains' own results are marked `[PENDING -- fill in after chain completion]` and will be
completed once each chain reaches a terminal status and its final cold verification
(`scripts/melitz_production_final_verify_2026-07-31.jl`) has run.**

Governing prompt: launch a restartable production search of the successful profiled-(A) D20
Melitz method (`docs/melitz_d20_profiled_A_welfare_continuation_2026-07-30.md`,
`docs/melitz_profiledA_parallel_speed_and_cutoff_portfolio_2026-07-30.md`) and produce the best
verified incumbents obtainable within a fixed overnight budget. Does not test whether profiling
works (already established), does not build another cutoff-gradient method/chamber
graph/parameterization/direct high-dimensional welfare optimizer, does not modify the Ricardian
implementation, does not continuously optimize the free cutoff coordinates.

Repo: `trade_robustness_modular`, branch `melitz/fullD-delta-star`, HEAD at launch `169f77d`.

## Established result reproduced at preflight (2026-07-31)

Cold-reverified fresh, under current code, before launch (`scripts/melitz_production_preflight_2026-07-31.jl`,
log `docs/key_results/production_delta0p5_2026-07-31/logs/preflight_2026-07-31.log`):

- Anchor `Delta0 = 0.4832764950` (target `0.483276`), `GT0 = 6.290641%`.
- Stored fixed-q profiled-A headline point: `GT=7.096891%`, `Delta*=0.4990186631` (target
  `GT~7.0969%`, `Delta*~0.499019`), `FiniteSolved`, matches the stored value to `<1e-6`.

## Production scope

- `D=20`, `W=80,000`, `delta=0.5`, real-D20 (`noah_D20`, focal=fra, `sigma=2.5`, `seed=1`).
- Three free-cutoff anchors (the 3/6 that survived the prior session's own Phase 7 feasibility
  screen -- `docs/melitz_profiledA_parallel_speed_and_cutoff_portfolio_2026-07-30.md`):
  `current_calibration`, `reduced_q_pre_switch`, `reduced_q_post_switch`.
- Both gains-from-trade directions (upper, lower) per anchor -- six independent production
  chains, each a separate OS process (5 Julia threads, BLAS=1, one independent KNITRO session,
  no shared mutable state).
- Method per chain: hold the anchor's free-`q` coordinates fixed; reconstruct the full `q`
  vector through the existing q-gravity map at every welfare point; genuinely optimize `A`
  (fixed-q A-middle loop, `solve_melitz_fixed_q_A_profile_v2`); reconstruct `f`; fully reoptimize
  and verify the inner LFD; adaptive two-start policy (`melitz_middle_two_start_adaptive!`) at
  every point (primary continuation start; compensated cellwise-`p*` fallback per the adaptive
  policy's own trigger conditions).
- Welfare search: wide sequential expansion (initial step `0.5` GT-percentage-points, growth up
  to `1.5x` on acceptance, capped at `2.0` percentage points) until a local boundary bracket is
  found, then safeguarded linear interpolation (clamped to the interior 20-80% of the bracket,
  falling back to an ordinary midpoint when interpolation fails or the local profile is
  nonmonotone), at most 3 boundary-polish evaluations, stopping at
  `0 <= delta - Delta* <= 0.002` or a bracket width `<=0.01` GT-percentage-points.
- Limits per chain: 6h wall time, 40 distinct profiled welfare points, 3 boundary-polish points
  after bracketing, stop after 6 consecutive non-improving attempts, no unbounded retries.
- Checkpoint after every completed welfare point (`docs/key_results/production_delta0p5_2026-07-31/checkpoints/<anchor>_<direction>.jls`),
  atomic write, resumable (fingerprint-validated, cold-reverified incumbent on resume).

Full launch mechanics, exact commands, hardware/software versions:
[`docs/key_results/production_delta0p5_2026-07-31/provenance_2026-07-31.md`](key_results/production_delta0p5_2026-07-31/provenance_2026-07-31.md).

## Source map (reused, not modified)

| concern | file |
|---|---|
| repaired fixed-q A-middle-loop driver (cap handling, incumbent retention, FC/GA dedup) | `src/melitz/fixed_q_a_middle_loop.jl` (`solve_melitz_fixed_q_A_profile_v2`) |
| adaptive two-start policy | `src/melitz/fixed_q_a_middle_loop.jl` (`melitz_middle_two_start_adaptive!`) |
| exact A gradient | `src/melitz/exact_a_gradient.jl` |
| fast O(W+D) focal-link update | `src/melitz/moment_operator.jl` (`melitz_update_moment_operator!`) |
| cutoff-anchor construction (reduced-q basis) | `src/melitz/reduced_q_controller.jl`, `reduced_q_subspace.jl` |
| this campaign's production driver (per anchor x direction chain) | `scripts/melitz_production_chain_2026-07-31.jl` |
| crash-aware launch/resume wrapper | `scripts/melitz_production_chain_launch_2026-07-31.sh` |
| 6-way orchestrator (manual/reproducible use) | `scripts/melitz_production_launch_all_2026-07-31.sh` |
| final cold verification (fresh process, per chain) | `scripts/melitz_production_final_verify_2026-07-31.jl` |
| preflight | `scripts/melitz_production_preflight_2026-07-31.jl` |

## Results

`[PENDING -- fill in after chain completion]`

### Aggregate upper/lower incumbent table

`[PENDING]`

| direction | winning anchor | best verified GT | Delta* | fixed-A/f Delta at same GT | improvement | wide points | polish points |
|---|---|---:|---:|---:|---:|---:|---:|
| upper | | | | | | | |
| lower | | | | | | | |

### Per-chain detail

`[PENDING -- one subsection per chain: current_calibration/upper, current_calibration/lower,
reduced_q_pre_switch/upper, reduced_q_pre_switch/lower, reduced_q_post_switch/upper,
reduced_q_post_switch/lower -- final status, best incumbent, wall time, point counts,
nonmonotonicity notes, any crash/resume events.]`

## Required outputs

1. This document.
2. Per-chain machine-readable point history: `docs/key_results/production_delta0p5_2026-07-31/points/<anchor>_<direction>_points.csv`.
3. Per-chain checkpoint directory: `docs/key_results/production_delta0p5_2026-07-31/checkpoints/`.
4. Aggregate upper/lower incumbent table: see Results above (also
   `docs/key_results/production_delta0p5_2026-07-31/aggregate_summary_2026-07-31.csv`, `[PENDING]`).
5. Fixed-(A/f) comparison results: `docs/key_results/production_delta0p5_2026-07-31/final_verification/*_final_verify.csv`, `[PENDING]`.
6. Launch and resume scripts: `scripts/melitz_production_chain_launch_2026-07-31.sh`,
   `scripts/melitz_production_launch_all_2026-07-31.sh` (same script resumes -- idempotent).
7. Provenance manifest: `docs/key_results/production_delta0p5_2026-07-31/provenance_2026-07-31.md`.

## Final questions

`[PENDING -- answered once all six chains reach a terminal status and are cold-verified]`

1. Best verified upper-bound gains-from-trade incumbent?
2. Best verified lower-bound incumbent?
3. Which cutoff anchor won in each direction?
4. `Delta*` at each winner?
5. How much worse was the fixed-(A/f) profile at the same welfare point?
6. How many wide exploration points and boundary-polish points were used?
7. Did safeguarded interpolation reduce the need for bisection?
8. Did any chain display materially nonmonotone profiled behavior?
9. Did any process crash, and was checkpoint recovery successful?
10. How much variation was there across cutoff anchors?
11. Is there sufficient evidence that the outer procedure performs productive nuisance-parameter
    search beyond calibration?

A verified feasible incumbent, the best result in this finite cutoff-anchor portfolio, and a
global optimum are three different things -- this campaign reports the second, never claims the
third.
