# Provenance manifest -- D20 profiled-A production campaign, delta=0.5 (2026-07-31)

- **Repo**: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular`
- **Branch**: `melitz/fullD-delta-star`
- **HEAD commit at campaign launch**: `169f77d4a9eedb908a078095629a0be77d9c61ca`
  ("Implement and wire in the focal-link O(W*D)->O(W+D) prefix-sum optimization (2026-07-30)")
- **Ancestry**: `169f77d` (focal-link fast path, wired into production, no opt-out) ->
  `0622db1` (Melitz profiledA parallel speed and cutoff portfolio) -> `b301aa0` (D20 profiled-A
  welfare continuation at delta=0.5) -- both prior governing-prompt docs read in full before this
  campaign was designed.
- Committed locally at the end of this session; **not pushed** (this repo's own standing
  "confirm before pushing to production" convention).

## Hardware

- Host: `demand.mit.edu`, `Linux 4.18.0-553.137.1.el8_10.x86_64`
- CPU: Intel(R) Xeon(R) Platinum 8270 @ 2.70GHz, 208 logical CPUs
- RAM: 3.0 TiB total

## Software

- Julia: `juliaup` toolchain, `v1.12.6` (`$HOME/.juliaup/bin/julia` -- NOT `/opt/shared_sw`, which
  is a broken 1.10.11 install per this repo's own standing memory note)
- KNITRO: Artelys Knitro 13.0.1, academic license (`/opt/shared_sw/knitro/13.0.1`)
- BLAS: `LinearAlgebra.BLAS.set_num_threads(1)` explicitly in every process (verified by
  `melitz_thread_startup_report()`'s own printed line at the start of every script)

## Process / thread layout

Six independent OS processes (NOT Julia tasks inside one process), one per anchor x direction:

| chain | anchor | direction | Julia threads | BLAS threads | KNITRO session |
|---|---|---|---:|---:|---|
| 1 | `current_calibration` | upper | 5 | 1 | independent |
| 2 | `current_calibration` | lower | 5 | 1 | independent |
| 3 | `reduced_q_pre_switch` | upper | 5 | 1 | independent |
| 4 | `reduced_q_pre_switch` | lower | 5 | 1 | independent |
| 5 | `reduced_q_post_switch` | upper | 5 | 1 | independent |
| 6 | `reduced_q_post_switch` | lower | 5 | 1 | independent |

Total budget: 30 Julia threads across 6 processes (this session's own `5x4`/`4x5`-throughput
finding from `docs/melitz_profiledA_parallel_speed_and_cutoff_portfolio_2026-07-30.md` Phase 6
used a 20-thread/5-process budget; this campaign uses 5 threads/process x 6 processes = 30
threads total, matching the governing prompt's own explicit "Julia threads=5" instruction, not
re-optimized here). No shared mutable bundles or solver sessions -- each process builds its own
fresh `MelitzCCBundle`/`MelitzInnerSession` from the real-D20 fixture.

Each chain launched via `scripts/melitz_production_chain_launch_2026-07-31.sh <anchor>
<direction>`, itself invoking `julia --project=. -t 5 scripts/melitz_production_chain_2026-07-31.jl
<anchor> <direction>`.

## Data / QMC / state fingerprint

- `W=80,000`, real-D20 (`noah_D20`, focal=`fra`, `sigma=2.5`, `seed=1`) throughout.
- Anchor `theta0` loaded from `docs/key_results/melitz_qbw_phase3_theta_q_2026-07-29.csv`
  (`realD20_seed1_W80000`, key `0.5`).
- `real_data/noah_D20/*.csv` mtimes (checked at campaign launch, all four files):
  `2026-07-23 16:56:15` -- inherited data-currency caveat from the prior two sessions (the user
  flagged mid-session on 2026-07-30 that "the underlying data has just changed"; this campaign
  uses the SAME on-disk snapshot the prior two profiled-A sessions used, unchanged since then).
- Reduced-q basis: `PowerScaledQBandwidth(1e-3, 80_000, 0.5)`, `target_switches=100`
  (`melitz_build_reduced_q_stage`) -- EXACT reuse of the negative-switch audit's own basis
  direction, identical to Phase 7/8 of the parallel-speed-and-portfolio session.
- Anchor cutoff perturbations (Phase 7-verified survivors of the feasibility screen only):
  - `current_calibration`: `q_free0` unchanged.
  - `reduced_q_pre_switch`: `q_free0 + (-1.0)*6.31e-3*b_q`.
  - `reduced_q_post_switch`: `q_free0 + (-1.0)*1.26e-2*b_q`.
- Per-chain fingerprint (validated on every resume, `scripts/melitz_production_chain_2026-07-31.jl`):
  `(D, W, seed, sigma, focal, anchor_label, direction, hash(q_free_fixed), hash(theta0), git_commit)`.
  A mismatch on the state-defining fields (D/W/seed/sigma/anchor/theta0) aborts the resume loudly
  rather than silently continuing on stale/mismatched state; a git-commit mismatch alone only
  warns (code may have changed harmlessly between restarts).

## Preflight (run before launch, `scripts/melitz_production_preflight_2026-07-31.jl`)

Log: `docs/key_results/production_delta0p5_2026-07-31/logs/preflight_2026-07-31.log`. All checks
passed:

- Anchor `Delta0` cold-reproduces `0.4832764950` (target `0.483276`).
- Stored headline point (`scripts/melitz_d20_profiledA_continuation_state_2026-07-30.jls`,
  upper-direction extreme) cold-reverifies fresh: `GT=7.096891%`, `Delta*=0.4990186631`
  (target `GT~7.0969%`, `Delta*~0.499019`), `FiniteSolved`, matching stored value to `<1e-6`.
- LFD recovery agrees with the cold-reverified `Delta` at the headline point.
- One ordinary profiled point at the `current_calibration` anchor (`g0`):
  `Delta_incumbent=0.287953`, `FiniteSolved`, no worse than the verified start (`0.483276`),
  `unique_inner_solves(64) <= unique_A_points(64)` (exact equality), `cache_hits=57`.
- `BLAS.get_num_threads()==1`; `forbid_dense_fallback=true` (no dense G); repaired v2 driver
  (`cap_handling=:barrier`, `cap_barrier_multiple=5.0`) with callback cache active; focal-link
  fast path wired into production unconditionally as of commit `169f77d` (no opt-out flag, so it
  cannot have been silently disabled).

## Exact launch commands

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Preflight (once, before launch):
julia --project=. -t 10 scripts/melitz_production_preflight_2026-07-31.jl

# Six production chains (launched as six SEPARATE harness-tracked background processes this
# session; equivalent standalone form for manual/reproducible use):
bash scripts/melitz_production_chain_launch_2026-07-31.sh current_calibration upper
bash scripts/melitz_production_chain_launch_2026-07-31.sh current_calibration lower
bash scripts/melitz_production_chain_launch_2026-07-31.sh reduced_q_pre_switch upper
bash scripts/melitz_production_chain_launch_2026-07-31.sh reduced_q_pre_switch lower
bash scripts/melitz_production_chain_launch_2026-07-31.sh reduced_q_post_switch upper
bash scripts/melitz_production_chain_launch_2026-07-31.sh reduced_q_post_switch lower

# Or all six at once (blocks until all exit):
bash scripts/melitz_production_launch_all_2026-07-31.sh

# Resume (idempotent -- re-run the SAME command; a finished chain exits immediately, a
# partially-run chain resumes from its own last checkpoint):
bash scripts/melitz_production_chain_launch_2026-07-31.sh <anchor> <direction>

# Final cold verification (fresh process, per chain, after the chain reaches a terminal status):
julia --project=. -t 5 scripts/melitz_production_final_verify_2026-07-31.jl <anchor> <direction>
```

## Output locations

- Checkpoints: `docs/key_results/production_delta0p5_2026-07-31/checkpoints/<anchor>_<direction>.jls`
- Per-point CSV history: `docs/key_results/production_delta0p5_2026-07-31/points/<anchor>_<direction>_points.csv`
- Logs (per launch attempt): `docs/key_results/production_delta0p5_2026-07-31/logs/`
- Crash artifacts (only written if a chain's SIGSEGV/crash recurs at the identical welfare-point
  target twice in a row, or the wrapper exhausts its restart budget): `docs/key_results/production_delta0p5_2026-07-31/crash_artifacts/<chain>/`
- Final cold-verification CSVs: `docs/key_results/production_delta0p5_2026-07-31/final_verification/`
