# Flexible-theta post-omit-ROW matched practical-value comparison — 2026-07-25 (task §15)

**STATUS: IN PROGRESS. Every number in this document is either (a) explicitly marked VERIFIED
with a real committed log citation, or (b) explicitly marked PENDING. No number here is
estimated, guessed, or carried over from any other session's output (a separate concurrent
Claude Code session on this shared host, scratchpad UUID `f5095a84-24d9-4a6a-b9db-7ccf77ef9ba6`,
was running its own, differently-scoped D=20 comparison script
(`run_d20_300s_matched_comparison_2026-07-25.jl`, comparing two internal a-space gp-coordinate
variants against each other) — that is NOT this port's work and none of its numbers appear
below.**

## Methodology (both arms, both deltas)

Both arms use `full_aod_diag/d4_exact/matched_comparison_fixed_vs_flexible_A.jl`:
- Identical genuine calibrated starting point: `ctx0.θ0_up[ctx0.free_idx]` (the real `A_od`/`gp`
  calibration block, NOT `zfree=0` — see CLAUDE.md standing warning).
- Identical data/draws: `W=80,000`, pseudorandom seed `20260719`, `destination_sample=:exclude_row`.
- Identical algorithm config: both `run_polish_checkpointed` (fixed arm) and
  `run_polish_checkpointed_flexible_theta_A` (flexible arm) load the SAME
  `csw_outer_wallclock_sr1.opt` file by default (`algorithm auto`, `hessopt 3`=SR1) — confirmed
  by direct inspection of the option file, not assumed.
- Identical screens/cache/incumbent logic: both call into the same `screened_eval`/`DualBank`/
  `SafeExactCache`/`incumbent_logic.jl` machinery (flexible arm via `screened_eval_flexible_A`,
  which delegates to the unmodified `screened_eval`).
- Identical gradient backend: `price_cache_backend=:cplus` explicitly passed to both arms.
- `find_smallest=true` (upper-kappa direction, matching the brief's §0 established baseline
  numbers' own convention).
- 600-second outer-solver wall-clock budget (`maxtime_real=600.0`) for the initial comparison.
- Cold-verification of the champion at the end of each arm (a fresh `warm=false` evaluation,
  independent of anything the outer solve itself cached).

## Delta=1, 600s

| Arm | kappa | wall (s) | n_eval | knitro_status | cold-verify Delta_dual |
|---|---|---|---|---|---|
| fixed | PENDING | PENDING | PENDING | PENDING | PENDING |
| flexible | PENDING | PENDING | PENDING | PENDING | PENDING |

Reference (task §0, established prior to this port, NOT re-derived here): at the OLD
pre-omit-ROW production context, 600s, delta=1: fixed=0.065159, old flexible z-space=0.054102,
new flexible a-space=0.069648. This port's own post-omit-ROW numbers (above) are a NEW
measurement in a genuinely different (rectangular, post-omit-ROW) production context and are
not expected to reproduce those exact figures — the comparison of interest is fixed-vs-flexible
WITHIN this port's own matched run, not against the pre-omit-ROW numbers.

## Delta=2, 600s

| Arm | kappa | wall (s) | n_eval | knitro_status | cold-verify Delta_dual |
|---|---|---|---|---|---|
| fixed | PENDING | PENDING | PENDING | PENDING | PENDING |
| flexible | PENDING | PENDING | PENDING | PENDING | PENDING |

## 2-hour delta=2 follow-up

Per task §15: run only if the 600s delta=2 result is favorable to flexible AND the branch is
otherwise stable. **Decision: PENDING, contingent on the 600s delta=2 result above.**

## Verdict

PENDING — filled once all real runs above complete and are cold-verified.
