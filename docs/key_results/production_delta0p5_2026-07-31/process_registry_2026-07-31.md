# Process registry -- launch time 2026-07-31 (recorded from host clock, ~2026-07-30 20:35 local)

All six chains launched via `scripts/melitz_production_chain_launch_2026-07-31.sh <anchor>
<direction>`, each as an independently harness-tracked background process (not nohup/disown'd),
5 Julia threads / 1 BLAS thread / one independent KNITRO session per process. Verified alive
(via `ps`) and past fixture-build startup (thread-startup report printed) within ~20-40s of
launch, per this repo's own "always check in early on long-running jobs" standing instruction.

| chain | anchor | direction | julia PID (final launch) | wrapper log |
|---|---|---:|---:|---|
| 1 | `current_calibration` | upper | 3352510 | `logs/current_calibration_upper_attempt1_20260730_203518.log` |
| 2 | `current_calibration` | lower | 3352604 | `logs/current_calibration_lower_attempt1_20260730_203520.log` |
| 3 | `reduced_q_pre_switch` | upper | 3352722 | `logs/reduced_q_pre_switch_upper_attempt1_20260730_203523.log` |
| 4 | `reduced_q_pre_switch` | lower | 3352804 | `logs/reduced_q_pre_switch_lower_attempt1_20260730_203525.log` |
| 5 | `reduced_q_post_switch` | upper | 3394649 | `logs/reduced_q_post_switch_upper_attempt1_20260730_204107.log` |
| 6 | `reduced_q_post_switch` | lower | 3394756 | `logs/reduced_q_post_switch_lower_attempt1_20260730_204109.log` |

Note: the julia PID shown is for the FINAL (working) launch of each chain. If a chain's wrapper
restarts it (transient crash recovery), a NEW PID is assigned each attempt -- the checkpoint file
(`checkpoints/<chain>.jls`) and per-attempt log filenames (`logs/<chain>_attempt<N>_<timestamp>.log`)
are the authoritative record across restarts, not this table's PID column.

## Live bug caught and fixed at launch (2026-07-31, ~20:36-20:41 local)

The two `reduced_q_post_switch` chains crashed on their very first (anchor) evaluation,
independently confirmed twice (`current_calibration`/`reduced_q_pre_switch` were unaffected).
**Root cause**: `scripts/melitz_production_chain_2026-07-31.jl`'s anchor-establishment step
built the fixed-q linear ordering/same-bin constraint system (`sys0`) and the initial feasible
start (`A_start0`) from `theta_plain0_d20` -- the CALIBRATION's own theta -- rather than the
ANCHOR's own theta (same `g0`, but with `q_free` replaced by `q_free_fixed`). For the
`current_calibration` anchor these are identical (`q_free_fixed == q_free0_d20`), so that chain
was unaffected; for `reduced_q_pre_switch`/`reduced_q_post_switch` they are DIFFERENT points, so
the wrong constraint rows were registered, the anchor's own true-feasible start point was
misclassified, and the (zero-gradient) fixed 50.0 cap-barrier value was returned as an
immediate 0-iteration "locally optimal" KNITRO solution -- failing the internal
`FiniteSolved`-required assertion on every attempt (deterministically, so the crash-aware
wrapper correctly identified "same target crashed twice" and would have stopped the chain
rather than retrying forever, exactly as designed -- the safety net worked, even though the
actual root cause was a code bug rather than the anticipated `mul_G!` SIGSEGV).

**Fix**: build `sys0`/`A_start0` from an anchor-specific `theta_anchor_plain0` (copies
`theta_plain0_d20` then overwrites the free-`q` block with `q_free_fixed`), matching
`scripts/melitz_phase7_cutoff_portfolio_2026-07-30.jl`'s own correct pattern. **Verified live**:
a focused smoke test of the `reduced_q_post_switch` anchor point after the fix reproduces
`Delta*=0.483265, FiniteSolved` -- an EXACT match to the Phase 7 cutoff-portfolio session's own
documented reference value for this anchor (`docs/melitz_profiledA_parallel_speed_and_cutoff_portfolio_2026-07-30.md`,
Phase 7 table). The two broken chains were killed, their stale crash-target/checkpoint markers
removed, and relaunched cleanly with the fixed driver at PIDs 3394649/3394756. The other four
chains (`current_calibration` x2, `reduced_q_pre_switch` x2) were never affected and ran
continuously throughout.

State fingerprints (D, W, seed, sigma, focal, anchor_label, direction, q_free_hash, theta0_hash,
git_commit) are printed by each chain at startup and re-validated on every resume; see each
chain's own attempt-1 log for the exact printed fingerprint, and `checkpoints/<chain>.jls`
(`state.fingerprint` field) for the persisted copy used for resume validation.
