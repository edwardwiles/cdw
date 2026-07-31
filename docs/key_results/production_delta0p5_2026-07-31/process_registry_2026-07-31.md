# Process registry -- launch time 2026-07-31 (recorded from host clock, ~2026-07-30 20:35 local)

All six chains launched via `scripts/melitz_production_chain_launch_2026-07-31.sh <anchor>
<direction>`, each as an independently harness-tracked background process (not nohup/disown'd),
5 Julia threads / 1 BLAS thread / one independent KNITRO session per process. Verified alive
(via `ps`) and past fixture-build startup (thread-startup report printed) within ~20-40s of
launch, per this repo's own "always check in early on long-running jobs" standing instruction.

| chain | anchor | direction | julia PID (attempt 1) | wrapper log |
|---|---|---:|---:|---|
| 1 | `current_calibration` | upper | 3352510 | `logs/current_calibration_upper_attempt1_20260730_203518.log` |
| 2 | `current_calibration` | lower | 3352604 | `logs/current_calibration_lower_attempt1_20260730_203520.log` |
| 3 | `reduced_q_pre_switch` | upper | 3352722 | `logs/reduced_q_pre_switch_upper_attempt1_20260730_203523.log` |
| 4 | `reduced_q_pre_switch` | lower | 3352804 | `logs/reduced_q_pre_switch_lower_attempt1_20260730_203525.log` |
| 5 | `reduced_q_post_switch` | upper | 3352857 | `logs/reduced_q_post_switch_upper_attempt1_20260730_203527.log` |
| 6 | `reduced_q_post_switch` | lower | 3352905 | `logs/reduced_q_post_switch_lower_attempt1_20260730_203529.log` |

Note: the julia PID shown is for launch attempt 1 only. If a chain's wrapper restarts it
(transient crash recovery), a NEW PID is assigned each attempt -- the checkpoint file
(`checkpoints/<chain>.jls`) and per-attempt log filenames (`logs/<chain>_attempt<N>_<timestamp>.log`)
are the authoritative record across restarts, not this table's PID column.

State fingerprints (D, W, seed, sigma, focal, anchor_label, direction, q_free_hash, theta0_hash,
git_commit) are printed by each chain at startup and re-validated on every resume; see each
chain's own attempt-1 log for the exact printed fingerprint, and `checkpoints/<chain>.jls`
(`state.fingerprint` field) for the persisted copy used for resume validation.
