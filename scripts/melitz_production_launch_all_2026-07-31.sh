#!/bin/bash
# Launches (or RESUMES -- this script is idempotent, see below) all six anchor x direction
# production chains as six SEPARATE OS processes
# (docs/melitz_d20_profiledA_production_delta0p5_2026-07-31.md governing prompt).
#
#   3 cutoff anchors (current_calibration, reduced_q_pre_switch, reduced_q_post_switch)
#   x 2 directions (upper, lower)
#   = 6 independent chains, each: 5 Julia threads, BLAS=1, one independent KNITRO session,
#     no shared mutable bundles or solver sessions (separate OS processes, not Julia tasks).
#
# RESUME: re-running this exact script after a partial/interrupted campaign is the correct way
# to resume -- each chain's own Julia driver (melitz_production_chain_2026-07-31.jl) checks for
# its own checkpoint file first and resumes from the last completed welfare point rather than
# restarting from calibration; a chain that already reached a terminal status (converged/
# budget_exhausted/max_points/stalled/max_wall) exits immediately when re-launched, a no-op.
#
# NOTE ON HOW THIS SESSION ACTUALLY LAUNCHED THE CAMPAIGN: this script backgrounds all six
# wrapper processes with `&` + `wait`, appropriate for a human operator running it directly in a
# terminal. The Claude Code session that authored this campaign instead invoked
# scripts/melitz_production_chain_launch_2026-07-31.sh six times as six SEPARATE
# harness-tracked background tool calls (one per anchor x direction), NOT via this script,
# specifically so each chain's completion is individually tracked/notified rather than only the
# combined `wait` here. Both approaches launch byte-identical underlying Julia processes; this
# script is provided for manual/reproducible standalone use.
set -uo pipefail
REPO2="/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
PRODDIR="$REPO2/docs/key_results/production_delta0p5_2026-07-31"
mkdir -p "$PRODDIR/logs" "$PRODDIR/checkpoints" "$PRODDIR/points" "$PRODDIR/pids"

ANCHORS=("current_calibration" "reduced_q_pre_switch" "reduced_q_post_switch")
DIRECTIONS=("upper" "lower")

PIDS=()
for A in "${ANCHORS[@]}"; do
  for D in "${DIRECTIONS[@]}"; do
    CHAINID="${A}_${D}"
    WRAPPER_LOG="$PRODDIR/logs/${CHAINID}_wrapper.log"
    bash "$REPO2/scripts/melitz_production_chain_launch_2026-07-31.sh" "$A" "$D" > "$WRAPPER_LOG" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    echo "$PID" > "$PRODDIR/pids/${CHAINID}.pid"
    echo "Launched $CHAINID  (wrapper PID $PID, log $WRAPPER_LOG)"
  done
done

echo ""
echo "All 6 chain wrappers launched. PIDs: ${PIDS[*]}"
echo "Waiting for all 6 to complete (up to 6h each; this script blocks until all exit)..."
wait
echo ""
echo "All 6 production chains have exited. Check docs/key_results/production_delta0p5_2026-07-31/checkpoints/*.jls for final status."
