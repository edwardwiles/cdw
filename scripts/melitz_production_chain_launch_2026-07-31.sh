#!/bin/bash
# Crash-aware launch/resume wrapper for ONE anchor x direction production chain
# (docs/melitz_d20_profiledA_production_delta0p5_2026-07-31.md).
#
# Usage: melitz_production_chain_launch_2026-07-31.sh <anchor_label> <direction>
#   anchor_label in (current_calibration, reduced_q_pre_switch, reduced_q_post_switch)
#   direction    in (upper, lower)
#
# Runs scripts/melitz_production_chain_2026-07-31.jl (5 Julia threads, BLAS=1, one
# independent KNITRO session). The Julia driver checkpoints after every completed welfare
# point and is itself resumable (reads its own checkpoint on restart). This wrapper's own job
# is ONLY to handle a process-level crash (e.g. the pre-existing, disclosed mul_G! SIGSEGV):
#
#   - clean exit (code 0)      -> chain finished on its own (converged/budget_exhausted/
#                                  max_points/stalled/max_wall); done, no restart.
#   - crash (non-zero exit)    -> compare the welfare-point target that was PENDING at crash
#                                  time (scripts/*.pending marker, written immediately before
#                                  each risky evaluation, cleared immediately after) against the
#                                  target that caused the PREVIOUS crash of this same chain:
#                                    * same target twice in a row -> STOP this chain permanently,
#                                      preserve a complete crash artifact (log + checkpoint), do
#                                      NOT retry indefinitely.
#                                    * different (or first) crash -> treat as transient, resume
#                                      (the Julia driver reads its own last-good checkpoint).
set -uo pipefail

ANCHOR="${1:?usage: melitz_production_chain_launch_2026-07-31.sh <anchor_label> <direction>}"
DIRECTION="${2:?usage: melitz_production_chain_launch_2026-07-31.sh <anchor_label> <direction>}"
REPO2="/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
CHAINID="${ANCHOR}_${DIRECTION}"
PRODDIR="$REPO2/docs/key_results/production_delta0p5_2026-07-31"
LOGDIR="$PRODDIR/logs"
CKPTDIR="$PRODDIR/checkpoints"
CRASHDIR="$PRODDIR/crash_artifacts/${CHAINID}"
mkdir -p "$LOGDIR" "$CKPTDIR"

CKPT="$CKPTDIR/${CHAINID}.jls"
PENDING="$CKPTDIR/${CHAINID}.pending"
CRASHFLAG="$CKPTDIR/${CHAINID}.last_crash_target"

export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

MAX_RESTARTS=8
attempt=0
while [ "$attempt" -lt "$MAX_RESTARTS" ]; do
  attempt=$((attempt + 1))
  TS=$(date +%Y%m%d_%H%M%S)
  LOGFILE="$LOGDIR/${CHAINID}_attempt${attempt}_${TS}.log"
  echo "[$CHAINID] launch attempt $attempt/$MAX_RESTARTS (PID will be logged below) -> $LOGFILE"

  cd "$REPO2" || exit 1
  julia --project=. -t 5 scripts/melitz_production_chain_2026-07-31.jl "$ANCHOR" "$DIRECTION" > "$LOGFILE" 2>&1 &
  JPID=$!
  echo "[$CHAINID] julia PID=$JPID  anchor=$ANCHOR  direction=$DIRECTION  threads=5  BLAS=1  attempt=$attempt"
  wait "$JPID"
  EXITCODE=$?
  echo "[$CHAINID] attempt $attempt (PID $JPID) exited with code $EXITCODE"

  if [ "$EXITCODE" -eq 0 ]; then
    echo "[$CHAINID] clean exit -- chain finished on its own. Done."
    exit 0
  fi

  if [ -f "$PENDING" ]; then
    THIS_TARGET=$(cat "$PENDING")
  else
    THIS_TARGET="none_pending"
  fi
  echo "[$CHAINID] crash detected (exit=$EXITCODE). pending target at crash time: $THIS_TARGET"

  PREV_TARGET=""
  [ -f "$CRASHFLAG" ] && PREV_TARGET=$(cat "$CRASHFLAG")

  if [ "$THIS_TARGET" != "none_pending" ] && [ "$THIS_TARGET" == "$PREV_TARGET" ]; then
    echo "[$CHAINID] SAME welfare-point target crashed TWICE in a row ($THIS_TARGET) -- stopping this chain permanently, preserving a complete crash artifact, NOT retrying indefinitely."
    mkdir -p "$CRASHDIR"
    cp -f "$LOGFILE" "$CRASHDIR/" 2>/dev/null
    [ -f "$CKPT" ] && cp -f "$CKPT" "$CRASHDIR/" 2>/dev/null
    {
      echo "chain=$CHAINID"
      echo "repeated_crash_target=$THIS_TARGET"
      echo "final_exit_code=$EXITCODE"
      echo "attempts=$attempt"
      date
    } > "$CRASHDIR/CRASH_SUMMARY.txt"
    exit 1
  fi

  echo "$THIS_TARGET" > "$CRASHFLAG"
  echo "[$CHAINID] different (or first) crash target -- treating as transient host/process failure, resuming from last checkpoint next attempt."
done

echo "[$CHAINID] exhausted $MAX_RESTARTS restart attempts without a clean exit -- stopping, preserving final state as a crash artifact (no unbounded retries)."
mkdir -p "$CRASHDIR"
cp -f "$LOGDIR"/"${CHAINID}"_attempt*.log "$CRASHDIR/" 2>/dev/null
[ -f "$CKPT" ] && cp -f "$CKPT" "$CRASHDIR/" 2>/dev/null
echo "chain=$CHAINID exhausted_restarts=$MAX_RESTARTS" > "$CRASHDIR/CRASH_SUMMARY.txt"
date >> "$CRASHDIR/CRASH_SUMMARY.txt"
exit 1
