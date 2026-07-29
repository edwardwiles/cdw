#!/usr/bin/env bash
# run_family_chain.sh -- per-cell OS-process-isolated, hard-capped chain runner for ONE
# family x ONE direction (2026-07-28 five-family overnight campaign, maxtime_real=3600s cap).
#
# The 2026-07-28 shakedown (30s cap) ran all 25 cells of a family/direction inside a single
# long-lived Julia process (campaign_cm_family_runner.jl / campaign_unrestricted_runner.jl looping
# over `for delta in DELTAS, st in starts`). That was fine at a ~90s/cell wall time (25 cells ~ 32
# min, trivial to babysit, a crash costs almost nothing). It is not fine at a 3600s/cell cap: a
# single hung callback (KNITRO is known, per CLAUDE.md, to sometimes not respect its own maxtime)
# could otherwise stall cells 2..25 of that family/direction for the rest of a many-hour budget,
# and there would be no way to enforce the task's external "hard process-group cap = 4200s" at
# anything finer than the whole 25-cell chain.
#
# This script keeps the exact same per-cell driver scripts and CLI (campaign_cm_family_runner.jl /
# campaign_unrestricted_runner.jl already accept a single delta/start via the DELTAS_OVERRIDE /
# STARTS_OVERRIDE CLI args) but invokes them ONCE PER CELL under `timeout`, so:
#   - the external 4200s hard cap (section 7) is enforced by the OS, per independent outer solve,
#     not just asserted;
#   - a hung/crashed cell only costs that one OS process -- the next cell starts regardless;
#   - automatic infrastructure retries (section 11, max 2 retries = 3 attempts) are real bash-level
#     retries of a fresh process, not merely a counter.
# No science changes: same manifest, same W/delta/draw_design/draw_seed, same maxtime_real passed
# to KNITRO internally (3600s) -- only the OS-process granularity around each cell changes.
#
# Usage:
#   run_family_chain.sh <family|unrestricted> <upper|lower> <manifest_json> <outroot> \
#       <maxtime_real> <threads> <hard_cap_s> [deltas_csv] [starts_csv]
set -uo pipefail  # NOT -e: a single cell's nonzero exit must not kill the whole chain

FAMILY="$1"; DIRECTION="$2"; MANIFEST="$3"; OUTROOT="$4"
MAXTIME="$5"; THREADS="$6"; HARDCAP="$7"
DELTAS_CSV="${8:-0.01,0.1,0.5,1.0,2.0}"
STARTS_CSV="${9:-1,2,3,4,5}"
MAX_ATTEMPTS=3

D4E="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$D4E/../.." && pwd)"
MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"
mkdir -p "$OUTROOT"; OUTROOT="$(cd "$OUTROOT" && pwd)"
LOGDIR="$OUTROOT/logs/cells"; mkdir -p "$LOGDIR"

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/13.0.1
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/13.0.1/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [ "$FAMILY" = "unrestricted" ]; then
  RUNNER="$D4E/campaign_unrestricted_runner.jl"
  cell_args() { echo "$DIRECTION" "$MANIFEST" "$OUTROOT" "$MAXTIME" "$1" "$2"; }
else
  RUNNER="$D4E/campaign_cm_family_runner.jl"
  cell_args() { echo "$FAMILY" "$DIRECTION" "$MANIFEST" "$OUTROOT" "$MAXTIME" "$1" "$2"; }
fi

IFS=',' read -ra DELTAS <<< "$DELTAS_CSV"
IFS=',' read -ra STARTS <<< "$STARTS_CSV"

echo "=== FAMILY CHAIN START family=$FAMILY direction=$DIRECTION hard_cap_s=$HARDCAP maxtime_real=$MAXTIME threads=$THREADS $(date) ==="

N_DONE=0; N_FAILED=0; N_RUN=0
for delta in "${DELTAS[@]}"; do
  for start in "${STARTS[@]}"; do
    ckdir="$OUTROOT/$FAMILY/$DIRECTION/delta_${delta}/start_${start}"
    if [ -f "$ckdir/DONE" ]; then
      echo "[$FAMILY/$DIRECTION delta=$delta start=$start] SKIP -- already DONE"
      N_DONE=$((N_DONE+1))
      continue
    fi
    attempt=1
    ok=0
    while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
      celllog="$LOGDIR/${FAMILY}_${DIRECTION}_d${delta}_s${start}_attempt${attempt}.log"
      echo "[$FAMILY/$DIRECTION delta=$delta start=$start] attempt $attempt/$MAX_ATTEMPTS (hard_cap=${HARDCAP}s) -> $celllog"
      timeout --kill-after=60s "${HARDCAP}s" \
        julia --project="$ROOT" -t "$THREADS" "$RUNNER" $(cell_args "$delta" "$start") \
        > "$celllog" 2>&1
      rc=$?
      if [ -f "$ckdir/DONE" ]; then
        echo "[$FAMILY/$DIRECTION delta=$delta start=$start] DONE (attempt $attempt, rc=$rc)"
        ok=1
        break
      fi
      echo "[$FAMILY/$DIRECTION delta=$delta start=$start] attempt $attempt FAILED (rc=$rc, timeout_hit=$([ $rc -eq 124 -o $rc -eq 137 ] && echo yes || echo no))"
      attempt=$((attempt+1))
    done
    if [ "$ok" -eq 1 ]; then
      N_RUN=$((N_RUN+1))
    else
      echo "[$FAMILY/$DIRECTION delta=$delta start=$start] EXHAUSTED $MAX_ATTEMPTS attempts -- leaving FAILED, continuing chain"
      mkdir -p "$ckdir"
      echo "exhausted $MAX_ATTEMPTS automatic retries at $(date)" > "$ckdir/FAILED"
      N_FAILED=$((N_FAILED+1))
    fi
  done
done

echo "=== FAMILY CHAIN END family=$FAMILY direction=$DIRECTION $(date) skipped_done=$N_DONE solved=$N_RUN failed=$N_FAILED ==="
[ "$N_FAILED" -eq 0 ]
exit $?
