#!/usr/bin/env bash
# run_family_chain_sigma3.sh -- per-cell OS-process-isolated, hard-capped chain runner for ONE
# family, BOTH directions, sigma3/W500k five-family production campaign (2026-07-30).
#
# Adapted from full_aod_diag/d4_exact/run_family_chain.sh (2026-07-28), which takes a single
# direction per invocation and is looped over both directions by two separate supervisor "waves"
# (all 5 families run upper, then all 5 run lower). This campaign's brief instead specifies
# START-MAJOR sequencing WITHIN a family: "Start 1: upper deltas...; lower deltas...; Start 2:
# ...; Start 3: ..." -- i.e. each start completes both directions before the next start begins,
# not "every start's upper, then every start's lower". Since every cell is fully independent
# (fresh context, no continuation, checkpointed/resumable -- see campaign_cm_family_runner_sigma3.jl/
# campaign_unrestricted_runner_sigma3.jl headers), this ordering does not change any result, only
# the sequence in which cells complete -- but it is implemented here to match the brief exactly
# rather than silently substituting an equivalent-but-different schedule.
#
# Usage:
#   run_family_chain_sigma3.sh <family|unrestricted> <manifest_json> <outroot> \
#       <maxtime_real> <threads> <hard_cap_s> [deltas_csv] [starts_csv]
set -uo pipefail  # NOT -e: a single cell's nonzero exit must not kill the whole chain

FAMILY="$1"; MANIFEST="$2"; OUTROOT="$3"
MAXTIME="$4"; THREADS="$5"; HARDCAP="$6"
DELTAS_CSV="${7:-0.01,0.1,0.5,1.0,2.0,5.0}"
STARTS_CSV="${8:-1,2,3}"
MAX_ATTEMPTS=3

DRIVERS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DRIVERS_DIR/../../.." && pwd)"
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
  RUNNER="$DRIVERS_DIR/campaign_unrestricted_runner_sigma3.jl"
  cell_args() { echo "$1" "$MANIFEST" "$OUTROOT" "$MAXTIME" "$2" "$3"; }   # direction delta start
else
  RUNNER="$DRIVERS_DIR/campaign_cm_family_runner_sigma3.jl"
  cell_args() { echo "$FAMILY" "$1" "$MANIFEST" "$OUTROOT" "$MAXTIME" "$2" "$3"; }
fi

IFS=',' read -ra DELTAS <<< "$DELTAS_CSV"
IFS=',' read -ra STARTS <<< "$STARTS_CSV"
DIRECTIONS=(upper lower)

echo "=== FAMILY CHAIN START family=$FAMILY hard_cap_s=$HARDCAP maxtime_real=$MAXTIME threads=$THREADS start_major $(date) ==="

N_DONE=0; N_FAILED=0; N_RUN=0
for start in "${STARTS[@]}"; do
  for direction in "${DIRECTIONS[@]}"; do
    for delta in "${DELTAS[@]}"; do
      ckdir="$OUTROOT/$FAMILY/$direction/delta_${delta}/start_${start}"
      if [ -f "$ckdir/DONE" ]; then
        echo "[$FAMILY/$direction delta=$delta start=$start] SKIP -- already DONE"
        N_DONE=$((N_DONE+1))
        continue
      fi
      attempt=1
      ok=0
      while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
        celllog="$LOGDIR/${FAMILY}_${direction}_d${delta}_s${start}_attempt${attempt}.log"
        echo "[$FAMILY/$direction delta=$delta start=$start] attempt $attempt/$MAX_ATTEMPTS (hard_cap=${HARDCAP}s) -> $celllog"
        # /usr/bin/time (GNU time, standalone binary) is NOT installed on this host -- confirmed
        # live 2026-07-30 (rc=127 "No such file or directory" on every single cell, which would
        # have broken all 180 real launch cells identically since this script IS what --launch
        # itself uses). Replaced with a background /proc/<pid>/status poller for peak RSS
        # (VmHWM), written in the same "Maximum resident set size (kbytes): N" grep-compatible
        # format the rollup script (rollup_summaries.jl) and run_preflight_smokes_6to10.sh's own
        # `grep "Maximum resident set size"` already expect -- no downstream format change needed.
        t_start=$(date +%s.%N)
        timeout --kill-after=60s "${HARDCAP}s" \
          julia --project="$ROOT" -t "$THREADS" "$RUNNER" $(cell_args "$direction" "$delta" "$start") \
          > "$celllog" 2>&1 &
        jpid=$!
        peak_kb=0
        while kill -0 "$jpid" 2>/dev/null; do
          for cpid in $(pgrep -P "$jpid" 2>/dev/null) "$jpid"; do
            [ -r "/proc/$cpid/status" ] || continue
            kb=$(awk '/^VmHWM:/{print $2}' "/proc/$cpid/status" 2>/dev/null)
            [ -n "$kb" ] && [ "$kb" -gt "$peak_kb" ] 2>/dev/null && peak_kb=$kb
          done
          sleep 2
        done
        wait "$jpid"
        rc=$?
        t_end=$(date +%s.%N)
        {
          echo "	Maximum resident set size (kbytes): $peak_kb"
          echo "	Elapsed (wall clock) time (h:mm:ss or m:ss): $(awk -v s="$t_start" -v e="$t_end" 'BEGIN{printf "%.2f sec", e-s}')"
        } > "$celllog.rusage"
        if [ -f "$ckdir/DONE" ]; then
          echo "[$FAMILY/$direction delta=$delta start=$start] DONE (attempt $attempt, rc=$rc)"
          grep -m1 "Knitro using the" "$celllog" > "$ckdir/knitro_algorithm.txt" 2>/dev/null || true
          # RESOURCE_USAGE.csv raw material: the peak-RSS poller's "Maximum resident set size"
          # line above, kept per-cell for the rollup script to parse.
          ok=1
          break
        fi
        echo "[$FAMILY/$direction delta=$delta start=$start] attempt $attempt FAILED (rc=$rc, timeout_hit=$([ $rc -eq 124 -o $rc -eq 137 ] && echo yes || echo no))"
        attempt=$((attempt+1))
      done
      if [ "$ok" -eq 1 ]; then
        N_RUN=$((N_RUN+1))
      else
        echo "[$FAMILY/$direction delta=$delta start=$start] EXHAUSTED $MAX_ATTEMPTS attempts -- leaving FAILED, continuing chain"
        mkdir -p "$ckdir"
        echo "exhausted $MAX_ATTEMPTS automatic retries at $(date)" > "$ckdir/FAILED"
        N_FAILED=$((N_FAILED+1))
      fi
    done
  done
done

echo "=== FAMILY CHAIN END family=$FAMILY $(date) skipped_done=$N_DONE solved=$N_RUN failed=$N_FAILED ==="
[ "$N_FAILED" -eq 0 ]
exit $?
