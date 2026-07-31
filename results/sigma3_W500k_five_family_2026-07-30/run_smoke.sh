#!/usr/bin/env bash
# run_smoke.sh -- generic five-family PARALLEL smoke launcher for the sigma3/W500k campaign
# preflights (items 6, 7, 8, 10 of the required-preflights list). Launches all 5 families
# simultaneously (matching --launch's own process-group design -- this is a preflight of the
# real launch command, not a separate ad hoc path) via run_family_chain_sigma3.sh, at the given
# delta/starts/directions/maxtime, waits for all 5, and writes <report_name>.json read back by
# run_preflights.jl.
#
# Usage: run_smoke.sh <report_name> <deltas_csv> <starts_csv> <directions_csv> <maxtime_s>
#   report_name: e.g. upper_smoke_report -> writes upper_smoke_report.json in this directory
set -uo pipefail

REPORT_NAME="$1"; DELTAS_CSV="$2"; STARTS_CSV="$3"; DIRECTIONS_CSV="$4"; MAXTIME="${5:-60}"

CAMPAIGN_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$CAMPAIGN_ROOT/../.." && pwd)"
DRIVERS_DIR="$REPO_ROOT/campaign_inputs/sigma3_W500k_2026-07-30/drivers"
SMOKE_ROOT="$CAMPAIGN_ROOT/smoke/$REPORT_NAME"
rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT/logs"

FAMILIES=(unrestricted flexible_cm common_frechet origin_zc cm_meanzc)
HARD_CAP=$((MAXTIME + 120))   # generous headroom over the short smoke budget for KNITRO callback return
THREADS_PER_FAMILY=20

echo "=== SMOKE START report=$REPORT_NAME deltas=$DELTAS_CSV starts=$STARTS_CSV directions=$DIRECTIONS_CSV maxtime=${MAXTIME}s $(date) ==="

# run_family_chain_sigma3.sh iterates BOTH directions internally (start-major); to smoke only ONE
# direction (items 6/7), we pass a single-element DIRECTIONS_CSV -- the runner script itself only
# understands "upper,lower" as a fixed pair, so for a single-direction smoke we call the
# per-family JULIA RUNNER directly here instead of going through the chain wrapper (still the
# exact same runner script/args the real launch uses, just without the chain wrapper's own
# both-directions loop).
pids=()
declare -A FAM_OK
for fam in "${FAMILIES[@]}"; do
  logf="$SMOKE_ROOT/logs/${fam}.log"
  (
    export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
    export KNITRODIR=/opt/shared_sw/knitro/13.0.1
    export LD_LIBRARY_PATH=/opt/shared_sw/knitro/13.0.1/lib:${LD_LIBRARY_PATH:-}
    export PATH="$HOME/.juliaup/bin:$PATH"
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    IFS=',' read -ra DIRS <<< "$DIRECTIONS_CSV"
    exit_code=0
    for dir in "${DIRS[@]}"; do
      if [ "$fam" = "unrestricted" ]; then
        RUNNER="$DRIVERS_DIR/campaign_unrestricted_runner_sigma3.jl"
        ARGS=("$dir" "$CAMPAIGN_ROOT/start_manifest.json" "$SMOKE_ROOT" "$MAXTIME" "$DELTAS_CSV" "$STARTS_CSV")
      else
        RUNNER="$DRIVERS_DIR/campaign_cm_family_runner_sigma3.jl"
        ARGS=("$fam" "$dir" "$CAMPAIGN_ROOT/start_manifest.json" "$SMOKE_ROOT" "$MAXTIME" "$DELTAS_CSV" "$STARTS_CSV")
      fi
      timeout --kill-after=30s "${HARD_CAP}s" julia --project="$REPO_ROOT" -t "$THREADS_PER_FAMILY" "$RUNNER" "${ARGS[@]}" >> "$logf" 2>&1
      rc=$?
      [ "$rc" -ne 0 ] && exit_code=$rc
    done
    exit $exit_code
  ) &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

ok=$([ "$fail" -eq 0 ] && echo "true" || echo "false")
{
  echo "{"
  echo "  \"ok\": $ok,"
  echo "  \"summary\": \"deltas=$DELTAS_CSV starts=$STARTS_CSV directions=$DIRECTIONS_CSV maxtime=${MAXTIME}s -- see $SMOKE_ROOT/logs/*.log\","
  echo "  \"generated\": \"$(date -Is)\""
  echo "}"
} > "$CAMPAIGN_ROOT/${REPORT_NAME}.json"

echo "=== SMOKE END report=$REPORT_NAME ok=$ok $(date) ==="
[ "$fail" -eq 0 ]
exit $?
