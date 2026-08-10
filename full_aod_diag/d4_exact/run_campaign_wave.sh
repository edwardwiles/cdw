#!/usr/bin/env bash
# run_campaign_wave.sh -- launch all 5 family processes IN PARALLEL for one direction (2026-07-28
# five-family shakedown campaign). Each family gets its own OS process, its own log file, and
# threads_per_family Julia threads. No state is shared between the 5 processes beyond the
# read-only manifest file and the read-only production codebase.
#
# Usage:
#   ./run_campaign_wave.sh <upper|lower> <manifest_json> <outroot> <maxtime_real> <threads_per_family> [deltas_csv] [starts_csv]
set -euo pipefail

DIRECTION="$1"
MANIFEST="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
OUTROOT="$3"
mkdir -p "$OUTROOT"
OUTROOT="$(cd "$OUTROOT" && pwd)"
MAXTIME="$4"
THREADS="$5"
DELTAS_CSV="${6:-}"
STARTS_CSV="${7:-}"

# Resolve the script's own directory AND the project root (worktree root, one level up) to
# absolute paths WITHOUT cd-ing this orchestrator process itself -- an earlier version of this
# script did `cd "$(dirname "$0")"` before resolving MANIFEST/OUTROOT, which silently broke both
# (1) relative MANIFEST/OUTROOT paths (resolved against the wrong directory) and (2) `--project=.`
# (pointed at full_aod_diag/d4_exact instead of the worktree root, so Project.toml/Manifest.toml
# -- and every package they pin, e.g. SpecialFunctions -- were not found). Caught by the minimal
# smoke: all 5 families crashed in <10s with ArgumentError: Package SpecialFunctions not found.
D4E="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$D4E/../.." && pwd)"

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/13.0.1
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/13.0.1/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p "$OUTROOT"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"

echo "=== WAVE START direction=$DIRECTION maxtime=$MAXTIME threads_per_family=$THREADS deltas=[${DELTAS_CSV:-default}] starts=[${STARTS_CSV:-default}] $(date) ==="
echo "continuation_enabled = false"
echo "prior_delta_state_loaded = false"
echo "prior_direction_state_loaded = false"
echo "cross_start_state_loaded = false"

PIDS=()
NAMES=()

launch() {
  local name="$1"; shift
  local logf="$LOGDIR/${name}_${DIRECTION}.log"
  echo "launching $name -> $logf"
  ("$@") > "$logf" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

# 2026-08-09 (CROSS campaign integration): the family list is now a variable, so a wave can run the
# K_pair^2 cross-power ZC families INSTEAD of their diagonal counterparts without editing this file.
# Default is byte-identical to the historical hardcoded five, so an operator who sets nothing gets
# exactly the previous behavior.
#   CAMPAIGN_FAMILIES="flexible_cm common_frechet cm_meanzc_cross origin_zc_cross unrestricted"
# `unrestricted` is the one family with its own separate runner script (self-contained include list,
# matching smoke_delta1_unrestricted.jl's convention); every other name goes to the CM family runner,
# which validates it against its own whitelist and errors on an unknown one.
CAMPAIGN_FAMILIES="${CAMPAIGN_FAMILIES:-flexible_cm common_frechet cm_meanzc origin_zc unrestricted}"
echo "campaign_families = [$CAMPAIGN_FAMILIES]"
# shellcheck disable=SC2086  -- unquoted on purpose: word splitting IS the mechanism here
for fam in $CAMPAIGN_FAMILIES; do
  if [ "$fam" = "unrestricted" ]; then
    launch "$fam" julia --project="$ROOT" -t "$THREADS" "$D4E/campaign_unrestricted_runner.jl" "$DIRECTION" "$MANIFEST" "$OUTROOT" "$MAXTIME" "$DELTAS_CSV" "$STARTS_CSV"
  else
    launch "$fam" julia --project="$ROOT" -t "$THREADS" "$D4E/campaign_cm_family_runner.jl" "$fam" "$DIRECTION" "$MANIFEST" "$OUTROOT" "$MAXTIME" "$DELTAS_CSV" "$STARTS_CSV"
  fi
done

echo "launched PIDs: ${PIDS[*]}"

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[${NAMES[$i]}] exited 0"
  else
    rc=$?
    echo "[${NAMES[$i]}] EXITED NONZERO ($rc)"
    FAIL=1
  fi
done

echo "=== WAVE END direction=$DIRECTION $(date) FAIL=$FAIL ==="
exit $FAIL
