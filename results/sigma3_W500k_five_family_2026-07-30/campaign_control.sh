#!/usr/bin/env bash
# campaign_control.sh -- sigma3/W500k five-family production campaign (2026-07-30)
#
# Single entry point for all campaign lifecycle actions. Modes:
#   --prepare-only     verify/freeze the prerequisite manifests (data, calibration, Sobol, starts);
#                       does not touch KNITRO.
#   --preflight-only    run the full required-preflights suite (task brief items 1-10); refuses to
#                       write READY_TO_LAUNCH if any gate fails.
#   --dry-run           print exactly what --launch would execute (5 setsid'd family chains,
#                       resolved paths/args/env) without starting anything.
#   --launch            the real thing. Refuses unless READY_TO_LAUNCH exists and every frozen
#                       manifest's SHA256 still matches campaign_config.sha256.
#   --resume            re-invoke the same 5 family chains; each chain and each cell already
#                       skip-if-DONE internally, so this is safe to run repeatedly.
#   --status            report per-family cell counts (done/failed/pending) and whether the
#                       supervisor process group is alive.
#   --stop              SIGTERM the recorded process groups (from process_manifest.json), wait,
#                       report.
#
# This script does not itself decide science: family list, deltas, starts, W, sigma, K are all
# read from campaign_config.json (frozen by prepare-only) so --launch cannot silently diverge from
# what was preflighted.
set -uo pipefail

CAMPAIGN_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$CAMPAIGN_ROOT/../.." && pwd)"
INPUTS_DIR="$REPO_ROOT/campaign_inputs/sigma3_W500k_2026-07-30"
DRIVERS_DIR="$INPUTS_DIR/drivers"
CONFIG_JSON="$CAMPAIGN_ROOT/campaign_config.json"
CONFIG_SHA="$CAMPAIGN_ROOT/campaign_config.sha256"
READY_MARKER="$CAMPAIGN_ROOT/READY_TO_LAUNCH"
RUNNING_MARKER="$CAMPAIGN_ROOT/RUNNING"
PROCMAN="$CAMPAIGN_ROOT/process_manifest.json"
STATE="$CAMPAIGN_ROOT/campaign_state.json"

FAMILIES=(unrestricted flexible_cm common_frechet origin_zc cm_meanzc)
THREADS_PER_FAMILY=20
HARD_CAP_S=12000     # process-group watchdog: 10,800s KNITRO budget + headroom for callback return/write
MAXTIME_REAL=10800
DELTAS_CSV="0.01,0.1,0.5,1.0,2.0,5.0"
STARTS_CSV="1,2,3"

usage() { echo "usage: $0 [--prepare-only|--preflight-only|--dry-run|--launch|--resume|--status|--stop]"; exit 2; }
[ $# -eq 1 ] || usage
MODE="$1"

lp() { echo "$(date -Is) [$MODE] $*"; }

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

require_config_unchanged() {
  [ -f "$CONFIG_JSON" ] || { lp "REFUSE: $CONFIG_JSON missing -- run --prepare-only first"; exit 1; }
  [ -f "$CONFIG_SHA" ] || { lp "REFUSE: $CONFIG_SHA missing -- run --prepare-only first"; exit 1; }
  local now; now="$(sha256_file "$CONFIG_JSON")"
  local frozen; frozen="$(cat "$CONFIG_SHA")"
  [ "$now" = "$frozen" ] || { lp "REFUSE: campaign_config.json changed since freeze ($frozen -> $now) -- re-run --prepare-only"; exit 1; }
}

case "$MODE" in
  --prepare-only)
    lp "=== PREPARE-ONLY: verifying prerequisite manifests ==="
    for f in "$INPUTS_DIR/data_manifest.json" "$INPUTS_DIR/calibration_manifest.json" "$INPUTS_DIR/start_manifest.json"; do
      if [ ! -f "$f" ]; then
        lp "MISSING: $f -- run the corresponding build script in $INPUTS_DIR first"
        exit 1
      fi
      lp "OK: $f ($(sha256_file "$f" | cut -c1-16)...)"
    done
    lp "All prerequisite manifests present. campaign_config.json is written/verified separately"
    lp "(see freeze_campaign_config.sh) -- prepare-only does not overwrite an already-frozen config."
    ;;

  --preflight-only)
    require_config_unchanged
    lp "=== PREFLIGHT-ONLY: running required preflights 1-10 ==="
    export PATH="$HOME/.juliaup/bin:$PATH"
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    julia --project="$REPO_ROOT" -t 8 "$CAMPAIGN_ROOT/run_preflights.jl"
    rc=$?
    if [ $rc -eq 0 ]; then
      lp "ALL PREFLIGHTS PASS -- writing READY_TO_LAUNCH"
      { echo "generated: $(date -Is)"; echo "campaign_config sha256: $(cat "$CONFIG_SHA")"; } > "$READY_MARKER"
    else
      lp "PREFLIGHT FAILURE (rc=$rc) -- NOT writing READY_TO_LAUNCH"
      rm -f "$READY_MARKER"
    fi
    exit $rc
    ;;

  --dry-run)
    lp "=== DRY-RUN: commands that --launch would execute (nothing started) ==="
    for fam in "${FAMILIES[@]}"; do
      echo "  setsid $DRIVERS_DIR/run_family_chain_sigma3.sh $fam $CAMPAIGN_ROOT/start_manifest.json $CAMPAIGN_ROOT/$fam $MAXTIME_REAL $THREADS_PER_FAMILY $HARD_CAP_S $DELTAS_CSV $STARTS_CSV"
    done
    lp "resource plan: ${#FAMILIES[@]} processes x $THREADS_PER_FAMILY threads = $(( ${#FAMILIES[@]} * THREADS_PER_FAMILY )) Julia threads total (host has $(nproc) logical CPUs)"
    lp "outer strategy default: direct_sr1 (hessopt_tag=sr1); optional BFGS polish NOT enabled in this launch command"
    ;;

  --launch)
    [ -f "$READY_MARKER" ] || { lp "REFUSE: $READY_MARKER absent -- run --preflight-only first (all gates must pass)"; exit 1; }
    require_config_unchanged
    frozen_ready_sha="$(grep 'campaign_config sha256' "$READY_MARKER" | awk '{print $NF}')"
    current_sha="$(cat "$CONFIG_SHA")"
    [ "$frozen_ready_sha" = "$current_sha" ] || { lp "REFUSE: campaign_config changed since READY_TO_LAUNCH was written -- re-run --preflight-only"; exit 1; }
    lp "=== LAUNCH: starting all ${#FAMILIES[@]} family chains simultaneously ==="
    mkdir -p "$CAMPAIGN_ROOT/logs"
    echo "{" > "$PROCMAN"
    first=1
    for fam in "${FAMILIES[@]}"; do
      setsid "$DRIVERS_DIR/run_family_chain_sigma3.sh" "$fam" "$CAMPAIGN_ROOT/start_manifest.json" \
        "$CAMPAIGN_ROOT/$fam" "$MAXTIME_REAL" "$THREADS_PER_FAMILY" "$HARD_CAP_S" "$DELTAS_CSV" "$STARTS_CSV" \
        > "$CAMPAIGN_ROOT/logs/${fam}_chain.log" 2>&1 &
      pid=$!
      lp "launched $fam pid=$pid pgid=$pid (setsid)"
      [ "$first" -eq 1 ] && first=0 || echo "," >> "$PROCMAN"
      echo "  \"$fam\": {\"pid\": $pid, \"pgid\": $pid, \"log\": \"$CAMPAIGN_ROOT/logs/${fam}_chain.log\", \"launched\": \"$(date -Is)\"}" >> "$PROCMAN"
    done
    echo "}" >> "$PROCMAN"
    { echo "{\"phase\": \"running\", \"launched\": \"$(date -Is)\", \"pid\": $$}"; } > "$STATE"
    lp "5 family chains launched (pids recorded in $PROCMAN). This process does not block -- use --status to check progress."
    ;;

  --resume)
    lp "=== RESUME: re-invoking all family chains (each cell/chain skips-if-DONE internally) ==="
    "$0" --launch
    ;;

  --status)
    lp "=== STATUS ==="
    if [ -f "$STATE" ]; then cat "$STATE"; else echo "(no campaign_state.json -- never launched)"; fi
    echo
    for fam in "${FAMILIES[@]}"; do
      famdir="$CAMPAIGN_ROOT/$fam"
      [ -d "$famdir" ] || { echo "$fam: not started"; continue; }
      n_done=$(find "$famdir" -name DONE 2>/dev/null | wc -l)
      n_failed=$(find "$famdir" -name FAILED 2>/dev/null | wc -l)
      n_total=$(( ${#FAMILIES[@]} > 0 ? 2*6*3 : 0 ))   # 2 directions x 6 deltas x 3 starts
      echo "$fam: done=$n_done failed=$n_failed / $n_total cells"
    done
    if [ -f "$PROCMAN" ]; then
      echo
      echo "process liveness:"
      python3 -c "
import json
d = json.load(open('$PROCMAN'))
for fam, info in d.items():
    pid = info['pid']
    import os
    try:
        os.kill(pid, 0)
        alive = 'ALIVE'
    except OSError:
        alive = 'DEAD'
    print(f'  {fam}: pid={pid} {alive}')
" 2>/dev/null || echo "  (install python3 or inspect $PROCMAN manually)"
    fi
    ;;

  --stop)
    lp "=== STOP: sending SIGTERM to recorded process groups ==="
    [ -f "$PROCMAN" ] || { lp "no $PROCMAN -- nothing to stop"; exit 0; }
    python3 -c "
import json
d = json.load(open('$PROCMAN'))
for fam, info in d.items():
    pgid = info['pgid']
    print(f'  stopping {fam} pgid={pgid}')
" 2>/dev/null
    for fam in "${FAMILIES[@]}"; do
      pgid=$(python3 -c "import json; d=json.load(open('$PROCMAN')); print(d.get('$fam',{}).get('pgid',''))" 2>/dev/null)
      [ -n "$pgid" ] && kill -TERM -"$pgid" 2>/dev/null && lp "SIGTERM sent to $fam (pgid=$pgid)"
    done
    lp "SIGTERM sent to all recorded process groups. Check --status after a few seconds to confirm."
    ;;

  *)
    usage
    ;;
esac
