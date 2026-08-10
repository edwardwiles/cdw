#!/usr/bin/env bash
# run_full_campaign_supervisor.sh -- top-level detached supervisor for the five-family overnight
# campaign (2026-07-28): launches all 5 family chains in parallel for the upper wave, waits for
# all 5 to reach a durable terminal state (all cells DONE or FAILED-after-retries), then does the
# same for the lower wave. Each family chain is run_family_chain.sh, which is itself a per-cell
# isolated/hard-capped/resumable sequential queue over that family's 25 (delta,start) cells (see
# that script's own header for why cell-level process isolation is needed once cells can run up to
# 3600s -- not needed at the 30s shakedown scale, needed now).
#
# Usage:
#   run_full_campaign_supervisor.sh <manifest_json> <outroot> <maxtime_real> <threads_per_family> <hard_cap_s>
set -uo pipefail

MANIFEST="$1"; OUTROOT="$2"; MAXTIME="$3"; THREADS="$4"; HARDCAP="$5"
D4E="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUTROOT"; OUTROOT="$(cd "$OUTROOT" && pwd)"
STATE="$OUTROOT/campaign_state.json"
PROCMAN="$OUTROOT/process_manifest.json"
SUPLOG="$OUTROOT/supervisor.log"

# 2026-08-09 (CROSS campaign integration): env-overridable, so a campaign can run the K_pair^2
# cross-power ZC families INSTEAD of their diagonal counterparts without editing this file. Default
# is byte-identical to the historical hardcoded five. Set e.g.
#   CAMPAIGN_FAMILIES="flexible_cm common_frechet cm_meanzc_cross origin_zc_cross unrestricted"
# The list is validated here (fail fast, before any chain is launched) rather than only inside the
# Julia runner -- a typo'd family name would otherwise start N-1 real chains and only kill the Nth,
# leaving a half-run campaign to clean up. Note run_campaign_wave.sh reads the SAME variable, and
# this supervisor exports it below so the two can never disagree.
# shellcheck disable=SC2206
FAMILIES=(${CAMPAIGN_FAMILIES:-flexible_cm common_frechet cm_meanzc origin_zc unrestricted})
VALID_FAMILIES=(flexible_cm common_frechet cm_meanzc origin_zc cm_meanzc_cross origin_zc_cross unrestricted)
for fam in "${FAMILIES[@]}"; do
  ok=0
  for v in "${VALID_FAMILIES[@]}"; do [ "$fam" = "$v" ] && ok=1 && break; done
  if [ "$ok" -ne 1 ]; then
    echo "run_full_campaign_supervisor.sh: unknown family '$fam' in CAMPAIGN_FAMILIES; valid: ${VALID_FAMILIES[*]}" >&2
    exit 2
  fi
done
export CAMPAIGN_FAMILIES="${FAMILIES[*]}"

lp() { echo "$(date -Is) $*" | tee -a "$SUPLOG"; }

write_state() {
  # minimal hand-rolled JSON (no jq dependency assumed) -- consistent with json_lite.jl's own scope
  local phase="$1"
  {
    echo "{"
    echo "  \"phase\": \"$phase\","
    echo "  \"updated\": \"$(date -Is)\","
    echo "  \"supervisor_pid\": $$,"
    echo "  \"outroot\": \"$OUTROOT\","
    echo "  \"maxtime_real\": $MAXTIME,"
    echo "  \"hard_cap_s\": $HARDCAP,"
    echo "  \"threads_per_family\": $THREADS"
    echo "}"
  } > "$STATE"
}

write_state "starting"
lp "=== CAMPAIGN SUPERVISOR START pid=$$ outroot=$OUTROOT maxtime_real=$MAXTIME threads_per_family=$THREADS hard_cap_s=$HARDCAP ==="

echo "{" > "$PROCMAN"
first=1
run_wave() {
  local direction="$1"
  lp "--- WAVE $direction: launching ${#FAMILIES[@]} family chains in parallel ---"
  declare -A PIDS
  for fam in "${FAMILIES[@]}"; do
    setsid "$D4E/run_family_chain.sh" "$fam" "$direction" "$MANIFEST" "$OUTROOT" "$MAXTIME" "$THREADS" "$HARDCAP" \
      > "$OUTROOT/logs/${fam}_${direction}_chain.log" 2>&1 &
    pid=$!
    PIDS[$fam]=$pid
    lp "launched $fam/$direction pid=$pid pgid=$pid (setsid)"
    if [ "$first" -eq 1 ]; then first=0; else echo "," >> "$PROCMAN"; fi
    echo "  \"${fam}_${direction}\": {\"pid\": $pid, \"pgid\": $pid, \"log\": \"$OUTROOT/logs/${fam}_${direction}_chain.log\"}" >> "$PROCMAN"
  done
  write_state "wave_${direction}_running"
  local fail=0
  for fam in "${FAMILIES[@]}"; do
    if wait "${PIDS[$fam]}"; then
      lp "[$fam/$direction] chain exited 0 (all cells DONE, no unretried failures)"
    else
      rc=$?
      lp "[$fam/$direction] chain exited $rc (one or more cells FAILED after max automatic retries -- see per-cell FAILED markers)"
      fail=1
    fi
  done
  write_state "wave_${direction}_complete"
  return $fail
}

run_wave upper
UPPER_FAIL=$?
run_wave lower
LOWER_FAIL=$?

echo "}" >> "$PROCMAN"
write_state "complete"
lp "=== CAMPAIGN SUPERVISOR END upper_fail=$UPPER_FAIL lower_fail=$LOWER_FAIL $(date) ==="
exit $((UPPER_FAIL + LOWER_FAIL))
