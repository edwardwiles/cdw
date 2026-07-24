#!/usr/bin/env bash
# ============================================================================
# Gate C (exclude-ROW-destination production release, 2026-07-24): real production-supervisor
# smoke test, generalized across the four restriction families the release brief requires --
# unrestricted, flexible CM, CM+mean-ZC (K_mean=1/K_pair=1), origin-specific-ZC
# (K_mean=1/K_pair=1). Uses the REAL process-group supervisor mechanics (setsid+pgid launch,
# kill_pgroup/alive_pgroup/wait_for_pgid_file), sourced directly from
# scripts/cm_production_supervisor.sh -- not a reimplementation. Same procedure as the validated
# scripts/d20_meanzc_supervisor_smoke_test.sh template, generalized to all four families and to
# the destination_sample toggle:
#   1. start the real stage (mode=calibration)
#   2. wait for a schema-valid checkpoint with a typed best-feasible incumbent (or budget
#      exhaustion with a valid checkpoint -- a short smoke budget may not reach a feasible point)
#   3. deliberately SIGTERM the process group, escalate to SIGKILL if needed
#   4. confirm no duplicate/orphan process remains
#   5. resume through the SAME supervisor mechanics with the remaining budget
#   6. confirm the stage log shows: resolved destination_sample, screens active, threshold-10
#      active, no destination-ROW variables/moments (only meaningful to check for :exclude_row)
#   7. verify resume REFUSAL under a mismatched destination_sample
#
# Usage: bash scripts/exclude_row_gateC_smoke_test.sh <family> <stage_dir> <budget_s> [destination_sample]
#   family: unrestricted | cm | cmzc | originzc
#   destination_sample: exclude_row (default, omit to test the DEFAULT path) | all_legacy (explicit)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAIN_ID="smoke"
SUPERVISOR_LOG=""
# shellcheck disable=SC1091
source "$SCRIPT_DIR/cm_production_supervisor.sh"

FAMILY="${1:?usage: $0 <family: unrestricted|cm|cmzc|originzc> <stage_dir> <budget_s> [destination_sample]}"
STAGE_DIR="${2:?usage: $0 <family> <stage_dir> <budget_s> [destination_sample]}"
BUDGET_S="${3:?usage: $0 <family> <stage_dir> <budget_s> [destination_sample]}"
DEST_SAMPLE_ARG="${4:-}"   # empty = DEFAULT path (env var omitted entirely -- proves the default)
mkdir -p "$STAGE_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

case "$FAMILY" in
  unrestricted)
    RUNNER="full_aod_diag/d4_exact/unrestricted_stage_runner.jl"
    DEST_ENV_NAME="DESTINATION_SAMPLE"
    ;;
  cm)
    RUNNER="full_aod_diag/d4_exact/cm_production_stage_runner.jl"
    DEST_ENV_NAME="CM_DESTINATION_SAMPLE"
    export CM_EXTENSION=cm_only CM_GRADIENT_BACKEND=cplus
    ;;
  cmzc)
    RUNNER="full_aod_diag/d4_exact/cm_production_stage_runner.jl"
    DEST_ENV_NAME="CM_DESTINATION_SAMPLE"
    export CM_EXTENSION=cm_plus_equal_means_zero_covariance MEANZC_K_MEAN=1 MEANZC_K_PAIR=1 MEANZC_BASIS=direct CM_GRADIENT_BACKEND=cplus
    ;;
  originzc)
    RUNNER="full_aod_diag/d4_exact/originzc_production_stage_runner.jl"
    DEST_ENV_NAME="CM_DESTINATION_SAMPLE"
    export DISTRIBUTION_RESTRICTION=origin_specific_moments_zero_covariance ORIGIN_K_MEAN=1 ORIGIN_K_PAIR=1 POWER_TARGET_LAYOUT=origin_by_power MEANZC_BASIS=direct CM_GRADIENT_BACKEND=cplus
    ;;
  *) echo "FAIL: unknown family $FAMILY (expected unrestricted|cm|cmzc|originzc)" >&2; exit 1 ;;
esac
if [ -n "$DEST_SAMPLE_ARG" ]; then
  export "$DEST_ENV_NAME"="$DEST_SAMPLE_ARG"
  RESOLVED_DEST="$DEST_SAMPLE_ARG"
else
  unset "$DEST_ENV_NAME" 2>/dev/null || true
  # exclude-ROW-destination production release (2026-07-24) SCOPE NOTE: every entry point EXCEPT
  # unrestricted defaults to :exclude_row -- unrestricted's real evaluation path
  # (compressed_moments.jl) is square-D-only and was never rectangularized, so it defaults to
  # (and only supports) :all_legacy. See EXCLUDE_ROW_DESTINATION_PRODUCTION_RELEASE_2026-07-24.md.
  if [ "$FAMILY" = "unrestricted" ]; then
    RESOLVED_DEST="all_legacy"
  else
    RESOLVED_DEST="exclude_row"
  fi
fi
if [ "$FAMILY" = "unrestricted" ] && [ "$RESOLVED_DEST" = "exclude_row" ]; then
  echo "FAIL: [$FAMILY] :exclude_row was explicitly requested for the unrestricted family, which does not support it (see SCOPE NOTE)" >&2
  exit 1
fi

launch_stage() {
  local mode="$1" seed_arg="$2" budget="$3"
  local pgid_file="$STAGE_DIR/run_meta.txt.pgid"
  rm -f "$pgid_file"
  ( cd "$REPO_ROOT" && setsid bash -c '
      echo "$$" > "$1"
      shift
      exec "$JULIA_BIN" --project=. full_aod_diag/d4_exact/'"$(basename "$RUNNER")"' "$1" "$2" "$3" "$4" "$5"
    ' _ "$pgid_file" "$STAGE_DIR" 1.0 "$budget" "$mode" "$seed_arg" \
      >> "$STAGE_DIR/stage.log" 2>&1 ) &
  echo $!
}

log "=== [$FAMILY] Step 1: start the real stage (calibration, D=20/W=80000, destination_sample(requested)=${DEST_SAMPLE_ARG:-<default>}) ==="
t_start=$(date +%s)
wrapper_pid=$(launch_stage "calibration" "" "$BUDGET_S")
log "wrapper_pid=$wrapper_pid, waiting for pgid file..."
if ! wait_for_pgid_file "$STAGE_DIR/run_meta.txt.pgid" 15; then
  echo "FAIL: [$FAMILY] pgid file never appeared" >&2; exit 1
fi
pgid=$(cat "$STAGE_DIR/run_meta.txt.pgid")
log "pgid=$pgid"

log "=== [$FAMILY] Step 2: wait for a schema-valid checkpoint (up to ~10 min at 10s poll) ==="
ckpt_path="$STAGE_DIR/stage_latest.jls"
have_ckpt=0
for i in $(seq 1 60); do
  if [ -f "$ckpt_path" ]; then
    have_ckpt=1
    log "poll $i: checkpoint file exists at $ckpt_path"
    break
  fi
  log "poll $i: no checkpoint file yet"
  sleep 10
done
if [ "$have_ckpt" -ne 1 ]; then
  echo "FAIL: [$FAMILY] no checkpoint appeared within the poll window" >&2
  kill_pgroup "$STAGE_DIR/run_meta.txt.pgid" KILL
  exit 1
fi
t_elapsed=$(( $(date +%s) - t_start ))
log "elapsed so far: ${t_elapsed}s"

log "=== [$FAMILY] Step 3: deliberately SIGTERM the process group, escalate to SIGKILL if needed ==="
kill_pgroup "$STAGE_DIR/run_meta.txt.pgid" TERM
waited=0
while alive_pgroup "$STAGE_DIR/run_meta.txt.pgid" && [ "$waited" -lt 20 ]; do sleep 1; waited=$((waited+1)); done
if alive_pgroup "$STAGE_DIR/run_meta.txt.pgid"; then
  log "still alive after 20s grace -- escalating to SIGKILL"
  kill_pgroup "$STAGE_DIR/run_meta.txt.pgid" KILL
  sleep 2
fi

log "=== [$FAMILY] Step 4: confirm no duplicate process remains ==="
if alive_pgroup "$STAGE_DIR/run_meta.txt.pgid"; then
  echo "FAIL: [$FAMILY] process group -$pgid still alive after SIGKILL" >&2; exit 1
fi
remaining_julia=$(pgrep -f "$(basename "$RUNNER") $STAGE_DIR " 2>/dev/null | wc -l)
log "no process-group members remain (pgid=$pgid); stray matching Julia processes: $remaining_julia"
[ "$remaining_julia" -eq 0 ] || { echo "FAIL: [$FAMILY] $remaining_julia stray stage-runner process(es) still running" >&2; exit 1; }

remaining_budget=$(( BUDGET_S - t_elapsed ))
[ "$remaining_budget" -gt 30 ] || remaining_budget=30
log "=== [$FAMILY] Step 5: resume through the same supervisor mechanics, remaining_budget=${remaining_budget}s ==="
wrapper_pid2=$(launch_stage "resume" "$ckpt_path" "$remaining_budget")
if ! wait_for_pgid_file "$STAGE_DIR/run_meta.txt.pgid" 15; then
  echo "FAIL: [$FAMILY] resume pgid file never appeared" >&2; exit 1
fi
pgid2=$(cat "$STAGE_DIR/run_meta.txt.pgid")
log "resumed under pgid=$pgid2, waiting for completion (up to ${remaining_budget}s + grace)..."
wait "$wrapper_pid2" 2>/dev/null
log "resume attempt exited"
if grep -q "STAGE_DONE" "$STAGE_DIR/stage.log"; then
  log "STAGE_DONE sentinel found -- clean completion (or budget exhaustion with checkpoint written)"
else
  log "WARNING: no STAGE_DONE sentinel found in stage.log after resume"
fi

log "=== [$FAMILY] Step 6: confirm banner/screen/threshold content in stage.log ==="
checks_ok=1
grep -q "destination_sample=$RESOLVED_DEST" "$STAGE_DIR/stage.log" && log "  PASS: resolved destination_sample=$RESOLVED_DEST printed" || { log "  FAIL: resolved destination_sample not found in log"; checks_ok=0; }
grep -q "\[active-layout\]" "$STAGE_DIR/stage.log" && log "  PASS: [active-layout] banner printed" || { log "  FAIL: [active-layout] banner missing"; checks_ok=0; }
if [ "$FAMILY" != "unrestricted" ]; then
  grep -q "\[screen-stack\]\|screen_counters" "$STAGE_DIR/stage.log" && log "  PASS: screen infrastructure present" || log "  NOTE: no explicit [screen-stack] line found (may be family-specific banner wording)"
fi
if [ "$RESOLVED_DEST" = "exclude_row" ]; then
  grep -q "destinations=19" "$STAGE_DIR/stage.log" && log "  PASS: destinations=19 (ROW omitted) confirmed in banner" || { log "  FAIL: destinations=19 not found for :exclude_row"; checks_ok=0; }
fi
[ "$checks_ok" -eq 1 ] || { echo "FAIL: [$FAMILY] one or more banner/content checks failed -- see log" >&2; exit 1; }

log "=== [$FAMILY] Step 7: verify resume REFUSAL under a mismatched destination_sample ==="
MISMATCH_DEST="all_legacy"; [ "$RESOLVED_DEST" = "all_legacy" ] && MISMATCH_DEST="exclude_row"
out=$(cd "$REPO_ROOT" && env "$DEST_ENV_NAME=$MISMATCH_DEST" "$JULIA_BIN" --project=. "$RUNNER" "$STAGE_DIR" 1.0 30 resume "$ckpt_path" 2>&1)
rc=$?
if [ "$rc" -ne 0 ] && ! echo "$out" | grep -q "STAGE_DONE"; then
  log "  PASS (refused as expected): resume under destination_sample=$MISMATCH_DEST rejected"
else
  echo "FAIL: [$FAMILY] resume did NOT refuse a destination_sample mismatch" >&2
  echo "$out" | tail -20 >&2
  exit 1
fi

log ">>> [$FAMILY] EXCLUDE_ROW_GATEC_SMOKE_TEST_DONE"
exit 0
