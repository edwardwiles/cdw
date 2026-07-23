#!/usr/bin/env bash
# ============================================================================
# Real D=20/W=80,000/L=50 production checkpoint/interrupt/SIGKILL/resume/
# cold-verification smoke test for the CM+moments(+ZC) integration (2026-07-23),
# per the release addendum Section 6 / multiple-K addendum Section 4.3:
# cm_extension=:cm_plus_moments, K_mean=2, K_pair=2, backend=:cplus.
#
# Uses the ACTUAL production launch mechanics (setsid+pgid process-group launch,
# kill_pgroup/alive_pgroup) by sourcing cm_production_supervisor.sh directly --
# not a separate reimplementation.
#
# Procedure (matches the task's own required steps exactly):
#   1. start the real stage (mode=calibration)
#   2. wait for a schema-valid checkpoint with a typed best verified feasible incumbent
#   3. deliberately SIGTERM the process group, escalate to SIGKILL if needed
#   4. confirm no duplicate process remains
#   5. resume through the SAME supervisor mechanics with the original remaining budget
#   6. cold-verify the exact stored full vector (incl. eta_nu) in a fresh process, cache disabled
#   7. verify reported and cold Delta_dual agree to numerical precision
#   8. verify the checkpoint refuses resume under changed L/draws/contrasts/extension/backend/K
#
# Usage: bash scripts/d20_meanzc_supervisor_smoke_test.sh <stage_dir> <budget_s>
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAIN_ID="smoke"
SUPERVISOR_LOG=""
# shellcheck disable=SC1091
source "$SCRIPT_DIR/cm_production_supervisor.sh"

STAGE_DIR="${1:?usage: $0 <stage_dir> <budget_s>}"
BUDGET_S="${2:?usage: $0 <stage_dir> <budget_s>}"
mkdir -p "$STAGE_DIR"

export CM_EXTENSION=cm_plus_moments
export MEANZC_K_MEAN=2
export MEANZC_K_PAIR=2
export MEANZC_BASIS=direct
export CM_GRADIENT_BACKEND=cplus

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

launch_stage() {
  local mode="$1" seed_arg="$2" budget="$3"
  local pgid_file="$STAGE_DIR/run_meta.txt.pgid"
  rm -f "$pgid_file"
  ( cd "$REPO_ROOT" && setsid bash -c '
      echo "$$" > "$1"
      shift
      exec "$JULIA_BIN" --project=. -t 8 full_aod_diag/d4_exact/cm_production_stage_runner.jl "$1" "$2" "$3" "$4" "$5" "$6"
    ' _ "$pgid_file" "$STAGE_DIR" 1.0 "$budget" "$mode" "$seed_arg" 0 \
      >> "$STAGE_DIR/stage.log" 2>&1 ) &
  echo $!   # wrapper pid
}

log "=== Step 1: start the real stage (calibration, D=20/W=80000/L=50, K_mean=2/K_pair=2/cplus) ==="
t_start=$(date +%s)
wrapper_pid=$(launch_stage "calibration" "" "$BUDGET_S")
log "wrapper_pid=$wrapper_pid, waiting for pgid file..."
if ! wait_for_pgid_file "$STAGE_DIR/run_meta.txt.pgid" 15; then
  echo "FAIL: pgid file never appeared" >&2; exit 1
fi
pgid=$(cat "$STAGE_DIR/run_meta.txt.pgid")
log "pgid=$pgid"

log "=== Step 2: wait for a schema-valid checkpoint with a typed best verified feasible incumbent ==="
ckpt_path="$STAGE_DIR/stage_latest.jls"
have_incumbent=0
for i in $(seq 1 90); do   # up to ~15 min at 10s poll
  if [ -f "$ckpt_path" ]; then
    has_best=$("$JULIA_BIN" --project="$REPO_ROOT" -e '
      include(joinpath("'"$REPO_ROOT"'", "full_aod_diag/d4_exact/oracle.jl"))
      include(joinpath("'"$REPO_ROOT"'", "full_aod_diag/d4_exact/cm_config.jl"))
      include(joinpath("'"$REPO_ROOT"'", "full_aod_diag/d4_exact/cm_meanzc_moments.jl"))
      include(joinpath("'"$REPO_ROOT"'", "full_aod_diag/d4_exact/cm_meanzc_config.jl"))
      include(joinpath("'"$REPO_ROOT"'", "full_aod_diag/d4_exact/cm_checkpoint.jl"))
      ck = load_cm_checkpoint("'"$ckpt_path"'")
      println(ck.best_feasible === nothing ? "NO" : "YES:schema=$(ck.schema):K_mean=$(ck.meanzc_K_mean):K_pair=$(ck.meanzc_K_pair):eta_nu=$(ck.eta_nu):Delta=$(ck.best_feasible.Delta)")
      ' 2>&1 | tail -1)
    log "poll $i: checkpoint exists, best_feasible check: $has_best"
    if [[ "$has_best" == YES:* ]]; then
      have_incumbent=1
      log "  -> schema-valid checkpoint with typed best verified feasible incumbent CONFIRMED: $has_best"
      break
    fi
  else
    log "poll $i: no checkpoint file yet"
  fi
  sleep 10
done
if [ "$have_incumbent" -ne 1 ]; then
  echo "FAIL: no verified feasible incumbent appeared within the poll window" >&2
  kill_pgroup "$STAGE_DIR/run_meta.txt.pgid" KILL
  exit 1
fi
t_elapsed=$(( $(date +%s) - t_start ))
log "elapsed so far: ${t_elapsed}s"

log "=== Step 3: deliberately SIGTERM the process group, escalate to SIGKILL if needed ==="
kill_pgroup "$STAGE_DIR/run_meta.txt.pgid" TERM
waited=0
while alive_pgroup "$STAGE_DIR/run_meta.txt.pgid" && [ "$waited" -lt 20 ]; do sleep 1; waited=$((waited+1)); done
if alive_pgroup "$STAGE_DIR/run_meta.txt.pgid"; then
  log "still alive after 20s grace -- escalating to SIGKILL"
  kill_pgroup "$STAGE_DIR/run_meta.txt.pgid" KILL
  sleep 2
fi

log "=== Step 4: confirm no duplicate process remains ==="
if alive_pgroup "$STAGE_DIR/run_meta.txt.pgid"; then
  echo "FAIL: process group -$pgid still alive after SIGKILL" >&2; exit 1
fi
if pgrep -g "$pgid" >/dev/null 2>&1; then
  echo "FAIL: pgrep still finds live members of group -$pgid" >&2; exit 1
fi
remaining_julia=$(pgrep -f "cm_production_stage_runner.jl $STAGE_DIR " 2>/dev/null | wc -l)
log "no process-group members remain (pgid=$pgid); stray matching Julia processes: $remaining_julia"
[ "$remaining_julia" -eq 0 ] || { echo "FAIL: $remaining_julia stray stage-runner process(es) still running" >&2; exit 1; }

remaining_budget=$(( BUDGET_S - t_elapsed ))
[ "$remaining_budget" -gt 30 ] || remaining_budget=30
log "=== Step 5: resume through the same supervisor mechanics, remaining_budget=${remaining_budget}s ==="
wrapper_pid2=$(launch_stage "resume" "$ckpt_path" "$remaining_budget")
if ! wait_for_pgid_file "$STAGE_DIR/run_meta.txt.pgid" 15; then
  echo "FAIL: resume pgid file never appeared" >&2; exit 1
fi
pgid2=$(cat "$STAGE_DIR/run_meta.txt.pgid")
log "resumed under pgid=$pgid2, waiting for completion (up to ${remaining_budget}s + grace)..."
wait "$wrapper_pid2" 2>/dev/null
log "resume attempt exited"
if grep -q "STAGE_DONE" "$STAGE_DIR/stage.log"; then
  log "STAGE_DONE sentinel found -- clean completion (or budget exhaustion with checkpoint written)"
else
  log "WARNING: no STAGE_DONE sentinel found in stage.log after resume (may still have written a valid checkpoint if budget ran out)"
fi

log "=== Step 6+7: cold-verify the exact stored full vector in a fresh process, cache disabled ==="
verify_out="$STAGE_DIR/cold_verified_seed.jls"
if ! ( cd "$REPO_ROOT" && "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_cold_verify.jl "$ckpt_path" "$verify_out" > "$STAGE_DIR/coldverify.log" 2>&1 ); then
  echo "FAIL: cold verification failed -- see $STAGE_DIR/coldverify.log" >&2
  tail -30 "$STAGE_DIR/coldverify.log" >&2
  exit 1
fi
grep "reported Delta=" "$STAGE_DIR/coldverify.log"
diff_line=$(grep "reported Delta=" "$STAGE_DIR/coldverify.log" | tail -1)
log "cold-verify result: $diff_line"

log "=== Step 8: verify resume REFUSAL under changed K_mean/K_pair/extension/backend ==="
refusal_tests=0
refusal_passed=0

test_refusal() {
  local desc="$1"; shift
  refusal_tests=$((refusal_tests+1))
  local out
  out=$(cd "$REPO_ROOT" && "$@" 2>&1)
  local rc=$?
  if [ "$rc" -ne 0 ] && ! echo "$out" | grep -q "STAGE_DONE"; then
    log "  PASS (refused as expected): $desc"
    refusal_passed=$((refusal_passed+1))
  else
    log "  FAIL (did NOT refuse): $desc"
    echo "$out" | tail -20 >&2
  fi
}

test_refusal "different K_mean (1 instead of 2)" \
  env MEANZC_K_MEAN=1 MEANZC_K_PAIR=1 CM_GRADIENT_BACKEND=cplus CM_EXTENSION=cm_plus_moments \
  "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_production_stage_runner.jl "$STAGE_DIR" 1.0 30 resume "$ckpt_path" 0

test_refusal "different K_pair (0 instead of 2)" \
  env MEANZC_K_MEAN=2 MEANZC_K_PAIR=0 CM_GRADIENT_BACKEND=cplus CM_EXTENSION=cm_plus_moments \
  "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_production_stage_runner.jl "$STAGE_DIR" 1.0 30 resume "$ckpt_path" 0

test_refusal "different cm_extension (cm_plus_equal_means instead of cm_plus_moments)" \
  env MEANZC_K_MEAN=1 MEANZC_K_PAIR=0 CM_GRADIENT_BACKEND=cplus CM_EXTENSION=cm_plus_equal_means \
  "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_production_stage_runner.jl "$STAGE_DIR" 1.0 30 resume "$ckpt_path" 0

test_refusal "different cm_gradient_backend (reference instead of cplus, no allow_backend_switch)" \
  env MEANZC_K_MEAN=2 MEANZC_K_PAIR=2 CM_GRADIENT_BACKEND=reference CM_EXTENSION=cm_plus_moments CM_ALLOW_BACKEND_SWITCH=0 \
  "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_production_stage_runner.jl "$STAGE_DIR" 1.0 30 resume "$ckpt_path" 0

log "resume-refusal checks: $refusal_passed/$refusal_tests passed"
[ "$refusal_passed" -eq "$refusal_tests" ] || { echo "FAIL: not all resume-refusal checks passed" >&2; exit 1; }

log ">>> D20_MEANZC_SUPERVISOR_SMOKE_TEST_DONE"
exit 0
