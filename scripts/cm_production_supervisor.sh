#!/usr/bin/env bash
# ============================================================================
# Supervisor for the 2026-07-22 D=20 CM production campaign.
#
# Runs ONE chain (a sequence of delta stages 0.1 -> 0.5 -> 1.0 -> 2.0), each stage
# initialized from the PRECEDING stage's cold-verified best-feasible incumbent (never
# the terminal iterate). Each stage has an inclusive 1-hour wall budget across ALL of
# that stage's own (re)starts -- a restart after a detected stall consumes the
# REMAINING budget, it does not get a fresh hour.
#
# Watchdog policy (per the brief):
#   - polls every $POLL_INTERVAL_S seconds
#   - does NOT classify an ordinary 90-130s CM callback as a hang (that is far below
#     the 600s threshold below)
#   - "probable stall" only after >= 10 minutes (600s) with BOTH (a) no new line in the
#     stage's log file (the heartbeat_interval_s=30s timer inside run_cm_upper_checkpointed
#     guarantees a log line at least that often when the process is healthy -- see
#     docs/CM_PRODUCTION_STATE_2026-07-22.md's account of the Phase 4 hang, where this
#     exact heartbeat stopped producing output) and (b) no checkpoint file mtime advance
#   - checks process state (ps STAT column) and CPU time as corroborating evidence, not
#     as the sole trigger (a genuinely busy KN_solve call can sit at STAT=R indefinitely
#     and that is NORMAL, not a stall signal by itself)
#   - graceful SIGTERM first, a grace period to exit, SIGKILL only if still alive
#   - every restart is appended to $RESTART_LOG with timestamp + reason
#   - never launches two stage-runner processes against the same CKPT_DIR
#
# Usage:
#   scripts/cm_production_supervisor.sh <chain_id 1|2|3> <ckpt_root_dir>
#
# Independent directories per chain are the CALLER's responsibility (pass a distinct
# ckpt_root_dir per chain, e.g. production_runs/cm_campaign_2026-07-22/chain{1,2,3});
# this script never assumes or constructs a shared path across chains.
# ============================================================================
set -uo pipefail

CHAIN_ID="${1:?usage: cm_production_supervisor.sh <chain_id> <ckpt_root_dir>}"
CKPT_ROOT="${2:?usage: cm_production_supervisor.sh <chain_id> <ckpt_root_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4X_DIR="$(cd "$SCRIPT_DIR/../full_aod_diag/d4_exact" && pwd)"
REPO_ROOT="$(cd "$D4X_DIR/../.." && pwd)"

JULIA_BIN="${JULIA_BIN:-$HOME/.juliaup/bin/julia}"   # NEVER /opt/shared_sw -- see memory note
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-20}"   # <= 20 per the brief's hard cap
export OPENBLAS_NUM_THREADS=1   # NOT bounded by JULIA_NUM_THREADS automatically -- must set explicitly
if [ -f "$REPO_ROOT/.knitro_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.knitro_env.sh"
fi

# All four tunables are env-overridable ONLY for deployment smoke-testing (documented in
# docs/CM_PRODUCTION_LAUNCHER_2026-07-22.md's smoke-test section) -- production launches must
# use the defaults below (1 hour / 10 minutes), never override STALL_THRESHOLD_S downward for a
# real campaign.
STAGE_WALL_S="${STAGE_WALL_S:-3600}"            # inclusive 1-hour wall limit per delta stage
POLL_INTERVAL_S="${POLL_INTERVAL_S:-20}"
STALL_THRESHOLD_S="${STALL_THRESHOLD_S:-600}"    # >=10 minutes with no evidence of progress
GRACE_TERM_S="${GRACE_TERM_S:-30}"                # wait this long after SIGTERM before escalating to SIGKILL
DELTAS_OVERRIDE="${DELTAS_OVERRIDE:-}"
if [ -n "$DELTAS_OVERRIDE" ]; then
  read -r -a DELTAS <<< "$DELTAS_OVERRIDE"
else
  DELTAS=(0.1 0.5 1.0 2.0)
fi

mkdir -p "$CKPT_ROOT"
RESTART_LOG="$CKPT_ROOT/restarts.log"
SUPERVISOR_LOG="$CKPT_ROOT/supervisor.log"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

slog() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [chain $CHAIN_ID] $*" | tee -a "$SUPERVISOR_LOG"; }

slog "=== supervisor starting === commit=$COMMIT ckpt_root=$CKPT_ROOT julia_threads=$JULIA_NUM_THREADS"

# Runs one stage to completion (STAGE_WALL_S inclusive across restarts), handling hang
# detection/restart internally. Args: delta stage_dir mode seed_arg [chain_perturb_seed]
# On success, prints the path to the stage's own "<label>_latest.jls" checkpoint on stdout
# (last line) so the caller can cold-verify it.
run_stage_with_watchdog() {
  local delta="$1" stage_dir="$2" mode="$3" seed_arg="$4" perturb_seed="${5:-0}"
  mkdir -p "$stage_dir"
  local log_file="$stage_dir/stage.log"
  local ckpt_latest="$stage_dir/stage_latest.jls"
  local stage_deadline=$(( $(date +%s) + STAGE_WALL_S ))
  local cur_mode="$mode" cur_seed="$seed_arg"

  while true; do
    local now; now=$(date +%s)
    local remaining=$(( stage_deadline - now ))
    if [ "$remaining" -le 0 ]; then
      slog "delta=$delta: stage wall budget ($STAGE_WALL_S s) exhausted, not restarting again"
      break
    fi

    : > "$stage_dir/run_meta.txt"
    {
      echo "commit=$COMMIT"
      echo "chain=$CHAIN_ID"
      echo "delta=$delta"
      echo "stage=$mode"
      echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "remaining_budget_s=$remaining"
    } >> "$stage_dir/run_meta.txt"

    slog "delta=$delta: launching stage runner (mode=$cur_mode remaining=${remaining}s)"
    # NOTE: --project=. must resolve to REPO_ROOT's Project.toml (where SpecialFunctions etc.
    # are declared), not full_aod_diag/d4_exact (which has no Project.toml of its own) -- every
    # production/shakedown script in this tree is invoked this same way, cwd=repo root, script
    # given as a repo-relative path. Confirmed live this session: running from D4X_DIR instead
    # fails fast with "Package SpecialFunctions not found in current path."
    ( cd "$REPO_ROOT" && "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_production_stage_runner.jl \
        "$stage_dir" "$delta" "$remaining" "$cur_mode" "$cur_seed" "$perturb_seed" \
        >> "$log_file" 2>&1 ) &
    local pid=$!
    echo "$pid" > "$stage_dir/run_meta.txt.pid"
    slog "delta=$delta: pid=$pid log=$log_file"

    local last_size=-1 last_progress_t
    last_progress_t=$(date +%s)
    local hung=0 exited_clean=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL_INTERVAL_S"
      now=$(date +%s)
      if [ "$now" -ge "$stage_deadline" ]; then
        slog "delta=$delta: stage wall budget hit while pid=$pid still running -- terminating for wall-limit, not stall"
        kill -TERM "$pid" 2>/dev/null
        sleep "$GRACE_TERM_S"
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
        break
      fi

      local cur_size=0
      [ -f "$log_file" ] && cur_size=$(stat -c %s "$log_file" 2>/dev/null || echo 0)
      local ckpt_mtime=0
      [ -f "$ckpt_latest" ] && ckpt_mtime=$(stat -c %Y "$ckpt_latest" 2>/dev/null || echo 0)

      if [ "$cur_size" != "$last_size" ] || [ "$ckpt_mtime" -gt "$last_progress_t" ]; then
        last_size="$cur_size"
        last_progress_t="$now"
        continue
      fi

      local stall_for=$(( now - last_progress_t ))
      if [ "$stall_for" -ge "$STALL_THRESHOLD_S" ]; then
        # Corroborate with process state/CPU before declaring -- log it either way, since an
        # ordinary long single callback would NOT reach this branch at all: heartbeat_interval_s=30
        # guarantees a log line at least every 30s when healthy, so >=600s of silence already means
        # something well beyond "one slow callback" per the brief's own 90-130s baseline.
        local ps_info; ps_info=$(ps -o pid,stat,etimes,pcpu,cmd -p "$pid" 2>/dev/null | tail -n1)
        slog "delta=$delta: ** PROBABLE STALL ** pid=$pid no log growth / checkpoint advance for ${stall_for}s (>=${STALL_THRESHOLD_S}s threshold). ps: $ps_info"
        echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta pid=$pid reason=probable_stall stall_for=${stall_for}s ps=[$ps_info]" >> "$RESTART_LOG"

        slog "delta=$delta: sending SIGTERM to pid=$pid (graceful first)"
        kill -TERM "$pid" 2>/dev/null
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$GRACE_TERM_S" ]; do
          sleep 2; waited=$(( waited + 2 ))
        done
        if kill -0 "$pid" 2>/dev/null; then
          slog "delta=$delta: pid=$pid still alive after ${GRACE_TERM_S}s grace -- escalating to SIGKILL"
          kill -KILL "$pid" 2>/dev/null
          echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta pid=$pid action=SIGKILL" >> "$RESTART_LOG"
        else
          echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta pid=$pid action=SIGTERM_succeeded" >> "$RESTART_LOG"
        fi
        hung=1
        break
      fi
    done

    wait "$pid" 2>/dev/null
    local exit_code=$?
    if [ "$hung" -eq 0 ]; then
      if grep -q "STAGE_DONE" "$log_file" 2>/dev/null; then
        exited_clean=1
        slog "delta=$delta: stage runner completed cleanly (exit=$exit_code)"
      else
        slog "delta=$delta: stage runner exited (exit=$exit_code) WITHOUT the STAGE_DONE sentinel -- treating as a non-hang failure, not restarting automatically. Check $log_file."
        echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta pid=$pid action=exited_no_sentinel exit_code=$exit_code" >> "$RESTART_LOG"
      fi
    fi

    if [ "$exited_clean" -eq 1 ]; then
      break
    fi
    if [ "$hung" -eq 1 ]; then
      if [ ! -f "$ckpt_latest" ]; then
        slog "delta=$delta: hung with NO checkpoint ever written -- cannot resume, aborting this stage. Manual intervention required."
        return 1
      fi
      slog "delta=$delta: restarting from latest checkpoint $ckpt_latest"
      cur_mode="resume"
      cur_seed="$ckpt_latest"
      continue
    fi
    # Non-hang, non-clean exit (a real error) -- do not loop forever.
    return 1
  done

  if [ -f "$ckpt_latest" ]; then
    echo "$ckpt_latest"
    return 0
  else
    slog "delta=$delta: no checkpoint produced -- stage failed"
    return 1
  fi
}

# ---- chain state machine: walk the delta ladder, cold-verifying between stages ----
prev_seed_file=""
mode="calibration"
seed_arg=""

for delta in "${DELTAS[@]}"; do
  stage_dir="$CKPT_ROOT/delta_${delta}"
  ckpt_path=$(run_stage_with_watchdog "$delta" "$stage_dir" "$mode" "$seed_arg" "$CHAIN_ID")
  status=$?
  if [ "$status" -ne 0 ] || [ -z "$ckpt_path" ]; then
    slog "delta=$delta: FAILED -- aborting chain $CHAIN_ID (no further deltas will run)"
    exit 1
  fi

  slog "delta=$delta: cold-verifying $ckpt_path"
  verify_out="$stage_dir/cold_verified_seed.jls"
  if ! ( cd "$REPO_ROOT" && "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_cold_verify.jl "$ckpt_path" "$verify_out" \
        >> "$stage_dir/coldverify.log" 2>&1 ); then
    slog "delta=$delta: COLD VERIFICATION FAILED -- see $stage_dir/coldverify.log -- aborting chain $CHAIN_ID"
    exit 1
  fi
  slog "delta=$delta: cold verification passed -> $verify_out"

  mode="seed_w0"
  seed_arg="$verify_out"
done

slog "=== chain $CHAIN_ID complete: all deltas (${DELTAS[*]}) finished and cold-verified ==="
