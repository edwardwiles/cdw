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
# State machine (four distinct terminal/transient outcomes per stage attempt --
# see docs/CM_PRODUCTION_LAUNCHER_2026-07-22.md for the full rationale):
#   clean_solver_completion    -- STAGE_DONE sentinel seen in THIS attempt's own log
#                                  region; stage is done, do not restart.
#   wall_budget_exhausted      -- the stage's inclusive wall budget ran out (whether the
#                                  process was still running or had just been resumed);
#                                  terminated safely, NOT restarted (budget is gone), and
#                                  treated as a normal (non-fatal) ending as long as a
#                                  schema-2 checkpoint exists -- the caller cold-verifies
#                                  best_feasible and only fails the stage if that fails.
#   probable_stall             -- >=STALL_THRESHOLD_S with no log growth/checkpoint
#                                  advance, budget NOT yet exhausted; terminated and
#                                  RESUMED from the latest checkpoint against the SAME
#                                  original stage deadline (no fresh budget granted).
#   unexpected_process_failure -- nonzero exit, no STAGE_DONE sentinel, not caused by a
#                                  deliberate wall/stall termination; fails the stage
#                                  outright, no automatic restart loop.
#
# stdout of run_stage_with_watchdog is reserved EXCLUSIVELY for its single return value
# (the checkpoint path), so it can be safely captured via command substitution. All
# logging (slog) goes to stderr (and is still appended to $SUPERVISOR_LOG) -- this is a
# structural fix, not a formatting one: before this fix, `ckpt_path=$(run_stage_with_watchdog ...)`
# captured EVERY slog line printed during the call (tee'd to stdout), not just the final
# checkpoint path. See scripts/test_cm_production_supervisor_return_path.sh for a
# shell-level regression test of exactly this.
#
# Usage:
#   scripts/cm_production_supervisor.sh <chain_id 1|2|3> <ckpt_root_dir>
#
# Independent directories per chain are the CALLER's responsibility (pass a distinct
# ckpt_root_dir per chain, e.g. production_runs/cm_campaign_2026-07-22/chain{1,2,3});
# this script never assumes or constructs a shared path across chains.
#
# Env overrides (production launches must use the defaults -- see the tunables block
# below for which ones are smoke-test-only):
#   RESUME_CAMPAIGN=1   required to launch into an already-nonempty ckpt_root_dir; the
#                        script refuses to start (exit 1, no side effects) into a
#                        nonempty directory otherwise, since a stale checkpoint or a
#                        stale STAGE_DONE sentinel from a PRIOR campaign could otherwise
#                        make a fresh failed attempt look like it inherited a completed
#                        stage.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4X_DIR="$(cd "$SCRIPT_DIR/../full_aod_diag/d4_exact" && pwd)"
REPO_ROOT="$(cd "$D4X_DIR/../.." && pwd)"

export JULIA_BIN="${JULIA_BIN:-$HOME/.juliaup/bin/julia}"   # NEVER /opt/shared_sw -- see memory note
# MUST be exported (found live, 2026-07-23): the process-group launch fix below runs
# `setsid bash -c '...'`, a genuinely NEW bash process (not a subshell fork of this script), which
# only inherits EXPORTED variables from the environment -- a plain (non-exported) shell variable
# silently expands to empty inside it, causing `exec "$JULIA_BIN" ...` to fail with
# "exec: : not found" and the stage runner never launching at all. The OLD (pre-fix) launch,
# `( cd ... && "$JULIA_BIN" ... ) &`, was a plain subshell fork of THIS script and so never needed
# JULIA_BIN to be exported -- this export only became necessary because of the setsid/bash -c
# rewrite, confirmed by a live smoke-test failure (15 minutes with zero checkpoint output, root
# cause found in stage.log: "_: line 3: exec: : not found").
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
EXPECTED_CKPT_SUFFIX="_latest.jls"   # every CMCheckpoint (schema-2) is written as "<label>_latest.jls"

# ----------------------------------------------------------------------------
# Process-group termination fix (2026-07-23, mean/ZC release addendum): the CM-C+ release
# report flagged that `pid=$!` after `( cd ... && "$JULIA_BIN" ... ) &` captures the SUBSHELL's
# PID, not necessarily the Julia/KNITRO process itself -- `kill -TERM/-KILL "$pid"` could
# therefore leave a Julia/KNITRO grandchild alive as an orphan if the subshell forked rather than
# exec'd into it. Fix: launch under `setsid bash -c '...; exec julia ...'` -- regardless of
# whatever fork depth setsid(1) itself uses internally, the bash script's own `$$` (captured
# BEFORE it execs into Julia) is, by construction, the PID of the process on which setsid()
# was actually called -- i.e. the new session/process-group leader's PID, preserved across the
# subsequent `exec` into Julia (exec never changes PID). Recording that PID to a file lets every
# kill below target the ENTIRE process group (`kill -SIG -- -$pgid`), not a single PID guess.
# ----------------------------------------------------------------------------
kill_pgroup() {
  local pgid_file="$1" sig="$2"
  [ -s "$pgid_file" ] || { slog "kill_pgroup: no pgid file at $pgid_file -- cannot signal process group"; return 1; }
  local pgid; pgid=$(cat "$pgid_file" 2>/dev/null)
  [ -n "$pgid" ] || { slog "kill_pgroup: empty pgid in $pgid_file"; return 1; }
  slog "kill_pgroup: sending SIG$sig to process group -$pgid"
  kill "-$sig" -- "-$pgid" 2>/dev/null
}
alive_pgroup() {
  local pgid_file="$1"
  [ -s "$pgid_file" ] || return 1
  local pgid; pgid=$(cat "$pgid_file" 2>/dev/null)
  [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null
}
# Waits up to $2 seconds (default 5) for the pgid file to appear (written asynchronously by the
# just-launched background job) -- a genuine race at process-launch time, not a symptom of a bug.
wait_for_pgid_file() {
  local pgid_file="$1" timeout_s="${2:-5}" waited=0
  while [ ! -s "$pgid_file" ] && [ "$waited" -lt "$timeout_s" ]; do
    sleep 0.2; waited=$(( waited + 1 ))
  done
  [ -s "$pgid_file" ]
}

# CM-C+ production integration 2026-07-23: defaults to the production backend (:cplus);
# CM_GRADIENT_BACKEND=reference selects the documented fallback/validation backend for the
# whole campaign instead. Exported so cm_production_stage_runner.jl (launched as a child
# process below) inherits it without any extra plumbing.
export CM_GRADIENT_BACKEND="${CM_GRADIENT_BACKEND:-cplus}"
export CM_ALLOW_BACKEND_SWITCH="${CM_ALLOW_BACKEND_SWITCH:-0}"

# CM+moments(+ZC) production integration (2026-07-23): defaults to :cm_only (unchanged production
# default -- the extension is an explicit opt-in). Exported so cm_production_stage_runner.jl
# (launched as a descendant process below) inherits it without any extra plumbing, same pattern
# as CM_GRADIENT_BACKEND above.
export CM_EXTENSION="${CM_EXTENSION:-cm_only}"
export MEANZC_K_MEAN="${MEANZC_K_MEAN:-0}"
export MEANZC_K_PAIR="${MEANZC_K_PAIR:-0}"
export MEANZC_BASIS="${MEANZC_BASIS:-direct}"
export MEANZC_ETA_NU0="${MEANZC_ETA_NU0:-}"

# slog: ALL output goes to stderr (and $SUPERVISOR_LOG, once it is set) -- NEVER stdout.
# This is what keeps `ckpt_path=$(run_stage_with_watchdog ...)` safe: stdout is reserved
# exclusively for run_stage_with_watchdog's single final `echo "$ckpt_latest"`.
slog() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] [chain ${CHAIN_ID:-?}] $*"
  if [ -n "${SUPERVISOR_LOG:-}" ]; then
    echo "$line" | tee -a "$SUPERVISOR_LOG" >&2
  else
    echo "$line" >&2
  fi
}

# Validates a checkpoint path returned by run_stage_with_watchdog before it is trusted for
# anything (cold-verification, seeding the next stage). Checks, in order: nonempty, exactly
# one line (no embedded newline -- the structural signature of log contamination), an
# existing regular file, located inside the expected stage directory, and named like a
# schema-2 CMCheckpoint ("<label>_latest.jls").
validate_ckpt_path() {
  local path="$1" expect_dir="$2"
  if [ -z "$path" ]; then
    slog "VALIDATION FAILED: empty checkpoint path returned"
    return 1
  fi
  case "$path" in
    *$'\n'*)
      slog "VALIDATION FAILED: checkpoint path return value contains multiple lines (possible log contamination): [$path]"
      return 1
      ;;
  esac
  if [ ! -f "$path" ]; then
    slog "VALIDATION FAILED: returned checkpoint path is not an existing regular file: $path"
    return 1
  fi
  local real_path real_dir
  real_path="$(readlink -f "$path" 2>/dev/null || echo "$path")"
  real_dir="$(readlink -f "$expect_dir" 2>/dev/null || echo "$expect_dir")"
  case "$real_path" in
    "$real_dir"/*) : ;;
    *)
      slog "VALIDATION FAILED: returned checkpoint path is not inside the expected stage directory ($expect_dir): $path"
      return 1
      ;;
  esac
  case "$path" in
    *"$EXPECTED_CKPT_SUFFIX") : ;;
    *)
      slog "VALIDATION FAILED: returned checkpoint path does not end with the expected schema-2 checkpoint filename ($EXPECTED_CKPT_SUFFIX): $path"
      return 1
      ;;
  esac
  return 0
}

# Runs one stage to completion (STAGE_WALL_S inclusive across restarts), handling
# stall detection/restart internally. Args: delta stage_dir mode seed_arg [chain_perturb_seed]
#
# Return contract: on success (clean_solver_completion OR wall_budget_exhausted with a
# checkpoint on disk), prints EXACTLY the checkpoint path -- one line, nothing else -- to
# stdout and returns 0. On failure (unexpected_process_failure, or wall-budget-exhausted /
# hung-with-no-checkpoint-ever-written), prints NOTHING to stdout and returns 1. Every
# other message this function produces (progress, stall detection, restarts) goes through
# slog (stderr + $SUPERVISOR_LOG), never stdout.
run_stage_with_watchdog() {
  local delta="$1" stage_dir="$2" mode="$3" seed_arg="$4" perturb_seed="${5:-0}"
  mkdir -p "$stage_dir"
  local log_file="$stage_dir/stage.log"
  local ckpt_latest="$stage_dir/stage_latest.jls"
  local stage_deadline=$(( $(date +%s) + STAGE_WALL_S ))
  local cur_mode="$mode" cur_seed="$seed_arg"
  local exit_reason=""

  while true; do
    local now; now=$(date +%s)
    local remaining=$(( stage_deadline - now ))
    if [ "$remaining" -le 0 ]; then
      # Budget was exhausted between restarts (e.g. a prior stall's grace period consumed
      # the last of it) rather than while a process was actively running -- same
      # wall_budget_exhausted classification as the in-loop case below.
      exit_reason="wall_budget_exhausted"
      slog "delta=$delta: stage wall budget (${STAGE_WALL_S}s) exhausted before a new attempt could launch"
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
      echo "cm_gradient_backend=$CM_GRADIENT_BACKEND"
      echo "cm_extension=$CM_EXTENSION"
      echo "meanzc_K_mean=$MEANZC_K_MEAN"
      echo "meanzc_K_pair=$MEANZC_K_PAIR"
      echo "meanzc_basis=$MEANZC_BASIS"
    } >> "$stage_dir/run_meta.txt"

    # Per-attempt sentinel scoping (item 3): record the log's byte size BEFORE this
    # attempt's own output is appended, so a STAGE_DONE sentinel left behind by an
    # EARLIER (possibly stalled/killed) attempt at this same stage_dir can never be
    # mistaken for THIS attempt's own completion. The log file itself stays one
    # cumulative file per stage (matches "resume" mode's own expectation of appending
    # to the same stage.log across restarts, for a single continuous narrative of the
    # stage), but STAGE_DONE is only ever searched for after this offset.
    local log_offset_before=0
    [ -f "$log_file" ] && log_offset_before=$(stat -c %s "$log_file" 2>/dev/null || echo 0)

    slog "delta=$delta: launching stage runner (mode=$cur_mode remaining=${remaining}s)"
    # NOTE: --project=. must resolve to REPO_ROOT's Project.toml (where SpecialFunctions etc.
    # are declared), not full_aod_diag/d4_exact (which has no Project.toml of its own) -- every
    # production/shakedown script in this tree is invoked this same way, cwd=repo root, script
    # given as a repo-relative path. Confirmed live this session: running from D4X_DIR instead
    # fails fast with "Package SpecialFunctions not found in current path."
    local pgid_file="$stage_dir/run_meta.txt.pgid"
    rm -f "$pgid_file"
    ( cd "$REPO_ROOT" && setsid bash -c '
        echo "$$" > "$1"
        shift
        exec "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_production_stage_runner.jl "$1" "$2" "$3" "$4" "$5" "$6"
      ' _ "$pgid_file" "$stage_dir" "$delta" "$remaining" "$cur_mode" "$cur_seed" "$perturb_seed" \
        >> "$log_file" 2>&1 ) &
    local pid=$!
    echo "$pid" > "$stage_dir/run_meta.txt.pid"
    if wait_for_pgid_file "$pgid_file" 10; then
      slog "delta=$delta: wrapper_pid=$pid pgid=$(cat "$pgid_file") log=$log_file"
    else
      slog "delta=$delta: WARNING: pgid file never appeared at $pgid_file within 10s -- process-group kill will not be available for this attempt (falling back to wrapper-pid-only signaling, the pre-fix behavior)"
    fi

    local last_size=-1 last_progress_t
    last_progress_t=$(date +%s)
    local hung=0 wall_exhausted=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL_INTERVAL_S"
      now=$(date +%s)
      if [ "$now" -ge "$stage_deadline" ]; then
        slog "delta=$delta: stage wall budget hit while pid=$pid still running -- terminating for wall-limit, not stall"
        local pgid_now=""; [ -s "$pgid_file" ] && pgid_now=$(cat "$pgid_file" 2>/dev/null)
        echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta wrapper_pid=$pid pgid=$pgid_now reason=wall_budget_exhausted" >> "$RESTART_LOG"
        if alive_pgroup "$pgid_file"; then
          kill_pgroup "$pgid_file" TERM
        else
          slog "delta=$delta: no live process group (pgid file missing/stale) -- falling back to wrapper-pid TERM"
          kill -TERM "$pid" 2>/dev/null
        fi
        sleep "$GRACE_TERM_S"
        if alive_pgroup "$pgid_file"; then
          slog "delta=$delta: process group -$pgid_now still alive after ${GRACE_TERM_S}s grace -- escalating to SIGKILL"
          kill_pgroup "$pgid_file" KILL
          echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta wrapper_pid=$pid pgid=$pgid_now action=SIGKILL_pgroup" >> "$RESTART_LOG"
        elif kill -0 "$pid" 2>/dev/null; then
          # pgroup already gone but the wrapper subshell itself is somehow still alive (should not
          # normally happen given the exec chain) -- kill it directly too, belt and suspenders.
          kill -KILL "$pid" 2>/dev/null
          echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta wrapper_pid=$pid pgid=$pgid_now action=SIGKILL_wrapper_fallback" >> "$RESTART_LOG"
        fi
        # Confirm cleanup: no process in this group should remain.
        if [ -n "$pgid_now" ] && pgrep -g "$pgid_now" >/dev/null 2>&1; then
          slog "delta=$delta: WARNING: pgrep still finds live members of process group -$pgid_now after kill sequence"
        else
          slog "delta=$delta: confirmed no process-group members remain (pgid=$pgid_now)"
        fi
        wall_exhausted=1
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

        local pgid_now=""; [ -s "$pgid_file" ] && pgid_now=$(cat "$pgid_file" 2>/dev/null)
        slog "delta=$delta: sending SIGTERM to process group -$pgid_now (graceful first)"
        if alive_pgroup "$pgid_file"; then
          kill_pgroup "$pgid_file" TERM
        else
          slog "delta=$delta: no live process group (pgid file missing/stale) -- falling back to wrapper-pid TERM"
          kill -TERM "$pid" 2>/dev/null
        fi
        local waited=0
        while alive_pgroup "$pgid_file" && [ "$waited" -lt "$GRACE_TERM_S" ]; do
          sleep 2; waited=$(( waited + 2 ))
        done
        if alive_pgroup "$pgid_file"; then
          slog "delta=$delta: process group -$pgid_now still alive after ${GRACE_TERM_S}s grace -- escalating to SIGKILL"
          kill_pgroup "$pgid_file" KILL
          echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta wrapper_pid=$pid pgid=$pgid_now action=SIGKILL_pgroup" >> "$RESTART_LOG"
        elif kill -0 "$pid" 2>/dev/null; then
          kill -KILL "$pid" 2>/dev/null
          echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta wrapper_pid=$pid pgid=$pgid_now action=SIGKILL_wrapper_fallback" >> "$RESTART_LOG"
        else
          echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta wrapper_pid=$pid pgid=$pgid_now action=SIGTERM_succeeded" >> "$RESTART_LOG"
        fi
        if [ -n "$pgid_now" ] && pgrep -g "$pgid_now" >/dev/null 2>&1; then
          slog "delta=$delta: WARNING: pgrep still finds live members of process group -$pgid_now after kill sequence"
        else
          slog "delta=$delta: confirmed no process-group members remain (pgid=$pgid_now)"
        fi
        hung=1
        break
      fi
    done

    wait "$pid" 2>/dev/null
    local exit_code=$?

    local sentinel_found=0
    if [ "$hung" -eq 0 ] && [ "$wall_exhausted" -eq 0 ]; then
      if [ -f "$log_file" ] && tail -c "+$((log_offset_before + 1))" "$log_file" 2>/dev/null | grep -q "STAGE_DONE"; then
        sentinel_found=1
      fi
    fi

    if [ "$wall_exhausted" -eq 1 ]; then
      exit_reason="wall_budget_exhausted"
      slog "delta=$delta: exit_reason=wall_budget_exhausted (terminated for wall-limit, not restarting -- inclusive stage budget is gone)"
      echo "exit_reason=wall_budget_exhausted" >> "$stage_dir/run_meta.txt"
      break
    elif [ "$sentinel_found" -eq 1 ]; then
      exit_reason="clean_solver_completion"
      slog "delta=$delta: exit_reason=clean_solver_completion (exit=$exit_code, STAGE_DONE sentinel confirmed in this attempt's own log region)"
      echo "exit_reason=clean_solver_completion" >> "$stage_dir/run_meta.txt"
      break
    elif [ "$hung" -eq 1 ]; then
      if [ ! -f "$ckpt_latest" ]; then
        exit_reason="probable_stall_no_checkpoint"
        slog "delta=$delta: exit_reason=probable_stall_no_checkpoint (hung with NO checkpoint ever written -- cannot resume, aborting this stage; manual intervention required)"
        echo "exit_reason=probable_stall_no_checkpoint" >> "$stage_dir/run_meta.txt"
        return 1
      fi
      slog "delta=$delta: exit_reason=probable_stall -- restarting from latest checkpoint $ckpt_latest (same original deadline, no fresh budget)"
      echo "exit_reason=probable_stall" >> "$stage_dir/run_meta.txt"
      cur_mode="resume"
      cur_seed="$ckpt_latest"
      continue
    else
      # Nonzero/zero exit, no STAGE_DONE sentinel in THIS attempt's own log region, not
      # caused by a deliberate wall/stall termination -- a real, unexpected process
      # failure (Julia exception, KNITRO hard error, etc). Do not loop forever.
      exit_reason="unexpected_process_failure"
      slog "delta=$delta: exit_reason=unexpected_process_failure (exit=$exit_code) -- exited WITHOUT the STAGE_DONE sentinel and without a deliberate wall/stall termination. Check $log_file."
      echo "exit_reason=unexpected_process_failure exit_code=$exit_code" >> "$stage_dir/run_meta.txt"
      echo "$(date '+%Y-%m-%d %H:%M:%S') chain=$CHAIN_ID delta=$delta pid=$pid action=exited_no_sentinel exit_code=$exit_code reason=unexpected_process_failure" >> "$RESTART_LOG"
      return 1
    fi
  done

  # Reached only via `break` above -- clean_solver_completion or wall_budget_exhausted.
  if [ -f "$ckpt_latest" ]; then
    echo "$ckpt_latest"
    return 0
  else
    slog "delta=$delta: exit_reason=$exit_reason but no checkpoint exists at $ckpt_latest -- stage failed"
    echo "exit_reason=${exit_reason}_no_checkpoint" >> "$stage_dir/run_meta.txt"
    return 1
  fi
}

# ---- chain state machine: walk the delta ladder, cold-verifying between stages ----
main() {
  CHAIN_ID="${1:?usage: cm_production_supervisor.sh <chain_id> <ckpt_root_dir>}"
  CKPT_ROOT="${2:?usage: cm_production_supervisor.sh <chain_id> <ckpt_root_dir>}"

  RESUME_CAMPAIGN="${RESUME_CAMPAIGN:-0}"
  if [ -d "$CKPT_ROOT" ] && [ -n "$(ls -A "$CKPT_ROOT" 2>/dev/null)" ] && [ "$RESUME_CAMPAIGN" != "1" ]; then
    echo "ERROR: $CKPT_ROOT already exists and is nonempty. Refusing to launch a fresh campaign" \
         "into it -- a stale checkpoint or a stale STAGE_DONE sentinel from a prior campaign" \
         "could otherwise make a fresh failed attempt look like it inherited a completed stage." \
         "Pass RESUME_CAMPAIGN=1 if you explicitly intend to resume/continue this exact" \
         "campaign directory." >&2
    return 1
  fi

  mkdir -p "$CKPT_ROOT"
  RESTART_LOG="$CKPT_ROOT/restarts.log"
  SUPERVISOR_LOG="$CKPT_ROOT/supervisor.log"
  COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

  slog "=== supervisor starting === commit=$COMMIT ckpt_root=$CKPT_ROOT julia_threads=$JULIA_NUM_THREADS resume_campaign=$RESUME_CAMPAIGN cm_gradient_backend=$CM_GRADIENT_BACKEND"

  local mode="calibration"
  local seed_arg=""

  for delta in "${DELTAS[@]}"; do
    local stage_dir="$CKPT_ROOT/delta_${delta}"
    local ckpt_path
    ckpt_path=$(run_stage_with_watchdog "$delta" "$stage_dir" "$mode" "$seed_arg" "$CHAIN_ID")
    local status=$?
    if [ "$status" -ne 0 ]; then
      slog "delta=$delta: FAILED -- aborting chain $CHAIN_ID (no further deltas will run)"
      return 1
    fi
    if ! validate_ckpt_path "$ckpt_path" "$stage_dir"; then
      slog "delta=$delta: FAILED -- returned checkpoint path failed validation -- aborting chain $CHAIN_ID"
      return 1
    fi

    slog "delta=$delta: cold-verifying $ckpt_path"
    local verify_out="$stage_dir/cold_verified_seed.jls"
    if ! ( cd "$REPO_ROOT" && "$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_cold_verify.jl "$ckpt_path" "$verify_out" \
          >> "$stage_dir/coldverify.log" 2>&1 ); then
      slog "delta=$delta: COLD VERIFICATION FAILED (no verified feasible incumbent exists) -- see $stage_dir/coldverify.log -- aborting chain $CHAIN_ID"
      return 1
    fi
    slog "delta=$delta: cold verification passed -> $verify_out"

    mode="seed_w0"
    seed_arg="$verify_out"
  done

  slog "=== chain $CHAIN_ID complete: all deltas (${DELTAS[*]}) finished and cold-verified ==="
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
  exit $?
fi
