#!/usr/bin/env bash
# ============================================================================
# Shell-level regression tests for the supervisor's 4-way state machine
# (clean_solver_completion / wall_budget_exhausted / probable_stall / unexpected_process_failure)
# plus the stale-STAGE_DONE-offset guard and the nonempty-campaign-directory guard.
#
# No Julia/KNITRO involved (fake JULIA_BIN stubs) -- fast, deterministic, pure bash.
# For the REAL-KNITRO end-to-end versions of the wall-budget/stall/clean-completion
# paths, see the campaign smoke-test procedure in
# docs/CM_PRODUCTION_LAUNCHER_2026-07-22.md ("Full supervisor smoke tests").
#
# Usage: scripts/test_cm_production_supervisor_state_machine.sh
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

PASS_COUNT=0
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "PASS: $*"; }

# shellcheck disable=SC1091
source "$SCRIPT_DIR/cm_production_supervisor.sh"

# ============================================================================
# Test 1: clean_solver_completion -- fake julia finishes immediately with the
# sentinel and a checkpoint; must classify as clean_solver_completion, no restart.
# ============================================================================
t1() {
  local T="$TMPDIR_TEST/t1"; mkdir -p "$T"
  local julia="$T/fake_julia.sh"
  cat > "$julia" <<'STUB'
#!/usr/bin/env bash
stage_dir="$3"
echo "fake julia: finishing immediately"
: > "$stage_dir/stage_latest.jls"
echo "STAGE_DONE"
exit 0
STUB
  chmod +x "$julia"

  JULIA_BIN="$julia" STAGE_WALL_S=60 POLL_INTERVAL_S=1 STALL_THRESHOLD_S=5 GRACE_TERM_S=1 \
    bash -c '
      source "'"$SCRIPT_DIR"'/cm_production_supervisor.sh"
      CHAIN_ID=t1; RESTART_LOG="'"$T"'/restarts.log"; SUPERVISOR_LOG="'"$T"'/supervisor.log"; COMMIT=x
      stage_dir="'"$T"'/delta_0.1"
      out=$(run_stage_with_watchdog "0.1" "$stage_dir" "calibration" "" "0")
      status=$?
      echo "STATUS=$status OUT=$out" > "'"$T"'/result.txt"
    '
  local status out
  status=$(grep -o 'STATUS=[0-9]*' "$T/result.txt" | cut -d= -f2)
  out=$(sed -e 's/^STATUS=[0-9]* OUT=//' "$T/result.txt")
  [ "$status" = "0" ] || fail "t1: expected status 0, got $status"
  [ "$out" = "$T/delta_0.1/stage_latest.jls" ] || fail "t1: unexpected checkpoint path: $out"
  grep -q "exit_reason=clean_solver_completion" "$T/supervisor.log" || fail "t1: supervisor.log missing exit_reason=clean_solver_completion"
  grep -q "exit_reason=clean_solver_completion" "$T/delta_0.1/run_meta.txt" || fail "t1: run_meta.txt missing exit_reason=clean_solver_completion"
  [ "$(grep -c 'launching stage runner' "$T/supervisor.log")" = "1" ] || fail "t1: expected exactly 1 launch (no restart) for a clean run"
  pass "clean_solver_completion classified correctly, single launch, no restart"
}

# ============================================================================
# Test 2: wall_budget_exhausted -- fake julia writes a checkpoint then runs
# forever; the stage's own wall budget (shorter than the stall threshold) must
# fire first, terminate the process, and classify as wall_budget_exhausted with
# a checkpoint returned -- NOT as unexpected_process_failure.
# ============================================================================
t2() {
  local T="$TMPDIR_TEST/t2"; mkdir -p "$T"
  local julia="$T/fake_julia.sh"
  cat > "$julia" <<'STUB'
#!/usr/bin/env bash
stage_dir="$3"
: > "$stage_dir/stage_latest.jls"
echo "fake julia: checkpoint written, now running past the wall budget"
sleep 60
echo "STAGE_DONE"
exit 0
STUB
  chmod +x "$julia"

  # STALL_THRESHOLD_S (20s) deliberately longer than STAGE_WALL_S (4s) so the wall-budget
  # path fires first and the stall path never triggers -- isolates the two classifications.
  JULIA_BIN="$julia" STAGE_WALL_S=4 POLL_INTERVAL_S=1 STALL_THRESHOLD_S=20 GRACE_TERM_S=1 \
    bash -c '
      source "'"$SCRIPT_DIR"'/cm_production_supervisor.sh"
      CHAIN_ID=t2; RESTART_LOG="'"$T"'/restarts.log"; SUPERVISOR_LOG="'"$T"'/supervisor.log"; COMMIT=x
      stage_dir="'"$T"'/delta_0.1"
      out=$(run_stage_with_watchdog "0.1" "$stage_dir" "calibration" "" "0")
      status=$?
      echo "STATUS=$status OUT=$out" > "'"$T"'/result.txt"
    '
  local status out
  status=$(grep -o 'STATUS=[0-9]*' "$T/result.txt" | cut -d= -f2)
  out=$(sed -e 's/^STATUS=[0-9]* OUT=//' "$T/result.txt")
  [ "$status" = "0" ] || fail "t2: expected status 0 (checkpoint exists, not a failure), got $status"
  [ "$out" = "$T/delta_0.1/stage_latest.jls" ] || fail "t2: unexpected checkpoint path: $out"
  grep -q "exit_reason=wall_budget_exhausted" "$T/supervisor.log" || fail "t2: supervisor.log missing exit_reason=wall_budget_exhausted"
  grep -q "reason=wall_budget_exhausted" "$T/restarts.log" || fail "t2: restarts.log missing reason=wall_budget_exhausted"
  ! grep -q "unexpected_process_failure" "$T/supervisor.log" || fail "t2: misclassified as unexpected_process_failure"
  [ "$(grep -c 'launching stage runner' "$T/supervisor.log")" = "1" ] || fail "t2: wall-budget exhaustion must NOT restart the stage"
  pass "wall_budget_exhausted classified correctly (not restarted, not treated as failure)"
}

# ============================================================================
# Test 3: stale STAGE_DONE offset guard -- seed the stage's log with a STAGE_DONE
# line BEFORE launching a fresh attempt that stalls (never legitimately finishes).
# Without the byte-offset fix, the stale sentinel from a prior attempt would make
# this stalling attempt look like it completed cleanly.
# ============================================================================
t3() {
  local T="$TMPDIR_TEST/t3"; mkdir -p "$T/delta_0.1"
  # Simulate a stale sentinel left behind by an earlier (unrelated/failed) attempt at this
  # same stage_dir, present in the log file BEFORE this test's own attempt is launched.
  echo "STAGE_DONE" > "$T/delta_0.1/stage.log"
  local julia="$T/fake_julia.sh"
  cat > "$julia" <<'STUB'
#!/usr/bin/env bash
stage_dir="$3"
: > "$stage_dir/stage_latest.jls"
echo "fake julia: this attempt will stall/run past budget, never prints its own STAGE_DONE"
sleep 60
STUB
  chmod +x "$julia"

  # Short wall budget bounds total test runtime; the real assertion is about
  # classification, not about which specific terminal state (stall-restart-loop vs
  # eventual wall-budget exhaustion) is reached first.
  JULIA_BIN="$julia" STAGE_WALL_S=8 POLL_INTERVAL_S=1 STALL_THRESHOLD_S=3 GRACE_TERM_S=1 \
    bash -c '
      source "'"$SCRIPT_DIR"'/cm_production_supervisor.sh"
      CHAIN_ID=t3; RESTART_LOG="'"$T"'/restarts.log"; SUPERVISOR_LOG="'"$T"'/supervisor.log"; COMMIT=x
      stage_dir="'"$T"'/delta_0.1"
      out=$(run_stage_with_watchdog "0.1" "$stage_dir" "calibration" "" "0")
      status=$?
      echo "STATUS=$status OUT=$out" > "'"$T"'/result.txt"
    '
  grep -q "exit_reason=clean_solver_completion" "$T/supervisor.log" && fail "t3: stale STAGE_DONE from a prior attempt was incorrectly treated as this attempt's own completion"
  grep -q "PROBABLE STALL" "$T/supervisor.log" || fail "t3: expected a probable-stall detection (the seeded stale sentinel must not short-circuit stall detection)"
  pass "stale STAGE_DONE sentinel from a prior attempt correctly ignored (byte-offset scoping works)"
}

# ============================================================================
# Test 4: nonempty campaign directory guard -- main() must refuse to launch into
# an already-nonempty CKPT_ROOT unless RESUME_CAMPAIGN=1.
# ============================================================================
t4() {
  local T="$TMPDIR_TEST/t4"; mkdir -p "$T/chain"
  echo "leftover from a prior run" > "$T/chain/supervisor.log"

  local julia="$T/fake_julia.sh"
  cat > "$julia" <<'STUB'
#!/usr/bin/env bash
echo "should not be reached"
exit 1
STUB
  chmod +x "$julia"

  if JULIA_BIN="$julia" bash -c 'source "'"$SCRIPT_DIR"'/cm_production_supervisor.sh"; main t4 "'"$T"'/chain"' 2>"$T/stderr.log"; then
    fail "t4: main() should have refused to launch into a nonempty campaign dir without RESUME_CAMPAIGN=1"
  fi
  grep -qi "refusing" "$T/stderr.log" || fail "t4: expected an explicit refusal message on stderr"
  pass "refuses to launch into a nonempty campaign directory without RESUME_CAMPAIGN=1"
}

t1
t2
t3
t4

echo ""
echo "ALL $PASS_COUNT STATE-MACHINE TESTS PASSED"
exit 0
