#!/usr/bin/env bash
# ============================================================================
# Shell-level regression test: proves that logging inside run_stage_with_watchdog
# CANNOT contaminate the checkpoint-path value returned via command substitution,
# even across a stall-detect-and-restart cycle (which is exactly the scenario that
# used to leak dozens of slog lines into `ckpt_path=$(...)` before the stderr fix).
#
# No Julia/KNITRO involved -- JULIA_BIN is pointed at a fake stub so this runs in
# a few seconds and needs nothing beyond bash.
#
# Usage: scripts/test_cm_production_supervisor_return_path.sh
# Exits 0 and prints PASS on success; exits nonzero with a description on failure.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# ---- fake julia stub ----
# argv (as invoked by run_stage_with_watchdog): --project=. <script.jl> <stage_dir> <delta>
# <remaining> <mode> <seed_arg> <perturb_seed>
FAKE_JULIA="$TMPDIR_TEST/fake_julia.sh"
cat > "$FAKE_JULIA" <<'STUB'
#!/usr/bin/env bash
stage_dir="$3"
mode="$6"
if [ "$mode" = "resume" ]; then
  echo "fake julia: resumed cleanly, finishing"
  echo "STAGE_DONE"
  : > "$stage_dir/stage_latest.jls"
  exit 0
else
  echo "fake julia: starting, wrote an early checkpoint, about to (simulate) hang"
  : > "$stage_dir/stage_latest.jls"
  sleep 30
  echo "STAGE_DONE"
  exit 0
fi
STUB
chmod +x "$FAKE_JULIA"

export JULIA_BIN="$FAKE_JULIA"
export STAGE_WALL_S=120
export POLL_INTERVAL_S=1
export STALL_THRESHOLD_S=3
export GRACE_TERM_S=2

# Source (not execute) the supervisor: this registers slog/run_stage_with_watchdog/
# validate_ckpt_path/main as functions without running main (guarded by the
# BASH_SOURCE==0 check at the bottom of the file), which is exactly the point --
# it lets this test call run_stage_with_watchdog directly.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/cm_production_supervisor.sh"

CHAIN_ID="return_path_test"
CKPT_ROOT="$TMPDIR_TEST/chain"
mkdir -p "$CKPT_ROOT"
RESTART_LOG="$CKPT_ROOT/restarts.log"
SUPERVISOR_LOG="$CKPT_ROOT/supervisor.log"
COMMIT="testcommit"
stage_dir="$CKPT_ROOT/delta_1.0"

out="$(run_stage_with_watchdog "1.0" "$stage_dir" "calibration" "" "0" 2>"$TMPDIR_TEST/stderr.log")"
status=$?

# ---- sanity: the scenario actually exercised a stall+restart (i.e. this is a real test,
# not a vacuously-passing one) ----
grep -q "PROBABLE STALL" "$SUPERVISOR_LOG" || fail "test scenario never triggered a stall -- test is not exercising the contamination path"
grep -q "reason=probable_stall" "$RESTART_LOG" || fail "restarts.log missing the expected probable_stall record"
[ "$(grep -c 'launching stage runner' "$SUPERVISOR_LOG")" -ge 2 ] || fail "expected at least 2 launch attempts (stall + resume), supervisor.log shows fewer"

# ---- the actual contamination check ----
[ "$status" -eq 0 ] || fail "run_stage_with_watchdog returned nonzero ($status); stderr:\n$(cat "$TMPDIR_TEST/stderr.log")"
[ -n "$out" ] || fail "captured stdout (ckpt_path) is empty"
case "$out" in
  *$'\n'*) fail "captured stdout contains an embedded newline -- log contamination reproduced:\n[$out]" ;;
esac
[ "$out" = "$stage_dir/stage_latest.jls" ] || fail "captured stdout is not exactly the expected checkpoint path. got: [$out]"
case "$out" in
  *"PROBABLE STALL"*|*"launching stage runner"*|*"chain $CHAIN_ID"*)
    fail "captured stdout contains a slog-originated substring -- contamination reproduced: [$out]" ;;
esac

validate_ckpt_path "$out" "$stage_dir" || fail "validate_ckpt_path rejected the returned path"

# ---- stderr, by contrast, SHOULD carry the log lines (proves slog isn't just silenced) ----
grep -q "PROBABLE STALL" "$TMPDIR_TEST/stderr.log" || fail "stderr unexpectedly missing slog output -- slog should log to stderr, not go silent"

echo "PASS: run_stage_with_watchdog's stdout stayed a clean single-line checkpoint path through a real stall+restart cycle ($(grep -c 'launching stage runner' "$SUPERVISOR_LOG") launch attempts, $(wc -l < "$SUPERVISOR_LOG") supervisor.log lines, 0 of which leaked into the return value)."
exit 0
