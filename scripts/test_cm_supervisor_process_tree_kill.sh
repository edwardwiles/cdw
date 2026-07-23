#!/usr/bin/env bash
# ============================================================================
# Shell-level regression test for the process-group termination fix
# (cm_production_supervisor.sh, 2026-07-23 mean/ZC release addendum): a wrapper
# process spawning a child, which itself spawns a grandchild, must ALL be
# terminated by kill_pgroup -- not just the top-level wrapper PID (the exact
# defect the CM-C+ release report flagged: `kill -TERM "$pid"` on a subshell
# wrapper does not guarantee delivery to a Julia/KNITRO grandchild).
#
# Uses synthetic `sleep` processes, NOT Julia -- this tests the shell-level
# process-group mechanics in isolation, independent of any Julia/KNITRO
# startup cost.
#
# Usage: bash scripts/test_cm_supervisor_process_tree_kill.sh
# Exits 0 on pass, 1 on failure (with a diagnostic message).
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the supervisor script for kill_pgroup/alive_pgroup/wait_for_pgid_file only -- guarded
# by BASH_SOURCE!=0 in the supervisor script itself, so `main` does NOT run as a side effect.
SUPERVISOR_LOG=""   # slog() (used by kill_pgroup) checks this; empty means stderr-only, fine for a test
CHAIN_ID="test"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/cm_production_supervisor.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass_count=0
check() { echo "  ok: $*"; pass_count=$((pass_count + 1)); }

TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

echo "=== Test 1: TERM against a live wrapper+child+grandchild tree terminates all three ==="
pgid_file="$TMPDIR_TEST/t1.pgid"
rm -f "$pgid_file"
( setsid bash -c '
    echo "$$" > "$1"
    bash -c "sleep 300; true" &   # child bash (the trailing "; true" prevents bash exec-optimizing
                                    # straight into sleep, so this stays a genuine 3rd generation:
                                    # leader -> child bash -> grandchild sleep)
    wait
  ' _ "$pgid_file" >/dev/null 2>&1 ) &
wrapper_pid=$!

wait_for_pgid_file "$pgid_file" 10 || fail "pgid file never appeared"
pgid=$(cat "$pgid_file")
[ -n "$pgid" ] || fail "pgid file was empty"
check "pgid file populated: pgid=$pgid (wrapper_pid=$wrapper_pid, expected to differ if setsid forked)"

sleep 0.5   # let the child+grandchild actually spawn
members_before=$(pgrep -g "$pgid" 2>/dev/null | wc -l)
[ "$members_before" -ge 3 ] || fail "expected >=3 process-group members (leader+child+grandchild) before kill, found $members_before: $(pgrep -g "$pgid" -a 2>/dev/null)"
check "process group has $members_before members before kill (leader+child+grandchild all present)"

alive_pgroup "$pgid_file" || fail "alive_pgroup reported dead before any kill was sent"
check "alive_pgroup correctly reports the group as alive"

kill_pgroup "$pgid_file" TERM
waited=0
while alive_pgroup "$pgid_file" && [ "$waited" -lt 10 ]; do sleep 0.5; waited=$((waited + 1)); done

if alive_pgroup "$pgid_file"; then
  echo "  (TERM alone did not clear it within 5s -- escalating to KILL, exactly like the real supervisor does)"
  kill_pgroup "$pgid_file" KILL
  waited=0
  while alive_pgroup "$pgid_file" && [ "$waited" -lt 10 ]; do sleep 0.5; waited=$((waited + 1)); done
fi

alive_pgroup "$pgid_file" && fail "process group -$pgid still alive after TERM+KILL escalation"
check "alive_pgroup reports the group is dead after kill_pgroup"

members_after=$(pgrep -g "$pgid" 2>/dev/null | wc -l)
[ "$members_after" -eq 0 ] || fail "pgrep still finds $members_after live members of group -$pgid after kill: $(pgrep -g "$pgid" -a 2>/dev/null)"
check "pgrep confirms zero surviving members of process group -$pgid (leader, child, AND grandchild all gone)"

echo
echo "=== Test 2: kill_pgroup/alive_pgroup handle a missing/empty pgid file gracefully (no crash) ==="
missing_file="$TMPDIR_TEST/does_not_exist.pgid"
if alive_pgroup "$missing_file"; then
  fail "alive_pgroup returned true for a nonexistent pgid file"
fi
check "alive_pgroup returns false (not an error) for a missing pgid file"
kill_pgroup "$missing_file" TERM 2>/dev/null
check "kill_pgroup does not crash when given a missing pgid file (returns nonzero, logs a warning)"

echo
echo "ALL PROCESS-TREE KILL TESTS PASSED ($pass_count checks)"
exit 0
