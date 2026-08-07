#!/usr/bin/env bash
# fix/fullA-lower-limit-and-hotpath-2026-08-06, task §14: static repository guard.
#
# Rejects the reintroduction of a hardcoded lower_limit=-50 (or any get(...,-50) silent-default
# pattern) or the dead ThresholdAbortState/resolve_threshold_for_delta early-abort mechanism
# anywhere in full_aod_diag/d4_exact/ EXCEPT an explicit allowlist of diagnostic/test/legacy
# files. Mirrors scripts/static_bundle_guard_2026-07-30.sh's own convention exactly (same
# allowlist-glob mechanism, same plain-POSIX-grep requirement for CI/plain-shell portability).
#
# The production invariant this guards: inner_lower_limit is a REQUIRED kwarg (no default) on
# d20_real_setup/d20_real_setup_design/run_cm_upper_checkpointed/run_originzc_upper_checkpointed/
# run_polish_checkpointed_unified -- see docs/audits/fullA-lower-limit-and-hotpath-2026-08-06/
# MASTER.md and LOWER_LIMIT_SOURCE_AUDIT.csv for the full audit this allowlist is derived from.
#
# Usage: scripts/static_lower_limit_guard_2026-08-06.sh   (exits 1 on any violation)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D4E="$REPO_ROOT/full_aod_diag/d4_exact"
cd "$D4E" || exit 2

# Files allowed to contain the forbidden patterns -- diagnostic/test-only by construction, or the
# module that legitimately DEFINES the (now dead-on-the-production-path) mechanism. Glob patterns,
# matched against the basename. Derived from LOWER_LIMIT_SOURCE_AUDIT.csv's own "action" column --
# every entry here was individually confirmed NOT reachable from run_cm_upper_checkpointed/
# run_originzc_upper_checkpointed/run_polish_checkpointed_unified.
ALLOWLIST_GLOBS=(
  "test_*.jl"                                  # explicit diagnostic tests may pass -50/construct
                                                 # ThresholdAbortState directly to exercise the
                                                 # mechanism (task §5's own "diagnostic tests may
                                                 # explicitly pass another value" carve-out)
  "context.jl"                                  # d4_exact_setup -- synthetic D4 diagnostic economy,
                                                 # explicitly documented (context_real_d20.jl's own
                                                 # header) as separate from the real D20 production
                                                 # path; NOT touched by this task
  "context_scaled.jl"                           # d_exact_setup_scaled -- same rationale
  "smoothed_consistent.jl"                      # diagnostic-only bundle construction
  "*bench*.jl" "*benchmark*.jl" "*profile*.jl" "*diag*.jl" "*investigate*.jl" "*audit*.jl"
  "compare_*.jl" "solve_*.jl" "verify_*.jl" "validate_*.jl" "c8_*.jl" "c9_*.jl" "c10_*.jl" "c12*.jl"
  "c13_*.jl" "c14_*.jl" "c30_*.jl" "c33_*.jl" "run_fullA_*.jl"
  "cm_hessian_subblock_profiling.jl"
)

is_allowlisted() {
  local base
  base=$(basename "$1")
  for pat in "${ALLOWLIST_GLOBS[@]}"; do
    # shellcheck disable=SC2053
    [[ "$base" == $pat ]] && return 0
  done
  return 1
}

VIOLATIONS=0

check_pattern() {
  local label="$1" pattern="$2"
  local hits
  hits=$(grep -rnE "$pattern" --include='*.jl' . 2>/dev/null) || true
  [[ -z "$hits" ]] && return 0
  while IFS= read -r line; do
    local file="${line%%:*}"
    is_allowlisted "$file" && continue
    echo "VIOLATION [$label]: $line"
    VIOLATIONS=$((VIOLATIONS + 1))
  done <<< "$hits"
}

echo "=== static_lower_limit_guard_2026-08-06.sh: scanning $D4E ==="

check_pattern "hardcoded-lower-limit-neg50"     'lower_limit[[:space:]]*=[[:space:]]*-50(\.0)?[^0-9]'
check_pattern "silent-default-get-lower-limit"  'get\([^)]*lower_limit[^)]*-50'
check_pattern "threshold-abort-state-live-construct" 'ThresholdAbortState\([^)]+\)'
# NOTE: bare `ThresholdAbortState()` (empty parens) is the SAFE inert default (threshold=Inf) and
# is deliberately NOT flagged -- it appears legitimately as a struct field default
# (cc_algo/PsiObjectiveBundle.jl / operator_psi_bundle.jl's own `threshold_state = ThresholdAbortState()`)
# and in this task's own explanatory comments. Only a construction with a real argument (a live,
# non-Inf threshold) is the dangerous pattern this guard exists to catch.
check_pattern "resolve-threshold-for-delta-call" 'resolve_threshold_for_delta\('
check_pattern "hidden-default-inner-lower-limit" 'inner_lower_limit[[:space:]]*::[[:space:]]*Float64[[:space:]]*='

echo "=== $VIOLATIONS violation(s) ==="
if [[ $VIOLATIONS -gt 0 ]]; then
  echo "FAIL: forbidden lower_limit/threshold-abort pattern(s) found outside the allowlist."
  exit 1
fi
echo "PASS: no forbidden patterns found outside the allowlist."
exit 0
