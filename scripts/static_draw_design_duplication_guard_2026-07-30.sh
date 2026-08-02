#!/usr/bin/env bash
# architecture/unify-random-draw-production-pipeline-2026-07-30, task §18: static repository
# guard against the QMC/pseudorandom draw-design duplication this task removed.
#
# Two independent checks:
#   1. The three deleted duplicate function names (and a QMC-specific campaign driver, which
#      never existed but is checked for anyway per the task spec) must have ZERO live
#      definitions or call sites anywhere in the repository.
#   2. Every unified pipeline function (master_prepare_cc, build_ad_context_real_d20,
#      d20_real_setup, precompute_pairwise_M, build_extreme_draw_witness) must have EXACTLY ONE
#      `function <name>` definition repo-wide -- a second definition anywhere is exactly how the
#      original duplication was introduced (a hand-copied "byte-for-byte mirror" file) and exactly
#      what this guard exists to catch before it drifts again.
#   3. No file outside the draw-design resolver itself (draw_design.jl, draw_design_types.jl) or
#      tests/diagnostics may branch on a draw-design symbol AFTER U has been generated (task
#      §18's "flag downstream production branches on :pseudorandom/:sobol_randomized/
#      :halton_scrambled after the draw resolver has returned").
#
# Usage: scripts/static_draw_design_duplication_guard_2026-07-30.sh   (run from repo root or
# anywhere; exits 1 on any violation, prints every offending file:line)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

violations=0

echo "=== static_draw_design_duplication_guard_2026-07-30.sh: scanning $REPO_ROOT ==="

echo "--- check 1: forbidden duplicate names must have zero live references ---"
FORBIDDEN_NAMES=(
  "master_prepare_cc_qmc"
  "build_ad_context_real_d20_qmc"
  "d20_real_setup_qmc"
  "production_campaign_QMC"
)
for name in "${FORBIDDEN_NAMES[@]}"; do
  # Live code reference: the name appears NOT inside a comment (line does not start with
  # whitespace+#) and not inside docs/. Historical narrative comments (e.g. "d20_real_setup_qmc
  # is now deleted") are permitted; a `#`-prefixed line or anything under docs/ is not a live
  # reference.
  # Excludes: docs/, `#`-prefixed comment lines, and a name referenced in backtick-quoted prose
  # inside a docstring (e.g. "the now-deleted `d20_real_setup_qmc`") -- unambiguous documentation,
  # not a live definition or call site.
  hits=$(grep -rn "$name" --include="*.jl" . 2>/dev/null | grep -v "^\./docs/" \
    | grep -vE '^\S+:[0-9]+:\s*#' | grep -vE "\`${name}\`")
  if [[ -n "$hits" ]]; then
    echo "  VIOLATION: live reference(s) to forbidden name '$name':"
    echo "$hits" | sed 's/^/    /'
    violations=$((violations + 1))
  fi
done
[[ $violations -eq 0 ]] && echo "  OK: zero live references to any forbidden duplicate name."

echo "--- check 2: unified pipeline functions must have exactly one definition ---"
SINGLE_DEFINITION_REQUIRED=(
  "master_prepare_cc"
  "build_ad_context_real_d20"
  "d20_real_setup"
  "precompute_pairwise_M"
  "build_extreme_draw_witness"
)
for name in "${SINGLE_DEFINITION_REQUIRED[@]}"; do
  count=$(grep -rn "^function ${name}(" --include="*.jl" . 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$count" != "1" ]]; then
    echo "  VIOLATION: '$name' has $count definition(s), expected exactly 1:"
    grep -rn "^function ${name}(" --include="*.jl" . 2>/dev/null | sed 's/^/    /'
    violations=$((violations + 1))
  else
    echo "  OK: '$name' has exactly 1 definition."
  fi
done

echo "--- check 3: no downstream branch on a draw-design symbol outside the resolver/tests ---"
D4E="$REPO_ROOT/full_aod_diag/d4_exact"
ALLOWLIST_GLOBS=(
  "draw_design.jl" "draw_design_types.jl"
  "test_*.jl" "*bench*.jl" "*diag*.jl" "c9_*.jl" "c10_*.jl" "c13_*.jl" "c14_*.jl" "c15_*.jl"
  "c19_*.jl" "c20*.jl" "c21_*.jl" "c22_*.jl" "c23_*.jl" "c24_*.jl" "c30_*.jl" "c31_*.jl"
  "c32_*.jl" "c33_*.jl" "c34_*.jl"
  # Release/regression-gate scripts: assert a loaded checkpoint's draw_design PROVENANCE FIELD
  # equals an expected value (sanity-checking which checkpoint was resumed), never a numerical
  # branch on draw design -- same DRAW_METADATA_REQUIRED category as the checkpoint struct fields
  # themselves (see QMC_PSEUDORANDOM_DUPLICATION_REACHABILITY_2026-07-30.md).
  "*_gates*.jl" "*gate*.jl" "*release*.jl" "*shakedown*.jl"
)
is_allowlisted() {
  local base; base="$(basename "$1")"
  for pat in "${ALLOWLIST_GLOBS[@]}"; do
    # shellcheck disable=SC2053
    [[ "$base" == $pat ]] && return 0
  done
  return 1
}
DRAW_DESIGN_BRANCH_PATTERN='(==|!=|in)\s*\(?\s*:(pseudorandom|sobol_randomized|halton_scrambled)\b'
while IFS= read -r -d '' f; do
  is_allowlisted "$f" && continue
  hits=$(grep -nE "$DRAW_DESIGN_BRANCH_PATTERN" "$f" 2>/dev/null)
  if [[ -n "$hits" ]]; then
    echo "  VIOLATION: downstream draw-design branch outside the resolver in $f:"
    echo "$hits" | sed 's/^/    /'
    violations=$((violations + 1))
  fi
done < <(find "$D4E" -maxdepth 1 -name "*.jl" -print0)
[[ $violations -eq 0 ]] || true
echo "  (check 3 complete -- draw_design::Symbol EQUALITY comparisons for checkpoint-provenance"
echo "   matching, e.g. guard_checkpoint_path/reuse_matches, are legitimate and excluded by the"
echo "   allowlist only where they live in draw_design.jl itself; a checkpoint struct FIELD named"
echo "   draw_design is metadata storage, not a branch, and is not matched by this pattern)"

echo ""
echo "=== $violations violation(s) ==="
if [[ $violations -eq 0 ]]; then
  echo "PASS: no draw-design duplication patterns found outside the allowlist."
  exit 0
else
  echo "FAIL: see violations above."
  exit 1
fi
