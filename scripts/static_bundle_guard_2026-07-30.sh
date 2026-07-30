#!/usr/bin/env bash
# architecture/production-operator-bundle-hardening-2026-07-30, task §13: static repository guard.
#
# Rejects forbidden production-module patterns anywhere in full_aod_diag/d4_exact/ EXCEPT an
# explicit allowlist of diagnostic/test files. This supplements (does not replace) the runtime/
# type-level safeguards in production_bundle_api.jl/dense_reference_diagnostics.jl -- a file can
# pass this scan and still be wrong if it dodges the string patterns below; the fatal live
# assertion (assert_production_operator_bundle!) is the actual enforcement mechanism. This script
# catches the class of regression the postmortem describes BEFORE a review even has to read the
# diff closely: a hardcoded `moment_representation = :dense_reference` default, a direct dense
# bundle construction, or a static "bundle_type=OperatorPsiBundle" print re-appearing outside an
# allowlisted diagnostic/test file.
#
# Usage: scripts/static_bundle_guard_2026-07-30.sh   (run from repo root or anywhere; exits 1 on
# any violation, prints every offending file:line)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D4E="$REPO_ROOT/full_aod_diag/d4_exact"
cd "$D4E" || exit 2

# Files allowed to contain the forbidden patterns -- diagnostic/test-only by construction (task
# §13's own "allowlist only explicit test and diagnostic paths"). Glob patterns, matched against
# the basename.
ALLOWLIST_GLOBS=(
  "test_*.jl"
  "dense_reference_diagnostics.jl"          # the ONE sanctioned dense-construction path
  "operator_psi_bundle.jl"                  # defines select_G_from_H(::OperatorPsiBundle,...) -- the fail-fast method itself, not a use
  "production_bundle_api.jl"                # defines the try/catch structural check that CALLS select_G_from_H to prove it throws -- not a production use
  "no_dense_g_counters.jl"                  # defines the counters, references the concept in docstrings only
  "postmerge_smoke_diagnostics.jl"          # explicitly diagnostic (its own header says so); superseded by the live manifest, cleanup candidate per task §15/§17
  "confirm_frechet_operator_default_flip_2026-07-29.jl"  # historical investigation script
  "*bench*.jl" "*benchmark*.jl" "*profile*.jl" "*diag*.jl" "*investigate*.jl" "c8_*.jl" "c9_*.jl" "c10_*.jl" "c12*.jl" "c13_*.jl" "c14_*.jl" "c30_*.jl" "c33_*.jl"
  "cm_hessian_subblock_profiling.jl" "cross_hessian_live_stash_2026-07-28.jl"
  "harmonization_d4_equivalence_gate_2026-07-29.jl"   # equivalence-test-shaped script, not named test_*
  "check_flexcm_d20_verification.jl"                  # "Ad hoc check" (own header) comparing both bundles at one point, not named test_*
  # --- audited 2026-07-30, DENSE_REFERENCE_REACHABILITY_AUDIT_2026-07-30.md Finding 2 ---
  # The 5 low-level family builders/converters still support BOTH moment_representation values --
  # deliberately NOT ripped out, because ~10 retained equivalence tests
  # (test_operator_no_H_bundle_equivalence_*.jl) call them directly with both values as their whole
  # purpose. The production invariant is enforced one layer up, at prepare_production_run/the 3
  # real drivers (which no longer accept or forward any representation choice at all -- verified by
  # test_all_family_real_production_entrypoints_operator_bundle.jl) -- not by making these builders
  # single-mode. Allowlisted here for that reason, not because the pattern is absent.
  "cm_production_bundle.jl" "cm_frechet_level.jl" "cm_meanzc_production.jl" "cm_originzc_production.jl" "compressed_live.jl"
  "cm_meanzc_moments.jl" "cm_originzc_moments.jl"     # the SAME builders' own moments!-closure helpers, same rationale
  # --- select_G_from_H: ~30 pre-existing call sites in genuinely still-live DENSE-mode production
  # code (the :dense_reference/:compressed evaluation paths this task does not remove -- a
  # different, still-supported axis, see DENSE_REFERENCE_REACHABILITY_AUDIT_2026-07-30.md Finding
  # 3). None are reachable when ctx.obj is OperatorPsiBundle: OperatorPsiBundle's own
  # select_G_from_H method (operator_psi_bundle.jl) throws unconditionally, so these call sites can
  # only ever fire on a genuine PsiObjectiveBundleImplicit -- i.e. inside an explicit diagnostic/
  # comparison run or a family/mode this task's 5 families' production path never selects. Audited,
  # not newly introduced; allowlisted rather than deleted (deleting live dense-mode support that
  # other code paths still depend on is out of this task's scope).
  "chunked_hessian.jl" "cm_frechet_cplus.jl" "cm_frechet_lookup_production.jl" "cm_hessian_architectures.jl"
  "cm_lookup_live_knitro.jl" "cm_lookup_production.jl" "cm_meanzc_lookup_production.jl" "cm_originzc_lookup_production.jl"
  "d20_originzc_fixedpoint_gates.jl" "debug_frechet_lookup_unit.jl" "fast_range_screen.jl" "flexible_theta.jl"
  "flexible_theta_aspace_production.jl" "infeasibility_screen.jl" "operator_verification.jl" "oracle_fast.jl"
  "audit_jach.jl" "common_marginals_interval.jl" "common_marginals_moments.jl" "context.jl" "context_real_d20.jl"
  "context_scaled.jl" "qmc_context_real_d20.jl" "restricted_workspace_d20_correctness.jl" "smoothed_consistent.jl"
  "c12b_interval_common_marginals_moments.jl"
  # --- comment-only historical references to the postmortem's own now-fixed false claim, not a
  # live print (all 4 remaining hits, checked by hand 2026-07-30) ---
  "campaign_unrestricted_runner.jl" "production_backend_manifest.jl" "smoke_delta1_unrestricted.jl"
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
  # Plain POSIX grep, not rg -- must run standalone (CI/plain shell) where rg may not be on PATH.
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

echo "=== static_bundle_guard_2026-07-30.sh: scanning $D4E ==="

check_pattern "hardcoded-dense-default"  'moment_representation\s*=\s*:dense_reference'
check_pattern "direct-dense-construction" 'PsiObjectiveBundleImplicit\('
check_pattern "select-g-from-h-use"       'select_G_from_H\('
check_pattern "static-bundle-type-claim"  'bundle_type\s*=\s*OperatorPsiBundle'
check_pattern "driver-representation-kwarg" 'moment_representation\s*::\s*(Symbol|Union\{Nothing,\s*Symbol\})'

echo "=== $VIOLATIONS violation(s) ==="
if [[ $VIOLATIONS -gt 0 ]]; then
  echo "FAIL: forbidden production-module pattern(s) found outside the allowlist."
  exit 1
fi
echo "PASS: no forbidden patterns found outside the allowlist."
exit 0
