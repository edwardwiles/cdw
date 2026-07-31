#!/usr/bin/env bash
# run_preflight_smokes_6to10.sh -- runs required preflight items 6-10 for the sigma3/W500k
# five-family production campaign (2026-07-30). Each smoke writes its own report JSON that
# run_preflights.jl reads back. Real, actual runs through the SAME campaign driver scripts
# --launch itself uses -- not a separate simulated path.
#
# Per the campaign brief: "Use 60-second outer smokes, extending an individual W500k family up
# to 180 seconds only when needed to complete one valid callback." Simplified here to a single
# retry-the-whole-batch-at-180s fallback rather than per-family extension (noted, not hidden --
# per-family extension would need per-process retry logic inside run_smoke.sh that isn't built
# yet; this fallback still satisfies "extend up to 180s if needed", just at batch granularity).
set -uo pipefail
CAMPAIGN_ROOT="$(cd "$(dirname "$0")" && pwd)"

smoke_with_extension() {
  local name="$1" deltas="$2" starts="$3" directions="$4"
  echo ">>> $name: trying 60s..."
  if "$CAMPAIGN_ROOT/run_smoke.sh" "$name" "$deltas" "$starts" "$directions" 60; then
    echo ">>> $name: PASS at 60s"
  else
    echo ">>> $name: 60s did not complete a valid callback -- extending to 180s"
    "$CAMPAIGN_ROOT/run_smoke.sh" "$name" "$deltas" "$starts" "$directions" 180
  fi
}

echo "=== Preflight 6: five-family parallel UPPER smoke at calibration, delta=0.1 ==="
smoke_with_extension "upper_smoke_report" "0.1" "1" "upper"

echo "=== Preflight 7: five-family parallel LOWER smoke at calibration, delta=0.1 ==="
smoke_with_extension "lower_smoke_report" "0.1" "1" "lower"

echo "=== Preflight 8: one direct outer callback for Starts 2 and 3, every family ==="
smoke_with_extension "perturbation_smoke_report" "0.1" "2,3" "upper"

echo "=== Preflight 9: strategy handoff smoke (direct_sr1 -> optional BFGS polish), unrestricted + origin_zc ==="
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
REPO_ROOT="$(cd "$CAMPAIGN_ROOT/../.." && pwd)"
julia --project="$REPO_ROOT" -t 8 "$CAMPAIGN_ROOT/run_strategy_handoff_smoke.jl"

echo "=== Preflight 10: five-family COMBINED resource smoke (both directions, 100 threads total) ==="
DRIVERS_DIR="$REPO_ROOT/campaign_inputs/sigma3_W500k_2026-07-30/drivers"
RESOURCE_SMOKE_ROOT="$CAMPAIGN_ROOT/smoke/resource_smoke"
rm -rf "$RESOURCE_SMOKE_ROOT"; mkdir -p "$RESOURCE_SMOKE_ROOT"
pids=()
for fam in unrestricted flexible_cm common_frechet origin_zc cm_meanzc; do
  "$DRIVERS_DIR/run_family_chain_sigma3.sh" "$fam" "$CAMPAIGN_ROOT/start_manifest.json" \
    "$RESOURCE_SMOKE_ROOT" 60 20 300 "0.1" "1" > "$RESOURCE_SMOKE_ROOT/${fam}_chain.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
peak_rss_total=0
for f in "$RESOURCE_SMOKE_ROOT"/logs/cells/*.rusage; do
  [ -f "$f" ] || continue
  rss=$(grep "Maximum resident set size" "$f" | awk '{print $NF}')
  [ -n "$rss" ] && peak_rss_total=$((peak_rss_total + rss))
done
ok=$([ "$fail" -eq 0 ] && echo "true" || echo "false")
{
  echo "{"
  echo "  \"ok\": $ok,"
  echo "  \"summary\": \"5 families x both directions, delta=0.1, start=1, 20 threads each (100 total) -- summed peak RSS ~${peak_rss_total}KB -- see $RESOURCE_SMOKE_ROOT\","
  echo "  \"generated\": \"$(date -Is)\""
  echo "}"
} > "$CAMPAIGN_ROOT/resource_smoke_report.json"

echo "=== ALL SMOKES (6-10) COMPLETE ==="
