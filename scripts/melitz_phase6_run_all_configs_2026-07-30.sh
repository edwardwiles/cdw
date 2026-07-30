#!/usr/bin/env bash
# Phase 6: runs all six fixed-total-20-core process/thread allocations SEQUENTIALLY (so each
# config's own wall-clock throughput measurement is not contaminated by another config
# competing for the same cores) against the SAME fixed job batch.
set -euo pipefail
cd "$(dirname "$0")/.."

for cfg in "1 20" "2 10" "4 5" "5 4" "10 2" "20 1"; do
    read -r NPROC TPP <<< "$cfg"
    LABEL="${NPROC}x${TPP}"
    echo ""
    echo "########################################################"
    echo "# Phase 6 config: $LABEL ($NPROC processes x $TPP threads)"
    echo "########################################################"
    bash scripts/melitz_phase6_launch_config_2026-07-30.sh "$NPROC" "$TPP" "$LABEL"
done

echo ""
echo "ALL PHASE 6 CONFIGS COMPLETE"
