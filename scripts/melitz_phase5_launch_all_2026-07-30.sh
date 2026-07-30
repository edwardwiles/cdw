#!/usr/bin/env bash
# Phase 5 (MANDATORY): launches melitz_phase5_thread_scaling_2026-07-30.jl as a separate clean
# Julia process for each T in {1,2,4,5,8,10,20}, BLAS threads pinned to 1 in every process
# (env vars below), writing one CSV per thread count
# (docs/key_results/melitz_phase5_thread_scaling_T<N>_2026-07-30.csv).
#
# Usage: bash scripts/melitz_phase5_launch_all_2026-07-30.sh [logdir]
set -euo pipefail
cd "$(dirname "$0")/.."
source .knitro_env.sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
LOGDIR="${1:-/tmp/melitz_phase5_logs}"
mkdir -p "$LOGDIR"

for T in 1 2 4 5 8 10 20; do
    echo "=== Launching T=$T ==="
    julia --project=. -t "$T" scripts/melitz_phase5_thread_scaling_2026-07-30.jl \
        > "$LOGDIR/phase5_T${T}.log" 2>&1
    echo "=== T=$T exit code $? ==="
done

echo "All thread counts complete. Logs in $LOGDIR"
