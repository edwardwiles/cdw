#!/usr/bin/env bash
# Phase II.12 thread-count sweep: launches scripts/melitz_parallel_gradient_benchmark.jl
# once per JULIA_NUM_THREADS value (a single Julia process cannot change Threads.nthreads()
# at runtime). OPENBLAS_NUM_THREADS/OMP_NUM_THREADS stay pinned at 1 for every launch
# (this repo's standing hard-cap policy) -- Threads.nthreads() is the ONLY thing varied.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.juliaup/bin:$PATH"

THREAD_COUNTS=(1 2 4 8 16 20 30 208)
LOG_DIR="${1:-/tmp/melitz_parallel_sweep}"
mkdir -p "$LOG_DIR"

for nt in "${THREAD_COUNTS[@]}"; do
    echo "=== JULIA_NUM_THREADS=${nt} ==="
    JULIA_NUM_THREADS="$nt" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        julia --project=. scripts/melitz_parallel_gradient_benchmark.jl \
        > "$LOG_DIR/nt_${nt}.log" 2>&1
    grep "THREADCOUNT_RESULT" "$LOG_DIR/nt_${nt}.log" || echo "  (no result line -- check $LOG_DIR/nt_${nt}.log)"
done

echo
echo "=== summary ==="
grep -h "THREADCOUNT_RESULT" "$LOG_DIR"/nt_*.log | sort -t= -k2 -n
