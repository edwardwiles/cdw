#!/bin/bash
# paper_upper_v1 Phase-I wave launcher.
#
# Launches exactly 2 starts x 5 families = 10 simultaneous family_start_chain.jl processes, each
# pinned to its own disjoint 10-physical-core taskset range, each with JULIA_NUM_THREADS=10 and
# OPENBLAS_NUM_THREADS=1 (avoid oversubscription -- 10 processes x 10 Julia threads = 100 cores,
# BLAS stays single-threaded per the project's own hard-cap rule). Blocks until ALL 10 processes
# in this wave have exited before returning (protocol addendum: never begin wave j+1 before wave
# j's both starts have completed their full five-family, four-delta discovery chains -- even if
# one start finishes first, its five slots sit idle rather than starting the next wave early).
#
# Usage:
#   ./launch_wave.sh <protocol_toml> <campaign_root> <start_id_1> <start_id_2> [first_core_offset]
#
# Example (wave 1, S0 and S1, cores 0-99):
#   ./launch_wave.sh protocols/paper_upper_v1.toml /bbkinghome/edav/repo_scratch/paper_upper_v1 S0 S1 0

set -euo pipefail

PROTOCOL_TOML="$1"
CAMPAIGN_ROOT="$2"
START_A="$3"
START_B="$4"
CORE_OFFSET="${5:-0}"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAMILIES=(UNRESTRICTED COMMON_MARGINALS COMMON_FRECHET ORIGIN_ZC CM_PLUS_ZC)

export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=10
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH="/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$CAMPAIGN_ROOT/phase1_discovery" "$CAMPAIGN_ROOT/logs"

echo "=== Wave launch: starts=($START_A,$START_B) core_offset=$CORE_OFFSET $(date -u +%FT%TZ) ==="

PIDS=()
SLOT=0
for START_ID in "$START_A" "$START_B"; do
    for FAMILY_ID in "${FAMILIES[@]}"; do
        LO=$((CORE_OFFSET + SLOT * 10))
        HI=$((LO + 9))
        LOG="$CAMPAIGN_ROOT/logs/${START_ID}_${FAMILY_ID}.log"
        echo "  launching $START_ID/$FAMILY_ID on cores $LO-$HI -> $LOG"
        taskset --cpu-list "$LO-$HI" julia --project="$SRC_DIR" -t 10 \
            "$SRC_DIR/paper_upper_v1_orchestrator/family_start_chain.jl" \
            "$PROTOCOL_TOML" "$FAMILY_ID" "$START_ID" "$CAMPAIGN_ROOT" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
        SLOT=$((SLOT + 1))
    done
done

echo "  wave PIDs: ${PIDS[*]}"
echo "  waiting for all ${#PIDS[@]} chains to complete..."

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "  !!! process $PID exited non-zero"
        FAILED=1
    fi
done

echo "=== Wave complete: starts=($START_A,$START_B) failed=$FAILED $(date -u +%FT%TZ) ==="
exit $FAILED
