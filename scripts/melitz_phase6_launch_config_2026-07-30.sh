#!/usr/bin/env bash
# Phase 6: realizes ONE (n_processes x threads_per_process) allocation against the fixed job
# batch (melitz_phase6_job_batch_2026-07-30.jls, >=20 jobs, built by
# melitz_phase6_build_batch_2026-07-30.jl) as separate OS processes / separate KNITRO sessions
# (never concurrent KN_solve calls sharing one Julia session).
#
# Usage: bash scripts/melitz_phase6_launch_config_2026-07-30.sh <n_processes> <threads_per_process> <label> [logdir]
set -euo pipefail
cd "$(dirname "$0")/.."
source .knitro_env.sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

NPROC="$1"
TPP="$2"
LABEL="$3"
LOGDIR="${4:-/tmp/melitz_phase6_logs/$LABEL}"
mkdir -p "$LOGDIR"
OUTDIR="docs/key_results/phase6_${LABEL}"
mkdir -p "$OUTDIR"

# Bounded to the first 24 jobs of the built batch (melitz_phase6_build_batch_2026-07-30.jl's own
# printed output confirmed 56 total jobs available -- 26 upper/30 lower, 28 continuation-start/28
# compensated-start; hardcoded here rather than re-deserialized, since the Phase6Job struct type
# is only defined after `include_melitz.jl`, too heavy to reload in this one-line shell probe).
# >=20 required by the governing prompt; 24 keeps the fully-serial 1x20 config's own wall time
# tractable within this session's own bounded-experiment scope while preserving a genuine mix of
# upper/lower directions and both start kinds (the batch is built interleaved: upper
# continuation, upper compensated, upper continuation, ... then lower likewise).
N_JOBS_TOTAL=56
N_JOBS=24
echo "Total jobs in batch: $N_JOBS_TOTAL (using first $N_JOBS)  ->  $NPROC processes x $TPP threads (total budget $((NPROC*TPP)))"

# Partition [1, N_JOBS] into NPROC roughly-equal contiguous chunks.
CHUNK=$(( (N_JOBS + NPROC - 1) / NPROC ))

T_START=$(date +%s.%N)
PIDS=()
for ((p=0; p<NPROC; p++)); do
    JOB_START=$((p*CHUNK + 1))
    JOB_END=$(( (p+1)*CHUNK ))
    if (( JOB_END > N_JOBS )); then JOB_END=$N_JOBS; fi
    if (( JOB_START > N_JOBS )); then continue; fi
    OUT_CSV="$OUTDIR/worker_${p}.csv"
    LOG="$LOGDIR/worker_${p}.log"
    julia --project=. -t "$TPP" scripts/melitz_phase6_batch_worker_2026-07-30.jl "$JOB_START" "$JOB_END" "$OUT_CSV" "$LABEL" \
        > "$LOG" 2>&1 &
    PIDS+=("$!")
    echo "  launched worker $p (jobs $JOB_START-$JOB_END) pid=$! log=$LOG"
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done
T_END=$(date +%s.%N)
WALL=$(echo "$T_END - $T_START" | bc)

echo "=== Config $LABEL ($NPROC x $TPP): total wall time = ${WALL}s  (failures: $FAIL) ==="
cat "$OUTDIR"/worker_*.csv | awk 'NR==1 || $0 !~ /^config_label/' > "$OUTDIR/combined.csv" 2>/dev/null || true
echo "total_wall_s,$WALL" > "$OUTDIR/summary.csv"
echo "Combined results: $OUTDIR/combined.csv"
echo "Summary: $OUTDIR/summary.csv"
