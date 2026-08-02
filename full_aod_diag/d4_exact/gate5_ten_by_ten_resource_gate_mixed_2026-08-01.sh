#!/bin/bash
# Gate 5 (production integration, 2026-08-01): MIXED-FAMILY ten-by-ten resource smoke.
# Launches 10 CONCURRENT fresh Julia processes x 10 Julia threads each (100 of this host's cores,
# disjoint taskset affinity) -- 5 running cm_meanzc, 5 running origin_zc, all with the
# all_optimized backend combo (the new production defaults), all at real production width
# W=100,000/K=3. Records wall time per process and whether all completed within a sane bound (no
# throughput collapse, no swap pressure) under REAL simultaneous cross-family host load -- the
# actual intended production deployment pattern (both families' campaigns run concurrently on the
# same host), not a single-family repeat of the prior closeout's own ten-by-ten gate.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.juliaup/bin:$PATH"

NPROC_PER_FAMILY=5
NTHREADS=10
BLAS_THREADS=8
OUTDIR="results/gate5_ten_by_ten_mixed_2026-08-01"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

echo "Launching $((NPROC_PER_FAMILY*2)) concurrent processes (mixed cm_meanzc/origin_zc) x $NTHREADS Julia threads (BLAS threads=$BLAS_THREADS), disjoint core affinity..."
pids=()
t_launch=$(date +%s)
core=0
for fam in cm_meanzc origin_zc; do
  for i in $(seq 0 $((NPROC_PER_FAMILY-1))); do
    core_start=$core
    core_end=$((core_start + NTHREADS - 1))
    core=$((core + NTHREADS))
    logf="$OUTDIR/${fam}_proc${i}.log"
    OPENBLAS_NUM_THREADS=$BLAS_THREADS OMP_NUM_THREADS=$BLAS_THREADS \
    ZC_FAMILY=$fam ZC_ARM=all_optimized ZC_W=100000 ZC_REP=mixed_t10_${fam}_${i} ZC_OUTDIR="$OUTDIR" \
      taskset -c ${core_start}-${core_end} julia -t $NTHREADS --project=. \
        full_aod_diag/d4_exact/gate5_single_solve_worker_2026-08-01.jl > "$logf" 2>&1 &
    pids+=($!)
  done
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
t_total=$(( $(date +%s) - t_launch ))

echo "All $((NPROC_PER_FAMILY*2)) processes finished in ${t_total}s wall (fail=$fail)"
echo ""
echo "=== per-process results ==="
grep -h "RESULT" "$OUTDIR"/*.log || echo "NO RESULT LINES FOUND -- see individual logs"

echo ""
echo "=== memory snapshot (informational) ==="
free -h

out_md="docs/GATE5_MIXED_TEN_BY_TEN_RESOURCE_GATE_2026-08-01.md"
{
  echo "# Gate 5: mixed-family ten-by-ten resource gate (2026-08-01)"
  echo ""
  echo "Config: $NPROC_PER_FAMILY cm_meanzc + $NPROC_PER_FAMILY origin_zc concurrent fresh Julia"
  echo "processes ($((NPROC_PER_FAMILY*2)) total) x $NTHREADS Julia threads each, BLAS threads=$BLAS_THREADS,"
  echo "disjoint taskset core affinity (0-$((NPROC_PER_FAMILY*2*NTHREADS-1)) of this host's cores),"
  echo "all running the all_optimized-backend single true-cold inner solve at W=100,000, K_mean=K_pair=3."
  echo ""
  echo "Total wall time for all $((NPROC_PER_FAMILY*2)) to complete: ${t_total}s."
  echo ""
  echo '```'
  grep -h "RESULT" "$OUTDIR"/*.log || true
  echo '```'
  echo ""
  echo "fail_flag=$fail (0 = all processes exited cleanly)"
} > "$out_md"
echo "Wrote $out_md"

if [ "$fail" -ne 0 ]; then
  echo "SOME_PROCESSES_FAILED"
  exit 1
fi
echo "GATE5_MIXED_TEN_BY_TEN_DONE"
