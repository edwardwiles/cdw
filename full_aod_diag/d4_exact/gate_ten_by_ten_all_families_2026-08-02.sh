#!/bin/bash
# gate_ten_by_ten_all_families_2026-08-02.sh -- production Hessian audit, task brief Section 19.
# Mixed ten-process resource smoke: 2 processes each of unrestricted/flexible_cm/common_frechet/
# origin_zc/cm_meanzc (all 5 families, not just the 2 the prior ZC-only gate5 script covered),
# 10 Julia threads/process, BLAS=8, disjoint taskset core ranges (100 of this host's cores),
# using the canonical audit harness (post-fix production HEAD) at W=100,000.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.juliaup/bin:$PATH"
source .knitro_env.sh

NTHREADS=10
BLAS_THREADS=8
OUTDIR="results/gate_ten_by_ten_all_families_2026-08-02"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

echo "Launching 10 concurrent processes (2 each: unrestricted/flexible_cm/common_frechet/origin_zc/cm_meanzc) x $NTHREADS Julia threads (BLAS threads=$BLAS_THREADS), disjoint core affinity..."
pids=()
t_launch=$(date +%s)
core=0
for fam in unrestricted flexible_cm common_frechet origin_zc cm_meanzc; do
  for i in 0 1; do
    core_start=$core
    core_end=$((core_start + NTHREADS - 1))
    core=$((core + NTHREADS))
    logf="$OUTDIR/${fam}_proc${i}.log"
    OPENBLAS_NUM_THREADS=$BLAS_THREADS OMP_NUM_THREADS=$BLAS_THREADS \
    AUDIT_FAMILY=$fam AUDIT_W=100000 AUDIT_NREPEAT=10 \
    AUDIT_OUTDIR="$OUTDIR" \
      taskset -c ${core_start}-${core_end} julia -t $NTHREADS --project=. \
        full_aod_diag/d4_exact/production_all_hessian_audit_harness_2026-08-02.jl > "$logf" 2>&1 &
    pids+=($!)
  done
done

echo "Launched ${#pids[@]} processes on cores 0-$((core-1)): ${pids[*]}"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
t_done=$(date +%s)
echo "All processes finished (or failed) in $((t_done - t_launch))s. fail_flag=$fail"

echo "=== per-process nStatus / completion check ==="
for fam in unrestricted flexible_cm common_frechet origin_zc cm_meanzc; do
  for i in 0 1; do
    logf="$OUTDIR/${fam}_proc${i}.log"
    status_line=$(grep -E "TRUE_COLD_INNER_SOLVE\] done" "$logf" || echo "MISSING")
    complete_line=$(grep -c "AUDIT HARNESS COMPLETE" "$logf" || echo "0")
    echo "$fam proc$i: complete=$complete_line  $status_line"
  done
done
exit $fail
