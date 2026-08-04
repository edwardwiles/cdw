#!/bin/bash
# run_campaign_chain_2026-08-04.sh -- one serial chain of the post-verifier-fix W=100k rerun +
# fresh K=3 campaign (task section 6/7/10/11). One process at a time within this chain: the next
# delta is never launched until the current delta's continuation_campaign_cell_driver.jl process
# has exited AND its report.jls is on disk (task section 12's "no mutable-seed races" requirement
# -- the driver itself reads the prior delta's finalized report.jls from CAMPAIGN_RESULTS_ROOT as
# its own inherited incumbent, so serial ordering here is load-bearing, not just tidiness).
#
# Usage:
#   run_campaign_chain_2026-08-04.sh <family> <direction> <core_start> <core_end> \
#       <explore_budget_s> <polish_budget_s> <delta1> [delta2 ...]
#
# A single delta's hard crash (nonzero exit) is logged and does NOT abort the rest of the chain --
# each delta is validated/seeded independently from the driver's own envelope logic, so later
# deltas can still proceed from whatever the envelope already has (task section 12: "a single
# failed candidate must not kill a chain").
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.juliaup/bin:$PATH"
source .knitro_env.sh

FAMILY="$1"; DIRECTION="$2"; CORE_START="$3"; CORE_END="$4"
EXPLORE_BUDGET_S="$5"; POLISH_BUDGET_S="$6"
shift 6
DELTAS=("$@")

NTHREADS=10
BLAS_THREADS=8   # matches this repo's own validated ten-by-ten resource gate (gate_ten_by_ten_all_families_2026-08-02.sh)
LOG_ROOT="/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/chain_logs/${FAMILY}_${DIRECTION}"
mkdir -p "$LOG_ROOT"

CHAIN_LABEL="${FAMILY}_${DIRECTION}"
echo "[$CHAIN_LABEL] chain start $(date -Iseconds) cores=${CORE_START}-${CORE_END} deltas=${DELTAS[*]} explore=${EXPLORE_BUDGET_S}s polish=${POLISH_BUDGET_S}s"
echo "[$CHAIN_LABEL] CAMPAIGN_MEANZC_K=${CAMPAIGN_MEANZC_K:-unset} CAMPAIGN_ORIGINZC_K=${CAMPAIGN_ORIGINZC_K:-unset} CAMPAIGN_W=${CAMPAIGN_W:-unset}"

fail_count=0
for delta in "${DELTAS[@]}"; do
  OUTPUT_DIR="/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/campaign_output/${FAMILY}/${DIRECTION}/delta_${delta}"
  mkdir -p "$OUTPUT_DIR"
  logf="$LOG_ROOT/delta_${delta}.log"
  t0=$(date +%s)
  echo "[$CHAIN_LABEL] delta=$delta START $(date -Iseconds)"
  OPENBLAS_NUM_THREADS=$BLAS_THREADS OMP_NUM_THREADS=$BLAS_THREADS \
    taskset -c ${CORE_START}-${CORE_END} julia -t $NTHREADS --project=. \
      full_aod_diag/d4_exact/continuation_campaign_cell_driver.jl \
      "$FAMILY" "$DIRECTION" "$delta" "$OUTPUT_DIR" "$EXPLORE_BUDGET_S" "$POLISH_BUDGET_S" \
      > "$logf" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then
    fail_count=$((fail_count + 1))
    echo "[$CHAIN_LABEL] delta=$delta FAILED (exit=$rc) after $((t1 - t0))s -- see $logf. Continuing to next delta (envelope keeps last valid incumbent)."
  else
    echo "[$CHAIN_LABEL] delta=$delta done, exit=0, wall=$((t1 - t0))s"
  fi
done

echo "[$CHAIN_LABEL] chain end $(date -Iseconds) fail_count=$fail_count / ${#DELTAS[@]} deltas"
exit $fail_count
