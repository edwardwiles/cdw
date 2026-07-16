#!/bin/bash
set -uo pipefail

cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"

LOG=sequential_gravity/batch_out_realD20_run.log

echo "=== Launch: $(date) ===" > "$LOG"

FAKEDATA=3 DVAL=20 DELTA_GRID=0.1,1.0,10.0 BOUND=both PARALLEL_INVERSION=true \
  OUTER_OPT_FILE=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/full_aod_diag/csw_outer_200.opt \
  OUT_DIR=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/sequential_gravity/batch_out_realD20 \
  REAL_DATA_DIR=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/real_data/noah_D20 \
  julia -t 19 --project=. sequential_gravity/run_profiled_production.jl >> "$LOG" 2>&1
BATCH_EXIT=$?

echo "=== Batch run finished: $(date), exit=$BATCH_EXIT ===" >> "$LOG"

if [ "$BATCH_EXIT" -eq 0 ]; then
  echo "=== Starting cold cross-verification: $(date) ===" >> "$LOG"
  DVAL=20 FAKEDATA=3 WVAL=8000 REAL_DATA_DIR=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/real_data/noah_D20 \
    VERIFY_BATCH_DIR=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/sequential_gravity/batch_out_realD20 \
    julia --project=. sequential_gravity/verify_batch_solutions.jl >> "$LOG" 2>&1
  VERIFY_EXIT=$?
  echo "=== Verification finished: $(date), exit=$VERIFY_EXIT ===" >> "$LOG"
fi

echo "=== ALL DONE: $(date) ===" >> "$LOG"
touch sequential_gravity/batch_out_realD20_ALLDONE
