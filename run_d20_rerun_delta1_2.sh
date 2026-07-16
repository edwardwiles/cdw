#!/bin/bash
set -uo pipefail

cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
PERF=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"

LOG=sequential_gravity/rerun_delta1_2_run.log
echo "=== Launch: $(date) ===" > "$LOG"

# DELTA_GRID includes 0.1 and 10.0 too (already-done checkpoints, will be skipped/reused
# for warm-start chaining) alongside the two new/rerun points: 1.0 (deleted checkpoint,
# forces fresh solve with higher maxit) and 2.0 (new).
FAKEDATA=3 DVAL=20 DELTA_GRID=0.1,1.0,2.0,10.0 BOUND=both PARALLEL_INVERSION=true \
  OUTER_OPT_FILE=$PERF/full_aod_diag/csw_outer_1000.opt \
  OUT_DIR=$PERF/sequential_gravity/batch_out_realD20 \
  REAL_DATA_DIR=$PERF/real_data/noah_D20 \
  julia -t 19 --project=. sequential_gravity/run_profiled_production.jl >> "$LOG" 2>&1
EXIT=$?

echo "=== Finished: $(date), exit=$EXIT ===" >> "$LOG"
touch sequential_gravity/rerun_delta1_2_ALLDONE
