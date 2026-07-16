#!/bin/bash
set -uo pipefail

cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
PERF=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"

LOG=sequential_gravity/coldstart_delta1_W800000_run.log
echo "=== Launch: $(date) ===" > "$LOG"

FAKEDATA=3 DVAL=20 DELTA_GRID=1.0 BOUND=upper PARALLEL_INVERSION=true WVAL=800000 \
  OUTER_OPT_FILE=$PERF/full_aod_diag/csw_outer_1000.opt \
  OUT_DIR=$PERF/sequential_gravity/batch_out_realD20_W800000 \
  REAL_DATA_DIR=$PERF/real_data/noah_D20 \
  julia -t 19 --project=. sequential_gravity/run_profiled_production.jl >> "$LOG" 2>&1
echo "=== Finished: $(date), exit=$? ===" >> "$LOG"
touch sequential_gravity/coldstart_delta1_W800000_ALLDONE
