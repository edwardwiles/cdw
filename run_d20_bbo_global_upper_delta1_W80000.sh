#!/bin/bash
set -uo pipefail

cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
PERF=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"

LOG=sequential_gravity/global_opt/logs/bbo_d20_upper_delta1_W80000_run.log
mkdir -p sequential_gravity/global_opt/logs
echo "=== Launch: $(date) ===" > "$LOG"

FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
  REAL_DATA_DIR=$PERF/real_data/noah_D20 \
  BBO_DELTA=1.0 BBO_MAXIMIZE_GP=false \
  BBO_MAXTIME=21600 BBO_POPSIZE=16 BBO_MAXIT=100 \
  julia -t 19 --project=. sequential_gravity/global_opt/run_bbo_d20_real.jl >> "$LOG" 2>&1
echo "=== finished: $(date), exit=$? ===" >> "$LOG"

touch sequential_gravity/global_opt/logs/bbo_d20_upper_delta1_W80000_ALLDONE
