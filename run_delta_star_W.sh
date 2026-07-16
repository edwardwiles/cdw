#!/bin/bash
set -uo pipefail
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"

W=$1
LOG=sequential_gravity/delta_star_schedule_W${W}_run.log
echo "=== Launch: $(date) ===" > "$LOG"

FAKEDATA=3 DVAL=20 WVAL=$W GAMMAP_STEP=0.001 \
  REAL_DATA_DIR=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/real_data/noah_D20 \
  SCHEDULE_OUT=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/sequential_gravity/delta_star_schedule_W${W}_out.jld2 \
  julia --project=. sequential_gravity/delta_star_schedule.jl >> "$LOG" 2>&1
echo "=== Finished: $(date), exit=$? ===" >> "$LOG"
touch sequential_gravity/delta_star_schedule_W${W}_ALLDONE
