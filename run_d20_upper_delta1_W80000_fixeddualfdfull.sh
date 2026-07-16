#!/bin/bash
set -uo pipefail

cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
PERF=/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.juliaup/bin:$PATH"

LOG=sequential_gravity/d20_upper_delta1_W80000_fixeddualfdfull_run.log
echo "=== Launch: $(date) ===" > "$LOG"
echo "=== real D=20 data, W=80000, BOUND=upper, delta=1.0, gradient_method=fixed_dual_fd_full ===" >> "$LOG"

# Separate OUT_DIR from the existing pointwise_ad D=20/W=80000 results (batch_out_realD20_W80000) --
# checkpoint filenames don't encode gradient_method, so reusing that dir would make the resume logic
# silently skip this run entirely (it would see seq_upper_delta1.0.jld2 already "done" and load the
# OLD pointwise_ad result instead of computing anything new).
FAKEDATA=3 DVAL=20 DELTA_GRID=1.0 BOUND=upper PARALLEL_INVERSION=true WVAL=80000 \
  GRADIENT_METHOD=fixed_dual_fd_full \
  OUTER_OPT_FILE=$PERF/full_aod_diag/csw_outer_1000.opt \
  OUT_DIR=$PERF/sequential_gravity/batch_out_realD20_W80000_fixeddualfdfull \
  REAL_DATA_DIR=$PERF/real_data/noah_D20 \
  julia -t 19 --project=. sequential_gravity/run_profiled_production.jl >> "$LOG" 2>&1
echo "=== Finished: $(date), exit=$? ===" >> "$LOG"
touch sequential_gravity/d20_upper_delta1_W80000_fixeddualfdfull_ALLDONE
