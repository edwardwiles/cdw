#!/usr/bin/env bash
set -u
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular
source .knitro_env.sh
sed -i -E "s/^maxit[[:space:]]+.*/maxit        100/" csw_outer_loop_settings_cluster.opt
start=$(date +%s)
julia --project=. master.jl > scratch_exp/results/verify_gravity_within_m100.log 2>&1
echo "EXIT $? WALL=$(($(date +%s)-start))" >> scratch_exp/results/verify_gravity_within_m100.log
# restore committed maxit
sed -i -E "s/^maxit[[:space:]]+.*/maxit        25/" csw_outer_loop_settings_cluster.opt
echo "GRAVITY_M100_DONE"
