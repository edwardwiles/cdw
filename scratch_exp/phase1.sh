#!/usr/bin/env bash
set -u
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular
bash scratch_exp/run_experiment.sh base_ana_bfgs 1 auto auto 25 0
bash scratch_exp/run_experiment.sh autodiff_bfgs 0 auto auto 25 0
echo "PHASE1_DONE"
