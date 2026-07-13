#!/usr/bin/env bash
set -u
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular
source .knitro_env.sh
R="bash scratch_exp/run_experiment.sh"

# Recommended combo: autodiff (no Dirac) + product_findiff Hessian
$R rec_ad_prodfd_m25  0 auto 4 25  0
$R rec_ad_prodfd_m100 0 auto 4 100 0

# B1 profiling (restore inner opt first so profiling uses the committed inner settings)
cp scratch_exp/opt_backups/ek_inner_loop_options.opt ek_inner_loop_options.opt
cp scratch_exp/opt_backups/csw_outer_loop_settings_cluster.opt csw_outer_loop_settings_cluster.opt
echo "=== B1 profile_inner.jl ==="
julia --project=. profile_inner.jl > scratch_exp/results/profile_inner.log 2>&1
echo "profile exit $?"

# A2 exact objective-Hessian callback (uses scratch_exp/opt_exacthess.opt, hessopt=1, maxit 100)
echo "=== A2 test_exact_hessian.jl ==="
julia --project=. test_exact_hessian.jl > scratch_exp/results/exact_hessian.log 2>&1
echo "exact_hessian exit $?"

echo "PHASE3_DONE_$(date +%H:%M:%S)"
