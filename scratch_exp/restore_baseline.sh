#!/usr/bin/env bash
# Restore the three KNITRO .opt files to their pristine committed state (the sweep harness
# rewrites them per-run). Source-code changes on the experiments branch are intentionally kept
# (documented in EXPERIMENTS_FINDINGS.md); use `git checkout main` for the untouched baseline.
set -eu
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular
cp scratch_exp/opt_backups/csw_outer_loop_settings_cluster.opt csw_outer_loop_settings_cluster.opt
cp scratch_exp/opt_backups/ek_inner_loop_options.opt          ek_inner_loop_options.opt
cp scratch_exp/opt_backups/ek_outer_loop_options.opt          ek_outer_loop_options.opt
echo "restored .opt files to committed baseline:"
grep -nE "^(algorithm|hessopt|maxit)" csw_outer_loop_settings_cluster.opt
