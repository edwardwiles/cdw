#!/usr/bin/env bash
# Generic single-experiment runner.
# Usage: run_experiment.sh <tag> <use_jac> <outer_algo> <outer_hessopt> <outer_maxit> <inner_algo>
#   tag           : label for logs/results
#   use_jac       : 1 analytic outer Jacobian, 0 ForwardDiff autodiff
#   outer_algo    : algorithm value for csw_outer_loop_settings_cluster.opt (0..5)
#   outer_hessopt : hessopt value for csw (auto|1|2|3|4|5|6  -> we pass raw string)
#   outer_maxit   : maxit for csw outer solve
#   inner_algo    : algorithm value for ek_inner_loop_options.opt (0..5)
set -u
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular
source .knitro_env.sh

TAG="$1"; USE_JAC="$2"; OALGO="$3"; OHESS="$4"; OMAXIT="$5"; IALGO="$6"; BETA="${7:-0.01}"
RES=scratch_exp/results
mkdir -p "$RES"
LOG="$RES/${TAG}.log"

# --- restore .opt files from pristine backups, then apply overrides ---
cp scratch_exp/opt_backups/csw_outer_loop_settings_cluster.opt csw_outer_loop_settings_cluster.opt
cp scratch_exp/opt_backups/ek_inner_loop_options.opt          ek_inner_loop_options.opt

sed -i -E "s/^algorithm[[:space:]]+.*/algorithm    ${OALGO}/"  csw_outer_loop_settings_cluster.opt
sed -i -E "s/^hessopt[[:space:]]+.*/hessopt      ${OHESS}/"    csw_outer_loop_settings_cluster.opt
sed -i -E "s/^maxit[[:space:]]+.*/maxit        ${OMAXIT}/"     csw_outer_loop_settings_cluster.opt
sed -i -E "s/^algorithm[[:space:]]+.*/algorithm    ${IALGO}/"  ek_inner_loop_options.opt

echo "=== EXP ${TAG}: use_jac=${USE_JAC} outer_algo=${OALGO} outer_hess=${OHESS} outer_maxit=${OMAXIT} inner_algo=${IALGO} ==="
echo "csw settings:"; grep -nE "^(algorithm|hessopt|maxit|opttol|feastol|eval_fcga)" csw_outer_loop_settings_cluster.opt
echo "inner settings:"; grep -nE "^(algorithm|hessopt|maxit)" ek_inner_loop_options.opt

echo "EXP_BETA=${BETA}"
start=$(date +%s)
# 25-min per-run safety cap so a pathological algorithm can't hang the overnight queue
timeout 1500 env EXP_USE_JAC="${USE_JAC}" EXP_BETA="${BETA}" julia --project=. run_master.jl > "$LOG" 2>&1
ec=$?
wall=$(( $(date +%s) - start ))

# newest bounds csv
CSV=$(ls -t NoJacob_*DR_1_*.csv 2>/dev/null | head -1)
BOUNDS=$(cat "$CSV" 2>/dev/null)

{
  echo "TAG=${TAG} exit=${ec} wall_s=${wall}"
  echo "bounds(δ,κlo,κhi)=${BOUNDS}"
  echo "--- OUTER_SOLVE lines ---"
  grep "OUTER_SOLVE" "$LOG"
  grep "OUTER_SOLVE_START\|OUTER_SOLVE_END\|EXP_USE_JAC" "$LOG"
  echo "======================================================================"
} | tee -a "$RES/SUMMARY.txt"
