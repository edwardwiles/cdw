#!/usr/bin/env bash
# Full experiment sweep. Each call: tag use_jac outer_algo outer_hess outer_maxit inner_algo [beta]
# outer solve uses csw_outer_loop_settings_cluster.opt; inner uses ek_inner_loop_options.opt.
# Baseline = analytic Jac, algorithm auto, hessopt auto, maxit 25 -> kappa [0.007437, 0.219549].
set -u
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular
R="bash scratch_exp/run_experiment.sh"

# ============ CRUX (A1/A3): analytic(with Dirac) vs autodiff(no Dirac), + convergence ============
# maxit 25 (re-run to capture feas_err/opt_err) then maxit 100 to see where each truly settles.
$R x_ana_m25       1 auto auto 25  0
$R x_autodiff_m25  0 auto auto 25  0
$R x_ana_m100      1 auto auto 100 0
$R x_autodiff_m100 0 auto auto 100 0

# ============ A3: beta sweep of the analytic-Jacobian SmoothDirac (beta->inf == no Dirac) ============
$R a3_beta_1em3   1 auto auto 25 0 0.001
$R a3_beta_1em1   1 auto auto 25 0 0.1
$R a3_beta_1e0    1 auto auto 25 0 1.0
$R a3_beta_1e2    1 auto auto 25 0 100.0

# ============ A2: Hessian option sweep (exact gradient throughout, analytic Jac) ============
$R a2_hess_prodfd 1 auto 4 25 0     # Hessian-vector products via finite-diff of EXACT gradient
$R a2_hess_sr1    1 auto 3 25 0     # dense SR1 quasi-Newton
$R a2_hess_lbfgs  1 auto 6 25 0     # limited-memory BFGS
$R a2_hess_bfgs   1 auto 2 25 0     # dense BFGS (explicit)

# ============ A4: outer algorithm sweep (analytic Jac, auto Hessian) ============
$R a4_o_direct    1 1 auto 25 0
$R a4_o_cg        1 2 auto 25 0
$R a4_o_active    1 3 auto 25 0
$R a4_o_sqp       1 4 auto 25 0

# ============ A4: inner algorithm sweep ============
$R a4_i_direct    1 auto auto 25 1
$R a4_i_active    1 auto auto 25 3

# ============ best-of convergence ============
$R conv_prodfd_m100 1 auto 4 100 0

echo "PHASE2_DONE_$(date +%H:%M:%S)"
