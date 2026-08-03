# Isolate whether the bug is in d(ell)/dg (the moment-column derivative) or in the
# dPsi-contraction/mu_link scaling step: compute op.ell numerically at g+-h (fixed A, fixed
# q_fixed matrix passed to melitz_f_from_Aq) and compare against the analytical
# f_jj*tail1_j-style / autarky-style formula, cell by cell (well, aggregated the same way).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
WT = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profiled-q-envelope-gradient-2026-07-31"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(WT, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

FIXTURE4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta04 = build_melitz_psi_bundle(FIXTURE4; outer_parameterization=:logcutoff, policy=policy_cap,
    backend=:matrix_free, forbid_dense_fallback=true)
ctx4 = obj4.γ
D4 = ctx4.D
session4 = MelitzInnerSession(obj4, ctx4, policy_cap)
theta_plain04 = melitz_unpower_theta_free(theta04, ctx4)
A04, f04, gpj04, fjj04, q04 = expand_free_theta_logcutoff(theta_plain04, ctx4)
A_free04 = theta_plain04[2:1+ctx4.D^2-1]
g0 = log(gpj04)
g_off = g0 - 0.08
j = ctx4.target_country

# Build op at g_off (base), read off ell + bin/rank.
base = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off), ctx4; coordinate=:logA)
@assert base.classification isa FiniteSolved
op = obj4.op
ell_base = copy(op.ell)
bin_j = copy(op.bin[:, j]); rank_j = copy(op.rank[:, j])
W = op.W

# Numerically perturb g (fixed A, fixed q04), re-run the operator update via
# melitz_middle_objective_and_gradient! (which internally calls solve_melitz_delta! ->
# melitz_update_operator_at_theta! -> update_moment_operator!), then read op.ell again.
h = 1e-4
rp = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off + h), ctx4; coordinate=:logA)
ell_p = copy(op.ell)
rm = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off - h), ctx4; coordinate=:logA)
ell_m = copy(op.ell)

dell_dg_numeric = (ell_p .- ell_m) ./ (2h)

# Analytical prediction: f_jj*1{bin_j[w]+1 >= rank_j[j]+1} + profit_autarky(z_w)/w_prime [z_orig_j[w]>=1]
_, f_, gpj_, fjj_, _ = expand_free_theta_logcutoff(melitz_fixed_q_state_theta(A_free04, q04, exp(g_off), ctx4), ctx4)
f_jj = fjj_
w_prime = ctx4.w_prime
z_orig_j = op.sorted_ctx.z_original[:, j]
z_power_j = op.sorted_ctx.z_power_original[:, j]
sigma = ctx4.sigma

dell_dg_pred = zeros(W)
rjj = rank_j[j]
for w in 1:W
    term1 = (bin_j[w] + 1 >= rjj + 1) ? f_jj : 0.0
    term2 = 0.0
    if z_orig_j[w] >= 1.0
        profit_autarky = w_prime * f_jj * (z_power_j[w] - 1.0)
        term2 = profit_autarky / w_prime
    end
    dell_dg_pred[w] = term1 + term2
end

diff = dell_dg_numeric .- dell_dg_pred
@printf("max|numeric - pred| = %.6g\n", maximum(abs.(diff)))
@printf("mean numeric = %.6g   mean pred = %.6g\n", sum(dell_dg_numeric)/W, sum(dell_dg_pred)/W)
@printf("sum numeric = %.6g   sum pred = %.6g   ratio = %.6g\n",
    sum(dell_dg_numeric), sum(dell_dg_pred), sum(dell_dg_numeric)/sum(dell_dg_pred))
# histogram of where they disagree
n_term1_only = count(w -> (bin_j[w]+1>=rjj+1) && !(z_orig_j[w]>=1.0), 1:W)
n_term2_only = count(w -> !(bin_j[w]+1>=rjj+1) && (z_orig_j[w]>=1.0), 1:W)
n_both = count(w -> (bin_j[w]+1>=rjj+1) && (z_orig_j[w]>=1.0), 1:W)
n_neither = W - n_term1_only - n_term2_only - n_both
@printf("n_term1_only=%d n_term2_only=%d n_both=%d n_neither=%d\n", n_term1_only, n_term2_only, n_both, n_neither)
@printf("rank_j[j]=%d  D=%d\n", rjj, D4)
