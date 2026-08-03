# d(ell)/dg is now numerically confirmed exact (melitz_profiled_envelope_ellcheck_2026-07-31.jl,
# ratio 1.00002). Directly test the FULL contraction (mu_link/M)*dot(dpsi,dell_dg) against the
# TRUE Delta secant, bypassing any remaining doubt about mu_link/M/dPsi conventions.
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

base = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off), ctx4; coordinate=:logA)
@assert base.classification isa FiniteSolved
x = base.classification.x
op = obj4.op
M = obj4.M
W = op.W
zeta = x[1]; mu = @view x[2:end]
mu_link = mu[op.layout.focal_link_index]
u = Vector{Float64}(undef, W); dpsi = Vector{Float64}(undef, W)
mul_G!(u, op, zeta, mu)
melitz_cc_dPsi!(dpsi, u)

bin_j = copy(op.bin[:, j]); rank_j = copy(op.rank[:, j])
_, f_, gpj_, fjj_, _ = expand_free_theta_logcutoff(melitz_fixed_q_state_theta(A_free04, q04, exp(g_off), ctx4), ctx4)
f_jj = fjj_
w_prime = ctx4.w_prime
z_orig_j = op.sorted_ctx.z_original[:, j]
z_power_j = op.sorted_ctx.z_power_original[:, j]
rjj = rank_j[j]

dell_dg = zeros(W)
for w in 1:W
    term1 = (bin_j[w] + 1 >= rjj + 1) ? f_jj : 0.0
    term2 = z_orig_j[w] >= 1.0 ? f_jj * (z_power_j[w] - 1.0) : 0.0   # profit_autarky/w_prime = f_jj*(zpow-1)
    dell_dg[w] = term1 + term2
end

pred_full_contraction_plus = (mu_link / M) * dot(dpsi, dell_dg)
pred_full_contraction_minus = -(mu_link / M) * dot(dpsi, dell_dg)
@printf("mu_link=%.6g M=%.6g dot(dpsi,dell_dg)=%.6g\n", mu_link, M, dot(dpsi, dell_dg))
@printf("pred (+mu_link/M)*dot = %.6g\n", pred_full_contraction_plus)
@printf("pred (-mu_link/M)*dot = %.6g\n", pred_full_contraction_minus)

for h in (1e-4, 1e-3)
    rp = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off + h), ctx4; coordinate=:logA)
    rm = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off - h), ctx4; coordinate=:logA)
    okp = rp.classification isa FiniteSolved; okm = rm.classification isa FiniteSolved
    secant = (okp && okm) ? (rp.Delta - rm.Delta) / (2h) : NaN
    @printf("h=%.0e  true secant=%.6g   relerr(+)=%.4g   relerr(-)=%.4g\n",
        h, secant, abs(pred_full_contraction_plus-secant)/abs(secant), abs(pred_full_contraction_minus-secant)/abs(secant))
end
