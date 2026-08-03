# Bisect the ~5x discrepancy in melitz_exact_g_gradient by printing every intermediate
# quantity (mu_link, M, f_jj, Sya, S1a) side by side with exact_a_gradient's OWN Step 3
# (already proven correct to relerr~3e-7 in melitz_profiled_envelope_harness_check_2026-07-31.jl).
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

base = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off), ctx4; coordinate=:logA)
@assert base.classification isa FiniteSolved
x = base.classification.x

_, f_, gpj_, fjj_, _ = expand_free_theta_logcutoff(melitz_fixed_q_state_theta(A_free04, q04, exp(g_off), ctx4), ctx4)
state = MelitzExpandedState(A04, f_, gpj_, fjj_)

op = obj4.op
D = op.D; W = op.W; M = obj4.M; j = ctx4.target_country
zeta = x[1]; mu = @view x[2:end]
mu_link = mu[op.layout.focal_link_index]
u = Vector{Float64}(undef, W); dpsi = Vector{Float64}(undef, W)
mul_G!(u, op, zeta, mu)
melitz_cc_dPsi!(dpsi, u)
z_orig_j = @view op.sorted_ctx.z_original[:, j]
z_power_j = @view op.sorted_ctx.z_power_original[:, j]
Sya = 0.0; S1a = 0.0
for s in 1:W
    if z_orig_j[s] >= 1.0
        global Sya += dpsi[s] * z_power_j[s]
        global S1a += dpsi[s]
    end
end
f_jj = state.f_jj
sigma = ctx4.sigma
w_prime = ctx4.w_prime
autarky_term = w_prime * f_jj * (Sya - S1a)
coefsm1_over_M = (sigma - 1.0) / M
link_coef = mu_link * coefsm1_over_M

@printf("D=%d W=%d M=%.6g j=%d\n", D, W, M, j)
@printf("mu_link=%.8g  f_jj=%.8g  Sya=%.8g  S1a=%.8g  (Sya-S1a)=%.8g\n", mu_link, f_jj, Sya, S1a, Sya-S1a)
@printf("autarky_term = w_prime*f_jj*(Sya-S1a) = %.8g   (w_prime=%.6g)\n", autarky_term, w_prime)
@printf("link_coef = mu_link*(sigma-1)/M = %.8g\n", link_coef)
dDelta_da_jj_manual = -(link_coef / w_prime) * autarky_term
@printf("MANUAL dDelta_da[j,j] = -link_coef/w_prime*autarky_term = %.8g\n", dDelta_da_jj_manual)

dDelta_dg_manual = (mu_link / M) * f_jj * (Sya - S1a)
@printf("MANUAL dDelta_dg = (mu_link/M)*f_jj*(Sya-S1a) = %.8g\n", dDelta_dg_manual)
@printf("check: -(1/(sigma-1))*dDelta_da_jj_manual = %.8g  (should equal dDelta_dg_manual if algebra is right)\n",
    -(1.0/(sigma-1))*dDelta_da_jj_manual)

# Now via the actual library functions:
grad_free_A, dDelta_da_full = melitz_exact_a_gradient(obj4, x, state, ctx4)
@printf("LIB dDelta_da_full[j,j] = %.8g\n", dDelta_da_full[j, j])
env_g = melitz_exact_g_gradient(obj4, x, state, ctx4)
@printf("LIB melitz_exact_g_gradient = %.8g\n", env_g)
