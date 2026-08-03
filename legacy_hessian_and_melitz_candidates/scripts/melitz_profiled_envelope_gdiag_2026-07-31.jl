# Isolated diagnostic: does melitz_exact_g_gradient match d(DeltaStar)/dg at TRULY FIXED
# (A,q) (no A-reprofiling, no q-gravity pivot movement -- melitz_middle_objective_and_gradient!
# passes q_fixed straight into melitz_f_from_Aq, bypassing the g-dependent q-pivot
# reconstruction entirely)? Isolates a genuine formula bug from the profiled-Phi run's own
# q-gravity-pivot-movement confound (the pivot cell's own reconstructed q value moves with g
# -- melitz_exact_g_gradient does not currently account for that channel).
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
session4 = MelitzInnerSession(obj4, ctx4, policy_cap)
theta_plain04 = melitz_unpower_theta_free(theta04, ctx4)
A04, f04, gpj04, fjj04, q04 = expand_free_theta_logcutoff(theta_plain04, ctx4)
A_free04 = theta_plain04[2:1+ctx4.D^2-1]
g0 = log(gpj04)
g_off = g0 - 0.08

base = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off), ctx4; coordinate=:logA)
println("base classification=", nameof(typeof(base.classification)), " Delta=", base.Delta)
@assert base.classification isa FiniteSolved

_, f_, gpj_, fjj_, _ = expand_free_theta_logcutoff(melitz_fixed_q_state_theta(A_free04, q04, exp(g_off), ctx4), ctx4)
state = MelitzExpandedState(A04, f_, gpj_, fjj_)
env_g = melitz_exact_g_gradient(obj4, base.classification.x, state, ctx4)
@printf("melitz_exact_g_gradient (fixed A,q) = %.8g\n", env_g)

for h in (1e-4, 1e-3, 1e-2)
    rp = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off + h), ctx4; coordinate=:logA)
    rm = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g_off - h), ctx4; coordinate=:logA)
    okp = rp.classification isa FiniteSolved; okm = rm.classification isa FiniteSolved
    secant = (okp && okm) ? (rp.Delta - rm.Delta) / (2h) : NaN
    relerr = (okp && okm) ? abs(env_g - secant) / max(1e-10, abs(secant)) : NaN
    @printf("h=%.0e  secant(fixed A,q; g-only)=%.6g  pred=%.6g  relerr=%.4g  (F/F=%s/%s)\n",
        h, secant, env_g, relerr, okp, okm)
end
