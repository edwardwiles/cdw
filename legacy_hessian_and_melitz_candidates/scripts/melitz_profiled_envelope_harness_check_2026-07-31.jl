# Sanity check: does the SAME test harness (melitz_middle_objective_and_gradient!, base.x,
# state construction) reproduce a KNOWN-GOOD result for the ALREADY-validated exact_a_gradient
# formula (specifically its Step-3 autarky term, direction = perturb A[j,j] only)? If this ALSO
# shows a large mismatch, the bug is in the test harness, not in melitz_exact_g_gradient.
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
println("base classification=", nameof(typeof(base.classification)), " Delta=", base.Delta)
@assert base.classification isa FiniteSolved

_, f_, gpj_, fjj_, _ = expand_free_theta_logcutoff(melitz_fixed_q_state_theta(A_free04, q04, exp(g_off), ctx4), ctx4)
state = MelitzExpandedState(A04, f_, gpj_, fjj_)

# Known-good exact A-gradient (free-coordinate basis), already validated in prior sessions.
grad_free_A, dDelta_da_full = melitz_exact_a_gradient(obj4, base.classification.x, state, ctx4)
println("exact A-block gradient (free coords): ", round.(grad_free_A; digits=6))

# `A_free04`'s own coordinate that packs A[j,j] directly (module pivot_map machinery).
pm = melitz_pivot_map(ctx4)
jj_idx = pm.A_jj_index
println("A_jj free-coordinate index = ", jj_idx, "  predicted d(Delta)/d(A_free[jj_idx]) = ", grad_free_A[jj_idx])

for h in (1e-4, 1e-3, 1e-2)
    Ap = copy(A_free04); Ap[jj_idx] += h
    Am = copy(A_free04); Am[jj_idx] -= h
    rp = melitz_middle_objective_and_gradient!(session4, Ap, q04, exp(g_off), ctx4; coordinate=:logA)
    rm = melitz_middle_objective_and_gradient!(session4, Am, q04, exp(g_off), ctx4; coordinate=:logA)
    okp = rp.classification isa FiniteSolved; okm = rm.classification isa FiniteSolved
    secant = (okp && okm) ? (rp.Delta - rm.Delta) / (2h) : NaN
    pred = grad_free_A[jj_idx]
    relerr = (okp && okm) ? abs(pred - secant) / max(1e-10, abs(secant)) : NaN
    @printf("h=%.0e  secant(A_jj coord)=%.6g  pred=%.6g  relerr=%.4g  (F/F=%s/%s)\n",
        h, secant, pred, relerr, okp, okm)
end
