# Quick scan to find a D4 welfare offset with a non-degenerate profiled Delta (the anchor's
# own Delta, 3.4e-6, sits far below the middle-loop KNITRO solver's own opttol_abs=1e-4 --
# scan for an offset where Phi(g) is O(0.001-0.5), informative for Phase 1 validation.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
WT = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profiled-q-envelope-gradient-2026-07-31"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(WT, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT = joinpath(WT, "melitz_middle_loop_opt_2026-07-30.opt")

FIXTURE4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta04 = build_melitz_psi_bundle(FIXTURE4; outer_parameterization=:logcutoff, policy=policy_cap,
    backend=:matrix_free, forbid_dense_fallback=true)
ctx4 = obj4.γ
session4 = MelitzInnerSession(obj4, ctx4, policy_cap)
theta_plain04 = melitz_unpower_theta_free(theta04, ctx4)
A04, f04, gpj04, fjj04, q04 = expand_free_theta_logcutoff(theta_plain04, ctx4)
A_free04 = theta_plain04[2:1+ctx4.D^2-1]
g0 = log(gpj04)

function profile_A(session, q_target, gpj_target, A_free_start, ctx; max_evals=150, box=0.1)
    theta_for_sys = melitz_fixed_q_state_theta(A_free_start, q_target, gpj_target, ctx)
    sys = melitz_fixed_q_middle_constraint_system(theta_for_sys, ctx, session.obj)
    A_proj = melitz_project_start_to_middle_constraints(copy(A_free_start), sys, ctx)
    session.obj.use_cached_x = false; session.obj.x .= NaN
    res = solve_melitz_fixed_q_A_profile_v2(session, q_target, gpj_target, A_proj, ctx;
        coordinate=:logA, max_evals=max_evals, box=box, outer_loop_opt=MIDDLE_OPT,
        sys=sys, cap_handling=:barrier, cap_barrier_multiple=5.0)
    return res
end

for delta_g in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)
    for sgn in (-1, 1)
        g_off = g0 + sgn * delta_g
        r = profile_A(session4, q04, exp(g_off), A_free04, ctx4)
        cls = r.r_incumbent isa FiniteSolved ? "FiniteSolved" :
              r.r_incumbent isa AboveEvaluationCap ? "AboveEvaluationCap" : "InfiniteDeltaCertified"
        @printf("sgn=%+d  delta_g=%.3f  g=%.5f  class=%-20s Delta=%.6g\n", sgn, delta_g, g_off, cls, r.Delta_incumbent)
        flush(stdout)
    end
end
