# Quick, single-purpose follow-up to Phase 5: at frac=0.50 (where fixed-A/f itself is
# NumericalFailure), Phase 5's A-only/f-only/full nuisance stages ALL failed immediately with
# a KNITRO grad_callback -502 "could not evaluate at the initial point" error -- a DIVERGENCE
# from the original 2026-07-28 gamma-profile session's own finding at this exact frac (full
# A/f nuisance search there found a genuine nStatus=-101 rescue, Delta=5.05e-1). Hypothesis:
# Phase 5's own script called the nuisance solves with obj.x left at whatever state the
# IMMEDIATELY PRECEDING (failed) fixed-A/f solve_melitz_delta! call left it in -- a poisoned/
# runaway dual warm start, the same failure mode this codebase's own
# `MelitzInnerSession`/anomaly-diagnostic docs document (Phase 4 of
# docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md: "even a freshly-
# capped re-solve from a stale runaway warm start returns NumericalFailure ... purely from the
# poisoned starting dual"). This single re-run (ONE nuisance solve, over Phase 5's own 12-solve
# budget by exactly 1, disclosed) checks whether a NEUTRAL warm start (obj.x reset) changes the
# outcome, to correctly attribute the divergence rather than leaving it ambiguous.

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

CAP = 10.0
policy = CappedEvaluation(CAP)
d4 = build_d4_fixture(; policy=policy)
ctx, obj, theta0 = d4.ctx, d4.obj, d4.theta0
n = length(theta0); D = ctx.D
j = ctx.target_country
state0 = melitz_outer_state(theta0, ctx)
lambda_jj = state0.equilibrium.trade_flow[j, j] / state0.equilibrium.expenditure[j]
wratio = 1.0 / ctx.w[j]
sigma = ctx.sigma
kappa_of_g(g, wratio, sigma) = wratio * exp(g)^(1 / (sigma - 1))
g_of_kappa(kappa, wratio, sigma) = log((kappa / wratio)^(sigma - 1))
kappa_pareto = kappa_of_g(theta0[1], wratio, sigma)
kappa_min = lambda_jj^(1 / (sigma - 1))
frac = 0.50
kappa_f = kappa_pareto + frac * (kappa_min - kappa_pareto)
g_f = g_of_kappa(kappa_f, wratio, sigma)
theta_fixed = copy(theta0); theta_fixed[1] = g_f

inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
block_free_mask(n, D, free_g, free_A, free_f) = (nA = D^2 - 1; m = falses(n); m[1] = free_g; m[2:1+nA] .= free_A; m[2+nA:end] .= free_f; m)
mask_A = block_free_mask(n, D, false, true, false)

# Reproduce Phase 5's own failing sequence first: fixed-A/f attempt (expected NumericalFailure), THEN A-only with NO neutral reset (reproduces the -502).
session = MelitzInnerSession(obj, ctx, policy)
r_fixed = solve_melitz_delta!(session, theta_fixed, policy; warm_start_source=:previous)
println("fixed A/f at frac=0.50: ", typeof(r_fixed), "   obj.x state after: ||x||=", norm(obj.x), "  any NaN=", any(isnan, obj.x))
flush(stdout)

println("\n--- A-only WITHOUT neutral reset (reproduces Phase 5's own -502) ---")
resA_stale = solve_melitz_nuisance_min_delta(ctx, obj, theta_fixed; free_mask=mask_A, radius=0.3,
    gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
    policy=policy, forbid_dense_fallback=true)
println("nStatus=", resA_stale.nStatus, "  verified=", resA_stale.r_final.verified)
flush(stdout)

println("\n--- A-only WITH neutral reset (obj.x .= NaN; use_cached_x=false) ---")
obj.x .= NaN
obj.use_cached_x = false
resA_neutral = solve_melitz_nuisance_min_delta(ctx, obj, theta_fixed; free_mask=mask_A, radius=0.3,
    gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
    policy=policy, forbid_dense_fallback=true)
println("nStatus=", resA_neutral.nStatus, "  verified=", resA_neutral.r_final.verified,
        resA_neutral.r_final.verified ? "  Delta=$(resA_neutral.r_final.Delta)" : "")
flush(stdout)

open(joinpath(REPO, "docs", "key_results", "melitz_post_consolidation_phase5b_neutral_warmstart_check_2026-07-28.csv"), "w") do io
    println(io, "variant,nStatus,verified,Delta")
    println(io, "stale_warmstart,$(resA_stale.nStatus),$(resA_stale.r_final.verified),$(resA_stale.r_final.verified ? resA_stale.r_final.Delta : NaN)")
    println(io, "neutral_warmstart,$(resA_neutral.nStatus),$(resA_neutral.r_final.verified),$(resA_neutral.r_final.verified ? resA_neutral.r_final.Delta : NaN)")
end
println("\nDONE Phase 5b (post-consolidation, diagnostic follow-up).")
