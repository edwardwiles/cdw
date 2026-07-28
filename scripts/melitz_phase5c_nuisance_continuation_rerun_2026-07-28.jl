# Follow-up to Phase 5 (docs/melitz_post_consolidation_validation_2026-07-28.md), fixing the
# gap Phase 5b diagnosed: the original 2026-07-28 gamma-profile session's own nuisance script
# (scripts/melitz_phase6_d4_staged_nuisance_profile_2026-07-28.jl) threads a CONTINUATION
# warm-start dual (AND theta, for the A/f block) across the fraction grid for the A-only/
# f-only stages -- each fraction's A-only stage starts from the PRECEDING fraction's own
# successful A-only dual+theta, not from a cold/neutral state at the new fraction's own g.
# Phase 5's script omitted this (each fraction was independent), which is why frac=0.50/0.65
# failed outright there (Phase 5b ruled out a stale-dual/consolidation cause; the real gap was
# never having a valid warm start in the first place, since fixed-A/f itself fails at those g).
#
# Re-runs the FULL 4-fraction grid (continuation requires processing in order: 0.10 feeds 0.35
# feeds 0.50 feeds 0.65) via the consolidated API. Budget: <=12 nuisance solves again (matching
# Phase 5's own original budget for this narrowly-scoped follow-up).

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

CAP = 10.0
policy = CappedEvaluation(CAP)
FRACTIONS = [0.10, 0.35, 0.50, 0.65]

kappa_of_g(g, wratio, sigma) = wratio * exp(g)^(1 / (sigma - 1))
g_of_kappa(kappa, wratio, sigma) = log((kappa / wratio)^(sigma - 1))
block_free_mask(n, D, free_g, free_A, free_f) = (nA = D^2 - 1; m = falses(n); m[1] = free_g; m[2:1+nA] .= free_A; m[2+nA:end] .= free_f; m)
ok_status(ns) = ns in (0, -101, -102, -103)

d4 = build_d4_fixture(; policy=policy)
ctx, obj, theta0 = d4.ctx, d4.obj, d4.theta0
n = length(theta0); D = ctx.D
j = ctx.target_country
state0 = melitz_outer_state(theta0, ctx)
lambda_jj = state0.equilibrium.trade_flow[j, j] / state0.equilibrium.expenditure[j]
wratio = 1.0 / ctx.w[j]
sigma = ctx.sigma
kappa_pareto = kappa_of_g(theta0[1], wratio, sigma)
kappa_min = lambda_jj^(1 / (sigma - 1))

inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
mask_A = block_free_mask(n, D, false, true, false)
mask_f = block_free_mask(n, D, false, false, true)
mask_full = block_free_mask(n, D, false, true, true)

fixed_session = MelitzInnerSession(obj, ctx, policy)

rows = NamedTuple[]
n_nuisance_solves = Ref(0)

warm_x_A = nothing; warm_theta_A = copy(theta0)
warm_x_f = nothing; warm_theta_f = copy(theta0)

for frac in FRACTIONS
    kappa_f = kappa_pareto + frac * (kappa_min - kappa_pareto)
    g_f = g_of_kappa(kappa_f, wratio, sigma)
    theta_fixed = copy(theta0); theta_fixed[1] = g_f
    println("="^100); @printf("frac=%.2f  g=%.6f\n", frac, g_f)

    r_fixed = solve_melitz_delta!(fixed_session, theta_fixed, policy; warm_start_source=:previous)
    Delta_fixed = r_fixed isa FiniteSolved ? r_fixed.Delta : NaN
    println("  fixed A/f: $(typeof(r_fixed))  Delta=$Delta_fixed")
    flush(stdout)

    # A-only: theta STARTS from the preceding successful A-only theta (A/f block carried
    # forward), g overwritten to this fraction's target; dual warm-started from the preceding
    # successful A-only dual (nothing at frac=0.10, the first grid point).
    theta_start_A = copy(warm_theta_A); theta_start_A[1] = g_f
    resA = solve_melitz_nuisance_min_delta(ctx, obj, theta_start_A; free_mask=mask_A, radius=0.3,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
        warm_start_x=warm_x_A, policy=policy, forbid_dense_fallback=true)
    n_nuisance_solves[] += 1
    okA = ok_status(resA.nStatus) && resA.r_final.verified
    DeltaA = resA.r_final.verified ? resA.r_final.Delta : NaN
    @printf("  A-only:    nStatus=%4d  Delta=%s  ok=%s\n", resA.nStatus, isnan(DeltaA) ? "NA" : @sprintf("%.4e", DeltaA), okA)
    if okA
        global warm_x_A = resA.r_final.dual_x
        global warm_theta_A = copy(resA.theta_final)
    end
    flush(stdout)

    theta_start_f = copy(warm_theta_f); theta_start_f[1] = g_f
    resF = solve_melitz_nuisance_min_delta(ctx, obj, theta_start_f; free_mask=mask_f, radius=0.3,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
        warm_start_x=warm_x_f, policy=policy, forbid_dense_fallback=true)
    n_nuisance_solves[] += 1
    okF = ok_status(resF.nStatus) && resF.r_final.verified
    DeltaF = resF.r_final.verified ? resF.r_final.Delta : NaN
    @printf("  f-only:    nStatus=%4d  Delta=%s  ok=%s\n", resF.nStatus, isnan(DeltaF) ? "NA" : @sprintf("%.4e", DeltaF), okF)
    if okF
        global warm_x_f = resF.r_final.dual_x
        global warm_theta_f = copy(resF.theta_final)
    end
    flush(stdout)

    # full: seeded from the BETTER of THIS fraction's own A-only/f-only (theta AND dual), per
    # the original session's own convention -- falls back to a cold solve at theta_fixed only
    # if NEITHER restricted stage converged.
    candidates = NamedTuple[]
    okA && push!(candidates, (theta=resA.theta_final, dual=resA.r_final.dual_x, Delta=resA.r_final.Delta))
    okF && push!(candidates, (theta=resF.theta_final, dual=resF.r_final.dual_x, Delta=resF.r_final.Delta))
    if isempty(candidates)
        resFull = solve_melitz_nuisance_min_delta(ctx, obj, theta_fixed; free_mask=mask_full, radius=0.3,
            gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
            warm_start_x=nothing, policy=policy, forbid_dense_fallback=true)
    else
        best = candidates[argmin([c.Delta for c in candidates])]
        resFull = solve_melitz_nuisance_min_delta(ctx, obj, copy(best.theta); free_mask=mask_full, radius=0.3,
            gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
            warm_start_x=best.dual, policy=policy, forbid_dense_fallback=true)
    end
    n_nuisance_solves[] += 1
    okFull = ok_status(resFull.nStatus) && resFull.r_final.verified
    DeltaFull = resFull.r_final.verified ? resFull.r_final.Delta : NaN
    @printf("  full:      nStatus=%4d  Delta=%s  ok=%s\n", resFull.nStatus, isnan(DeltaFull) ? "NA" : @sprintf("%.4e", DeltaFull), okFull)
    flush(stdout)

    deltas = Dict{String,Float64}()
    isfinite(Delta_fixed) && (deltas["fixed"] = Delta_fixed)
    okA && (deltas["A_only"] = DeltaA)
    okF && (deltas["f_only"] = DeltaF)
    okFull && (deltas["full"] = DeltaFull)
    best_label = isempty(deltas) ? "none" : first(sort(collect(keys(deltas)), by=k -> deltas[k]))
    best_delta = isempty(deltas) ? NaN : deltas[best_label]

    if isfinite(Delta_fixed) && isfinite(best_delta)
        @assert best_delta <= Delta_fixed + 1e-9 "REGRESSION: reported best ($best_delta) worse than fixed A/f ($Delta_fixed) at frac=$frac"
    end

    push!(rows, (frac=frac, g=g_f, Delta_fixed=Delta_fixed, Delta_A_only=DeltaA, Delta_f_only=DeltaF,
                  Delta_full=DeltaFull, best_source=best_label, best_Delta=best_delta))
    println("  BEST: $best_label  Delta=$best_delta")
    flush(stdout)
end

println("\nTotal nuisance solves: ", n_nuisance_solves[], " (budget: <=12)")

open(joinpath(OUTDIR, "melitz_post_consolidation_phase5c_nuisance_continuation_rerun_2026-07-28.csv"), "w") do io
    println(io, "frac,g,Delta_fixed,Delta_A_only,Delta_f_only,Delta_full,best_source,best_Delta")
    for r in rows
        println(io, "$(r.frac),$(r.g),$(r.Delta_fixed),$(r.Delta_A_only),$(r.Delta_f_only),$(r.Delta_full),$(r.best_source),$(r.best_Delta)")
    end
end
println("\nWrote docs/key_results/melitz_post_consolidation_phase5c_nuisance_continuation_rerun_2026-07-28.csv")
println("\nDONE Phase 5c (post-consolidation, continuation-threaded rerun).")
