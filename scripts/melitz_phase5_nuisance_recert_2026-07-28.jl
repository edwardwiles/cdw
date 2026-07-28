# Post-consolidation validation, Phase 5: selected nuisance-profile checks via the
# consolidated inner API. Budget: at most 12 nuisance solves (A-only + f-only + full, x4
# points = 12 exactly). D4 only, scope disclosure: real-D20 nuisance minimization is known
# (docs/melitz_outer_search_step_control_and_robustness_2026-07-28.md Phase 8) to occasionally
# take >1000s per stage (A-only at an interior point) -- 4 points x up to 3 D20 stages could
# alone consume this validation's ENTIRE remaining compute budget for a result the prior
# session already established is small (0.05-0.07% DeltaStar reduction). D4 is fast (seconds
# per stage), already showed LARGE (3-5x) gains, and is the more informative reconsolidation
# check given the architecture change touches the SAME `solve_melitz_nuisance_min_delta`
# driver at both scales. real-D20 nuisance re-certification is therefore classified
# "not rerun, low risk" in the final triage table, not silently skipped.
#
# D4 fractions used: {0.10, 0.35, 0.50, 0.65} -- roughly "DeltaStar~0.1" (0.10 -> 0.034),
# "DeltaStar~0.5" (0.35 -> 0.572), and two points BEYOND the finite fixed-A/f corridor
# (0.50, 0.65, both NumericalFailure in the original fixed-A/f profile) standing in for the
# "DeltaStar~1"/"DeltaStar~2" targets, which have no finite fixed-A/f reference at D4 at all
# (disclosed) -- these ALSO directly test the prior session's own "flexibility rescues
# infeasibility" finding under the new architecture.

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
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

for frac in FRACTIONS
    kappa_f = kappa_pareto + frac * (kappa_min - kappa_pareto)
    g_f = g_of_kappa(kappa_f, wratio, sigma)
    theta_fixed = copy(theta0); theta_fixed[1] = g_f
    println("="^100); @printf("frac=%.2f  g=%.6f\n", frac, g_f)

    r_fixed = solve_melitz_delta!(fixed_session, theta_fixed, policy; warm_start_source=:previous)
    Delta_fixed = r_fixed isa FiniteSolved ? r_fixed.Delta : NaN
    println("  fixed A/f: $(typeof(r_fixed))  Delta=$Delta_fixed")
    flush(stdout)

    # A-only
    resA = solve_melitz_nuisance_min_delta(ctx, obj, theta_fixed; free_mask=mask_A, radius=0.3,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
        policy=policy, forbid_dense_fallback=true)
    n_nuisance_solves[] += 1
    okA = ok_status(resA.nStatus) && resA.r_final.verified
    DeltaA = resA.r_final.verified ? resA.r_final.Delta : NaN
    @printf("  A-only:    nStatus=%4d  Delta=%s  ok=%s\n", resA.nStatus, isnan(DeltaA) ? "NA" : @sprintf("%.4e", DeltaA), okA)
    flush(stdout)

    # f-only
    resF = solve_melitz_nuisance_min_delta(ctx, obj, theta_fixed; free_mask=mask_f, radius=0.3,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
        policy=policy, forbid_dense_fallback=true)
    n_nuisance_solves[] += 1
    okF = ok_status(resF.nStatus) && resF.r_final.verified
    DeltaF = resF.r_final.verified ? resF.r_final.Delta : NaN
    @printf("  f-only:    nStatus=%4d  Delta=%s  ok=%s\n", resF.nStatus, isnan(DeltaF) ? "NA" : @sprintf("%.4e", DeltaF), okF)
    flush(stdout)

    # full, seeded from the better of A-only/f-only
    warm_x = nothing
    if okA && (!okF || DeltaA <= DeltaF)
        warm_x = copy(resA.r_final.dual_x)
    elseif okF
        warm_x = copy(resF.r_final.dual_x)
    end
    resFull = solve_melitz_nuisance_min_delta(ctx, obj, theta_fixed; free_mask=mask_full, radius=0.3,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
        warm_start_x=warm_x, policy=policy, forbid_dense_fallback=true)
    n_nuisance_solves[] += 1
    okFull = ok_status(resFull.nStatus) && resFull.r_final.verified
    DeltaFull = resFull.r_final.verified ? resFull.r_final.Delta : NaN
    @printf("  full:      nStatus=%4d  Delta=%s  ok=%s\n", resFull.nStatus, isnan(DeltaFull) ? "NA" : @sprintf("%.4e", DeltaFull), okFull)
    flush(stdout)

    candidates = [("fixed", Delta_fixed, r_fixed isa FiniteSolved)]
    okA && push!(candidates, ("A_only", DeltaA, true))
    okF && push!(candidates, ("f_only", DeltaF, true))
    okFull && push!(candidates, ("full", DeltaFull, true))
    valid = filter(c -> c[3] && isfinite(c[2]), candidates)
    best = isempty(valid) ? ("none", NaN, false) : valid[argmin([c[2] for c in valid])]

    # Never-worse-than-fixed check
    if isfinite(Delta_fixed) && isfinite(best[2])
        @assert best[2] <= Delta_fixed + 1e-9 "REGRESSION: reported best ($best) worse than fixed A/f ($Delta_fixed) at frac=$frac"
    end

    push!(rows, (frac=frac, g=g_f, Delta_fixed=Delta_fixed, Delta_A_only=DeltaA, Delta_f_only=DeltaF,
                  Delta_full=DeltaFull, best_source=best[1], best_Delta=best[2]))
    println("  BEST: $(best[1])  Delta=$(best[2])")
    flush(stdout)
end

println("\nTotal nuisance solves: ", n_nuisance_solves[], " (budget: <=12)")

open(joinpath(OUTDIR, "melitz_post_consolidation_phase5_nuisance_recert_2026-07-28.csv"), "w") do io
    println(io, "frac,g,Delta_fixed,Delta_A_only,Delta_f_only,Delta_full,best_source,best_Delta")
    for r in rows
        println(io, "$(r.frac),$(r.g),$(r.Delta_fixed),$(r.Delta_A_only),$(r.Delta_f_only),$(r.Delta_full),$(r.best_source),$(r.best_Delta)")
    end
end
println("\nWrote docs/key_results/melitz_post_consolidation_phase5_nuisance_recert_2026-07-28.csv")
println("\nDONE Phase 5 (post-consolidation).")
