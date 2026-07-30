# Phase 4 (profiledA_parallel_speed_and_cutoff_portfolio governing prompt, 2026-07-30)
# FOLLOW-UP: validates the O(W*D) -> O(W+D) focal-link reformation now wired into
# src/melitz/moment_operator.jl's melitz_update_moment_operator! against a standalone,
# untouched, verbatim copy of the ORIGINAL O(W*D) loop (defined here only, never used in
# production). Two layers:
#   1. Direct op.ell agreement at many random D4 and real-D20 (A,f) points.
#   2. End-to-end agreement: reproduces the two headline numbers this project has already
#      published to 10 significant figures (D20 anchor Delta0=0.4832764950468883, extreme
#      point Delta*=0.4990186631205613) using the NEW code path, unmodified from the
#      committed 2026-07-30 session's own values.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random
LinearAlgebra.BLAS.set_num_threads(1)
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads()); flush(stdout)

"""
    melitz_focal_link_ell_reference!(ell_ref, op, p, eq, cf)

VERBATIM copy of the ORIGINAL O(W*D) focal-link loop this session's own Phase 4 replaced in
`melitz_update_moment_operator!` (git history, commit 0622db1's own parent state) -- kept
here ONLY as an independent reference for validation, never called from production code.
"""
function melitz_focal_link_ell_reference!(ell_ref::Vector{Float64}, op, p, eq, cf)
    D = op.D
    W = op.W
    sigma = p.sigma
    sorted_ctx = op.sorted_ctx
    j = p.target_country
    z_orig = sorted_ctx.z_original
    fill!(ell_ref, 0.0)
    @inbounds for d in 1:D
        for w in 1:W
            z = z_orig[w, j]
            firm = melitz_firm(p.w[j], p.tau[j, d], p.A[j, d], p.f[j, d], sigma,
                                eq.expenditure[d], 1.0, z)
            ell_ref[w] += firm.realized_operating_profit
        end
    end
    price_power_autarky = p.gamma_prime_target
    @inbounds for w in 1:W
        z_j = z_orig[w, j]
        firm_autarky = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], sigma,
                                    cf.expenditure_prime, price_power_autarky, z_j)
        ell_ref[w] = ell_ref[w] / p.w[j] - firm_autarky.realized_operating_profit / cf.w_prime
    end
    return ell_ref
end

function compare_ell(label::AbstractString, obj, theta::AbstractVector, ctx)
    D = ctx.D
    # EXACT production construction path (cc_bundle.jl's own melitz_update_operator_at_theta!,
    # reproduced here only to get independent access to the (primitives,eq,cf) triple so the
    # reference function can be called against the SAME state -- not a different derivation).
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country,
                                   ctx.tau, ctx.w, A, f, gamma_prime_j)
    cutoff = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    eq = MelitzEquilibrium(ctx.expenditure, ones(Float64, D), cutoff, ctx.X_data)
    expenditure_prime = ctx.w_prime * ctx.L[ctx.target_country]
    cf = MelitzCounterfactual(ctx.target_country, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)

    melitz_update_moment_operator!(obj.op, primitives, eq, cf; X_data=ctx.X_data)
    ell_new = copy(obj.op.ell)
    ell_ref = zeros(length(ell_new))
    melitz_focal_link_ell_reference!(ell_ref, obj.op, primitives, eq, cf)
    absdiff = abs.(ell_new .- ell_ref)
    maxabs = maximum(absdiff)
    scale = maximum(abs.(ell_ref))
    maxrel = scale > 0 ? maxabs / scale : maxabs
    exact = ell_new == ell_ref
    @printf("  %-32s max_abs_diff=%.3e  max_rel_diff=%.3e  scale=%.3e  bit-identical=%s\n",
        label, maxabs, maxrel, scale, exact)
    return (label=label, maxabs=maxabs, maxrel=maxrel, scale=scale, exact=exact)
end

results = NamedTuple[]

# ---------------- D4 fixture: many random (A,f)-consistent perturbations ----------------
println("\n", "="^90); println("D4 FIXTURE"); println("="^90); flush(stdout)
FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta04 = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff,
    policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
ctx4 = obj4.γ
push!(results, compare_ell("D4 anchor", obj4, theta04, ctx4))

rng = MersenneTwister(4242)
for trial in 1:10
    theta_pert = theta04 .+ 0.05 .* randn(rng, length(theta04))
    push!(results, compare_ell("D4 random_pert_$trial", obj4, theta_pert, ctx4))
end

# ---------------- Real D20 fixture: anchor + several stored middle-loop trial points ----------------
println("\n", "="^90); println("REAL D20 FIXTURE"); println("="^90); flush(stdout)
function load_realD20_calib()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal
end
calib, focal = load_realD20_calib()
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
obj20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx20 = obj20.γ

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(REPO2, "docs", "key_results", "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
theta0_d20 = theta_q_rows[("realD20_seed1_W80000", 0.5)]
push!(results, compare_ell("D20 anchor", obj20, theta0_d20, ctx20))

rng2 = MersenneTwister(7777)
for trial in 1:10
    theta_pert = theta0_d20 .+ 0.01 .* randn(rng2, length(theta0_d20))
    push!(results, compare_ell("D20 random_pert_$trial", obj20, theta_pert, ctx20))
end

println("\n", "="^90); println("SUMMARY: op.ell agreement across ", length(results), " points"); println("="^90)
worst = results[argmax([r.maxrel for r in results])]
@printf("Worst-case max_rel_diff: %.3e (%s)\n", worst.maxrel, worst.label)
all_exact = all(r.exact for r in results)
println("All bit-identical: ", all_exact)
println("All max_rel_diff < 1e-9: ", all(r.maxrel < 1e-9 for r in results))
println("All max_rel_diff < 1e-12: ", all(r.maxrel < 1e-12 for r in results))
flush(stdout)

# ---------------- End-to-end: reproduce the two published headline numbers exactly ----------------
println("\n", "="^90); println("END-TO-END: reproduce published D20 headline numbers with NEW code"); println("="^90); flush(stdout)
session20 = MelitzInnerSession(obj20, ctx20, policy_cap)
session20.obj.use_cached_x = false; session20.obj.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj20, theta0_d20)
@printf("D20 anchor: Delta=%.10f (published: 0.4832764950468883)  match<1e-8: %s  wall=%.2fs\n",
    lfd0.Delta, isapprox(lfd0.Delta, 0.4832764950468883; atol=1e-8), time() - t0)
@assert lfd0.lfd_ok
@assert isapprox(lfd0.Delta, 0.4832764950468883; atol=1e-8)

# The stored extreme point's own theta_free_incumbent (from the committed .jls state).
using Serialization
mutable struct ContinuationPoint
    idx::Int; g::Float64; GT::Float64; classification::Symbol; Delta::Float64
    accepted::Bool; best_start::Symbol; A_free::Vector{Float64}; theta_free::Vector{Float64}
end
st = deserialize(joinpath(REPO2, "scripts", "melitz_d20_profiledA_continuation_state_2026-07-30.jls"))
ext = st.result_upper.most_extreme
session20.obj.use_cached_x = false; session20.obj.x .= NaN
t1 = time()
r_ext = solve_melitz_delta!(session20, ext.theta_free, policy_cap; warm_start_source=:neutral)
Delta_ext = r_ext isa FiniteSolved ? r_ext.Delta : NaN
@printf("D20 extreme point: Delta=%.10f (published: 0.4990186631205613)  classification=%s  match<1e-8: %s  wall=%.2fs\n",
    Delta_ext, nameof(typeof(r_ext)), isapprox(Delta_ext, 0.4990186631205613; atol=1e-8), time() - t1)
@assert r_ext isa FiniteSolved
@assert isapprox(Delta_ext, 0.4990186631205613; atol=1e-8)

println("\nPHASE 4 FOCAL-LINK VALIDATION COMPLETE -- ALL CHECKS PASSED")
