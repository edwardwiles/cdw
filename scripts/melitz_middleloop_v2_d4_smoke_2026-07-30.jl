# Quick D4 smoke test of the ADDENDUM-repaired middle-loop driver
# (solve_melitz_fixed_q_A_profile_v2, src/melitz/fixed_q_a_middle_loop.jl) before spending real
# D20 wall-clock time on it. Not part of the final deliverable -- a throwaway correctness check.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

policy_cap = CappedEvaluation(10.0)
FIXTURE4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta04 = build_melitz_psi_bundle(FIXTURE4; outer_parameterization=:logcutoff, policy=policy_cap,
    backend=:matrix_free, forbid_dense_fallback=true)
ctx4 = obj4.γ
session4 = MelitzInnerSession(obj4, ctx4, policy_cap)
theta_plain04 = melitz_unpower_theta_free(theta04, ctx4)
A04, f04, gpj04, fjj04, q04 = expand_free_theta_logcutoff(theta_plain04, ctx4)
nA4 = ctx4.D^2 - 1
A_free04 = theta_plain04[2:1+nA4]
sys4 = melitz_fixed_q_middle_constraint_system(theta_plain04, ctx4, obj4)

r0 = melitz_middle_objective_and_gradient!(session4, A_free04, q04, gpj04, ctx4; coordinate=:logA)
@printf("anchor start classification=%s Delta=%.6g\n", nameof(typeof(r0.classification)), r0.Delta)
@assert r0.classification isa FiniteSolved

println("\n--- test 1: v2 driver at the D4 anchor, coordinate=:logA ---")
t0 = time()
res = solve_melitz_fixed_q_A_profile_v2(session4, q04, gpj04, A_free04, ctx4;
    coordinate=:logA, max_evals=60, box=1.0, sys=sys4)
@printf("nStatus=%d incumbent_source=%s Delta_incumbent=%.8g Delta_start_verified=%.8g Delta_terminal_verified=%.8g\n",
    res.nStatus, res.incumbent_source, res.Delta_incumbent, res.Delta_start_verified, res.Delta_terminal_verified)
@printf("n_fc=%d n_ga=%d F/C/I=%d/%d/%d unique_A_points=%d unique_inner_solves=%d cache_hits=%d wall=%.2fs\n",
    res.n_fc_calls, res.n_ga_calls, res.n_finite_solved, res.n_above_cap, res.n_infinite_certified,
    res.unique_A_points, res.unique_inner_solves, res.cache_hits, res.wall_s)
@assert res.Delta_incumbent <= res.Delta_start_verified + 1e-6
@assert res.unique_inner_solves <= res.unique_A_points
@assert res.r_incumbent isa FiniteSolved
println("test 1 OK")

println("\n--- test 2: v2 driver with a DELIBERATELY BAD start (way outside feasible A region) to force AboveEvaluationCap trials and check throw/backtrack behavior ---")
rng = MersenneTwister(7)
A_bad = A_free04 .+ 3.0 .* randn(rng, length(A_free04))
A_bad_proj = melitz_project_start_to_middle_constraints(A_bad, sys4, ctx4)
r_bad_start = melitz_middle_objective_and_gradient!(session4, A_bad_proj, q04, gpj04, ctx4; coordinate=:logA)
@printf("bad-start classification=%s Delta=%.6g\n", nameof(typeof(r_bad_start.classification)), r_bad_start.Delta)
t0b = time()
res_bad = solve_melitz_fixed_q_A_profile_v2(session4, q04, gpj04, A_bad_proj, ctx4;
    coordinate=:logA, max_evals=60, box=1.0, sys=sys4)
@printf("nStatus=%d incumbent_source=%s Delta_incumbent=%s Delta_start_verified=%s wall=%.2fs n_classified=%d (F/C/I=%d/%d/%d)\n",
    res_bad.nStatus, res_bad.incumbent_source, string(res_bad.Delta_incumbent), string(res_bad.Delta_start_verified),
    res_bad.wall_s, length(res_bad.eval_log), res_bad.n_finite_solved, res_bad.n_above_cap, res_bad.n_infinite_certified)
if isfinite(res_bad.Delta_start_verified)
    @assert res_bad.Delta_incumbent <= res_bad.Delta_start_verified + 1e-6
end
println("test 2 OK (v2 handled a bad/capped start without crashing; see printed classification counts)")

println("\n--- test 3: repeat request at the identical A point (cache-hit check) ---")
stats_check = MelitzMiddleCacheStats()
ec = MelitzExactPointCache(); bc = MelitzMiddleBadPointCache()
r_a = melitz_middle_objective_and_gradient_cached!(session4, A_free04, q04, gpj04, ctx4, ec, bc, stats_check; coordinate=:logA)
r_b = melitz_middle_objective_and_gradient_cached!(session4, A_free04, q04, gpj04, ctx4, ec, bc, stats_check; coordinate=:logA)
@printf("first: cache_hit=%s  second (identical x): cache_hit=%s\n", r_a.cache_hit, r_b.cache_hit)
@assert r_a.cache_hit == false
@assert r_b.cache_hit == true
@assert isapprox(r_a.Delta, r_b.Delta; atol=1e-12)
@assert isapprox(r_a.grad_free, r_b.grad_free; atol=1e-10)
@assert stats_check.n_actual_inner_solves == 1
@assert stats_check.n_cache_hits == 1
@assert length(stats_check.seen_keys) == 1
println("test 3 OK (repeat point reused cache, no second inner solve)")

println("\nALL D4 SMOKE TESTS PASSED")
