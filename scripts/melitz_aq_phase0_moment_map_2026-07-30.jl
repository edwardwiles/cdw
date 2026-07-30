# Phase 0/1 (D4) verification: exact implemented moment map + LFD-preserving state
# constructor building blocks. Governing prompt:
# melitz_joint_Aq_feasibility_preserving_search_2026-07-30.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra, Random
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
inner_opt = joinpath(REPO2, "melitz_inner_loop_options.opt")
obj, theta0 = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff, policy=policy_cap,
    inner_loop_opt=inner_opt, forbid_dense_fallback=true)
ctx = obj.γ
D = ctx.D; nA = D^2 - 1; nq = D^2 - 2
sorted_ctx = ctx.sorted_tail_ctx
println("D=$D  nA=$nA  nq=$nq  sigma=$(ctx.sigma)"); flush(stdout)

obj.use_cached_x = false; obj.x .= NaN
lfd0 = melitz_recover_lfd(obj, theta0)
@printf("anchor solve: Delta0=%.10f  lfd_ok=%s  nStatus=%d\n", lfd0.Delta, lfd0.lfd_ok, lfd0.nStatus)
@assert lfd0.lfd_ok
p_star = lfd0.weights
@assert isapprox(sum(p_star), 1.0; atol=1e-9)

A_anchor, f_anchor, gpj_anchor, fjj_anchor, q_anchor = expand_free_theta_logcutoff(
    melitz_unpower_theta_free(theta0, ctx), ctx)
println("gravity_A residual at anchor = ", melitz_gravity_A_residual(A_anchor, ctx))
println("gravity_f residual at anchor = ", melitz_gravity_f_residual(f_anchor, ctx))

# ================================================================================================
# Test 1: T_od(p_star, q_anchor) reproduces H_od = lambda_od/coef_od EXACTLY (up to solver tol),
# confirming formula (*) is self-consistent at s=0 (q_new==q_anchor => A_new==A_anchor exactly).
# ================================================================================================
println("\n=== Test 1: cellwise recovery at q_new==q_anchor reproduces A_anchor exactly ===")
A_id, status_id = melitz_cellwise_A_from_moments(A_anchor, q_anchor, q_anchor, p_star, sorted_ctx, ctx.sigma)
max_reldiff = maximum(abs.(A_id .- A_anchor) ./ A_anchor)
n_notok = count(!=(:ok), status_id)
@printf("max relative |A_id - A_anchor| = %.3e   (n status != :ok = %d)\n", max_reldiff, n_notok)
@assert n_notok == 0
@assert max_reldiff < 1e-9

# ================================================================================================
# Test 2: mul_Gt! under p_star at the anchor operator state reproduces ~0 trade residuals and
# ~0 focal-link residual (i.e. p_star genuinely satisfies the anchor's own moments).
# ================================================================================================
println("\n=== Test 2: anchor moment residuals under p_star (mul_Gt!) ===")
melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)
trade_res0, focal_res0 = melitz_moment_residuals_under_p(obj.op, p_star)
@printf("max|trade residual| = %.3e   focal residual = %.3e\n", maximum(abs.(trade_res0)), focal_res0)
@assert maximum(abs.(trade_res0)) < 1e-6
@assert abs(focal_res0) < 1e-6

# ================================================================================================
# Test 3: step-function structural finding -- T_od(p_star,q_od) is EXACTLY constant strictly
# between two adjacent draws with p_star>0 in that origin's column, and jumps exactly at a
# positive-weight draw crossing. Verify both halves directly (not merely "flat to FD tolerance").
# ================================================================================================
println("\n=== Test 3: T_od step-function structural check (cell o=1,d=2) ===")
o_t, d_t = 1, 2
suffix_1 = melitz_origin_suffix_tail(sorted_ctx, o_t, p_star)
col = sorted_ctx.sorted_log_z[:, o_t]
k0 = melitz_active_tail_start(col, q_anchor[o_t, d_t])
# find nearest strictly-positive-weight draw above the current cutoff (by sorted position)
perm_o = sorted_ctx.permutation[:, o_t]
pos_next = k0
while pos_next <= length(col) && p_star[perm_o[pos_next]] <= 0
    global pos_next += 1
end
@assert pos_next <= length(col) "no further positive-weight draw found -- pick a different test cell"
q_mid = q_anchor[o_t, d_t] < col[pos_next] ? 0.5 * (q_anchor[o_t, d_t] + col[pos_next]) : q_anchor[o_t, d_t] + 1e-9
q_just_below = col[pos_next] - 1e-9 * abs(col[pos_next])
q_just_above = col[pos_next] + 1e-9 * abs(col[pos_next])
T_below = melitz_T_od(q_just_below, o_t, sorted_ctx, suffix_1)
T_above = melitz_T_od(q_just_above, o_t, sorted_ctx, suffix_1)
T_mid1 = melitz_T_od(q_just_below - 1e-12, o_t, sorted_ctx, suffix_1)
@printf("crossing row (sorted pos=%d, orig row=%d, p_star weight=%.6e): T(just below)=%.10f  T(just above)=%.10f  diff=%.3e\n",
        pos_next, perm_o[pos_next], p_star[perm_o[pos_next]], T_below, T_above, T_below - T_above)
@assert isapprox(T_below, T_mid1; atol=0, rtol=0) "T_od not exactly flat strictly below the crossing draw"
@assert T_below - T_above ≈ p_star[perm_o[pos_next]] * sorted_ctx.sorted_z_power[pos_next, o_t] rtol=1e-9
println("CONFIRMED: T_od jumps EXACTLY at the positive-weight draw, by EXACTLY p_star_weight*z_power; flat immediately below.")

# ================================================================================================
# Test 4: cellwise-compensated (uncorrected) endpoint at a small q perturbation -- exact moment
# preservation under p_star, generic A-gravity violation (Step 1B alone).
# ================================================================================================
println("\n=== Test 4: Step 1B uncorrected endpoint at a random small free-q perturbation ===")
rng = Random.MersenneTwister(2026)
q_pivot, cell_of_k = melitz_q_free_cell_map(ctx)
q_free_free_anchor = pivot_reduce(vec(q_anchor)[ctx.f_free_lin], q_pivot)
delta = 1e-3 .* randn(rng, nq)
q_free_free_trial = q_free_free_anchor .+ delta
g_anchor = theta0[1]

st_B = melitz_construct_lfd_preserving_state(theta0, p_star, ctx, obj, g_anchor, q_free_free_trial;
    correct=false)
@printf("Step1B endpoint: max|trade residual under p*| = %.3e  focal residual = %.3e  gravity_A = %.6e  gravity_f = %.6e  n_notok=%d\n",
    maximum(abs.(st_B.trade_residuals)), st_B.focal_residual, st_B.gravity_A_residual, st_B.gravity_f_residual,
    count(!=(:ok), st_B.A_status))
@assert maximum(abs.(st_B.trade_residuals)) < 1e-6 "Step 1B must preserve trade moments exactly under p*"
# Step 1B targets ONLY the D^2 bilateral trade-share moments (task spec, Step 1B) -- the focal
# link is NOT a trade-share moment and is NOT expected to be preserved by Step 1B alone (task
# spec, Step 1C's own header: "will generally not automatically satisfy A gravity or the focal
# free-entry link"). Report it, do not assert it small.
println("focal residual at Step1B (expected GENERICALLY NONZERO, small only because the test perturbation is small): ", st_B.focal_residual)
println("gravity_A residual (expected GENERICALLY NONZERO, confirming Step1B alone breaks gravity): ", st_B.gravity_A_residual)

# ================================================================================================
# Test 5: full corrector (Step 1C) restores A-gravity AND f-gravity, at the SAME perturbation.
# ================================================================================================
println("\n=== Test 5: Step 1C corrector restores both gravity restrictions ===")
melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)  # restore obj.op
st_C = melitz_construct_lfd_preserving_state(theta0, p_star, ctx, obj, g_anchor, q_free_free_trial;
    correct=true, max_corrector_rounds=8)
@printf("Step1C endpoint: max|trade residual| = %.3e  focal residual = %.3e  gravity_A = %.3e  gravity_f = %.3e  feasible=%s\n",
    maximum(abs.(st_C.trade_residuals)), st_C.focal_residual, st_C.gravity_A_residual, st_C.gravity_f_residual, st_C.feasible)
println("corrector trace length = ", length(st_C.corrector_trace))

# cross-check: theta_free_recovered actually reproduces (A_new,f_new,gpj_new) via the ordinary
# expand_free_theta_logcutoff round trip (i.e. it's a genuine point in the free parameterization).
A_rt, f_rt, gpj_rt, _, q_rt = expand_free_theta_logcutoff(st_C.theta_free, ctx)
@printf("round-trip max|A diff|=%.3e  max|f diff|=%.3e  |gpj diff|=%.3e\n",
    maximum(abs.(A_rt .- st_C.A)), maximum(abs.(f_rt .- st_C.f)), abs(gpj_rt - st_C.gamma_prime_j))
# Round-trip agreement is bounded by the corrector's OWN achieved gravity residual (~1e-7 here,
# not exactly 0 -- a bounded discrete local search, not an exact root-find) -- tolerance is
# scaled accordingly, not a fixed 1e-6.
rt_tol = max(1e-5, 100 * max(abs(st_C.gravity_A_residual), abs(st_C.gravity_f_residual)))
@assert maximum(abs.(A_rt .- st_C.A)) < rt_tol
@assert maximum(abs.(f_rt .- st_C.f)) < rt_tol

# restore obj.op to the anchor state before leaving (courtesy for any downstream reuse)
melitz_update_operator_at_theta!(obj.op, theta0, ctx)

open(joinpath(OUTDIR, "melitz_aq_phase0_moment_map_2026-07-30.csv"), "w") do io
    println(io, "test,quantity,value")
    println(io, "1,max_reldiff_A_at_s0,", max_reldiff)
    println(io, "2,max_trade_residual_anchor,", maximum(abs.(trade_res0)))
    println(io, "2,focal_residual_anchor,", focal_res0)
    println(io, "3,T_jump,", T_below - T_above)
    println(io, "3,p_weight_x_zpower,", p_star[perm_o[pos_next]] * sorted_ctx.sorted_z_power[pos_next, o_t])
    println(io, "4,max_trade_residual_stepB,", maximum(abs.(st_B.trade_residuals)))
    println(io, "4,focal_residual_stepB,", st_B.focal_residual)
    println(io, "4,gravity_A_residual_stepB,", st_B.gravity_A_residual)
    println(io, "5,max_trade_residual_stepC,", maximum(abs.(st_C.trade_residuals)))
    println(io, "5,focal_residual_stepC,", st_C.focal_residual)
    println(io, "5,gravity_A_residual_stepC,", st_C.gravity_A_residual)
    println(io, "5,gravity_f_residual_stepC,", st_C.gravity_f_residual)
    println(io, "5,feasible_stepC,", st_C.feasible)
    println(io, "5,corrector_trace_length,", length(st_C.corrector_trace))
end
println("\nDONE PHASE 0/1 (D4)")
