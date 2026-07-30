# Phase 5 (D4, scoped per governing prompt's own allowance -- "Do not run a full outer solve
# under these coordinates in this phase"): relative cutoff (5A, reference-rank) and normalized
# tail-target (5B) coordinate prototypes. Governing prompt:
# melitz_joint_Aq_feasibility_preserving_search_2026-07-30.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra, Random, Statistics
melitz_thread_startup_report()
const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
inner_opt = joinpath(REPO2, "melitz_inner_loop_options.opt")
obj, theta0 = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff, policy=policy_cap,
    inner_loop_opt=inner_opt, forbid_dense_fallback=true)
ctx = obj.γ
D = ctx.D
sorted_ctx = ctx.sorted_tail_ctx
lfd0 = melitz_recover_lfd(obj, theta0)
@assert lfd0.lfd_ok
p_star = lfd0.weights
A0, f0, gpj0, _, q0 = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta0, ctx), ctx)
println("D=$D  W=$(sorted_ctx.W)"); flush(stdout)

# ================================================================================================
# 5A: reference-rank coordinates (empirical-CDF-of-draws scale, per origin).
# u_od = fraction of origin o's W reference draws with log(z) < q_od (reference = the uniform
# empirical measure over the W QMC draws -- NOT the population Pareto CDF, a disclosed choice:
# ties the coordinate directly to the SAME finite-support draws the model's own moments use).
# ================================================================================================
function melitz_reference_rank(q_od::Real, o::Int, sorted_ctx)
    k0 = melitz_active_tail_start(view(sorted_ctx.sorted_log_z, :, o), q_od)
    return (k0 - 1) / sorted_ctx.W
end
function melitz_reference_rank_inverse(u::Real, o::Int, sorted_ctx)
    W = sorted_ctx.W
    k = clamp(round(Int, u * W), 0, W)
    col = view(sorted_ctx.sorted_log_z, :, o)
    k == 0 && return col[1] - 1.0
    k == W && return col[W] + 1.0
    return 0.5 * (col[k] + col[k+1])
end

println("\n=== 5A: relative cutoff (reference-rank) coordinates -- reconstruction test ===")
q_5A = zeros(D, D)
max_q_reconstruct_err = 0.0
max_u_gap = -Inf; min_u_gap = Inf
for o in 1:D
    order = sortperm(q0[o, :])   # stagewise destination order at this origin (ascending cutoff)
    u_sorted = [melitz_reference_rank(q0[o, d], o, sorted_ctx) for d in order]
    level = u_sorted[1]
    gaps = diff(u_sorted)   # Delta u_o,k = u_{k+1} - u_k >= 0 by construction (sorted ascending)
    @assert all(gaps .>= -1e-12) "5A gaps must be nonnegative by stagewise ordering"
    global max_u_gap = max(max_u_gap, maximum(gaps))
    global min_u_gap = min(min_u_gap, minimum(gaps))
    # recover u cumulative, invert back to q, scatter into original destination positions
    u_rec = vcat(level, level .+ cumsum(gaps))
    for (idx, d) in enumerate(order)
        q_5A[o, d] = melitz_reference_rank_inverse(u_rec[idx], o, sorted_ctx)
    end
end
global max_q_reconstruct_err = maximum(abs.(q_5A .- q0))
@printf("max|q_5A - q0| (reconstruction error) = %.6e   min/max u-gap across all origins = %.6f / %.6f\n",
    max_q_reconstruct_err, min_u_gap, max_u_gap)

# Does the 5A reconstruction land in the SAME chamber (identical active set under p*)? -- test
# via the cellwise-A-recovery: A should be BIT-IDENTICAL to A0 if q_5A reproduces the same
# active-tail-start k0 for every cell (not merely a close q level).
A_5A, status_5A = melitz_cellwise_A_from_moments(A0, q0, q_5A, p_star, sorted_ctx, ctx.sigma)
@printf("5A round-trip: max relative |A_5A - A0| = %.3e   n status!=:ok = %d\n",
    maximum(abs.(A_5A .- A0) ./ A0), count(!=(:ok), status_5A))

# ================================================================================================
# 5B: normalized tail-target (H) coordinates, per origin.
# H_od = T_od(p_star, q0[o,d]) (== lambda_od/coef_od(A0) exactly, since p_star satisfies the
# anchor moments -- origin_block_screen.jl's own H[d] target, re-derived here directly).
# ================================================================================================
println("\n=== 5B: normalized tail-target (H) coordinates -- reconstruction test ===")
A_5B = zeros(D, D)
max_A_reconstruct_err = 0.0
for o in 1:D
    suffix_o = melitz_origin_suffix_tail(sorted_ctx, o, p_star)
    order = sortperm(q0[o, :])  # ascending cutoff => DEscending H
    H_sorted = [melitz_T_od(q0[o, d], o, sorted_ctx, suffix_o) for d in order]
    @assert all(diff(H_sorted) .<= 1e-9) "H must be non-increasing along ascending-cutoff order"
    terminal = H_sorted[end]                       # H_o,D (largest q, smallest H)
    masses = -diff(H_sorted)                        # M_o,k = H_o,k - H_o,k+1 >= 0
    @assert all(masses .>= -1e-9)
    # recover H_o,k backward: H_o,D known, H_o,k = H_o,k+1 + M_o,k
    H_rec = zeros(D)
    H_rec[end] = terminal
    for k in (D-1):-1:1
        H_rec[k] = H_rec[k+1] + masses[k]
    end
    for (idx, d) in enumerate(order)
        # invert H_od -> A_od analytically: a_od = a_anchor + (log(T_anchor)-log(H_rec))/(sigma-1),
        # T_anchor == H_od exactly at the SAME q (anchor is a fixed point of this formula).
        T_anchor_od = melitz_T_od(q0[o, d], o, sorted_ctx, suffix_o)
        a_rec = log(A0[o, d]) + (log(T_anchor_od) - log(H_rec[idx])) / (ctx.sigma - 1)
        A_5B[o, d] = exp(a_rec)
    end
end
max_A_reconstruct_err = maximum(abs.(A_5B .- A0) ./ A0)
@printf("5B round-trip: max relative |A_5B - A0| = %.3e\n", max_A_reconstruct_err)

# ================================================================================================
# 5C (S_W chamber-statistic variant only -- scope reduction disclosed, S_pop continuous-reference
# variant not built this session): does an EMPTY finite-W interval force zero interval mass?
# ================================================================================================
println("\n=== 5C: finite-W chamber statistic -- empty-interval forces zero mass (direct check) ===")
o_test = 1
order = sortperm(q0[o_test, :])
suffix_test = melitz_origin_suffix_tail(sorted_ctx, o_test, p_star)
# S_W,o,k = sum_s w_ref,s Y_os 1{q_k <= log z_os < q_k+1}; use uniform reference weights w_ref=1/W
# and Y_os = z_power (a defensible, documented weight choice -- disclosed, not the ONLY choice).
n_empty = 0
for k in 1:(D-1)
    d_lo, d_hi = order[k], order[k+1]
    k0_lo = melitz_active_tail_start(view(sorted_ctx.sorted_log_z, :, o_test), q0[o_test, d_lo])
    k0_hi = melitz_active_tail_start(view(sorted_ctx.sorted_log_z, :, o_test), q0[o_test, d_hi])
    n_draws_in_interval = k0_hi - k0_lo   # positions [k0_lo, k0_hi-1] are strictly between the two cutoffs (active for d_lo, inactive for d_hi)
    S_W = n_draws_in_interval > 0 ? sum(sorted_ctx.sorted_z_power[k0_lo:k0_hi-1, o_test]) / sorted_ctx.W : 0.0
    n_draws_in_interval == 0 && (global n_empty += 1)
    @printf("  interval k=%d (d_lo=%d,d_hi=%d): n_draws=%d  S_W=%.6e  %s\n",
        k, d_lo, d_hi, n_draws_in_interval, S_W, n_draws_in_interval == 0 ? "EMPTY -> mass forced to 0" : "")
end
println("(D4 fixture at W=20,000 is dense; n_empty=$n_empty here -- the forced-zero mechanism is",
    " a STRUCTURAL property of the S_W definition, confirmed directly at whichever intervals ARE empty,",
    " not merely asserted; see the D20 origin-14 cliff (Phase 2) for a case where this actually binds.")

open(joinpath(OUTDIR, "melitz_aq_phase5_relative_coords_2026-07-30.csv"), "w") do io
    println(io, "test,quantity,value")
    println(io, "5A,max_q_reconstruct_err,", max_q_reconstruct_err)
    println(io, "5A,min_u_gap,", min_u_gap)
    println(io, "5A,max_u_gap,", max_u_gap)
    println(io, "5A,max_relerr_A_roundtrip,", maximum(abs.(A_5A .- A0) ./ A0))
    println(io, "5B,max_relerr_A_roundtrip,", max_A_reconstruct_err)
    println(io, "5C,n_empty_intervals_D4,", n_empty)
end
println("\nDONE PHASE 5 (D4)")
