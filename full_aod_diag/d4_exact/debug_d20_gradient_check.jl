using Random
using LinearAlgebra: norm

include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0

ctx_fixed = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
theta_star = 1.0 / ctx_fixed.μHat
sigma = ctx_fixed.σ
theta_min = 2 * (sigma - 1) * 1.05
theta_max = 3 * theta_star
ctx = make_flexible_theta(ctx_fixed; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :theta_decoupled_aspace)
xy = precompute_aspace_XY(ctx)
Ddest = _flex_ddest(ctx); D = ctx.D
n_free_A = D * Ddest - 1
rsc = build_ranged_screen_context(ctx)

x_free_fixed = CS.pack_free(ctx_fixed.θ0_up, ctx_fixed.m)
gp0 = x_free_fixed[1]
logA_full0 = log.(reshape(x_free_fixed[2:end], D, Ddest))
pgc0 = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
w_ext_start_a = reduce_to_w_ext_A(theta_star, gp0, logA_full0, pgc0, xy)

r0, d0 = screened_eval_flexible_A_verify(w_ext_start_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
lp("base: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual)

ctx_frozen = freeze_theta_ctx(ctx, d0.mu)
pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
pe_here = pivot_elim_from_cache(pgc, d0.mu)
xf_reduced = vcat(d0.gp, d0.xf[3:end])
base = BaseDualState(xf_reduced, r0.θ_full, r0.zeta, r0.lambda, copy(ctx.obj.arg1), r0.inner_status)
grad_pool = build_grad_workspace_pool(W)
lfix_c_ws = build_lfix_factorized_workspace(D, Ddest, W)
gfull_reduced, meta_g = composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws;
    base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
lp("gfull_reduced length=", length(gfull_reduced), " (expect D*Ddest=", D*Ddest, ")")
lp("gfull_reduced[1] (gp grad, z-space, pre-rescale)=", gfull_reduced[1])
predicted_g_a = gfull_reduced[2:end] .* (-d0.theta)
lp("predicted_g_a length=", length(predicted_g_a), " (expect D*Ddest-1=", n_free_A, ")")
lp("norm(predicted_g_a)=", norm(predicted_g_a), " max|predicted_g_a|=", maximum(abs.(predicted_g_a)))

for (trial, seed, h_fd) in ((1, 4242, 1e-4), (2, 4242, 1e-5), (3, 4242, 1e-6), (4, 777, 1e-4))
    rng = MersenneTwister(seed)
    dir = randn(rng, n_free_A); dir ./= norm(dir)
    w_plus = copy(w_ext_start_a); w_plus[3:end] .+= h_fd .* dir
    w_minus = copy(w_ext_start_a); w_minus[3:end] .-= h_fd .* dir
    r_plus, _ = screened_eval_flexible_A_verify(w_plus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
    r_minus, _ = screened_eval_flexible_A_verify(w_minus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
    fd_directional = (r_plus.Delta_dual - r_minus.Delta_dual) / (2 * h_fd)
    predicted_directional = sum(predicted_g_a .* dir)
    rel_err = abs(fd_directional - predicted_directional) / max(abs(fd_directional), 1e-8)
    lp("trial ", trial, " seed=", seed, " h=", h_fd, ": status(+/-)=", r_plus.inner_status, "/", r_minus.inner_status,
       " Delta(+/-)=", r_plus.Delta_dual, "/", r_minus.Delta_dual,
       " analytic=", predicted_directional, " fd=", fd_directional, " rel_err=", rel_err)
end

# One-hot probes on a handful of individual coordinates (isolate whether it's a specific cell)
lp(">>> one-hot probes on first 5 a_nonpivot coordinates")
for k in 1:5
    h_fd = 1e-4
    w_plus = copy(w_ext_start_a); w_plus[2+k] += h_fd
    w_minus = copy(w_ext_start_a); w_minus[2+k] -= h_fd
    r_plus, _ = screened_eval_flexible_A_verify(w_plus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
    r_minus, _ = screened_eval_flexible_A_verify(w_minus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
    fd_k = (r_plus.Delta_dual - r_minus.Delta_dual) / (2 * h_fd)
    analytic_k = predicted_g_a[k]
    rel_err_k = abs(fd_k - analytic_k) / max(abs(fd_k), 1e-8)
    lp("  k=", k, " status(+/-)=", r_plus.inner_status, "/", r_minus.inner_status, " analytic=", analytic_k, " fd=", fd_k, " rel_err=", rel_err_k)
end
flush(stdout)
