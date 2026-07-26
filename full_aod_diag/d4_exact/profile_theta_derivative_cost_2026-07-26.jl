# Ad hoc profiling: quantify the per-gradient-callback cost asymmetry between the shared
# analytic 380-coordinate C+ gradient pass and the flexible-mode theta secant's 3x
# full-moments-reconstruction + 2x redundant pivot-cache-rebuild pattern in cb_G!
# (c10_d20_production_driver_unified.jl). Diagnostic only, not part of the gate suite.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))

const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
theta_star = 1.0 / ctx0.μHat
sigma = ctx0.σ
theta_min = 2 * (sigma - 1) * 1.05
theta_max = 3 * theta_star
ctx_flex = make_flexible_theta(ctx0; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :powered_aspace)
xy = precompute_aspace_XY(ctx_flex)
pgc = build_pivot_elimination_cheap(ctx_flex; mu_probe1 = 1.0/theta_min, mu_probe2 = 1.0/theta_max)

x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
D = ctx0.D; Ddest = ctx0.D_dest
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
layout = make_layout(trade_elasticity_mode = :flexible, A_coordinate_mode = :powered_aspace)
w0 = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout)

rsc = build_ranged_screen_context(ctx_flex)
sc = ScreenCounters(); n_eval = Ref(0)
d0 = decode_outer_unified(w0, ctx_flex, layout, pgc, xy)
r0, _ = screened_eval(d0.xf, ctx_flex, rsc, sc, n_eval; warm = false)
println("cold start: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual)

grad_pool = build_grad_workspace_pool(W)
lfix_c_ws = build_lfix_factorized_workspace(D, Ddest, W)
inner_x_fixed = copy(ctx_flex.obj.x)
ctx_frozen = freeze_theta_ctx(ctx_flex, d0.mu)
pe_here = pivot_elim_from_cache(pgc, d0.mu)
xf_reduced = vcat(d0.gp, d0.xf[3:end])
base = BaseDualState(collect(d0.xf), r0.θ_full, r0.zeta, r0.lambda, copy(ctx_flex.obj.arg1), r0.inner_status)
base_frozen = BaseDualState(xf_reduced, base.θ_full0, base.ζstar, base.λstar, base.m_star, base.inner_status)

# warm up (JIT)
composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws; base = base_frozen, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
theta_fixed_dual_delta_pivot_A(w0, inner_x_fixed, ctx_flex, xy)
build_pivot_elimination_cheap(ctx_flex; mu_probe1 = 1.0/theta_min, mu_probe2 = 1.0/theta_max)

println("\n=== A) shared analytic C+ gradient pass (all 380 gp/A coordinates, ONE call) ===")
t1 = @elapsed a1 = @allocated composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws; base = base_frozen, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
println("  wall=", round(t1*1000, digits=2), "ms  allocated=", round(a1/1024, digits=1), "KiB")

println("\n=== B) ONE theta_fixed_dual_delta_pivot_A call (theta secant needs 2 of these per gradient) ===")
t2 = @elapsed a2 = @allocated theta_fixed_dual_delta_pivot_A(w0, inner_x_fixed, ctx_flex, xy)
println("  wall=", round(t2*1000, digits=2), "ms  allocated=", round(a2/1024, digits=1), "KiB")

println("\n=== C) build_pivot_elimination_cheap alone (rebuilt TWICE per gradient inside B, redundantly -- pgc is already available) ===")
t3 = @elapsed a3 = @allocated build_pivot_elimination_cheap(ctx_flex; mu_probe1 = 1.0/theta_min, mu_probe2 = 1.0/theta_max)
println("  wall=", round(t3*1000, digits=2), "ms  allocated=", round(a3/1024, digits=1), "KiB")

println("\n=== D) the base-point moments!/reconstruct_full call cb_G! does a THIRD time after the secant ===")
θ_full_base = CS.reconstruct_full(d0.xf, ctx_flex.m)
t4 = @elapsed a4 = @allocated ctx_flex.obj.moments!(@view(ctx_flex.obj.H[:, 1]), CS.select_G_from_H(ctx_flex.obj, ctx_flex.obj.H), θ_full_base, ctx_flex.obj.U, ctx_flex.obj)
println("  wall=", round(t4*1000, digits=2), "ms  allocated=", round(a4/1024, digits=1), "KiB")

println("\n=== Estimated real cb_G! cost split (flexible mode) ===")
theta_block_est = 2*t2 + t4   # B called twice (w_plus, w_minus) + D once
println("  A (shared 380-coord analytic gradient): ", round(t1*1000, digits=1), "ms")
println("  theta block (2x B + 1x D):               ", round(theta_block_est*1000, digits=1), "ms  (",
        round(100*theta_block_est/(t1+theta_block_est), digits=1), "% of total cb_G! wall-time)")
println("  of which C (redundant pivot-cache rebuild, x2 inside B): ", round(2*t3*1000, digits=1), "ms")
