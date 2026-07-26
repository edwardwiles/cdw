# Runtime/allocation gates (task §11): old brute-force theta block vs new theta_cplus block vs
# the shared 380-coordinate A/gp C+ block, warmed, real D=20/W=80,000.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "theta_cplus.jl"))

const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
theta_star = 1.0 / ctx0.μHat
sigma = ctx0.σ
theta_min = 2 * (sigma - 1) * 1.05
theta_max = 3 * theta_star
ctx0 = attach_compressed_factual_workspace(ctx0, ctx0.D, ctx0.D_dest, W)
ctx0 = attach_canonical_price_precompute_workspace(ctx0)
ctx0 = attach_hard_score_b_cache(ctx0)
ctx_flex = make_flexible_theta(ctx0; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :powered_aspace)
xy = precompute_aspace_XY(ctx_flex)
D = ctx0.D; Ddest = ctx0.D_dest
pgc = build_pivot_elimination_cheap(ctx_flex; mu_probe1 = 1.0/theta_min, mu_probe2 = 1.0/theta_max)
theta_ws = build_theta_cplus_workspace(D, Ddest, W; h_theta = 1e-3)

x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
layout = make_layout(trade_elasticity_mode = :flexible, A_coordinate_mode = :powered_aspace)
w0 = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout)

rsc = build_ranged_screen_context(ctx_flex)
sc = ScreenCounters(); n_eval = Ref(0)
d0 = decode_outer_unified(w0, ctx_flex, layout, pgc, xy)
r0, _ = screened_eval(d0.xf, ctx_flex, rsc, sc, n_eval; warm = false)
println("cold start: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual)
base = BaseDualState(collect(d0.xf), r0.θ_full, r0.zeta, r0.lambda, copy(ctx_flex.obj.arg1), r0.inner_status)

grad_pool = build_grad_workspace_pool(W)
lfix_c_ws = build_lfix_factorized_workspace(D, Ddest, W)
xf_reduced = vcat(d0.gp, d0.xf[3:end])
ctx_frozen = freeze_theta_ctx(ctx_flex, d0.mu)
pe_here = pivot_elim_from_cache(pgc, d0.mu)
base_frozen = BaseDualState(xf_reduced, base.θ_full0, base.ζstar, base.λstar, base.m_star, base.inner_status)

# warm up
composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws; base = base_frozen, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
theta_fixed_dual_delta_pivot_A(w0, copy(ctx_flex.obj.x), ctx_flex, xy)
theta_cplus_secant(w0, ctx_flex, xy, pgc, base, theta_ws)

println("\n=== A) shared analytic C+ gradient (380 coords, ONE call) ===")
tA = @elapsed aA = @allocated composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws; base = base_frozen, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
println("  wall=", round(tA*1000, digits=2), "ms  allocated=", round(aA/1024/1024, digits=2), "MiB")

println("\n=== OLD theta block (2x theta_fixed_dual_delta_pivot_A, matching the old cb_G!) ===")
inner_x_fixed = copy(ctx_flex.obj.x)
h_theta = 1e-3
told = @elapsed aold = @allocated begin
    w_plus = copy(w0); w_plus[1] += h_theta
    D_plus = theta_fixed_dual_delta_pivot_A(w_plus, inner_x_fixed, ctx_flex, xy)
    w_minus = copy(w0); w_minus[1] -= h_theta
    D_minus = theta_fixed_dual_delta_pivot_A(w_minus, inner_x_fixed, ctx_flex, xy)
    θ_full_base = CS.reconstruct_full(d0.xf, ctx_flex.m)
    ctx_flex.obj.moments!(@view(ctx_flex.obj.H[:, 1]), CS.select_G_from_H(ctx_flex.obj, ctx_flex.obj.H), θ_full_base, ctx_flex.obj.U, ctx_flex.obj)
    ctx_flex.obj.H[:, 2] .= 1.0
end
println("  wall=", round(told*1000, digits=1), "ms  allocated=", round(aold/1024/1024, digits=2), "MiB")

println("\n=== NEW theta block (theta_cplus_secant, no 3rd reconstruction) ===")
tnew = @elapsed anew = @allocated theta_cplus_secant(w0, ctx_flex, xy, pgc, base, theta_ws)
println("  wall=", round(tnew*1000, digits=2), "ms  allocated=", round(anew/1024/1024, digits=3), "MiB")

println("\n=== SUMMARY ===")
println("  A (shared 380-coord gradient):     ", round(tA*1000, digits=1), "ms, ", round(aA/1024/1024, digits=1), "MiB")
println("  OLD theta block:                    ", round(told*1000, digits=1), "ms, ", round(aold/1024/1024, digits=1), "MiB  (", round(100*told/(tA+told), digits=1), "% of old total cb_G!)")
println("  NEW theta block:                    ", round(tnew*1000, digits=1), "ms, ", round(anew/1024/1024, digits=3), "MiB  (", round(100*tnew/(tA+tnew), digits=1), "% of new total cb_G!)")
println("  THETA_BLOCK_SPEEDUP = ", round(told/tnew, digits=2), "x")
println("  THETA_BLOCK_ALLOCATION_REDUCTION = ", round(100*(1 - anew/aold), digits=2), "%")
println("  old total cb_G! (A+old theta):      ", round((tA+told)*1000, digits=1), "ms")
println("  new total cb_G! (A+new theta):      ", round((tA+tnew)*1000, digits=1), "ms")
println("  TOTAL_CBG_SPEEDUP = ", round((tA+told)/(tA+tnew), digits=2), "x")
