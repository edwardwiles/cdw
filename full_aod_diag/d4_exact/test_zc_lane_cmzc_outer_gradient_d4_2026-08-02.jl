# ZC lane task (2026-08-02), Phase II outer gradient: combined A/gp + eta/nu outer gradient for
# CM+ZC's real reduced+widened context, gated against COMPLETE fixed-dual finite differences.
# Direct sibling of test_zc_lane_originzc_outer_gradient_d4_2026-08-02.jl (which validated this
# decisively for origin-ZC) -- same "fixed-dual" FD methodology, same shared A/gp engine, same
# per-coordinate adaptive-bandwidth-matching discipline (this repo's own documented FD pitfall).
# CM-grid (C) targets are fixed campaign configuration (task's own instruction: "do not invent
# gradients for fixed target parameters") -- restriction_contrib0_cmzc! folds their contribution
# into q0 as a constant, like every restriction block, but no C-block outer-parameter gradient is
# built here, matching origin-ZC's own Z-only scope.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_meanzc_lookup_kernels.jl", "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl"]
    include(joinpath(D4X, f))
end

# NOTE (2026-08-02, found live via bisection on the origin-ZC sibling gate): profiled_operator_
# bundle_2026-08-01.jl (the unrestricted family's own second, independent KNITRO driver) causes
# cross-talk with a real family KNITRO solve when both are loaded in the same process -- silent
# degenerate solve, misleadingly-reported nStatus=0/-103. Not loaded here; the two small helpers it
# would have provided are inlined below instead.
function evaluate_profiled_point end
function build_profiled_ab_spec_pe(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}())
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = global_overrides)
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gauge = build_anchor_gauge(z_calib, spec)
    pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
    return spec, gauge, pe
end
function reduce_calibration_to_w_profiled(ctx, pe::PivotGravityElimOnRetained)
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gp0 = θ0[3+D]
    return reduce_to_w_profiled(gp0, z_calib, pe)
end

for f in ["profiled_lfix_incremental_2026-08-01.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"Fixed-dual Delta_dual at a perturbed (w_profiled, nu_full), zeta*/lambda* held FIXED at the real
solved point -- rebuilds G fresh via obj.moments! (the SAME production closure every real solve
uses), never a hand-derived formula."
function delta_dual_fixed_dual(obj, ζstar::Float64, λstar::Vector{Float64}, w_profiled::Vector{Float64},
        νfull::Vector{Float64}, ctx, pe, W::Int)
    decoded = decode_outer_profiled(w_profiled, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    θ_ext = vcat(θ_full, νfull)
    K = Vector{Float64}(undef, W)
    G = Matrix{Float64}(undef, W, obj.d - 1)
    obj.moments!(K, G, θ_ext, ctx.U, obj)
    r = fill(-ζstar, W) .- G * λstar
    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + ζstar
    return -f
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)
D = ctx.D
W = size(ctx.U, 1)

spec, gauge, pe = build_profiled_ab_spec_pe(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
w_profiled_calib = reduce_calibration_to_w_profiled(ctx, pe)
n_total = outer_dim_profiled(pe)
println("outer_dim_profiled = $n_total (1 gp + $(n_total-1) free-A)")

const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
νvec0 = fill(1.0, K_MEAN)

aug_reduced = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm
cctx_reduced = build_cm_meanzc_bin_ctx(ctx, aug_reduced; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
ctx_cm_reduced = (obj = obj_reduced, m = ctx.m)

println("="^78); println("STEP 1: REDUCED CM+ZC inner solve at calibration"); println("="^78)
base_reduced = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_reduced, cctx_reduced)
check("REDUCED inner solve feasible/optimal-within-tolerance", base_reduced.inner_status in (0, -100, -101, -103))
n_econ = layout.total_reduced_economic_moments
β_full = base_reduced.λstar
@printf("nStatus=%d  zeta*=%.8f  n_econ=%d  length(beta)=%d  obj.d=%d\n",
    base_reduced.inner_status, base_reduced.ζstar, n_econ, length(β_full), obj_reduced.d)

println("\n" * "="^78); println("STEP 2: analytic combined gradient at calibration"); println("="^78)
bins_u32 = cctx_reduced.Bidx isa Matrix{UInt32} ? cctx_reduced.Bidx : Matrix{UInt32}(cctx_reduced.Bidx)
fctx = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced, cctx_reduced, bins_u32)
K_v = Vector{Float64}(undef, W); G_v = Matrix{Float64}(undef, W, obj_reduced.d - 1)
obj_reduced.moments!(K_v, G_v, vcat(collect(θ_full_calib), νvec0), ctx.U, obj_reduced)
r_v = fill(-base_reduced.ζstar, W) .- G_v * β_full
m_weights = similar(r_v); obj_reduced.dPsi!(m_weights, r_v)
mean_m = sum(m_weights) / W
cf_reduced = cctx_reduced.core_cf_ref[]
cf_reduced isa CompressedFactual || error("core_cf_ref[] is not a CompressedFactual after solve")
ev = (result = (beta = β_full, zeta = base_reduced.ζstar), st = (cf = cf_reduced, layout = layout),
      theta_full = collect(θ_full_calib), obj = (M = W,), m_weights = m_weights, nu_full = νvec0)

cache_diag = build_shared_profiled_lfix_cache(w_profiled_calib, fctx, ctx, ev)
println("q0 vs q_true diagnostic: max|cache.q0 - q_true|=", maximum(abs.(cache_diag.q0 .- r_v)))

g_Agp, meta = shared_family_outer_gradient(w_profiled_calib, ctx, fctx, ev)
g_eta = d_delta_dual_d_eta_nu_vec(β_full, aug_reduced, νvec0; mean_m = mean_m)
println("g_Agp: length=$(length(g_Agp))  g_Agp[1](dK/dgp)=$(g_Agp[1])")
println("g_eta: length=$(length(g_eta))  g_eta=$(g_eta)")

println("\n" * "="^78); println("STEP 3: gate vs COMPLETE fixed-dual finite differences (calibration point)"); println("="^78)
h_A = 1e-4; h_eta = 1e-4
max_rel_err_A = 0.0
for coord in 1:n_total
    h_c = coord == 1 ? h_A : meta.h_used[coord]
    wp = copy(w_profiled_calib); wp[coord] += h_c
    Kp = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, wp, νvec0, ctx, pe, W)
    wm = copy(w_profiled_calib); wm[coord] -= h_c
    Km = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, wm, νvec0, ctx, pe, W)
    fd = (Kp - Km) / (2h_c)
    err = abs(fd - g_Agp[coord])
    relerr = err / max(1.0, abs(fd))
    global max_rel_err_A = max(max_rel_err_A, relerr)
    label = coord == 1 ? "gp" : "A_free[$(coord-1)]"
    @printf("  %-14s analytic=%14.8g  FD(h=%.2e)=%14.8g  rel_err=%.3e\n", label, g_Agp[coord], h_c, fd, relerr)
end
check("A/gp gradient matches FD at calibration (max_rel_err < 5e-3)", max_rel_err_A < 5e-3)

max_rel_err_eta = 0.0
for k in 1:K_MEAN
    νp = copy(νvec0); νp[k] += h_eta
    Kp = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, w_profiled_calib, νp, ctx, pe, W)
    νm = copy(νvec0); νm[k] -= h_eta
    Km = delta_dual_fixed_dual(obj_reduced, base_reduced.ζstar, β_full, w_profiled_calib, νm, ctx, pe, W)
    fd_nu = (Kp - Km) / (2h_eta)
    fd_eta = fd_nu * νvec0[k]
    err = abs(fd_eta - g_eta[k])
    relerr = err / max(1.0, abs(fd_eta))
    global max_rel_err_eta = max(max_rel_err_eta, relerr)
    @printf("  eta[%d]         analytic=%14.8g  FD=%14.8g  rel_err=%.3e\n", k, g_eta[k], fd_eta, relerr)
end
check("eta/nu gradient matches FD at calibration (max_rel_err < 5e-3)", max_rel_err_eta < 5e-3)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
