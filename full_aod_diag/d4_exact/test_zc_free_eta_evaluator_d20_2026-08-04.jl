# fix/profiled-functional-readiness-closeout-2026-08-03, task §9: D20 scale extension of
# test_zc_free_eta_evaluator_d4_2026-08-04.jl -- SAME gate (real production evaluator surface,
# fixed-dual FD ground truth via a separate dense-reference bundle, adaptive per-coordinate
# bandwidth), at real D20 data. W_VAL controls scale (set below; pass as ARGS[1] to override).
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_stable_layout_digest_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl",
          "profiled_zc_free_eta_2026-08-04.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"Fixed-dual Delta_dual at a perturbed (w_econ, eta_nu), zeta*/lambda* held FIXED."
function delta_dual_fixed_dual(obj, ζstar::Float64, λstar::Vector{Float64}, w_econ::Vector{Float64},
        eta_nu::Vector{Float64}, ctx, pe, W::Int)
    decoded = decode_outer_profiled(w_econ, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    θ_ext = vcat(θ_full, exp.(eta_nu))
    K = Vector{Float64}(undef, W)
    G = Matrix{Float64}(undef, W, obj.d - 1)
    obj.moments!(K, G, θ_ext, ctx.U, obj)
    r = fill(-ζstar, W) .- G * λstar
    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + ζstar
    return -f
end

"Gate against fixed-dual FD -- SUBSET of coordinates at D20 (gp, 3 representative A_free spread
across the vector, ALL eta) per the task's own D20 'selected coordinates from every block'
instruction, not the full sweep D4 used."
function gate_family_selected(label::String, obj, w_econ::Vector{Float64}, eta_nu::Vector{Float64}, ev,
        g_ext::Vector{Float64}, ctx, pe, W::Int, h_used::Vector{Float64}, econ_coords::Vector{Int};
        h_gp = 1e-4, h_eta = 1e-4, tol = 5e-3)
    n_econ = length(w_econ); n_eta_d = length(eta_nu)
    ζstar = ev.result.zeta; λstar = ev.result.beta
    max_rel = 0.0
    for coord in econ_coords
        h_econ = coord == 1 ? h_gp : h_used[coord]
        wp = copy(w_econ); wp[coord] += h_econ
        wm = copy(w_econ); wm[coord] -= h_econ
        Kp = delta_dual_fixed_dual(obj, ζstar, λstar, wp, eta_nu, ctx, pe, W)
        Km = delta_dual_fixed_dual(obj, ζstar, λstar, wm, eta_nu, ctx, pe, W)
        fd = (Kp - Km) / (2h_econ)
        err = abs(fd - g_ext[coord]); relerr = err / max(1.0, abs(fd))
        max_rel = max(max_rel, relerr)
        @printf("  [%s] econ[%d]  analytic=%14.8g  FD=%14.8g  rel_err=%.3e\n", label, coord, g_ext[coord], fd, relerr)
    end
    for k in 1:n_eta_d
        ep = copy(eta_nu); ep[k] += h_eta
        em = copy(eta_nu); em[k] -= h_eta
        Kp = delta_dual_fixed_dual(obj, ζstar, λstar, w_econ, ep, ctx, pe, W)
        Km = delta_dual_fixed_dual(obj, ζstar, λstar, w_econ, em, ctx, pe, W)
        fd = (Kp - Km) / (2h_eta)
        err = abs(fd - g_ext[n_econ+k]); relerr = err / max(1.0, abs(fd))
        max_rel = max(max_rel, relerr)
        @printf("  [%s] eta[%d]   analytic=%14.8g  FD=%14.8g  rel_err=%.3e\n", label, k, g_ext[n_econ+k], fd, relerr)
    end
    check("$label: selected [econ;eta] gradient matches FD (max_rel_err < $tol)", max_rel < tol)
    return max_rel
end

W_VAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
println("W_VAL = $W_VAL"); flush(stdout)
t0 = time()
println("PID=", getpid()); flush(stdout)

t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; W = size(ctx.U, 1)
korea_idx, brazil_idx = 14, 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
w_profiled_calib = reduce_to_w_profiled(θ0[3+D], z_calib, pe)
n_total = length(w_profiled_calib)
econ_coords = unique(vcat(1, 2, min(3, n_total), n_total ÷ 2, n_total))
@printf("outer_dim_profiled = %d, testing econ coords %s\n", n_total, econ_coords); flush(stdout)

println("="^90); println("FAMILY origin-ZC: free-eta evaluator D20 gate, W=$W_VAL"); println("="^90); flush(stdout)
layout_o = OriginByPowerLayout(D, 1, 0)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
obj_fd_oz = aug_reduced_oz.obj_cm
octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, fill(1.0, D))
eta0_oz = zeros(D)
t_solve_oz = @elapsed ev_oz = evaluate_profiled_originzc_point(w_profiled_calib, eta0_oz, fctx_oz, pes_oz)
@printf("origin-ZC solve: %.2fs\n", t_solve_oz); flush(stdout)
check("origin-ZC free-eta D20: inner solve converges", ev_oz.result.inner_status == 0)
g_ext_oz, meta_oz = reduced_originzc_outer_gradient_with_eta(w_profiled_calib, eta0_oz, ctx, fctx_oz, ev_oz)
gate_family_selected("origin-ZC@calib(D20,W=$W_VAL)", obj_fd_oz, w_profiled_calib, eta0_oz, ev_oz, g_ext_oz, ctx, pe, W, meta_oz.h_used, econ_coords)

println("\n" * "="^90); println("FAMILY CM+ZC: free-eta evaluator D20 gate, W=$W_VAL"); println("="^90); flush(stdout)
const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
aug_reduced_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = true, inner_fg_backend = :dense_reference, profiled_layout = layout)
bins_u32 = cctx_reduced_cz.Bidx isa Matrix{UInt32} ? cctx_reduced_cz.Bidx : Matrix{UInt32}(cctx_reduced_cz.Bidx)
obj_fd_cz = aug_reduced_cz.obj_cm
fctx_cz = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced_cz, cctx_reduced_cz, bins_u32)
pes_cz = CMZCPointEvalState(cctx_reduced_cz, fill(1.0, K_MEAN))
eta0_cz = zeros(K_MEAN)
t_solve_cz = @elapsed ev_cz = evaluate_profiled_cmzc_point(w_profiled_calib, eta0_cz, fctx_cz, pes_cz)
@printf("CM+ZC solve: %.2fs\n", t_solve_cz); flush(stdout)
check("CM+ZC free-eta D20: inner solve feasible/optimal", ev_cz.result.inner_status in (0, -100, -101, -103))
g_ext_cz, meta_cz = reduced_cmzc_outer_gradient_with_eta(w_profiled_calib, eta0_cz, ctx, fctx_cz, ev_cz)
gate_family_selected("CM+ZC@calib(D20,W=$W_VAL)", obj_fd_cz, w_profiled_calib, eta0_cz, ev_cz, g_ext_cz, ctx, pe, W, meta_cz.h_used, econ_coords)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
ALL_PASS[] || exit(1)
