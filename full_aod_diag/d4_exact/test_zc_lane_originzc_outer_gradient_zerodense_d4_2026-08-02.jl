# Phase2b task (2026-08-02): TRUE zero-dense sibling of
# test_zc_lane_originzc_outer_gradient_d4_2026-08-02.jl.
#
# The existing dense-FG gate solves origin-ZC's inner problem via `archOZ_base_state` (the
# DENSE-width :dense_reference FG evaluator). This file solves through `reduced_originzc_base_state`
# -- `profiled_reduced_originzc_lookup_kernels_2026-08-02.jl`'s own genuinely matrix-free
# `ReducedOriginZCOperatorState`/`inner_loop_KNITRO_reduced_originzc` driver, the SAME incantation
# `test_profiled_reduced_originzc_operator_and_autodiff_2026-08-02.jl` already uses and gates via
# NO_DENSE_G_COUNTERS. `OriginZCFamilyCtx` is built from the SAME `aug_reduced` regardless of which
# driver solved the inner problem (it only reads POST-SOLVE `ev.result.beta`/`ev.nu_full`/`ev.st.cf`).
#
# `m_weights`/`mean_m` (needed by `d_delta_dual_d_eta_origin_vec`) are recovered WITHOUT any dense G:
# a single extra FG evaluation of the solved point through the SAME zero-dense `st` functor refreshes
# `st.arg1 = dPsi(q*)` in place.
#
# Gate: NO_DENSE_G_COUNTERS deltas (economic/ZC) are asserted EXACTLY ZERO across BOTH inner solves
# (calibration + perturbed point) + BOTH outer gradient evaluations -- not just the inner solve. The
# FD ground truth (`delta_dual_fixed_dual`, reused verbatim from the existing dense gate) necessarily
# rebuilds a dense reference G via `obj.moments!` and is measured OUTSIDE that bracket, exactly as
# every other gate in this codebase does for its own FD reference.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
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
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end

"Inlined copy, same rationale/cross-talk-avoidance note as the ZC-lane sibling gates."
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

function evaluate_profiled_point end
for f in ["profiled_lfix_incremental_2026-08-01.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"Fixed-dual Delta_dual, IDENTICAL to the existing dense gate's own helper."
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

layout_o = OriginByPowerLayout(D, 1, 0)   # K_mean=1, K_pair=0 -- the one safe config
νvec0 = fill(1.0, D)   # n_eta = K_mean*D = D

aug_reduced = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm   # dense-reference closure, used ONLY as the FD ground truth below
octx_reduced = build_originzc_core_hess_ctx(aug_reduced, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)

before_counters = deepcopy(NO_DENSE_G_COUNTERS[])

println("="^78); println("STEP 1: TRUE ZERO-DENSE reduced origin-ZC inner solve at calibration"); println("="^78)
base_reduced = reduced_originzc_base_state(x_free_calib, ctx, layout, octx_reduced, νvec0)
check("REDUCED (operator) inner solve converges", base_reduced.inner_status == 0)
n_econ = layout.total_reduced_economic_moments
n_eta_total = n_eta(layout_o)
β_full = base_reduced.λstar
@printf("nStatus=%d  zeta*=%.8f  n_econ=%d  n_eta=%d  length(beta)=%d  n_fg=%d\n",
    base_reduced.inner_status, base_reduced.ζstar, n_econ, n_eta_total, length(β_full), base_reduced.n_fg)

println("\n" * "="^78); println("STEP 2: analytic combined gradient at calibration (zero-dense throughout)"); println("="^78)
fctx = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced)

st_reduced = base_reduced.st
x_full = vcat(base_reduced.ζstar, β_full)
g_buf = zeros(length(x_full))
f_check = st_reduced(x_full, g_buf)
m_weights = copy(st_reduced.arg1)
mean_m = sum(m_weights) / W
cf_reduced = octx_reduced.core_cf_ref[]
cf_reduced isa CompressedFactual || error("core_cf_ref[] is not a CompressedFactual after solve")
# Performance closeout task (2026-08-02): see the identical fix/comment in
# test_zc_lane_cmzc_outer_gradient_zerodense_d4_2026-08-02.jl -- this mock ev needs `decoded` to
# match evaluate_profiled_originzc_point's own real return shape now that the shared gradient
# engine's build_price_winner_base_cache reads ev.decoded.xf.
decoded_calib = decode_outer_profiled(w_profiled_calib, ctx, pe)
ev = (result = (beta = β_full, zeta = base_reduced.ζstar), st = (cf = cf_reduced, layout = layout),
      theta_full = collect(θ_full_calib), obj = (M = W,), m_weights = m_weights, nu_full = νvec0,
      decoded = decoded_calib)

g_Agp, meta = shared_family_outer_gradient(w_profiled_calib, ctx, fctx, ev; threaded = false)
g_eta = d_delta_dual_d_eta_origin_vec(β_full, aug_reduced, νvec0; mean_m = mean_m)
println("g_Agp: length=$(length(g_Agp))  g_Agp[1](dK/dgp)=$(g_Agp[1])")
println("g_eta: length=$(length(g_eta))  g_eta[1:3]=$(g_eta[1:min(3,end)])")

println("\n" * "="^78); println("STEP 3: gate vs COMPLETE fixed-dual finite differences (calibration point)"); println("="^78)
h_A = 1e-4; h_eta = 1e-4
max_rel_err_A = 0.0; max_abs_err_A = 0.0
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
    global max_abs_err_A = max(max_abs_err_A, err)
    label = coord == 1 ? "gp" : "A_free[$(coord-1)]"
    @printf("  %-14s analytic=%14.8g  FD(h=%.2e)=%14.8g  abs_err=%.3e  rel_err=%.3e\n", label, g_Agp[coord], h_c, fd, err, relerr)
end
check("A/gp gradient matches FD at calibration (max_rel_err < 5e-3)", max_rel_err_A < 5e-3)

max_rel_err_eta = 0.0
n_eta_test = min(n_eta_total, D)
for k in 1:n_eta_test
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

println("\n" * "="^78); println("STEP 4: gate at a PERTURBED (non-calibration) point -- TRUE zero-dense re-solve"); println("="^78)
w_profiled_pert = w_profiled_calib .+ 0.02 .* randn(n_total)
νvec_pert = νvec0 .* exp.(0.05 .* randn(D))
decoded_pert = decode_outer_profiled(w_profiled_pert, ctx, pe)
θ_full_pert = CS.reconstruct_full(decoded_pert.xf, ctx.m)
octx_reduced2 = build_originzc_core_hess_ctx(aug_reduced, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
x_free_pert = θ_full_pert[ctx.free_idx]
base_pert = reduced_originzc_base_state(x_free_pert, ctx, layout, octx_reduced2, νvec_pert)
check("REDUCED (operator) inner solve converges at perturbed point", base_pert.inner_status == 0)
β_pert = base_pert.λstar
fctx2 = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced)

st_pert = base_pert.st
x_full_pert = vcat(base_pert.ζstar, β_pert)
g_buf_pert = zeros(length(x_full_pert))
st_pert(x_full_pert, g_buf_pert)
mw2 = copy(st_pert.arg1)
cf_pert = octx_reduced2.core_cf_ref[]
ev2 = (result = (beta = β_pert, zeta = base_pert.ζstar), st = (cf = cf_pert, layout = layout),
       theta_full = θ_full_pert, obj = (M = W,), m_weights = mw2, nu_full = νvec_pert,
       decoded = decoded_pert)
g_Agp2, meta2 = shared_family_outer_gradient(w_profiled_pert, ctx, fctx2, ev2; threaded = false)
after_counters = NO_DENSE_G_COUNTERS[]

max_rel_err_A2 = 0.0
test_coords = unique(vcat(1, 2, pe.pivot_pos == 2 ? 3 : 2, rand(2:n_total, min(4, n_total - 1))))
for coord in test_coords
    h_c = coord == 1 ? h_A : meta2.h_used[coord]
    wp = copy(w_profiled_pert); wp[coord] += h_c
    Kp = delta_dual_fixed_dual(obj_reduced, base_pert.ζstar, β_pert, wp, νvec_pert, ctx, pe, W)
    wm = copy(w_profiled_pert); wm[coord] -= h_c
    Km = delta_dual_fixed_dual(obj_reduced, base_pert.ζstar, β_pert, wm, νvec_pert, ctx, pe, W)
    fd = (Kp - Km) / (2h_c)
    err = abs(fd - g_Agp2[coord])
    relerr = err / max(1.0, abs(fd))
    global max_rel_err_A2 = max(max_rel_err_A2, relerr)
    label = coord == 1 ? "gp" : "A_free[$(coord-1)]"
    @printf("  %-14s analytic=%14.8g  FD(h=%.2e)=%14.8g  rel_err=%.3e\n", label, g_Agp2[coord], h_c, fd, relerr)
end
check("A/gp gradient matches FD at perturbed point (max_rel_err < 5e-3)", max_rel_err_A2 < 5e-3)

println("\n" * "="^78); println("NO_DENSE_G_COUNTERS delta across BOTH inner solves + BOTH outer gradient evaluations"); println("="^78)
d_econ = after_counters.dense_economic_G_materializations - before_counters.dense_economic_G_materializations
d_zc = after_counters.dense_ZC_G_materializations - before_counters.dense_ZC_G_materializations
@printf("  delta dense_economic_G_materializations=%d  delta dense_ZC_G_materializations=%d\n", d_econ, d_zc)
check("ZERO dense economic G materializations across both solves+gradients", d_econ == 0)
check("ZERO dense ZC G materializations across both solves+gradients", d_zc == 0)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
