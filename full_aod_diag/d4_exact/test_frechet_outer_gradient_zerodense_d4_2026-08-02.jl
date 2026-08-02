# Phase2b task (2026-08-02): combined outer-gradient decisive-PASS gate for common Fréchet,
# following the ZC-lane gates' exact "complete fixed-dual FD" methodology, mirroring
# test_flexcm_outer_gradient_zerodense_d4_2026-08-02.jl exactly (see that file's own header for the
# full rationale -- identical reasoning applies here).
#
# Common Fréchet, like flexible CM, has NO restriction outer parameter in real production (level
# targets are fixed campaign configuration, never a live KNITRO decision variable -- confirmed by
# grep: no `d_delta_dual_d_eta*` function exists for common Fréchet anywhere in this directory). So
# this family's "combined outer gradient" is JUST the A/gp block.
#
# The inner solve is ALREADY genuinely zero-dense here: `evaluate_profiled_frechet_point`
# (profiled_restricted_family_adapters_2026-08-02.jl) delegates entirely to `reduced_frechet_base_state`
# (profiled_reduced_frechet_lookup_kernels_2026-08-02.jl's own `ReducedCMFrechetLookupState`/
# `inner_loop_KNITRO_reduced_frechet` driver, with `archC_frechet_hess_cb_builder` -- NOT flexible
# CM's own Hessian builder, which never dispatches the level block at all).
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
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl"]
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
          "profiled_restricted_family_adapters_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"Fixed-dual Delta_dual (no nu/eta -- common Fréchet has no restriction outer parameter)."
function delta_dual_fixed_dual_noeta(obj, ζstar::Float64, λstar::Vector{Float64}, w_profiled::Vector{Float64},
        ctx, pe, W::Int)
    decoded = decode_outer_profiled(w_profiled, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    K = Vector{Float64}(undef, W)
    G = Matrix{Float64}(undef, W, obj.d - 1)
    obj.moments!(K, G, θ_full, ctx.U, obj)
    r = fill(-ζstar, W) .- G * λstar
    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + ζstar
    return -f
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)
W = size(ctx.U, 1)

spec, gauge, pe = build_profiled_ab_spec_pe(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
w_profiled_calib = reduce_calibration_to_w_profiled(ctx, pe)
n_total = outer_dim_profiled(pe)
println("outer_dim_profiled = $n_total (1 gp + $(n_total-1) free-A)")

L = 10; contrasts = :anchored
aug_reduced = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_reduced.level_targets
obj_dense = aug_reduced.obj_cm   # dense-reference closure, used ONLY as the FD ground truth below
cctx_f = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference,
                           threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
fctx = build_frechet_family_ctx(ctx, spec, pe, layout, cctx_f, level_targets)

v = validate_family_layout_contract(fctx)
check("Frechet: layout contract validates", v.pe === pe && v.layout === layout)

before_counters = deepcopy(NO_DENSE_G_COUNTERS[])
println("="^78); println("STEP 1: TRUE ZERO-DENSE reduced common-Frechet inner solve at calibration"); println("="^78)
ev = evaluate_profiled_frechet_point(w_profiled_calib, fctx)
check("Frechet: real KNITRO solve reaches nStatus in {0,-100,-101,-103}", ev.result.inner_status in (0, -100, -101, -103))
@printf("nStatus=%d  zeta*=%.8f  length(beta)=%d  n_fg=%d\n",
    ev.result.inner_status, ev.result.zeta, length(ev.result.beta), ev.result.n_fg_calls)

println("\n" * "="^78); println("STEP 2: analytic combined (A/gp-only) gradient at calibration (zero-dense throughout)"); println("="^78)
g_Agp, meta = shared_family_outer_gradient(w_profiled_calib, ctx, fctx, ev)
after_counters = NO_DENSE_G_COUNTERS[]
println("g_Agp: length=$(length(g_Agp))  g_Agp[1](dK/dgp)=$(g_Agp[1])")

println("\n" * "="^78); println("STEP 2b: NO_DENSE_G_COUNTERS delta across inner solve + outer gradient evaluation"); println("="^78)
d_econ = after_counters.dense_economic_G_materializations - before_counters.dense_economic_G_materializations
d_cm = after_counters.dense_CM_G_materializations - before_counters.dense_CM_G_materializations
d_frechet = after_counters.dense_Frechet_G_materializations - before_counters.dense_Frechet_G_materializations
@printf("  delta dense_economic_G_materializations=%d  delta dense_CM_G_materializations=%d  delta dense_Frechet_G_materializations=%d\n",
    d_econ, d_cm, d_frechet)
check("ZERO dense economic G materializations across solve+gradient", d_econ == 0)
check("ZERO dense CM-grid G materializations across solve+gradient", d_cm == 0)
check("ZERO dense Frechet-level G materializations across solve+gradient", d_frechet == 0)

println("\n" * "="^78); println("STEP 3: gate vs COMPLETE fixed-dual finite differences (calibration point)"); println("="^78)
h_A = 1e-4
max_rel_err_A = 0.0
for coord in 1:n_total
    h_c = coord == 1 ? h_A : meta.h_used[coord]
    wp = copy(w_profiled_calib); wp[coord] += h_c
    Kp = delta_dual_fixed_dual_noeta(obj_dense, ev.result.zeta, ev.result.beta, wp, ctx, pe, W)
    wm = copy(w_profiled_calib); wm[coord] -= h_c
    Km = delta_dual_fixed_dual_noeta(obj_dense, ev.result.zeta, ev.result.beta, wm, ctx, pe, W)
    fd = (Kp - Km) / (2h_c)
    err = abs(fd - g_Agp[coord])
    relerr = err / max(1.0, abs(fd))
    global max_rel_err_A = max(max_rel_err_A, relerr)
    label = coord == 1 ? "gp" : "A_free[$(coord-1)]"
    @printf("  %-14s analytic=%14.8g  FD(h=%.2e)=%14.8g  rel_err=%.3e\n", label, g_Agp[coord], h_c, fd, relerr)
end
check("A/gp gradient matches FD at calibration (max_rel_err < 5e-3)", max_rel_err_A < 5e-3)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
