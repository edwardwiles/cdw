# fix/profiled-functional-readiness-closeout-2026-08-03, section 3.2 (D20/W=20,000 half): the D4
# gates for common_frechet (test_frechet_threaded_profiled_d4_2026-08-03.jl) and cm_meanzc
# (test_zc_lane_cmzc_threaded_profiled_d4_2026-08-02.jl) both PASS. This extends both to real
# D20/W=20,000, 10 threads, per the task's own requirement not to infer production-scale behavior
# from D4 alone.
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
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

println("Threads.nthreads()=", Threads.nthreads()); flush(stdout)

const W_VAL = 20_000
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D
korea_idx, brazil_idx = 14, 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
W = size(ctx.U, 1)
L_VAL = 50

println("="^90); println("common_frechet, D20/W=20,000"); println("="^90); flush(stdout)
aug_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L_VAL, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_f.level_targets
cctx_f_s = build_cm_bin_ctx(ctx, aug_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
    threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
cctx_f_t = build_cm_bin_ctx(ctx, aug_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
    threaded_bins = true, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
K_probe = Vector{Float64}(undef, W); G_probe = Matrix{Float64}(undef, W, aug_f.obj_cm.d)
aug_f.obj_cm.moments!(K_probe, G_probe, collect(θ_full_calib), ctx.U, aug_f.obj_cm)
h_len_f = (cctx_f_s.NCORE + cctx_f_s.ncm) * (cctx_f_s.NCORE + cctx_f_s.ncm + 1) ÷ 2
Random.seed!(9302)
n_f = aug_f.obj_cm.outer_constr_index
xs_f = [zeros(n_f), 0.01 .* randn(n_f), 0.05 .* randn(n_f)]
maxdiff_f = 0.0
t_serial_f = 0.0; t_threaded_f = 0.0
for (pi_, x) in enumerate(xs_f)
    _prep_dual_index_for_archC!(cctx_f_s, aug_f.obj_cm, x)
    h_s = zeros(h_len_f)
    hessian_cm_frechet_structured!(h_s, aug_f.obj_cm, cctx_f_s, level_targets)  # warmup
    global t_serial_f += @elapsed hessian_cm_frechet_structured!(h_s, aug_f.obj_cm, cctx_f_s, level_targets)

    _prep_dual_index_for_archC!(cctx_f_t, aug_f.obj_cm, x)
    h_t = zeros(h_len_f)
    hessian_cm_frechet_structured_v2!(h_t, aug_f.obj_cm, cctx_f_t, level_targets; threaded_bins = true, tls = cctx_f_t.tls)  # warmup
    global t_threaded_f += @elapsed hessian_cm_frechet_structured_v2!(h_t, aug_f.obj_cm, cctx_f_t, level_targets; threaded_bins = true, tls = cctx_f_t.tls)

    d = maximum(abs.(h_s .- h_t))
    global maxdiff_f = max(maxdiff_f, d)
    check("common_frechet D20/W20k pt$pi_: threaded matches serial (max|Δ|=$(d))", d < 1e-8)
end
@printf("common_frechet: serial=%.4fs threaded=%.4fs (avg over %d calls each) maxdiff=%.4e\n",
    t_serial_f / length(xs_f), t_threaded_f / length(xs_f), length(xs_f), maxdiff_f)

println("="^90); println("cm_meanzc (CM+ZC), D20/W=20,000"); println("="^90); flush(stdout)
K_MEAN, K_PAIR = 1, 1
νvec0_cm = fill(1.0, K_MEAN)
aug_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_VAL, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_cz_s = build_cm_meanzc_bin_ctx(ctx, aug_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
cctx_cz_t = build_cm_meanzc_bin_ctx(ctx, aug_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = true, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
cctx_cz_s.nu_ref[] = collect(νvec0_cm); cctx_cz_t.nu_ref[] = collect(νvec0_cm)
K_probe2 = Vector{Float64}(undef, W); G_probe2 = Matrix{Float64}(undef, W, aug_cz.obj_cm.d)
aug_cz.obj_cm.moments!(K_probe2, G_probe2, vcat(collect(θ_full_calib), νvec0_cm), ctx.U, aug_cz.obj_cm)
h_len_cz = (cctx_cz_s.NCORE + cctx_cz_s.ncm) * (cctx_cz_s.NCORE + cctx_cz_s.ncm + 1) ÷ 2
Random.seed!(9303)
n_cz = aug_cz.obj_cm.outer_constr_index
xs_cz = [zeros(n_cz), 0.01 .* randn(n_cz), 0.05 .* randn(n_cz)]
maxdiff_cz = 0.0
t_serial_cz = 0.0; t_threaded_cz = 0.0
for (pi_, x) in enumerate(xs_cz)
    _prep_dual_index_for_archC!(cctx_cz_s, aug_cz.obj_cm, x)
    h_s = zeros(h_len_cz)
    hessian_cm_structured!(h_s, aug_cz.obj_cm, cctx_cz_s)  # warmup
    global t_serial_cz += @elapsed hessian_cm_structured!(h_s, aug_cz.obj_cm, cctx_cz_s)

    _prep_dual_index_for_archC!(cctx_cz_t, aug_cz.obj_cm, x)
    h_t = zeros(h_len_cz)
    hessian_cm_structured_v2!(h_t, aug_cz.obj_cm, cctx_cz_t; threaded_bins = true, tls = cctx_cz_t.tls)  # warmup
    global t_threaded_cz += @elapsed hessian_cm_structured_v2!(h_t, aug_cz.obj_cm, cctx_cz_t; threaded_bins = true, tls = cctx_cz_t.tls)

    d = maximum(abs.(h_s .- h_t))
    global maxdiff_cz = max(maxdiff_cz, d)
    check("cm_meanzc D20/W20k pt$pi_: threaded matches serial (max|Δ|=$(d))", d < 1e-8)
end
@printf("cm_meanzc: serial=%.4fs threaded=%.4fs (avg over %d calls each) maxdiff=%.4e\n",
    t_serial_cz / length(xs_cz), t_threaded_cz / length(xs_cz), length(xs_cz), maxdiff_cz)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
exit(ALL_PASS[] ? 0 : 1)
