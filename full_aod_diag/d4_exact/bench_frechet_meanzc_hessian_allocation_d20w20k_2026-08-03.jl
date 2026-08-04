# fix/profiled-functional-readiness-closeout-2026-08-03, section 3.1: allocation bytes + callback
# time for common_frechet's and cm_meanzc's REDUCED/profiled Hessian evaluators at D20/W=20,000 --
# the measurement the prior inner-readiness session's own report flagged as missing (it established
# numerical equivalence for the mul!/H_EM mirror-loop fixes but never archived a clear allocation
# table). Reports @allocated bytes for ONE post-warmup Hessian call (JIT-compiled first, measured
# second, per the codebase's own established allocation-benchmark convention), wall-clock for that
# same call, and the NO_DENSE_G_COUNTERS delta across it (must be zero -- these are REDUCED/
# operator-FG evaluators, not dense-G paths). Serial (threaded_bins=false) only -- the threaded_bins
# comparison itself is a separate gate (test_frechet_threaded_profiled_d4_2026-08-03.jl /
# test_zc_lane_cmzc_threaded_profiled_d4_2026-08-02.jl).
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
          "zc_restriction_operator.jl",
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
using Printf, LinearAlgebra, Random
flush(stdout)
println("PID=", getpid()); flush(stdout)

const W_VAL = 20_000
const D_VAL, DDEST_VAL, L_VAL = 20, 19, 50

t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; Ddest = ctx.D_dest
korea_idx, brazil_idx = 14, 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
W = size(ctx.U, 1)

function bench_hessian!(label::String, cctx, obj, hess_fn::Function, h_len::Int, x::Vector{Float64})
    _prep_dual_index_for_archC!(cctx, obj, x)
    h = zeros(h_len)
    hess_fn(h, obj, cctx)   # warmup / JIT
    c0 = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations
    t = @elapsed hess_fn(h, obj, cctx)
    alloc = @allocated hess_fn(h, obj, cctx)
    c1 = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations
    @printf("[%s] wall=%.4fs  allocated=%d bytes (%.2f MiB)  dense_G_delta=%d\n",
        label, t, alloc, alloc / 2^20, c1 - c0)
    return (label = label, wall = t, allocated = alloc, dense_g_delta = c1 - c0)
end

rows = NamedTuple[]

println("="^90); println("common_frechet"); println("="^90); flush(stdout)
aug_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L_VAL, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_f.level_targets
cctx_f = build_cm_bin_ctx(ctx, aug_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
    threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
K_probe = Vector{Float64}(undef, W); G_probe = Matrix{Float64}(undef, W, aug_f.obj_cm.d)
aug_f.obj_cm.moments!(K_probe, G_probe, collect(θ_full_calib), ctx.U, aug_f.obj_cm)
h_len_f = (cctx_f.NCORE + cctx_f.ncm) * (cctx_f.NCORE + cctx_f.ncm + 1) ÷ 2
x0_f = zeros(aug_f.obj_cm.outer_constr_index)
push!(rows, bench_hessian!("common_frechet (serial)", cctx_f, aug_f.obj_cm,
    (h, obj, cctx) -> hessian_cm_frechet_structured!(h, obj, cctx, level_targets), h_len_f, x0_f))

println("="^90); println("cm_meanzc (CM+ZC)"); println("="^90); flush(stdout)
K_MEAN, K_PAIR = 1, 1
νvec0_cm = fill(1.0, K_MEAN)
aug_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_VAL, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_cz = build_cm_meanzc_bin_ctx(ctx, aug_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
cctx_cz.nu_ref[] = collect(νvec0_cm)
K_probe2 = Vector{Float64}(undef, W); G_probe2 = Matrix{Float64}(undef, W, aug_cz.obj_cm.d)
aug_cz.obj_cm.moments!(K_probe2, G_probe2, collect(θ_full_calib), ctx.U, aug_cz.obj_cm)
h_len_cz = (cctx_cz.NCORE + cctx_cz.ncm) * (cctx_cz.NCORE + cctx_cz.ncm + 1) ÷ 2
x0_cz = zeros(aug_cz.obj_cm.outer_constr_index)
push!(rows, bench_hessian!("cm_meanzc (serial)", cctx_cz, aug_cz.obj_cm,
    (h, obj, cctx) -> hessian_cm_structured!(h, obj, cctx), h_len_cz, x0_cz))

println()
println("="^90); println("SUMMARY"); println("="^90)
for r in rows
    @printf("%-28s wall=%.4fs  allocated=%10d bytes (%7.2f MiB)  dense_G_delta=%d\n",
        r.label, r.wall, r.allocated, r.allocated / 2^20, r.dense_g_delta)
end
ok = all(r.dense_g_delta == 0 for r in rows)
println("\nBENCH_RESULT: ", ok ? "PASS (zero dense-G materialization for both families)" : "FAIL (dense-G materialized)")
flush(stdout)
exit(ok ? 0 : 1)
