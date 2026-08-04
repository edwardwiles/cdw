# fix/profiled-functional-readiness-closeout-2026-08-03, task §4.2 stage 2: promote the
# W=20,000-classified-infeasible unrestricted/common_frechet points (shift=6.0, both fast:
# 0.31s/1.03s at W=20,000, real 2026-08-04 result in
# warmstart_fastreject_unrestricted_frechet_2026-08-04.log) to real W=100,000.
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
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_stable_layout_digest_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "reduced_restricted_family_verification_2026-08-03.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf

t0 = time()
println("PID=", getpid()); flush(stdout)

function try_status(label::String, thunk)
    t = @elapsed result = try
        r = thunk()
        (status = r.inner_status, n_fg = r.n_fg_calls === nothing ? -1 : -1)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        m = match(r"nStatus=(-?\d+)", e.msg)
        m === nothing && rethrow()
        (status = parse(Int, m.captures[1]), n_fg = -1)
    end
    @printf("  [%s] wall=%.2fs  nStatus=%d\n", label, t, result.status)
    flush(stdout)
    return (label = label, wall = t, status = result.status)
end

W_VAL = 100_000
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D
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
w_calib = reduce_to_w_profiled(θ0[3+D], z_calib, pe)

SHIFT = 6.0
w_shift = copy(w_calib); w_shift[2:end] .+= SHIFT
decoded_shift = decode_outer_profiled(w_shift, ctx, pe)

println("="^90); println("unrestricted shift=$SHIFT at real W=100,000"); println("="^90); flush(stdout)
r_u = try_status("unrestricted shift=$SHIFT W100k", () -> begin
    ev = evaluate_profiled_point(w_shift, ctx, spec, pe)
    (inner_status = ev.result.inner_status, n_fg_calls = nothing)
end)

println("\n" * "="^90); println("common_frechet shift=$SHIFT at real W=100,000"); println("="^90); flush(stdout)
aug_cf = build_cm_frechet_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_cf.level_targets
cctx_fr = build_cm_bin_ctx(ctx, aug_cf; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
r_fr = try_status("common_frechet shift=$SHIFT W100k", () -> reduced_frechet_base_state(decoded_shift.xf, ctx, layout, cctx_fr, level_targets))

println("\n" * "="^90); println("VERDICT"); println("="^90)
@printf("unrestricted:    nStatus=%d (expect -300)\n", r_u.status)
@printf("common_frechet:  nStatus=%d (expect -300)\n", r_fr.status)
ok = r_u.status == -300 && r_fr.status == -300
println(ok ? "FASTREJECT_W100K_CONFIRMED" : "FASTREJECT_W100K_UNEXPECTED -- investigate")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
ok || exit(1)
