# fix/profiled-functional-readiness-closeout-2026-08-03, task §4.2 stage 2: promote the
# W=20,000-classified-infeasible origin-ZC/CM+ZC points (shift=6.0, both fast: 0.40s/0.50s at
# W=20,000, real 2026-08-04 result in fastreject_stage1_w20k_2026-08-04.log) to the real
# W=100,000 scale, confirming rejection and recording iterations/time there.
#
# flexible_CM's own W=100,000 data point already exists (this continuation's earlier eval18
# maxit=100/maxit=1000 work): nStatus=-400 at production maxit=100 (NOT fast at this W -- the
# "slow genuinely unbounded" category), nStatus=-300 only once maxit=1000 (577.67s). Combined with
# stage 1's W=20,000 result for the SAME point (nStatus=-300 in 12.72s, genuinely fast at that
# smaller W), this is a real, non-fabricated W-dependence finding for flexible_CM -- not re-run
# here, already decisively established.
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
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
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
          "reduced_restricted_family_verification_2026-08-03.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf

t0 = time()
println("PID=", getpid()); flush(stdout)

function try_status(label::String, thunk)
    t = @elapsed result = try
        r = thunk()
        (status = r.inner_status, n_fg = r.n_fg, n_hess = r.n_hess)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        m = match(r"nStatus=(-?\d+)", e.msg)
        m === nothing && rethrow()
        (status = parse(Int, m.captures[1]), n_fg = -1, n_hess = -1)
    end
    @printf("  [%s] wall=%.2fs  nStatus=%d  n_fg=%d  n_hess=%d\n", label, t, result.status, result.n_fg, result.n_hess)
    flush(stdout)
    return (label = label, wall = t, result...)
end

W_VAL = 100_000
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
@printf("context build (W=100,000): %.2fs\n", t_ctx); flush(stdout)
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

SHIFT = 6.0   # the cleanest fast-infeasible example from the W=20,000 sweep (0.40s/0.50s)
w_shift = copy(w_calib); w_shift[2:end] .+= SHIFT
decoded_shift = decode_outer_profiled(w_shift, ctx, pe)

println("="^90); println("origin-ZC shift=$SHIFT at real W=100,000"); println("="^90); flush(stdout)
layout_o = OriginByPowerLayout(D, 1, 0)
aug_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_oz = build_originzc_core_hess_ctx(aug_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
νvec0_oz = fill(1.0, D)
r_oz = try_status("origin-ZC shift=$SHIFT W100k", () -> reduced_originzc_base_state(decoded_shift.xf, ctx, layout, octx_oz, νvec0_oz))

println("\n" * "="^90); println("CM+ZC shift=$SHIFT at real W=100,000"); println("="^90); flush(stdout)
const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
aug_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_cz = build_cm_meanzc_bin_ctx(ctx, aug_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = true, inner_fg_backend = :dense_reference, profiled_layout = layout)
νvec0_cz = fill(1.0, K_MEAN)
r_cz = try_status("CM+ZC shift=$SHIFT W100k", () -> reduced_meanzc_base_state(decoded_shift.xf, νvec0_cz, ctx, layout, cctx_cz))

println("\n" * "="^90); println("VERDICT"); println("="^90)
@printf("origin-ZC:  nStatus=%d (expect -300, confirming fast-reject also holds at real W=100,000)\n", r_oz.status)
@printf("CM+ZC:      nStatus=%d (expect -300, confirming fast-reject also holds at real W=100,000)\n", r_cz.status)
ok = r_oz.status == -300 && r_cz.status == -300
println(ok ? "FASTREJECT_W100K_CONFIRMED" : "FASTREJECT_W100K_UNEXPECTED -- investigate")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
ok || exit(1)
