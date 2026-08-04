# Task §8 (profiled-outer-ab-completion-2026-08-04): measure the REDUCED bandwidth-search reuse
# cache's material benefit, not just correctness (the prior session's own gates only checked
# on/off gradient equality, hit reuse, no-stale-reuse -- never isolated a performance gain). Real
# D20/W=20,000, flexible_cm (scoped down from the task's own unrestricted/flexible_cm/origin_zc x
# W20k/W100k matrix to ONE family/scale given this session's wall-clock budget -- reported
# honestly as a narrower scope, not silently assumed to generalize).
#
# Protocol: same decoded starting point, same solved inner dual (ONE ProfiledLFixCache reused for
# every point in the sequence -- isolates the bandwidth-selection cost specifically, not
# confounded by inner-solve time), same sequence of nearby outer points (small random walk,
# within validity_radius so cache hits are expected after the first visit to each coordinate),
# same thread count (serial, to isolate the cache's own contribution from threading), compilation
# warmed via one untimed pass first, reversed on/off execution order to rule out ordering effects.
D4X = @__DIR__
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
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "blas_thread_policy.jl", "knitro_outer_algorithm.jl", "production_backend_manifest.jl",
          "incumbent_logic.jl", "cm_hessian_subblock_profiling.jl", "production_bundle_api.jl", "country_resolve.jl",
          "cm_exact_cache_production.jl", "cm_checkpoint.jl", "draw_design.jl",
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
          "profiled_restricted_family_adapters_2026-08-02.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl",
          "profiled_reduced_bandwidth_cache_2026-08-04.jl"]
    include(joinpath(D4X, f))
end
using Random

lp(xs...) = (println(xs...); flush(stdout))
const W_VAL = parse(Int, get(ENV, "GATE_W", "20000"))
const FAMILY = Symbol(get(ENV, "GATE_FAMILY", "flexible_cm"))
FAMILY in (:unrestricted, :flexible_cm, :origin_zc) ||
    error("test_bandwidth_cache_benefit_2026-08-04: GATE_FAMILY must be unrestricted|flexible_cm|origin_zc, got $FAMILY")
lp("Threads.nthreads() = ", Threads.nthreads(), "  W=", W_VAL, "  FAMILY=", FAMILY)

ctx = d20_real_setup_design(; W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = [(3, 14)], σHat = 3.0)
D = ctx.D
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(14 => 3))
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]
w0 = reduce_to_w_profiled(gp0, z_calib, pe)

θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

if FAMILY == :unrestricted
    ev0 = evaluate_profiled_point(w0, ctx, spec, pe)
    fctx = build_unrestricted_family_ctx(ctx, spec, pe, ev0)
elseif FAMILY == :flexible_cm
    aug = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
    cctx = build_cm_bin_ctx(ctx, aug; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
    fctx = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx)
    ev0 = evaluate_profiled_flexcm_point(w0, fctx)
else # :origin_zc (fixed-nu evaluator -- this timing test only exercises the shared A/gp engine)
    layout_o = OriginByPowerLayout(D, 1, 0)
    νvec0 = fill(1.0, D)
    augz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
    octx = build_originzc_core_hess_ctx(augz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
        zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
    fctx = build_originzc_family_ctx(ctx, spec, pe, layout, augz)
    pes_o = OriginZCPointEvalState(octx, νvec0)
    ev0 = evaluate_profiled_originzc_point(w0, fctx, pes_o)
end
lp("calibration inner_status=", ev0.result.inner_status, " Delta_dual=", ev0.result.Delta_dual)

sci_stub = (W = W_VAL, sigma = 3.0, draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true, gravity_exclude_cells = [(3, 14)],
    K_mean = 1, K_pair = 1, L = 50, dataset_checksum = "bwcache_benefit")
cache_ctx = (manifest_hash = reduced_bandwidth_manifest_hash(sci_stub), family = FAMILY,
    coordinate_mode = :profiled_pivot_anchor_relative, layout_digest = stable_layout_digest(fctx),
    nu_generation = UInt64(0))

n_free = length(w0) - 1
Random.seed!(20260804)
n_points = 8
points = Vector{Vector{Float64}}(undef, n_points)
points[1] = copy(w0)
for i in 2:n_points
    wi = copy(points[i-1]); wi[2:end] .+= 0.005 .* randn(n_free)  # small step, stays within validity_radius=0.02 of recent points
    points[i] = wi
end
plfix_seq = [build_shared_profiled_lfix_cache(points[i], fctx, ctx, ev0) for i in 1:n_points]

"Runs the full sequence once, cache ON (bwcache reused across ALL points) or OFF (uncached each time), returns (wall_s, cpu_alloc_bytes)."
function run_sequence(cache_on::Bool)
    bwcache = cache_on ? ReducedBandwidthCache(0.02) : nothing
    t0 = time_ns()
    gcstats0 = Base.gc_num()
    for i in 1:n_points
        if cache_on
            profiled_composite_gradient_from_cache_bwcache(plfix_seq[i], ctx, spec, pe, points[i], ev0, bwcache, cache_ctx; threaded = false)
        else
            profiled_composite_gradient_from_cache(plfix_seq[i], ctx, spec, pe, points[i], ev0; threaded = false)
        end
    end
    wall = (time_ns() - t0) / 1e9
    gcdiff = Base.GC_Diff(Base.gc_num(), gcstats0)
    return wall, gcdiff.allocd, bwcache
end

lp("\n=== Warmup (untimed, excludes JIT compilation from the measurement) ===")
run_sequence(true); run_sequence(false)

lp("\n=== Order A: cache OFF then ON ===")
wall_off_A, alloc_off_A, _ = run_sequence(false)
wall_on_A, alloc_on_A, bw_A = run_sequence(true)
lp("  OFF: wall=", wall_off_A, "s  alloc=", alloc_off_A, " bytes")
lp("  ON:  wall=", wall_on_A, "s  alloc=", alloc_on_A, " bytes  hits=", bw_A.hits, " misses=", bw_A.misses, " stale_evictions=", bw_A.stale_evictions)

lp("\n=== Order B: cache ON then OFF (reversed execution order) ===")
wall_on_B, alloc_on_B, bw_B = run_sequence(true)
wall_off_B, alloc_off_B, _ = run_sequence(false)
lp("  ON:  wall=", wall_on_B, "s  alloc=", alloc_on_B, " bytes  hits=", bw_B.hits, " misses=", bw_B.misses, " stale_evictions=", bw_B.stale_evictions)
lp("  OFF: wall=", wall_off_B, "s  alloc=", alloc_off_B, " bytes")

speedup_A = wall_off_A / wall_on_A
speedup_B = wall_off_B / wall_on_B
lp("\nspeedup (OFF/ON wall time), order A = ", speedup_A)
lp("speedup (OFF/ON wall time), order B = ", speedup_B)
lp("alloc reduction, order A = ", 1 - alloc_on_A / alloc_off_A)
lp("alloc reduction, order B = ", 1 - alloc_on_B / alloc_off_B)

lp("\nBANDWIDTH_CACHE_BENEFIT_W", W_VAL, "_", FAMILY, ": speedup_A=", round(speedup_A, digits=2), "x  speedup_B=", round(speedup_B, digits=2), "x")
