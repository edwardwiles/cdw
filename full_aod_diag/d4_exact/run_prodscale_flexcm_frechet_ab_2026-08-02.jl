# Performance closeout task (2026-08-02), Sections 4/12 extension: production-scale FULL-vs-REDUCED
# A/B for flexible_CM and common_Frechet -- the two families with NO ZC restriction block (no
# H_EZ/H_CZ, no zc_ez_backend/hcz_prep_backend at all), so unlike CM+ZC/origin-ZC there is no
# BASELINE-vs-OPTIMIZED backend axis to isolate here; this just confirms the reduced/profiled path
# (already unaffected by this task's Section 6/7/8 fixes, since it shares none of that dispatch
# code) is at parity with or faster than FULL for these two families too, completing "all five
# families" coverage for Section 4/12 (unrestricted is separately out of scope -- Section 11's own
# allocation note, no dedicated production-context wrapper the same way).
#
# Reuses run_coldsolve_{flexcm,frechet}_w100k_2026-08-02.jl's own REDUCED builder chain verbatim
# and gate3_compile_free_backend_ab_2026-08-01.jl's own genuine-cold FULL methodology verbatim.
# Must be launched as a FRESH julia process.
const D4X = @__DIR__
t0_total = time()
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "country_resolve.jl", "winners.jl", "oracle.jl",
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
          "operator_verification.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, Random
lp(xs...) = (println(xs...); flush(stdout))
lp("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s")

const W_VAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const D_VAL, DDEST_VAL, L_VAL = 20, 19, 50
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true

t_ctx = @elapsed ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; Ddest = ctx.D_dest
@assert D == D_VAL && Ddest == DDEST_VAL "expected D=$D_VAL/Ddest=$DDEST_VAL, got D=$D/Ddest=$Ddest"
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
W = size(ctx.U, 1)

t_full = @elapsed begin
    pcx_flexcm = build_cm_production_context(ctx, CS; L = L_VAL, contrasts = :anchored)
    pcx_frechet = build_cm_frechet_production_context(ctx, CS; L = L_VAL, contrasts = :anchored, cm_hessian_backend = :structured)
end
@printf("FULL context build: %.2fs\n", t_full); flush(stdout)

korea_idx = 14; brazil_idx = 3
t_layout = @elapsed begin
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
    cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
    has_france = cf_probe.cf_col > 0
    layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
    reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
end
@printf("REDUCED layout build: %.2fs\n", t_layout); flush(stdout)

t_reduced = @elapsed begin
    aug_reduced_flexcm = build_cm_augmented_obj_archB(ctx, CS; L = L_VAL, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
    cctx_reduced_flexcm = build_cm_bin_ctx(ctx, aug_reduced_flexcm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
    aug_reduced_frechet = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L_VAL, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
    cctx_reduced_frechet = build_cm_bin_ctx(ctx, aug_reduced_frechet; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false,
        core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
    level_targets_frechet = aug_reduced_frechet.level_targets
end
@printf("REDUCED object/cctx build: %.2fs\n", t_reduced); flush(stdout)

function timed_full_flexcm()
    pcx_flexcm.ctx_cm.obj.x .= NaN
    reset_no_dense_g_counters!(); prof_reset!()
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K_, base, verify = cm_production_value_verified_screened(x_free_calib, pcx_flexcm)
    wall = time() - t0
    return (family = "flexcm", arm = "FULL", wall_s = wall, iters = CS.INNER_ITERS_TOTAL[] - iters0,
        nStatus = base.inner_status, dense_econ = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations)
end
function timed_full_frechet()
    pcx_frechet.ctx_cm.obj.x .= NaN
    reset_no_dense_g_counters!(); prof_reset!()
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K_, base, verify = cm_frechet_production_value_verified_screened(x_free_calib, pcx_frechet)
    wall = time() - t0
    return (family = "frechet", arm = "FULL", wall_s = wall, iters = CS.INNER_ITERS_TOTAL[] - iters0,
        nStatus = base.inner_status, dense_econ = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations)
end
function timed_reduced_flexcm()
    reset_no_dense_g_counters!(); prof_reset!()
    t0 = time()
    base = reduced_cm_base_state(x_free_calib, ctx, layout, cctx_reduced_flexcm)
    wall = time() - t0
    return (family = "flexcm", arm = "REDUCED", wall_s = wall, iters = base.n_fg,
        nStatus = base.inner_status, dense_econ = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations)
end
function timed_reduced_frechet()
    reset_no_dense_g_counters!(); prof_reset!()
    t0 = time()
    base = reduced_frechet_base_state(x_free_calib, ctx, layout, cctx_reduced_frechet, level_targets_frechet)
    wall = time() - t0
    return (family = "frechet", arm = "REDUCED", wall_s = wall, iters = base.n_fg,
        nStatus = base.inner_status, dense_econ = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations)
end

lp("="^100); lp("THROWAWAY warm-up pass (discarded)")
timed_full_flexcm(); timed_full_frechet(); timed_reduced_flexcm(); timed_reduced_frechet()
lp("warm-up done, discarded")

lp("="^100); lp("REAL MEASUREMENTS  W=$(W_VAL)")
rows = NamedTuple[]
for (label, fn) in [("flexcm FULL", timed_full_flexcm), ("flexcm REDUCED", timed_reduced_flexcm),
                     ("frechet FULL", timed_full_frechet), ("frechet REDUCED", timed_reduced_frechet)]
    r = fn(); push!(rows, r)
    @printf("  [%-16s] wall=%8.3fs iters=%-4d nStatus=%-4d dense_econ=%d\n", label, r.wall_s, r.iters, r.nStatus, r.dense_econ)
    flush(stdout)
end

outpath = joinpath(D4X, "..", "..", "docs", "PRODSCALE_FLEXCM_FRECHET_AB_W$(W_VAL)_2026-08-02.csv")
open(outpath, "w") do io
    println(io, "family,arm,wall_s,iters,nStatus,dense_econ")
    for r in rows
        println(io, "$(r.family),$(r.arm),$(r.wall_s),$(r.iters),$(r.nStatus),$(r.dense_econ)")
    end
end
lp("Wrote ", outpath)
@printf("TOTAL WALL: %.2fs\n", time() - t0_total)
lp("DONE")
