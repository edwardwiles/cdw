# test_hessian_allocation_regression_2026-08-02.jl -- permanent regression safeguards (production
# Hessian audit, §20). Tolerance-ceiling gates, not fragile exact-wall-time assertions (per the
# task brief's own "use tolerance ceilings... do not add fragile exact wall-time unit tests").
#
# Thresholds below are derived from THIS audit's own real measurements (W=20,000, real D=20
# calibration point, post-fix production HEAD) with a 2x safety margin -- generous enough to
# survive normal machine-to-machine/run-to-run variance, tight enough to catch a REGRESSION back
# toward the two allocation hotspots this audit found and fixed (commit 8f1151e): cm_meanzc was
# 1,996,064 bytes/call pre-fix, common_frechet was 561,504 bytes/call pre-fix -- a regression back
# to either of those would trip its own family's ceiling by 9-14x, nowhere near the 2x margin.
#
# Usage: julia --project=. -t 10 full_aod_diag/d4_exact/test_hessian_allocation_regression_2026-08-02.jl
const D4X = @__DIR__
cd(D4X)

const _CM_FAMILY_LIST = [
    "draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
    "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
    "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
    "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
    "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
    "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
    "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
    "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
    "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
    "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
    "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
    "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
    "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
    "hcz_drawchunk_candidate_2026-07-29.jl",
    "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl",
    "c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
    "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl",
]
for f in unique(_CM_FAMILY_LIST)
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random, Statistics, Test, Dates
using Base.Threads: nthreads

const W_TEST = 20_000
const K_ZC = 3
const CM_L = 50

# 2x margin over this audit's own real post-fix measurements (bytes_per_call, W=20000/100000
# alike -- confirmed W-independent, see PRODUCTION_HESSIAN_ALLOCATION_BASELINE_2026-08-02.csv).
const ALLOC_CEILING_BYTES = Dict(
    :unrestricted    => 30_000,      # measured 11,872-12,784
    :flexible_cm     => 100_000,     # measured 41,456-41,632
    :common_frechet  => 100_000,     # measured 41,504-42,160 post-fix (was 561,504+ pre-fix)
    :origin_zc       => 100_000,     # measured 47,920-48,148
    :cm_meanzc       => 150_000,     # measured 70,392-71,544 post-fix (was 1,995,792+ pre-fix)
)
# Expected dual dimension at this audit's canonical scientific config (D=20/L=50/K=3) -- a change
# here signals a live-dimension drift the task brief's own "confirm live dimensions" requirement
# is meant to catch, not just a cosmetic difference.
const EXPECTED_N = Dict(:unrestricted => 382, :flexible_cm => 1332, :common_frechet => 1382,
                         :origin_zc => 1012, :cm_meanzc => 1962)

println("="^100)
println("HESSIAN ALLOCATION/BACKEND/DIMENSION REGRESSION SUITE -- ", Dates.now())
println("julia_threads=", nthreads(), " BLAS_threads=", BLAS.get_num_threads())
println("="^100)

@testset "Hessian allocation/backend/dimension regression" begin
    ctx0 = d20_real_setup_design(W = W_TEST, δ = 1.0, find_smallest = true,
        draw_design = :sobol_randomized, draw_seed = 20260719,
        destination_sample = :exclude_row, exclude_diagonal_gravity = true,
        gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]

    @testset "unrestricted" begin
        ctx = build_unrestricted_operator_ctx(ctx0)
        obj = ctx.obj
        θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
        K_hard, inner_x, nStatus, n_fg, n_hess, st = inner_loop_internal_compressed(obj, θ_full0, ctx)
        @test nStatus in (0, -100, -101, -103)
        x_state = collect(inner_x); n = length(x_state); hlen = n * (n + 1) ÷ 2
        @test n == EXPECTED_N[:unrestricted]
        h = Vector{Float64}(undef, hlen); fake_req = (x = x_state,); fake_res = (hess = h,)
        _callbackEvalH_inner_compressed!(nothing, nothing, fake_req, fake_res, st)   # JIT warm-up
        b = @allocated _callbackEvalH_inner_compressed!(nothing, nothing, fake_req, fake_res, st)
        @test b <= ALLOC_CEILING_BYTES[:unrestricted]
        @test obj isa OperatorPsiBundle   # confirms the no-dense-H production default is actually active
    end

    @testset "flexible_cm" begin
        SNAPS = nested_grid_sequence([10, 20, 50])
        ctx_cm, aug, bins, cctx = build_cm_production_context(ctx0, CS; L = CM_L, contrasts = :orthonormal, probs = SNAPS[CM_L])
        base = archC_base_state(x_free_calib, ctx_cm, cctx)
        @test base.inner_status in (0, -100, -101, -103)
        x_state = vcat(base.ζstar, base.λstar); n = length(x_state); hlen = n * (n + 1) ÷ 2
        @test n == EXPECTED_N[:flexible_cm]
        cb = archC_hess_cb_builder(cctx)
        h = Vector{Float64}(undef, hlen); fake_req = (x = x_state,); fake_res = (hess = h,)
        reset_core_hessian_counters!()
        cb(nothing, nothing, fake_req, fake_res, ctx_cm.obj)
        b = @allocated cb(nothing, nothing, fake_req, fake_res, ctx_cm.obj)
        @test b <= ALLOC_CEILING_BYTES[:flexible_cm]
        @test cctx.core_hessian_backend == :exact_winner_pair_parallel
        @test cctx.cm_cross_hessian_backend == :winner_bin
        @test CORE_HESSIAN_COUNTERS[].dense_core_fallback_calls == 0
    end

    @testset "common_frechet" begin
        SNAPS = nested_grid_sequence([10, 20, 50])
        pcx = build_cm_frechet_production_context(ctx0, CS; L = CM_L, contrasts = :orthonormal, probs = SNAPS[CM_L], cm_hessian_backend = :structured)
        base = archC_frechet_base_state(x_free_calib, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
        @test base.inner_status in (0, -100, -101, -103)
        x_state = vcat(base.ζstar, base.λstar); n = length(x_state); hlen = n * (n + 1) ÷ 2
        @test n == EXPECTED_N[:common_frechet]
        cb = archC_frechet_hess_cb_builder(pcx.cctx, pcx.aug.level_targets)
        h = Vector{Float64}(undef, hlen); fake_req = (x = x_state,); fake_res = (hess = h,)
        reset_core_hessian_counters!()
        cb(nothing, nothing, fake_req, fake_res, pcx.ctx_cm.obj)
        b = @allocated cb(nothing, nothing, fake_req, fake_res, pcx.ctx_cm.obj)
        @test b <= ALLOC_CEILING_BYTES[:common_frechet]
        @test pcx.cctx.core_hessian_backend == :exact_winner_pair_parallel
        @test CORE_HESSIAN_COUNTERS[].dense_core_fallback_calls == 0
    end

    @testset "cm_meanzc" begin
        ctx_cm, aug, cctx, bins = build_cm_meanzc_production_context(ctx0, CS; L = CM_L, K_mean = K_ZC, K_pair = K_ZC,
            contrasts = :orthonormal, meanzc_basis = :direct)
        νvec = Float64.(factorial.(1:K_ZC))
        base = archC_meanzc_base_state(x_free_calib, νvec, ctx_cm, cctx)
        @test base.inner_status in (0, -100, -101, -103)
        x_state = vcat(base.ζstar, base.λstar); n = length(x_state); hlen = n * (n + 1) ÷ 2
        @test n == EXPECTED_N[:cm_meanzc]
        cb = archC_hess_cb_builder(cctx)
        h = Vector{Float64}(undef, hlen); fake_req = (x = x_state,); fake_res = (hess = h,)
        reset_core_hessian_counters!()
        cb(nothing, nothing, fake_req, fake_res, ctx_cm.obj)
        b = @allocated cb(nothing, nothing, fake_req, fake_res, ctx_cm.obj)
        @test b <= ALLOC_CEILING_BYTES[:cm_meanzc]
        @test cctx.zc_gram_backend == :blas_syrk
        @test cctx.hcz_prep_backend == :draw_chunk_reordered
        @test cctx.zc_ez_backend == :drawmajor_v2
        @test CORE_HESSIAN_COUNTERS[].dense_core_fallback_calls == 0
    end

    @testset "origin_zc" begin
        layout = OriginByPowerLayout(ctx0.D, K_ZC, K_ZC)
        nu0 = Vector{Float64}(undef, n_eta(layout))
        for k in 1:K_ZC, o in 1:ctx0.D
            nu0[target_index(layout, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
        end
        pcx = build_originzc_production_context(ctx0, CS, layout; fg_backend = :operator, moment_representation = :operator)
        base = archOZ_base_state(x_free_calib, nu0, pcx.ctx_cm)
        @test base.inner_status in (0, -100, -101, -103)
        x_state = vcat(base.ζstar, base.λstar); n = length(x_state); hlen = n * (n + 1) ÷ 2
        @test n == EXPECTED_N[:origin_zc]
        cb = archA_partitioned_hess_cb_builder(pcx.octx)
        h = Vector{Float64}(undef, hlen); fake_req = (x = x_state,); fake_res = (hess = h,)
        reset_core_hessian_counters!()
        cb(nothing, nothing, fake_req, fake_res, pcx.ctx_cm.obj)
        b = @allocated cb(nothing, nothing, fake_req, fake_res, pcx.ctx_cm.obj)
        @test b <= ALLOC_CEILING_BYTES[:origin_zc]
        @test pcx.octx.zc_gram_backend == :blas_syrk
        @test pcx.octx.zc_ez_backend == :drawmajor_v2
        @test CORE_HESSIAN_COUNTERS[].dense_core_fallback_calls == 0
    end
end
