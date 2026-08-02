# Genuine-cold ZC Hessian K3 closeout task (2026-08-01) -- CORRECTED true-cold single-solve A/B.
#
# Supersedes zc_direct_true_cold_inner_ab_2026-08-01.jl's fresh-Julia-process-per-measurement
# methodology, which was found (2026-08-01, cross-session catch) to conflate JIT compilation cost
# with actual solve time -- every fresh process pays full first-call compilation for the entire
# KNITRO/objective/Hessian call graph, which is NOT small for this codebase. Confirmed: this
# session's own W=100,000 cm_meanzc reference number (143.1s) was ~5.4x a validated compile-free
# baseline (26.7s, same point/config, unmodified production code, another session's
# profile_full_table_2026-08-01.jl on campaign/sigma3-W500k-fullA-10x10-launch-2026-08-01).
#
# This script adopts that session's own validated idiom exactly: ONE throwaway warm-up pass
# (discarded) triggers JIT compilation once; every subsequent REAL measurement then runs in the
# SAME already-JIT-warm process, with `obj.x .= NaN` (this codebase's own established cold-start
# reset, c9_phase3c_correctness_d4.jl:43) immediately before each solve -- a genuine cold KNITRO
# start with zero compile-time contamination. Real KNITRO iteration count (`CS.INNER_ITERS_TOTAL[]`
# diff) is tracked alongside wall time, exactly as the reference script does.
const _D4E = @__DIR__
for f in ["draw_design.jl", "country_resolve.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl",
          "hcz_reordered_candidate_2026-08-01.jl", "hez_drawmajor_candidate_2026-08-01.jl",
          "hez_drawmajor_v2_candidate_2026-08-01.jl", "hzz_chunked_syrk_candidate_2026-08-01.jl",
          "lfix_buffer_reuse.jl", "bandwidth_cache_policy.jl", "fast_range_screen.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates, Statistics, Random, LinearAlgebra
lp(xs...) = (println(xs...); flush(stdout))

const DELTA = 0.01
const FIND_SMALLEST = true
const DEST_SAMPLE = :exclude_row
const DRAW_DESIGN = :sobol_randomized
const DRAW_SEED = 20260719
const EXCLUDE_DIAGONAL_GRAVITY = true
const SIGMA = 3.0
const K = 3
const CM_L = 50

function build_ctxs(W)
    t_ctx = time()
    ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
        draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED, destination_sample = DEST_SAMPLE,
        exclude_diagonal_gravity = EXCLUDE_DIAGONAL_GRAVITY, σHat = SIGMA,
        gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea())
    lp(">> ctx built (W=", W, ") in ", round(time() - t_ctx, digits = 1), "s")
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    pcx_meanzc = build_cm_meanzc_production_context(ctx, CS; L = CM_L, K_mean = K, K_pair = K, contrasts = :orthonormal)
    oz_layout = OriginByPowerLayout(ctx.D, K, K)
    pcx_originzc = build_originzc_production_context(ctx, CS, oz_layout; fg_backend = :operator, moment_representation = :operator)
    nu_meanzc = Float64.(factorial.(1:K))
    nu_originzc = begin
        nu = Vector{Float64}(undef, n_eta(oz_layout))
        for k in 1:K, o in 1:ctx.D
            nu[target_index(oz_layout, o, k)] = mean(@view (ctx.U .^ k)[:, o])
        end
        nu
    end
    return ctx, x_free_calib, pcx_meanzc, pcx_originzc, nu_meanzc, nu_originzc
end

ARM_BACKENDS = Dict(
    :reference      => (:reference, :draw_chunk_thread_local, :winner_bin),
    :hzz_only       => (:blas_syrk, :draw_chunk_thread_local, :winner_bin),
    :hcz_only       => (:reference, :draw_chunk_reordered,    :winner_bin),
    :hez_only       => (:reference, :draw_chunk_thread_local, :drawmajor_v2),
    :all_optimized  => (:blas_syrk, :draw_chunk_reordered,    :drawmajor_v2),
)

function timed_meanzc(pcx, x_free_calib, nu, arm::Symbol)
    hzz_be, hcz_be, hez_be = ARM_BACKENDS[arm]
    pcx.cctx.zc_gram_backend = hzz_be
    pcx.cctx.hcz_prep_backend = hcz_be
    pcx.cctx.zc_ez_backend = hez_be
    pcx.ctx_cm.obj.x .= NaN   # established cold-start reset idiom -- genuine cold KNITRO start, zero compile cost
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K_, base, verify = cm_meanzc_production_value_verified_screened(x_free_calib, nu, pcx)
    wall = time() - t0
    iters = CS.INNER_ITERS_TOTAL[] - iters0
    return (arm = arm, wall_s = wall, iters = iters, nStatus = base.inner_status, Delta_dual = get(verify, :Delta_dual, NaN))
end

function timed_originzc(pcx, x_free_calib, nu, arm::Symbol)
    hzz_be, _, hez_be = ARM_BACKENDS[arm]
    pcx.octx.zc_gram_backend = hzz_be
    pcx.octx.zc_ez_backend = hez_be
    pcx.ctx_cm.obj.x .= NaN
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K_, base, verify = cm_originzc_production_value_verified_screened(x_free_calib, nu, pcx)
    wall = time() - t0
    iters = CS.INNER_ITERS_TOTAL[] - iters0
    return (arm = arm, wall_s = wall, iters = iters, nStatus = base.inner_status, Delta_dual = get(verify, :Delta_dual, NaN))
end

const ARMS = [:reference, :hzz_only, :hcz_only, :hez_only, :all_optimized]
const ORIGINZC_ARMS = [:reference, :hzz_only, :hez_only, :all_optimized]

lp("="^100); lp(">> THROWAWAY warm-up pass at W=100,000 (discarded, purely to trigger JIT compilation for every backend path)")
begin
    ctx, x_free_calib, pcx_meanzc, pcx_originzc, nu_meanzc, nu_originzc = build_ctxs(100_000)
    for arm in ARMS
        timed_meanzc(pcx_meanzc, x_free_calib, nu_meanzc, arm)
    end
    for arm in ORIGINZC_ARMS
        timed_originzc(pcx_originzc, x_free_calib, nu_originzc, arm)
    end
    lp(">> warm-up done, discarded")
end

rows = NamedTuple[]
for W in [100_000, 500_000]
    lp("="^100); lp("REAL W=", W, "  ", now())
    ctx, x_free_calib, pcx_meanzc, pcx_originzc, nu_meanzc, nu_originzc = build_ctxs(W)
    for arm in ARMS
        r = timed_meanzc(pcx_meanzc, x_free_calib, nu_meanzc, arm)
        @printf("  [cm_meanzc|W=%-7d|%-14s] wall=%8.3fs iters=%-4d nStatus=%-4d Delta=%.6e\n", W, arm, r.wall_s, r.iters, r.nStatus, r.Delta_dual)
        push!(rows, (family = "cm_meanzc", W = W, arm = arm, wall_s = r.wall_s, iters = r.iters, nStatus = r.nStatus, Delta_dual = r.Delta_dual))
    end
    for arm in ORIGINZC_ARMS
        r = timed_originzc(pcx_originzc, x_free_calib, nu_originzc, arm)
        @printf("  [origin_zc|W=%-7d|%-14s] wall=%8.3fs iters=%-4d nStatus=%-4d Delta=%.6e\n", W, arm, r.wall_s, r.iters, r.nStatus, r.Delta_dual)
        push!(rows, (family = "origin_zc", W = W, arm = arm, wall_s = r.wall_s, iters = r.iters, nStatus = r.nStatus, Delta_dual = r.Delta_dual))
    end
end

outpath = joinpath(_D4E, "..", "..", "docs", "ZC_COMPILE_FREE_BACKEND_AB_2026-08-01.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "family,W,arm,wall_s,iters,nStatus,Delta_dual")
    for r in rows
        println(io, "$(r.family),$(r.W),$(r.arm),$(r.wall_s),$(r.iters),$(r.nStatus),$(r.Delta_dual)")
    end
end
lp("Wrote ", outpath)
lp("="^100); lp("DONE"); lp("="^100)
