# Performance closeout task (2026-08-02), Sections 4/5/9/12: production-dimension (D=20, Ddest=19,
# L=50, K_mean=3, K_pair=3) genuine-cold FULL-vs-REDUCED A/B for origin-ZC and CM+ZC, at W=100,000
# (first pass) and W=500,000 (second pass, launched separately -- much slower per family).
#
# Reuses EXISTING, already-validated harnesses verbatim rather than inventing a new comparison
# method:
#   - FULL arm: gate3_compile_free_backend_ab_2026-08-01.jl's own `build_ctxs`/production-context
#     builders and cold-reset methodology (`obj.x .= NaN` before each timed call).
#   - REDUCED arm: run_prodscale_{cmzc,originzc}_2026-08-02.jl's own reduced-path builder chain
#     (build_profiled_economic_moment_layout -> build_reduced_base_obj_for_family ->
#     build_cm_meanzc_augmented_obj/build_originzc_augmented_obj -> build_cm_meanzc_bin_ctx/
#     build_originzc_core_hess_ctx) and base-state entry points
#     (reduced_meanzc_base_state/reduced_originzc_base_state), unchanged.
#
# REDUCED arms isolate this task's Section 6/7/8 fixes incrementally by varying ONLY the backend
# selector kwargs on an otherwise-identical profiled cctx/octx (no code branch, no git checkout):
#   REDUCED_BASELINE:          zc_ez_backend=:winner_bin (forces the pre-fix base-kernel fallback --
#                               :winner_bin is not :drawmajor_v2, so it hits the same `else` branch
#                               the unconditional pre-fix call always used), hcz_prep_backend=
#                               :origin_owned (hcz_prep_dispatch!'s :origin_owned branch calls the
#                               SAME bin_zc_cross_hessian_fill! the pre-fix code called directly),
#                               threaded_bins=false.
#   REDUCED_DRAWMAJOR_ONLY:    zc_ez_backend=:drawmajor_v2 (Section 6 fix alone), hcz_prep_backend=
#                               :origin_owned, threaded_bins=false.
#   REDUCED_DRAWMAJOR_PLUS_HCZ: zc_ez_backend=:drawmajor_v2, hcz_prep_backend=:draw_chunk_reordered
#                               (Sections 6+7), threaded_bins=false.
#   REDUCED_ALL_ACCEPTED:      zc_ez_backend=:drawmajor_v2, hcz_prep_backend=:draw_chunk_reordered,
#                               threaded_bins=true (Sections 6+7+8).
# origin-ZC has no CM-grid/H_CZ/threaded_bins concept at all (confirmed, no such fields on
# OriginZCCoreHessCtx) -- only REDUCED_BASELINE and REDUCED_ALL_ACCEPTED (=:drawmajor_v2) apply.
#
# Must be launched as a FRESH julia process. Usage: julia run_prodscale_full_vs_reduced_ab_2026-08-02.jl <W>
const D4X = @__DIR__
t0_total = time()
# Include order matters in this codebase (later files assume types defined by earlier ones) --
# this list is gate3_compile_free_backend_ab_2026-08-01.jl's own proven order (it already builds
# the FULL production contexts successfully) verbatim, with context.jl/context_real_d20.jl spliced
# in at the front (needed for d20_real_setup, not gate3's own d20_real_setup_design), and the
# REDUCED-path-specific files appended at the end in run_prodscale_{cmzc,originzc}_2026-08-02.jl's
# own proven relative order among themselves -- reusing two independently-working include lists
# rather than re-deriving one from scratch.
for f in ["context.jl", "context_real_d20.jl",
          "draw_design.jl", "country_resolve.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
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
          "lfix_buffer_reuse.jl", "bandwidth_cache_policy.jl", "fast_range_screen.jl",
          "cm_checkpoint.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "operator_verification.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl",
          "reduced_originzc_verification_2026-08-02.jl",
          "cm_meanzc_lookup_kernels.jl", "cm_meanzc_lookup_production.jl",
          "recover_full_a_2026-07-31.jl", "reduced_recovery_from_lfd_2026-08-01.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, Dates, Statistics, Random, LinearAlgebra
lp(xs...) = (println(xs...); flush(stdout))
lp("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s")

const W_VAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const D_VAL, DDEST_VAL, L_VAL, K_MEAN, K_PAIR = 20, 19, 50, 3, 3
const DELTA = 1.0

CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true

lp("="^100); lp("FULL-vs-REDUCED production-scale A/B: D=$D_VAL Ddest=$DDEST_VAL L=$L_VAL K_mean=$K_MEAN K_pair=$K_PAIR W=$W_VAL")
lp("="^100)

t_ctx = @elapsed ctx = d20_real_setup(W = W_VAL, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; Ddest = ctx.D_dest
@assert D == D_VAL && Ddest == DDEST_VAL "expected D=$D_VAL/Ddest=$DDEST_VAL, got D=$D/Ddest=$Ddest"
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
W = size(ctx.U, 1)

# ---- FULL contexts (dense-Hessian-optimized production path) ----
t_full = @elapsed begin
    pcx_meanzc_full = build_cm_meanzc_production_context(ctx, CS; L = L_VAL, K_mean = K_MEAN, K_pair = K_PAIR, contrasts = :orthonormal)
    oz_layout_full = OriginByPowerLayout(ctx.D, K_MEAN, K_PAIR)
    pcx_originzc_full = build_originzc_production_context(ctx, CS, oz_layout_full; fg_backend = :operator, moment_representation = :operator)
end
nu_meanzc = Float64.(factorial.(1:K_MEAN))
nu_originzc_full = begin
    nu = Vector{Float64}(undef, n_eta(oz_layout_full))
    for k in 1:K_MEAN, o in 1:ctx.D
        nu[target_index(oz_layout_full, o, k)] = mean(@view (ctx.U .^ k)[:, o])
    end
    nu
end
@printf("FULL context build: %.2fs\n", t_full); flush(stdout)

# ---- REDUCED layout + base objects (shared across all REDUCED arms) ----
korea_idx = 14; brazil_idx = 3
t_layout = @elapsed begin
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
    cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
    has_france = cf_probe.cf_col > 0
    layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
    reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
end
@printf("REDUCED layout build: %.2fs  total_reduced_econ=%d  france=%s\n", t_layout, layout.total_reduced_economic_moments, has_france); flush(stdout)

νvec0_meanzc = [Float64(factorial(k)) for k in 1:K_MEAN]
layout_o = OriginByPowerLayout(D, K_MEAN, K_PAIR)
νfull0_originzc = vcat([fill(Float64(factorial(k)), D) for k in 1:K_MEAN]...)

aug_reduced_meanzc = build_cm_meanzc_augmented_obj(ctx, CS; L = L_VAL, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
aug_reduced_originzc = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)

CMZC_ARMS = Dict(
    :REDUCED_BASELINE           => (zc_ez_backend = :winner_bin,   hcz_prep_backend = :origin_owned,       use_threaded_bins = false),
    :REDUCED_DRAWMAJOR_ONLY     => (zc_ez_backend = :drawmajor_v2, hcz_prep_backend = :origin_owned,       use_threaded_bins = false),
    :REDUCED_DRAWMAJOR_PLUS_HCZ => (zc_ez_backend = :drawmajor_v2, hcz_prep_backend = :draw_chunk_reordered, use_threaded_bins = false),
    :REDUCED_ALL_ACCEPTED       => (zc_ez_backend = :drawmajor_v2, hcz_prep_backend = :draw_chunk_reordered, use_threaded_bins = true),
)
ORIGINZC_ARMS = Dict(
    :REDUCED_BASELINE     => (zc_ez_backend = :winner_bin,),
    :REDUCED_ALL_ACCEPTED => (zc_ez_backend = :drawmajor_v2,),
)

# build_cm_meanzc_bin_ctx/build_originzc_core_hess_ctx do NOT accept zc_ez_backend/hcz_prep_backend
# as constructor kwargs (confirmed by reading their signatures) -- those are plain mutable fields
# on CMBinHessCtx/OriginZCCoreHessCtx, set via their own ZC_EZ_BACKEND_DEFAULT[]/
# HCZ_PREP_BACKEND_DEFAULT[] Refs at construction and meant to be flipped post-construction, exactly
# how gate3_compile_free_backend_ab_2026-08-01.jl's own timed_meanzc/timed_originzc do it
# (`pcx.cctx.zc_ez_backend = hez_be`, etc.). ONE shared cctx/octx per family, built once with
# threaded_bins=true so cctx.tls exists for every arm (use_threaded_bins is then just a bool flip,
# not a rebuild) -- mutated in place per arm below, not rebuilt per arm.
cctx_reduced_meanzc = build_cm_meanzc_bin_ctx(ctx, aug_reduced_meanzc; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = true,
    inner_fg_backend = :dense_reference, profiled_layout = layout)
octx_reduced_originzc = build_originzc_core_hess_ctx(aug_reduced_originzc, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)

function apply_cmzc_arm!(arm_cfg)
    cctx_reduced_meanzc.zc_ez_backend = arm_cfg.zc_ez_backend
    cctx_reduced_meanzc.hcz_prep_backend = arm_cfg.hcz_prep_backend
    cctx_reduced_meanzc.use_threaded_bins = arm_cfg.use_threaded_bins
    return cctx_reduced_meanzc
end
function apply_originzc_arm!(arm_cfg)
    octx_reduced_originzc.zc_ez_backend = arm_cfg.zc_ez_backend
    return octx_reduced_originzc
end

rows = NamedTuple[]
prof_rows_by_arm = Dict{String,Vector{NamedTuple}}()

function record_prof!(key)
    prof_rows_by_arm[key] = prof_summary()
end

# ---- FULL arm timing (cold reset methodology from gate3) ----
function timed_full_meanzc()
    pcx_meanzc_full.cctx.zc_gram_backend = :blas_syrk
    pcx_meanzc_full.cctx.hcz_prep_backend = :draw_chunk_reordered
    pcx_meanzc_full.cctx.zc_ez_backend = :drawmajor_v2
    pcx_meanzc_full.ctx_cm.obj.x .= NaN
    reset_no_dense_g_counters!(); prof_reset!()
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K_, base, verify = cm_meanzc_production_value_verified_screened(x_free_calib, nu_meanzc, pcx_meanzc_full)
    wall = time() - t0
    record_prof!("cmzc|FULL|W$(W_VAL)")
    return (family = "cm_meanzc", arm = "FULL", wall_s = wall, iters = CS.INNER_ITERS_TOTAL[] - iters0,
        nStatus = base.inner_status, Delta_dual = get(verify, :Delta_dual, NaN),
        dense_econ = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations)
end
function timed_full_originzc()
    pcx_originzc_full.octx.zc_gram_backend = :blas_syrk
    pcx_originzc_full.octx.zc_ez_backend = :drawmajor_v2
    pcx_originzc_full.ctx_cm.obj.x .= NaN
    reset_no_dense_g_counters!(); prof_reset!()
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K_, base, verify = cm_originzc_production_value_verified_screened(x_free_calib, nu_originzc_full, pcx_originzc_full)
    wall = time() - t0
    record_prof!("originzc|FULL|W$(W_VAL)")
    return (family = "origin_zc", arm = "FULL", wall_s = wall, iters = CS.INNER_ITERS_TOTAL[] - iters0,
        nStatus = base.inner_status, Delta_dual = get(verify, :Delta_dual, NaN),
        dense_econ = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations)
end

function timed_reduced_meanzc(arm::Symbol)
    cctx = apply_cmzc_arm!(CMZC_ARMS[arm])
    reset_no_dense_g_counters!(); prof_reset!()
    t0 = time()
    base = reduced_meanzc_base_state(x_free_calib, νvec0_meanzc, ctx, layout, cctx)
    wall = time() - t0
    record_prof!("cmzc|$(arm)|W$(W_VAL)")
    c = NO_DENSE_G_COUNTERS[]
    return (family = "cm_meanzc", arm = String(arm), wall_s = wall, iters = base.n_fg,
        nStatus = base.inner_status, Delta_dual = NaN, dense_econ = c.dense_economic_G_materializations,
        drawmajor_v2_dispatch = c.drawmajor_v2_dispatch_count, drawmajor_v2_fallback = c.drawmajor_v2_fallback_count,
        draw_chunk_reordered_dispatch = c.draw_chunk_reordered_dispatch_count, draw_chunk_reordered_fallback = c.draw_chunk_reordered_fallback_count)
end
function timed_reduced_originzc(arm::Symbol)
    octx = apply_originzc_arm!(ORIGINZC_ARMS[arm])
    reset_no_dense_g_counters!(); prof_reset!()
    t0 = time()
    base = reduced_originzc_base_state(x_free_calib, ctx, layout, octx, νfull0_originzc)
    wall = time() - t0
    record_prof!("originzc|$(arm)|W$(W_VAL)")
    c = NO_DENSE_G_COUNTERS[]
    return (family = "origin_zc", arm = String(arm), wall_s = wall, iters = base.n_fg,
        nStatus = base.inner_status, Delta_dual = NaN, dense_econ = c.dense_economic_G_materializations,
        drawmajor_v2_dispatch = c.drawmajor_v2_dispatch_count, drawmajor_v2_fallback = c.drawmajor_v2_fallback_count,
        draw_chunk_reordered_dispatch = 0, draw_chunk_reordered_fallback = 0)
end

lp("="^100); lp("THROWAWAY warm-up pass (discarded, JIT compile every code path once)")
timed_full_meanzc(); timed_full_originzc()
for arm in keys(CMZC_ARMS); timed_reduced_meanzc(arm); end
for arm in keys(ORIGINZC_ARMS); timed_reduced_originzc(arm); end
lp("warm-up done, discarded"); flush(stdout)

lp("="^100); lp("REAL MEASUREMENTS  W=$(W_VAL)  ", now())
r = timed_full_meanzc(); push!(rows, r)
@printf("  [cm_meanzc |%-26s] wall=%8.3fs iters=%-4d nStatus=%-4d\n", r.arm, r.wall_s, r.iters, r.nStatus); flush(stdout)
for arm in [:REDUCED_BASELINE, :REDUCED_DRAWMAJOR_ONLY, :REDUCED_DRAWMAJOR_PLUS_HCZ, :REDUCED_ALL_ACCEPTED]
    r = timed_reduced_meanzc(arm); push!(rows, r)
    @printf("  [cm_meanzc |%-26s] wall=%8.3fs iters=%-4d nStatus=%-4d  drawmajor_v2(d=%d,f=%d) draw_chunk_reordered(d=%d,f=%d)\n",
        r.arm, r.wall_s, r.iters, r.nStatus, r.drawmajor_v2_dispatch, r.drawmajor_v2_fallback, r.draw_chunk_reordered_dispatch, r.draw_chunk_reordered_fallback)
    flush(stdout)
end
r = timed_full_originzc(); push!(rows, r)
@printf("  [origin_zc |%-26s] wall=%8.3fs iters=%-4d nStatus=%-4d\n", r.arm, r.wall_s, r.iters, r.nStatus); flush(stdout)
for arm in [:REDUCED_BASELINE, :REDUCED_ALL_ACCEPTED]
    r = timed_reduced_originzc(arm); push!(rows, r)
    @printf("  [origin_zc |%-26s] wall=%8.3fs iters=%-4d nStatus=%-4d  drawmajor_v2(d=%d,f=%d)\n",
        r.arm, r.wall_s, r.iters, r.nStatus, r.drawmajor_v2_dispatch, r.drawmajor_v2_fallback)
    flush(stdout)
end

outpath = joinpath(D4X, "..", "..", "docs", "PRODSCALE_FULL_VS_REDUCED_AB_W$(W_VAL)_2026-08-02.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "family,arm,wall_s,iters,nStatus,dense_econ,drawmajor_v2_dispatch,drawmajor_v2_fallback,draw_chunk_reordered_dispatch,draw_chunk_reordered_fallback")
    for r in rows
        dj = get(r, :drawmajor_v2_dispatch, -1); fj = get(r, :drawmajor_v2_fallback, -1)
        dc = get(r, :draw_chunk_reordered_dispatch, -1); fc = get(r, :draw_chunk_reordered_fallback, -1)
        println(io, "$(r.family),$(r.arm),$(r.wall_s),$(r.iters),$(r.nStatus),$(r.dense_econ),$(dj),$(fj),$(dc),$(fc)")
    end
end
lp("Wrote ", outpath)

# ---- Section 5: fixed-state callback decomposition, printed for FULL and REDUCED_BASELINE ----
profpath = joinpath(D4X, "..", "..", "docs", "PRODSCALE_CALLBACK_DECOMPOSITION_W$(W_VAL)_2026-08-02.csv")
open(profpath, "w") do io
    println(io, "arm_key,label,n,total_s,mean_s,median_s")
    for (key, prows) in prof_rows_by_arm
        for row in prows
            total_s = row.n * row.mean_s
            println(io, "$(key),$(row.label),$(row.n),$(total_s),$(row.mean_s),$(row.median_s)")
        end
    end
end
lp("Wrote ", profpath)

lp("="^100); @printf("TOTAL WALL: %.2fs\n", time() - t0_total); lp("="^100)
lp("DONE")
