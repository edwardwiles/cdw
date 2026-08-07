# fix/fullA-lower-limit-and-hotpath-2026-08-06, task P1: live Profile.Allocs trace of the real
# production common-Frechet (two-family) Hessian callback at D20/W=100,000/L=50.
#
# Reuses the EXACT same wiring as smoke_frechet_public_driver_tls_2026-08-06.jl (the real
# run_cm_upper_checkpointed public driver, not a hand-rolled reimplementation) -- this is a
# deliberate low-risk choice: rather than hand-invoking hessian_cm_structured_v2! directly (which
# would require correctly reconstructing its exact pre-call state, a real chance of a silent setup
# bug producing a WRONG allocation number), wrap the real driver call in Profile.Allocs and filter
# the resulting allocation records post-hoc by backtrace, isolating exactly the frames attributable
# to the real Hessian callback.
#
# JULIA_NUM_THREADS=1 deliberately for THIS run only: cctx.use_threaded_bins (hence which function,
# hessian_cm_structured_v2! vs the serial hessian_cm_structured!, actually executes) is controlled
# by the threaded_bins KWARG below, not by Threads.nthreads() -- so passing threaded_bins=true still
# exercises the real production code path even at nthreads()=1. Single-threaded execution keeps
# every allocation's backtrace attributable to one clean call stack, avoiding any risk of
# Profile.Allocs misattributing allocations across threads. Thread-SCALING is task P2's question,
# not this one -- P1 only needs to identify WHICH LINES allocate, not how volume changes with
# thread count.
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_cplus.jl", "cm_meanzc_lookup_production.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Profile, Profile.Allocs

lp(xs...) = (println(xs...); flush(stdout))

lp("nthreads()=", Threads.nthreads())

ctx = d20_real_setup_design(W = 100_000, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    inner_lower_limit = -10.0)
pe = build_pivot_elimination(ctx)
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
L = 50
probs = cm_equal_grid_probs(L)
CKPT = mktempdir()

lp("="^90); lp("P1 allocation trace: common-Frechet TWO-FAMILY through the real public driver"); lp("="^90)
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true

Profile.Allocs.@profile sample_rate=0.1 begin
    global result = run_cm_upper_checkpointed(w0; W = 100_000, delta = 1.0,
        draw_design = :sobol_randomized, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs,
        include_truncated_moment = true,   # TWO-FAMILY (fam2), matching the profiling target
        cm_hessian_backend = :structured, threaded_bins = true,
        exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
        marginal_restriction = :common_frechet,
        A_coordinate_mode = :powered_aspace,
        inner_lower_limit = -10.0,
        ckpt_dir = CKPT, run_id = "p1_allocs_frechet", label = "p1_allocs_frechet",
        checkpoint_interval_s = 90.0, maxtime_real = 90.0, verbose = true)
end
lp("knitro_status=", result.knitro_status, " n_eval=", result.n_eval, " n_grad=", result.n_grad)

pcx_live = CM_LIVE_PCX_STASH[]
if pcx_live !== nothing
    lp("cctx.use_threaded_bins=", pcx_live.cctx.use_threaded_bins, " n_families=", pcx_live.cctx.n_families)
end

lp("="^90); lp("Fetching Profile.Allocs results..."); lp("="^90)
profdata = Profile.Allocs.fetch()
lp("total allocation records captured: ", length(profdata.allocs))

function frame_names(alloc)
    st = alloc.stacktrace
    return [string(f.func) for f in st]
end

is_hessian_frame(names) = any(n -> occursin("hessian_cm_structured", n) || occursin("archC_frechet_hess_cb_builder", n) ||
                                    occursin("_fill_frechet_level_blocks", n) || occursin("_fill_cm_HEE", n) ||
                                    occursin("build_bin_tables", n) || occursin("fill_cm_HCC", n) ||
                                    occursin("winner_pair_cross_hessian", n) || occursin("pack_upper_cm_hessian", n) ||
                                    occursin("prefix_sum_tables", n), names)

hess_allocs = filter(a -> is_hessian_frame(frame_names(a)), profdata.allocs)
lp("allocation records attributed to the Hessian-callback call tree: ", length(hess_allocs))

# Group by (top hessian-related frame, type) -> (count, total bytes)
site_bytes = Dict{Tuple{String,String},Tuple{Int,Int}}()
for a in hess_allocs
    names = frame_names(a)
    idx = findfirst(n -> occursin("hessian_cm_structured", n) || occursin("archC_frechet_hess_cb_builder", n) ||
                          occursin("_fill_frechet_level_blocks", n) || occursin("_fill_cm_HEE", n) ||
                          occursin("build_bin_tables", n) || occursin("fill_cm_HCC", n) ||
                          occursin("winner_pair_cross_hessian", n) || occursin("pack_upper_cm_hessian", n) ||
                          occursin("prefix_sum_tables", n), names)
    site = idx === nothing ? "unknown" : names[idx]
    tyname = string(a.type)
    key = (site, tyname)
    cnt, tot = get(site_bytes, key, (0, 0))
    site_bytes[key] = (cnt + 1, tot + a.size)
end

lp("="^90); lp("TOP ALLOCATION SITES (by total bytes), Hessian-callback call tree only"); lp("="^90)
sorted_sites = sort(collect(site_bytes); by = kv -> -kv[2][2])
total_hess_bytes = sum(v[2] for v in values(site_bytes); init = 0)
lp(@sprintf("TOTAL bytes attributed to Hessian call tree: %.3e (%.2f MB)", total_hess_bytes, total_hess_bytes / 2^20))
for (i, ((site, ty), (cnt, bytes))) in enumerate(sorted_sites)
    i > 15 && break
    @printf("%2d. %-45s %-30s count=%6d  bytes=%.3e (%.2f MB)  %.1f%%\n",
        i, site, ty, cnt, bytes, bytes / 2^20, 100 * bytes / max(total_hess_bytes, 1))
end

lp("="^90); lp("DONE"); lp("="^90)
