# Part C: complete-inner-solve smoke for HCZ_PREP_BACKEND_DEFAULT flip candidate (2026-07-29).
# Real D=20/W=100,000 cm_meanzc, run_cm_upper_checkpointed, comparing :origin_owned (current
# default) vs :draw_chunk_thread_local (candidate) end-to-end through real KNITRO.
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
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
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(D4X, f))
end
using Random, Printf, LinearAlgebra, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const W = 100_000
const DELTA = 1.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "hcz_complete_inner_solve_smoke_2026-07-29")
mkpath(OUTROOT)
BLAS.set_num_threads(1)

function cmzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    return vcat(w_a_calib, log.(Float64.(factorial.(1:1))))
end

ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
w0 = cmzc_w0(ctx0, pe0, theta0, xy0)

results = NamedTuple[]
for backend in (:origin_owned, :draw_chunk_thread_local)
    HCZ_PREP_BACKEND_DEFAULT[] = backend
    lp("\n=== HCZ_PREP_BACKEND=$backend ===")
    t0 = time()
    result = run_cm_upper_checkpointed(w0;
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal,
        probs = nested_grid_sequence([10, 20, 50])[50],
        cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
        ckpt_dir = OUTROOT, run_id = "hcz_smoke_$(backend)", label = "hcz_smoke_$(backend)",
        checkpoint_interval_s = 3600.0, maxtime_real = 60.0, verbose = false)
    wall = time() - t0
    lp("[RESULT $backend] wall=", @sprintf("%.1f", wall), "s status=", result.knitro_status,
       " n_eval=", result.n_eval, " n_grad=", result.n_grad, " kappa=", result.kappa)
    push!(results, (backend = backend, wall = wall, status = result.knitro_status, n_eval = result.n_eval, n_grad = result.n_grad, kappa = result.kappa))
end

println("\n=== SUMMARY ===")
for r in results
    println(rpad(string(r.backend), 24), " wall=", round(r.wall, digits=1), "s status=", r.status, " n_eval=", r.n_eval, " n_grad=", r.n_grad, " kappa=", r.kappa)
end
if length(results) == 2 && all(r -> r.status in (0,-100,-101,-103,-401), results)
    kdiff = abs(results[1].kappa - results[2].kappa)
    println("kappa diff = ", kdiff)
end
println("DONE.")
