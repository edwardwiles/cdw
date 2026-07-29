# H_ZZ realistic resource gate worker (2026-07-28 selective-release continuation).
#
# Single-family, single-process worker for the H_ZZ backend resource gate (task brief §8, scoped
# down from the full 5-process realistic plan given time constraints -- see
# docs/HZZ_REALISTIC_RESOURCE_GATE_2026-07-28.md for what was and wasn't covered). Meant to be
# launched as an independent OS process (matching how families actually run in production, per the
# task brief's own "because families run as separate processes" framing) -- run twice (isolated)
# and then twice CONCURRENTLY (two simultaneous processes) to see whether BLAS-thread
# oversubscription or KNITRO-process concurrency degrades the :blas_gemm H_ZZ backend's isolated
# win.
#
# Usage: OPENBLAS_NUM_THREADS=<N> OMP_NUM_THREADS=<N> FAMILY=origin_zc|cm_meanzc BLAS_THREADS=<N> \
#            RUN_TAG=<label> julia --project=<worktree-root> -t 4 hzz_resource_gate_worker_2026-07-28.jl
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
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(D4X, f))
end
using Random, Printf, LinearAlgebra, Statistics, Dates
lp(xs...) = (println(xs...); flush(stdout))

const FAMILY = get(ENV, "FAMILY", "origin_zc")
const BLAS_THREADS = parse(Int, get(ENV, "BLAS_THREADS", "8"))
const RUN_TAG = get(ENV, "RUN_TAG", "solo")
BLAS.set_num_threads(BLAS_THREADS)
lp("[$FAMILY|$RUN_TAG] Threads.nthreads()=", Threads.nthreads(), " BLAS threads set to ", BLAS_THREADS,
   " (actual: ", BLAS.get_num_threads(), ")")

ZC_GRAM_BACKEND_DEFAULT[] = :blas_gemm
ZC_GRAM_THREADED_WORKERS_DEFAULT[] = 1   # irrelevant for :blas_gemm (BLAS threads govern it, not this)

const W = 100_000
const DELTA = 1.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "hzz_resource_gate_2026-07-28", "$(FAMILY)_$(RUN_TAG)_blas$(BLAS_THREADS)")
mkpath(OUTROOT)

function originzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    return vcat(w_a_calib, log.(nu0))
end

function cmzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    return vcat(w_a_calib, log.(Float64.(factorial.(1:1))))
end

t0 = time()
local result, status_str, kappa_str
if FAMILY == "origin_zc"
    ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
    pe0 = build_pivot_elimination(ctx0)
    theta0 = cm_fixed_theta(ctx0)
    xy0 = precompute_cm_aspace_xy(ctx0)
    w0 = originzc_w0(ctx0, pe0, theta0, xy0)
    result = run_originzc_upper_checkpointed(w0;
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
        ckpt_dir = OUTROOT, run_id = "hzz_$(FAMILY)_$(RUN_TAG)", label = "hzz_$(FAMILY)_$(RUN_TAG)",
        checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = true)
elseif FAMILY == "cm_meanzc"
    ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
    pe0 = build_pivot_elimination(ctx0)
    theta0 = cm_fixed_theta(ctx0)
    xy0 = precompute_cm_aspace_xy(ctx0)
    w0 = cmzc_w0(ctx0, pe0, theta0, xy0)
    result = run_cm_upper_checkpointed(w0;
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal,
        probs = nested_grid_sequence([10, 20, 50])[50],
        cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
        ckpt_dir = OUTROOT, run_id = "hzz_$(FAMILY)_$(RUN_TAG)", label = "hzz_$(FAMILY)_$(RUN_TAG)",
        checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = true)
else
    error("unknown FAMILY=$FAMILY")
end
wall = time() - t0
status_str = string(result.knitro_status)
kappa_str = hasproperty(result, :kappa) ? string(result.kappa) : "n/a"
lp("[RESULT $FAMILY|$RUN_TAG|blas$BLAS_THREADS] wall=", @sprintf("%.1f", wall), "s status=", status_str,
   " n_eval=", result.n_eval, " n_grad=", result.n_grad, " kappa=", kappa_str)

csvpath = joinpath(D4X, "..", "..", "docs", "HZZ_REALISTIC_RESOURCE_GATE_2026-07-28.csv")
mkpath(dirname(csvpath))
line = "$(FAMILY),$(RUN_TAG),$(BLAS_THREADS),$(wall),$(status_str),$(result.n_eval),$(result.n_grad),$(kappa_str)\n"
# Multiple concurrent processes append to this SAME file -- use a simple O_APPEND open (atomic for
# small writes on a local filesystem) rather than read-modify-write, so concurrent workers don't
# clobber each other's rows.
open(csvpath, "a") do io
    write(io, line)
end
lp("Appended result row to ", csvpath)
lp("DONE.")
